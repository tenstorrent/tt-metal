# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Denoiser forward PCC + std-ratio across timestep t at 4096 tokens (1024px regime).
# The 1024px schedule front-loads LOW-t steps; the earlier forward test only probed
# t=0.5. If PCC/std-ratio degrade at low t, that's a t-dependent compute bug (time
# embedding / AdaLN) that only manifests in the loop. Loads the 9.3B model once.

import gc

import pytest
import torch
import transformers
from loguru import logger
from safetensors.torch import load_file

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4Transformer
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....reference.ideogram4 import modeling_ideogram4
from ....reference.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)
from ....reference.ideogram4.dequant import dequant_fp8_state_dict
from ....utils import tensor
from ....utils.tensor import bf16_tensor

FP8 = "/localdev/cglagovich/ideogram-4-fp8"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_denoiser_forward_t_sweep(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    dt = torch.bfloat16
    cfg = modeling_ideogram4.Ideogram4Config()
    grid = 64
    image_len = grid * grid

    tok = transformers.AutoTokenizer.from_pretrained(FP8, subfolder="tokenizer")
    hf = transformers.AutoModel.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype=dt)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    esd = dequant_fp8_state_dict(load_file(f"{FP8}/text_encoder/model.safetensors"))
    lm.load_state_dict(
        {k[len("language_model.") :]: v for k, v in esd.items() if k.startswith("language_model.")}, strict=False
    )
    lm.eval()
    text = tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "a red panda reading a book under a cherry tree"}]}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    n_text = ids.shape[1]
    caps = {}
    hh = [
        lm.layers[i].register_forward_hook(
            lambda m, i_, o, i=i: caps.__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach())
        )
        for i in QWEN3_VL_ACTIVATION_LAYERS
    ]
    with torch.no_grad():
        lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for h in hh:
        h.remove()
    llm_text = (
        torch.stack([caps[i] for i in QWEN3_VL_ACTIVATION_LAYERS], 0).permute(1, 2, 3, 0).reshape(1, n_text, -1).to(dt)
    )
    del hf, lm, esd, caps
    gc.collect()

    seq = n_text + image_len
    llm_features = torch.zeros(1, seq, cfg.llm_features_dim, dtype=dt)
    llm_features[:, :n_text] = llm_text
    x = torch.randn(1, seq, cfg.in_channels, dtype=dt)
    indicator = torch.full((1, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :n_text] = LLM_TOKEN_INDICATOR
    position_ids = torch.zeros(1, seq, 3, dtype=torch.long)
    tp = torch.arange(n_text)
    position_ids[:, :n_text] = torch.stack([tp, tp, tp], dim=1)
    a = torch.arange(grid)
    position_ids[:, n_text:, 0] = IMAGE_POSITION_OFFSET
    position_ids[:, n_text:, 1] = IMAGE_POSITION_OFFSET + a.repeat_interleave(grid)
    position_ids[:, n_text:, 2] = IMAGE_POSITION_OFFSET + a.repeat(grid)
    segment_ids = torch.zeros(1, seq, dtype=torch.long)

    sd = dequant_fp8_state_dict(load_file(f"{FP8}/transformer/diffusion_pytorch_model.safetensors"))
    ref = modeling_ideogram4.Ideogram4Transformer(cfg).to(dt).eval()
    ref.load_state_dict(sd, strict=False)

    pc = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    tt = Ideogram4Transformer(
        emb_dim=cfg.emb_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        intermediate_size=cfg.intermediate_size,
        adaln_dim=cfg.adanln_dim,
        in_channels=cfg.in_channels,
        llm_features_dim=cfg.llm_features_dim,
        norm_eps=cfg.norm_eps,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear),
        parallel_config=pc,
    )
    tt.load_torch_state_dict(sd)

    cos, sin = ref.rotary_emb(position_ids)
    llm_mask = (indicator == LLM_TOKEN_INDICATOR).float().unsqueeze(-1)
    img_mask = (indicator == OUTPUT_IMAGE_INDICATOR).float().unsqueeze(-1)
    image_idx = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.int32)
    mask = (indicator == OUTPUT_IMAGE_INDICATOR)[0]

    tt_cos = bf16_tensor(cos.unsqueeze(1), device=mesh_device)
    tt_sin = bf16_tensor(sin.unsqueeze(1), device=mesh_device)
    tt_llm = bf16_tensor(llm_features, device=mesh_device)
    tt_x = bf16_tensor(x, device=mesh_device)
    tt_idx = tensor.from_torch(image_idx, device=mesh_device, dtype=ttnn.uint32)
    tt_lm, tt_im = bf16_tensor(llm_mask, device=mesh_device), bf16_tensor(img_mask, device=mesh_device)

    for t_val in (0.02, 0.1, 0.3, 0.5, 0.9):
        t = torch.tensor([t_val])
        with torch.no_grad():
            ref_out = ref(
                llm_features=llm_features,
                x=x,
                t=t,
                position_ids=position_ids,
                segment_ids=segment_ids,
                indicator=indicator,
            )
        t_sin = Ideogram4Transformer.sinusoidal_embedding(t, cfg.emb_dim)
        tt_out = tt(
            x=tt_x,
            llm_features=tt_llm,
            t_sin=bf16_tensor(t_sin.unsqueeze(1), device=mesh_device),
            cos=tt_cos,
            sin=tt_sin,
            image_indicator_index=tt_idx,
            llm_token_mask=tt_lm,
            output_image_mask=tt_im,
            spatial_sequence_length=seq,
        )
        tt_torch = tensor.to_torch(tt_out, mesh_axes=[None, None, None])
        ri, ti = ref_out[:, mask].float(), tt_torch[:, mask].float()
        # PCC inline (avoid raising) + std ratio
        rf, tf = ri.flatten(), ti.flatten()
        pcc = torch.corrcoef(torch.stack([rf, tf]))[0, 1].item()
        logger.info(
            f"T-SWEEP t={t_val}: PCC={pcc:.5f} std_ratio={ti.std().item()/ri.std().item():.4f} (ref_std={ri.std():.3f} tt_std={ti.std():.3f})"
        )
