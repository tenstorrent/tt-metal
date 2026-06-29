# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# End-to-end Ideogram 4.0 text-to-image on a single Blackhole device using the
# real (dequantized fp8) weights. To fit one device, this runs the conditional
# branch only (gw=1, no CFG): host Qwen3-VL encode -> tt Ideogram4Transformer
# denoise loop (Euler) -> tt Flux2 VAE decode -> PNG.
# =============================================================================

import gc

import pytest
import torch
import transformers
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger
from PIL import Image
from safetensors.torch import load_file

import ttnn

from ....models.transformers.transformer_ideogram4 import Ideogram4Transformer
from ....models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ....parallel.config import DiTParallelConfig, ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....pipelines.ideogram4.pipeline import Ideogram4DecodeStage
from ....pipelines.ideogram4.sampler import Ideogram4Sampler
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
PROMPT = "a watercolor painting of a red panda reading a book under a cherry tree, soft morning light"
OUT_PATH = "/localdev/cglagovich/ideogram4_sample.png"


def _encode_prompt(prompt, llm_features_dim):
    """Host Qwen3-VL encode -> real [1, n_text, 13*4096] features (then free the encoder)."""
    tok = transformers.AutoTokenizer.from_pretrained(FP8, subfolder="tokenizer")
    hf = transformers.AutoModel.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype=torch.bfloat16)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    enc_sd = dequant_fp8_state_dict(load_file(f"{FP8}/text_encoder/model.safetensors"))
    lm.load_state_dict(
        {k[len("language_model.") :]: v for k, v in enc_sd.items() if k.startswith("language_model.")}, strict=False
    )
    lm.eval()
    del enc_sd

    text = tok.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}], add_generation_prompt=True, tokenize=False
    )
    ids = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
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
    feats = torch.cat([caps[i] for i in QWEN3_VL_ACTIVATION_LAYERS], dim=-1).to(torch.bfloat16)  # [1, n_text, 53248]
    n_text = ids.shape[1]
    del hf, lm, caps
    gc.collect()
    return feats, n_text


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(("height", "width", "num_steps"), [(512, 512, 12)], ids=["512px_12steps"])
def test_generate_image(*, mesh_device: ttnn.MeshDevice, height: int, width: int, num_steps: int) -> None:
    torch.manual_seed(1234)
    torch_dtype = torch.bfloat16
    config = modeling_ideogram4.Ideogram4Config()
    patch, ae = 2, 8
    grid_h, grid_w = height // (patch * ae), width // (patch * ae)
    num_img = grid_h * grid_w

    # ---- 1. encode (host) ----
    llm_text, n_text = _encode_prompt(PROMPT, config.llm_features_dim)
    seq = n_text + num_img
    logger.info(f"encoded prompt: {n_text} text tokens, {num_img} image tokens ({grid_h}x{grid_w})")

    # packed [text | image] conditioning
    llm_features = torch.zeros(1, seq, config.llm_features_dim, dtype=torch_dtype)
    llm_features[:, :n_text] = llm_text
    indicator = torch.full((1, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :n_text] = LLM_TOKEN_INDICATOR
    position_ids = torch.zeros(1, seq, 3, dtype=torch.long)
    tp = torch.arange(n_text)
    position_ids[:, :n_text] = torch.stack([tp, tp, tp], dim=1)
    hh = torch.arange(grid_h).repeat_interleave(grid_w)
    ww = torch.arange(grid_w).repeat(grid_h)
    position_ids[:, n_text:, 0] = IMAGE_POSITION_OFFSET
    position_ids[:, n_text:, 1] = IMAGE_POSITION_OFFSET + hh
    position_ids[:, n_text:, 2] = IMAGE_POSITION_OFFSET + ww

    # host-precomputed rope + masks (fixed across steps)
    rope = modeling_ideogram4.Ideogram4MRoPE(
        head_dim=config.emb_dim // config.num_heads, base=config.rope_theta, mrope_section=config.mrope_section
    )
    cos, sin = rope(position_ids)
    tt_cos = bf16_tensor(cos.unsqueeze(1).to(torch_dtype), device=mesh_device)
    tt_sin = bf16_tensor(sin.unsqueeze(1).to(torch_dtype), device=mesh_device)
    tt_llm = bf16_tensor(llm_features, device=mesh_device)
    tt_llm_mask = bf16_tensor((indicator == LLM_TOKEN_INDICATOR).to(torch.float32).unsqueeze(-1), device=mesh_device)
    tt_img_mask = bf16_tensor((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.float32).unsqueeze(-1), device=mesh_device)
    tt_idx = tensor.from_torch(
        (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.int32), device=mesh_device, dtype=ttnn.uint32
    )

    # ---- 2. conditional transformer (device) ----
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    sd = dequant_fp8_state_dict(load_file(f"{FP8}/transformer/diffusion_pytorch_model.safetensors"))
    transformer = Ideogram4Transformer(
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        adaln_dim=config.adanln_dim,
        in_channels=config.in_channels,
        llm_features_dim=config.llm_features_dim,
        norm_eps=config.norm_eps,
        mesh_device=mesh_device,
        ccl_manager=ccl,
        parallel_config=parallel_config,
    )
    transformer.load_torch_state_dict(sd)
    del sd
    gc.collect()

    sampler = Ideogram4Sampler.from_preset("V4_TURBO_12", height=height, width=width)
    z = torch.randn(1, num_img, config.in_channels, dtype=torch.float32)

    for i in reversed(range(num_steps)):
        t_val, s_val = sampler.times_for_step(i)
        pos_x = torch.zeros(1, seq, config.in_channels, dtype=torch_dtype)
        pos_x[:, n_text:] = z.to(torch_dtype)
        t_sin = Ideogram4Transformer.sinusoidal_embedding(torch.tensor([t_val]), config.emb_dim)
        out = transformer(
            x=bf16_tensor(pos_x, device=mesh_device),
            llm_features=tt_llm,
            t_sin=bf16_tensor(t_sin.unsqueeze(1), device=mesh_device),
            cos=tt_cos,
            sin=tt_sin,
            image_indicator_index=tt_idx,
            llm_token_mask=tt_llm_mask,
            output_image_mask=tt_img_mask,
            spatial_sequence_length=seq,
        )
        v = tensor.to_torch(out, mesh_axes=[None, None, None])[:, n_text:].float()  # cond-only (gw=1)
        z = z + v * (s_val - t_val)
        logger.info(f"step {num_steps - i}/{num_steps} (t={t_val:.3f} -> {s_val:.3f})")

    del transformer
    gc.collect()
    ttnn.synchronize_device(mesh_device)

    # ---- 3. VAE decode (device) ----
    vae_sd = load_file(f"{FP8}/vae/diffusion_pytorch_model.safetensors")
    akl = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        norm_num_groups=32,
    )
    akl.load_state_dict({k: v for k, v in vae_sd.items() if not k.startswith("bn.")}, strict=False)
    akl = akl.to(torch_dtype).eval()
    vae = Ideogram4VAEDecoder.from_torch(
        akl,
        mesh_device=mesh_device,
        parallel_config=VAEParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1)),
        ccl_manager=ccl,
    )
    decode_stage = Ideogram4DecodeStage(vae, mesh_device=mesh_device, patch=patch)

    tt_z = bf16_tensor(z, device=mesh_device)
    decoded = decode_stage.decode(tt_z, grid_h=grid_h, grid_w=grid_w)
    img = Ideogram4DecodeStage.to_images(decoded)[0].cpu().numpy()  # [H, W, 3] uint8

    Image.fromarray(img).save(OUT_PATH)
    logger.info(f"saved generated image to {OUT_PATH}  shape={img.shape}  range=[{img.min()},{img.max()}]")
    assert img.shape == (height, width, 3)
    assert img.std() > 1.0, "image is flat — generation likely failed"
