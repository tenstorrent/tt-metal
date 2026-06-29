# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# =============================================================================
# Recheck each Ideogram 4.0 module against the REAL shipped weights: the fp8
# checkpoint (ideogram-ai/ideogram-4-fp8) dequantized to bf16 (dequant.py) is
# loaded into both the reference and the tt module, and PCC is re-asserted. This
# replaces the random-init structural check with the actual model weights.
# =============================================================================

import pytest
import torch
import transformers
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from loguru import logger
from safetensors.torch import load_file

import ttnn

from ....encoders.qwen3vl.model_qwen3vl import Qwen3VlTextEncoder, create_rope_tensors
from ....models.transformers.transformer_ideogram4 import Ideogram4Transformer
from ....models.vae.vae_ideogram4 import Ideogram4VAEDecoder
from ....parallel.config import DiTParallelConfig, ParallelFactor, VAEParallelConfig
from ....parallel.manager import CCLManager
from ....reference.ideogram4 import modeling_ideogram4
from ....reference.ideogram4.constants import OUTPUT_IMAGE_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS
from ....reference.ideogram4.dequant import dequant_fp8_state_dict
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor
from .test_transformer_ideogram4_model import _build_model_inputs

FP8 = "/localdev/cglagovich/ideogram-4-fp8"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_transformer_real_weights(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    llm_len, image_len = 64, 256
    seq_len = llm_len + image_len

    sd = dequant_fp8_state_dict(load_file(f"{FP8}/transformer/diffusion_pytorch_model.safetensors"))
    config = modeling_ideogram4.Ideogram4Config()
    ref = modeling_ideogram4.Ideogram4Transformer(config).to(torch_dtype).eval()
    ref.load_state_dict(sd, strict=False)

    llm_features, x, t, position_ids, segment_ids, indicator = _build_model_inputs(config, 1, llm_len, image_len)
    llm_features, x = llm_features.to(torch_dtype), x.to(torch_dtype)
    with torch.no_grad():
        ref_out = ref(
            llm_features=llm_features, x=x, t=t, position_ids=position_ids, segment_ids=segment_ids, indicator=indicator
        )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    tt = Ideogram4Transformer(
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        adaln_dim=config.adanln_dim,
        in_channels=config.in_channels,
        llm_features_dim=config.llm_features_dim,
        norm_eps=config.norm_eps,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear),
        parallel_config=parallel_config,
    )
    tt.load_torch_state_dict(sd)

    cos, sin = ref.rotary_emb(position_ids)
    t_sin = Ideogram4Transformer.sinusoidal_embedding(t, config.emb_dim)
    llm_mask = (indicator == 3).to(torch.float32).unsqueeze(-1)
    img_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.float32).unsqueeze(-1)
    image_idx = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.int32)

    tt_out = tt(
        x=bf16_tensor(x, device=mesh_device),
        llm_features=bf16_tensor(llm_features, device=mesh_device),
        t_sin=bf16_tensor(t_sin.unsqueeze(1), device=mesh_device),
        cos=bf16_tensor(cos.unsqueeze(1), device=mesh_device),
        sin=bf16_tensor(sin.unsqueeze(1), device=mesh_device),
        image_indicator_index=tensor.from_torch(image_idx, device=mesh_device, dtype=ttnn.uint32),
        llm_token_mask=bf16_tensor(llm_mask, device=mesh_device),
        output_image_mask=bf16_tensor(img_mask, device=mesh_device),
        spatial_sequence_length=seq_len,
    )
    tt_torch = tensor.to_torch(tt_out, mesh_axes=[None, None, None])
    mask = (indicator == OUTPUT_IMAGE_INDICATOR)[0]
    logger.info("transformer REAL weights:")
    assert_quality(ref_out[:, mask], tt_torch[:, mask], pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vae_real_weights(*, mesh_device: ttnn.MeshDevice) -> None:
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16

    vae_sd = load_file(f"{FP8}/vae/diffusion_pytorch_model.safetensors")  # diffusers layout, not fp8
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

    parallel_config = VAEParallelConfig(tensor_parallel=ParallelFactor(factor=1, mesh_axis=1))
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    tt = Ideogram4VAEDecoder.from_torch(akl, mesh_device=mesh_device, parallel_config=parallel_config, ccl_manager=ccl)

    z = torch.randn(1, 32, 32, 32, dtype=torch_dtype)
    with torch.no_grad():
        ref = akl.decode(z).sample
    tt_z = ttnn.from_torch(
        z.permute(0, 2, 3, 1),
        dtype=ttnn.bfloat16,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
        layout=ttnn.TILE_LAYOUT,
    )
    tt_out = ttnn.to_torch(ttnn.get_device_tensors(tt(tt_z))[0]).permute(0, 3, 1, 2)
    logger.info("VAE REAL weights:")
    assert_quality(ref.float(), tt_out.float(), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("seq_len", [128])
def test_text_encoder_real_weights(*, mesh_device: ttnn.MeshDevice, seq_len: int) -> None:
    torch.manual_seed(0)
    sd = dequant_fp8_state_dict(load_file(f"{FP8}/text_encoder/model.safetensors"))
    sd = {k[len("language_model.") :]: v for k, v in sd.items() if k.startswith("language_model.")}

    hf = transformers.AutoModel.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype=torch.bfloat16)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    lm.load_state_dict(sd, strict=False)  # load the Ideogram-shipped (dequantized) weights
    lm.eval()
    cfg = lm.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    mrope_section, rope_theta = cfg.rope_scaling["mrope_section"], cfg.rope_scaling["rope_theta"]

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len))
    caps = {}
    handles = [
        lm.layers[i].register_forward_hook(
            lambda m, inp, out, i=i: caps.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach())
        )
        for i in QWEN3_VL_ACTIVATION_LAYERS
    ]
    with torch.no_grad():
        lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    for h in handles:
        h.remove()
    golden = [caps[i].float() for i in QWEN3_VL_ACTIVATION_LAYERS]

    enc = Qwen3VlTextEncoder(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        hidden_act="silu",
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=rope_theta,
        mrope_section=mrope_section,
        activation_layers=QWEN3_VL_ACTIVATION_LAYERS,
        device=mesh_device,
    )
    enc.load_torch_state_dict(sd)
    cos, sin = create_rope_tensors(1, seq_len, None, head_dim, rope_theta, mrope_section)
    tt_ids = ttnn.from_torch(ids, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    tt_caps = enc.forward(
        tt_ids,
        attention_mask=None,
        pos_embeds=(bf16_tensor(cos, device=mesh_device), bf16_tensor(sin, device=mesh_device)),
    )

    for layer_idx, g, tt_t in zip(QWEN3_VL_ACTIVATION_LAYERS, golden, tt_caps):
        logger.info(f"text encoder REAL weights layer {layer_idx}:")
        assert_quality(g, tensor.to_torch(tt_t, mesh_axes=[None, None, None]), pcc=0.99)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_transformer_real_inputs(*, mesh_device: ttnn.MeshDevice) -> None:
    """Transformer with REAL llm_features (from the actual encoder) + a noise latent.

    The per-module recheck fed random N(0,1) llm_features, which are out-of-distribution
    for the trained llm_cond_proj; this uses real encoder features to settle whether the
    ~0.966 was an OOD-input artifact vs a real fidelity defect.
    """
    import gc

    from ....reference.ideogram4.constants import IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR

    torch.manual_seed(0)
    torch_dtype = torch.bfloat16
    config = modeling_ideogram4.Ideogram4Config()

    # ---- real llm_features from the HF Qwen3-VL encoder (host), then free it ----
    tok = transformers.AutoTokenizer.from_pretrained(f"{FP8}", subfolder="tokenizer")
    hf = transformers.AutoModel.from_pretrained("Qwen/Qwen3-VL-8B-Instruct", torch_dtype=torch_dtype)
    lm = hf.language_model if hasattr(hf, "language_model") else hf.model.language_model
    enc_sd = dequant_fp8_state_dict(load_file(f"{FP8}/text_encoder/model.safetensors"))
    lm.load_state_dict(
        {k[len("language_model.") :]: v for k, v in enc_sd.items() if k.startswith("language_model.")}, strict=False
    )
    lm.eval()

    msgs = [{"role": "user", "content": [{"type": "text", "text": "a red panda reading a book under a cherry tree"}]}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
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
    llm_text = torch.cat([caps[i] for i in QWEN3_VL_ACTIVATION_LAYERS], dim=-1).to(torch_dtype)  # [1, n_text, 53248]
    del hf, lm, caps
    gc.collect()

    # ---- packed sequence [text | image], real positions/indicator (per pipeline) ----
    image_len = 256
    grid = 16
    seq = n_text + image_len
    llm_features = torch.zeros(1, seq, config.llm_features_dim, dtype=torch_dtype)
    llm_features[:, :n_text] = llm_text
    x = torch.randn(1, seq, config.in_channels, dtype=torch_dtype)  # noise latent (correct)
    t = torch.tensor([0.5])
    indicator = torch.full((1, seq), OUTPUT_IMAGE_INDICATOR, dtype=torch.long)
    indicator[:, :n_text] = LLM_TOKEN_INDICATOR
    position_ids = torch.zeros(1, seq, 3, dtype=torch.long)
    tp = torch.arange(n_text)
    position_ids[:, :n_text] = torch.stack([tp, tp, tp], dim=1)
    hh2 = torch.arange(grid).repeat_interleave(grid)
    ww2 = torch.arange(grid).repeat(grid)
    position_ids[:, n_text:, 0] = IMAGE_POSITION_OFFSET
    position_ids[:, n_text:, 1] = IMAGE_POSITION_OFFSET + hh2
    position_ids[:, n_text:, 2] = IMAGE_POSITION_OFFSET + ww2
    segment_ids = torch.zeros(1, seq, dtype=torch.long)

    sd = dequant_fp8_state_dict(load_file(f"{FP8}/transformer/diffusion_pytorch_model.safetensors"))
    ref = modeling_ideogram4.Ideogram4Transformer(config).to(torch_dtype).eval()
    ref.load_state_dict(sd, strict=False)
    with torch.no_grad():
        ref_out = ref(
            llm_features=llm_features, x=x, t=t, position_ids=position_ids, segment_ids=segment_ids, indicator=indicator
        )

    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=1, mesh_axis=1),
        sequence_parallel=ParallelFactor(factor=1, mesh_axis=0),
    )
    tt = Ideogram4Transformer(
        emb_dim=config.emb_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        adaln_dim=config.adanln_dim,
        in_channels=config.in_channels,
        llm_features_dim=config.llm_features_dim,
        norm_eps=config.norm_eps,
        mesh_device=mesh_device,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear),
        parallel_config=parallel_config,
    )
    tt.load_torch_state_dict(sd)

    cos, sin = ref.rotary_emb(position_ids)
    t_sin = Ideogram4Transformer.sinusoidal_embedding(t, config.emb_dim)
    llm_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.float32).unsqueeze(-1)
    img_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.float32).unsqueeze(-1)
    image_idx = (indicator == OUTPUT_IMAGE_INDICATOR).to(torch.int32)
    tt_out = tt(
        x=bf16_tensor(x, device=mesh_device),
        llm_features=bf16_tensor(llm_features, device=mesh_device),
        t_sin=bf16_tensor(t_sin.unsqueeze(1), device=mesh_device),
        cos=bf16_tensor(cos.unsqueeze(1), device=mesh_device),
        sin=bf16_tensor(sin.unsqueeze(1), device=mesh_device),
        image_indicator_index=tensor.from_torch(image_idx, device=mesh_device, dtype=ttnn.uint32),
        llm_token_mask=bf16_tensor(llm_mask, device=mesh_device),
        output_image_mask=bf16_tensor(img_mask, device=mesh_device),
        spatial_sequence_length=seq,
    )
    tt_torch = tensor.to_torch(tt_out, mesh_axes=[None, None, None])
    mask = (indicator == OUTPUT_IMAGE_INDICATOR)[0]
    logger.info("transformer REAL weights + REAL llm_features:")
    assert_quality(ref_out[:, mask], tt_torch[:, mask], pcc=0.99)
