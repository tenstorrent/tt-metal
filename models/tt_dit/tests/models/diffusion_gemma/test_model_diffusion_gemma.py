# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end model parity: TT ``DiffusionGemmaForBlockDiffusion`` vs the actual
HF ``DiffusionGemmaForBlockDiffusion``.

Exercises the full text-only path::

    encoder text model → per-layer KV → decoder text model → lm_head → softcap

The 51 GB pretrained checkpoint isn't required — we build a tiny config on the
fly and load HF-initialized weights into TT. Multimodal (vision) is skipped;
the multimodal encoder has its own dedicated test.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_model_diffusion_gemma.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma._state_utils import per_layer_moe_substates
from ....models.transformers.diffusion_gemma.encoder_model import DiffusionGemmaEncoderModel
from ....models.transformers.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion, DiffusionGemmaModel
from ....models.transformers.diffusion_gemma.text_decoder import DiffusionGemmaDecoderModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
from ....utils.test import line_params, ring_params

# Looser thresholds: full stack (encoder + decoder + lm_head + tanh softcap) accumulates
# rounding from every component, so PCC stays at 0.99 but allclose has more headroom.
# End-to-end model = encoder → per-layer KV → decoder → lm_head → tanh softcap. The tanh
# softcap compresses large logits into [-30, 30] which tends to compress absolute drift too;
# on the other hand, chained matmul + softmax multiplies rounding across layers. Not yet
# run — set to a permissive layer-tolerance baseline and tighten with observed data.
PCC_THRESHOLD = 0.999
ALLCLOSE_ATOL = 5.5e-1
ALLCLOSE_RTOL = 5e-2


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("encoder_len", [32])
@pytest.mark.parametrize("canvas_len", [32])
def test_model_for_block_diffusion(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    encoder_len: int,
    canvas_len: int,
) -> None:
    """TT DiffusionGemmaForBlockDiffusion vs HF (text-only, tiny config)."""
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaForBlockDiffusion as HFForBlockDiffusion,
    )
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig

    torch.manual_seed(0)
    dtype = torch.float32

    text_config = DiffusionGemmaTextConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=64,
        moe_intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_global_key_value_heads=2,
        head_dim=32,
        global_head_dim=32,
        num_experts=8,
        top_k_experts=2,
        num_hidden_layers=6,
        sliding_window=64,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=2048,
        pad_token_id=0,
        final_logit_softcapping=30.0,
    )
    hf_config = DiffusionGemmaConfig(text_config=text_config, vision_config=None)
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if text_config.num_global_key_value_heads % tp_factor != 0:
        pytest.skip(f"num_global_kv_heads doesn't divide tp_factor={tp_factor}")

    # ---- HF reference model ----------------------------------------------------------------
    hf_model = HFForBlockDiffusion(hf_config).to(dtype).eval()
    # HF ties (a) encoder.layers ↔ decoder.layers, (b) encoder.embed ↔ decoder.embed ↔ lm_head,
    # (c) encoder.norm ↔ decoder.norm. With a fresh (random-init) construct these are independent;
    # mirror the tying here so the HF forward output exactly equals an equivalent TT run.
    hf_encoder = hf_model.model.encoder.language_model
    hf_decoder = hf_model.model.decoder
    hf_decoder.embed_tokens.weight = hf_encoder.embed_tokens.weight
    for li in range(text_config.num_hidden_layers):
        hf_decoder.layers[li].load_state_dict(hf_encoder.layers[li].state_dict(), strict=False)
    hf_decoder.norm.weight = hf_encoder.norm.weight
    hf_model.lm_head.weight = hf_decoder.embed_tokens.weight

    # ---- Inputs ----------------------------------------------------------------------------
    input_ids = torch.randint(low=1, high=text_config.vocab_size, size=(B, encoder_len), dtype=torch.long)
    decoder_input_ids = torch.randint(low=1, high=text_config.vocab_size, size=(B, canvas_len), dtype=torch.long)
    encoder_position_ids = torch.arange(encoder_len, dtype=torch.long).unsqueeze(0)
    decoder_position_ids = torch.arange(encoder_len, encoder_len + canvas_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        hf_out = hf_model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        hf_logits = hf_out.logits  # [B, canvas, vocab]

    # ---- TT model build + state load ------------------------------------------------------
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    hf_state = hf_model.state_dict()
    text_kwargs = dict(
        vocab_size=text_config.vocab_size,
        hidden_size=text_config.hidden_size,
        intermediate_size=text_config.intermediate_size,
        num_hidden_layers=text_config.num_hidden_layers,
        layer_types=list(text_config.layer_types),
        num_attention_heads=text_config.num_attention_heads,
        num_key_value_heads=text_config.num_key_value_heads,
        num_global_key_value_heads=text_config.num_global_key_value_heads,
        head_dim=text_config.head_dim,
        global_head_dim=text_config.global_head_dim,
        sliding_window=text_config.sliding_window,
        num_experts=text_config.num_experts,
        top_k_experts=text_config.top_k_experts,
        moe_intermediate_size=text_config.moe_intermediate_size,
        rms_norm_eps=text_config.rms_norm_eps,
        max_position_embeddings=text_config.max_position_embeddings,
        sliding_rope_theta=text_config.rope_parameters["sliding_attention"]["rope_theta"],
        full_rope_theta=text_config.rope_parameters["full_attention"]["rope_theta"],
        full_partial_rotary_factor=text_config.rope_parameters["full_attention"]["partial_rotary_factor"],
        pad_token_id=text_config.pad_token_id,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_links=num_links,
        topology=topology,
    )

    encoder_moe = per_layer_moe_substates(
        hf_state, num_layers=text_config.num_hidden_layers, prefix="model.encoder.language_model."
    )
    decoder_moe = per_layer_moe_substates(hf_state, num_layers=text_config.num_hidden_layers, prefix="model.decoder.")

    encoder = DiffusionGemmaEncoderModel(
        text_kwargs={**text_kwargs, "moe_state_dicts": encoder_moe},
        vision_kwargs=None,
        multimodal_hidden_size=text_config.hidden_size,
        text_hidden_size=text_config.hidden_size,
        rms_norm_eps=text_config.rms_norm_eps,
        image_token_id=getattr(hf_config, "image_token_id", 0),
        pad_token_id=text_config.pad_token_id,
        mesh_device=mesh_device,
    )
    decoder = DiffusionGemmaDecoderModel(**{**text_kwargs, "moe_state_dicts": decoder_moe})
    tt_model = DiffusionGemmaForBlockDiffusion(
        model=DiffusionGemmaModel(encoder=encoder, decoder=decoder),
        text_hidden_size=text_config.hidden_size,
        vocab_size=text_config.vocab_size,
        final_logit_softcapping=text_config.final_logit_softcapping,
        mesh_device=mesh_device,
    )
    tt_model.load_state_dict(hf_state)

    # ---- Masks: build via HF utilities so TT sees identical inputs -----------------------
    with torch.no_grad():
        inputs_embeds_for_mask = hf_encoder.embed_tokens(input_ids)
    mask_kwargs = {
        "config": text_config,
        "inputs_embeds": inputs_embeds_for_mask,
        "attention_mask": None,
        "past_key_values": None,
        "position_ids": encoder_position_ids,
    }
    hf_encoder_masks = {
        "full_attention": create_causal_mask(**mask_kwargs),
        "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
    }
    tt_encoder_masks = {
        lt: (
            ttnn.from_torch(m.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
            if m is not None
            else None
        )
        for lt, m in hf_encoder_masks.items()
    }
    # Decoder masks: bidirectional over [encoder | canvas]. Additive zero = full attention.
    decoder_mask_dict = {
        "full_attention": torch.zeros(B, 1, canvas_len, encoder_len + canvas_len, dtype=torch.bfloat16),
        "sliding_attention": torch.zeros(B, 1, canvas_len, encoder_len + canvas_len, dtype=torch.bfloat16),
    }
    tt_decoder_masks = {
        lt: ttnn.from_torch(m, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for lt, m in decoder_mask_dict.items()
    }

    # ---- TT forward ----------------------------------------------------------------------
    tt_input_ids = ttnn.from_torch(input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_decoder_input_ids = ttnn.from_torch(
        decoder_input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    tt_logits = tt_model(
        input_ids=tt_input_ids,
        position_ids=encoder_position_ids,
        encoder_attention_masks=tt_encoder_masks,
        decoder_input_ids=tt_decoder_input_ids,
        decoder_position_ids=decoder_position_ids,
        decoder_attention_masks=tt_decoder_masks,
        self_conditioning_signal=None,
    )
    tt_logits_torch = local_device_to_torch(tt_logits)
    if tt_logits_torch.ndim == 4 and tt_logits_torch.shape[0] == 1:
        tt_logits_torch = tt_logits_torch.squeeze(0)

    logger.info(f"hf_logits: {hf_logits.shape}, tt_logits: {tt_logits_torch.shape}")
    assert_quality(hf_logits, tt_logits_torch, pcc=PCC_THRESHOLD)

    abs_diff = (hf_logits - tt_logits_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(hf_logits, tt_logits_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
