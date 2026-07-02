# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end decoder parity: TT ``DiffusionGemmaDecoderModel`` vs the actual
HF ``DiffusionGemmaDecoderModel``.

Setup: build HF encoder + decoder, run HF encoder to fill a DynamicCache, then
compare the HF decoder forward against our TT decoder fed the same per-layer
K/V tensors. This isolates the decoder logic (self-conditioning + cross-attn +
RoPE routing + final norm).

    pytest models/tt_dit/tests/models/diffusion_gemma/test_text_decoder.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma._state_utils import per_layer_moe_substates
from ....models.transformers.diffusion_gemma.text_decoder import DiffusionGemmaDecoderModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch, typed_tensor
from ....utils.test import line_params, ring_params

# Text decoder = embed + self_conditioning + N × DiffusionGemmaLayer (with cross-attn to
# encoder KV cache) + final norm. Extrapolating from the encoder layer test (PCC 99.94%,
# max abs 0.53). Cross-attn concat may add slight drift; not yet run.
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
def test_decoder_text_model(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    encoder_len: int,
    canvas_len: int,
) -> None:
    """TT DiffusionGemmaDecoderModel vs HF DiffusionGemmaDecoderModel."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaConfig
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaDecoderModel as HFDecoderModel,
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
    )
    hf_config = DiffusionGemmaConfig(text_config=text_config, vision_config=None)
    B = 1

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if text_config.num_global_key_value_heads % tp_factor != 0:
        pytest.skip(f"num_global_kv_heads doesn't divide tp_factor={tp_factor}")

    # HF encoder (text-only path — no vision tower in this config) + decoder.
    # `DiffusionGemmaEncoderModel` requires a vision_config; build a config with vision_config=None
    # by going through the text-only encoder directly.
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaEncoderTextModel as HFEncoderTextModel,
    )

    hf_encoder = HFEncoderTextModel(text_config).to(dtype).eval()
    hf_decoder = HFDecoderModel(hf_config).to(dtype).eval()
    # Tie encoder ↔ decoder layer weights as HF does (`DiffusionGemmaModel._tied_weights_keys`).
    hf_decoder.embed_tokens.weight = hf_encoder.embed_tokens.weight
    for li in range(text_config.num_hidden_layers):
        # Tied weights: norm + all layer-internal weights + scales + expert weights.
        hf_decoder.layers[li].load_state_dict(hf_encoder.layers[li].state_dict(), strict=False)
    hf_decoder.norm.weight = hf_encoder.norm.weight

    input_ids = torch.randint(low=1, high=text_config.vocab_size, size=(B, encoder_len), dtype=torch.long)
    decoder_input_ids = torch.randint(low=1, high=text_config.vocab_size, size=(B, canvas_len), dtype=torch.long)
    position_ids = torch.arange(encoder_len, dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        enc_out = hf_encoder(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            past_key_values=None,
        )
        past_key_values = enc_out.past_key_values

        # Decoder mask (bidirectional, full attention) over kv_length = encoder_len + canvas_len
        # and q_length = canvas_len. Additive zero = attend to all.
        decoder_mask_dict = {
            "full_attention": torch.zeros(B, 1, canvas_len, encoder_len + canvas_len, dtype=dtype),
            "sliding_attention": torch.zeros(B, 1, canvas_len, encoder_len + canvas_len, dtype=dtype),
        }
        hf_dec_out = hf_decoder(
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            self_conditioning_logits=None,
            self_conditioning_mask=None,
            decoder_attention_mask=decoder_mask_dict,
            decoder_position_ids=torch.arange(encoder_len, encoder_len + canvas_len, dtype=torch.long).unsqueeze(0),
        )
        hf_last = hf_dec_out.last_hidden_state

    # Extract per-layer K/V from HF DynamicCache for TT consumption.
    per_layer_kv_torch: list[tuple[torch.Tensor, torch.Tensor]] = []
    for li in range(text_config.num_hidden_layers):
        layer = past_key_values.layers[li]
        per_layer_kv_torch.append((layer.keys.detach().clone(), layer.values.detach().clone()))

    # TT setup.
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    hf_dec_state = hf_decoder.state_dict()
    moe_state_dicts = per_layer_moe_substates(hf_dec_state, num_layers=text_config.num_hidden_layers)

    tt_decoder = DiffusionGemmaDecoderModel(
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
        moe_state_dicts=moe_state_dicts,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_links=num_links,
        topology=topology,
    )
    tt_decoder.load_state_dict(hf_dec_state)

    # Upload inputs.
    tt_decoder_input_ids = ttnn.from_torch(
        decoder_input_ids, device=mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    decoder_position_ids = torch.arange(encoder_len, encoder_len + canvas_len, dtype=torch.long).unsqueeze(0)

    # Upload per-layer encoder K/V, sharded on the kv-head axis along the TP mesh axis.
    tt_encoder_kv: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
    for k, v in per_layer_kv_torch:
        tt_k = typed_tensor(k, dtype=ttnn.bfloat16, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
        tt_v = typed_tensor(v, dtype=ttnn.bfloat16, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
        tt_encoder_kv.append((tt_k, tt_v))

    tt_dec_masks = {
        lt: ttnn.from_torch(m.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        for lt, m in decoder_mask_dict.items()
    }

    tt_h = tt_decoder(
        tt_decoder_input_ids,
        decoder_position_ids,
        encoder_kv_cache=tt_encoder_kv,
        decoder_attention_masks=tt_dec_masks,
        self_conditioning_signal=None,
    )
    tt_h_torch = local_device_to_torch(tt_h).squeeze(0)

    logger.info(f"hf_last: {hf_last.shape}, tt_h: {tt_h_torch.shape}")
    assert_quality(hf_last, tt_h_torch, pcc=PCC_THRESHOLD)

    abs_diff = (hf_last - tt_h_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(hf_last, tt_h_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
