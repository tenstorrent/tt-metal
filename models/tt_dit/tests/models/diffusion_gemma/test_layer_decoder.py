# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decoder-mode parity: TT ``DiffusionGemmaLayer`` (decoder, with encoder KV cache) vs
the actual ``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaDecoderTextLayer``.

The encoder KV cache is populated with synthetic random tensors (representing
post-RoPE encoder K and post-v_norm V — matching what the real encoder would
produce). Both HF and TT consume the *same* cache tensors so the layer's
cross-attention path is what's being validated.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_layer_decoder.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.diffusion_gemma._state_utils import per_layer_moe_substates
from ....models.transformers.diffusion_gemma.layer import DiffusionGemmaLayer
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch, typed_tensor
from ....utils.test import ring_params

# Decoder layer = encoder layer with cross-attention concat over pre-populated encoder KV.
PCC_THRESHOLD = 0.9993
ALLCLOSE_ATOL = 4e-1
ALLCLOSE_RTOL = 3e-2


def _build_tiny_config(num_layers: int = 6):
    """Match sizes used by ``test_layer.py``: real Gemma4 shapes so the
    sparse_matmul kernel doesn't hang on shape-mismatched compiled binaries at tiny sizes.
    Weights are still random-init — no HF checkpoint required.
    """
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig

    return DiffusionGemmaTextConfig(
        hidden_size=2816,
        intermediate_size=2112,
        moe_intermediate_size=704,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_experts=128,
        top_k_experts=8,
        num_hidden_layers=num_layers,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=8192,
    )


def _moe_substate(layer_state: dict) -> dict:
    rekeyed = {f"layers.0.{k}": v for k, v in layer_state.items()}
    return per_layer_moe_substates(rekeyed, num_layers=1)[0]


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((1, 8), 1, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_1x8"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("layer_type", "layer_idx_in_config"),
    [
        pytest.param("sliding_attention", 0, id="sliding"),
        pytest.param("full_attention", 5, id="full"),
    ],
)
@pytest.mark.parametrize("canvas_len", [256])
@pytest.mark.parametrize("encoder_len", [32])
def test_decoder_layer(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    layer_type: str,
    layer_idx_in_config: int,
    canvas_len: int,
    encoder_len: int,
) -> None:
    """TT decoder layer vs HF DiffusionGemmaDecoderTextLayer (cross-attn to encoder cache)."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaDecoderTextLayer
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    torch.manual_seed(0)
    dtype = torch.float32

    hf_config = _build_tiny_config()
    B = 1
    is_sliding = layer_type == "sliding_attention"
    head_dim = hf_config.head_dim if is_sliding else hf_config.global_head_dim
    num_kv_heads = hf_config.num_key_value_heads if is_sliding else hf_config.num_global_key_value_heads

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    if num_kv_heads % tp_factor != 0:
        pytest.skip(f"num_kv_heads={num_kv_heads} doesn't divide tp_factor={tp_factor}")

    # HF reference decoder layer.
    hf_layer = DiffusionGemmaDecoderTextLayer(hf_config, layer_idx=layer_idx_in_config).to(dtype).eval()

    # Synthetic encoder cache: random K (post-RoPE) and V (post-v_norm).
    enc_k = torch.randn(B, num_kv_heads, encoder_len, head_dim, dtype=dtype)
    enc_v = torch.randn(B, num_kv_heads, encoder_len, head_dim, dtype=dtype)

    # Populate HF DynamicCache for the relevant layer.
    cache = DynamicCache(config=hf_config)
    # The cache only holds entries for indices we update; ensure the slot exists.
    cache.update(enc_k.clone(), enc_v.clone(), layer_idx_in_config)

    # RoPE tables for the canvas (decoder positions continue after the encoder cache).
    hf_rope = Gemma4TextRotaryEmbedding(hf_config).eval()
    canvas_position_ids = torch.arange(encoder_len, encoder_len + canvas_len, dtype=torch.long).unsqueeze(0)
    canvas_hidden = torch.randn(B, canvas_len, hf_config.hidden_size, dtype=dtype)

    # Bidirectional decoder mask: canvas queries attend to all (encoder cache + canvas).
    # Shape (B, 1, canvas_len, encoder_len + canvas_len), additive (zeros).
    decoder_mask = torch.zeros(B, 1, canvas_len, encoder_len + canvas_len, dtype=dtype)

    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(canvas_hidden, canvas_position_ids, layer_type=layer_type)
        torch_out = hf_layer(
            hidden_states=canvas_hidden,
            position_embeddings=(hf_cos_full, hf_sin_full),
            attention_mask=decoder_mask,
            position_ids=canvas_position_ids,
            past_key_values=cache,
        )

    # Half-dim cos/sin for TT.
    half = head_dim // 2
    cos_half = hf_cos_full[..., :half].unsqueeze(1).to(torch.bfloat16)
    sin_half = hf_sin_full[..., :half].unsqueeze(1).to(torch.bfloat16)
    tt_cos = ttnn.from_torch(cos_half, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin_half, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    layer_state = hf_layer.state_dict()
    moe_state = _moe_substate(layer_state)

    tt_layer = DiffusionGemmaLayer(
        is_sliding=is_sliding,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.intermediate_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        sliding_window=hf_config.sliding_window if is_sliding else None,
        num_experts=hf_config.num_experts,
        top_k_experts=hf_config.top_k_experts,
        moe_intermediate_size=hf_config.moe_intermediate_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        moe_state_dict=moe_state,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
        num_links=num_links,
        topology=topology,
    )
    tt_layer.load_state_dict(layer_state)

    # Upload encoder K/V, sharded on the kv-head axis along the TP mesh axis.
    tt_enc_k = typed_tensor(enc_k, dtype=ttnn.bfloat16, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)
    tt_enc_v = typed_tensor(enc_v, dtype=ttnn.bfloat16, device=mesh_device, mesh_axis=tp_axis, shard_dim=1)

    tt_hidden = bf16_tensor(canvas_hidden.unsqueeze(0), device=mesh_device)
    tt_mask = ttnn.from_torch(
        decoder_mask.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )

    tt_out, _, _ = tt_layer(tt_hidden, tt_cos, tt_sin, attention_mask=tt_mask, encoder_kv=(tt_enc_k, tt_enc_v))
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(
        torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), f"allclose failed: max abs={abs_diff.max().item():.3e}"
