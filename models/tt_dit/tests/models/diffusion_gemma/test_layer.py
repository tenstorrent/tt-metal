# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TT ``DiffusionGemmaLayer`` (encoder + decoder modes) vs the
actual ``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaEncoderTextLayer``
/ ``DiffusionGemmaDecoderTextLayer``.

Tiny config (8 experts, 256 hidden, 64 intermediate) to keep MoE memory bounded.

    pytest models/tt_dit/tests/models/diffusion_gemma/test_layer.py -s
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
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params, ring_params

# Full layer = attention (5 matmuls + SDPA) + dense MLP (3 matmuls) + MoE branch (router +
# sparse_matmul) + 7 RMSNorms + residual adds + TP all-gathers. Observed: PCC 99.9465%
# (sliding) / 99.9425% (full), max abs 0.336 / 0.529 (full has larger head_dim=512 hence
# larger accumulator). Tight to observed with modest headroom.
PCC_THRESHOLD = 0.999
ALLCLOSE_ATOL = 5.5e-1
ALLCLOSE_RTOL = 5e-2


def _build_tiny_config(num_layers: int = 6, hidden: int = 2816):
    """Construct a real DiffusionGemmaTextConfig matching the sizes used by
    ``models/demos/gemma4/tests/unit/test_moe.py``.

    The demos/gemma4 sparse_matmul kernel is compiled for shapes derived from the
    real Gemma4 hyperparameters; substantially smaller values (e.g. hidden=256,
    moe_intermediate_size=64) cause the compiled kernel binary to mismatch the
    tensor shapes at runtime and the op hangs.  Weights are still random-init
    so no HF checkpoint download is required.

    ``num_layers`` is kept low (6) purely so we don't allocate 30 layers of experts
    for a single-layer test.
    """
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaTextConfig

    return DiffusionGemmaTextConfig(
        hidden_size=hidden,
        intermediate_size=2816,
        moe_intermediate_size=2112,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_global_key_value_heads=2,
        head_dim=256,
        global_head_dim=512,
        num_experts=8,
        top_k_experts=4,
        num_hidden_layers=num_layers,
        sliding_window=1024,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=8192,
    )


def _moe_substate(layer_state: dict) -> dict:
    """Pluck router/experts entries from a flat single-layer HF state-dict."""
    # Wraps per_layer_moe_substates for the single-layer case: pretend this layer is
    # ``layers.0.*`` so we can reuse the same helper used by the multi-layer paths.
    rekeyed = {f"layers.0.{k}": v for k, v in layer_state.items()}
    return per_layer_moe_substates(rekeyed, num_layers=1)[0]


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((2, 4), 0, 1, ring_params, ttnn.Topology.Ring, id="wh_t3k_tp2"),
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
@pytest.mark.parametrize("seq_len", [128])
def test_encoder_layer(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    layer_type: str,
    layer_idx_in_config: int,
    seq_len: int,
) -> None:
    """TT DiffusionGemmaLayer (encoder mode) vs HF DiffusionGemmaEncoderTextLayer."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaEncoderTextLayer
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

    # HF reference layer.
    hf_layer = DiffusionGemmaEncoderTextLayer(hf_config, layer_idx=layer_idx_in_config).to(dtype).eval()

    # RoPE tables (full-dim) from HF.
    hf_rope = Gemma4TextRotaryEmbedding(hf_config).eval()
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden_states = torch.randn(B, seq_len, hf_config.hidden_size, dtype=dtype)

    # Causal mask matching what HF eager attention will see.
    causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype)
    causal_mask = causal_mask + torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(dtype)

    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(hidden_states, position_ids, layer_type=layer_type)
        torch_out = hf_layer(
            hidden_states=hidden_states,
            position_embeddings=(hf_cos_full, hf_sin_full),
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=None,
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

    # Strip out MoE substate (constructed at __init__) from the HF state.
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

    tt_hidden = bf16_tensor(hidden_states.unsqueeze(0), device=mesh_device)
    tt_attn_mask = ttnn.from_torch(
        causal_mask.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_out, _, _ = tt_layer(tt_hidden, tt_cos, tt_sin, attention_mask=tt_attn_mask, encoder_kv=None)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(
        torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), f"allclose failed: max abs={abs_diff.max().item():.3e}"
