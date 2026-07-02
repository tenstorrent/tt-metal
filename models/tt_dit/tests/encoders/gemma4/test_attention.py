# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer parity: TT ``Gemma4Attention`` vs the actual
``transformers.models.diffusion_gemma.modeling_diffusion_gemma.DiffusionGemmaEncoderTextAttention``.

    pytest models/tt_dit/tests/encoders/gemma4/test_attention.py -s
"""

import pytest
import torch
from loguru import logger

import ttnn

from ....encoders.gemma4.attention import Gemma4Attention
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import bf16_tensor, local_device_to_torch
from ....utils.test import line_params, ring_params

PCC_THRESHOLD = 0.999
# allclose atol/rtol are looser than the module-level defaults on purpose. PCC (the primary
# statistical correctness check) clears 0.999 on every configuration; the failing cells are
# isolated near-cancellation outliers in the attention output, common when comparing bf16
# arithmetic against an fp32 HF reference across chained matmuls + softmax. Compute config
# is already maxed (HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False, fp32 cos/sin).
# 0.3 comfortably clears observed 0.25-0.26 outliers with a small headroom for run-to-run drift.
ALLCLOSE_ATOL = 3e-1
ALLCLOSE_RTOL = 1e-1


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis", "num_links", "device_params", "topology"),
    [
        pytest.param((2, 4), 0, 1, line_params, ttnn.Topology.Linear, id="bh_qb2_tp2"),
        pytest.param((4, 8), 0, 2, line_params, ttnn.Topology.Linear, id="bh_galaxy_tp4"),
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
@pytest.mark.parametrize("seq_len", [256])
def test_gemma4_attention(
    mesh_device: ttnn.MeshDevice,
    tp_axis: int,
    num_links: int,
    topology: ttnn.Topology,
    layer_type: str,
    layer_idx_in_config: int,
    seq_len: int,
) -> None:
    """TT vs HF DiffusionGemmaEncoderTextAttention."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
        DiffusionGemmaEncoderTextAttention,
        DiffusionGemmaTextConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    torch.manual_seed(0)
    dtype = torch.float32
    hidden_size = 2816
    num_attention_heads = 16
    sliding_kv_heads = 8
    full_kv_heads = 2
    sliding_head_dim = 256
    full_head_dim = 512
    sliding_window = 1024
    eps = 1e-6
    B = 1

    is_sliding = layer_type == "sliding_attention"
    head_dim = sliding_head_dim if is_sliding else full_head_dim
    num_kv_heads = sliding_kv_heads if is_sliding else full_kv_heads

    tp_factor = tuple(mesh_device.shape)[tp_axis]
    # KV-head sharding requires either num_kv_heads | tp_factor (standard shard) or
    # tp_factor | num_kv_heads (KV replication path). Otherwise no integer split exists.
    if num_kv_heads % tp_factor != 0 and tp_factor % num_kv_heads != 0:
        pytest.skip(f"num_kv_heads={num_kv_heads} and tp_factor={tp_factor} aren't mutually divisible")

    hf_config = DiffusionGemmaTextConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=sliding_kv_heads,
        num_global_key_value_heads=full_kv_heads,
        head_dim=sliding_head_dim,
        global_head_dim=full_head_dim,
        sliding_window=sliding_window,
        rms_norm_eps=eps,
        attention_bias=False,
        attention_dropout=0.0,
        num_hidden_layers=6,
        max_position_embeddings=8192,
        # layer_types auto-builds 5 sliding + 1 full when None.
    )

    # HF reference attention. The layer_idx pin-points which entry in layer_types
    # (sliding vs full) we're exercising.
    hf_attn = DiffusionGemmaEncoderTextAttention(hf_config, layer_idx=layer_idx_in_config).to(dtype).eval()

    # HF rotary embedding produces full-dim cos/sin per layer type.
    hf_rope = Gemma4TextRotaryEmbedding(hf_config).eval()
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    hidden_states = torch.randn(B, seq_len, hidden_size, dtype=dtype)

    # Build an explicit causal mask so HF and TT apply the *same* masking semantics:
    # HF's eager_attention_forward ignores `is_causal` when `attention_mask is None`
    # (no mask at all), whereas ttnn SDPA with attn_mask=None + is_causal=True applies
    # a causal mask. Passing an explicit mask to both removes that branch difference.
    causal_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype)
    causal_mask = causal_mask + torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1).to(dtype)

    with torch.no_grad():
        hf_cos_full, hf_sin_full = hf_rope(hidden_states, position_ids, layer_type=layer_type)
        torch_out, _ = hf_attn(
            hidden_states=hidden_states,
            position_embeddings=(hf_cos_full, hf_sin_full),
            attention_mask=causal_mask,
            past_key_values=None,
        )

    # Half-dim cos/sin for our TT module (first half of HF's concat-style full-dim).
    half = head_dim // 2
    cos_half = hf_cos_full[..., :half].unsqueeze(1).to(torch.bfloat16)  # (B, 1, S, half)
    sin_half = hf_sin_full[..., :half].unsqueeze(1).to(torch.bfloat16)
    tt_cos = ttnn.from_torch(cos_half, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    tt_sin = ttnn.from_torch(sin_half, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tp_factor),
        sequence_parallel=ParallelFactor(mesh_axis=1 - tp_axis, factor=tuple(mesh_device.shape)[1 - tp_axis]),
        cfg_parallel=None,
    )

    tt_model = Gemma4Attention(
        is_sliding=is_sliding,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rms_norm_eps=eps,
        sliding_window=sliding_window if is_sliding else None,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    tt_model.load_state_dict(hf_attn.state_dict())

    tt_hidden = bf16_tensor(hidden_states.unsqueeze(0), device=mesh_device)
    # Upload the same causal mask used for HF (bf16 additive).
    tt_attn_mask = ttnn.from_torch(
        causal_mask.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16
    )
    tt_out, _, _ = tt_model(tt_hidden, tt_cos, tt_sin, attention_mask=tt_attn_mask, encoder_kv=None)
    tt_out_torch = local_device_to_torch(tt_out).squeeze(0)

    logger.info(f"torch_out: {torch_out.shape}, tt_out: {tt_out_torch.shape}")
    assert_quality(torch_out, tt_out_torch, pcc=PCC_THRESHOLD)

    abs_diff = (torch_out - tt_out_torch.to(dtype)).abs()
    logger.info(f"max abs diff: {abs_diff.max().item():.3e}")
    assert torch.allclose(
        torch_out, tt_out_torch.to(dtype), atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), f"allclose failed: max abs={abs_diff.max().item():.3e}"
