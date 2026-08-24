# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level test for ttnn.transformer.fused_qkv_sdpa, which reads Q, K and V out of one fused
projection output so the head split never runs as its own op.

Two things are checked, and the second is the one that matters. Against torch SDPA it shows the op
computes attention; against `nlp_create_qkv_heads` + `scaled_dot_product_attention` it shows the
strided reader addresses the fused tensor the same way the split op does. A base or stride off by
one head still scores well against torch on random data, so only the second comparison pins the
addressing down.

The reader kernel is JIT-built at invocation, so running on hardware is the only way to catch a
break in it.
"""

import torch
import pytest
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def _fused_qkv(q, k, v):
    """Pack per-head q/k/v into the [B, 1, S, 3*H*DH] layout a fused qkv matmul produces."""
    b, h, s, dh = q.shape
    # [B, H, S, DH] -> [B, 1, S, H*DH]: the head axis moves behind the sequence.
    flat = lambda t: t.permute(0, 2, 1, 3).reshape(b, 1, s, h * dh)
    return torch.cat([flat(q), flat(k), flat(v)], dim=-1)


@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim, q_chunk, k_chunk",
    [
        (576, 16, 64, 192, 576),  # the Janus-Pro vision tower's geometry
        (256, 8, 64, 128, 256),
        (128, 4, 32, 32, 128),  # head_dim at exactly one tile
        (512, 16, 128, 128, 256),  # k_chunk < seq_len: several inner iterations
    ],
    ids=["janus_576x16x64", "s256_h8", "s128_dh32", "s512_dh128"],
)
@pytest.mark.parametrize(
    "dtype, pcc_threshold",
    [
        (ttnn.bfloat16, 0.99),
        # bfloat8_b is what the tower actually feeds the op.
        (ttnn.bfloat8_b, 0.98),
    ],
    ids=["bf16", "bf8"],
)
def test_fused_qkv_sdpa_vs_torch(device, dtype, pcc_threshold, seq_len, num_heads, head_dim, q_chunk, k_chunk):
    torch.manual_seed(42)
    b = 1
    scale = head_dim**-0.5

    q = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)

    qkv_tt = ttnn.from_torch(_fused_qkv(q, k, v), device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    out_tt = ttnn.transformer.fused_qkv_sdpa(
        qkv_tt,
        num_heads,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)

    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), is_causal=False, scale=scale
    )

    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"
    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(f"fused_qkv_sdpa vs torch dtype={dtype} s={seq_len} h={num_heads} dh={head_dim} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"


@pytest.mark.parametrize(
    "seq_len, num_heads, head_dim, q_chunk, k_chunk",
    [
        (576, 16, 64, 192, 576),
        (256, 8, 64, 128, 256),
    ],
    ids=["janus_576x16x64", "s256_h8"],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.bfloat8_b], ids=["bf16", "bf8"])
def test_fused_qkv_sdpa_matches_split_path(device, dtype, seq_len, num_heads, head_dim, q_chunk, k_chunk):
    """The fused reader must land on the same tiles nlp_create_qkv_heads hands to SDPA.

    Both paths start from one fused tensor and run the same compute kernel at the same fidelity, so
    the only thing that can differ is addressing -- which makes near-equality the right bar. A wrong
    base or stride reads a different head and shows up here even though torch would not catch it.
    """
    torch.manual_seed(0)
    b = 1
    scale = head_dim**-0.5

    q = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, head_dim, dtype=torch.bfloat16)
    qkv_tt = ttnn.from_torch(_fused_qkv(q, k, v), device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    fused = ttnn.to_torch(
        ttnn.transformer.fused_qkv_sdpa(
            qkv_tt,
            num_heads,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    ).to(torch.float32)

    q_h, k_h, v_h = ttnn.experimental.nlp_create_qkv_heads(
        qkv_tt,
        num_heads=num_heads,
        num_kv_heads=num_heads,
        transpose_k_heads=False,
    )
    split = ttnn.to_torch(
        ttnn.transformer.scaled_dot_product_attention(
            q_h,
            k_h,
            v_h,
            is_causal=False,
            scale=scale,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    ).to(torch.float32)

    assert fused.shape == split.shape, f"shape mismatch: {fused.shape} vs {split.shape}"
    passing, pcc = comp_pcc(split, fused, 0.9999)
    logger.info(f"fused_qkv_sdpa vs split path dtype={dtype} s={seq_len} h={num_heads} pcc={pcc}")
    assert passing, f"fused reader disagrees with nlp_create_qkv_heads: PCC {pcc}"
