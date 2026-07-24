# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level smoke test for windowed (block-diagonal) attention via
ttnn.transformer.scaled_dot_product_attention(..., cu_window_seqlens=...).

This intentionally lives next to the other SDPA op tests (instead of only under
models/demos/qwen25_vl/) so that any change to the shared SDPA kernel helpers
(e.g. dataflow_common.hpp / write_block) exercises the windowed writer kernel in
a pre-merge / per-commit gate. Device kernels are JIT-built at op invocation, so
running the op on hardware is the only way to catch a kernel-signature break -
which is exactly what slipped through in #45015 and broke Qwen2.5-VL nightly.

Correctness is checked against torch SDPA with the equivalent block-diagonal
window mask.
"""

import os
import torch
import pytest
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def windowed_mask(seq_len, cu_window_seqlens):
    """Block-diagonal mask: token i attends only to tokens in the same window."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
    for i in range(1, len(cu_window_seqlens)):
        start, end = cu_window_seqlens[i - 1], cu_window_seqlens[i]
        mask[start:end, start:end] = 0.0
    return mask


@pytest.mark.parametrize(
    "seq_len, chunk, cu_window_seqlens",
    [
        (128, 32, [0, 64, 128]),  # two equal tile-aligned windows
        (128, 32, [0, 32, 96, 128]),  # three uneven windows
        (256, 32, [0, 64, 128, 256]),  # larger sequence
        (96, 64, [0, 33, 64, 96]),  # sequence padded to chunk size; windowed mask owns padding
        (129, 64, [0, 32, 97, 129]),  # partial final tile plus chunk padding
    ],
    ids=["s128_w2", "s128_w3", "s256_w3", "s96_padded_chunk", "s129_partial_tile"],
)
@pytest.mark.parametrize("num_heads", [1, 8])
@pytest.mark.parametrize(
    "dtype, pcc_threshold",
    [
        (ttnn.bfloat16, 0.99),
        # bfloat8_b mirrors the dtype Qwen2.5-VL actually feeds the op
        # (vision_attention.py typecasts q/k/v to bf8 before the call); looser
        # PCC accounts for the reduced input precision.
        (ttnn.bfloat8_b, 0.98),
    ],
    ids=["bf16", "bf8"],
)
# Both dest-accumulation modes are covered: fp32_dest_acc_en selects different compute paths
# (False -> streaming on Blackhole, True -> standard), so both must stay correct.
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32acc", "no_fp32acc"])
def test_windowed_sdpa_smoke(
    device, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens, fp32_dest_acc_en
):
    torch.manual_seed(42)
    b, dh = 1, 128
    scale = dh**-0.5

    q = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    k = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)
    v = torch.randn(b, num_heads, seq_len, dh, dtype=torch.bfloat16)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )

    q_tt = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    k_tt = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    v_tt = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    cu_tt = ttnn.from_torch(
        torch.tensor(cu_window_seqlens, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt,
        k_tt,
        v_tt,
        is_causal=False,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        cu_window_seqlens=cu_tt,
    )
    out = ttnn.to_torch(out_tt).to(torch.float32)

    mask = windowed_mask(seq_len, cu_window_seqlens).unsqueeze(0).unsqueeze(0)
    gt = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k.to(torch.float32), v.to(torch.float32), attn_mask=mask, scale=scale
    )

    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(f"windowed SDPA dtype={dtype} s={seq_len} heads={num_heads} windows={cu_window_seqlens} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"
