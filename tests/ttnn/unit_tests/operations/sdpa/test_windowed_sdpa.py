# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Post-commit smoke for windowed SDPA (cu_window_seqlens): catches kernel breaks like #45015.

Full sweep: tests/ttnn/nightly/unit_tests/operations/sdpa/test_windowed_sdpa.py.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.unit_tests.operations.sdpa.windowed_sdpa_test_utils import (
    reference_sdpa,
    sdpa_configs,
    to_cu_tensor,
    windowed_mask,
)


def run_windowed_sdpa(device, q, k, v, cu_window_seqlens, is_causal, dtype=ttnn.bfloat16, **op_kwargs):
    tt = lambda t: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.transformer.scaled_dot_product_attention(
        tt(q),
        tt(k),
        tt(v),
        is_causal=is_causal,
        scale=q.shape[-1] ** -0.5,
        cu_window_seqlens=to_cu_tensor(cu_window_seqlens, device),
        **op_kwargs,
    )
    return ttnn.to_torch(out_tt).to(torch.float32)


@pytest.mark.parametrize(
    "is_causal, fp32_dest_acc_en, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens",
    [
        (False, True, ttnn.bfloat16, 0.99, 8, 1024, 128, [0, 200, 480, 730, 1024]),
        (True, True, ttnn.bfloat16, 0.99, 8, 1024, 128, [0, 200, 480, 730, 1024]),
        (True, False, ttnn.bfloat16, 0.99, 1, 129, 64, [0, 32, 97, 129]),
        # bf8: what Qwen2.5-VL feeds the op.
        (False, False, ttnn.bfloat8_b, 0.98, 8, 96, 64, [0, 33, 64, 96]),
        (True, True, ttnn.bfloat8_b, 0.98, 8, 1024, 128, [0, 128, 256, 384, 512, 640, 768, 896, 1024]),
    ],
    ids=[
        "bidir-fp32acc-bf16-s1024_w4_straddle",
        "causal-fp32acc-bf16-s1024_w4_straddle",
        "causal-no_fp32acc-bf16-s129_partial_tile",
        "bidir-no_fp32acc-bf8-s96_padded_chunk",
        "causal-fp32acc-bf8-s1024_w8_aligned",
    ],
)
def test_windowed_sdpa_smoke(
    device, is_causal, fp32_dest_acc_en, dtype, pcc_threshold, num_heads, seq_len, chunk, cu_window_seqlens
):
    torch.manual_seed(42)
    q, k, v = (torch.randn(1, num_heads, seq_len, 128, dtype=torch.bfloat16) for _ in range(3))
    program_config, compute_kernel_config = sdpa_configs(device, chunk, chunk, fp32_dest_acc_en)

    out = run_windowed_sdpa(
        device,
        q,
        k,
        v,
        cu_window_seqlens,
        is_causal,
        dtype=dtype,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal), 128**-0.5)
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"

    passing, pcc = comp_pcc(gt, out, pcc_threshold)
    logger.info(f"windowed SDPA causal={is_causal} dtype={dtype} s={seq_len} windows={cu_window_seqlens} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"


@pytest.mark.parametrize(
    "is_causal, fp32_dest_acc_en, seq_len, q_chunk, k_chunk, cu_window_seqlens",
    [
        # Windows shorter than a tile, including a 1-token window.
        (True, True, 256, 64, 64, [0, 5, 32, 33, 60, 256]),
        # Empty windows (repeated boundaries) between real ones.
        (False, False, 256, 64, 128, [0, 100, 100, 200, 200, 256]),
    ],
    ids=["causal-fp32acc-sub_tile", "bidir-no_fp32acc-empty_windows"],
)
def test_windowed_sdpa_edges(device, is_causal, fp32_dest_acc_en, seq_len, q_chunk, k_chunk, cu_window_seqlens):
    torch.manual_seed(1234)
    q, k, v = (torch.randn(1, 4, seq_len, 64, dtype=torch.bfloat16) for _ in range(3))
    program_config, compute_kernel_config = sdpa_configs(device, q_chunk, k_chunk, fp32_dest_acc_en)

    out = run_windowed_sdpa(
        device,
        q,
        k,
        v,
        cu_window_seqlens,
        is_causal,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal), 64**-0.5)

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed SDPA edges causal={is_causal} s={seq_len} windows={cu_window_seqlens} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"
    # Every row attends to itself, so output must be finite.
    assert torch.isfinite(out).all(), "non-finite output rows"


def test_windowed_sdpa_q_token_offset(device):
    """Q sharded on the sequence, offset passed as a tensor; concatenated shards must match unsharded."""
    torch.manual_seed(42)
    seq_len, num_shards, cu_window_seqlens = 256, 4, [0, 96, 160, 256]
    shard_rows = seq_len // num_shards
    q, k, v = (torch.randn(1, 8, seq_len, 128, dtype=torch.bfloat16) for _ in range(3))
    program_config, compute_kernel_config = sdpa_configs(device, 32, 32)

    shards = []
    for shard in range(num_shards):
        offset = shard * shard_rows
        offset_tt = ttnn.from_torch(
            torch.tensor([offset], dtype=torch.int32), device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
        )
        shards.append(
            run_windowed_sdpa(
                device,
                q[:, :, offset : offset + shard_rows, :].contiguous(),
                k,
                v,
                cu_window_seqlens,
                is_causal=True,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config,
                windowed_q_token_offset_tensor=offset_tt,
            )
        )
    out = torch.cat(shards, dim=-2)
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal=True), 128**-0.5)

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed SDPA q-offset shards={num_shards} windows={cu_window_seqlens} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"


def test_windowed_causal_sdpa_gqa(device):
    """Issue #57920: Qwen3-Embedding-4B shape (32 Q / 8 KV heads), non-tile-multiple windows."""
    torch.manual_seed(0)
    windows = [300, 212, 512]
    seq_len = sum(windows)
    cu_window_seqlens = [0] + torch.tensor(windows).cumsum(0).tolist()
    q = torch.randn(1, 32, seq_len, 128, dtype=torch.bfloat16)
    k, v = (torch.randn(1, 8, seq_len, 128, dtype=torch.bfloat16) for _ in range(2))
    program_config, compute_kernel_config = sdpa_configs(device, 64, 256)

    out = run_windowed_sdpa(
        device,
        q,
        k,
        v,
        cu_window_seqlens,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal=True), 128**-0.5)
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed causal SDPA GQA windows={windows} pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"

    last_rows = [end - 1 for end in cu_window_seqlens[1:]]
    passing, pcc = comp_pcc(gt[:, :, last_rows, :], out[:, :, last_rows, :], 0.99)
    assert passing, f"last-token rows PCC below threshold: {pcc}"


def test_windowed_sdpa_output_concat_heads(device):
    torch.manual_seed(7)
    b, nqh, nkv, seq_len, dh = 1, 8, 2, 512, 128
    cu_window_seqlens = [0, 150, 300, 512]
    q = torch.randn(b, nqh, seq_len, dh, dtype=torch.bfloat16)
    k, v = (torch.randn(b, nkv, seq_len, dh, dtype=torch.bfloat16) for _ in range(2))
    program_config, compute_kernel_config = sdpa_configs(device, 128, 128, fp32_dest_acc_en=False)

    out = run_windowed_sdpa(
        device,
        q,
        k,
        v,
        cu_window_seqlens,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        output_concat_heads=True,
    )
    gt = reference_sdpa(q, k, v, windowed_mask(seq_len, cu_window_seqlens, is_causal=True), dh**-0.5)
    gt = gt.permute(0, 2, 1, 3).reshape(b, 1, seq_len, nqh * dh)
    assert out.shape == gt.shape, f"shape mismatch: {out.shape} vs {gt.shape}"

    passing, pcc = comp_pcc(gt, out, 0.99)
    logger.info(f"windowed SDPA concat-heads causal pcc={pcc}")
    assert passing, f"PCC below threshold: {pcc}"
