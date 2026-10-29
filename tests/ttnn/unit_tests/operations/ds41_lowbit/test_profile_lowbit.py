# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler bench (run_safe_pytest.sh --profile), one card, DeepSeek-V4.1 2x2 per-chip shapes (8x4 where noted):
the index score with bf16 vs bfloat4_b q / k, the writes (bfp_round of the index query / a chunk's keys, ckv_bfp4
bfloat8_b window records) next to what the model runs today, and sparse_sdpa's window as bf16 vs bfloat8_b records (the
compressed layers' kv_window; the window-only layers' kv), with plain fp8 rows (FP8_E4M3, no scale) as a lower bound of
an e4m3 + scale window. Each op runs WARM + REPS times; DEVICE KERNEL DURATION per op from the CSV."""

import importlib.util

import pytest
import torch

import ttnn

sp = importlib.util.spec_from_file_location("ref", "ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/reference.py")
ref = importlib.util.module_from_spec(sp)
sp.loader.exec_module(ref)

REPS = 5
DRAM = ttnn.DRAM_MEMORY_CONFIG
K, WIN, TOPK, H, KC = 512, 128, 640, 64, 128


def _ckc(fp32=True, fid=ttnn.MathFidelity.HiFi2):
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=False
    )


def _t(x, dt, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(x, dtype=dt, layout=layout, device=device, memory_config=DRAM)


@pytest.mark.parametrize("ratio, T", [(2, 27648), (1, 55296)])
def test_index_score(device, ratio, T):
    """indexer_score_dsa, 32 heads x 128, 512 queries (a 2x2 chip), the last chunk of 55296 tokens, HiFi2 + fp32 DEST:
    q / k bf16 (today) vs bfloat4_b (vs bfloat8_b)."""
    g = torch.Generator().manual_seed(0)
    Sq, nh, D = 512, 32, 128
    q = torch.randn(1, nh, Sq, D, generator=g)
    k = torch.randn(1, 1, T, D, generator=g)
    w = _t((torch.randn(1, 1, Sq, nh, generator=g) / 64), ttnn.bfloat16, device)
    cfg = ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=32, head_group_size=0)
    for dt in (ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b):
        tq, tk = _t(q, dt, device), _t(k, dt, device)
        for _ in range(1 + REPS):
            ttnn.bringup.indexer_score_dsa(
                tq,
                tk,
                w,
                chunk_start_idx=55296 - Sq,
                key_compression_ratio=ratio,
                program_config=cfg,
                compute_kernel_config=_ckc(),
                kv_len=T,
            ).deallocate()
        tq.deallocate()
        tk.deallocate()
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("rows", [13824, 27648])
def test_index_writes(device, rows):
    """Today the indexer tiles the chip's whole key slab every chunk (ROW_MAJOR -> TILE, [rows, 128] bf16); with a
    bfloat4_b key table it rounds only the chunk's new keys (512 / 1024 rows) and the query [1, 32, 512, 128]."""
    g = torch.Generator().manual_seed(0)
    slab = _t(torch.randn(1, 1, rows, 128, generator=g), ttnn.bfloat16, device, ttnn.ROW_MAJOR_LAYOUT)
    for _ in range(1 + REPS):
        ttnn.to_layout(slab, ttnn.TILE_LAYOUT, memory_config=DRAM).deallocate()
    new = _t(torch.randn(1, 1, 1024 if rows == 27648 else 512, 128, generator=g), ttnn.bfloat16, device)
    table = ttnn.from_torch(
        torch.zeros(1, 1, rows, 128), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM
    )
    for _ in range(1 + REPS):
        ttnn.bringup.bfp_round(new, out=table, row0=0)
    q = _t(torch.randn(1, 32, 512, 128, generator=g), ttnn.bfloat16, device)
    for _ in range(1 + REPS):
        ttnn.bringup.bfp_round(q, ttnn.bfloat4_b).deallocate()
    ttnn.synchronize_device(device)


def test_window_write(device):
    """The window table of a chunk (128 carry + 2048 rows): today bf16 TILE -> ROW_MAJOR; bfloat8_b records instead."""
    g = torch.Generator().manual_seed(0)
    n = 2176
    x = _t(torch.randn(1, 1, n, K, generator=g), ttnn.bfloat16, device)
    for _ in range(1 + REPS):
        ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM).deallocate()
    rec = ttnn.zeros([1, 1, n, 576], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    for _ in range(1 + REPS):
        ttnn.bringup.ckv_bfp4(x, rec, 0, dtype=ttnn.bfloat8_b)
    ttnn.synchronize_device(device)


def _idx(S, Tw, T, n_top, g):
    idx = torch.full((1, 1, S, WIN + n_top), -1, dtype=torch.int64)
    for s in range(S):
        idx[0, 0, s, :WIN] = torch.randperm(Tw, generator=g)[:WIN]
        if n_top:
            idx[0, 0, s, WIN:] = torch.randperm(T, generator=g)[:n_top]
    return idx


@pytest.mark.parametrize("S", [160, 512])
def test_sdpa_window(device, S):
    """64 heads x 512, S queries (512: a 2x2 chip; 160: an 8x4 chip at chunk 5120).
    Compressed layers: ROW_BFP4 (512 top-k records) with the bf16 window vs the bfloat8_b window records.
    Window-only layers (128 window keys): BF16 vs ROW_BFP8 vs FP8_E4M3 (plain fp8 rows, the e4m3 window's lower
    bound: no per-32 scale multiply)."""
    g = torch.Generator().manual_seed(0)
    T, Tw = 65536, 2176
    rows = ref.encode_rows(torch.randn(T, K, generator=g).bfloat16())
    win = torch.randn(Tw, K, generator=g).bfloat16()
    rec8 = ref.encode_rows(win, mbits=7)
    rm = ttnn.ROW_MAJOR_LAYOUT
    t_rows = _t(rows.view(1, 1, T, -1), ttnn.uint8, device, rm)
    t_win = _t(win.view(1, 1, Tw, K), ttnn.bfloat16, device, rm)
    t_win8 = _t(rec8.view(1, 1, Tw, 576), ttnn.uint8, device, rm)
    t_winfp8 = _t(win.float().view(1, 1, Tw, K), ttnn.fp8_e4m3, device, rm)
    sink = _t(torch.zeros(1, 1, 1, H).bfloat16(), ttnn.bfloat16, device, rm)
    q = _t((torch.randn(1, H, S, K, generator=g) * K**-0.5).bfloat16(), ttnn.bfloat16, device, rm)
    i_csa = _t(_idx(S, Tw, T, 512, g).to(torch.int32), ttnn.uint32, device, rm)
    i_win = _t(_idx(S, Tw, T, 0, g).to(torch.int32), ttnn.uint32, device, rm)
    common = dict(scale=1.0, k_chunk_size=KC, compute_kernel_config=_ckc(), attention_sink=sink)
    F = ttnn.bringup.SparseKVFormat
    runs = [
        lambda: ttnn.bringup.sparse_sdpa(q, t_rows, i_csa, K, kv_format=F.ROW_BFP4, kv_window=t_win, **common),
        lambda: ttnn.bringup.sparse_sdpa(q, t_rows, i_csa, K, kv_format=F.ROW_BFP4, kv_window=t_win8, **common),
        lambda: ttnn.bringup.sparse_sdpa(q, t_win, i_win, K, kv_format=F.BF16, **common),
        lambda: ttnn.bringup.sparse_sdpa(q, t_win8, i_win, K, kv_format=F.ROW_BFP8, **common),
    ]
    runs.append(lambda: ttnn.bringup.sparse_sdpa(q, t_winfp8, i_win, K, kv_format=F.FP8_E4M3, **common))
    for run in runs:
        for _ in range(1 + REPS):
            run().deallocate()
    ttnn.synchronize_device(device)
