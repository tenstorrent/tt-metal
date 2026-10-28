# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Device-profiler bench (run_safe_pytest.sh --profile): the compressed-KV table write and the attention read,
CKV_FORMAT=bf16 vs bfp4_rows, one card. Each op runs WARM + REPS times; the CSV's DEVICE KERNEL DURATION is read per
op signature by summarize_profile.py."""

import importlib.util

import pytest
import torch

import ttnn

sp = importlib.util.spec_from_file_location("ref", "ttnn/ttnn/bringup/ckv_bfp4_ttnn/tests/reference.py")
ref = importlib.util.module_from_spec(sp)
sp.loader.exec_module(ref)

K, WIN, TOPK, H, KC = 512, 128, 640, 64, 128
REPS = 5
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _ckc():
    return ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=False
    )


def _rm(t, dt, device):
    return ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)


@pytest.mark.parametrize("n", [512, 1280])
def test_write(device, n):
    """bf16: gathered tiles -> ROW_MAJOR -> slice_write into the bf16 table. bfp4_rows: ckv_bfp4 into the row
    records (+ the rounded bf16 tiles -> ROW_MAJOR, which the model also makes)."""
    T = 8192
    x = torch.randn(1, 1, n, K).bfloat16()
    g = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
    tab16 = ttnn.zeros(
        [1, 1, T, K], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM
    )
    tab4 = ttnn.zeros([1, 1, T, 320], dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=DRAM)
    for _ in range(1 + REPS):
        rm = ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        ttnn.experimental.slice_write(rm, tab16, [0, 0, 64, 0], [1, 1, 64 + n, K], [1, 1, 1, 1])
        rm.deallocate()
    for _ in range(1 + REPS):
        r = ttnn.bringup.ckv_bfp4(g, tab4, 64, rounded=True, memory_config=DRAM)
        rm = ttnn.to_layout(r, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
        r.deallocate()
        rm.deallocate()
    for _ in range(1 + REPS):
        ttnn.bringup.ckv_bfp4(g, tab4, 64, rounded=False, memory_config=DRAM)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("S", [160, 512])
def test_read(device, S):
    """sparse_sdpa: bf16 table vs ROW_BFP4 records + bf16 window; 128 window + 512 top-k keys per query, 64 heads."""
    gen = torch.Generator().manual_seed(0)
    T, Tw = 65536, 5248
    x = torch.randn(T, K, generator=gen)
    rows = ref.encode_rows(x.bfloat16())
    vals = ref.decode_rows(rows).bfloat16()
    win = torch.randn(Tw, K, generator=gen).bfloat16()
    t_rows = _rm(rows.view(1, 1, T, -1), ttnn.uint8, device)
    t_win = _rm(win.view(1, 1, Tw, K), ttnn.bfloat16, device)
    t_tab = _rm(torch.cat([win, vals]).view(1, 1, -1, K), ttnn.bfloat16, device)
    sink = _rm(torch.zeros(1, 1, 1, H).bfloat16(), ttnn.bfloat16, device)
    q = _rm((torch.randn(1, H, S, K, generator=gen) * K**-0.5).bfloat16(), ttnn.bfloat16, device)
    idx = torch.empty(1, 1, S, TOPK, dtype=torch.int64)
    for s in range(S):
        idx[0, 0, s, :WIN] = torch.randperm(Tw, generator=gen)[:WIN]
        idx[0, 0, s, WIN:] = torch.randperm(T, generator=gen)[: TOPK - WIN]
    i4 = _rm(idx.to(torch.int32), ttnn.uint32, device)
    i16 = idx.clone()
    i16[..., WIN:] += Tw
    i16 = _rm(i16.to(torch.int32), ttnn.uint32, device)
    common = dict(scale=1.0, k_chunk_size=KC, compute_kernel_config=_ckc(), attention_sink=sink)
    for _ in range(1 + REPS):
        ttnn.bringup.sparse_sdpa(q, t_tab, i16, K, kv_format=ttnn.bringup.SparseKVFormat.BF16, **common).deallocate()
    for _ in range(1 + REPS):
        ttnn.bringup.sparse_sdpa(
            q, t_rows, i4, K, kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4, kv_window=t_win, **common
        ).deallocate()
    ttnn.synchronize_device(device)
