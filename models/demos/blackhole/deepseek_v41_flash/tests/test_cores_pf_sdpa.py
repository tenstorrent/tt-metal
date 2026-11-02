# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-chip sparse_sdpa timing with unsorted vs per-row sorted gather indices (cores-for-prefill study): is the random 512 B row gather DRAM-locality bound?
Synthetic: fp8 pool of R rows, 512 query rows x 32 heads (bf16 RM), idx [1,1,512,640] = a 128-row contiguous window + 512 random pool rows (like the prefill's window + top-k)."""
import os
import time

import pytest
import torch

import ttnn

R = int(os.environ.get("CSD_ROWS", "65536"))
REPS = 10


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("mode", ["random"])
@pytest.mark.parametrize("kc", [128])
def test_cores_pf_sdpa(device, mode, kc):
    dev = device
    S, TOPK, W = int(os.environ.get("CSD_S", "512")), 512, 128
    g = torch.Generator().manual_seed(0)
    pool = ttnn.from_torch(
        torch.randn(1, 1, R, 512, generator=g).float(),
        device=dev,
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    q = ttnn.from_torch(
        torch.randn(1, 32, S, 512, generator=g).to(torch.bfloat16),
        device=dev,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    base = torch.arange(S).view(S, 1) + 8192  # the chunk's own rows at the end of an 8k context
    win = base + torch.arange(-W + 1, 1).view(1, W)
    if mode == "local":  # top-k clustered near the query (best case for DRAM locality)
        top = (base - 200 - torch.randint(0, 1024, (S, TOPK), generator=g)).clamp(min=0)
    else:
        top = torch.randint(0, 8192, (S, TOPK), generator=g)
    idx = torch.cat([win, top], dim=1)
    if mode == "sorted":
        idx = idx.sort(dim=1).values
    tidx = ttnn.from_torch(
        idx.to(torch.int32).reshape(1, 1, S, W + TOPK),
        device=dev,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    def run():
        return ttnn.transformer.sparse_sdpa(
            q,
            pool,
            tidx,
            512,
            kv_format=ttnn.transformer.SparseKVFormat.FP8_E4M3,
            scale=0.044,
            k_chunk_size=kc,
            compute_kernel_config=ckc,
        )

    o = run()
    ttnn.synchronize_device(dev)
    ttnn.deallocate(o)
    t = time.perf_counter()
    for _ in range(REPS):
        o = run()
        ttnn.deallocate(o)
    ttnn.synchronize_device(dev)
    print(
        f"SDPA {mode} kc{kc}: {(time.perf_counter() - t) / REPS * 1e6:.1f} us per call (S={S}, single chip)", flush=True
    )
