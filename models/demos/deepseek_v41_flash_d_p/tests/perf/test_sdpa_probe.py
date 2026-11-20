# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.81 probe (not a gate): what bounds the compressed layers' sparse_sdpa (ROW_BFP4 records + a bf16 window, 64
heads x 160 queries a chip, [window 128 | top-k 512], sink, the model's attention compute config, max_grid 11 x 10).
Device time per call (real-time profiler, every chip) while varying heads, queries and top-k keys.

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_sdpa_probe.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import torch

import ttnn
from models.demos.common.bringup.testing import profiler
from models.demos.common.bringup.testing.harness import mesh_parametrize

_spec = importlib.util.spec_from_file_location(
    "ckv_bfp4_reference",
    Path(ttnn.__file__).parent / "bringup" / "ckv_bfp4_ttnn" / "tests" / "reference.py",
)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

ITERS = int(os.environ.get("P81_ITERS", "20"))
K_DIM, WIN, KC = 512, 128, 128
T = 76800
TW = 288


def _dev(mesh, t, dtype):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _case(mesh, H, S, top, g):
    q = (torch.randn(1, H, S, K_DIM, generator=g) * K_DIM**-0.5).to(torch.bfloat16)
    win = torch.randn(TW, K_DIM, generator=g).to(torch.bfloat16)
    x = torch.randn(T, K_DIM, generator=g)
    rows = ref.encode_rows(x.bfloat16())
    idx = torch.full((1, 1, S, WIN + top), -1, dtype=torch.int64)
    for s in range(S):
        idx[0, 0, s, :WIN] = torch.arange(s + 1, s + 1 + WIN) % TW
        idx[0, 0, s, WIN : WIN + top] = torch.randperm(T, generator=g)[:top]
    sink = (torch.randn(H, generator=g) * 0.5).to(torch.bfloat16)
    return (
        _dev(mesh, q, ttnn.bfloat16),
        _dev(mesh, rows.view(1, 1, T, -1), ttnn.uint8),
        _dev(mesh, idx.to(torch.int32), ttnn.uint32),
        _dev(mesh, win.view(1, 1, TW, K_DIM), ttnn.bfloat16),
        _dev(mesh, sink.view(1, 1, 1, -1), ttnn.bfloat16),
    )


@mesh_parametrize
def test_sdpa_probe(mesh_device):
    mesh = mesh_device
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    grid = ttnn.CoreCoord(11, 10)
    g = torch.Generator().manual_seed(0)
    cases = []
    for H, S, top in (
        (64, 160, 512),
        (32, 160, 512),
        (64, 110, 512),
        (64, 220, 512),
        (64, 160, 128),
        (64, 160, 256),
        (64, 160, 1024),
    ):
        q, kv, idx, win, sink = _case(mesh, H, S, top, g)

        def run(q=q, kv=kv, idx=idx, win=win, sink=sink):
            return ttnn.bringup.sparse_sdpa(
                q,
                kv,
                idx,
                K_DIM,
                kv_format=ttnn.bringup.SparseKVFormat.ROW_BFP4,
                scale=1.0,
                k_chunk_size=KC,
                compute_kernel_config=ckc,
                attention_sink=sink,
                kv_window=win,
                max_grid=grid,
            )

        try:
            ttnn.deallocate(run())
        except Exception as ex:  # a variant the kernel build refuses (the sfpi ICE seen at H 32)
            print(f"[p81] sdpa H{H} S{S} top{top}: FAILED {str(ex).splitlines()[0][:120]}")
            continue
        cases.append((f"H{H} S{S} top{top}", run))
    profiler.enable(mesh, rt=True)
    try:
        for tag, run in cases:
            profiler.signpost(tag)
            for _ in range(ITERS):
                ttnn.deallocate(run())
        profiler.signpost("end")
        profiler.finish_rt()
        prof = profiler.result()
    finally:
        profiler.disable()
    for tag, _ in cases:
        print(f"[p81] sdpa {tag:24s} {prof['kernel_ns'].get(tag, float('nan')) / ITERS / 1e6:.4f} ms")
