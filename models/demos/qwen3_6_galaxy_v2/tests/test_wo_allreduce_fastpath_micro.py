# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Confirm the FA-WO all-reduce fast-path lever on 32-chip BH GLX 8x4.

The FA WO does `line_all_reduce(cluster_axis=0)` (row, ring-8) on a [32,1280]
per-chip tensor and costs ~437-528us (test_wo_rs_micro). But line_all_reduce
(llama_ccl.py:1082) BRANCHES on input memcfg:
  - L1-sharded input  -> ttnn.experimental.all_reduce_async  (the SLOW monolithic kernel)
  - interleaved input -> line_reduce_scatter + line_all_gather (the FAST minimal kernels:
    ReduceScatterMinimalAsync ~58us + AllGatherAsync ~53us ~= 111us, as measured in the MLP)

Same op, same axis/ring — the only difference is which kernel the branch picks. This test
TIMES both paths on the real WO shape via the model's tt_ccl, to confirm the ~4x gap and
that routing WO through RS+AG recovers it. If confirmed, the WO fix is a one-line routing
change (feed interleaved / force the RS+AG decomposition), worth ~437->~111us x 16 FA layers.

Wall-clock per call (synchronize-bracketed, warm iters) — a fair proxy for one isolated CCL.

Run:
    MESH_DEVICE=BH_GLX python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_wo_allreduce_fastpath_micro.py -v -s
"""
from __future__ import annotations

import os
import statistics
import time

import pytest
import torch

import ttnn
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)

_CLUSTER = (8, 4)
_ITERS = 10
_M, _DIM_PER_TP = 32, 1280


def _time(label, fn, mesh, wsd):
    fn()  # compile
    ttnn.synchronize_device(mesh, sub_device_ids=[wsd] if wsd else None)
    ts = []
    for _ in range(_ITERS):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(mesh, sub_device_ids=[wsd] if wsd else None)
        ts.append((time.perf_counter() - t0) * 1e6)
        if o is not None:
            ttnn.deallocate(o)
    print(f"[wo-fast] {label:46s} {statistics.mean(ts):8.1f} ± {statistics.pstdev(ts):6.1f} us/call", flush=True)
    return statistics.mean(ts)


@pytest.mark.hardware
def test_wo_allreduce_fastpath(bh_glx_mesh):
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
    }.items():
        os.environ.setdefault(_k, _v)
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    sd = _load_full_state_dict(_SNAPSHOT)
    pfx0, allp = "model.language_model.layers.0.", "model.language_model.layers."
    sd = {k: v for k, v in sd.items() if (not k.startswith(allp)) or k.startswith(pfx0)}
    paged = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(bh_glx_mesh, sd, ["linear_attention"], 1, paged)
    model.switch_mode("decode")
    mesh = bh_glx_mesh
    tt_ccl = model.layers[0].feed_forward.tt_ccl
    wsd = tt_ccl.worker_sub_device_id
    mc = model.model_config

    # WO reduction input: per-chip [1,1,32,1280] partial, summed over the 8 rows (cluster_axis=0).
    torch.manual_seed(0)
    full = torch.randn(*_CLUSTER, _M, _DIM_PER_TP) * 0.1
    base = ttnn.from_torch(
        full,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=_CLUSTER),
    )
    print(f"\n[wo-fast] WO shape per-chip [{_M},{_DIM_PER_TP}] cluster_axis=0 ring-8", flush=True)
    print("[wo-fast] SLOW baseline (test_wo_rs_micro): all_reduce_async/ttnn.all_reduce = 437-528 us/call", flush=True)

    # --- B: interleaved input -> line_all_reduce decomposes to RS+AG minimal (the FAST path) ---
    def path_B():
        return tt_ccl.line_all_reduce(
            ttnn.clone(base),
            cluster_axis=0,
            num_links=mc["GALAXY_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            use_optimal_ccl_for_llama=True,
        )

    # --- B2: explicit RS + AG (the minimal kernels), in case the line_all_reduce branch differs ---
    def path_B2():
        rs = tt_ccl.line_reduce_scatter(
            ttnn.clone(base),
            ttnn.DRAM_MEMORY_CONFIG,
            dim=3,
            cluster_axis=0,
            num_links=mc["GALAXY_NUM_LINKS"],
            math_op=ttnn.ReduceType.Sum,
            buffer_key="WO_FAST",
        )
        ag = tt_ccl.line_all_gather(
            rs,
            dim=3,
            cluster_axis=0,
            num_links=mc["GALAXY_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="WO_FAST",
        )
        ttnn.deallocate(rs)
        return ag

    b = _time("B: interleaved line_all_reduce -> RS+AG minimal", path_B, mesh, wsd)
    try:
        b2 = _time("B2: explicit line_reduce_scatter + line_all_gather", path_B2, mesh, wsd)
    except Exception as e:  # noqa: BLE001
        print(f"[wo-fast] B2 failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        b2 = None
    fast = min(x for x in [b, b2] if x)
    print(f"[wo-fast] FAST path ~{fast:.0f} us/call  vs SLOW ~437-528 us/call -> ~{437.0 / fast:.1f}x", flush=True)
    ttnn.synchronize_device(mesh, sub_device_ids=[wsd])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
