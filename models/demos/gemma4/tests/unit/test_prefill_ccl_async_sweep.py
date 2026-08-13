# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Optional async vs sync TP all-reduce isolate at prefill height.

    unset TT_METAL_DEVICE_PROFILER
    HF_MODEL=google/gemma-4-31B-it GEMMA4_CCL_ASYNC_SWEEP=1 pytest \\
        models/demos/gemma4/tests/unit/test_prefill_ccl_async_sweep.py -k 1x8 -sv --timeout=900
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allreduce

from ..test_factory import parametrize_mesh_with_fabric

_HIDDEN = 5376
_REPEATS = 3
_TRACE_REPS = 20

_ARMS = (
    ("sync_prod", {"GEMMA4_CCL_ASYNC": "0"}),
    ("async_w1c1", {"GEMMA4_CCL_ASYNC": "1", "GEMMA4_CCL_NUM_WORKERS": "1", "GEMMA4_CCL_CHUNKS_PER_SYNC": "1"}),
    ("async_w2c2", {"GEMMA4_CCL_ASYNC": "1", "GEMMA4_CCL_NUM_WORKERS": "2", "GEMMA4_CCL_CHUNKS_PER_SYNC": "2"}),
    ("async_w1c2", {"GEMMA4_CCL_ASYNC": "1", "GEMMA4_CCL_NUM_WORKERS": "1", "GEMMA4_CCL_CHUNKS_PER_SYNC": "2"}),
    ("async_w2c1", {"GEMMA4_CCL_ASYNC": "1", "GEMMA4_CCL_NUM_WORKERS": "2", "GEMMA4_CCL_CHUNKS_PER_SYNC": "1"}),
    ("async_w2c10", {"GEMMA4_CCL_ASYNC": "1", "GEMMA4_CCL_NUM_WORKERS": "2", "GEMMA4_CCL_CHUNKS_PER_SYNC": "10"}),
)


def _upload_distinct(mesh_device, shards):
    stacked = torch.cat(shards, dim=1)
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)],
            ttnn.MeshShape(*mesh_device.shape),
        ),
    )
    return ttnn.from_torch(
        stacked,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _all_device_torch(tensor):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(tensor)]


def _time_trace(mesh_device, capture_fn, *, reps=_TRACE_REPS):
    out = capture_fn()
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = capture_fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    out.deallocate(True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    us = (time.perf_counter() - t0) / reps * 1e6
    ttnn.release_trace(mesh_device, tid)
    return us


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 96_000_000, "l1_small_size": 24576})
@pytest.mark.parametrize("seq_len", [2048], ids=lambda n: f"M{n}")
def test_prefill_async_ccl_sweep(seq_len, mesh_device, reset_seeds, monkeypatch):
    if os.environ.get("GEMMA4_CCL_ASYNC_SWEEP", "0").lower() not in ("1", "true", "yes"):
        pytest.skip("set GEMMA4_CCL_ASYNC_SWEEP=1 to run async vs sync timing")
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("needs TP>=2")

    torch.manual_seed(0)
    n_dev = mesh_device.get_num_devices()
    shape = (1, 1, seq_len, _HIDDEN)
    shards = [torch.randn(shape, dtype=torch.bfloat16) * float(i + 1) for i in range(n_dev)]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device)
    inp = _upload_distinct(mesh_device, shards)
    monkeypatch.setenv("GEMMA4_CCL_ASYNC", "0")
    ref = ttnn.all_reduce(
        ttnn.clone(inp),
        cluster_axis=mesh_config.tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ref_host = _all_device_torch(ref)
    ref.deallocate(True)

    rows = []
    for name, env in _ARMS:
        for k in (
            "GEMMA4_CCL_ASYNC",
            "GEMMA4_CCL_NUM_WORKERS",
            "GEMMA4_CCL_CHUNKS_PER_SYNC",
            "GEMMA4_CCL_NUM_BUFFERS",
        ):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        try:
            cand = ccl_allreduce(ttnn.clone(inp), mesh_config, ccl_manager, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            equal = all(torch.equal(a, b) for a, b in zip(ref_host, _all_device_torch(cand)))
            cand.deallocate(True)
        except Exception as e:
            logger.error(f"{name}: equal-check failed: {e}")
            rows.append((name, False, None))
            continue
        times = []
        try:
            for _ in range(_REPEATS):
                times.append(
                    _time_trace(
                        mesh_device,
                        lambda: ccl_allreduce(
                            ttnn.clone(inp), mesh_config, ccl_manager, memory_config=ttnn.DRAM_MEMORY_CONFIG
                        ),
                    )
                )
            best = min(times)
        except Exception as e:
            logger.error(f"{name}: timing failed: {e}")
            rows.append((name, equal, None))
            continue
        logger.info(
            f"async sweep M={seq_len} {name}: equal={equal} min={best:.1f}us times={[round(t,1) for t in times]}"
        )
        assert equal, f"{name} not bit-exact vs fused at M={seq_len}"
        rows.append((name, equal, best))

    timed = [r for r in rows if r[2] is not None]
    assert timed, "no successful arms"
    winner = min(timed, key=lambda r: r[2])
    logger.info(f"async sweep winner M={seq_len}: {winner[0]} {winner[2]:.1f}us")
    inp.deallocate(True)
