# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolate prefill-shaped TP all-reduce knobs (split RS+AG vs fused).

Decode already swept ``num_workers_per_link`` / ``chunks_per_sync`` on
``[1,1,32,5376]`` bf16 (344 KB, latency-bound) and shipped ``w=1, c=1``. Prefill
payloads are larger (M=96 ~1 MB, M=2048 ~22 MB). Default test: production
``ccl_allreduce`` is ``torch.equal`` to fused ``ttnn.all_reduce``. Optional
timing matrix (``GEMMA4_CCL_SWEEP=1``):

    unset TT_METAL_DEVICE_PROFILER
    HF_MODEL=google/gemma-4-31B-it GEMMA4_CCL_SWEEP=1 pytest \\
        models/demos/gemma4/tests/unit/test_prefill_ccl_rs_sweep.py -k 1x8 -sv --timeout=600
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager, ccl_allreduce, ccl_sync_rs_buffers

from ..test_factory import parametrize_mesh_with_fabric

_HIDDEN = 5376
_RS_ARMS = (
    (1, 1),
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 2),
    (4, 1),
)
_REPEATS = 3
_TRACE_REPS = 20


def _upload_distinct(mesh_device, shards, *, dtype, layout, memory_config):
    """One [1,1,M,H] activation per device with distinct values (shard cat dim=1)."""
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
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=mapper,
    )


def _all_device_torch(tensor):
    return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(tensor)]


def _split_allreduce(tensor, mesh_config, ccl_manager, *, workers, chunks, buffers):
    scattered = ttnn.reduce_scatter(
        tensor,
        dim=3,
        cluster_axis=mesh_config.tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_workers_per_link=workers,
        chunks_per_sync=chunks,
        num_buffers_per_channel=buffers,
    )
    gathered = ttnn.all_gather(
        scattered,
        dim=3,
        cluster_axis=mesh_config.tp_axis,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scattered.deallocate(True)
    return gathered


def _time_trace(mesh_device, capture_fn, *, reps=_TRACE_REPS):
    """Compile, capture, replay ``reps`` times. Returns us/call (wall / reps)."""
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


@parametrize_mesh_with_fabric()
@pytest.mark.parametrize("seq_len", [96, 2048], ids=lambda n: f"M{n}")
def test_prefill_ccl_allreduce_matches_fused(seq_len, mesh_device, reset_seeds):
    """Production split all-reduce (height-aware knobs) is bit-exact vs fused."""
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("needs TP>=2")

    torch.manual_seed(0)
    n_dev = mesh_device.get_num_devices()
    shape = (1, 1, seq_len, _HIDDEN)
    shards = [torch.randn(shape, dtype=torch.bfloat16) * float(i + 1) for i in range(n_dev)]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device)
    inp = _upload_distinct(
        mesh_device, shards, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    split = ccl_allreduce(ttnn.clone(inp), mesh_config, ccl_manager, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    fused = ttnn.all_reduce(
        ttnn.clone(inp),
        cluster_axis=mesh_config.tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    split_host = _all_device_torch(split)
    fused_host = _all_device_torch(fused)
    split.deallocate(True)
    fused.deallocate(True)
    inp.deallocate(True)
    assert all(
        torch.equal(a, b) for a, b in zip(split_host, fused_host)
    ), f"ccl_allreduce != fused all_reduce at M={seq_len}"


@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 64_000_000})
@pytest.mark.parametrize("seq_len", [96, 2048], ids=lambda n: f"M{n}")
def test_prefill_rs_knob_sweep(seq_len, mesh_device, reset_seeds):
    """Optional metal-trace RS worker/chunk matrix. Opt in with GEMMA4_CCL_SWEEP=1."""
    if os.environ.get("GEMMA4_CCL_SWEEP", "0").lower() not in ("1", "true", "yes"):
        pytest.skip("set GEMMA4_CCL_SWEEP=1 to run the RS knob timing matrix")
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    if tp < 2:
        pytest.skip("needs TP>=2")

    torch.manual_seed(0)
    n_dev = mesh_device.get_num_devices()
    shape = (1, 1, seq_len, _HIDDEN)
    shards = [torch.randn(shape, dtype=torch.bfloat16) * float(i + 1) for i in range(n_dev)]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device)
    buffers = ccl_sync_rs_buffers()
    inp = _upload_distinct(
        mesh_device, shards, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
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
    for workers, chunks in _RS_ARMS:
        cand = _split_allreduce(inp, mesh_config, ccl_manager, workers=workers, chunks=chunks, buffers=buffers)
        equal = all(torch.equal(a, b) for a, b in zip(ref_host, _all_device_torch(cand)))
        cand.deallocate(True)
        times = [
            _time_trace(
                mesh_device,
                lambda w=workers, c=chunks: _split_allreduce(
                    inp, mesh_config, ccl_manager, workers=w, chunks=c, buffers=buffers
                ),
            )
            for _ in range(_REPEATS)
        ]
        best = min(times)
        rows.append((workers, chunks, equal, best))
        logger.info(
            f"prefill RS M={seq_len} w={workers} c={chunks} equal={equal} "
            f"min={best:.1f}us times={[round(t, 1) for t in times]}"
        )
        assert equal, f"split RS w={workers} c={chunks} M={seq_len} != fused all_reduce"

    winner = min(rows, key=lambda r: r[3])
    logger.info(f"prefill RS winner M={seq_len}: w={winner[0]} c={winner[1]} {winner[3]:.1f}us")
    inp.deallocate(True)
