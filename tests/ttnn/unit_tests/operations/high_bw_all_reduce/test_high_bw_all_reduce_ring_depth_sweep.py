# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Why is the 2-link ring the slowest cell? Sweeps the pipeline depths (landing / final / staging
slots per reducer, chunk size) on the None-Ring and on the same 4-chip snake line, at 2 links with
the 14 KiB router. If the ring is credit-round-trip-bound, depth moves it and the snake line less.
Run with `-s` to see the SWEEP lines."""

import statistics
import sys

import pytest
import torch
import ttnn

from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce
from ttnn.operations.high_bw_all_reduce import high_bw_all_reduce_program_descriptor as pd
from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

# The op imports its descriptor lazily from sys.modules; patching a duplicate module is a silent no-op.
assert pd is sys.modules["ttnn.operations.high_bw_all_reduce.high_bw_all_reduce_program_descriptor"]

_ROUTER = ttnn.FabricRouterConfig()
_ROUTER.max_packet_payload_size_bytes = 14 * 1024

# name -> knob overrides (baseline = the shipped knobs)
VARIANTS = {
    "baseline": {},
    "final2": {"FINAL_EFFECTIVE_DEPTH": 2},
    "recv4": {"RECV_EFFECTIVE_DEPTH": 4},
    "staging2": {"STAGING_DEPTH_PER_REDUCER": 2},
    "chunk32k": {"CHUNK_BYTES_TARGET": 32768},
    "chunk32k_deep": {
        "CHUNK_BYTES_TARGET": 32768,
        "FINAL_EFFECTIVE_DEPTH": 3,
        "RECV_EFFECTIVE_DEPTH": 4,
        "STAGING_DEPTH_PER_REDUCER": 2,
    },
    "credit1": {"CREDIT_BATCH_CHUNKS": 1},
    "reducers2": {"REDUCERS_PER_LANE": 2},
}


def _profile(mesh_device, run):
    _, records = profile_realtime_program(mesh_device, run, collect_all=True, record_timeout_seconds=5.0)
    programs = {}
    for record in records:
        if any("high_bw_all_reduce" in s for s in record["kernel_sources"]):
            programs[record["runtime_id"]] = max(programs.get(record["runtime_id"], 0.0), record["duration_ns"])
    assert programs, "realtime profiler returned no high_bw_all_reduce program"
    return sum(programs.values())


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": _ROUTER,
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        }
    ],
    ids=["router_14k"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
@pytest.mark.parametrize(
    "cluster_axis,topology",
    [(None, ttnn.Topology.Ring), (None, ttnn.Topology.Linear), (0, ttnn.Topology.Linear)],
    ids=["none-ring", "none-linear", "axis0-linear"],
)
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_sweep(mesh_device, variant, cluster_axis, topology, monkeypatch):
    for knob, value in VARIANTS[variant].items():
        assert hasattr(pd, knob), knob
        monkeypatch.setattr(pd, knob, value)
    depths = []
    ring_depths = pd._ring_depths
    monkeypatch.setattr(pd, "_ring_depths", lambda b: depths.append(b) or ring_depths(b))

    shape = (1, 1, 4096, 4096)
    torch.manual_seed(0)
    rows, cols = tuple(mesh_device.shape)
    stacked = torch.randn((rows, cols, *shape), dtype=torch.bfloat16)
    dims = (0, 1) if cluster_axis is None else (cluster_axis,)
    expected = stacked.float().sum(dim=dims, keepdim=True).expand_as(stacked)
    glob = torch.cat([torch.cat(list(stacked[r]), dim=1) for r in range(rows)], dim=0)
    t = ttnn.from_torch(
        glob,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    run = lambda: high_bw_all_reduce(t, cluster_axis=cluster_axis, topology=topology, num_links=2)
    out = run()
    ttnn.synchronize_device(mesh_device)
    for idx, dt in enumerate(ttnn.get_device_tensors(out)):
        r, c = divmod(idx, cols)
        assert torch.allclose(ttnn.to_torch(dt).float(), expected[r, c], rtol=0.02, atol=0.1)

    _profile(mesh_device, run)
    median_ns = statistics.median(_profile(mesh_device, run) for _ in range(5))
    size = torch.Size(shape).numel() * 2
    group = rows * cols if cluster_axis is None else rows
    per_dir = size * ((group - 1) / group if topology == ttnn.Topology.Ring else 1.0)
    batch = depths[-1] if depths else None
    print(
        f"SWEEP {cluster_axis}-{topology.name} variant={variant} credit_batch={batch} "
        f"depths(recv,final)={ring_depths(batch) if batch else None} median={median_ns / 1e3:.1f}us "
        f"per_link_dir={per_dir / 2 / median_ns:.2f}GB/s"
    )
