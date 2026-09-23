# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""CCL benchmark, modeled on nccl-tests.

    tech_reports/CCLs/run_bench.sh

Gated on TTNN_RUN_CCL_BANDWIDTH_BENCHMARK=1, so ordinary collection skips it.

Every machine setting is a constant below, each overridable by an environment
variable so one checkout can drive several configurations without editing this
file. Only the architecture is detected, and only to pick a line rate. A ring
cell whose devices have no wraparound link is skipped rather than measured as a
line on a ring fabric.

nccl-tests sweeps bytes (-b 1K -e 16G -f 2) and derives the element count from
the dtype. This does the same, then rounds each target to a whole number of
tiles, so achieved sizes shift slightly between dtypes.
"""

import json
import os
from pathlib import Path

import pytest
import torch

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tracy import signpost

# ------------------------------------------------------------------- tables
#
# Machine facts the settings below are derived from. None of them read the
# environment.

TILE = 32
ELEMS_PER_TILE = TILE * TILE
DTYPES = {
    # name: (ttnn dtype, bytes per 32x32 tile, pcc threshold)
    "bfloat16": (ttnn.bfloat16, 2048, 0.999),
    "bfloat8_b": (ttnn.bfloat8_b, 1088, 0.99),
    "float32": (ttnn.float32, 4096, 0.9999),
}

FABRIC_CONFIGS = {
    "ring": ttnn.FabricConfig.FABRIC_1D_RING,
    "line": ttnn.FabricConfig.FABRIC_1D,
}

# Per link per direction, GB/s
LINE_RATE_GBPS = {"wormhole_b0": 12.5, "blackhole": 50.0}

# Largest fabric packet payload each architecture accepts. Over it, the conftest
# skips the whole run.
MAX_PACKET_PAYLOAD = {"wormhole_b0": 7616, "blackhole": 15232}

try:
    ARCH = ttnn.get_arch_name()
except Exception:
    ARCH = None


# ----------------------------------------------------------------- settings
#
# Each setting takes its value from the matching environment variable, or the
# default shown. run_bench.sh sets them; edit the defaults to change what a
# bare pytest invocation does.

BENCHMARK_ENV = "TTNN_RUN_CCL_BANDWIDTH_BENCHMARK"


def _env(name, default, cast=str):
    raw = os.environ.get(name)
    return default if raw is None else cast(raw)


def _shape(raw):
    return tuple(int(x) for x in raw.lower().split("x"))


def _list(raw):
    return raw.split(",")


MESH_SHAPE = _env("CCL_MESH", (1, 8), _shape)  # mesh to open
TOPOLOGIES = _env("CCL_TOPOLOGY", ["line", "ring"], _list)  # line | ring
CLUSTER_AXIS = _env("CCL_AXIS", 1, int)  # mesh axis the collective runs along
SUBMESH_SHAPES = _env(
    "CCL_SUBMESHES", [(1, 2), (1, 4), (1, 8)], lambda r: [_shape(x) for x in r.split(",")]
)  # one run per shape

MEMORIES = _env("CCL_MEMORY", ["dram"], _list)  # dram | l1
DTYPE = _env("CCL_DTYPE", "bfloat16")  # bfloat16 | bfloat8_b | float32
TT_DTYPE, TILE_BYTES, PCC = DTYPES[DTYPE]

# The most whole pages that fit one hardware packet, capped at the four segments
# a scatter write carries.
PACKET_PAYLOAD = _env("CCL_PACKET", min(MAX_PACKET_PAYLOAD.get(ARCH, 4352) // TILE_BYTES, 4) * TILE_BYTES, int)
LINE_RATE = _env("CCL_LINE_RATE", LINE_RATE_GBPS.get(ARCH), float)

OPS = _env("CCL_OPS", ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"], _list)

BYTE_TARGETS = [1 << k for k in range(10, 35)]  # 1 KiB .. 16 GiB
ITERS = _env("CCL_ITERS", 20, int)
ITERS_LARGE = _env("CCL_ITERS_LARGE", 5, int)
LARGE_BYTES = 1 << 30  # fewer iterations above this
CHECK_MAX_BYTES = 1 << 30  # correctness runs below this only

# Generated files all live in the report's data directory, which is gitignored.
CONFIG_LOG = Path(os.environ.get("TT_METAL_HOME", ".")) / "tech_reports" / "CCLs" / "data" / "ccl_bench_configs.jsonl"

pytestmark = pytest.mark.skipif(
    os.environ.get(BENCHMARK_ENV) != "1",
    reason=f"CCL bandwidth benchmark is gated on {BENCHMARK_ENV}=1",
)


def mem_config(memory):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED,
        ttnn.BufferType.L1 if memory == "l1" else ttnn.BufferType.DRAM,
    )


def _router_config(payload):
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = payload
    return cfg


def device_params(topology):
    return {
        "fabric_config": FABRIC_CONFIGS[topology],
        "trace_region_size": 500000,
        "fabric_router_config": _router_config(PACKET_PAYLOAD),
    }


# ------------------------------------------------------------------ geometry


def plan(op, target_bytes, n):
    """Tile geometry for one cell, or None if it cannot be expressed.

    Every op splits something n ways and each piece must be a whole tile, so
    the total array is at least 1024*n elements however it is shaped. Targets
    below that have no tiled shape and are dropped.

    Shape rule: the split dimension is 64 tiles once the tensor is big enough,
    and the other dimension takes the rest. 64 tiles is 2048 elements, wide
    enough that packet fill has stopped depending on it.
    """
    tiles = max(1, round(target_bytes / TILE_BYTES))

    split_tiles = 64 if tiles >= 64 else (tiles // n) * n
    if split_tiles == 0:
        return None
    other_tiles = max(1, round(tiles / split_tiles))

    # all_to_all splits the height, the rest split the last dim
    h_tiles, w_tiles = (split_tiles, other_tiles) if op == "all_to_all" else (other_tiles, split_tiles)
    if (h_tiles if op == "all_to_all" else w_tiles) % n:
        return None

    h, w = h_tiles * TILE, w_tiles * TILE
    count = h * w
    if count < ELEMS_PER_TILE * n:
        return None

    if op == "all_gather":
        dev_shape, dev_in, dev_out = [1, 1, h, w // n], count // n, count
    elif op == "reduce_scatter":
        dev_shape, dev_in, dev_out = [1, 1, h, w], count, count // n
    else:
        dev_shape, dev_in, dev_out = [1, 1, h, w], count, count
    return {
        "dev_shape": dev_shape,
        "count": count,
        "num_pages": h_tiles * w_tiles,
        "bytes": h_tiles * w_tiles * TILE_BYTES,
        "dev_in": dev_in,
        "dev_out": dev_out,
    }


def cells(op, n, max_bytes=None):
    out = []
    for target in BYTE_TARGETS:
        if max_bytes and target > max_bytes:
            continue
        if plan(op, target, n):
            out.append(target)
    return out


def _cases(max_bytes=None):
    return [
        pytest.param(
            device_params(topology),
            topology,
            memory,
            op,
            shape,
            target,
            id=f"{op}-{topology}-{memory}-n{shape[CLUSTER_AXIS]}-{target}B",
        )
        for topology in TOPOLOGIES
        for memory in MEMORIES
        for op in OPS
        for shape in SUBMESH_SHAPES
        # A ring can only close across the whole axis.
        if topology == "line" or shape[CLUSTER_AXIS] == MESH_SHAPE[CLUSTER_AXIS]
        for target in cells(op, shape[CLUSTER_AXIS], max_bytes)
    ]


CASE_ARGS = "device_params, topology, memory, op, submesh_shape, target_bytes"
PERF_CASES = _cases()
CHECK_CASES = _cases(CHECK_MAX_BYTES)[::7]  # a sample across the grid, not every cell


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="session", autouse=True)
def banner():
    print(
        "\n"
        "================= CCL benchmark =================\n"
        f"  mesh          {MESH_SHAPE}\n"
        f"  topology      {', '.join(TOPOLOGIES)}\n"
        f"  cluster axis  {CLUSTER_AXIS}\n"
        f"  submeshes     {'  '.join(str(s) for s in SUBMESH_SHAPES)}\n"
        f"  memory        {', '.join(m.upper() for m in MEMORIES)} interleaved\n"
        f"  dtype         {DTYPE}  ({TILE_BYTES} B pages)\n"
        f"  packet        {PACKET_PAYLOAD} B\n"
        f"  arch          {ARCH}  (line rate {LINE_RATE} GB/s per link per direction)\n"
        f"  iters         {ITERS}, {ITERS_LARGE} above {LARGE_BYTES} B\n"
        f"  targets       {BYTE_TARGETS[0]} .. {BYTE_TARGETS[-1]} B, x2\n"
        "=================================================\n"
    )


def get_submesh(mesh_device, shape):
    """Carve the requested shape out of the full mesh.

    Opening a small mesh directly fails fabric bring-up on its boundary links,
    so the parent is always opened whole. The conftest closes submeshes with it.
    """
    shape = tuple(shape)
    if shape == tuple(mesh_device.shape):
        return mesh_device
    return mesh_device.create_submesh(ttnn.MeshShape(*shape), ttnn.MeshCoordinate(0, 0))


def run_op(op, tt_in, mem):
    if op == "all_gather":
        return ttnn.all_gather(tt_in, dim=3, memory_config=mem, cluster_axis=CLUSTER_AXIS)
    if op == "all_reduce":
        return ttnn.all_reduce(tt_in, memory_config=mem, cluster_axis=CLUSTER_AXIS)
    if op == "reduce_scatter":
        return ttnn.reduce_scatter(tt_in, dim=3, memory_config=mem, cluster_axis=CLUSTER_AXIS)
    return ttnn.experimental.all_to_all_async_generic(
        tt_in, in_dim=3, out_dim=2, memory_config=mem, cluster_axis=CLUSTER_AXIS
    )


# ---------------------------------------------------------------- perf test


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(CASE_ARGS, PERF_CASES, indirect=["device_params"])
@pytest.mark.timeout(3600)
def test_perf(mesh_device, device_params, topology, memory, op, submesh_shape, target_bytes):
    submesh = get_submesh(mesh_device, submesh_shape)
    n = submesh.shape[CLUSTER_AXIS]
    mem = mem_config(memory)
    p = plan(op, target_bytes, n)
    iters = ITERS_LARGE if p["bytes"] > LARGE_BYTES else ITERS

    out = tt_in = trace_id = None
    capturing = False
    try:
        # Uninitialised on-device allocation. from_torch builds the tensor on the
        # host first, which dominates the run at multi-GB sizes. Inside the try
        # because an L1 run is expected to outgrow the buffer and must skip.
        tt_in = ttnn.allocate_tensor_on_device(ttnn.Shape(p["dev_shape"]), TT_DTYPE, ttnn.TILE_LAYOUT, submesh, mem)
        # The same demotion the CCL ops apply, so this is the topology they run.
        resolved = ttnn.get_usable_topology(tt_in, cluster_axis=CLUSTER_AXIS).name
        if topology == "ring" and resolved != "Ring":
            pytest.skip(f"no wraparound link along axis {CLUSTER_AXIS} at n={n}")
        out = run_op(op, tt_in, mem)
        ttnn.synchronize_device(submesh)
        # Free it before capturing, or the traced run allocates a second output
        # and peak memory doubles.
        out.deallocate()
        out = None

        trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
        capturing = True
        out = run_op(op, tt_in, mem)
        ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
        capturing = False
        ttnn.synchronize_device(submesh)

        signpost("start")
        for _ in range(iters):
            ttnn.execute_trace(submesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(submesh)
        signpost("stop")

        CONFIG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_LOG.open("a") as f:
            f.write(
                json.dumps(
                    {
                        "op": op,
                        "n": n,
                        "topology": topology,
                        "resolved": resolved,
                        "fabric": FABRIC_CONFIGS[topology].name,
                        "mesh": list(MESH_SHAPE),
                        "submesh": list(submesh.shape),
                        "cluster_axis": CLUSTER_AXIS,
                        "memory": memory,
                        "dtype": DTYPE,
                        "arch": ARCH,
                        "line_rate_gbps": LINE_RATE,
                        "packet": PACKET_PAYLOAD,
                        "target_bytes": target_bytes,
                        "bytes": p["bytes"],
                        "count": p["count"],
                        "num_pages": p["num_pages"],
                        "page_size": TILE_BYTES,
                        "shape": p["dev_shape"],
                        "iters": iters,
                    }
                )
                + "\n"
            )
    except Exception as e:
        pytest.skip(f"{op} n={n} {target_bytes}B: {str(e)[:200]}")
    finally:
        if capturing:
            # A queue left in capture mode stops the device closing, which then
            # fails every later test in the process.
            try:
                ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
            except Exception:
                print("end_trace_capture failed; device may be unusable")
        if trace_id is not None:
            ttnn.release_trace(submesh, trace_id)
        for t in (out, tt_in):
            if t is not None:
                t.deallocate()


# ------------------------------------------------------------- correctness


@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize(CASE_ARGS, CHECK_CASES, indirect=["device_params"])
@pytest.mark.timeout(1800)
def test_correctness(mesh_device, device_params, topology, memory, op, submesh_shape, target_bytes):
    """Not timed. Uses real data, so it takes the slow host allocation path."""
    submesh = get_submesh(mesh_device, submesh_shape)
    n = submesh.shape[CLUSTER_AXIS]
    mem = mem_config(memory)
    p = plan(op, target_bytes, n)
    torch.manual_seed(0)
    h, w = p["dev_shape"][2], p["dev_shape"][3]

    if op in ("all_gather", "all_to_all"):
        golden = torch.rand(1, 1, h, w * n).bfloat16()
        tt_in = ttnn.from_torch(
            golden,
            dtype=TT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=mem,
            mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=3),
        )
        expected = (
            [golden] * n if op == "all_gather" else [golden[:, :, i * h // n : (i + 1) * h // n, :] for i in range(n)]
        )
    else:
        base = torch.rand(1, 1, h, w).bfloat16()
        tt_in = ttnn.from_torch(
            base,
            dtype=TT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        summed = base.float() * n
        expected = (
            [summed] * n if op == "all_reduce" else [summed[:, :, :, i * w // n : (i + 1) * w // n] for i in range(n)]
        )

    out = run_op(op, tt_in, mem)
    ttnn.synchronize_device(submesh)
    for i, dev_out in enumerate(ttnn.get_device_tensors(out)):
        ok, msg = comp_pcc(ttnn.to_torch(dev_out), expected[i], PCC)
        assert ok, f"{op} n={n} device {i}: {msg}"
    out.deallocate()
    tt_in.deallocate()
