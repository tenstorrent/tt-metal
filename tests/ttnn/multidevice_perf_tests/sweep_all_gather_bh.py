# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# all_gather perf sweep on a BH (1,8) box, run under tracy by run_all_gather_sweep_bh.py (see there for the method
# and the analysis). It uses only the public ttnn.all_gather API and the fabric router config, so the same file runs
# on any branch for a comparison. Configured by environment:
#
#   AG_TOPO      ring | line                                   (one per process)
#   AG_SWEEPS    comma list of size, page, holdout             (default: all)
#   AG_CELLS     comma list of fnmatch patterns over cell ids, e.g. "size/*,holdout/l1_*" (default: all)
#   AG_COLUMNS   comma list of algorithm columns: auto, uc_n1, uc_n2, uc_n3, mc (default: auto). A forced column sets
#                TTNN_ALL_GATHER_FORCE for its calls; builds without the override ignore it, so only auto compares
#                across branches.
#   AG_ITERS     traced replays per cell (default: 50 up to 4 MB, where the spread is larger, else 20)
#   AG_MANIFEST  JSON-lines file: one record per cell, in run order
#   AG_CHECK     1 = functional mode: no trace, two calls (the second hits the program cache), bit-exact compare
#
# Per cell: one compile call, a trace capture of one call, AG_ITERS replays each followed by a device sync (isolated
# ops), then ReadDeviceProfiler. A cell whose compile call raises is logged and recorded as status "error".

import fnmatch
import json
import os

import pytest
import torch
import ttnn
from loguru import logger

PAYLOAD = 15168  # BH's tuned fabric payload; 15232 B is the slower 4-slot router tier
FORCE_ENV = "TTNN_ALL_GATHER_FORCE"
COLUMNS = {"auto": None, "uc_n1": "unicast:n=1", "uc_n2": "unicast:n=2", "uc_n3": "unicast:n=3", "mc": "multicast"}

DTYPES = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "fp32": ttnn.float32}
TILE_BYTES = {"bf16": 2048, "bfp8": 1088, "fp32": 4096}
ELEM_BYTES = {"bf16": 2, "fp32": 4}
MiB = 1 << 20


def _cell(sweep, name, layout, dtype, shape, dim, mem_in="dram", mem_out="dram"):
    return dict(
        cell=f"{sweep}/{name}",
        sweep=sweep,
        layout=layout,
        dtype=dtype,
        shape=shape,
        dim=dim,
        mem_in=mem_in,
        mem_out=mem_out,
    )


def _size_cells():
    # bf16 tile [1,1,R,8192] gathered on dim 3; below 1 MB, R = 32 and a narrower row (R < 32 would pad tiles).
    widths = {"64K": (32, 1024), "256K": (32, 4096)}
    rows = {"1M": 64, "2M": 128, "4M": 256, "8M": 512, "26M": 1600, "42M": 2560}
    cells = [_cell("size", k, "tile", "bf16", [1, 1, r, w], 3) for k, (r, w) in widths.items()]
    return cells + [_cell("size", k, "tile", "bf16", [1, 1, r, 8192], 3) for k, r in rows.items()]


def _page_cells():
    # row-major bf16 gathered on dim 0, so every device's block is whole pages; 2 MiB and 25 MiB (26.2 MB) outputs
    cells = []
    for total_name, total in (("2M", 2 * MiB), ("26M", 25 * MiB)):
        for page in (256, 512, 1024, 2048, 4096, 8192, 16384):
            shape = [8, 1, total // (8 * page), page // 2]
            cells.append(_cell("page", f"{total_name}_{page}B", "rm", "bf16", shape, 0))
    return cells


# Disjoint from the bf16 tile cells the choice model is calibrated on.
HOLDOUT = [
    _cell("holdout", "rm_rowseg256_16M", "rm", "bf16", [1, 1, 8192, 1024], 3),  # 256 B per device per row
    _cell("holdout", "rm_rowseg4K_16M", "rm", "bf16", [1, 1, 512, 16384], 3),
    _cell("holdout", "fp32_8M", "tile", "fp32", [1, 1, 256, 8192], 3),
    _cell("holdout", "bfp8_9M", "tile", "bfp8", [1, 1, 1024, 8192], 3),
    _cell("holdout", "fp32_dim0_16M", "tile", "fp32", [8, 1, 512, 1024], 0),
    _cell("holdout", "bfp8_dim0_16M", "tile", "bfp8", [8, 1, 1920, 1024], 0),
    _cell("holdout", "l1_out_2M", "tile", "bf16", [1, 1, 128, 8192], 3, mem_out="l1"),
    _cell("holdout", "l1_l1_2M", "tile", "bf16", [1, 1, 128, 8192], 3, mem_in="l1", mem_out="l1"),
    _cell("holdout", "nd_dram_8M", "tile", "bf16", [1, 1, 512, 8192], 3, "nd_dram:1x1x64x1024", "nd_dram:1x1x64x1024"),
    _cell("holdout", "hs_l1_in_4M", "tile", "bf16", [1, 1, 2048, 1024], 3, mem_in="hs_l1:32x128x64"),
]

SWEEPS = {"size": _size_cells(), "page": _page_cells(), "holdout": HOLDOUT}


def out_bytes(c):
    elems = 1
    for d in c["shape"]:
        elems *= d
    return elems * TILE_BYTES[c["dtype"]] // 1024 if c["layout"] == "tile" else elems * ELEM_BYTES[c["dtype"]]


def page_bytes(c):
    # the output page; equal to the input page unless a row-major tensor is gathered on its last dim
    return TILE_BYTES[c["dtype"]] if c["layout"] == "tile" else c["shape"][-1] * ELEM_BYTES[c["dtype"]]


def memory_config(tag):
    """dram | l1 | nd_dram:<shard shape> | hs_l1:<shard h>x<shard w>x<cores>"""
    if tag == "dram":
        return ttnn.DRAM_MEMORY_CONFIG
    if tag == "l1":
        return ttnn.L1_MEMORY_CONFIG
    kind, arg = tag.split(":")
    dims = [int(x) for x in arg.split("x")]
    if kind == "nd_dram":
        # A DRAM bank id is the core's x alone, so the grid is one row of 8 banks.
        cores = ttnn.num_cores_to_corerangeset(8, ttnn.CoreCoord(8, 1), row_wise=True)
        spec = ttnn.NdShardSpec(
            ttnn.Shape(dims), cores, ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D
        )
        return ttnn.MemoryConfig(ttnn.BufferType.DRAM, spec)
    if kind == "hs_l1":
        h, w, n = dims
        cores = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(8, 8), row_wise=True)
        shard = ttnn.ShardSpec(cores, (h, w), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard)
    raise ValueError(f"unknown memory tag {tag!r}")


def _env_list(name, default):
    return [x.strip() for x in os.environ.get(name, default).split(",") if x.strip()]


def selected_cells():
    patterns = _env_list("AG_CELLS", "*")
    cells = [c for s in _env_list("AG_SWEEPS", ",".join(SWEEPS)) for c in SWEEPS[s]]
    return [c for c in cells if any(fnmatch.fnmatch(c["cell"], p) for p in patterns)]


def _router_config():
    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = PAYLOAD
    return cfg


_TOPOLOGIES = {
    "ring": ttnn.FabricConfig.FABRIC_1D_RING,
    "line": ttnn.FabricConfig.FABRIC_1D,
}
_TOPO = os.environ.get("AG_TOPO", "line")
assert _TOPO in _TOPOLOGIES, f"AG_TOPO must be one of {list(_TOPOLOGIES)}"


def _set_column(column):
    if COLUMNS[column] is None:
        os.environ.pop(FORCE_ENV, None)
    else:
        os.environ[FORCE_ENV] = COLUMNS[column]


def _tensors(mesh_device, c):
    layout = ttnn.TILE_LAYOUT if c["layout"] == "tile" else ttnn.ROW_MAJOR_LAYOUT
    torch_dtype = torch.float32 if c["dtype"] == "fp32" else torch.bfloat16
    torch.manual_seed(0)
    tt_in = ttnn.from_torch(
        torch.randn(c["shape"], dtype=torch_dtype),
        layout=layout,
        dtype=DTYPES[c["dtype"]],
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=c["dim"]),
    )
    tt_in = ttnn.to_device(tt_in, mesh_device, memory_config=memory_config(c["mem_in"]))
    out = ttnn.from_torch(
        torch.zeros(c["shape"], dtype=torch_dtype),
        layout=layout,
        dtype=DTYPES[c["dtype"]],
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.to_device(out, mesh_device, memory_config=memory_config(c["mem_out"]))
    return tt_in, out


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": _TOPOLOGIES[_TOPO], "trace_region_size": 200000, "fabric_router_config": _router_config()}],
    indirect=True,
    ids=[_TOPO],
)
def test_all_gather_sweep(mesh_device, device_params):
    columns = _env_list("AG_COLUMNS", "auto")
    unknown = set(columns) - set(COLUMNS)
    assert not unknown, f"unknown AG_COLUMNS {sorted(unknown)}; known: {list(COLUMNS)}"
    check_mode = os.environ.get("AG_CHECK") == "1"
    manifest = os.environ.get("AG_MANIFEST")
    num_devices = mesh_device.get_num_devices()
    failed = []
    for c in selected_cells():
        nbytes = out_bytes(c)
        iters = int(os.environ.get("AG_ITERS", 50 if nbytes <= 4 * MiB else 20))
        tt_in, out = _tensors(mesh_device, c)
        expected = None
        if check_mode:
            expected = ttnn.to_torch(tt_in, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=c["dim"]))
        for column in columns:
            _set_column(column)
            label = f"{c['cell']} {_TOPO} {column}"
            rec = dict(c, topology=_TOPO, column=column, bytes=nbytes, page_bytes=page_bytes(c), iters=iters)
            rec.update(devices=num_devices, payload=PAYLOAD, status="ok")
            try:
                ttnn.all_gather(tt_in, dim=c["dim"], output_tensor=out)  # compile (or the first checked call)
                ttnn.synchronize_device(mesh_device)
            except RuntimeError as e:
                logger.error(f"AGPERF error {label}: {e}")
                failed.append(label)
                rec["status"] = "error"
            if rec["status"] == "ok" and check_mode:
                ttnn.all_gather(tt_in, dim=c["dim"], output_tensor=out)  # program-cache hit
                ttnn.synchronize_device(mesh_device)
                for d, dev_out in enumerate(ttnn.get_device_tensors(out)):
                    if not torch.equal(ttnn.to_torch(dev_out), expected):
                        logger.error(f"AGPERF mismatch {label}: device {d}")
                        failed.append(label)
                        rec["status"] = "mismatch"
                        break
            elif rec["status"] == "ok":
                trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                ttnn.all_gather(tt_in, dim=c["dim"], output_tensor=out)
                ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
                ttnn.synchronize_device(mesh_device)
                for _ in range(iters):
                    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh_device)
                ttnn.release_trace(mesh_device, trace_id)
                ttnn.ReadDeviceProfiler(mesh_device)  # per cell, so the profiler buffers never overflow
            if manifest:
                with open(manifest, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            logger.info(f"AGPERF {rec['status']} {label} {nbytes} B iters={iters}")
        ttnn.deallocate(out)
        ttnn.deallocate(tt_in)
    os.environ.pop(FORCE_ENV, None)
    assert not failed, f"{len(failed)} cell(s) failed: {failed}"
