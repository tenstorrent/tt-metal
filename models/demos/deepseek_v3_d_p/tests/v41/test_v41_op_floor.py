# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The per-program floor of a traced step on the 8x4 Galaxy: N tiny programs captured in one trace, the replay's
device time divided by N.

No weights. Each case is a kind of program and a count N:

* ``add1``: ``ttnn.add`` of one bf16 ``[1, 1, 32, 32]`` tile in L1 with itself, the output feeding the next call
  (one core; the chain serializes the programs);
* ``addg``: the same on one tile per compute core (the whole grid, one program);
* ``add_alt``: two ``add1`` chains alternating, one in interleaved L1 (core 0,0), one sharded on core (5,5), so
  consecutive programs run on disjoint cores (whether the dispatcher preloads a program while the previous runs);
* ``generic1``: a one-core ``generic_op`` (``compact_leading_rows`` on one uint32 row in DRAM, in place: the
  kernel reads the row and writes nothing);
* ``ag_sp``: ``ttnn.all_gather`` of one bf16 tile over the SP axis (8 chips, 2 links), as the decode routing
  gathers its index tile;
* ``ag_tp``: ``all_gather_async`` of one tile over the TP axis (4 chips), ``V41Collectives.tp_all_gather``;
* ``rs_tp``: ``reduce_scatter_minimal_async`` of ``[1, 1, 32, 128]`` over TP, ``V41Collectives.tp_reduce_scatter``;
* ``mixed``: ``add1`` with an ``ag_sp`` of the chain's tile every fifth program (the decode layer's shape: 9
  collectives in 23 launches);
* ``add1_ag``: one ``ag_sp`` first, then ``add1`` (whether one collective in the trace changes the local programs'
  dispatch after it, as ``mixed`` suggested).

Per case: the N calls run eagerly once (compile, program cache) and once timed, then are captured in one trace and
replayed ``V41_FLOOR_REPLAYS`` times (default 20) after a warm replay, the host's issue time of ``execute_trace``
and the wall to the device sync both recorded; ``us_traced`` = wall / (replays x N) is the per-program cost under
trace (dispatch, launch and the kernel's own time). One ``FLOOR {json}`` line a case; ``V41_FLOOR_OUT`` collects
them in a JSON file. ``V41_FLOOR_KINDS`` and ``V41_FLOOR_NS`` (comma lists) choose the cases. ``V41_FLOOR_PROFILE=1``
under ``tracy`` reads the device profiler after the eager calls and puts one replay between the ``start`` / ``stop``
signposts (the eager calls between ``eager-start`` / ``eager-stop``), so the ops CSV splits each op into its kernel
duration and the gap before it (``OP TO OP LATENCY``); profile one kind at one N per run with ``V41_FLOOR_REPLAYS=1``
(the report's import of 18,000 ops on 32 chips was killed at 450 GB of host memory).

Run: pytest models/demos/deepseek_v3_d_p/tests/v41/test_v41_op_floor.py -k torus-xy-8x4
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tt.v41.attention import compact_leading_rows
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.layout import SP_AXIS

try:
    from tracy import signpost
except ModuleNotFoundError:  # not under the tracy wrapper

    def signpost(_name: str) -> None:
        pass


TRACE_REGION = 512 * 1024 * 1024
KINDS = ("add1", "addg", "add_alt", "generic1", "ag_sp", "ag_tp", "rs_tp", "mixed", "add1_ag")
COLLECTIVES = ("ag_sp", "ag_tp", "rs_tp")
NS = (100, 500, 2000, 5000)
MIXED_PERIOD = 5


def _env_list(name: str, default, cast=str) -> list:
    raw = os.environ.get(name)
    return list(default) if not raw else [cast(v) for v in raw.split(",") if v]


def _replicated(mesh_device, host, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _among(t, tensors) -> bool:
    return any(t is u for u in tensors)


class _Chain:
    """``N`` programs of one kind, each call's output the next call's input where the kind allows it."""

    def __init__(self, mesh_device, kind: str, coll: V41Collectives):
        self.kind, self.coll = kind, coll
        grid = mesh_device.compute_with_storage_grid_size()
        self.cores = grid.x * grid.y
        self.mem = [ttnn.L1_MEMORY_CONFIG]
        if kind in ("add1", "mixed", "ag_sp", "add1_ag"):
            self.x0 = _replicated(
                mesh_device,
                torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
                ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        elif kind == "add_alt":
            core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 5))})
            shard = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(core, [32, 32], ttnn.ShardOrientation.ROW_MAJOR),
            )
            self.mem = [ttnn.L1_MEMORY_CONFIG, shard]
            self.x0 = _replicated(
                mesh_device,
                torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
                ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            self.x1 = _replicated(
                mesh_device, torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16, memory_config=shard
            )
        elif kind == "addg":
            self.x0 = _replicated(
                mesh_device,
                torch.zeros(1, 1, 32, 32 * self.cores, dtype=torch.bfloat16),
                ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        elif kind == "generic1":
            self.x0 = _replicated(
                mesh_device, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        elif kind == "ag_tp":
            self.x0 = _replicated(mesh_device, torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), ttnn.bfloat16)
        elif kind == "rs_tp":
            self.x0 = _replicated(mesh_device, torch.zeros(1, 1, 32, 128, dtype=torch.bfloat16), ttnn.bfloat16)
        else:
            raise ValueError(kind)

    def _ag_sp(self, t):
        # as the decode routing's gather (``tt/v41/routing.py``) less its ``num_links``, deprecated and ignored
        return ttnn.all_gather(t, dim=2, cluster_axis=SP_AXIS, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _collective(self, t):
        if self.kind in ("ag_sp", "mixed", "add1_ag"):
            return self._ag_sp(t)
        if self.kind == "ag_tp":
            return self.coll.tp_all_gather(t, dim=3)
        if self.kind == "rs_tp":
            return self.coll.tp_reduce_scatter(t, dim=3)
        raise ValueError(self.kind)

    def _local(self, t, chain: int = 0):
        if self.kind == "generic1":
            compact_leading_rows(t, 1, 32)  # in place: the same buffer is the next call's input
            return t
        return ttnn.add(t, t, memory_config=self.mem[chain])

    def is_collective(self, i: int) -> bool:
        if self.kind == "add1_ag":
            return i == 0
        return self.kind in COLLECTIVES or (self.kind == "mixed" and i % MIXED_PERIOD == MIXED_PERIOD - 1)

    def run(self, n: int) -> list:
        """``n`` calls; returns the tensors still allocated at the end (freed by the caller). A collective's output
        is not chained (its shape differs); it is freed when the next call is issued."""
        heads = [self.x0] + ([self.x1] if self.kind == "add_alt" else [])
        cur, side = list(heads), None
        for i in range(n):
            if side is not None:
                ttnn.deallocate(side)
                side = None
            c = i % len(cur)
            if self.is_collective(i):
                side = self._collective(cur[c])
            else:
                nxt = self._local(cur[c], c)
                if cur[c] is not nxt and not _among(cur[c], heads):
                    ttnn.deallocate(cur[c])
                cur[c] = nxt
        return [t for t in cur + [side] if t is not None and not _among(t, heads)]

    def free(self):
        ttnn.deallocate(self.x0)
        if self.kind == "add_alt":
            ttnn.deallocate(self.x1)


def _sync(mesh_device):
    ttnn.synchronize_device(mesh_device)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device, device_params", galaxy_meshes(trace_region_size=TRACE_REGION), indirect=True)
def test_v41_op_floor(mesh_device, device_params):
    sp, tp = tuple(mesh_device.shape)
    coll = V41Collectives(mesh_device)
    kinds = _env_list("V41_FLOOR_KINDS", KINDS)
    ns = _env_list("V41_FLOOR_NS", NS, int)
    replays = int(os.environ.get("V41_FLOOR_REPLAYS", 20))
    profile = os.environ.get("V41_FLOOR_PROFILE") == "1"
    out = os.environ.get("V41_FLOOR_OUT")
    records = []
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(
        f"mesh {sp}x{tp}, grid {grid.x}x{grid.y}, fabric {device_params.get('fabric_config')}, links {coll.num_links}"
    )
    for kind in kinds:
        chain = _Chain(mesh_device, kind, coll)
        for t in chain.run(2):  # compile and warm the program cache
            ttnn.deallocate(t)
        _sync(mesh_device)
        for n in ns:
            t0 = time.perf_counter()
            # eager reference
            signpost("eager-start")
            te = time.perf_counter()
            tail = chain.run(n)
            issue_eager = time.perf_counter() - te
            _sync(mesh_device)
            us_eager = (time.perf_counter() - te) * 1e6 / n
            signpost("eager-stop")
            for t in tail:
                ttnn.deallocate(t)
            if profile:
                ttnn.ReadDeviceProfiler(mesh_device)
            # capture
            tc = time.perf_counter()
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            tail = chain.run(n)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            capture_s = time.perf_counter() - tc
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)  # warm replay
            _sync(mesh_device)
            if profile:
                signpost("start")
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                _sync(mesh_device)
                ttnn.ReadDeviceProfiler(mesh_device)
                signpost("stop")
            issue = 0.0
            tr = time.perf_counter()
            for _ in range(replays):
                ti = time.perf_counter()
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                issue += time.perf_counter() - ti
            _sync(mesh_device)
            wall = time.perf_counter() - tr
            # one replay at a time, synchronized, for the per-replay fixed cost (not under the profiler: every op
            # it records costs host memory at the report, 18,000 tiny ops on 32 chips were 450 GB)
            singles = [wall / replays] if profile else []
            for _ in range(0 if profile else 5):
                ts = time.perf_counter()
                ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
                _sync(mesh_device)
                singles.append(time.perf_counter() - ts)
            ttnn.release_trace(mesh_device, tid)
            for t in tail:
                ttnn.deallocate(t)
            rec = dict(
                kind=kind,
                n=n,
                replays=replays,
                us_traced=round(wall * 1e6 / (replays * n), 3),
                us_eager=round(us_eager, 2),
                issue_eager_us=round(issue_eager * 1e6 / n, 2),
                replay_ms=round(wall * 1e3 / replays, 4),
                issue_ms=round(issue * 1e3 / replays, 4),
                single_replay_ms=round(min(singles) * 1e3, 4),
                capture_s=round(capture_s, 3),
                case_s=round(time.perf_counter() - t0, 1),
            )
            records.append(rec)
            logger.info(f"FLOOR {json.dumps(rec)}")
        chain.free()
        _sync(mesh_device)
    if out:
        with open(out, "w") as f:
            json.dump(dict(mesh=[sp, tp], grid=[grid.x, grid.y], links=coll.num_links, cases=records), f, indent=1)
