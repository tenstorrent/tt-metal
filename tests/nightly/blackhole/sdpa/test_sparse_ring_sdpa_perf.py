# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Op-level timing of the block-sparse ring SDPA (`ttnn.transformer.sparse_ring_sdpa`) on Blackhole.

Each case times one op as a captured trace: a compile call, one capture, then REPS replays, each replay timed with
`ttnn.synchronize_device` (device time plus one synchronize, no per-call host dispatch). Modes:

  dense     the dense `ring_joint_scaled_dot_product_attention` at its measured chunks
  sparse    the sparse op with pooled K / V (every non-kept block one pooled entry)
  kept      the sparse op without pooled K / V (kept blocks only)
  gather    the fused ring K / V all-gather alone (`ring_attention_all_gather_async`): the communication floor

Shapes are the MiniMax-H3 per-device rows at 5 / 10 / 15 s, 14 heads per device, D = 128, on the 4x8 Galaxy
(TP=4 heads x SP=8 ring) or a single ring. Masks are synthetic (`make_mask`) at block densities 0.20 / 0.07 / 0.05,
the measured 5 s densities at tau 1 / 2 / 3. Every case logs one line and appends one JSON record to
`<basetemp>/sparse_ring_sdpa_perf*/results.jsonl` (pass `--basetemp` to choose the directory). Skipped on CI.
"""

import json
import os
import statistics
import time

import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    call_sdpa,
    close_ring_joint_sdpa_runtime,
    open_ring_joint_sdpa_runtime,
)
from tests.nightly.blackhole.sdpa.test_sparse_ring_sdpa import BLOCK, D, Case, Geometry, _density, make_mask, pool

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="Performance test - skip on CI")

REPS = 6
TRACE_REGION_SIZE = 200_000_000
WORKING_POINTS = {4736: "5s", 9184: "10s", 13664: "15s"}
DENSE_CHUNKS = {4736: (320, 384), 9184: (256, 512), 13664: (256, 512)}  # measured (q_chunk, k_chunk)
DENSITIES = [0.20, 0.07, 0.05]  # 5 s block densities at tau 1 / 2 / 3
K_CHUNKS = [256, 384, 512]  # production: 256 at 5 s, 512 at 10 / 15 s


@pytest.fixture(scope="module")
def runtime():
    rt = open_ring_joint_sdpa_runtime(MESH_CONFIG, trace_region_size=TRACE_REGION_SIZE)
    yield rt
    close_ring_joint_sdpa_runtime(rt)


@pytest.fixture(scope="module")
def results_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("sparse_ring_sdpa_perf") / "results.jsonl"
    logger.info(f"sparse ring SDPA perf results: {path}")
    return path


def traced_ms(mesh_device, fn):
    """Compile call, one trace capture, REPS timed replays: the replay times in ms. `fn` returns the tensor it
    allocates, or None when it writes only persistent buffers."""
    _free(fn())
    ttnn.synchronize_device(mesh_device)
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        times.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh_device, trace_id)
    _free(out)
    return times


def _free(t):
    if t is not None:
        ttnn.deallocate(t)


class Bench:
    """Device inputs for one shape (the last one used; the previous is freed) and its dense / gather baselines."""

    def __init__(self, rt):
        self.rt, self.rows, self.case, self.tensors, self.baselines = rt, None, None, (), {}

    def shape(self, rows):
        if rows != self.rows:
            self._free()
            geo = Geometry(rows, rows * MESH_CONFIG.sp_size, K_CHUNKS[0], dense_chunks=DENSE_CHUNKS[rows])
            case = Case(self.rt, geo)
            shape = (1, case.H, case.S * rows, D)
            g = torch.Generator().manual_seed(rows)
            self.qkv = [torch.randn(shape, generator=g, dtype=torch.bfloat16) for _ in range(3)]
            self.tensors = tuple(case._upload(x, case.seq_dims) for x in self.qkv)
            self.pooled = tuple(case._upload(pool(x, geo, case.S), case.dims(None, 1)) for x in self.qkv[1:])
            self.rows, self.case = rows, case
        return self.case

    def _free(self):
        if self.case is not None:
            for t in (*self.tensors, *self.pooled, self.case.buf_k, self.case.buf_v):
                ttnn.deallocate(t)
        self.case, self.tensors, self.pooled, self.qkv = None, (), (), ()

    def dense(self):
        rt, case = self.rt, self.case
        q, k, v = self.tensors
        return lambda: call_sdpa(
            q,
            k,
            v,
            case.geo.logical_n,
            False,
            False,
            case.buf_k,
            case.buf_v,
            case.dense_config,
            rt.compute_kernel_config,
            rt.ccl_semaphore_handles,
            rt.num_links,
            rt.sp_axis,
            rt.mesh_device,
            rt.topology,
            rt.worker_sub_device_id,
            rt.ccl_column,
        )

    def gather(self):
        rt, case = self.rt, self.case
        _, k, v = self.tensors

        def fn():  # writes the persistent buffers only: nothing to free
            ttnn.experimental.ring_attention_all_gather_async(
                [k, v],
                persistent_output_buffer=[case.buf_k, case.buf_v],
                dim=2,
                multi_device_global_semaphore=rt.ccl_semaphore_handles,
                num_links=rt.num_links,
                cluster_axis=rt.sp_axis,
                mesh_device=rt.mesh_device,
                topology=rt.topology,
                subdevice_id=rt.worker_sub_device_id,
            )

        return fn

    def sparse(self, t_mask, k_chunk, pooled):
        rt, case = self.rt, self.case
        q, k, v = self.tensors
        pk, pv = self.pooled if pooled else (None, None)
        config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=rt.sdpa_compute_grid,
            q_chunk_size=BLOCK,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )
        return lambda: ttnn.transformer.sparse_ring_sdpa(
            q,
            k,
            v,
            t_mask,
            pk,
            pv,
            persistent_output_buffer_k=case.buf_k,
            persistent_output_buffer_v=case.buf_v,
            logical_n=case.geo.logical_n,
            program_config=config,
            compute_kernel_config=rt.compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=rt.ccl_semaphore_handles,
            num_links=rt.num_links,
            cluster_axis=rt.sp_axis,
            mesh_device=rt.mesh_device,
            topology=rt.topology,
            subdevice_id=rt.worker_sub_device_id,
            ccl_core_grid_offset=(rt.ccl_column, 0),
            use_column_major_ccl=True,
            block_size=BLOCK,
        )

    def baseline(self, rows):
        """(dense, gather) replay times for `rows`, timed on first use."""
        if rows not in self.baselines:
            self.shape(rows)
            mesh = self.rt.mesh_device
            self.baselines[rows] = (traced_ms(mesh, self.dense()), traced_ms(mesh, self.gather()))
        return self.baselines[rows]


@pytest.fixture(scope="module")
def bench(runtime):
    b = Bench(runtime)
    yield b
    b._free()


def record(results_file, rows, mode, times, *, k_chunk=None, density=None, dense=None, gather=None):
    op_ms = statistics.median(times)
    rec = {
        "shape": WORKING_POINTS[rows],
        "rows": rows,
        "mesh": f"{MESH_CONFIG.tp_size}x{MESH_CONFIG.sp_size}",
        "mode": mode,
        "k_chunk": k_chunk,
        "density": density,
        "op_ms": op_ms,
        "all_ms": times,
        "speedup_vs_dense": None if dense is None else statistics.median(dense) / op_ms,
        "gap_to_gather_ms": None if gather is None else op_ms - statistics.median(gather),
    }
    extra = ""
    if rec["speedup_vs_dense"] is not None:
        extra += f", {rec['speedup_vs_dense']:.2f}x vs dense"
    if rec["gap_to_gather_ms"] is not None:
        extra += f", {rec['gap_to_gather_ms']:+.3f} ms vs gather"
    logger.info(
        f"[{rec['shape']} rows={rows} {rec['mesh']}] {mode:<6} k={k_chunk} d={density}: {op_ms:.3f} ms "
        f"(all {' '.join(f'{t:.3f}' for t in times)}){extra}"
    )
    with open(results_file, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


@pytest.mark.parametrize("rows", list(WORKING_POINTS), ids=list(WORKING_POINTS.values()))
def test_dense_and_gather_perf(bench, results_file, rows):
    """The two baselines: the dense op and the gather it overlaps."""
    dense, gather = bench.baseline(rows)
    record(results_file, rows, "dense", dense, k_chunk=DENSE_CHUNKS[rows][1], gather=gather)
    record(results_file, rows, "gather", gather)


@pytest.mark.parametrize("pooled", [True, False], ids=["sparse", "kept"])
@pytest.mark.parametrize("density", DENSITIES, ids=[f"d{d}" for d in DENSITIES])
@pytest.mark.parametrize("k_chunk", K_CHUNKS, ids=[f"k{k}" for k in K_CHUNKS])
@pytest.mark.parametrize("rows", list(WORKING_POINTS), ids=list(WORKING_POINTS.values()))
def test_sparse_ring_sdpa_perf(bench, results_file, rows, k_chunk, density, pooled):
    """The sparse op on a synthetic mask, against the dense op and the gather floor of the same shape."""
    dense, gather = bench.baseline(rows)
    case = bench.shape(rows)
    mask = make_mask(case.geo, case.S, case.H, density=density, seed=rows)
    block_density = _density(mask, case.geo, case.S)
    t_mask = case._upload(mask, case.dims(0, 1), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    del mask
    try:
        times = traced_ms(case.rt.mesh_device, bench.sparse(t_mask, k_chunk, pooled))
    finally:
        ttnn.deallocate(t_mask)
    record(
        results_file,
        rows,
        "sparse" if pooled else "kept",
        times,
        k_chunk=k_chunk,
        density=round(block_density, 4),
        dense=dense,
        gather=gather,
    )
