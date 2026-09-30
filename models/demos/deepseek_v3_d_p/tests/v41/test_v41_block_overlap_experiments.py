# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Block op-overlap experiments E0, E1, E3 (bead 8y7.9.11; design study ``block-overlap-design.md`` §6).

LoudBox 2x4 (SP = 2, TP = 4), traced replay, device-synchronized wall time (the traced-span domain of the G2 profile).
Each case logs one ``OVERLAP {json}`` line; the assertions only guard that the experiment ran and measured what it
names (the numbers are results, not contracts):

* **E0** (sub-device switch cost): the same matmuls on an 11 x 8 grid, as (a) one trace, (b) split into trace segments
  with no manager switch, (c) segments with a load / clear of a 2 / 8-row manager (the Plan B partition) around each
  second matmul, replayed like ``SubDeviceTraceController`` (non-blocking segments, host load / clear between them).
  Cost per switch pair = (c - a) / pairs; per segment boundary = (b - a) / boundaries.
* **E1** (issue order): two 55-core sub-devices (rows 0-4 / 5-9), chains of 3 dependent compute-bound matmuls, issued
  grouped (A1 A2 A3 B1 B2 B3) vs interleaved (A1 B1 A2 B2 A3 B3), plus one long op then a chain (the TtMoe pattern).
  Concurrent chains take about max(A, B); lock-step issue (hypothesis H-CQ) makes grouped take about A + 2/3 B.
* **E3** (fused wo_b + TP reduce-scatter): per chip [2560, 2048] x [2048, 5120] (bf16 x bfp8, HiFi2) then the TP
  reduce-scatter of dim 3, as today (``ttnn.linear`` + ``V41Collectives.tp_reduce_scatter``, Linear) vs
  ``minimal_matmul_strided_reduce_scatter_async`` (Ring only; run only on a ring TP axis, see the test). Accuracy vs
  the fp32 product of the device inputs, determinism by repeat, time.
* **E3b** (Plan A2 without a fused collective): wq_a and wkv as one matmul and one TP all-reduce plus two slices, vs
  two of each. Bit-equality of the matmul columns, accuracy, time.
"""

import json
import time

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.timing_events import phase
from models.demos.deepseek_v3_d_p.tests.v41.test_v41_trace import MESH
from models.demos.deepseek_v3_d_p.tt.v41.attention import DENSE_COMPUTE_CONFIG
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from tests.ttnn.utils_for_testing import comp_pcc

REPLAYS = 10
ROUNDS = 3


def _log(**rec):
    logger.info(f"OVERLAP {json.dumps(rec)}")


def _tensor(mesh_device, shape, dtype=ttnn.bfloat16, scale=1.0, mapper=None):
    return ttnn.from_torch(
        (torch.randn(shape) * scale).to(torch.bfloat16),
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _rows(x0, y0, x1, y1):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})


class _Segmented:
    """Capture of a step list into trace segments split at ``("load", id)`` / ``("clear",)`` / ``("split",)``
    markers; replay as ``SubDeviceTraceController.replay`` does (segments non-blocking, host load / clear between)."""

    def __init__(self, mesh_device, steps):
        self.mesh_device, self.program, self.keep = mesh_device, [], []
        tid = None
        for step in steps:
            if callable(step):
                if tid is None:
                    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
                self.keep.append(step())
                continue
            if tid is not None:
                ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
                self.program.append(("trace", tid))
                tid = None
            if step[0] == "load":
                mesh_device.load_sub_device_manager(step[1])
            elif step[0] == "clear":
                mesh_device.clear_loaded_sub_device_manager()
            if step[0] != "split":
                self.program.append(step)
        if tid is not None:
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            self.program.append(("trace", tid))

    def replay(self):
        for step in self.program:
            if step[0] == "trace":
                ttnn.execute_trace(self.mesh_device, step[1], cq_id=0, blocking=False)
            elif step[0] == "load":
                self.mesh_device.load_sub_device_manager(step[1])
            else:
                self.mesh_device.clear_loaded_sub_device_manager()

    def timed_us(self, replays=REPLAYS):
        """Mean wall per replay of ``replays`` back-to-back replays (one device synchronize), after a warm one."""
        self.replay()
        ttnn.synchronize_device(self.mesh_device)
        t0 = time.perf_counter()
        for _ in range(replays):
            self.replay()
        ttnn.synchronize_device(self.mesh_device)
        return (time.perf_counter() - t0) * 1e6 / replays

    def release(self):
        for step in self.program:
            if step[0] == "trace":
                ttnn.release_trace(self.mesh_device, step[1])
        self.keep.clear()


def _mm(a, b, grid, sd=None):
    kwargs = {} if sd is None else {"sub_device_id": sd}
    return ttnn.matmul(
        a,
        b,
        core_grid=grid,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=DENSE_COMPUTE_CONFIG,
        **kwargs,
    )


# --- E0 ------------------------------------------------------------------------------------------------------------
E0_PAIRS = 8
E0_M = 4096  # rows of each E0 matmul: about 0.4 ms on 88 cores


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_e0_sub_device_switch_cost(mesh_device, device_params):
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    small_rows = 2
    mgr = mesh_device.create_sub_device_manager(
        [ttnn.SubDevice([_rows(0, 0, gx - 1, small_rows - 1)]), ttnn.SubDevice([_rows(0, small_rows, gx - 1, gy - 1)])],
        0,
    )
    big = ttnn.SubDeviceId(1)
    core_grid = ttnn.CoreGrid(x=gx, y=gy - small_rows)
    with phase("weights", test="e0"):
        x = _tensor(mesh_device, (1, 1, E0_M, 2048))
        w = _tensor(mesh_device, (1, 1, 2048, 2048), dtype=ttnn.bfloat8_b, scale=2048**-0.5)
    full = lambda: _mm(x, w, core_grid)
    on_sd = lambda: _mm(x, w, core_grid, big)
    results = {}
    with phase("compute", test="e0"):
        # compile both program variants (full-grid manager, and the split manager)
        full()
        mesh_device.load_sub_device_manager(mgr)
        on_sd()
        mesh_device.clear_loaded_sub_device_manager()
        ttnn.synchronize_device(mesh_device)
        variants = {
            "one_trace": [full] * (2 * E0_PAIRS),
            "split_only": sum([[full, full, ("split",)] for _ in range(E0_PAIRS)], [])[:-1],
            "switch": sum([[full, ("load", mgr), on_sd, ("clear",)] for _ in range(E0_PAIRS)], []),
        }
        segs = {name: _Segmented(mesh_device, steps) for name, steps in variants.items()}
        for _ in range(ROUNDS):  # rounds interleave the variants, so drift hits all of them
            for name, seg in segs.items():
                results[name] = min(results.get(name, float("inf")), seg.timed_us())
        for name, seg in segs.items():
            results[f"{name}_segments"] = sum(s[0] == "trace" for s in seg.program)
            seg.release()
    mesh_device.remove_sub_device_manager(mgr)
    per_mm = results["one_trace"] / (2 * E0_PAIRS)
    switch_pair = (results["switch"] - results["one_trace"]) / E0_PAIRS
    boundary = (results["split_only"] - results["one_trace"]) / (E0_PAIRS - 1)
    # the switch variant has 2 * pairs segments: 2 * pairs - 1 boundaries, each a host load or clear
    per_switch_boundary = (results["switch"] - results["one_trace"]) / (2 * E0_PAIRS - 1)
    _log(
        exp="E0",
        pairs=E0_PAIRS,
        grid=[gx, gy],
        partition=f"rows 0-{small_rows - 1} / {small_rows}-{gy - 1}",
        per_matmul_us=per_mm,
        switch_pair_us=switch_pair,
        segment_boundary_us=boundary,
        per_switch_boundary_us=per_switch_boundary,
        **results,
    )
    assert results["switch_segments"] == 2 * E0_PAIRS


# --- E1 ------------------------------------------------------------------------------------------------------------
E1_M, E1_K = 2560, 3072


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_e1_issue_order(mesh_device, device_params):
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    half = gy // 2
    mgr = mesh_device.create_sub_device_manager(
        [ttnn.SubDevice([_rows(0, 0, gx - 1, half - 1)]), ttnn.SubDevice([_rows(0, half, gx - 1, gy - 1)])], 0
    )
    sd = [ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)]
    core_grid = ttnn.CoreGrid(x=gx, y=half)
    with phase("weights", test="e1"):
        xs = [_tensor(mesh_device, (1, 1, E1_M, E1_K)) for _ in sd]
        x_long = _tensor(mesh_device, (1, 1, 3 * E1_M, E1_K))
        w = _tensor(mesh_device, (1, 1, E1_K, E1_K), dtype=ttnn.bfloat8_b, scale=E1_K**-0.5)

    def chain(side):
        """The 3 steps of chain ``side``; each consumes the previous output (a real dependency)."""
        state = {"t": xs[side]}

        def step():
            state["t"] = _mm(state["t"], w, core_grid, sd[side])
            return state["t"]

        return [step] * 3

    long_op = lambda: _mm(x_long, w, core_grid, sd[0])
    orders = {
        "A_alone": lambda: chain(0),
        "B_alone": lambda: chain(1),
        "grouped": lambda: chain(0) + chain(1),
        "interleaved": lambda: [s for pair in zip(chain(0), chain(1)) for s in pair],
        "long_alone": lambda: [long_op],
        "long_then_chain": lambda: [long_op] + chain(1),
        "chain_then_long": lambda: chain(1) + [long_op],
    }
    results = {}
    with phase("compute", test="e1"):
        # every replay is load, body, clear: the clear waits for both sub-devices, so replays do not overlap each
        # other and each variant carries the same switch cost (E0)
        segs = {}
        for name, make in orders.items():
            mesh_device.load_sub_device_manager(mgr)
            for step in make():  # compile / warm
                step()
            mesh_device.clear_loaded_sub_device_manager()
            ttnn.synchronize_device(mesh_device)
            segs[name] = _Segmented(mesh_device, [("load", mgr)] + make() + [("clear",)])
        segs["switch_only"] = _Segmented(mesh_device, [("load", mgr), ("clear",)])
        for _ in range(ROUNDS):
            for name, seg in segs.items():
                results[name] = min(results.get(name, float("inf")), seg.timed_us())
        for seg in segs.values():
            seg.release()
    mesh_device.remove_sub_device_manager(mgr)
    a, b = results["A_alone"], results["B_alone"]
    _log(
        exp="E1",
        m=E1_M,
        k=E1_K,
        cores_per_sd=gx * half,
        serial_sum_us=a + b,
        concurrent_bound_us=max(a, b),
        lockstep_prediction_us=a + 2 * b / 3,
        **results,
    )
    assert all(v > 0 for v in results.values())


# --- E3 ------------------------------------------------------------------------------------------------------------
E3_M, E3_K, E3_N = 2560, 2048, 5120  # per chip: rows = chunk 5120 / SP, K = O_GROUPS * O_LORA_RANK / TP, N = hidden
E3_ITERS = 8
# (grid x, grid y, M block, K block, N block, subblock h, subblock w, chunk width in mm blocks), in tiles
E3_CONFIGS = [
    (10, 8, 5, 8, 8, 1, 8, 1),
    (10, 8, 10, 4, 8, 2, 4, 1),
    (10, 8, 5, 8, 16, 1, 8, 1),
    (10, 8, 5, 8, 8, 1, 8, 2),
    (8, 8, 5, 8, 10, 1, 5, 1),
]


def _timed_trace_us(mesh_device, fn, iters=E3_ITERS, label=None):
    """Min over replays of the per-call wall; under the device profiler, ``label`` signposts the replays (the ops
    CSV then attributes kernel durations per case) and the profiler buffer is read after them."""
    fn()
    ttnn.synchronize_device(mesh_device)
    if label:
        signpost(label)
    keep = []
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(iters):
        keep.append(fn())
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
    best = float("inf")
    for _ in range(REPLAYS):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)
        best = min(best, (time.perf_counter() - t0) * 1e6 / iters)
    ttnn.release_trace(mesh_device, tid)
    if label:
        ttnn.ReadDeviceProfiler(mesh_device)
    return best


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_e3_wo_b_fused_reduce_scatter(mesh_device, device_params):
    assert tuple(mesh_device.shape) == (2, 4)
    tp = mesh_device.shape[TP_AXIS]
    coll = V41Collectives(mesh_device)
    shard = lambda dim: ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(None, dim))
    torch.manual_seed(0)
    with phase("weights", test="e3"):
        a = _tensor(mesh_device, (1, 1, E3_M, E3_K * tp), mapper=shard(3))
        w = _tensor(
            mesh_device, (1, 1, E3_K * tp, E3_N), dtype=ttnn.bfloat8_b, scale=(E3_K * tp) ** -0.5, mapper=shard(2)
        )
    with phase("oracle", test="e3"):
        dev = lambda t: [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]
        a_c, w_c = dev(a)[:tp], dev(w)[:tp]  # mesh row 0; row 1 holds the same shards
        golden = sum(x @ y for x, y in zip(a_c, w_c))[0, 0]
        slice_n = E3_N // tp
        want = [golden[:, c * slice_n : (c + 1) * slice_n] for c in range(tp)]

    def check(name, out):
        got = dev(out)
        assert len(got) == 2 * tp and list(got[0].shape) == [1, 1, E3_M, slice_n], (name, got[0].shape)
        pccs = [comp_pcc(want[i % tp], got[i][0, 0], 0.0)[1] for i in range(len(got))]
        return min(pccs), got

    results = {}
    with phase("compute", test="e3"):
        base = lambda: coll.tp_reduce_scatter(ttnn.linear(a, w, compute_kernel_config=DENSE_COMPUTE_CONFIG))
        mm_only = lambda: ttnn.linear(a, w, compute_kernel_config=DENSE_COMPUTE_CONFIG)
        pcc, ref = check("baseline", base())
        repeat = torch.equal(torch.stack([g for g in dev(base())]), torch.stack(ref))
        results["baseline"] = dict(us=_timed_trace_us(mesh_device, base), pcc=pcc, deterministic=repeat)
        results["baseline_mm"] = dict(us=_timed_trace_us(mesh_device, mm_only))
        rs_in = mm_only()
        results["baseline_rs"] = dict(us=_timed_trace_us(mesh_device, lambda: coll.tp_reduce_scatter(rs_in)))
        _log(exp="E3", case="baseline", **results["baseline"], mm_us=results["baseline_mm"]["us"])

        # The fused op validates Ring only. On the LoudBox (FABRIC_2D, TP axis not wrapped) its first call hung the
        # device (dispatch timeout, 2026-09-30, bead 8y7.9.11), so it runs only where the TP axis is a ring.
        configs = E3_CONFIGS if coll.tp_topology == ttnn.Topology.Ring else []
        if not configs:
            _log(exp="E3", case="fused", skipped=f"TP topology {coll.tp_topology} is not Ring")
        for gx, gy, mb, kb, nb, sh, sw, cw in configs:
            name = f"fused_g{gx}x{gy}_m{mb}k{kb}n{nb}_s{sh}x{sw}_cw{cw}"
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=mb,
                K_block_size=kb,
                N_block_size=nb,
                subblock_h=sh,
                subblock_w=sw,
                compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
            )

            def fused(cfg=cfg, gy=gy, cw=cw):
                _, rs = ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                    a,
                    w,
                    3,
                    coll.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=TP_AXIS),
                    ttnn.CoreCoord(0, gy),
                    compute_kernel_config=DENSE_COMPUTE_CONFIG,
                    num_links=coll.num_links,
                    memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                    rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=ttnn.Topology.Ring,
                    cluster_axis=TP_AXIS,
                    config=cfg,
                    barrier_semaphore=coll.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=TP_AXIS),
                    chunk_width_in_mm_blocks=cw,
                )
                return rs

            try:
                pcc, first = check(name, fused())
            except Exception as err:  # config rejected by the op's validation: record and go on
                results[name] = dict(error=str(err).splitlines()[0][:300])
                _log(exp="E3", case=name, **results[name])
                continue
            runs = [torch.stack(dev(fused())) for _ in range(3)]
            deterministic = all(torch.equal(r, torch.stack(first)) for r in runs)
            vs_base = max((g - r).abs().max().item() for g, r in zip(first, ref))
            results[name] = dict(
                us=_timed_trace_us(mesh_device, fused), pcc=pcc, deterministic=deterministic, max_abs_vs_base=vs_base
            )
            _log(exp="E3", case=name, **results[name])
    _log(exp="E3", case="summary", results=results)
    assert results["baseline"]["deterministic"]


# --- E3b: Plan A2 without a fused collective (merged wq_a + wkv, one TP all-reduce) --------------------------------
A2_M, A2_K, A2_NQ, A2_NKV = 2560, 1280, 1280, 512  # per chip: rows, hidden / TP, q_lora_rank, head_dim


def _merged_grid_sweep(mesh_device, x, w, block, reference):
    """Time the merged [M, K] x [K, NQ + NKV] matmul on narrower 2D-multicast grids (same K block, so the same
    per-tile accumulation order); returns {grid: us} and whether each is bit-equal to the default-grid result."""
    mt, nt = A2_M // 32, w.shape[-1] // 32
    out = {}
    for gx, gy in ((7, 10), (8, 10), (11, 8), (7, 8)):
        pm, pn = -(-mt // gy), -(-nt // gx)
        sub_w = max(v for v in range(1, 9) if pn % v == 0)
        sub_h = max(h for h in range(1, 9) if pm % h == 0 and h * sub_w <= 8)
        cfg = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(gx, gy),
            in0_block_w=block,
            out_subblock_h=sub_h,
            out_subblock_w=sub_w,
            per_core_M=pm,
            per_core_N=pn,
            transpose_mcast=False,
            fuse_batch=False,
            fused_activation=None,
        )
        run = lambda cfg=cfg: ttnn.linear(x, w, program_config=cfg, compute_kernel_config=DENSE_COMPUTE_CONFIG)
        got = [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(run())]
        out[f"merged_mm_{gx}x{gy}_us"] = _timed_trace_us(mesh_device, run, label=f"merged_mm_{gx}x{gy}")
        out[f"merged_mm_{gx}x{gy}_bit_equal"] = all(torch.equal(a, b) for a, b in zip(got, reference))
    return out


@pytest.mark.timeout(900)
@pytest.mark.parametrize("mesh_device, device_params", MESH, indirect=True)
def test_v41_e3b_merged_q_kv_projection(mesh_device, device_params):
    """wq_a and wkv share their input and are both row-parallel: one [K, NQ + NKV] matmul and one all-reduce, then
    two slices, vs today's two matmuls and two all-reduces (``tt/v41/attention.py`` forward)."""
    from models.demos.deepseek_v3_d_p.tt.v41.attention import DENSE_IN0_BLOCK_W, dense_program_config

    tp = mesh_device.shape[TP_AXIS]
    coll = V41Collectives(mesh_device)
    shard = lambda dim: ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(None, dim))
    torch.manual_seed(0)
    with phase("weights", test="e3b"):
        x = _tensor(mesh_device, (1, 1, A2_M, A2_K * tp), mapper=shard(3))
        wq_t = torch.randn(1, 1, A2_K * tp, A2_NQ) * (A2_K * tp) ** -0.5
        wkv_t = torch.randn(1, 1, A2_K * tp, A2_NKV) * (A2_K * tp) ** -0.5
        mk = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=shard(2),
        )
        wq, wkv, wqkv = mk(wq_t), mk(wkv_t), mk(torch.cat([wq_t, wkv_t], dim=-1))
    assert DENSE_IN0_BLOCK_W["wq_a"] == DENSE_IN0_BLOCK_W["wkv"]
    block = DENSE_IN0_BLOCK_W["wq_a"]
    dense = lambda w: ttnn.linear(
        x,
        w,
        program_config=dense_program_config(A2_M, A2_K, w.shape[-1], block),
        compute_kernel_config=DENSE_COMPUTE_CONFIG,
    )
    separate = lambda: (coll.tp_all_reduce(dense(wq)), coll.tp_all_reduce(dense(wkv)))

    def merged():
        both = coll.tp_all_reduce(dense(wqkv))
        return (
            ttnn.slice(both, [0, 0, 0, 0], [1, 1, A2_M, A2_NQ]),
            ttnn.slice(both, [0, 0, 0, A2_NQ], [1, 1, A2_M, A2_NQ + A2_NKV]),
        )

    dev = lambda t: [ttnn.to_torch(d).float() for d in ttnn.get_device_tensors(t)]
    with phase("oracle", test="e3b"):
        xs = dev(x)[:tp]
        wq_d, wkv_d = dev(wq)[:tp], dev(wkv)[:tp]
        want = [sum(a @ b for a, b in zip(xs, ws)) for ws in (wq_d, wkv_d)]
    with phase("compute", test="e3b"):
        sep, mer = [[dev(t) for t in f()] for f in (separate, merged)]
        pcc = lambda outs: min(comp_pcc(want[i], o, 0.0)[1] for i in range(2) for o in outs[i])
        bit_equal = [all(torch.equal(a, b) for a, b in zip(sep[i], mer[i])) for i in range(2)]
        max_abs = [max((a - b).abs().max().item() for a, b in zip(sep[i], mer[i])) for i in range(2)]
        mm_equal = [
            all(torch.equal(a, b[..., sl]) for a, b in zip(dev(dense(w)), dev(dense(wqkv))))
            for w, sl in ((wq, slice(0, A2_NQ)), (wkv, slice(A2_NQ, None)))
        ]
        rec = dict(
            separate_us=_timed_trace_us(mesh_device, separate, label="separate"),
            merged_us=_timed_trace_us(mesh_device, merged, label="merged"),
            separate_mm_us=_timed_trace_us(mesh_device, lambda: (dense(wq), dense(wkv)), label="separate_mm"),
            merged_mm_us=_timed_trace_us(mesh_device, lambda: dense(wqkv), label="merged_mm"),
            **_merged_grid_sweep(mesh_device, x, wqkv, block, dev(dense(wqkv))),
            pcc_separate=pcc(sep),
            pcc_merged=pcc(mer),
            matmul_bit_equal_q_kv=mm_equal,
            all_reduce_bit_equal_q_kv=bit_equal,
            max_abs_q_kv=max_abs,
        )
    _log(exp="E3b", **rec)
    assert rec["pcc_merged"] > 0.999
