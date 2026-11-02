# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of ``custom_ops/mhc_pre_block`` (no device; the audit of 2026-10-22, _agents/mhc_a1_audit):

* the kernel ABI: every ``get_compile_time_arg_val`` / ``get_arg_val`` index in the three kernel sources against
  the vectors ``MhcPreBlock.compile_time_args`` / ``runtime_args_for_core`` pack (NoC tables, accessor offsets);
* the work plan over the grids this op can meet (12x10 with two-column cores, 13x10 with idle cores, ...);
* the CB plan against the kernels' in-flight tile counts and the receive-slot invariants the senders rely on
  (``GATHER1 == gy``, ``GATHER2 == gx``, ``PRE == RS2 == 1``: a sender writes slot ``y`` / ``x`` of the receiver's
  buffer *base*, which is only right while the buffer never wraps past one round);
* a discrete-event model of the gather tree (reader / writer / compute of every core as coroutines, random
  interleavings, semaphore counts, slot writes that land before the receiver reserved): the coordinator must see
  every core's contribution exactly once per round, the pre / rsqrt tiles must reach every core, no slot may be
  overwritten before it was consumed, every semaphore must be back at zero when the launch ends, and a second launch
  on the un-reset state (what a replayed trace is if the dispatcher did not re-initialise) must complete too.

    pytest --noconftest models/demos/glm53_flash/tests/custom_ops/test_mhc_pre_block_host.py -q
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import pytest

ttnn = pytest.importorskip("ttnn")
from models.demos.glm53_flash.custom_ops.mhc_pre_block.op import BF16_TILE, FP32_TILE, KERNEL_DIR, MhcPreBlock

N = 4
CT = 128  # C = 4096
GRIDS = [(12, 10), (13, 10), (8, 8), (14, 10), (10, 10), (11, 6)]


# ---------------------------------------------------------------------------------------------- kernel ABI
def _src(name: str) -> str:
    return (KERNEL_DIR / name).read_text()


def _ct_indices(src: str) -> dict[str, str]:
    """``name -> index expression`` of every ``get_compile_time_arg_val(...)`` declaration."""
    pat = re.compile(r"(?:constexpr|const)\s+(?:uint32_t|bool|auto)\s+(\w+)\s*=\s*get_compile_time_arg_val\(([^)]*)\)")
    return {m.group(1): m.group(2).strip() for m in pat.finditer(src)}


def _rt_indices(src: str) -> dict[str, int]:
    pat = re.compile(r"const\s+(?:uint32_t|bool)\s+(\w+)\s*=\s*get_arg_val<uint32_t>\((\d+)\)")
    return {m.group(1): int(m.group(2)) for m in pat.finditer(src)}


def _fake_tables(gx: int, gy: int):
    return [100 + x for x in range(gx)], [200 + y for y in range(gy)]


def _ct_vectors(gx: int, gy: int, exact: bool, n_reader_acc: int = 5, n_writer_acc: int = 3):
    noc_x, noc_y = _fake_tables(gx, gy)
    reader_acc = [900 + i for i in range(n_reader_acc)]  # one config word per interleaved accessor
    writer_acc = [950 + i for i in range(n_writer_acc)]
    return MhcPreBlock.compile_time_args(
        n=N,
        gx=gx,
        gy=gy,
        Ct=CT,
        noc_x=noc_x,
        noc_y=noc_y,
        frep=MhcPreBlock.FREP if exact else 1,
        iters=20,
        eps=1e-6,
        norm_eps=1e-6,
        C=CT * 32,
        rms_eps=1e-6,
        debug=False,
        exact=exact,
        reader_accessors=reader_acc,
        writer_accessors=writer_acc,
    )


@pytest.mark.parametrize("gx,gy", [(12, 10), (13, 10)])
def test_reader_abi(gx, gy):
    CB = MhcPreBlock.CB
    src = _src("reader_mhc_pre_block.cpp")
    ct = _ct_indices(src)
    reader_ct, _, _ = _ct_vectors(gx, gy, exact=False)
    expect = {
        "cb_x": CB["X"],
        "cb_fn": CB["FN"],
        "cb_zero": CB["ZERO"],
        "cb_gather1": CB["GATHER1"],
        "cb_gather2": CB["GATHER2"],
        "cb_pre": CB["PRE"],
        "cb_rs2": CB["RS2"],
        "cb_w": CB["W"],
        "cb_consts": CB["CONSTS"],
        "n": N,
        "GY": gy,
        "GX": gx,
        "Ct": CT,
        "sem_arrive1": MhcPreBlock.SEM_ARRIVE1,
        "sem_arrive2": MhcPreBlock.SEM_ARRIVE2,
        "sem_pre": MhcPreBlock.SEM_PRE,
        "sem_rs2": MhcPreBlock.SEM_RS2,
    }
    for name, value in expect.items():
        idx = int(ct[name])
        assert reader_ct[idx] == value, (name, idx, reader_ct[idx], value)
    base = MhcPreBlock.READER_NOC_Y_BASE
    assert re.search(r"constexpr uint32_t NOC_Y_BASE = (\d+);", src).group(1) == str(base)
    assert ct["frep"].replace(" ", "") == "NOC_Y_BASE+GY" and ct["cb_mask"].replace(" ", "") == "NOC_Y_BASE+GY+1"
    assert re.search(r"TensorAccessorArgs<NOC_Y_BASE \+ GY \+ 2>", src)
    assert reader_ct[base : base + gy] == _fake_tables(gx, gy)[1]  # the NoC y table
    assert reader_ct[base + gy] == 1 and reader_ct[base + gy + 1] == CB["MASK"]
    assert reader_ct[base + gy + 2 :] == [900 + i for i in range(5)]  # x, fn, consts, w, mask accessors
    # the column fan-out reads the table by runtime index: kernel_compile_time_args[NOC_Y_BASE + y]
    assert "kernel_compile_time_args[NOC_Y_BASE + y]" in src
    rt = _rt_indices(src)
    reader_rt, _, _ = MhcPreBlock.runtime_args_for_core(
        xi=3,
        yi=2,
        count=1,
        c0=7,
        noc_x=_fake_tables(gx, gy)[0],
        noc_y=_fake_tables(gx, gy)[1],
        x_addr=11,
        fn_addr=12,
        consts_addr=13,
        w_addr=14,
        mask_addr=15,
        post_addr=16,
        comb_addr=17,
        hn_addr=18,
    )
    assert reader_rt[rt["x_addr"]] == 11 and reader_rt[rt["fn_addr"]] == 12 and reader_rt[rt["consts_addr"]] == 13
    assert reader_rt[rt["w_addr"]] == 14 and reader_rt[rt["mask_addr"]] == 15
    assert reader_rt[rt["count"]] == 1 and reader_rt[rt["c0"]] == 7 and reader_rt[rt["my_noc_x"]] == 103
    assert re.search(r"leader = get_arg_val<uint32_t>\(6\)", src) and reader_rt[6] == 0
    assert re.search(r"coordinator = get_arg_val<uint32_t>\(7\)", src) and reader_rt[7] == 0


@pytest.mark.parametrize("gx,gy", [(12, 10), (13, 10)])
def test_writer_abi(gx, gy):
    CB = MhcPreBlock.CB
    src = _src("writer_mhc_pre_block.cpp")
    ct = _ct_indices(src)
    _, writer_ct, _ = _ct_vectors(gx, gy, exact=False)
    expect = {
        "cb_part": CB["PART"],
        "cb_zero": CB["ZERO"],
        "cb_coef_out": CB["COEF_OUT"],
        "cb_hsq": CB["HSQ"],
        "cb_rs2out": CB["RS2OUT"],
        "cb_out": CB["OUT"],
        "cb_pre": CB["PRE"],
        "cb_rs2": CB["RS2"],
        "cb_gather1": CB["GATHER1"],
        "cb_gather2": CB["GATHER2"],
        "n": N,
        "GY": gy,
        "GX": gx,
        "Ct": CT,
        "sem_arrive1": MhcPreBlock.SEM_ARRIVE1,
        "sem_arrive2": MhcPreBlock.SEM_ARRIVE2,
        "sem_pre": MhcPreBlock.SEM_PRE,
        "sem_rs2": MhcPreBlock.SEM_RS2,
    }
    for name, value in expect.items():
        idx = int(ct[name])
        assert writer_ct[idx] == value, (name, idx, writer_ct[idx], value)
    base = MhcPreBlock.WRITER_NOC_X_BASE
    assert re.search(r"constexpr uint32_t NOC_X_BASE = (\d+);", src).group(1) == str(base)
    assert re.search(r"constexpr uint32_t NOC_Y0 = NOC_X_BASE \+ GX;", src)
    assert re.search(r"TensorAccessorArgs<NOC_Y0 \+ 1>", src)
    noc_x, noc_y = _fake_tables(gx, gy)
    assert writer_ct[base : base + gx] == noc_x and writer_ct[base + gx] == noc_y[0]
    assert writer_ct[base + gx + 1 :] == [950 + i for i in range(3)]  # post, comb, hn accessors
    assert "kernel_compile_time_args[NOC_X_BASE + x]" in src and "get_compile_time_arg_val(NOC_Y0)" in src
    rt = _rt_indices(src)
    _, writer_rt, _ = MhcPreBlock.runtime_args_for_core(
        xi=3,
        yi=2,
        count=1,
        c0=7,
        noc_x=noc_x,
        noc_y=noc_y,
        x_addr=11,
        fn_addr=12,
        consts_addr=13,
        w_addr=14,
        mask_addr=15,
        post_addr=16,
        comb_addr=17,
        hn_addr=18,
    )
    assert writer_rt[rt["post_addr"]] == 16 and writer_rt[rt["comb_addr"]] == 17 and writer_rt[rt["hn_addr"]] == 18
    assert writer_rt[rt["count"]] == 1 and writer_rt[rt["c0"]] == 7
    # the leader of column 3 is (noc_x[3], noc_y[0]); this core's slot there is its row; the coordinator is
    # (noc_x[0], noc_y[0]) and the leaders' slot there is their column
    assert writer_rt[rt["leader_x"]] == 103 and writer_rt[rt["leader_y"]] == 200 and writer_rt[rt["my_y"]] == 2
    assert writer_rt[rt["coord_x"]] == 100 and writer_rt[rt["coord_y"]] == 200 and writer_rt[rt["my_x"]] == 3


@pytest.mark.parametrize("exact", [False, True])
def test_compute_abi(exact):
    CB = MhcPreBlock.CB
    src = _src("compute_mhc_pre_block.cpp")
    ct = _ct_indices(src)
    _, _, compute_ct = _ct_vectors(12, 10, exact=exact)
    for name in MhcPreBlock.COMPUTE_CB_ORDER:
        idx = int(ct[f"CB_{name}"])
        assert compute_ct[idx] == CB[name], (name, idx)
    base = MhcPreBlock.COMPUTE_PARAM_BASE
    params = {"N": N, "GY": 10, "GX": 12, "ITERS": 20}
    for name, value in params.items():
        assert compute_ct[int(ct[name])] == value, name
    assert int(ct["N"]) == base and int(ct["NORM_EPS_BITS"]) == base + 7
    assert compute_ct[int(ct["CB_PARTS"])] == CB["PARTS"] and compute_ct[int(ct["CB_DBG"])] == CB["DBG"]
    assert compute_ct[int(ct["DEBUG"])] == 0 and compute_ct[int(ct["EXACT_PROJ"])] == int(exact)
    assert compute_ct[int(ct["CB_MASK"])] == CB["MASK"]
    assert len(compute_ct) == base + 13
    # the packer-format cache is per launch (reset at kernel_main entry), never a function-local static
    assert "static uint32_t last" not in src
    assert re.search(r"void kernel_main\(\) \{\s*g_pack_last = 0xFFFFFFFFu;", src)
    # every data-movement kernel ends with the full barrier (late atomics hang the next program's NOC init)
    for k in ("reader_mhc_pre_block.cpp", "writer_mhc_pre_block.cpp"):
        body = _src(k)
        assert body.rstrip().endswith("noc_async_full_barrier();\n}".rstrip()), k


# ---------------------------------------------------------------------------------------------- work plan
@pytest.mark.parametrize("gx,gy", GRIDS)
def test_plan_work(gx, gy):
    ncores = gx * gy
    work = MhcPreBlock.plan_work(ncores, CT, MhcPreBlock.MAX_COLS)
    assert len(work) == ncores
    counts = [c for c, _ in work]
    assert sum(counts) == CT and max(counts) <= MhcPreBlock.MAX_COLS
    starts = [s for _, s in work]
    seen = []
    for cnt, c0 in work:
        seen += list(range(c0, c0 + cnt))
    assert seen == list(range(CT))  # every column tile exactly once, in core order
    idle = [k for k, c in enumerate(counts) if c == 0]
    assert idle == list(range(ncores - len(idle), ncores))  # idle cores only at the end
    two = [k for k, c in enumerate(counts) if c == 2]
    assert two == list(range(max(0, CT - ncores)))  # two-column cores only at the start
    assert starts[0] == 0


def test_plan_work_rejects_small_grid():
    rejected = False
    try:
        MhcPreBlock.plan_work(60, CT, MhcPreBlock.MAX_COLS)  # 60 cores x 2 = 120 < 128
    except AssertionError:
        rejected = True
    assert rejected, "a grid that cannot hold the column tiles must be rejected at construction"


# ---------------------------------------------------------------------------------------------- CB plan
@pytest.mark.parametrize("gx,gy", [(12, 10), (13, 10)])
@pytest.mark.parametrize("exact", [False, True])
def test_cb_plan(gx, gy, exact):
    frep = MhcPreBlock.FREP if exact else 1
    plan = MhcPreBlock.cb_plan(N, gx, gy, frep)
    assert set(plan) == set(MhcPreBlock.CB)
    mc = MhcPreBlock.MAX_COLS
    # the receive buffers: a sender writes slot y (x) of the receiver's buffer BASE and the receiver pushes /
    # pops exactly one round per phase, so the buffers must hold exactly one round (no wrap mid-round)
    assert plan["GATHER1"][0] == gy and plan["GATHER2"][0] == gx
    assert plan["PRE"][0] == 1 and plan["RS2"][0] == 1
    # the inputs of a two-column core; the three exact bf16 parts of one column; the per-column outputs
    assert plan["X"][0] == N * mc and plan["FN"][0] == N * mc * frep and plan["W"][0] == mc
    assert plan["HI"][0] == N and plan["LO"][0] == N and plan["LO2"][0] == N
    assert plan["PARTS"][0] == mc and plan["H"][0] == mc
    # a leader's compute pushes its own partial and then the column sum before the writer must have drained
    assert plan["PART"][0] >= 2
    assert plan["COEF_OUT"][0] == N + 2 and plan["CONSTS"][0] == 8 and plan["ZERO"][0] == 1 and plan["MASK"][0] == 1
    assert plan["HSQ"][0] >= 1 and plan["OUT"][0] >= 1 and plan["RS2OUT"][0] >= 1
    total = sum(t * (FP32_TILE if d == ttnn.float32 else BF16_TILE) for t, d in plan.values())
    assert total <= 512 * 1024, total  # every core allocates the whole plan: keep it well under L1 (1.5 MB)


# ---------------------------------------------------------------------------------------------- gather tree
class _Core:
    """One core's state: semaphores, the receive slots a remote writer lands in, the local CB FIFOs."""

    def __init__(self, x, y, count, gx, gy):
        self.x, self.y, self.count = x, y, count
        self.leader, self.coord = y == 0, (x == 0 and y == 0)
        self.sem = {"A1": 0, "A2": 0, "PRE": 0, "RS2": 0}
        # receive slots: (payload, consumed); a write into a slot whose payload is still unconsumed is a hazard
        self.gather1 = [None] * gy
        self.gather2 = [None] * gx
        self.pre_slot = None
        self.rs2_slot = None
        # local FIFOs (payload lists) and their capacities
        self.fifo = {"PART": [], "HSQ": [], "COEF": [], "RS2OUT": [], "H": [], "OUT": [], "PRE": [], "RS2": []}
        self.cap = {"PART": 2, "HSQ": 2, "COEF": N + 2, "RS2OUT": 1, "H": 2, "OUT": 2, "PRE": 1, "RS2": 1}
        self.gather1_pushed = False  # the reader pushed the round's slots to the compute kernel
        self.gather2_pushed = False
        self.pre_seen = None
        self.rs2_seen = None
        self.hazards: list[str] = []
        self.max_occ = Counter()

    def write_slot(self, slots, idx, payload, what):
        if slots[idx] is not None and not slots[idx][1]:
            self.hazards.append(f"{what} slot {idx} of ({self.x},{self.y}) overwritten before it was consumed")
        slots[idx] = (payload, False)

    def push(self, name, payload):
        assert len(self.fifo[name]) < self.cap[name], (name, self.x, self.y)
        self.fifo[name].append(payload)
        self.max_occ[name] = max(self.max_occ[name], len(self.fifo[name]))

    def pop(self, name):
        return self.fifo[name].pop(0)


def _run_launch(cores, gx, gy, rng, reset_sems=True):
    """One launch of the three kernels on every core as coroutines; returns the set of hazards."""
    grid = {(c.x, c.y): c for c in cores}
    ALL = Counter((c.x, c.y) for c in cores if c.count)  # every worker exactly once
    if reset_sems:
        for c in cores:
            c.sem = dict.fromkeys(c.sem, 0)
    for c in cores:
        c.gather1_pushed = c.gather2_pushed = False
        c.pre_pushed = False
        c.pre_seen = c.rs2_seen = None
    checks: list[str] = []

    def reader(c: _Core):
        if c.leader:
            yield lambda: c.sem["A1"] >= gy
            c.sem["A1"] = 0
            for i, s in enumerate(c.gather1):
                if s is None:
                    checks.append(f"leader ({c.x},{c.y}) round 1 slot {i} empty")
            c.gather1_pushed = True
        if c.coord:
            yield lambda: c.sem["A2"] >= gx
            c.sem["A2"] = 0
            c.gather2_pushed = True
        yield lambda: c.sem["PRE"] >= 1 and len(c.fifo["PRE"]) < c.cap["PRE"]
        c.sem["PRE"] = 0
        payload, _ = c.pre_slot
        c.pre_slot = (payload, True)
        if c.leader:
            for yy in range(1, gy):
                dst = grid[(c.x, yy)]
                if dst.pre_slot is not None and not dst.pre_slot[1]:
                    dst.hazards.append(f"pre slot of ({dst.x},{dst.y}) overwritten before consumption")
                dst.pre_slot = (payload, False)
                yield lambda: True  # the write lands, then the semaphore (barrier in between)
                dst.sem["PRE"] += 1
        c.push("PRE", payload)
        c.pre_pushed = True
        if c.leader:
            yield lambda: c.sem["A1"] >= gy and not c.gather1_pushed
            c.sem["A1"] = 0
            c.gather1_pushed = True
        if c.coord:
            yield lambda: c.sem["A2"] >= gx and not c.gather2_pushed
            c.sem["A2"] = 0
            c.gather2_pushed = True
        yield lambda: c.sem["RS2"] >= 1 and len(c.fifo["RS2"]) < c.cap["RS2"]
        c.sem["RS2"] = 0
        payload, _ = c.rs2_slot
        c.rs2_slot = (payload, True)
        if c.leader:
            for yy in range(1, gy):
                dst = grid[(c.x, yy)]
                if dst.rs2_slot is not None and not dst.rs2_slot[1]:
                    dst.hazards.append(f"rs2 slot of ({dst.x},{dst.y}) overwritten before consumption")
                dst.rs2_slot = (payload, False)
                yield lambda: True
                dst.sem["RS2"] += 1
        c.push("RS2", payload)

    def writer(c: _Core):
        leader = grid[(c.x, 0)]
        coord = grid[(0, 0)]

        def send_to_leader(name):
            if c.count:
                yield lambda: len(c.fifo[name]) >= 1
                payload = c.fifo[name][0]
            else:
                payload = Counter()  # the zero tile
            leader.write_slot(leader.gather1, c.y, payload, "gather1")
            yield lambda: True
            leader.sem["A1"] += 1
            if c.count:
                c.pop(name)

        def send_to_coord():
            yield lambda: len(c.fifo["PART"]) >= 1
            payload = c.fifo["PART"][0]
            coord.write_slot(coord.gather2, c.x, payload, "gather2")
            yield lambda: True
            coord.sem["A2"] += 1
            c.pop("PART")

        def fan_out(payload, which):
            for xx in range(gx):
                dst = grid[(xx, 0)]
                slot = dst.pre_slot if which == "PRE" else dst.rs2_slot
                if slot is not None and not slot[1]:
                    dst.hazards.append(f"{which} slot of leader ({xx},0) overwritten before consumption")
                if which == "PRE":
                    dst.pre_slot = (payload, False)
                else:
                    dst.rs2_slot = (payload, False)
                yield lambda: True
                dst.sem[which] += 1

        yield from send_to_leader("PART")
        if c.leader:
            yield from send_to_coord()
        if c.coord:
            yield lambda: len(c.fifo["COEF"]) >= N + 2
            yield from fan_out(c.fifo["COEF"][0], "PRE")
            for _ in range(N + 2):
                c.pop("COEF")
        if not c.count:
            # the writer fix of 2026-10-22: an idle core's round-2 zero tile waits for the pre tile (the reader's
            # CB_PRE push), which exists only after its leader finished round 1. Without this gate the model
            # reproduces the hazard on 13x10: the leader of column 12 sums round 1 over stale slots, then deadlocks.
            yield lambda: c.pre_pushed
        yield from send_to_leader("HSQ")
        if c.leader:
            yield from send_to_coord()
        if c.coord:
            yield lambda: len(c.fifo["RS2OUT"]) >= 1
            yield from fan_out(c.fifo["RS2OUT"][0], "RS2")
            c.pop("RS2OUT")
        for _ in range(c.count):
            yield lambda: len(c.fifo["OUT"]) >= 1
            c.pop("OUT")

    def compute(c: _Core):
        if c.count:
            yield lambda: len(c.fifo["PART"]) < c.cap["PART"]
            c.push("PART", Counter({(c.x, c.y): 1}))
        if c.leader:
            yield lambda: c.gather1_pushed and len(c.fifo["PART"]) < c.cap["PART"]
            total = Counter()
            for i in range(gy):
                payload, _ = c.gather1[i]
                total += payload
                c.gather1[i] = (payload, True)
            c.gather1_pushed = False
            c.push("PART", total)
        if c.coord:
            yield lambda: c.gather2_pushed
            total = Counter()
            for i in range(gx):
                payload, _ = c.gather2[i]
                total += payload
                c.gather2[i] = (payload, True)
            c.gather2_pushed = False
            if total != ALL:
                checks.append(f"round-1 sum {dict(total)} != every worker once")
            for _ in range(N + 2):
                yield lambda: len(c.fifo["COEF"]) < c.cap["COEF"]
                c.push("COEF", total)
        if c.count:
            yield lambda: len(c.fifo["PRE"]) >= 1
            pre = c.pop("PRE")
            c.pre_seen = pre
            for _ in range(c.count):
                yield lambda: len(c.fifo["H"]) < c.cap["H"]
                c.push("H", pre)
            yield lambda: len(c.fifo["HSQ"]) < c.cap["HSQ"]
            c.push("HSQ", Counter({(c.x, c.y): 1}))
        if c.leader:
            yield lambda: c.gather1_pushed and len(c.fifo["PART"]) < c.cap["PART"]
            total = Counter()
            for i in range(gy):
                payload, _ = c.gather1[i]
                total += payload
                c.gather1[i] = (payload, True)
            c.gather1_pushed = False
            c.push("PART", total)
        if c.coord:
            yield lambda: c.gather2_pushed
            total = Counter()
            for i in range(gx):
                payload, _ = c.gather2[i]
                total += payload
                c.gather2[i] = (payload, True)
            c.gather2_pushed = False
            if total != ALL:
                checks.append(f"round-2 sum {dict(total)} != every worker once")
            yield lambda: len(c.fifo["RS2OUT"]) < c.cap["RS2OUT"]
            c.push("RS2OUT", total)
        if c.count:
            yield lambda: len(c.fifo["H"]) >= c.count and len(c.fifo["RS2"]) >= 1
            rs2 = c.pop("RS2")
            c.rs2_seen = rs2
            for _ in range(c.count):
                c.pop("H")
                yield lambda: len(c.fifo["OUT"]) < c.cap["OUT"]
                c.push("OUT", rs2)
        else:
            # an idle core still receives the two fan-outs (its reader pushes them; nobody pops): drain them here
            yield lambda: len(c.fifo["PRE"]) >= 1 and len(c.fifo["RS2"]) >= 1
            c.pre_seen = c.pop("PRE")
            c.rs2_seen = c.pop("RS2")

    threads = []
    for c in cores:
        for gen in (reader(c), writer(c), compute(c)):
            threads.append([gen, None, False])  # generator, pending condition, done
    for t in threads:
        t[1] = next(t[0])
    while True:
        live = [t for t in threads if not t[2]]
        if not live:
            break
        runnable = [t for t in live if t[1]()]
        if not runnable:
            stuck = sum(1 for t in live)
            raise AssertionError(f"deadlock: {stuck} coroutines blocked; hazards so far {_hazards(cores)} {checks}")
        t = rng.choice(runnable)
        try:
            t[1] = next(t[0])
        except StopIteration:
            t[2] = True
    for c in cores:
        if c.pre_seen != ALL:
            checks.append(f"({c.x},{c.y}) pre tile {None if c.pre_seen is None else dict(c.pre_seen)} != all")
        if c.rs2_seen != ALL:
            checks.append(f"({c.x},{c.y}) rs2 tile != all")
        for k, v in c.sem.items():
            if v != 0:
                checks.append(f"({c.x},{c.y}) semaphore {k} = {v} at launch end")
        for name, q in c.fifo.items():
            if q:
                checks.append(f"({c.x},{c.y}) CB {name} holds {len(q)} tiles at launch end")
    return _hazards(cores) + checks


def _hazards(cores):
    out = []
    for c in cores:
        out += c.hazards
        c.hazards = []
    return out


@pytest.mark.parametrize("gx,gy", [(12, 10), (13, 10), (3, 2), (1, 1), (2, 3)])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gather_tree_model(gx, gy, seed):
    ncores = gx * gy
    ct = min(CT, ncores * MhcPreBlock.MAX_COLS) if ncores * MhcPreBlock.MAX_COLS < CT else CT
    work = MhcPreBlock.plan_work(ncores, ct, MhcPreBlock.MAX_COLS)
    rng = random.Random(seed)
    cores = [_Core(x, y, work[x * gy + y][0], gx, gy) for x in range(gx) for y in range(gy)]
    problems = _run_launch(cores, gx, gy, rng)
    assert not problems, problems
    # a second launch on the same state without the dispatcher's semaphore re-initialisation (the waiters reset
    # every semaphore themselves) and with the receive slots still holding the consumed round-2 tiles
    problems = _run_launch(cores, gx, gy, rng, reset_sems=False)
    assert not problems, problems
    occ = Counter()
    for c in cores:
        for k, v in c.max_occ.items():
            occ[k] = max(occ[k], v)
    plan = MhcPreBlock.cb_plan(N, gx, gy, 1)
    names = {"PART": "PART", "HSQ": "HSQ", "COEF": "COEF_OUT", "RS2OUT": "RS2OUT", "H": "H", "OUT": "OUT"}
    for k, name in names.items():
        assert occ[k] <= plan[name][0], (k, occ[k], plan[name][0])


def test_kernel_sources_present():
    for k in ("reader_mhc_pre_block.cpp", "writer_mhc_pre_block.cpp", "compute_mhc_pre_block.cpp"):
        assert (Path(KERNEL_DIR) / k).exists()
