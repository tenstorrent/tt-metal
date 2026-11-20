# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The in-program collective (fused/collective/collective.py) on the mesh the model opens.

On the mock Galaxy (HY4_MOCK=1) every case compiles its kernels and checks CB / L1 / kernel sizes; nothing runs.
On the board a case runs three rounds (both parities of the landing, then the first again) on fresh data and
compares every member chip with the host reference bit for bit: gathers, Float32 sums in the device's order, and
BFloat16 sums as the Float32 sum rounded to BFloat16 half away from zero (see compare). Smallest first:

    pytest <this file> -k "test_collective and pair"      (HY4_COLLECTIVE_CASE=<id>: one case)
    pytest <this file> -k "test_collective and (row0 or col0)"
    pytest <this file> -k "test_collective and (axis or both)"
    pytest <this file> -k test_stale_state
    pytest <this file> -k test_latency
    pytest <this file> -k test_timing_first     (alone, before the first test_breakdown)
    pytest <this file> -k test_breakdown
    pytest <this file> -k test_marginal
    pytest <this file> -k test_collective_pair_to_galaxy   (all of it in one mesh open)

Result lines start with HY4_COLLECTIVE; the latency table also goes to $HY4_COLLECTIVE_RESULTS (a JSON file).

HY4_COLLECTIVE_DRY=1 with HY4_MOCK=1 walks the board's code path on the mock (test_collective_pair_to_galaxy
only): nothing executes, so nothing is right, but a mistake in this file's host code shows before a board run.
"""

import json
import os
import re
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.collective import collective as cl
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.tt import prefill_bridge as pb

GALAXY_MESH = (8, 4)
TRACE_REGION_BYTES = 32 << 20
RESULTS = Path(os.environ.get("HY4_COLLECTIVE_RESULTS", "generated/hy4_preview/collective_latency.json"))

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION_BYTES), GALAXY_MESH)],
    indirect=True,
    ids=["8x4-torus"],
)

FP32, BF16 = ttnn.float32, ttnn.bfloat16


DRY = bool(os.environ.get("HY4_COLLECTIVE_DRY")) and cl.is_mock()


def mock():
    """Compile only? True on the mock Galaxy, unless $HY4_COLLECTIVE_DRY asks for the board's code path there: a
    dry run of this file's host code. The mock executes nothing, so no result is right (``expect`` only logs
    then), and it cannot enqueue a program on part of the mesh, so only the 32-chip cases run."""
    return cl.is_mock() and not DRY


def expect(condition, message):
    if DRY:
        logger.info(
            f"HY4_COLLECTIVE dry run: {'ok' if condition else 'not met (expected on the mock)'}: {str(message)[:120]}"
        )
    else:
        assert condition, message


def rings_of(kind, shape):
    """The stages of a case: a list of ring sets, one per site. "axis1x4": the rings along axis 1, four times (one
    site used by four stages of one program)."""
    repeated = re.fullmatch(r"axis([01])x(\d+)", kind)
    if repeated:
        return [cl.axis_rings(shape, int(repeated.group(1)))] * int(repeated.group(2))
    return {
        "pair": [[[(0, 0), (0, 1)]]],
        "pairv": [[[(0, 0), (1, 0)]]],
        "row0": [cl.axis_rings(shape, 1, lines=[0])],
        "col0": [cl.axis_rings(shape, 0, lines=[0])],
        "axis1": [cl.axis_rings(shape, 1)],
        "axis0": [cl.axis_rings(shape, 0)],
        "both": [cl.axis_rings(shape, 1), cl.axis_rings(shape, 0)],
        # one site used three times in one program, then the other axis: stages 0-2 share a site
        "row0x3": [cl.axis_rings(shape, 1, lines=[0])] * 3,
        "bothx2": [cl.axis_rings(shape, 1), cl.axis_rings(shape, 1), cl.axis_rings(shape, 0), cl.axis_rings(shape, 0)],
    }[kind]


# id -> (rings, payload bytes, dtype, routes, mode[, parts: landing cores the payload is striped over])
CASES = {
    "pair-64B-fp32-unicast-reduce": ("pair", 64, FP32, "unicast", "reduce"),
    "pair-64B-fp32-unicast-gather": ("pair", 64, FP32, "unicast", "gather"),
    "pairv-4K-fp32-unicast-reduce": ("pairv", 4096, FP32, "unicast", "reduce"),
    "row0-64B-fp32-unicast-reduce": ("row0", 64, FP32, "unicast", "reduce"),
    "row0-48K-fp32-unicast-reduce": ("row0", 49152, FP32, "unicast", "reduce"),
    "row0-24K-bf16-unicast-reduce": ("row0", 24576, BF16, "unicast", "reduce"),
    "row0-4K-fp32-unicast-gather": ("row0", 4096, FP32, "unicast", "gather"),
    "row0-48K-fp32-unicast-land": ("row0", 49152, FP32, "unicast", "land"),
    "row0-48K-fp32-multicast-reduce": ("row0", 49152, FP32, "multicast", "reduce"),
    "row0x3-4K-fp32-unicast-reduce": ("row0x3", 4096, FP32, "unicast", "reduce"),
    "col0-64B-fp32-unicast-reduce": ("col0", 64, FP32, "unicast", "reduce"),
    "col0-48K-fp32-unicast-reduce": ("col0", 49152, FP32, "unicast", "reduce"),
    "col0-24K-bf16-multicast-reduce": ("col0", 24576, BF16, "multicast", "reduce"),
    "col0-4K-bf16-unicast-gather": ("col0", 4096, BF16, "unicast", "gather"),
    "axis1-64B-fp32-unicast-reduce": ("axis1", 64, FP32, "unicast", "reduce"),
    "axis1-48K-fp32-unicast-reduce": ("axis1", 49152, FP32, "unicast", "reduce"),
    "axis1-48K-fp32-multicast-reduce": ("axis1", 49152, FP32, "multicast", "reduce"),
    "axis0-24K-bf16-unicast-reduce": ("axis0", 24576, BF16, "unicast", "reduce"),
    "axis0-48K-fp32-multicast-reduce": ("axis0", 49152, FP32, "multicast", "reduce"),
    "axis1-4K-fp32-multicast-gather": ("axis1", 4096, FP32, "multicast", "gather"),
    "axis0-64B-fp32-unicast-gather": ("axis0", 64, FP32, "unicast", "gather"),
    "axis0-24K-bf16-multicast-land": ("axis0", 24576, BF16, "multicast", "land"),
    "both-64B-fp32-unicast-reduce": ("both", 64, FP32, "unicast", "reduce"),
    "both-4K-fp32-multicast-reduce": ("both", 4096, FP32, "multicast", "reduce"),
    "both-48K-fp32-multicast-reduce": ("both", 49152, FP32, "multicast", "reduce"),
    "both-24K-bf16-multicast-reduce": ("both", 24576, BF16, "multicast", "reduce"),
    "bothx2-4K-fp32-multicast-reduce": ("bothx2", 4096, FP32, "multicast", "reduce"),
    # striped over several landing cores
    "row0-48K-fp32-unicast-reduce-4cores": ("row0", 49152, FP32, "unicast", "reduce", 4),
    "row0-8K-fp32-unicast-gather-2cores": ("row0", 8192, FP32, "unicast", "gather", 2),
    "axis1-48K-fp32-multicast-reduce-12cores": ("axis1", 49152, FP32, "multicast", "reduce", 12),
    "axis0-24K-bf16-multicast-land-4cores": ("axis0", 24576, BF16, "multicast", "land", 4),
    "both-48K-fp32-multicast-reduce-4cores": ("both", 49152, FP32, "multicast", "reduce", 4),
}


def case_of(case):
    """(rings, payload bytes, dtype, routes, mode, parts) of a case id."""
    spec = CASES[case]
    return (*spec, 1) if len(spec) == 5 else spec


def build(mesh_device, kind, payload, dtype, routes, mode, parts=1, wait_limit=cl.DEFAULT_WAIT_LIMIT, timing=False):
    shape = tuple(mesh_device.shape)
    # Up to 7 landing cores sit beside the sender in row 0; more take row 1.
    landing_core = cl.LANDING_CORE if parts <= 7 else (0, 1)
    made, stages = {}, []  # a ring set listed twice is one site used twice
    for rings in rings_of(kind, shape):
        key = repr(rings)
        if key not in made:
            made[key] = cl.Site(
                mesh_device, rings, payload, dtype, landing_core, routes=routes, parts=parts, name=f"{kind}{len(made)}"
            )
        stages.append(made[key])
    return stages, cl.Collective(mesh_device, stages, mode, wait_limit=wait_limit, timing=timing)


def random_data(chips, tiles, dtype, seed):
    """Per-chip payloads with a spread of exponents, so the order of a sum shows in its low bits."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randn(chips, tiles * cl.TILE, cl.TILE, generator=g)
    data = data * torch.pow(2.0, torch.randint(-6, 7, data.shape, generator=g).float())
    return data.to(torch.bfloat16).float() if dtype == BF16 else data


def bits(x):
    return x.contiguous().view(torch.int32)


def sample(r, d, limit=6):
    """A few differing elements as "index: reference bits / device bits (values)"."""
    where = (bits(r) != bits(d)).nonzero().flatten()[:limit]
    return ", ".join(
        f"{int(i)}: {int(bits(r)[i]) & 0xFFFFFFFF:08x}/{int(bits(d)[i]) & 0xFFFFFFFF:08x} ({float(r[i]):.6g}/{float(d[i]):.6g})"
        for i in where
    )


def compare(name, collective, sites, data, payload, dtype, mode, strict=True):
    """Every member chip against the host reference, over the payload's elements, bit for bit: gathers, Float32
    sums (summed in the device's order), and BFloat16 sums with the Float32 sum of a stage rounded to BFloat16
    half away from zero. A BFloat16 sum that is not bit-equal still passes within one BF16 step of the reference
    when it has one stage (a second stage sums values that already differ). Returns a summary; with ``strict`` a
    violation raises."""
    mask = cl.payload_mask(sites[0].part_bytes, dtype, sites[0].tiles, sites[0].parts)
    if mode != "reduce":
        mask = mask.repeat(sites[0].members, 1)
    reference, got = collective.reference(data), collective.read()
    worst = {"exact": True, "mismatched": 0, "chips_wrong": 0, "max_abs": 0.0, "pcc": 1.0, "first": ""}
    lenient = dtype == BF16 and mode == "reduce" and len(sites) == 1
    for coord, ref in reference.items():
        r, d = ref[mask], got[coord].float()[mask]
        wrong = int((bits(r) != bits(d)).sum())
        ok = wrong == 0
        if not ok and lenient:
            step = torch.pow(2.0, torch.floor(torch.log2(r.abs().clamp_min(1e-30))) - 7)
            ok = bool(((r - d).abs() <= step).all())
        worst["mismatched"] = max(worst["mismatched"], wrong)
        worst["max_abs"] = max(worst["max_abs"], float((r - d).abs().max()))
        if r.numel() > 16:
            worst["pcc"] = min(worst["pcc"], float(torch.corrcoef(torch.stack([r.double(), d.double()]))[0, 1]))
        if wrong:
            worst["first"] = worst["first"] or f"{coord}: {wrong} of {r.numel()} differ: {sample(r, d)}"
        if not ok:
            worst["chips_wrong"] += 1
    worst["exact"] = worst["mismatched"] == 0
    if strict:
        expect(worst["chips_wrong"] == 0, f"{name}: {worst['chips_wrong']} chips wrong, first {worst['first']}")
    return worst


# One mesh open per group, smallest first.
GROUPS = {
    "pair": [c for c in CASES if c.startswith("pair")],
    "row0": [c for c in CASES if c.startswith("row0") and not c.endswith("cores")],
    "col0": [c for c in CASES if c.startswith("col0")],
    "axis": [c for c in CASES if c.startswith("axis") and not c.endswith("cores")],
    "both": [c for c in CASES if c.startswith("both") and not c.endswith("cores")],
    "striped": [c for c in CASES if c.endswith("cores")],
}
assert sorted(c for cases in GROUPS.values() for c in cases) == sorted(CASES)


def is_idle(site, rounds=None):
    """Counters zero, no status, every core of every member on the same round (``rounds`` when given)."""
    try:
        common = site.check_idle()
    except AssertionError:
        return False
    return rounds is None or common == rounds


def l1_allocated(mesh_device) -> int:
    return ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank


def run_case(mesh_device, case):
    kind, payload, dtype, routes, mode, parts = case_of(case)
    before = l1_allocated(mesh_device)
    sites, collective = build(mesh_device, kind, payload, dtype, routes, mode, parts)
    chips = mesh_device.get_num_devices()
    logger.info(
        f"HY4_COLLECTIVE {case}: {len(sites[0].where)} chips, rings of {[s.members for s in sites]}, "
        f"{parts} x {sites[0].tiles} tiles, chunk {sites[0].chunk_bytes}, "
        f"L1 per core {sum(s.l1_bytes() for s in collective.distinct)} B "
        f"(allocated before {before}, now {l1_allocated(mesh_device)})"
    )
    inputs = []
    try:
        if mock():
            inputs.append(collective.new_input())
            collective.compile(inputs[0])
            logger.info(f"HY4_COLLECTIVE {case}: compiled on the mock mesh")
            return
        # A dry run keeps to one round: the mock stops taking host reads after some ten thousand of them.
        for round_ in range(1 if DRY else 3):
            data = random_data(chips, parts * sites[0].tiles, dtype, seed=1000 * round_ + len(case))
            if round_ == 0:
                inputs.append(collective.new_input(data))
            else:
                collective.write_input(inputs[0], data)
            collective(inputs[0])
            ttnn.synchronize_device(mesh_device)
            try:
                worst = compare(case, collective, sites, data, payload, dtype, mode)
                for site in collective.distinct:
                    expect(is_idle(site, (round_ + 1) * sites.count(site)), f"{case}: {site.name} is not idle")
            except AssertionError:
                for site in collective.distinct:
                    logger.info(f"HY4_COLLECTIVE {case} round {round_} FAILED, {site.summary()}")
                raise
            logger.info(
                f"HY4_COLLECTIVE {case} round {round_}: exact={worst['exact']} mismatched={worst['mismatched']} "
                f"max_abs={worst['max_abs']:.3e} pcc={worst['pcc']:.8f}"
                + (f" first {worst['first']}" if worst["first"] else "")
            )
    finally:
        for tensor in inputs + [collective.output]:
            ttnn.deallocate(tensor)
        for site in collective.distinct:
            site.release()
        assert l1_allocated(mesh_device) == before, f"{case} left L1 allocated: {l1_allocated(mesh_device) - before} B"


@galaxy
@pytest.mark.parametrize("group", list(GROUPS), ids=list(GROUPS))
def test_collective(mesh_device, group):
    only = os.environ.get("HY4_COLLECTIVE_CASE")
    for case in GROUPS[group]:
        if only is None or only == case:
            run_case(mesh_device, case)


def stale_state(mesh_device, case="row0-64B-fp32-unicast-reduce", timing=False):
    """The stale-state story (test_stale_state). Returns the core clock's rate in Hz when the kernels are built
    with ``timing`` (the wait that runs out is then also measured in clock ticks), else 0."""
    kind, payload, dtype, routes, mode, parts = case_of(case)
    # A short bound: this runs into it on purpose (a second or so of polling per waiting core).
    polls = 20_000_000
    sites, collective = build(mesh_device, kind, payload, dtype, routes, mode, parts, wait_limit=polls, timing=timing)
    site = sites[0]
    if mock():
        collective.compile(collective.new_input())
        return
    chips = mesh_device.get_num_devices()
    data = random_data(chips, site.tiles, dtype, seed=7)
    x = collective.new_input(data)

    # 1. Counts and a status left behind, the rounds agreeing: the idle check sees it, reset() clears it.
    site.poke_ctrl(counter0=2, counter1=1, round_=5, status=0xDEAD)
    expect(not is_idle(site), "poked counters and status must not look idle")
    site.reset()
    zero = {"counter": [0, 0], "round": 0, "status": 0}
    expect(all(w == zero for cores in site.read_ctrl().values() for w in cores), "reset() zeroes the control words")
    collective(x)
    compare("after reset", collective, sites, data, payload, dtype, mode)
    expect(is_idle(site, 1), "idle after the run that follows reset()")

    # 2. One member a round ahead of the others (it finished a round the others did not).
    site.reset()
    site.poke_ctrl(rounds={site.rings[0][1]: 1})
    started = tracing.sync(mesh_device)
    collective(x)
    seconds = tracing.sync(mesh_device) - started
    hertz = 0.0
    logger.info(
        f"HY4_COLLECTIVE stale rounds: {polls} polls ran out in {seconds:.3f} s of host time, "
        f"{1e9 * seconds / polls:.1f} ns per poll at most"
    )
    if timing:
        waited = sorted(((c[0]["finish"] - c[0]["handed"]) & 0xFFFFFFFF) for c in site.read_stamps().values())
        hertz = waited[len(waited) // 2] / seconds if seconds > 0 else 0.0
        logger.info(
            f"HY4_COLLECTIVE stale rounds: the wait took {waited[0]} to {waited[-1]} clock ticks on the landing "
            f"cores: about {hertz / 1e6:.0f} MHz"
        )
    words = [w for cores in site.read_ctrl().values() for w in cores]
    logger.info(f"HY4_COLLECTIVE stale rounds: the kernels gave up and left {site.summary()}")
    expect(all(w["status"] >> 24 == 0xA1 for w in words), f"every member reports a landing timeout: {words}")
    expect(any(w["counter"] != [0, 0] for w in words), f"arrivals stay in the counters: {words}")
    expect(not is_idle(site), "members a round apart must not look idle")

    # 3. reset() and the same collective is exact again, twice (both parities).
    site.reset()
    for round_ in range(2):
        collective(x)
        compare("after stale rounds", collective, sites, data, payload, dtype, mode)
        expect(is_idle(site, round_ + 1), "idle after reset() and a run")
    logger.info("HY4_COLLECTIVE stale state: reset() restored the site")
    ttnn.deallocate(x)
    ttnn.deallocate(collective.output)
    site.release()
    return hertz


@galaxy
def test_stale_state(mesh_device):
    """What a run that stopped inside a collective leaves, and that Site.reset() clears it: counts in the arrival
    counters, a status, and members that disagree on the round. Members on different rounds wait on different
    parities, so the next collective cannot complete: with a wait bound the kernels give up, the program ends and
    every member reports a status instead of hanging the mesh. After reset() the collective is exact again."""
    stale_state(mesh_device)


# ---------------------------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------------------------
# payload label -> (bytes, the [rows, cols] of the tile tensor the ttnn op gets for the same bytes)
PAYLOADS = {"64B": (64, (4, 4)), "4KB": (4096, (32, 32)), "48KB": (49152, (96, 128))}
REPEATS = 20
AXIS = {"axis1": 1, "axis0": 0}


def ttnn_forms(mesh_device, kind, shape):
    """{name: fn}: the ttnn-op forms of a collective over ``kind`` on a tile tensor of ``shape`` in DRAM."""
    x = ttnn.from_torch(torch.randn(1, 1, *shape), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.float32)
    dram = ttnn.DRAM_MEMORY_CONFIG
    if kind == "both":
        return {
            "ttnn.all_reduce": lambda: ttnn.all_reduce(
                ttnn.all_reduce(x, cluster_axis=1, memory_config=dram), cluster_axis=0, memory_config=dram
            )
        }
    axis = AXIS[kind]
    return {
        "ttnn.all_reduce": lambda: ttnn.all_reduce(x, cluster_axis=axis, memory_config=dram),
        "ttnn.all_gather": lambda: ttnn.all_gather(x, dim=3, cluster_axis=axis, memory_config=dram),
    }


def fused_forms(kind, payload, tier):
    """(mode, routes, landing cores) of the fused collective measured for ``kind`` and a payload size. Tier 1 is
    what a fused layer would use (multicast; the 48 KB payload also striped over 4 and 12 landing cores, 12 and
    4 KB each); tier 2 the gather with its copy and the unicast routes."""
    if tier == 1:
        forms = [("reduce", "multicast", 1), ("land", "multicast", 1)]
        if payload == 49152:
            forms += [("reduce", "multicast", 4), ("reduce", "multicast", 12), ("land", "multicast", 4)]
    else:
        forms = [("gather", "multicast", 1), ("reduce", "unicast", 1), ("land", "unicast", 1)]
    stages = 2 if kind == "both" else 1  # a stage takes one sender op per landing core
    return [f for f in forms if (f[0] == "reduce" or kind != "both") and stages * f[2] <= cl.MAX_OPS]


# Calls per trace: one (the collective as a trace of its own: every replay sends its kernel binaries again) and
# many (the collective inside a long trace, binaries sent once: how PERF.md section 4 measures its per-op floors).
CALLS = (1, 16)


def traced(mesh_device, fn):
    """{calls per trace: us per call} by tracing.traced_ms, or one plain call in a dry run (the mock cannot capture
    a trace)."""
    if DRY:
        fn()
        return {calls: 0.0 for calls in CALLS}
    return {calls: 1e3 * tracing.traced_ms(mesh_device, fn, REPEATS, calls=calls)["replay_ms"] for calls in CALLS}


def show(us):
    return ", ".join(f"{value:.1f} us per call with {calls} per trace" for calls, value in us.items())


def latency(mesh_device, deadline=None, skip_routes=(), tiers=(1, 2)):
    """The latency table, {row: {form: {calls per trace: us per call}}} (added to the file of an earlier call that
    measured tier 1). ``deadline`` (time.perf_counter() value): stop starting measurements after it."""
    chips = mesh_device.get_num_devices()
    table = {"repeats": REPEATS, "idle_sync_us": 1e3 * tracing.idle_sync_ms(mesh_device), "rows": {}}
    if 1 not in tiers and RESULTS.is_file():
        table["rows"] = json.loads(RESULTS.read_text()).get("rows", {})
    late = lambda: deadline is not None and time.perf_counter() > deadline
    if 1 in tiers:
        tile = ttnn.from_torch(
            torch.randn(1, 1, 32, 32), layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16
        )
        table["rows"]["floor"] = {"ttnn.add, [32 x 32] bf16": traced(mesh_device, lambda: ttnn.add(tile, tile))}
        logger.info(
            f"HY4_COLLECTIVE latency floor: idle synchronize {table['idle_sync_us']:.1f} us; ttnn.add of one tile "
            f"{show(table['rows']['floor']['ttnn.add, [32 x 32] bf16'])}"
        )
    for tier in tiers:
        for kind in ("axis1", "both", "axis0"):
            for label, (payload, shape) in PAYLOADS.items():
                row = table["rows"].setdefault(f"{kind} {label}", {})
                for mode, routes, parts in fused_forms(kind, payload, tier):
                    if routes in skip_routes or late():
                        continue
                    name = f"fused {mode} {routes}" + (f" {parts} cores" if parts > 1 else "")
                    sites, collective = build(mesh_device, kind, payload, FP32, routes, mode, parts)
                    data = random_data(chips, parts * sites[0].tiles, FP32, seed=payload + len(kind))
                    x = collective.new_input(data)
                    row[name] = traced(mesh_device, lambda: collective(x))
                    worst = compare(
                        f"{kind} {label} {name}", collective, sites, data, payload, FP32, mode, strict=False
                    )
                    idle = all(is_idle(site) for site in collective.distinct)
                    logger.info(
                        f"HY4_COLLECTIVE latency {kind} {label} {name}: {show(row[name])}; after the replays "
                        f"exact={worst['exact']} chips_wrong={worst['chips_wrong']} idle={idle}"
                    )
                    ttnn.deallocate(x)
                    ttnn.deallocate(collective.output)
                    for site in collective.distinct:
                        site.release()
                    expect(idle and worst["chips_wrong"] == 0, f"{kind} {label} {name}: idle={idle} {worst['first']}")
                for name, fn in ttnn_forms(mesh_device, kind, shape).items() if tier == 1 else ():
                    if late():
                        continue
                    row[name] = traced(mesh_device, fn)
                    logger.info(f"HY4_COLLECTIVE latency {kind} {label} {name}: {show(row[name])}")
                RESULTS.write_text(json.dumps(table, indent=1))
    return table


# (rings, payload bytes, mode, routes, landing cores) of the collectives whose time is broken down
BREAKDOWNS = [
    ("axis1", 64, "land", "multicast", 1),
    ("axis1", 49152, "land", "multicast", 1),
    ("axis1", 49152, "reduce", "multicast", 1),
    ("axis1", 49152, "reduce", "multicast", 4),
    ("axis1", 49152, "reduce", "multicast", 12),
    ("axis0", 64, "land", "multicast", 1),
    ("axis0", 49152, "land", "multicast", 1),
    ("axis1", 49152, "land", "unicast", 1),
]
# name -> (later stamp, earlier stamp): the first five on a landing core, the rest on the sender core
PHASES = {
    "kernel start to hand_over": ("hand_over", "start"),
    "hand_over (copy to the sender, ready bit)": ("handed", "hand_over"),
    "handed to every member landed": ("landed", "handed"),
    "landed to finish (reduce: the sum)": ("finish", "landed"),
    "sender: start to connections open": ("sender_open", "sender_start"),
    "sender: open to first op ready": ("sender_ready", "sender_open"),
    "sender: ready to all sent": ("sender_sent", "sender_ready"),
    "sender: sent to connections closed": ("sender_closed", "sender_sent"),
}


def breakdown(mesh_device, hertz=0.0, deadline=None):
    """Where a collective's time goes, from wall-clock stamps the kernels leave (built with timing=True) in the last
    of two blocking replays of a trace: per phase the least, the median and the most over the chips, in
    microseconds when the clock's rate is known (``hertz``, from stale_state), in clock ticks otherwise. The phases
    of a landing core and of the sender core are each on that core's own clock."""
    chips = mesh_device.get_num_devices()
    unit, scale = ("us", 1e6 / hertz) if hertz > 0 else ("ticks", 1.0)
    for kind, payload, mode, routes, parts in BREAKDOWNS:
        if deadline is not None and time.perf_counter() > deadline:
            continue
        sites, collective = build(mesh_device, kind, payload, FP32, routes, mode, parts, timing=True)
        x = collective.new_input(random_data(chips, parts * sites[0].tiles, FP32, seed=payload))
        collective(x)
        if not DRY:
            trace_id, _ = tracing.capture(mesh_device, lambda: collective(x))
            for _ in range(2):
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
            ttnn.release_trace(mesh_device, trace_id)
        stamps = [cores[0] for cores in sites[0].read_stamps().values()]
        lines = []
        for phase, (later, earlier) in PHASES.items():
            took = sorted(scale * ((s[later] - s[earlier]) & 0xFFFFFFFF) for s in stamps)
            lines.append(f"{phase}: {took[0]:.1f} / {took[len(took) // 2]:.1f} / {took[-1]:.1f}")
        name = f"{kind} {payload} B {mode} {routes}" + (f" {parts} cores" if parts > 1 else "")
        logger.info(f"HY4_COLLECTIVE breakdown {name} ({unit}, min / median / max over chips): " + "; ".join(lines))
        idle = all(is_idle(site) for site in collective.distinct)
        ttnn.deallocate(x)
        ttnn.deallocate(collective.output)
        for site in collective.distinct:
            site.release()
        expect(idle, f"breakdown {name}: the site is not idle")


def marginal(mesh_device, deadline=None):
    """What one more collective costs inside a program: the traced time per call (16 calls per trace) of a reduce
    of 1, 2 and 4 stages over one site (each stage sums the stage before's sums), and (4 stages - 1 stage) / 3.
    The 48 KB payload is striped over 4 landing cores."""
    chips = mesh_device.get_num_devices()
    out = {}
    for kind in ("axis1", "axis0"):
        for label, payload, parts in (("64B", 64, 1), ("4KB", 4096, 1), ("48KB", 49152, 4)):
            if deadline is not None and time.perf_counter() > deadline:
                continue
            us, exact = {}, True
            for stages in (1, 2, 4):
                sites, collective = build(mesh_device, f"{kind}x{stages}", payload, FP32, "multicast", "reduce", parts)
                data = random_data(chips, parts * sites[0].tiles, FP32, seed=payload + stages)
                x = collective.new_input(data)
                us[stages] = traced(mesh_device, lambda: collective(x))[CALLS[-1]]
                worst = compare(
                    f"{kind}x{stages} {label}", collective, sites, data, payload, FP32, "reduce", strict=False
                )
                exact = exact and worst["chips_wrong"] == 0 and all(is_idle(site) for site in collective.distinct)
                ttnn.deallocate(x)
                ttnn.deallocate(collective.output)
                for site in collective.distinct:
                    site.release()
            out[f"{kind} {label}"] = dict(us, marginal=(us[4] - us[1]) / 3)
            logger.info(
                f"HY4_COLLECTIVE marginal {kind} {label} (multicast reduce, {parts} landing cores, {CALLS[-1]} calls per "
                f"trace): 1 / 2 / 4 stages in one program {us[1]:.1f} / {us[2]:.1f} / {us[4]:.1f} us, "
                f"{(us[4] - us[1]) / 3:.1f} us per added collective; exact and idle after the replays: {exact}"
            )
            expect(exact, f"marginal {kind} {label}: a staged reduce was not exact or not idle")
    return out


@galaxy
def test_marginal(mesh_device):
    """The cost of one more collective inside a program (see marginal)."""
    if mock():
        pytest.skip("the marginal cost needs the board")
    table = json.loads(RESULTS.read_text()) if RESULTS.is_file() else {}
    table["marginal_us"] = marginal(mesh_device)
    RESULTS.write_text(json.dumps(table, indent=1))


@galaxy
def test_timing_first(mesh_device):
    """The kernels built with timing stamps on two chips and 64 bytes: the first board run of that build, to be
    run alone before test_breakdown. The sum is exact, the site idle, and every stamp of both chips was written."""
    sites, collective = build(mesh_device, "pair", 64, FP32, "unicast", "reduce", timing=True)
    x = collective.new_input()
    if mock():
        collective.compile(x)
        return
    chips = mesh_device.get_num_devices()
    for round_ in range(2):
        data = random_data(chips, sites[0].tiles, FP32, seed=round_)
        collective.write_input(x, data)
        collective(x)
        ttnn.synchronize_device(mesh_device)
        compare("timing build", collective, sites, data, 64, FP32, "reduce")
        expect(is_idle(sites[0], round_ + 1), f"timing build: {sites[0].summary()}")
    stamps = sites[0].read_stamps()
    logger.info(f"HY4_COLLECTIVE timing build on two chips: stamps {stamps}")
    expect(all(value != 0 for cores in stamps.values() for value in cores[0].values()), f"a stamp is missing: {stamps}")


@galaxy
def test_breakdown(mesh_device):
    """The time of a collective by phase, from the kernels' own clock (see breakdown). Run test_timing_first, alone,
    before the first run of this on a new build of the kernels."""
    if mock():
        pytest.skip("the breakdown needs the board")
    hertz = stale_state(mesh_device, timing=True)
    breakdown(mesh_device, hertz)


@galaxy
def test_latency(mesh_device):
    """Traced latency of the standalone collective next to ttnn.all_reduce / ttnn.all_gather on the same bytes and
    mesh: 20 replays back to back, one synchronize, best of three rounds, with one call and with 16 calls per
    trace. Nothing is asserted on the times; every traced collective is checked against the reference after its
    replays."""
    if mock():
        pytest.skip("latency needs the board")
    latency(mesh_device)


@galaxy
def test_collective_pair_to_galaxy(mesh_device):
    """Everything above but the timing build (test_timing_first, test_breakdown) in ONE mesh open, smallest first,
    for a board with a long queue: the cases of every group, the stale-state story, then as much of the latency
    tables as fits the time budget ($HY4_COLLECTIVE_BUDGET_S, default 480 s). A case with wrong results is logged and the rest goes on (a route kind that failed is left out
    of the latency table); anything else, a device timeout above all, stops the test at once. Fails at the end if
    any case failed."""
    start = time.perf_counter()
    deadline = start + float(os.environ.get("HY4_COLLECTIVE_BUDGET_S", "480"))
    failed, broken_routes = [], set()
    for group in GROUPS:
        for case in GROUPS[group]:
            covered = sum(len(ring) for ring in rings_of(case_of(case)[0], tuple(mesh_device.shape))[0])
            if DRY and covered < mesh_device.get_num_devices():
                continue  # the mock cannot enqueue on part of the mesh
            if not mock() and time.perf_counter() > deadline:
                logger.info(f"HY4_COLLECTIVE {case}: left out, the time budget is spent")
                continue
            try:
                run_case(mesh_device, case)
            except AssertionError as failure:
                failed.append(case)
                if case_of(case)[2] == FP32:  # a Float32 case is exact or the route kind does not work
                    broken_routes.add(case_of(case)[3])
                logger.info(f"HY4_COLLECTIVE {case} FAILED: {str(failure)[:600]}")
    if mock():
        return
    logger.info(f"HY4_COLLECTIVE functional cases done after {time.perf_counter() - start:.0f} s, failed: {failed}")
    if "unicast" not in broken_routes:
        try:
            stale_state(mesh_device, "axis1-64B-fp32-unicast-reduce" if DRY else "row0-64B-fp32-unicast-reduce")
        except AssertionError as failure:
            failed.append("stale state")
            logger.info(f"HY4_COLLECTIVE stale state FAILED: {str(failure)[:600]}")
    logger.info(f"HY4_COLLECTIVE stale state done after {time.perf_counter() - start:.0f} s")
    for name, part in (
        ("latency, what a fused layer would use", lambda: latency(mesh_device, deadline, broken_routes, tiers=(1,))),
        ("marginal cost", lambda: marginal(mesh_device, deadline) if not broken_routes else None),
        ("latency, the other forms", lambda: latency(mesh_device, deadline, broken_routes, tiers=(2,))),
    ):
        try:
            part()
        except AssertionError as failure:
            failed.append(name)
            logger.info(f"HY4_COLLECTIVE {name} FAILED: {str(failure)[:600]}")
        logger.info(f"HY4_COLLECTIVE {name} done after {time.perf_counter() - start:.0f} s")
    logger.info(f"HY4_COLLECTIVE all done after {time.perf_counter() - start:.0f} s, failed: {failed}")
    assert not failed, failed
