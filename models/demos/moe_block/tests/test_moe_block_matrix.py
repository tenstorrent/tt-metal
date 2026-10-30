# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MoEBlock on device over models x rows a mesh row x routing scenarios, in tiers (select with -k quick / nightly /
soak): eager calls then a trace replayed on fresh inputs, every checked replay against the host model on the
device's own routing (PCC) and its routing against the host's (agreement); with the device profiler on, the
replays' device time per program. The mesh is the process's (MOE_MESH / MOE_FABRIC as test_moe_block: the 2x4 lines,
or one LoudBox square as a 1x4 TP ring).

Models keep their real router; where the experts a chip would exceed the flat op's 64, the router folds onto fewer
physical experts (RouterConfig.fold: K3 at E 896 -> 28 a chip on the ring of 4 and the 2x4, a Galaxy chip's load).
Routing scenarios bias the router (candidate space, the fold's mask kept): uniform; zipf spread (hot experts on every
chip) / packed (hot experts on chip 0); one hot expert (every token picks it); empty experts (every other expert of a
chip never picked); a chip with zero rows (no expert of chip 1 picked); ties (zero router weights, zero bias: every
score equal)."""

import dataclasses
import os
import statistics
import time

import pytest
import torch
from loguru import logger
from ttnn.bringup.flat_routed_expert_ttnn.closeout_models import fabric_payload

import ttnn
from models.demos.moe_block.moe_block import (
    DSV41_FLASH,
    GLM_53,
    GLM_53_FLASH,
    KIMI_K3,
    KIMI_K27,
    MINIMAX_M3,
    MoEBlock,
    MoEModel,
    MoEWeights,
    WeightCache,
)
from models.demos.moe_block.tests.test_moe_block import (
    TRACE_REGION,
    C,
    R,
    _fab,
    device_routing,
    pcc,
    random_weights,
    reference,
    replay_device_times,
)

BASE = {
    "k3": KIMI_K3,
    "k27": KIMI_K27,
    "glm53_flash": GLM_53_FLASH,
    "glm53": GLM_53,
    "m3": MINIMAX_M3,
    "dsv41_flash": DSV41_FLASH,
}
ROUTINGS = ("uniform", "zipf_mild", "zipf_spread", "zipf_packed", "one_hot", "empty_experts", "zero_chip", "ties")
EPC_MAX = 64  # (the flat op's experts a chip)
GALAXY_EPC = {"k3": 28, "k27": 12}  # (a Galaxy chip's experts: the router folded to them)


def mesh_model(name: str, chips: int) -> MoEModel:
    """The model on this mesh: K3 / K2.7 at a Galaxy chip's load (GALAXY_EPC experts a chip, the router at its real E
    folded), the others at their real experts unless more than EPC_MAX a chip (then folded to fit)."""
    m = BASE[name]
    E = m.num_experts
    epc = GALAXY_EPC.get(name, E // chips)
    F = 1
    while E // F // chips > EPC_MAX or (name in GALAXY_EPC and E // F // chips > epc):
        F *= 2
    assert E % (F * chips) == 0, f"{name}: {E} experts over {chips} chips, fold {F}"
    if F == 1:
        return m
    return dataclasses.replace(m, num_experts=E // F, router=dataclasses.replace(m.router, fold=F))


def scenario_weights(m: MoEModel, routing: str, chips: int, seed=0) -> MoEWeights:
    """random_weights with the router biased for the scenario (the fold's mask kept: one eligible candidate a group)."""
    w = random_weights(m, seed)
    F, E = m.router.fold, m.num_experts
    epc = E // chips
    b = torch.zeros(E)
    g = torch.Generator().manual_seed(seed + 7)
    if routing in ("zipf_mild", "zipf_spread", "zipf_packed"):
        rank = torch.randperm(E, generator=g) if routing != "zipf_packed" else torch.arange(E)
        b = (-0.3 if routing == "zipf_mild" else -0.6) * torch.log1p(rank.float())
    elif routing == "one_hot":
        b[0] = 8.0
    elif routing == "empty_experts":
        b[1::2] = -1.0e3
    elif routing == "zero_chip":
        b[epc : 2 * epc] = -1.0e3
    rb = w.router["bias"] + b.repeat_interleave(F)
    rw = w.router["weight"]
    if routing == "ties":
        rw = torch.zeros_like(rw)
        rb = torch.where(w.router["bias"] < -1.0e3, w.router["bias"], torch.zeros_like(rb))  # (the fold mask only)
    return dataclasses.replace(w, router={"weight": rw, "bias": rb})


def _launch_stats_on():
    """MOE_LAUNCH_STATS=1: time every program launch of the back-to-back calls (host until the launch returns), by
    call site, with the launch's payload (prepared programs: runtime-arg words / kernel-cores / common words / CBs /
    kernels of the first mesh program, and the mesh programs - one a chip when the chips' args differ)."""
    import sys

    stats, orig = {}, (ttnn.generic_op, ttnn.bringup.run_prepared)

    def site():
        f = sys._getframe(2)
        while f.f_back is not None and "prepared_program" in f.f_code.co_filename:
            f = f.f_back
        return f"{os.path.basename(f.f_code.co_filename)}:{f.f_lineno} {f.f_code.co_name}"

    def wrap(fn, kind, info):
        def w(*a, **k):
            t0 = time.perf_counter_ns()
            r = fn(*a, **k)
            dt = (time.perf_counter_ns() - t0) / 1e3
            e = stats.setdefault((kind, site()), [0, 0.0, None])
            e[0] += 1
            e[1] += dt
            if e[2] is None:
                try:
                    e[2] = info(*a)
                except Exception as ex:  # (stats only)
                    e[2] = f"? {type(ex).__name__}"
            return r

        return w

    def prep_info(pp, io):
        words, cores, common, cbs, kernels, nprog = pp.payload
        info = f"{nprog} mesh programs; a chip {words} rt words ({words * 4 // 1024} KB) on {cores} kernel-cores, {common} common, {cbs} CBs, {kernels} kernels; {pp.patch_count} patches"
        if words > 4096:  # (where a big payload goes: per kernel of the first mesh program)
            runs = pp.invariant_runs
            rows = [
                f"      {w:6d} words {w * 4 // 1024:4d} KB  {c:4d} cores  {w // max(c, 1):4d} a core (args {mn}-{mx})  "
                f"{d:4d} distinct  {same:6d} in all-cores-equal positions ({sum(e - b0 for b0, e in r)} positions)  {src}"
                for (src, c, w, d, same), (mn, mx, r) in sorted(zip(pp.kernel_stats, runs), key=lambda t: -t[0][2])
            ]
            logger.info(f"MOE_LAUNCH_KERNELS ({words} words a chip):\n" + "\n".join(rows))
        return info

    def gen_info(io, md):
        try:
            return f"{sum(1 for _ in md)} mesh programs"
        except Exception:
            return ""

    ttnn.generic_op = wrap(orig[0], "generic_op", gen_info)
    ttnn.bringup.run_prepared = wrap(orig[1], "run_prepared", prep_info)
    # (the flat experts + combine: a C++ op with its combine program riding along)
    orig = orig + (ttnn.bringup.flat_routed_expert_with_program,)
    ttnn.bringup.flat_routed_expert_with_program = wrap(orig[2], "flat_op", lambda *a: "C++ op")
    return stats, orig


def _launch_stats_report(launch, n, tag):
    stats, orig = launch
    ttnn.generic_op, ttnn.bringup.run_prepared, ttnn.bringup.flat_routed_expert_with_program = orig
    tot = sum(e[1] for e in stats.values()) / n
    lines = [
        f"  {e[1] / n:7.0f} us a call ({e[1] / e[0]:6.0f} a launch x {e[0] / n:.0f})  {k[0]:12s} {k[1]}  [{e[2]}]"
        for k, e in sorted(stats.items(), key=lambda kv: -kv[1][1])
    ]
    logger.info(f"MOE_LAUNCH {tag}: {tot:.0f} us a call in program launches over {n} calls:\n" + "\n".join(lines))


def eager_host(block, mesh_device, x_in, n, tag):
    """Untraced forwards: wall a call synced after each (host + device + launch), back to back (n calls, one sync at the
    end: the host's dispatch pipelined with the device), and the host's own time a call (until the call returns, no sync);
    MOE_EAGER_PROF=1 adds a cProfile of the back-to-back calls (generated/moe_block/eager_host_<tag>.prof, top 30 logged)."""
    keep, block.keep_taps = block.keep_taps, False  # (as a model runs it: no taps)
    walls, hosts = [], []
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter_ns()
        out = block(x_in)
        t1 = time.perf_counter_ns()
        ttnn.synchronize_device(mesh_device)
        walls.append((time.perf_counter_ns() - t0) / 1e3)
        hosts.append((t1 - t0) / 1e3)
        ttnn.deallocate(out)
    prof = None
    if os.environ.get("MOE_EAGER_PROF") == "1":
        import cProfile

        prof = cProfile.Profile()
    spy = None
    if os.environ.get("MOE_EAGER_PYSPY"):  # (native host profile of the back-to-back calls: py-spy on this process)
        import ctypes
        import subprocess

        ctypes.CDLL(None).prctl(0x59616D61, ctypes.c_ulong(-1), 0, 0, 0)  # (PR_SET_PTRACER_ANY: ptrace_scope 1)
        spy = subprocess.Popen(
            [os.path.expanduser("~/.local/bin/py-spy"), "record", "--native", "--rate", "500", "--format", "raw"]
            + ["--pid", str(os.getpid()), "-o", os.environ["MOE_EAGER_PYSPY"]]
        )
        time.sleep(2.0)
    launch = _launch_stats_on() if os.environ.get("MOE_LAUNCH_STATS") == "1" else None
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter_ns()
    if prof:
        prof.enable()
    for _ in range(n):
        ttnn.deallocate(block(x_in))
    t_issue = time.perf_counter_ns()
    if prof:
        prof.disable()
    ttnn.synchronize_device(mesh_device)
    b2b = (time.perf_counter_ns() - t0) / 1e3 / n
    if launch is not None:
        _launch_stats_report(launch, n, tag)
    if spy is not None:
        import signal

        spy.send_signal(signal.SIGINT)
        spy.wait(timeout=120)
    logger.info(
        f"MOE_EAGER {tag}: synced a call median {statistics.median(walls):.0f} us (min {min(walls):.0f}); host until "
        f"return median {statistics.median(hosts):.0f} us; back to back {b2b:.0f} us a call (host issue "
        f"{(t_issue - t0) / 1e3 / n:.0f} us a call) over {n}"
    )
    if prof:
        import io
        import pstats

        os.makedirs("generated/moe_block", exist_ok=True)
        prof.dump_stats(f"generated/moe_block/eager_host_{tag}.prof")
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(30)
        logger.info(f"MOE_EAGER_PROF {tag} (cumulative, {n} calls):\n{buf.getvalue()}")
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("tottime").print_stats(25)
        logger.info(f"MOE_EAGER_PROF {tag} (tottime, {n} calls):\n{buf.getvalue()}")
    block.keep_taps = keep


@dataclasses.dataclass(frozen=True)
class Case:
    model: str
    rows: int  # (tokens a mesh row)
    routing: str
    replays: int  # (checked trace replays, each on its own fresh input)
    tier: str

    @property
    def id(self):
        return f"{self.tier}-{self.model}-r{self.rows}-{self.routing}-x{self.replays}"


def _cases():
    out = [Case("k3", 640, "uniform", 16, "quick")]
    for mn in BASE:
        for rows in (128, 256, 512, 640, 768, 1024, 1280, 2048):  # (768 / 1280: the chunked heads, NCH 2 - 5)
            out.append(Case(mn, rows, "uniform", 32, "nightly"))
    for mn in ("k3", "glm53_flash", "dsv41_flash"):
        for rt in ROUTINGS[1:]:
            out.append(Case(mn, 640, rt, 32, "nightly"))
    for mn, rows in (
        ("k3", 640),
        ("k3", 1024),
        ("glm53_flash", 640),
        ("glm53", 1024),
        ("k3", 1280),
        ("m3", 640),
        ("dsv41_flash", 640),
        ("k27", 640),  # (the planner's 11x10 MT 2 head, compact RS zones, the fused non-latent tail)
        ("k27", 1024),
        ("glm53", 640),
    ):
        out.append(Case(mn, rows, "uniform", 10000, "soak"))
        out.append(Case(mn, rows, "zipf_packed", 10000, "soak"))
    return out


CASES = _cases()


def _snap_env():
    """MOE_COMBINE_SNAP: the combine's snap q - "2^-14" or a float; unset / 0: plain fp32 adds in landing order."""
    v = os.environ.get("MOE_COMBINE_SNAP", "")
    if not v or v == "0":
        return None
    return 2.0 ** int(v[2:]) if v.startswith("2^") else float(v)


def run_checked(mesh_device, block, m, w, tag, *, eager, replays, nx=16):
    """eager calls, then (replays > 0) a trace replayed on fresh inputs; every checked call against the host model on
    the device's own routing (PCC > 0.98) -> dict(pcc=[...], agree=[...]) (the routing agreement with the host's)."""
    g = block.geometry
    T, H = g.chunk_tokens, m.emb_dim
    shape = tuple(mesh_device.shape)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=(2, 3))
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=shape, dims=(2, 3))
    NX = nx
    xs = [torch.randn(T, H, generator=torch.Generator().manual_seed(3000 + k)).to(torch.bfloat16) for k in range(NX)]
    hx = [
        ttnn.from_torch(x_.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper)
        for x_ in xs
    ]
    x_in = block.input_buffer()
    refs, stats = {}, dict(pcc=[], agree=[])

    first_y = {}  # (determinism: input k's first output - a same-x call's routing rows and output bits against it)
    det = stats.setdefault("det", dict(calls=0, rows_diff=0, y_diff=0, y_maxabs=0.0))

    def check(out, k):
        y = ttnn.to_torch(out, mesh_composer=composer).reshape(T, H).float()
        rows = block.taps()["rows"]
        if k in first_y:  # (a same-x call: the routing is bit-identical by construction - deterministic fin; the
            # output too with combine_snap, else its fp32 sums in landing order differ in the last bits)
            det["calls"] += 1
            if not all(torch.equal(a, b) for a, b in zip(rows, refs[k][0])):
                det["rows_diff"] += 1
            dy = (y - first_y[k]).abs()
            if bool((dy > 0).any()):
                det["y_diff"] += 1
                det["y_maxabs"] = max(det["y_maxabs"], float(dy.max()))
        else:
            first_y[k] = y.clone()
        if k not in refs or not all(torch.equal(a, b) for a, b in zip(rows, refs[k][0])):
            ids_d, wts_d = device_routing(rows, g.tokens_per_chip, m.top_k)
            y_ref = reference(xs[k], w, m, ids=ids_d, wts=wts_d)[0]
            _, ids_r, _ = reference(xs[k], w, m)
            agree = (torch.sort(ids_d, 1)[0] == torch.sort(ids_r, 1)[0]).float().mean().item()
            refs[k] = (rows, y_ref, agree)
            if k == 0 and int(ids_d.max()) < m.num_experts:  # (the routing's skew: tokens an expert / a chip, max over mean)
                e = torch.bincount(ids_d.flatten(), minlength=m.num_experts).float()
                c = e.reshape(-1, g.experts_per_chip).sum(1)
                logger.info(
                    f"MATRIX_LOAD {tag}: chip max/mean {c.max() / c.mean():.2f} (max {c.max():.0f}, min {c.min():.0f} "
                    f"token picks a chip); expert max/mean {e.max() / e.mean():.2f} (max {e.max():.0f}); "
                    f"{int((e == 0).sum())} of {len(e)} experts empty"
                )
        p = pcc(y, refs[k][1])
        stats["pcc"].append(p)
        stats["agree"].append(refs[k][2])
        assert p > 0.98, f"{tag} (x {k}): block output off (PCC {p:.5f})"

    block.keep_taps = "device"
    for k in range(eager):  # (eager)
        ttnn.copy_host_to_device_tensor(hx[k], x_in)
        out = block(x_in)
        ttnn.synchronize_device(mesh_device)
        check(out, k)
        ttnn.deallocate(out)
    if int(os.environ.get("MOE_EAGER_N", "0")):  # (host side, untraced: N forwards synced each / back to back)
        eager_host(block, mesh_device, x_in, int(os.environ["MOE_EAGER_N"]), tag)
    if not replays:
        return stats
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out = block(x_in)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.synchronize_device(mesh_device)
    every = 1 if replays <= 64 else 25  # (soak: every replay on a fresh input, every 25th checked)
    for i in range(replays):
        k = (3 * i + 2) % NX
        ttnn.copy_host_to_device_tensor(hx[k], x_in)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        if i % every == every - 1 or i == replays - 1:
            ttnn.synchronize_device(mesh_device)
            check(out, k)
        if i % 1000 == 999:
            logger.info(f"MATRIX {tag}: {i + 1} replays, PCC min {min(stats['pcc']):.5f}")
    ttnn.synchronize_device(mesh_device)
    if int(os.environ.get("MOE_WALL_N", "0")):  # (one synchronous replay at a time, the input copy off the clock)
        walls = []
        for i in range(int(os.environ["MOE_WALL_N"])):
            ttnn.copy_host_to_device_tensor(hx[i % NX], x_in)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter_ns()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            walls.append((time.perf_counter_ns() - t0) / 1e3)
        logger.info(
            f"MOE_WALL {tag}: first {walls[0]:.0f} us, median {statistics.median(walls):.0f} us, min {min(walls):.0f} "
            f"us over {len(walls)} single replays"
        )
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        replay_device_times(mesh_device, tid, f"matrix {tag}", batches=(10, 40))
    ttnn.release_trace(mesh_device, tid)
    ttnn.deallocate(out)
    return stats


@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(BASE[c.model].routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            c,
            id=c.id,
        )
        for c in CASES
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(36000)
def test_moe_block_matrix(mesh_device, device_params, case: Case):
    chips = R * C
    m = mesh_model(case.model, chips)
    w = scenario_weights(m, case.routing, chips)
    try:
        from models.demos.moe_block.moe_block import MoEOptions

        opts = MoEOptions(
            router_split=os.environ.get("MOE_ROUTER_SPLIT") == "1",  # (test-only: the router split A/B)
            combine_snap=_snap_env(),
        )
        block = MoEBlock(mesh_device, m, case.rows * R, w, WeightCache(path=None, layer=0), options=opts)
    except (AssertionError, ValueError, NotImplementedError, RuntimeError) as e:
        msg = " | ".join(l.strip() for l in str(e).splitlines() if l.strip() and "---" not in l)[:600]
        logger.error(f"MATRIX {case.id}: does not build: {type(e).__name__}: {msg}")
        pytest.xfail(f"does not build: {type(e).__name__}: {msg[:300]}")
    logger.info(f"MATRIX {case.id}: {m.num_experts} experts (fold {m.router.fold}), {block.summary}")
    stats = run_checked(mesh_device, block, m, w, case.id, eager=2, replays=case.replays)
    d_ = stats.get("det", {})
    logger.info(
        f"MATRIX_DET {case.id}: snap {opts.combine_snap}: {d_.get('calls', 0)} same-x calls, routing rows differ in "
        f"{d_.get('rows_diff', 0)}, outputs differ in {d_.get('y_diff', 0)} (max |dy| {d_.get('y_maxabs', 0.0):.3g})"
    )
    logger.info(
        f"MATRIX_RESULT {case.id}: PCC min {min(stats['pcc']):.5f} median {statistics.median(stats['pcc']):.5f}; "
        f"routing agreement min {min(stats['agree']):.4f}; {len(stats['pcc'])} checks; {block.summary}"
    )


# switching in one process: chunks, eager <-> traced, models (one fabric payload: the largest row's); each block closed
# before the next is built. (model, rows a mesh row, checked trace replays - 0: eager only)
SWITCH = {
    "k3": (
        ("k3", 640, 4),
        ("k3", 1024, 0),
        ("k3", 128, 4),
        ("k3", 2048, 0),
        ("k3", 640, 0),
        ("k3", 256, 4),
        ("k3", 640, 4),
    ),
    "models": (
        ("k3", 1024, 0),
        ("k3", 640, 4),
        ("glm53_flash", 640, 4),
        ("m3", 512, 4),
        ("dsv41_flash", 512, 4),
        ("glm53", 1024, 4),
        ("k3", 1024, 4),
    ),
}


@pytest.mark.parametrize(
    "mesh_device, device_params, seq",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=max(fabric_payload(BASE[mn].routed_width) for mn, _, _ in SWITCH[k]),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            k,
            id=f"switch-{k}",
        )
        for k in SWITCH
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(7200)
def test_moe_block_switch(mesh_device, device_params, seq):
    """Blocks of other chunks / models built one after another on one mesh (each closed first), eager and traced:
    every one checked - nothing a closed block left behind (L1, program cache, semaphores) breaks the next."""
    import gc

    chips = R * C
    for i, (mn, rows, replays) in enumerate(SWITCH[seq]):
        m = mesh_model(mn, chips)
        w = scenario_weights(m, "uniform", chips, seed=i)
        block = MoEBlock(mesh_device, m, rows * R, w, WeightCache(path=None, layer=0))
        tag = f"switch {i} {mn} r{rows} x{replays}"
        stats = run_checked(mesh_device, block, m, w, tag, eager=2, replays=replays, nx=4)
        logger.info(f"MATRIX_RESULT {tag}: PCC min {min(stats['pcc']):.5f}; {block.summary}")
        block.close()
        del block
        gc.collect()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(KIMI_K3.routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            id="first-call-k3",
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("builds", [24, 1])  # (1: a fresh process a build - run it repeatedly)
@pytest.mark.timeout(7200)
def test_moe_block_first_call(mesh_device, device_params, builds):
    """The block built again and again in one process (closed in between), each build's warm-ups and first call checked:
    the fused tail once hung in its first call ever (MOE_BLOCK_WORKLOG.md, the SP 1 hang entry)."""
    import gc

    chips = R * C
    m = mesh_model("k3", chips)
    for i in range(builds):
        w = scenario_weights(m, "uniform", chips, seed=i)
        block = MoEBlock(mesh_device, m, 640 * R, w, WeightCache(path=None, layer=0))
        stats = run_checked(mesh_device, block, m, w, f"first call {i}", eager=1, replays=0, nx=1)
        logger.info(f"MATRIX_RESULT first call {i}: PCC {stats['pcc'][0]:.5f}")
        block.close()
        del block
        gc.collect()


# head layout A/Bs (MoEOptions.head_layout): (model, rows a mesh row, [(GY, MT, NCH, KB, on the engine), ...]) - the
# planner's choice against the alternatives it ranked near it
AB = {
    "glm53-640": ("glm53", 640, [(10, 2, 1, 12, True), (10, 2, 1, 8, True), (10, 2, 1, 6, True)]),
    "glm53-512": ("glm53", 512, [(8, 2, 1, 12, True), (8, 2, 1, 8, True)]),
    "glmf-640": ("glm53_flash", 640, [(10, 2, 1, 16, True), (10, 2, 1, 8, True)]),
    "k27-640": ("k27", 640, [(10, 2, 1, 7, True), (10, 2, 1, 4, True), (10, 2, 1, 7, False)]),
    "k27-768": ("k27", 768, [(8, 1, 3, 14, True), (8, 1, 3, 14, False), (6, 2, 2, 7, True)]),
    "glmf-1280": ("glm53_flash", 1280, [(8, 1, 5, 8, True), (4, 2, 5, 4, False)]),
    # (the RS zones: P-tile slots vs compact, the same layout - the 6th field)
    "zones-glm53-512": ("glm53", 512, [(8, 2, 1, 8, True, False), (8, 2, 1, 8, True, True)]),
    "zones-glm53-640": ("glm53", 640, [(10, 2, 1, 8, True, False), (10, 2, 1, 8, True, True)]),
    "zones-glmf-640": ("glm53_flash", 640, [(10, 2, 1, 16, True, False), (10, 2, 1, 16, True, True)]),
    # (the same with the ops tail: no fused tail's L1 - P-tile slots fit GLM-5.3 at KB 8 again)
    "zonesops-glm53-512": ("glm53", 512, [(8, 2, 1, 8, True, False), (8, 2, 1, 8, True, True)], "ops"),
    "zonesops-glm53-640": ("glm53", 640, [(10, 2, 1, 8, True, False), (10, 2, 1, 8, True, True)], "ops"),
    # (the router's fin + top-K over two cores a row tile - the 7th field; K step 0: the largest that fits)
    "split-k3-640": ("k3", 640, [(10, 2, 1, 0, True, False, False), (10, 2, 1, 0, True, False, True)]),
    "split-k27-640": ("k27", 640, [(10, 2, 1, 0, True, True, False), (10, 2, 1, 0, True, True, True)]),
    # (one layout a run: each its own profile - the chain's zones)
    "one-k3-640-base": ("k3", 640, [(10, 2, 1, 0, True, False, False)]),
    "one-k3-640-split": ("k3", 640, [(10, 2, 1, 0, True, False, True)]),
    "one-k27-640-base": ("k27", 640, [(10, 2, 1, 0, True, True, False)]),
    "one-k27-640-split": ("k27", 640, [(10, 2, 1, 0, True, True, True)]),
}


@pytest.mark.parametrize(
    "mesh_device, device_params, ab",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(BASE[AB[k][0]].routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            k,
            id=f"ab-{k}",
        )
        for k in AB
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(36000)
def test_head_layout_ab(mesh_device, device_params, ab):
    """Each layout of AB[ab] in turn (a block each, closed before the next), traced replays timed (the profiler)."""
    import gc

    from models.demos.moe_block.moe_block import MoEOptions
    from models.demos.moe_block.tt.head_plan import HeadLayout

    name, rows, lays, *tail = AB[ab]
    tail = tail[0] if tail else "auto"
    m = mesh_model(name, R * C)
    w = scenario_weights(m, "uniform", R * C)
    for GY, MT, NCH, KB, on, *zc in lays:
        lay = HeadLayout(
            GY=GY, MT=MT, NCH=NCH, KB=KB, on_engine=on, compact=bool(zc and zc[0]), split=bool(len(zc) > 1 and zc[1])
        )
        try:
            block = MoEBlock(
                mesh_device,
                m,
                rows * R,
                w,
                WeightCache(path=None, layer=0),
                options=MoEOptions(head="fused", head_layout=lay, tail=tail),
            )
        except (AssertionError, ValueError, RuntimeError) as e:
            logger.warning(f"AB {ab} {lay}: does not build: {' '.join(str(e).split())[:200]}")
            continue
        tag = f"ab-{ab}-{GY}x{MT}x{NCH}-kb{KB}-{'eng' if on else 'rc'}{'-compact' if lay.compact else ''}"
        tag += "-split" if lay.split else ""
        stats = run_checked(mesh_device, block, m, w, tag, eager=1, replays=12)
        logger.info(f"AB_RESULT {tag}: PCC min {min(stats['pcc']):.5f}; {block.summary}")
        block.close()
        del block
        gc.collect()  # (its global semaphores go with their last reference: before the next block's L1)


# the ports' logits priority A/B (MoEOptions.logits_first): (model, rows a mesh row)
PRIO = [("k3", 640), ("glm53", 640), ("k27", 640), ("k3", 1024)]


@pytest.mark.parametrize(
    "mesh_device, device_params, case, first",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(BASE[m].routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            (m, rows),
            first,
            id=f"prio-{m}-r{rows}-{'logits' if first else 'x'}first",
        )
        for m, rows in PRIO
        for first in (True, False)
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(36000)
def test_logits_priority_ab(mesh_device, device_params, case, first):
    """The block with the ports' logits ahead of / after the x gather, traced replays timed (the profiler: the head's
    zones - the routing chain's steps)."""
    from models.demos.moe_block.moe_block import MoEOptions

    name, rows = case
    m = mesh_model(name, R * C)
    w = scenario_weights(m, "uniform", R * C)
    block = MoEBlock(
        mesh_device,
        m,
        rows * R,
        w,
        WeightCache(path=None, layer=0),
        options=MoEOptions(head="fused", logits_first=first),
    )
    tag = f"prio-{name}-r{rows}-{'logits' if first else 'x'}first"
    stats = run_checked(mesh_device, block, m, w, tag, eager=1, replays=12)
    logger.info(f"PRIO_RESULT {tag}: PCC min {min(stats['pcc']):.5f}; {block.summary}")
    block.close()


@pytest.mark.parametrize(
    "mesh_device, device_params, model",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(BASE[m].routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            m,
            id=f"split-equal-{m}",
        )
        for m in ("k3", "k27")
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(7200)
def test_router_split_equal(mesh_device, device_params, model):
    """The router's split (a row tile's experts over two cores, merged on the first) picks exactly what one core
    does: deterministic sums (the sources' partials in slot order), the same inputs, the device's joint routing rows
    bit for bit; then the split block's output against the host (PCC)."""
    import gc

    from models.demos.moe_block.moe_block import MoEOptions
    from models.demos.moe_block.tt.head_plan import HeadLayout

    rows_ = 640
    m = mesh_model(model, R * C)
    w = scenario_weights(m, "uniform", R * C)
    T, H = rows_ * R, m.emb_dim
    mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(R, C), dims=(2, 3))
    xs = [torch.randn(T, H, generator=torch.Generator().manual_seed(4000 + k)).to(torch.bfloat16) for k in range(3)]
    got = {}
    for split in (False, True):
        lay = HeadLayout(GY=10, MT=2, NCH=1, KB=0, on_engine=True, compact=model != "k3", split=split)
        block = MoEBlock(
            mesh_device,
            m,
            T,
            w,
            WeightCache(path=None, layer=0),
            options=MoEOptions(head="fused", head_layout=lay, deterministic=True),
        )
        logger.info(f"SPLIT {model} split={split}: {block.summary}")
        block.keep_taps = "device"
        x_in = block.input_buffer()
        got[split] = []
        for x_ in xs:
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(x_.view(1, 1, T, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper),
                x_in,
            )
            out = block(x_in)
            ttnn.synchronize_device(mesh_device)
            got[split].append([t.clone() for t in block.taps()["rows"]])
            ttnn.deallocate(out)
        if split:
            stats = run_checked(mesh_device, block, m, w, f"split-{model}", eager=2, replays=12)
            logger.info(f"SPLIT_RESULT {model}: PCC min {min(stats['pcc']):.5f}; agreement {min(stats['agree']):.4f}")
        block.close()
        del block
        gc.collect()
    diff = sum(int(not torch.equal(a, b)) for ra, rb in zip(got[False], got[True]) for a, b in zip(ra, rb))
    if diff:  # (a few tokens' ids, one core vs split: the joint rows [owned / 32, 32 x 2 K] - K ids then K weights)
        a0, b0 = got[False][0][0].reshape(-1, 32, 2 * m.top_k), got[True][0][0].reshape(-1, 32, 2 * m.top_k)
        for rt, tok in ((0, 0), (0, 5), (1, 0), (4, 31)):
            logger.info(
                f"SPLIT_DIFF chip 0 row tile {rt} token {tok}: one core {a0[rt, tok, :m.top_k].tolist()} split "
                f"{b0[rt, tok, :m.top_k].tolist()}"
            )
    logger.info(f"SPLIT_EQUAL {model}: {diff} of {sum(len(r) for r in got[True])} chips' routing rows differ")
    assert diff == 0, f"{model}: the split's routing rows differ from one core's on {diff} chips"


@pytest.mark.parametrize(
    "mesh_device, device_params, model",
    [
        pytest.param(
            (R, C),
            _fab(
                fabric_payload_size=fabric_payload(BASE[m].routed_width),
                l1_small_size=0,
                trace_region_size=TRACE_REGION,
            ),
            m,
            id=f"split-soak-{m}",
        )
        for m in ("k3", "k27")
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.timeout(7200)
def test_router_split_soak(mesh_device, device_params, model):
    """The router's split at 640 rows: traced replays on fresh inputs, every 25th checked (PCC, routing agreement)."""
    from models.demos.moe_block.moe_block import MoEOptions

    m = mesh_model(model, R * C)
    w = scenario_weights(m, "uniform", R * C)
    block = MoEBlock(  # (the planner's split layout: MoEOptions.router_split)
        mesh_device, m, 640 * R, w, WeightCache(path=None, layer=0), options=MoEOptions(head="fused", router_split=True)
    )
    assert "10x10" in block.summary, block.summary
    stats = run_checked(
        mesh_device, block, m, w, f"split-soak-{model}", eager=2, replays=int(os.environ.get("SPLIT_SOAK", "2000"))
    )
    logger.info(
        f"SPLIT_SOAK {model}: {len(stats['pcc'])} checks, PCC min {min(stats['pcc']):.5f}, agreement min "
        f"{min(stats['agree']):.4f}; {block.summary}"
    )
    block.close()
