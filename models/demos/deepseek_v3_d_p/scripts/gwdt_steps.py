# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from gwdt_checks import (
    LogFollower,
    Result,
    Status,
    check_smi,
    classify_loop,
    classify_pytest,
    collected_node_ids,
    didt_bad_devices,
    dram_health,
    fabric_results,
    gddr_edc,
    gddr_result,
    missing_node_ids,
    mla_determinism_reps,
    read_tail,
    row_results,
    run_to_run_bad_devices,
    stress_result,
    sweep_bad_devices,
    sweep_skipped_windows,
    umd_metal_ids,
    write_ack_result,
)

SCRIPTS = Path(__file__).resolve().parent
TESTS = "models/demos/deepseek_v3_d_p/tests"
HANG_SECS = 120

MODEL_ENV = {"kimi_k2_7": "KIMI_K2_7", "glm53": "GLM5_3"}
MODEL_LABEL = {"kimi_k2_7": "K2.7", "glm53": "GLM5.3"}
MODEL_LAYERS = {"kimi_k2_7": "L61", "glm53": "L78"}
MODEL_PATH_VARS = {
    "kimi_k2_7": ["KIMI_K2_7_HF_MODEL", "TT_KIMI_PREFILL_TTNN_CACHE", "PREFILL_TRACE_DIR"],
    "glm53": ["GLM53_HF_MODEL", "TT_GLM53_PREFILL_TTNN_CACHE", "PREFILL_TRACE_DIR"],
}
SMOKE_ENV = {"ITERS_ID": "two_iters", "CHUNKS_ID": "chunks2", "LAYERS_ID": "L10"}
SOAK_ENV = {"ITERS_ID": "iters600", "CHUNKS_ID": "chunks20"}
SMOKE_LOOPS, SOAK_LOOPS = 1, 3
SOAK_ITERS = 600


@dataclass
class State:
    result: Result = field(default_factory=lambda: Result(Status.PENDING))
    log: str = ""
    start: float | None = None
    end: float | None = None
    loops: list[Result] = field(default_factory=list)
    rows: list[Result] = field(default_factory=list)
    now: str = ""
    attempt: int = 1
    env: dict[str, str] = field(default_factory=dict)
    opts: list[str] = field(default_factory=list)
    fallbacks: list["State"] = field(default_factory=list)


@dataclass
class Ctx:
    repo: Path
    run_dir: Path
    mode_co: bool
    tick: Callable[[], None] = lambda: None
    blocked_models: dict[str, str] = field(default_factory=dict)
    missing_paths: dict[str, list[str]] = field(default_factory=dict)
    collected: list[str] | None = None


@dataclass
class Step:
    key: str
    title: str
    group: str
    model: str | None
    blocking: bool
    run: Callable[[Ctx, State], Result]
    collect: Callable[[Ctx], list[str]] | None = None
    needs: list[str] = field(default_factory=list)
    state: State = field(default_factory=State)


def base_env(repo: Path) -> dict[str, str]:
    venv = repo / "python_env"
    return {
        **os.environ,
        "TT_METAL_HOME": str(repo),
        "PYTHONPATH": str(repo),
        "VIRTUAL_ENV": str(venv),
        "PATH": f"{venv / 'bin'}:{os.environ.get('PATH', '')}",
        "DS_PERF_IGNORE_POWER": "1",
        # Pinned host-memory H2D makes weight load ~10x slower; 0 keeps the copy path (#52893).
        "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0",
    }


def common_sh(repo: Path, model: str, extra: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    """(model ENV_VARS as a dict, PYTEST_TARGET) as common.sh resolves them for this env."""
    env = {**base_env(repo), "MODEL": MODEL_ENV[model], **(extra or {})}
    out = subprocess.run(
        ["bash", "-c", f'source {SCRIPTS}/common.sh x >/dev/null && printf "%s\\n%s\\n" "$ENV_VARS" "$PYTEST_TARGET"'],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    return dict(kv.split("=", 1) for kv in out[0].split()), out[1]


# tt-triage scripts run on a hang; keep in sync with HANG_TRIAGE in common.sh. Besides where each core is stuck
# (op mesh, callstacks), the op window shows the ops just before and after the hung one, check_noc_status catches
# posted-write/ack counters that never drained, and check_arc / check_eth_status cover firmware and links.
HANG_TRIAGE = (
    "--run=dump_op_mesh --run=dump_running_operations --run=dump_op_window --op-window=20 "
    "--run=check_dram_health --run=dump_callstacks --run=check_noc_status --run=check_arc --run=check_eth_status"
)


def triage_env(repo: Path, csv: Path) -> dict[str, str]:
    """Same hang handling as common.sh triage_env: triage on dispatch timeout, then kill the hung pytest."""
    py = repo / "python_env/bin/python"
    return {
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": str(HANG_SECS),
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": f"{py} {repo}/tools/tt-triage.py --disable-progress "
        f"{HANG_TRIAGE} --llm-output --llm-output-path={csv}; kill -9 $PPID",
    }


def sh(ctx: Ctx, cmd: str, log: Path, env: dict[str, str] | None = None, timeout: float | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout if timeout else None
    with open(log, "w") as f:
        p = subprocess.Popen(
            ["bash", "-c", cmd],
            cwd=ctx.repo,
            env={**base_env(ctx.repo), **(env or {})},
            stdout=f,
            stderr=subprocess.STDOUT,
        )
        while p.poll() is None:
            if deadline and time.time() > deadline:
                p.kill()
                p.wait()
                f.write(f"\nTIMEOUT after {timeout}s\n")
                return 124
            ctx.tick()
            time.sleep(1)
    return p.returncode


def step_dir(ctx: Ctx, key: str) -> Path:
    return ctx.run_dir / key.replace(":", "/")


def would_run(cmd: str) -> Result:
    return Result(Status.PASS, f"would run: {cmd}")


# Each pytest start costs ~30 s here (venv imports over NFS, cluster open/close in the DeepSeek conftest), and
# pytest-benchmark adds a `git describe` over thousands of loose refs; so all steps collect in one call, without it.
COLLECT_CMD = "pytest --collect-only -q -p no:benchmark "


def collected_ids(ctx: Ctx, ids: list[str], log: Path) -> list[str]:
    sh(ctx, COLLECT_CMD + " ".join(f"'{i}'" for i in ids), log, timeout=600)
    return collected_node_ids(log.read_text(errors="replace"))


def run_collect(ctx: Ctx, key: str, ids: list[str]) -> Result:
    """--co result for one step: from the collect gate's single pytest call when it ran, else its own."""
    log = step_dir(ctx, "collect") / "collect.log" if ctx.collected is not None else step_dir(ctx, key) / "collect.log"
    collected = ctx.collected if ctx.collected is not None else collected_ids(ctx, ids, log)
    if missing := missing_node_ids(ids, collected):
        return Result(Status.FAIL, f"{len(missing)}/{len(ids)} node ids do not collect, log {log}", missing)
    return Result(
        Status.PASS, f"collected: {ids[0].split('::')[-1]}" if len(ids) == 1 else f"collected {len(ids)} node ids"
    )


RESET_CMD = "tt-smi -glx_reset_auto"
SMI_CMD = "tt-smi -s --snapshot_no_tty"


def attempt_path(path: Path, attempt: int) -> Path:
    """A fallback's files sit next to the first run's: reset.log, reset_02.log."""
    return path if attempt == 1 else path.with_stem(f"{path.stem}_{attempt:02d}")


def save_smi(ctx: Ctx, d: Path, attempt: int = 1) -> None:
    """Device telemetry right after a device step, before the next reset, for post-mortem."""
    sh(ctx, SMI_CMD, attempt_path(d / "tt-smi.json", attempt), timeout=120)


# tt-smi's own "auto retry" does not cover a chip that misses firmware init after the IPMI cycle (seen once: GDDR
# training timeout on one chip, clean on the next reset), so a reset gets a few tries before the step fails.
RESET_TRIES = 3


def retry_path(log: Path, n: int) -> Path:
    """Retries of one reset sit next to it: reset.log, reset.retry2.log; log_01, log_01.retry2."""
    return log if n == 1 else log.with_name(f"{log.stem}.retry{n}{log.suffix}")


def pcie_buses() -> dict[str, int]:
    """PCIe bus id (0000:c7:00.0) -> /dev/tenstorrent index, from sysfs."""
    out = {}
    for dev in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            out[(dev / "device").resolve().name] = int(dev.name.split("!")[1])
        except (OSError, ValueError):
            continue
    return out


def check_gddr(ctx: Ctx, step: Step) -> None:
    """Every tt-smi snapshot a step saved: corrected GDDR EDC warns, uncorrected fails, reported per metal device,
    GDDR instance and bank. Metal ids come from UMD's PCIe-id line in the step's own logs."""
    # mtimes come from the kernel's coarse clock and can trail time.time() by a tick; the slack only needs to stay
    # under one step's runtime.
    d, since = step_dir(ctx, step.key), (step.state.start or 0) - 2
    snaps = sorted(p for p in d.rglob("tt-smi*.json") if p.stat().st_mtime >= since)
    found = []
    for snap in snaps:
        try:
            found += [(str(snap.relative_to(d)), e) for e in gddr_edc(json.loads(snap.read_text()))]
        except (OSError, ValueError):
            continue
    if not found:
        return
    metal_by_pcie: dict[int, int] = {}
    # The step's own logs first; mcast-ack's per-chip runs do not print the line, so then any log in the run.
    for log in [*sorted(d.rglob("log_*")), *sorted(ctx.run_dir.rglob("log_*"))]:
        with open(log, errors="replace") as f:
            if metal_by_pcie := umd_metal_ids(f.read(1 << 16)):
                break
    metal_by_bus = {bus: metal_by_pcie[i] for bus, i in pcie_buses().items() if i in metal_by_pcie}
    step.state.result = gddr_result(step.state.result, found, metal_by_bus)


def reset(ctx: Ctx, log: Path, state: State) -> Result | None:
    for n in range(1, RESET_TRIES + 1):
        if sh(ctx, RESET_CMD, retry_path(log, n), timeout=600) == 0:
            return None
    state.log = str(retry_path(log, RESET_TRIES))
    return Result(Status.FAIL, f"reset failed {RESET_TRIES}x: {RESET_CMD}, first log {log}")


def pytest_step(
    key: str,
    title: str,
    group: str,
    model: str | None,
    ids: list[str],
    opts: list[str] = [],
    details: Callable[[str], list[str]] = lambda log: [],
    env: dict[str, str] = {},
) -> Step:
    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return run_collect(ctx, key, ids)
        d, n = step_dir(ctx, key), state.attempt
        if group != "gate" and (failed := reset(ctx, attempt_path(d / "reset.log", n), state)):
            return failed
        log, csv = d / f"log_{n:02d}", d / f"crash_triage_{n:02d}.csv"
        state.log = str(log)
        cmd = "pytest -vs -rA " + " ".join(f"'{a}'" for a in ids + opts + state.opts)

        def tick():
            if log.is_file():
                state.rows = row_results(read_tail(log), ids)
            outer_tick()

        outer_tick, ctx.tick = ctx.tick, tick
        try:
            model_env = common_sh(ctx.repo, model)[0] if model else {}
            sh(ctx, cmd, log, {**model_env, **env, **state.env, **triage_env(ctx.repo, csv)})
        finally:
            ctx.tick = outer_tick
        save_smi(ctx, d, n)
        state.rows = row_results(read_tail(log), ids)
        if csv.is_file() and csv.stat().st_size > 0:
            return Result(Status.HANG, f"triage {csv}")
        text = log.read_text(errors="replace")
        r = classify_pytest(text, len(ids))
        r.details = details(text)
        return r

    return Step(key, title, group, model, True, run, lambda ctx: ids)


def loop_results(d: Path, loops: int) -> list[Result]:
    return [
        classify_loop(d / f"log_{i:02d}", d / f"crash_triage_{i:02d}.csv")
        if (d / f"log_{i:02d}").is_file()
        else Result(Status.PENDING)
        for i in range(1, loops + 1)
    ]


def stress_step(phase: str, model: str, loops: int, ids: dict[str, str], iters: int) -> Step:
    key = f"{phase}:{model}"
    ids = {**ids, "TRACE_ID": "traced", "DET_ID": "with_determinism"}

    def collect(ctx: Ctx) -> list[str]:
        return [common_sh(ctx.repo, model, ids)[1]]

    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return run_collect(ctx, key, collect(ctx))
        d = step_dir(ctx, key)
        env = {**ids, "MODEL": MODEL_ENV[model], "TRIAGE": "1", "PREFLIGHT": "0", "LOG_DIR": str(d)}

        followers: dict[int, LogFollower] = {}
        finished: dict[int, Result] = {}

        def tick():
            for i in range(1, loops + 1):
                log = d / f"log_{i:02d}"
                if i in finished or not log.is_file():
                    continue
                if "TEST_DONE_EXIT=" in read_tail(log, 4096):
                    finished[i] = classify_loop(log, d / f"crash_triage_{i:02d}.csv")
                    continue
                f = followers.setdefault(i, LogFollower(log))
                f.update()
                state.log = str(log)
                state.now = f"{phase} {i}/{loops}  " + f.line(iters, time.time())
            state.loops = [
                finished.get(i, Result(Status.RUN) if i in followers else Result(Status.PENDING))
                for i in range(1, loops + 1)
            ]
            outer_tick()

        outer_tick, ctx.tick = ctx.tick, tick
        try:
            sh(ctx, f"bash {SCRIPTS}/stress.sh {d.name} {loops}", d / "stress.log", env)
        finally:
            ctx.tick = outer_tick
        save_smi(ctx, d)
        state.loops = loop_results(d, loops)
        state.now = ""
        return stress_result(state.loops, d / "stress.log")

    return Step(key, f"{phase} {MODEL_LABEL[model]}", phase, model, True, run, collect)


def paths_step(models: list[str]) -> Step:
    fixed = [
        "python_env/bin/activate",
        "tests/didt/test_minimal_matmul.py",
        f"{TESTS}/utils/test_fabric_availability.py",
    ]

    def run(ctx: Ctx, state: State) -> Result:
        missing = [p for p in fixed if not (ctx.repo / p).exists()]
        details = [f"{p} MISSING" for p in missing]
        found = total = 0
        for m in models:
            env = common_sh(ctx.repo, m)[0]
            details.append(MODEL_LABEL[m])
            for var in MODEL_PATH_VARS[m]:
                total += 1
                if Path(env[var]).exists():
                    found += 1
                    details.append(f"  {var:<27} = {env[var]}")
                else:
                    details.append(f"  {var:<27} = {env[var]}   MISSING")
                    ctx.missing_paths.setdefault(m, []).append(var)
        if missing:
            return Result(Status.FAIL, f"{len(missing)} repo paths missing", details)
        return Result(Status.PASS if found == total else Status.WARN, f"{found}/{total} exist", details)

    return Step("paths", "paths", "gate", None, True, run)


def collect_step(steps: list[Step]) -> Step:
    def run(ctx: Ctx, state: State) -> Result:
        targets = [s for s in steps if s.collect and not missing_needs(s, ctx)]
        wanted = {s.key: s.collect(ctx) for s in targets}
        log = step_dir(ctx, "collect") / "collect.log"
        state.log = str(log)
        ctx.collected = collected_ids(ctx, [i for ids in wanted.values() for i in ids], log)
        missing = {key: m for key, ids in wanted.items() if (m := missing_node_ids(ids, ctx.collected))}
        if missing:
            # One unknown id aborts the whole call and lists nothing, so find the culprits one step at a time.
            ctx.collected = None
            missing = {
                key: m
                for key in missing
                if (
                    m := missing_node_ids(
                        wanted[key], collected_ids(ctx, wanted[key], step_dir(ctx, key) / "collect.log")
                    )
                )
            }
        bad = [f"{key}: {', '.join(m)}" for key, m in missing.items()]
        if bad:
            return Result(Status.FAIL, f"{len(bad)}/{len(targets)} steps have node ids that do not collect", bad)
        return Result(Status.PASS, f"{len(targets)}/{len(targets)} steps, {sum(map(len, wanted.values()))} node ids")

    return Step("collect", "collect", "gate", None, True, run)


def reset_gate() -> Step:
    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return would_run(RESET_CMD)
        return reset(ctx, step_dir(ctx, "reset") / "log_01", state) or Result(Status.PASS)

    return Step("reset", "reset", "gate", None, True, run)


def smi_step() -> Step:
    cmd = SMI_CMD

    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return would_run(cmd)
        out = step_dir(ctx, "tt-smi") / "tt-smi.json"
        state.log = str(out)
        if sh(ctx, cmd, out, timeout=120) != 0:
            return Result(Status.FAIL, f"{cmd} failed")
        try:
            return check_smi(json.loads(out.read_text()))
        except json.JSONDecodeError:
            return Result(Status.FAIL, "tt-smi output is not JSON")

    return Step("tt-smi", "tt-smi", "gate", None, True, run)


def fabric_step() -> Step:
    test = f"{TESTS}/utils/test_fabric_availability.py"
    inner = pytest_step("fabric", "fabric", "gate", None, [f"{test}::test_fabric_opens[FABRIC_2D_TORUS_XY]"])

    def run(ctx: Ctx, state: State) -> Result:
        r = inner.run(ctx, state)
        if ctx.mode_co or r.status in (Status.PASS, Status.HANG):
            return r
        log = step_dir(ctx, "fabric") / "all_configs.log"
        sh(ctx, f"pytest -rA '{test}'", log)
        opens = fabric_results(log.read_text(errors="replace"))
        r.details = [f"{name} {'opens' if ok else 'FAILS'}" for name, ok in opens.items()]
        return r

    return Step("fabric", "fabric", "gate", None, True, run, inner.collect)


def memory_step() -> Step:
    return Step("memory", "memory", "gate", None, False, lambda ctx, state: Result(Status.SKIP, "not implemented"))


def mla_step(key: str, reps: int) -> Step:
    ids = [
        f"{TESTS}/test_mla.py::test_mla_chunked_prefill"
        "[blackhole-sp_only-with_determinism-scalar-k2_7-deep-200k+5k-func-torus-xy-8x4]"
    ]
    inner = pytest_step(key, key, "gate", None, ids, [f"--mla-determinism-reps={reps}"])

    def run(ctx: Ctx, state: State) -> Result:
        r = inner.run(ctx, state)
        if ctx.mode_co or r.status == Status.HANG:
            return r
        if r.status == Status.PASS and mla_determinism_reps(Path(state.log).read_text(errors="replace")) != reps:
            r = Result(Status.FAIL, f"passed without the {reps} reps determinism line")
        d = step_dir(ctx, key)
        sh(
            ctx,
            "python tools/tt-triage.py --run=check_dram_health --dev=all --disable-progress "
            f"--triage-summary-path={d / 'dram.txt'} --sqlite-output-path={d / 'dram.sqlite'}",
            d / "dram.log",
            timeout=600,
        )
        dram = dram_health(d / "dram.sqlite")
        return Result(
            r.status if r.status != Status.PASS else dram.status, f"{r.reason}; {dram.reason}", r.details + dram.details
        )

    return Step(key, key, "gate", None, True, run, inner.collect)


def nd_outputs(log: str) -> list[str]:
    return [f"device {d}: {k} non-deterministic outputs" for d, k in sorted(didt_bad_devices(log).items())]


def run_to_run_cores(log: str) -> list[str]:
    return [
        f"device {d}: " + "; ".join(f"{where} {n} elements" for where, n in cores.items())
        for d, cores in sorted(run_to_run_bad_devices(log).items())
    ]


def sweep_elements(log: str) -> list[str]:
    return [f"device {d}: {n} elements differ run-to-run" for d, n in sorted(sweep_bad_devices(log).items())]


def matmul_step(
    name: str, ids: list[str], opts: list[str], details: Callable[[str], list[str]], env: dict[str, str] = {}
) -> Step:
    """A failure here blocks nothing: every matmul run and every model still runs, each after its own reset."""
    return pytest_step(f"matmul:{name}", f"matmul {name}", "matmul", None, ids, opts, details, env)


MATMUL_ITERS = 1000
RUN_TO_RUN_ITERS = 10
SWEEP_ITERS = 10
WRITE_ACK_ITERS = 10
SWEEP_GRID = "11x10"
SINGLE_GALAXY_MGD = "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto"


def core_sweep_step() -> Step:
    """One TT_SWEEP_GRID window at (0,0), GXxGY: 11x10 is the 110-core worker budget of the run-to-run test."""
    ids = ["tests/didt/scratch_core_sweep.py::test_core_sweep[blackhole-galaxy]"]
    env = {"TT_SWEEP_MODE": "grid", "TT_SWEEP_GRID": SWEEP_GRID, "TT_SWEEP_ITERS": str(SWEEP_ITERS)}
    inner = matmul_step("core-sweep", ids, ["-p", "no:randomly", "--timeout=0"], sweep_elements, env)

    def run(ctx: Ctx, state: State) -> Result:
        r = inner.run(ctx, state)
        if ctx.mode_co or r.status != Status.PASS:
            return r
        # The sweep catches a window's exception, counts it as skipped and still passes.
        skipped = sweep_skipped_windows(Path(state.log).read_text(errors="replace"))
        if skipped != 0:
            return Result(Status.FAIL, f"{SWEEP_GRID} window did not run, skipped windows: {skipped}")
        return r

    return Step(inner.key, inner.title, inner.group, None, True, run, inner.collect)


def write_ack_step(workload: str = "wkv_b1") -> Step:
    """One process per chip, each with its own cold cache and dispatch watchdog; logs land in the step dir via
    TMPDIR. workload wkv_b1 is the MLA projection the screen was written for (matmul:mcast-ack); duty is the
    duty-cycle-bursty-5 op alone on each chip (matmul:mcast-ack-duty), which hung 81:00.0 (T4:N1) and c2:00.0
    (T3:N2) on bh-glx-803 in 3 of 5 resets when wkv_b1 passed them."""
    key = "matmul:mcast-ack" if workload == "wkv_b1" else f"matmul:mcast-ack-{workload}"
    script = "tests/didt/test_mm_mcast_write_ack_failure.py"
    cmd = f"python {script}"
    test = {"wkv_b1": "test_mm_mcast_write_ack_failure_scan", "duty": "test_mm_mcast_write_ack_failure_duty"}[workload]
    ids = [f"{script}::{test}[blackhole-1x1]"]

    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return run_collect(ctx, key, ids)
        d, n = step_dir(ctx, key), state.attempt
        if failed := reset(ctx, attempt_path(d / "reset.log", n), state):
            return failed
        log = d / f"log_{n:02d}"
        state.log = str(log)
        # The script wipes its work dir under TMPDIR, so a rerun gets its own.
        tmp = d if n == 1 else d / f"tmp_{n:02d}"
        env = {"TMPDIR": str(tmp), "MATMUL_ITERS": str(WRITE_ACK_ITERS), "MATMUL_WORKLOAD": workload, **state.env}
        sh(ctx, cmd, log, env, timeout=1800)
        save_smi(ctx, d, n)
        return write_ack_result(log.read_text(errors="replace"))

    return Step(key, f"matmul {key.removeprefix('matmul:')}", "matmul", None, True, run, lambda ctx: ids)


def matmul_steps() -> list[Step]:
    return [
        matmul_step(
            "minimal",
            ["tests/didt/test_minimal_matmul.py::test_minimal_matmul[blackhole-bf16_HiFi2-galaxy]"],
            [f"--didt-workload-iterations={MATMUL_ITERS}", "--determinism-check-interval=50", "--timeout=0"],
            nd_outputs,
        ),
        matmul_step(
            "ff1",
            ["tests/didt/test_ff1_matmul.py::test_ff1_matmul[blackhole-galaxy-without_gelu]"],
            [f"--didt-workload-iterations={MATMUL_ITERS}", "--determinism-check-interval=50", "--timeout=0"],
            nd_outputs,
        ),
        # Each iteration is 1000 x (matmul + 5 cos) and one more matmul; only that last matmul is compared.
        matmul_step(
            "duty-cycle-bursty-5",
            ["tests/didt/test_duty_cycle.py::test_duty_cycle[blackhole-duty-5-rep-1000x-all]"],
            ["--didt-workload-iterations=10", "--determinism-check-interval=1", "--timeout", "1800"],
            nd_outputs,
            {"TT_MESH_GRAPH_DESC_PATH": SINGLE_GALAXY_MGD},
        ),
        matmul_step(
            "run-to-run",
            ["tests/didt/test_matmul_run_to_run_determinism.py::test_matmul_run_to_run_determinism[blackhole-galaxy]"],
            ["-p", "no:randomly", "--timeout=0"],
            run_to_run_cores,
            {"TT_DET_SWEEP_ITERS": str(RUN_TO_RUN_ITERS)},
        ),
        core_sweep_step(),
        write_ack_step(),
        write_ack_step("duty"),
    ]


THROTTLE = ({"TT_MM_THROTTLE_PERF": "5"}, [])
# Unthrottled, so a pass on 110 cores points at the co-active core count.
CORE_CAP = ({}, ["--grid-size", SWEEP_GRID])


def with_fallbacks(step: Step, fallbacks: list[tuple[dict[str, str], list[str]]]) -> Step:
    """A failed or hung ND run runs every fallback in order, each after its own reset, so each factor gets its own
    result. THROTTLE throttles matmul compute to 33%. The row keeps the first result; each fallback is a sub-row."""

    def run(ctx: Ctx, state: State) -> Result:
        r = step.run(ctx, state)
        if ctx.mode_co or r.status not in (Status.FAIL, Status.HANG):
            return r
        for n, (env, opts) in enumerate(fallbacks, 2):
            f = State(Result(Status.RUN), start=time.time(), attempt=n, env=env, opts=opts)
            state.fallbacks.append(f)
            try:
                f.result = step.run(ctx, f)
            except Exception as e:
                f.result = Result(Status.FAIL, f"{type(e).__name__}: {e}")
            f.end = time.time()
        return r

    return Step(step.key, step.title, step.group, step.model, step.blocking, run, step.collect)


def build_steps(models: list[str]) -> list[Step]:
    blocks = {
        "kimi_k2_7": [
            pytest_step(
                "block:kimi_k2_7",
                "block K2.7",
                "block",
                "kimi_k2_7",
                [
                    f"{TESTS}/test_prefill_block_chunked.py::test_kimi_prefill_block_chunked"
                    "[blackhole-kimi_k2_7-torus-xy-8x4-moe-gate_host-chunks2-iter2-with_determinism]"
                ],
            ),
            pytest_step(
                "block-random:kimi_k2_7",
                "block-random K2.7",
                "block-random",
                "kimi_k2_7",
                [
                    f"{TESTS}/test_prefill_block_chunked.py::test_kimi_prefill_block_chunked_random"
                    f"[blackhole-kimi_k2_7-torus-xy-8x4-{layer}-chunks2-iter100]"
                    for layer in ("dense_L0", "moe_L1")
                ],
            ),
        ],
        "glm53": [
            pytest_step(
                "block:glm53",
                "block GLM5.3",
                "block",
                "glm53",
                [
                    f"{TESTS}/test_prefill_block.py::test_glm_prefill_block"
                    f"[blackhole-glm53-with_determinism-iter2-{layer}-seq5120-torus-xy-8x4]"
                    for layer in ("dense", "moe")
                ],
            ),
            pytest_step(
                "block-random:glm53",
                "block-random GLM5.3",
                "block-random",
                "glm53",
                [
                    f"{TESTS}/test_prefill_block_chunked.py::test_glm_prefill_block_chunked_random"
                    f"[blackhole-glm53-torus-xy-8x4-{layer}-chunks2-iter100]"
                    for layer in ("dense_L0", "moe_full_L6", "moe_shared_L3")
                ],
            ),
        ],
    }
    model_steps = [s for m in models for s in blocks[m]]
    model_steps += [stress_step("smoke", m, SMOKE_LOOPS, SMOKE_ENV, 2) for m in models]
    model_steps += [stress_step("soak", m, SOAK_LOOPS, SOAK_ENV, SOAK_ITERS) for m in models]
    for s in model_steps:
        s.needs = path_needs(s.group, s.model)
    hardware = [
        reset_gate(),
        smi_step(),
        fabric_step(),
        memory_step(),
        mla_step("mla-smoke", 50),
        mla_step("mla-soak", 2500),
    ]
    matmul = [
        with_fallbacks(s, [THROTTLE, CORE_CAP] if s.key == "matmul:minimal" else [THROTTLE]) for s in matmul_steps()
    ]
    return [paths_step(models), collect_step(hardware + matmul + model_steps)] + hardware + matmul + model_steps


def path_needs(group: str, model: str) -> list[str]:
    hf, ttnn_cache, _trace = MODEL_PATH_VARS[model]
    if group == "block-random":
        # Random weights; the K2.7 config ships in the repo and GLM5.3 builds its own.
        return []
    # Without a golden trace, block runs random input and smoke/soak run synthetic tokens.
    return [hf, ttnn_cache]


def missing_needs(step: Step, ctx: Ctx) -> list[str]:
    return [v for v in step.needs if v in ctx.missing_paths.get(step.model, [])]


def run_all(steps: list[Step], ctx: Ctx, skip: set[str], on_update: Callable[[Step], None]) -> None:
    gate_failed = None
    for i, step in enumerate(steps):
        s = step.state
        if step.key in skip:
            s.result = Result(Status.SKIP, "by user")
        elif gate_failed is not None:
            s.result = Result(Status.PENDING, f"not run: gate {gate_failed} failed")
        elif step.model in ctx.blocked_models:
            s.result = Result(Status.PENDING, f"not run: {ctx.blocked_models[step.model]}")
        elif missing := missing_needs(step, ctx):
            s.result = Result(Status.PENDING, f"not run: {', '.join(missing)} missing")
        else:
            s.start, s.result = time.time(), Result(Status.RUN)
            on_update(step)
            try:
                s.result = step.run(ctx, s)
            except Exception as e:
                s.result = Result(Status.FAIL, f"{type(e).__name__}: {e}")
            if not ctx.mode_co:
                try:
                    check_gddr(ctx, step)
                except Exception as e:
                    s.result.details.append(f"GDDR check skipped: {type(e).__name__}: {e}")
            s.end = time.time()
            if s.result.status in (Status.FAIL, Status.HANG):
                if step.group == "gate" and step.blocking:
                    gate_failed = f"{i} {step.title}"
                elif step.model:
                    ctx.blocked_models.setdefault(step.model, f"{step.title} failed")
        on_update(step)
