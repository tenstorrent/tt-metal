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
    Result,
    Status,
    check_smi,
    classify_loop,
    classify_pytest,
    collected_node_ids,
    didt_bad_devices,
    fabric_results,
    progress,
)

SCRIPTS = Path(__file__).resolve().parent
TESTS = "models/demos/deepseek_v3_d_p/tests"
HANG_SECS = 120

MODEL_ENV = {"kimi_k2_7": "KIMI_K2_7", "glm53": "GLM5_3"}
MODEL_LABEL = {"kimi_k2_7": "K2.7", "glm53": "GLM5.3"}
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
    now: str = ""


@dataclass
class Ctx:
    repo: Path
    run_dir: Path
    mode_co: bool
    tick: Callable[[], None] = lambda: None
    blocked_models: dict[str, str] = field(default_factory=dict)


@dataclass
class Step:
    key: str
    title: str
    group: str
    model: str | None
    blocking: bool
    run: Callable[[Ctx, State], Result]
    collect: Callable[[Ctx], tuple[list[str], dict[str, str], int]] | None = None
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


def triage_env(repo: Path, csv: Path) -> dict[str, str]:
    """Same hang handling as common.sh triage_env: triage on dispatch timeout, then kill the hung pytest."""
    py = repo / "python_env/bin/python"
    return {
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": str(HANG_SECS),
        "TT_METAL_DISPATCH_TIMEOUT_COMMAND_TO_EXECUTE": f"{py} {repo}/tools/tt-triage.py --disable-progress "
        "--run=dump_op_mesh --run=check_dram_health --run=dump_callstacks --llm-output "
        f"--llm-output-path={csv}; kill -9 $PPID",
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


def collected_ids(ctx: Ctx, args: list[str], env: dict[str, str], log: Path) -> list[str]:
    sh(ctx, "pytest --collect-only -q " + " ".join(f"'{a}'" for a in args), log, env, timeout=600)
    return collected_node_ids(log.read_text(errors="replace"))


def run_collect(ctx: Ctx, key: str, args: list[str], env: dict[str, str], expected: int) -> Result:
    ids = collected_ids(ctx, args, env, step_dir(ctx, key) / "collect.log")
    if len(ids) != expected:
        return Result(Status.FAIL, f"{len(ids)} node ids collected, expected {expected}", ids)
    return Result(Status.PASS, f"collected: {ids[0]}" if expected == 1 else f"collected {expected} node ids", ids)


RESET_CMD = "tt-smi -glx_reset_auto"


def reset(ctx: Ctx, log: Path) -> Result | None:
    if sh(ctx, RESET_CMD, log, timeout=600) != 0:
        return Result(Status.FAIL, f"reset failed: {RESET_CMD}")
    return None


def pytest_step(
    key: str,
    title: str,
    group: str,
    model: str | None,
    args: list[str],
    expected: int,
    details: Callable[[str], list[str]] = lambda log: [],
) -> Step:
    def env_for(ctx: Ctx) -> dict[str, str]:
        return common_sh(ctx.repo, model)[0] if model else {}

    def collect(ctx: Ctx) -> tuple[list[str], dict[str, str], int]:
        return args, env_for(ctx), expected

    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return run_collect(ctx, key, args, env_for(ctx), expected)
        d = step_dir(ctx, key)
        if model and (failed := reset(ctx, d / "reset.log")):
            return failed
        log, csv = d / "log_01", d / "crash_triage_01.csv"
        state.log = str(log)
        cmd = "pytest -vs -rA " + " ".join(f"'{a}'" for a in args)
        sh(ctx, cmd, log, {**env_for(ctx), **triage_env(ctx.repo, csv)})
        if csv.is_file() and csv.stat().st_size > 0:
            return Result(Status.HANG, f"triage {csv}")
        text = log.read_text(errors="replace")
        r = classify_pytest(text, expected)
        r.details = details(text)
        return r

    return Step(key, title, group, model, True, run, collect)


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

    def collect(ctx: Ctx) -> tuple[list[str], dict[str, str], int]:
        env, target = common_sh(ctx.repo, model, ids)
        return [target], env, 1

    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return run_collect(ctx, key, *collect(ctx))
        d = step_dir(ctx, key)
        env = {**ids, "MODEL": MODEL_ENV[model], "TRIAGE": "1", "PREFLIGHT": "0", "LOG_DIR": str(d)}

        def tick():
            state.loops = loop_results(d, loops)
            live = [i for i in range(1, loops + 1) if (d / f"log_{i:02d}").is_file()]
            if live:
                log = d / f"log_{live[-1]:02d}"
                text = log.read_text(errors="replace")
                state.log = str(log)
                if "TEST_DONE_EXIT=" not in text:
                    state.loops[live[-1] - 1] = Result(Status.RUN)
                state.now = f"{phase} {live[-1]}/{loops}  " + progress(text, iters, time.time(), log.stat().st_mtime)
            outer_tick()

        outer_tick, ctx.tick = ctx.tick, tick
        try:
            sh(ctx, f"bash {SCRIPTS}/stress.sh {d.name} {loops}", d / "stress.log", env)
        finally:
            ctx.tick = outer_tick
        state.loops = loop_results(d, loops)
        state.now = ""
        bad = next((r for r in state.loops if r.status != Status.PASS), None)
        return bad or Result(Status.PASS, f"{loops}/{loops} loops passed")

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
                    ctx.blocked_models.setdefault(m, f"{var} missing")
        if missing:
            return Result(Status.FAIL, f"{len(missing)} repo paths missing", details)
        return Result(Status.PASS if found == total else Status.WARN, f"{found}/{total} exist", details)

    return Step("paths", "paths", "gate", None, True, run)


def collect_step(steps: list[Step]) -> Step:
    def run(ctx: Ctx, state: State) -> Result:
        targets = [s for s in steps if s.collect and s.model not in ctx.blocked_models]
        if ctx.mode_co:
            return Result(Status.PASS, f"{len(targets)} node ids, collected per step below")
        bad = []
        for s in targets:
            r = run_collect(ctx, s.key, *s.collect(ctx))
            if r.status != Status.PASS:
                bad.append(f"{s.key}: {r.reason}")
        if bad:
            return Result(Status.FAIL, f"{len(bad)}/{len(targets)} node ids do not collect", bad)
        return Result(Status.PASS, f"{len(targets)}/{len(targets)} node ids")

    return Step("collect", "collect", "gate", None, True, run)


def reset_gate() -> Step:
    def run(ctx: Ctx, state: State) -> Result:
        if ctx.mode_co:
            return would_run(RESET_CMD)
        return reset(ctx, step_dir(ctx, "reset") / "log_01") or Result(Status.PASS)

    return Step("reset", "reset", "gate", None, True, run)


def smi_step() -> Step:
    cmd = "tt-smi -s --snapshot_no_tty"

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
    inner = pytest_step("fabric", "fabric", "gate", None, [test, "-k", "FABRIC_2D_TORUS_XY"], 1)

    def run(ctx: Ctx, state: State) -> Result:
        r = inner.run(ctx, state)
        if ctx.mode_co or r.status == Status.PASS:
            return r
        log = step_dir(ctx, "fabric") / "all_configs.log"
        sh(ctx, f"pytest -rA '{test}'", log)
        opens = fabric_results(log.read_text(errors="replace"))
        r.details = [f"{name} {'opens' if ok else 'FAILS'}" for name, ok in opens.items()]
        return r

    return Step("fabric", "fabric", "gate", None, True, run, inner.collect)


def memory_step() -> Step:
    return Step("memory", "memory", "gate", None, False, lambda ctx, state: Result(Status.SKIP, "not implemented"))


def didt_step() -> Step:
    args = [
        "tests/didt/test_minimal_matmul.py",
        "-k",
        "galaxy and bf16_HiFi2",
        "--didt-workload-iterations=1000",
        "--determinism-check-interval=50",
        "--timeout",
        "1200",
    ]

    def bad_devices(log: str) -> list[str]:
        bad = didt_bad_devices(log)
        return [f"device {d}: {k} non-deterministic outputs" for d, k in sorted(bad.items())]

    return pytest_step("didt", "didt", "gate", None, args, 1, bad_devices)


def build_steps(models: list[str]) -> list[Step]:
    blocks = {
        "kimi_k2_7": [
            pytest_step(
                "block:kimi_k2_7",
                "block K2.7",
                "block",
                "kimi_k2_7",
                [
                    f"{TESTS}/test_prefill_block_chunked.py::test_kimi_prefill_block_chunked",
                    "-k",
                    "torus-xy-8x4 and chunks2 and with_determinism and iter2 and not iter25 and not iter2000",
                ],
                1,
            ),
            pytest_step(
                "block-random:kimi_k2_7",
                "block-random K2.7",
                "block-random",
                "kimi_k2_7",
                [
                    f"{TESTS}/test_prefill_transformer.py::test_kimi_prefill_transformer",
                    "-k",
                    "torus-xy-8x4 and 5_layers and e384_device and 5k and not 25k and iter2 and not iter25 "
                    "and not iter2000 and smoke and with_determinism and non_balanced and right_pad "
                    "and json_prompts and random",
                ],
                1,
            ),
        ],
        "glm53": [
            pytest_step(
                "block:glm53",
                "block GLM5.3",
                "block",
                "glm53",
                [
                    f"{TESTS}/test_prefill_block.py::test_glm_prefill_block",
                    "-k",
                    "seq5120 and torus-xy-8x4 and glm53 and iter2 and with_determinism",
                ],
                2,
            )
        ],
    }
    model_steps = [s for m in models for s in blocks[m]]
    model_steps += [stress_step("smoke", m, SMOKE_LOOPS, SMOKE_ENV, 2) for m in models]
    model_steps += [stress_step("soak", m, SOAK_LOOPS, SOAK_ENV, SOAK_ITERS) for m in models]
    hardware = [reset_gate(), smi_step(), fabric_step(), memory_step(), didt_step()]
    return [paths_step(models), collect_step(hardware + model_steps)] + hardware + model_steps


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
        else:
            s.start, s.result = time.time(), Result(Status.RUN)
            on_update(step)
            try:
                s.result = step.run(ctx, s)
            except Exception as e:
                s.result = Result(Status.FAIL, f"{type(e).__name__}: {e}")
            s.end = time.time()
            if s.result.status in (Status.FAIL, Status.HANG):
                if step.group == "gate" and step.blocking:
                    gate_failed = f"{i} {step.title}"
                elif step.model:
                    ctx.blocked_models.setdefault(step.model, f"{step.title} failed")
        on_update(step)
