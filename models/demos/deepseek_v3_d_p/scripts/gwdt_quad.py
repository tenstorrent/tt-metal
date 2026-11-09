# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""gwdt --quad: the prefill-runner hardware screen on 1 or 4 Blackhole galaxies, with no model weights.

Each galaxy is one pipeline stage of the real prefill runner (tt-run, one rank per host). The stage body is a fake
model (adapters/quad_screen.py): ttnn.clone pass-through, or K2.7 layer 0 + layer 1 with random weights. The H2D
producer feeds the same request over and over; every stage checks its input and output per device and writes
results under the run dir, which gwdt_quad_verdict turns into a per-stage verdict naming a galaxy or a link.
"""

import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from gwdt_checks import Result, Status
from gwdt_quad_verdict import load_hashes, load_ranks, overall, stage_verdicts
from gwdt_steps import Ctx, State, Step, base_env, step_dir, would_run

TOPO = "models/demos/common/prefill/runners/topology_configuration"
MGDS = {
    (4, "2d_torus_xy"): f"{TOPO}/ci/sc4_mgd.textproto",
    (4, "2d"): f"{TOPO}/pipeline_prefill_4galaxy_connected_fabric2d_mesh_graph_descriptor.textproto",
    (1, "2d_torus_xy"): f"{TOPO}/ci/sc1_mgd.textproto",
    (1, "2d"): "tt_metal/fabric/mesh_graph_descriptors/single_bh_galaxy_mesh_graph_descriptor.textproto",
}
MANIFESTS = "models/demos/deepseek_v3_d_p/tt/runners/manifests"
CONFIG_JSON = "models/demos/deepseek_v3_d_p/reference/kimi_k2_7/config.json"
DESCRIPTORS = Path("/data/scaleout_configs/bh_glx_exabox")
TORUS, MESH = "2d_torus_xy", "2d"

SERVICE_ID = "quad_screen"
CHUNK_SIZE = 5120
CHUNKS = 2
MAX_SEQ_LEN = CHUNK_SIZE * CHUNKS
LAYERS_PER_STAGE = 2
UNBOUNDED_REQUESTS = 10**9
TCP_INTERFACE = os.environ.get("PREFILL_TCP_INTERFACE", "ens5f0np0")

PASSTHROUGH_REQUESTS = 3
SMOKE_S = 300
SETUP_TIMEOUT_S = 2700
STALL_S = 300
SHUTDOWN_S = 120
POLL_S = 5
RESET_TRIES = 3

_SETUP = re.compile(r"\[pp rank (\d+)\] setup complete, entering request loop")
_KMD_LOCK = re.compile(r"Waiting for KMD lock .* held by another handle")
KMD_LOCK_S = 300
# A run that never reached the hardware question; its fallback would meet the same wall.
ENV_FAULT = "environment: "


def mgd_path(n_hosts: int, fabric: str) -> str:
    try:
        return MGDS[(n_hosts, fabric)]
    except KeyError:
        raise ValueError(f"no mesh-graph descriptor for {n_hosts} host(s) on fabric {fabric}") from None


def _bash(repo: Path, exports: dict[str, str], exe: str) -> str:
    env = "".join(f" export {k}={v};" for k, v in exports.items())
    return f"cd {repo}; source python_env/bin/activate;{env} exec {exe}"


def _common_env(repo: Path) -> dict[str, str]:
    return {
        "PYTHONPATH": str(repo),
        "TT_METAL_HOME": str(repo),
        "PYTHONUNBUFFERED": "1",
        # Pinned host-memory H2D makes weight upload ~10x slower (#52893).
        "TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES": "0",
        "PREFILL_H2D_SERVICE_ID": SERVICE_ID,
    }


def runner_cmd(repo: Path, hosts: list[str], mode: str, fabric: str, results_dir: Path) -> str:
    """tt-run passes only TT_/ARCH_/... prefixes to remote ranks, so every PREFILL_* goes inside the bash -lc."""
    exports = _common_env(repo) | {
        "PREFILL_MANIFEST": f"{repo}/{MANIFESTS}/quad_{mode}.json",
        "PREFILL_FABRIC_MODE": fabric,
        "PREFILL_QUAD_RESULTS_DIR": str(results_dir),
        "PREFILL_NUM_LAYERS": str(LAYERS_PER_STAGE * len(hosts)),
        # A device hang raises in the rank instead of blocking forever, and the rank file records it.
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": "120",
    }
    argv = [
        "python3",
        f"{repo}/ttnn/ttnn/distributed/ttrun.py",
        "--mesh-graph-descriptor",
        str(repo / mgd_path(len(hosts), fabric)),
        "--hosts",
        ",".join(hosts),
        "--tcp-interface",
        TCP_INTERFACE,
        "--force-rediscovery",
        "--mpi-args",
        "-x PATH --tag-output",
        "bash",
        "-lc",
        _bash(repo, exports, "python3 -m models.demos.common.prefill.runners.prefill_runner"),
    ]
    return shlex.join(argv)


def producer_cmd(
    repo: Path, rank0: str, mode: str, n_hosts: int, tokens_dir: Path, duration_s: int | None, max_requests: int | None
) -> str:
    """The H2D feeder runs on rank 0's host: it attaches to rank 0's H2D service and drains the ack ring."""
    exports = _common_env(repo) | {
        "PREFILL_MODEL": f"quad_{mode}",
        "PREFILL_NUM_USERS": "1",
        "PREFILL_CHUNK_SIZE": str(CHUNK_SIZE),
        "PREFILL_MAX_SEQ_LEN": str(MAX_SEQ_LEN),
        "PREFILL_NUM_LAYERS": str(LAYERS_PER_STAGE * n_hosts),
        "PREFILL_TRACE_DIR": str(tokens_dir),
        "PREFILL_PRODUCER_CHUNKS": str(CHUNKS),
        "PREFILL_PRODUCER_INTERLEAVE": "round_robin",
        "PREFILL_PRODUCER_MAX_REQUESTS": str(max_requests or UNBOUNDED_REQUESTS),
        "PREFILL_PRODUCER_DRAIN_ACKS": "1",
        "PREFILL_SEND_SHUTDOWN": "1",
        "PREFILL_H2D_CONNECT_TIMEOUT": "120",
    }
    if duration_s is not None:
        exports["PREFILL_PRODUCER_DURATION_S"] = str(duration_s)
    argv = ["mpirun", "-np", "1", "--host", f"{rank0}:1", "--bind-to", "none", "--tag-output"]
    argv += ["-x", "PATH", "-x", "LD_LIBRARY_PATH", "bash", "-lc"]
    argv.append(_bash(repo, exports, "python3 -m models.demos.common.prefill.runners.prefill_producer"))
    return shlex.join(argv)


def vocab_size(repo: Path) -> int:
    cfg = json.loads((repo / CONFIG_JSON).read_text())
    return int(cfg.get("text_config", cfg)["vocab_size"])


def write_tokens(d: Path, n: int, vocab: int, seed: int = 1234) -> Path:
    """A minimal trace dir (metadata.json token_ids) so the producer needs no golden trace."""
    rng = random.Random(seed)
    d.mkdir(parents=True, exist_ok=True)
    ids = [rng.randrange(vocab) for _ in range(n)]
    (d / "metadata.json").write_text(json.dumps({"token_ids": ids, "n_prompt_tokens": n}))
    return d


def _short(h: str) -> str:
    return h.split(".")[0]


def chain_ok(hosts: list[str], ring: list[str]) -> bool:
    """Every consecutive pair of the pipeline is cabled: adjacent in the physical ring, either direction."""
    ring = [_short(h) for h in ring]
    pos = {h: i for i, h in enumerate(ring)}
    hs = [_short(h) for h in hosts]
    if any(h not in pos for h in hs):
        return False
    n = len(ring)
    return all((pos[a] - pos[b]) % n in (1, n - 1) for a, b in zip(hs, hs[1:]))


def ring_order(repo: Path, hosts: list[str]) -> list[str] | None:
    """Physical ring order from the exabox cabling descriptors; None when they are not there to ask."""
    cabling, deployment = DESCRIPTORS / "cabling_descriptor.textproto", DESCRIPTORS / "deployment_descriptor.textproto"
    if not (cabling.is_file() and deployment.is_file()):
        return None
    out = subprocess.run(
        [sys.executable, str(repo / "tools/scaleout/exabox/resolve_host_ring_order.py"), "--hosts", ",".join(hosts)]
        + ["--cabling", str(cabling), "--deployment", str(deployment)],
        capture_output=True,
        text=True,
    )
    try:
        return json.loads(out.stdout)["ordered_hosts"]
    except (ValueError, KeyError, TypeError):
        return None


def setup_ranks(log_text: str) -> set[int]:
    return {int(m.group(1)) for m in _SETUP.finditer(log_text)}


def bringup_blocker(log_text: str) -> str | None:
    """Why bring-up is stuck when the cause is not the galaxy: another process still holds the chips."""
    if _KMD_LOCK.search(log_text):
        return (
            f"{ENV_FAULT}chips held by another process (UMD waits on the KMD lock); not a hardware verdict. "
            "Stop the leftover process on that host, then rerun"
        )
    return None


def rank0_host(ttrun_cwd: Path, hosts: list[str]) -> str:
    """tt-run's discovery decides which host is rank 0; its rankfile says so."""
    rankfiles = sorted(ttrun_cwd.glob("generated/ttrun/*/rankfile"), key=lambda p: p.stat().st_mtime)
    if rankfiles:
        m = re.search(r"^rank\s+0=(\S+)", rankfiles[-1].read_text(), re.M)
        if m:
            return m.group(1)
    return hosts[0]


def _is_local(host: str) -> bool:
    return _short(host) == _short(socket.gethostname())


def _host_cmd(host: str, cmd: str) -> list[str]:
    if _is_local(host):
        return ["bash", "-lc", cmd]
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", host, f"bash -lc {shlex.quote(cmd)}"]


def reset_hosts(repo: Path, hosts: list[str], d: Path) -> Result:
    """Kill leftovers, drop stale shm, reset every galaxy, in parallel. The [p] keeps pkill off its own shell."""
    cmd = (
        "pkill -9 -f '[p]refill_runner|[p]refill_producer'; rm -f /dev/shm/*prefill* /dev/shm/*quad_screen*; "
        f"PATH={repo}/python_env/bin:$PATH tt-smi -glx_reset_auto"
    )
    d.mkdir(parents=True, exist_ok=True)
    failed = []

    def one(host: str) -> None:
        for n in range(1, RESET_TRIES + 1):
            log = d / f"reset_{_short(host)}{'' if n == 1 else f'.retry{n}'}.log"
            with open(log, "w") as f:
                if subprocess.run(_host_cmd(host, cmd), stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode == 0:
                    return
        failed.append(host)

    threads = [threading.Thread(target=one, args=(h,)) for h in hosts]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failed:
        return Result(Status.FAIL, f"reset failed {RESET_TRIES}x on {', '.join(failed)}, logs {d}")
    return Result(Status.PASS, f"{len(hosts)} host(s) reset")


def _spawn(cmd: str, log: Path, cwd: Path, repo: Path) -> subprocess.Popen:
    f = open(log, "w")
    return subprocess.Popen(
        ["bash", "-c", cmd], cwd=cwd, env=base_env(repo), stdout=f, stderr=subprocess.STDOUT, start_new_session=True
    )


def _kill(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(p.pid, sig)
        except ProcessLookupError:
            return
        try:
            p.wait(wait)
            return
        except subprocess.TimeoutExpired:
            continue


class _Follower:
    def __init__(self, path: Path):
        self.path, self.pos, self.text = path, 0, ""

    def read(self) -> str:
        try:
            with open(self.path, errors="replace") as f:
                f.seek(self.pos)
                new = f.read()
                self.pos = f.tell()
        except FileNotFoundError:
            return ""
        self.text = (self.text + new)[-(1 << 20) :]
        return new


def _attempt_dir(ctx: Ctx, mode: str, duration_s, max_requests, attempt: int) -> Path:
    budget = f"{duration_s}s" if duration_s is not None else f"{max_requests}req"
    return ctx.run_dir / "quad" / f"{mode}_{budget}" / f"attempt_{attempt:02d}"


def run_once(
    ctx: Ctx, state: State, mode: str, fabric: str, hosts: list[str], duration_s: int | None, max_requests: int | None
) -> Result:
    d = _attempt_dir(ctx, mode, duration_s, max_requests, state.attempt)
    results, ttrun_cwd = d / "results", d / "ttrun-cwd"
    for p in (results, ttrun_cwd):
        p.mkdir(parents=True, exist_ok=True)
    reset = reset_hosts(ctx.repo, hosts, d / "reset")
    if reset.status != Status.PASS:
        return reset
    tokens = write_tokens(d / "tokens", MAX_SEQ_LEN, vocab_size(ctx.repo))
    runner_log, producer_log = d / "runner.log", d / "producer.log"
    state.log = str(runner_log)
    runner = producer = None
    hung = shutdown_stuck = False
    notes: list[str] = []
    try:
        runner = _spawn(runner_cmd(ctx.repo, hosts, mode, fabric, results), runner_log, ttrun_cwd, ctx.repo)
        follow, ready, start = _Follower(runner_log), set(), time.time()
        last_ready = start
        while len(ready) < len(hosts):
            now_ready = ready | setup_ranks(follow.read())
            if now_ready != ready:
                ready, last_ready = now_ready, time.time()
            blocker = bringup_blocker(follow.text) if time.time() - last_ready > KMD_LOCK_S else None
            if blocker is not None:
                r = Result(Status.FAIL, f"{blocker}; log {runner_log}")
                return r
            state.now = f"bring-up {len(ready)}/{len(hosts)} ranks ready ({fabric})"
            ctx.tick()
            if runner.poll() is not None:
                notes.append(f"runner exited rc={runner.returncode} during bring-up, log {runner_log}")
                break
            if time.time() - start > SETUP_TIMEOUT_S:
                hung = True
                notes.append(f"bring-up not done after {SETUP_TIMEOUT_S}s, log {runner_log}")
                break
            time.sleep(POLL_S)
        if len(ready) == len(hosts):
            rank0 = rank0_host(ttrun_cwd, hosts)
            cmd = producer_cmd(ctx.repo, rank0, mode, len(hosts), tokens, duration_s, max_requests)
            producer = _spawn(cmd, producer_log, ttrun_cwd, ctx.repo)
            fed = time.time()
            while producer.poll() is None:
                ranks = load_ranks(results, len(hosts), retries=1, delay=0)
                last = max([fed] + [r["last_progress"] for r in ranks.values() if r])
                iters = min((r["iterations"] for r in ranks.values() if r), default=0)
                state.now = f"{mode} {fabric}: {iters} iterations, {time.time() - fed:.0f}s"
                ctx.tick()
                if runner.poll() is not None:
                    notes.append(f"runner exited rc={runner.returncode} while serving, log {runner_log}")
                    break
                if time.time() - last > STALL_S:
                    hung = True
                    notes.append(f"no stage progressed for {STALL_S}s")
                    break
                time.sleep(POLL_S)
            if not hung and producer.poll() is not None:
                try:
                    runner.wait(SHUTDOWN_S)
                except subprocess.TimeoutExpired:
                    shutdown_stuck = True
                    notes.append(f"runner did not exit {SHUTDOWN_S}s after the shutdown sentinel")
    finally:
        _kill(producer)
        _kill(runner)

    ranks = load_ranks(results, len(hosts))
    # A rank stuck short of its shutdown is a hang; every rank done but the job not exiting is teardown only.
    all_done = all(r is not None and r["done"] for r in ranks.values())
    hung = hung or (shutdown_stuck and not all_done)
    rows = stage_verdicts(ranks, load_hashes(results, len(hosts)), hosts, hung)
    r = overall(rows, hung)
    if r.status == Status.PASS and shutdown_stuck:
        r = Result(Status.WARN, "every stage passed; the runner hung in teardown")
    elif r.status == Status.PASS:
        if producer is None or producer.returncode != 0:
            rc = None if producer is None else producer.returncode
            r = Result(Status.FAIL, f"producer rc={rc}, log {producer_log}")
        elif runner.returncode != 0:
            r = Result(Status.FAIL, f"runner rc={runner.returncode}, log {runner_log}")
    if notes:
        r.reason = f"{r.reason}; {'; '.join(notes)}"
    r.details = [f"{label}: {res.reason}" for label, res in rows]
    state.rows = [res for _, res in rows]
    return r


def build_quad_steps(hosts: list[str], soak_s: int, run_once=run_once) -> list[Step]:
    blocked: dict[str, str] = {}

    def reset_step() -> Step:
        def run(ctx: Ctx, state: State) -> Result:
            if ctx.mode_co:
                return would_run(f"tt-smi -glx_reset_auto on {','.join(hosts)}")
            state.log = str(step_dir(ctx, "quad:reset"))
            return reset_hosts(ctx.repo, hosts, step_dir(ctx, "quad:reset"))

        return Step("quad:reset", "reset", "gate", None, True, run)

    def test_step(key: str, title: str, mode: str, duration_s: int | None, max_requests: int | None) -> Step:
        def run(ctx: Ctx, state: State) -> Result:
            if mode != "passthrough" and "passthrough" in blocked:
                return Result(Status.PENDING, blocked["passthrough"])
            if ctx.mode_co:
                d = ctx.run_dir / "quad"
                return would_run(runner_cmd(ctx.repo, hosts, mode, TORUS, d / "results"))
            r = run_once(ctx, state, mode, TORUS, hosts, duration_s, max_requests)
            if r.status not in (Status.FAIL, Status.HANG) or r.reason.startswith(ENV_FAULT):
                return r
            f = State(Result(Status.RUN), start=time.time(), attempt=2, env={"PREFILL_FABRIC_MODE": MESH})
            state.fallbacks.append(f)
            try:
                f.result = run_once(ctx, f, mode, MESH, hosts, duration_s, max_requests)
            except Exception as e:
                f.result = Result(Status.FAIL, f"{type(e).__name__}: {e}")
            f.end = time.time()
            if mode == "passthrough" and f.result.status in (Status.FAIL, Status.HANG):
                blocked["passthrough"] = "not run: passthrough failed on torus 2D XY and on mesh 2D"
            return r

        return Step(key, title, "quad", None, True, run)

    return [
        reset_step(),
        test_step("quad:passthrough-smoke", "passthrough-smoke", "passthrough", None, PASSTHROUGH_REQUESTS),
        test_step("quad:block-random-smoke", "block-random-smoke", "block_random", SMOKE_S, None),
        test_step("quad:block-random-soak", "block-random-soak", "block_random", soak_s, None),
    ]
