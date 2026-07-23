# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""End-to-end galaxy test: the prefill runner (holds the mesh) + the scenario producer (device-less
client), as TWO processes, with a per-slot KV-cache PCC gate.

Each scenario spins up its OWN runner (per-scenario config) via the `_running_runner` context manager,
runs the producer against it, then tears it down. That fully isolates scenarios (independent runner
config, no cross-contamination, a crash in one doesn't block the others) at the cost of paying the
runner startup (full model load + kernel JIT) once PER scenario.
"""

import contextlib
import glob
import os
import signal
import subprocess
import sys
import time

import pytest

from models.common.utility_functions import is_blackhole, skip_for_slow_dispatch

CHUNK_SIZE = 5120
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", "2"))
SERVICE_ID = "ci_ds_prefill"
TABLE_PATH = "/tmp/ci_prefill_kv_table.pb"  # IPC rendezvous files; cleaned up around each scenario
DEVMAP_PATH = "/tmp/ci_prefill_kv_devmap.json"
# Logs live under generated/test_reports/ so the workflow's upload-artifact-with-job-uuid step
# (path /work/generated/test_reports/, prefix "test_reports_") uploads them as a downloadable artifact.
_REPORT_DIR = os.path.join(os.environ.get("TT_METAL_HOME", os.getcwd()), "generated", "test_reports")

_RUNNER_MODULE = "models.demos.common.prefill.runners.prefill_runner"
_PRODUCER_MODULE = "models.demos.common.prefill.runners.prefill_producer"

_READY_TIMEOUT_S = int(os.environ.get("PREFILL_CI_RUNNER_READY_TIMEOUT_S", "1200"))  # 20 min for load + JIT
_PRODUCER_TIMEOUT_S = int(os.environ.get("PREFILL_CI_PRODUCER_TIMEOUT_S", "900"))
_LOG_TAIL_LINES = int(os.environ.get("PREFILL_CI_LOG_TAIL_LINES", "200"))
_DESCRIPTOR = f"/dev/shm/tt_h2d_stream_service_{SERVICE_ID}.bin"

# --- launch mode: standard (bare pytest) vs CI (under an MPI launcher) --------------------------
# This test runs in one of two ways, and ONLY the child-process environment differs between them:
#
#   standard  -- `pytest .../test_producer_runner_e2e.py ...` invoked directly (local iteration).
#                The runner/producer children inherit this process's environment unchanged and come
#                up as standalone singletons. Historical behaviour; unchanged.
#
#   ci        -- launched under mpirun/prterun to match the blaze pipeline convention, e.g.
#                `mpirun --pernode ... python3 -m pytest ...`. The launcher exports OMPI_*/PMIX_*/
#                PRTE_* into this (pytest) process. If a child inherited them, the runner's
#                ttnn.init_distributed_context() (MPI_Init) would join the launcher's PMIx session
#                instead of standing up its own singleton; because we tear the long-lived runner
#                down with a signal and never call MPI_Finalize, prterun then reports an "abnormal
#                termination" and forces mpirun to exit non-zero -- masking an otherwise-green
#                pytest (the exact failure seen in CI). The CI path strips those launcher vars from
#                the child environment so the runner/producer detach from prterun and run as clean
#                singletons, identical to the standard path. This pytest process never calls
#                MPI_Init, so it exits with pytest's real return code.
#
# Auto-detected from the launcher env vars (so the mpirun wrapper works even with no flag) and
# pinned explicitly by the CI matrix entry via PREFILL_RUNNER_LAUNCH=ci; unset locally (standard).
_MPI_LAUNCHER_PREFIXES = ("OMPI_", "PMIX_", "PRTE_", "PRRTE_")


def _mpi_launcher_keys(env) -> list:
    """The launcher / PMIx-session variables present in `env` (empty when not under mpirun)."""
    return [k for k in env if k.startswith(_MPI_LAUNCHER_PREFIXES)]


def _launch_mode() -> str:
    """Return "ci" when launched under an MPI launcher (mpirun/prterun), else "standard".
    PREFILL_RUNNER_LAUNCH=ci|standard overrides the auto-detection."""
    forced = os.environ.get("PREFILL_RUNNER_LAUNCH", "").strip().lower()
    if forced in ("ci", "mpi", "mpirun"):
        return "ci"
    if forced in ("standard", "standalone", "local", "bare", "pytest"):
        return "standard"
    return "ci" if _mpi_launcher_keys(os.environ) else "standard"

pytestmark = [
    skip_for_slow_dispatch(),
    pytest.mark.skipif(not is_blackhole(), reason="prefill runner + H2DStreamService require a Blackhole galaxy"),
]

# Each scenario carries its OWN runner config (users / max_seq_len) + the producer schedule. Runners
# run sequentially (one at a time), so each only needs users*max_seq_len to fit the KV budget on its
# own. The producer inherits the runner's NUM_USERS (drives all its slots) unless overridden.
SCENARIOS = {
    # 1) Full-depth single user: 11 x 5120 = 56320 = the full Kimi golden trace. Deepest correctness gate.
    "single_user_full_depth": {
        "users": 1,
        "max_seq_len": 56320,
        "producer": {"PREFILL_PRODUCER_CHUNKS": "11", "PREFILL_PRODUCER_MAX_REQUESTS": "1"},
    },
    # 2) Round-robin across 4 users, 4 chunks each (u0c0, u1c0, u2c0, u3c0, u0c1, ...). Deterministic
    #    interleave correctness; breadth over depth so it fits the KV budget (4 x 20480).
    "round_robin_4users": {
        "users": 4,
        "max_seq_len": 20480,
        "producer": {
            "PREFILL_PRODUCER_INTERLEAVE": "round_robin",
            "PREFILL_PRODUCER_CHUNKS": "4",
            "PREFILL_PRODUCER_MAX_REQUESTS": "4",
            "PREFILL_PRODUCER_P_GAP": "0",
            "PREFILL_PRODUCER_P_BURST": "0",
        },
    },
    # 3) Random interleave across 8 users, 1-2 chunks each, with gaps/bursts + slot recycling (seeded =>
    #    reproducible). Chaotic multi-user correctness at the highest slot count (KV budget 8 x 10240).
    "random_8users": {
        "users": 8,
        "max_seq_len": 10240,
        "producer": {
            "PREFILL_PRODUCER_CHUNKS": "1,2",
            "PREFILL_PRODUCER_MAX_REQUESTS": "12",
            "PREFILL_PRODUCER_P_GAP": "0.1",
            "PREFILL_PRODUCER_P_BURST": "0.2",
        },
    },
}


def _transport_env(num_users: int, max_seq_len: int, **extra) -> dict:
    """Inherit the CI/dev env (weights cache, HF, golden trace) and add the shared orchestration knobs
    for this scenario's runner+producer. `extra` layers on the runner (MOCK_MIGRATION) or producer
    (schedule + CHECK_PCC) knobs.

    In the CI (mpirun) launch path the launcher's MPI/PMIx session variables are stripped from the
    returned child environment so the runner/producer run as standalone singletons detached from
    prterun (see the launch-mode note above). No-op in the standard path."""
    env = dict(os.environ)
    env.update(
        PREFILL_CHUNK_SIZE=str(CHUNK_SIZE),
        PREFILL_MAX_SEQ_LEN=str(max_seq_len),
        PREFILL_NUM_LAYERS=str(NUM_LAYERS),
        PREFILL_NUM_USERS=str(num_users),
        PREFILL_H2D_SERVICE_ID=SERVICE_ID,
        PREFILL_MIGRATION_TABLE_PATH=TABLE_PATH,
        PREFILL_MIGRATION_DEVICE_MAP_PATH=DEVMAP_PATH,
    )
    env.update(extra)
    if _launch_mode() == "ci":
        # CI path: detach the child from the mpirun/prterun session so it comes up as a standalone
        # singleton (see the launch-mode note above). Harmless when no launcher vars are present.
        for key in _mpi_launcher_keys(env):
            env.pop(key, None)
    return env


def _cleanup_ipc() -> None:
    for path in (TABLE_PATH, DEVMAP_PATH, *glob.glob(f"/dev/shm/*{SERVICE_ID}*")):
        try:
            os.remove(path)
        except OSError:
            pass


def _tail(path: str, n: int = _LOG_TAIL_LINES) -> str:
    try:
        with open(path) as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return f"(no log at {path})"


def _emit_log_group(title: str, path: str, n: int = _LOG_TAIL_LINES) -> None:
    """Echo the tail of `path` to stdout so it shows inline in the GitHub Actions step log, wrapped in
    a collapsible ::group:: when running under Actions. The FULL file is uploaded as an artifact
    (generated/test_reports/), so this is only the bounded inline view."""
    if not os.path.exists(path):
        return
    in_gha = os.environ.get("GITHUB_ACTIONS") == "true"
    print(f"::group::{title} (tail {n} lines)" if in_gha else f"\n===== {title} (tail {n} lines) =====", flush=True)
    print(_tail(path, n), flush=True)
    if in_gha:
        print("::endgroup::", flush=True)


@contextlib.contextmanager
def _running_runner(tag: str, num_users: int, max_seq_len: int):
    """Spin up ONE runner (mock-migration, request mode) for a scenario and tear it down. Yields once
    it has published the H2D descriptor + KV table + device map (i.e. it is serving)."""
    os.makedirs(_REPORT_DIR, exist_ok=True)
    log_path = os.path.join(_REPORT_DIR, f"ci_runner_{tag}.log")
    _cleanup_ipc()  # a stale table/descriptor from a prior scenario would make the readiness poll pass early
    env = _transport_env(num_users, max_seq_len, PREFILL_MOCK_MIGRATION="1")
    mode = _launch_mode()
    if mode == "ci":
        print(
            f"[producer-runner-e2e] launch mode=ci: detaching runner/producer from mpirun "
            f"(stripped {len(_mpi_launcher_keys(os.environ))} launcher env vars from child env)",
            flush=True,
        )
    else:
        print("[producer-runner-e2e] launch mode=standard (bare pytest; children inherit env)", flush=True)
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, "-m", _RUNNER_MODULE], env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + _READY_TIMEOUT_S
        while not (os.path.exists(_DESCRIPTOR) and os.path.exists(TABLE_PATH) and os.path.exists(DEVMAP_PATH)):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"runner [{tag}] exited early (rc={proc.returncode}) during startup:\n{_tail(log_path)}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"runner [{tag}] not ready within {_READY_TIMEOUT_S}s:\n{_tail(log_path)}")
            time.sleep(2.0)
        yield log_path
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)  # graceful; SIGKILL is the hard fallback
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        _emit_log_group(f"runner log [{tag}]", log_path)  # inline tail; the artifact has the full log
        _cleanup_ipc()


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_producer_runner_pcc(scenario):
    """Spin up a fresh runner for the scenario, drive it with the producer, and require the per-slot
    KV PCC gate to pass (the producer exits non-zero if any resident slot is below threshold)."""
    sc = SCENARIOS[scenario]
    prod_log = os.path.join(_REPORT_DIR, f"ci_producer_{scenario}.log")
    with _running_runner(scenario, sc["users"], sc["max_seq_len"]) as runner_log:
        env = _transport_env(sc["users"], sc["max_seq_len"], PREFILL_PRODUCER_CHECK_PCC="1", **sc["producer"])
        try:
            with open(prod_log, "w") as f:
                result = subprocess.run(
                    [sys.executable, "-m", _PRODUCER_MODULE],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=_PRODUCER_TIMEOUT_S,
                )
        finally:
            _emit_log_group(f"producer log [{scenario}]", prod_log)  # inline tail; the artifact has the full log
        assert result.returncode == 0, (
            f"producer scenario {scenario!r} failed (rc={result.returncode}; PCC below threshold or error). "
            f"See the grouped producer log above and the test_reports_* artifact. Runner tail:\n{_tail(runner_log)}"
        )
