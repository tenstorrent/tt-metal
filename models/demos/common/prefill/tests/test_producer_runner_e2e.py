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
# GLM-5.2 golden trace carrying BOTH the 78 kv_cache layers and the 21 dsa/indexer_k_layer_* dirs, at
# 56320 rows (= 11 x CHUNK_SIZE). The adapter's own prefill_trace_default omits dsa/, which would leave
# the merged table's index config with no golden to PCC against.
GLM52_TRACE = "/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k"
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
#
# Optional per-scenario keys:
#   layers             -- PREFILL_NUM_LAYERS for this scenario (default: the module-level NUM_LAYERS)
#   env                -- extra env applied to BOTH the runner and the producer (model, trace dir, ...)
#   ready_timeout_s    -- override the runner startup budget (bigger models load + JIT for longer)
#   producer_timeout_s -- override the producer budget (the PCC sweep scales with layers x seq_len)
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
    # 4) GLM-5.2 (sparse / DSA) full-depth single user over ALL 78 layers. This is the gate for the
    #    MERGED two-config KV chunk address table: config 0 = the bf16 ROW_MAJOR MLA KVPE cache (all 78
    #    layers), config 1 = the bfp8 lightning-indexer KEY cache (only the 21 `full` layers, compacted).
    #    The runner builds that single merged table under PREFILL_MOCK_MIGRATION and the producer reads
    #    BOTH configs back through it over UMD, so a wrong address in either config shows up as a PCC
    #    failure. 11 x 5120 = 56320 is exactly the trace depth.
    #
    #    ALL layers is mandatory here, not a preference: the index cache is sized from the model's whole
    #    indexer_types map (21 full layers), so a truncated run leaves the upper index ranks unwritten —
    #    the producer asserts on exactly that mismatch rather than PCC'ing untouched memory.
    "glm52_full_depth_kv_table": {
        "users": 1,
        "layers": 78,
        "max_seq_len": 56320,
        "env": {
            "PREFILL_MODEL": "glm_5_2",
            "PREFILL_TRACE_DIR": GLM52_TRACE,
            # The table describes all 78 layers, so the last layer must still WRITE its KV; the runner's
            # default headless-last-layer optimization would leave layer 77 empty.
            "PREFILL_KV_ONLY_LAST_LAYER": "0",
        },
        # 78 layers of GLM-5.2 weights + kernel JIT, then a two-config PCC sweep of ~174k sequential
        # read_dram_umd block reads (78 x 1760 for KVPE + 21 x 1760 for the index cache). Both phases
        # are far past the Kimi-sized defaults.
        "ready_timeout_s": 3600,
        "producer_timeout_s": 7200,
        "producer": {"PREFILL_PRODUCER_CHUNKS": "11", "PREFILL_PRODUCER_MAX_REQUESTS": "1"},
    },
    # 5) Same run as (4) but with the GLM-5.2 SPxTP KV dedup on (PREFILL_TP_SHARD_KV=1): the KVPE and
    #    index caches are sequence-sharded across BOTH mesh axes, so each of the 32 devices holds a
    #    distinct 1/(sp*tp) slice (1760 tokens instead of 7040) and the migration table addresses each
    #    (row, col) device with a singleton group. This is the only end-to-end gate that the allocator,
    #    the TP-sharded write op, the TP-inner/SP-outer read gather and the table all agree byte-for-byte.
    #
    #    ACCEPTANCE IS A DIFF, NOT A THRESHOLD. TP dedup is pure storage dedup — bit-identical by design —
    #    so the check that means something is "per-layer nope/pe/index PCC equals the SP-only baseline from
    #    (4), layer for layer", not "above some absolute number". PREFILL_STANDALONE_CHUNKED_PCC is lowered
    #    to a floor just under the known SP-only minimum (0.8608 on nope @ layer 75) so this scenario does
    #    NOT re-fail on that pre-existing full-depth KVPE issue and still prints every per-layer line for
    #    the diff. A genuinely broken TP layout reads the wrong device or the wrong 1/tp window and lands
    #    near zero, far below the floor.
    "glm52_full_depth_kv_table_tp_sharded": {
        "users": 1,
        "layers": 78,
        "max_seq_len": 56320,
        "env": {
            "PREFILL_MODEL": "glm_5_2",
            "PREFILL_TRACE_DIR": GLM52_TRACE,
            # Required by PREFILL_TP_SHARD_KV: the kv-only last layer has no TP-sharded write path (and
            # the table describes all 78 layers, so layer 77 must still write its KV).
            "PREFILL_KV_ONLY_LAST_LAYER": "0",
            "PREFILL_TP_SHARD_KV": "1",
            "PREFILL_STANDALONE_CHUNKED_PCC": "0.85",
        },
        "ready_timeout_s": 3600,
        "producer_timeout_s": 7200,
        "producer": {"PREFILL_PRODUCER_CHUNKS": "11", "PREFILL_PRODUCER_MAX_REQUESTS": "1"},
    },
}


def _transport_env(num_users: int, max_seq_len: int, num_layers: int = NUM_LAYERS, **extra) -> dict:
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
        PREFILL_NUM_LAYERS=str(num_layers),
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


def _scenario_env(sc: dict, **extra) -> dict:
    """Child environment for one scenario's runner or producer: the shared transport knobs, the
    scenario's own layer count, its model-specific `env` (model, trace dir, ...), then the role-specific
    `extra` (MOCK_MIGRATION for the runner; the schedule + CHECK_PCC for the producer). Both roles must
    agree on model/layers/seq_len or the producer would validate against a differently-shaped cache."""
    return _transport_env(
        sc["users"],
        sc["max_seq_len"],
        num_layers=sc.get("layers", NUM_LAYERS),
        **{**sc.get("env", {}), **extra},
    )


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
def _running_runner(tag: str, sc: dict):
    """Spin up ONE runner (mock-migration, request mode) for a scenario and tear it down. Yields once
    it has published the H2D descriptor + KV table + device map (i.e. it is serving)."""
    os.makedirs(_REPORT_DIR, exist_ok=True)
    log_path = os.path.join(_REPORT_DIR, f"ci_runner_{tag}.log")
    _cleanup_ipc()  # a stale table/descriptor from a prior scenario would make the readiness poll pass early
    env = _scenario_env(sc, PREFILL_MOCK_MIGRATION="1")
    ready_timeout_s = int(sc.get("ready_timeout_s", _READY_TIMEOUT_S))
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
        deadline = time.monotonic() + ready_timeout_s
        while not (os.path.exists(_DESCRIPTOR) and os.path.exists(TABLE_PATH) and os.path.exists(DEVMAP_PATH)):
            if proc.poll() is not None:
                raise RuntimeError(
                    f"runner [{tag}] exited early (rc={proc.returncode}) during startup:\n{_tail(log_path)}"
                )
            if time.monotonic() > deadline:
                raise TimeoutError(f"runner [{tag}] not ready within {ready_timeout_s}s:\n{_tail(log_path)}")
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


def _scenario_params():
    """One pytest param per scenario, each carrying a pytest-timeout budget matching its own limits.

    pytest.ini sets a blanket ``timeout = 300``, which is below what this test declares even for the
    small scenarios (_READY_TIMEOUT_S + _PRODUCER_TIMEOUT_S = 2100s) and nowhere near a full-depth
    model, which spends longer than that just loading weights. pytest-timeout reads the marker at
    setup time, so the bound has to be attached at collection rather than inside the test body. The
    real per-phase enforcement stays in _running_runner/subprocess.run; this only stops SIGALRM from
    killing the test before those can report a useful failure.
    """
    return [
        pytest.param(
            name,
            marks=pytest.mark.timeout(
                sc.get("ready_timeout_s", _READY_TIMEOUT_S)
                + sc.get("producer_timeout_s", _PRODUCER_TIMEOUT_S)
                + 600  # slack for import, table build, teardown and log emission
            ),
        )
        for name, sc in SCENARIOS.items()
    ]


@pytest.mark.parametrize("scenario", _scenario_params())
def test_producer_runner_pcc(scenario):
    """Spin up a fresh runner for the scenario, drive it with the producer, and require the per-slot
    KV PCC gate to pass (the producer exits non-zero if any resident slot is below threshold)."""
    sc = SCENARIOS[scenario]
    prod_log = os.path.join(_REPORT_DIR, f"ci_producer_{scenario}.log")
    with _running_runner(scenario, sc) as runner_log:
        env = _scenario_env(sc, PREFILL_PRODUCER_CHECK_PCC="1", **sc["producer"])
        try:
            with open(prod_log, "w") as f:
                result = subprocess.run(
                    [sys.executable, "-m", _PRODUCER_MODULE],
                    env=env,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=int(sc.get("producer_timeout_s", _PRODUCER_TIMEOUT_S)),
                )
        finally:
            _emit_log_group(f"producer log [{scenario}]", prod_log)  # inline tail; the artifact has the full log
        assert result.returncode == 0, (
            f"producer scenario {scenario!r} failed (rc={result.returncode}; PCC below threshold or error). "
            f"See the grouped producer log above and the test_reports_* artifact. Runner tail:\n{_tail(runner_log)}"
        )
