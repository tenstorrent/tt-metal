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
import threading
import time

import pytest

from models.common.utility_functions import is_blackhole, skip_for_slow_dispatch

CHUNK_SIZE = 5120
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", "2"))
# GLM-5.2 golden trace carrying BOTH the 78 kv_cache layers and the 21 dsa/indexer_k_layer_* dirs, at
# 56320 rows (= 11 x CHUNK_SIZE). The adapter's own prefill_trace_default omits dsa/, which would leave
# the merged table's index config with no golden to PCC against.
GLM52_TRACE = "/mnt/models/deepseek-prefill-cache/glm-traces/vllm-glm52-indexer-kcache-55k"
# GLM-5.2 pretrained checkpoint + the MTP weight cache tree (layer 78 block + the fused MTP weights),
# for the MTP4 scenario. The MTP weights are keyed on layer 78, which the trunk cache does not carry,
# so they live in their own tree with the same <variant>_<arch>_<N>dev/<sp>x<tp> leaf.
GLM52_HF_MODEL = os.environ.get("GLM52_HF_MODEL", "/mnt/models/deepseek-prefill-cache/GLM-5.2-FP8")
GLM52_MTP_TTNN_CACHE = os.environ.get(
    "TT_GLM52_MTP_TTNN_CACHE", "/mnt/models/deepseek-prefill-cache/glm52_mtp_ttnn_cache"
)
SERVICE_ID = "ci_ds_prefill"
TABLE_PATH = "/tmp/ci_prefill_kv_table.pb"  # IPC rendezvous files; cleaned up around each scenario
DEVMAP_PATH = "/tmp/ci_prefill_kv_devmap.json"
# Full logs are kept under generated/test_reports/ on the pod. NOTE: no workflow step uploads that
# directory today (blaze-models-prefill-tests-impl.yaml only uploads PREFILL_SUMMARIES/**/*.md), so
# in CI these files die with the pod -- what survives is the live stream (see _ChildStream below).
_REPORT_DIR = os.path.join(os.environ.get("TT_METAL_HOME", os.getcwd()), "generated", "test_reports")

_RUNNER_MODULE = "models.demos.common.prefill.runners.prefill_runner"
_PRODUCER_MODULE = "models.demos.common.prefill.runners.prefill_producer"

_READY_TIMEOUT_S = int(os.environ.get("PREFILL_CI_RUNNER_READY_TIMEOUT_S", "1200"))  # 20 min for load + JIT
_PRODUCER_TIMEOUT_S = int(os.environ.get("PREFILL_CI_PRODUCER_TIMEOUT_S", "900"))
_LOG_TAIL_LINES = int(os.environ.get("PREFILL_CI_LOG_TAIL_LINES", "200"))
_STREAM_LOGS = os.environ.get("PREFILL_CI_STREAM_LOGS", "1") == "1"  # live child output; see _ChildStream
_HEARTBEAT_S = float(os.environ.get("PREFILL_CI_HEARTBEAT_S", "30"))
_DESCRIPTOR = f"/dev/shm/tt_h2d_stream_service_{SERVICE_ID}.bin"
_T0 = time.monotonic()

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
    # 5) GLM-5.2 with MTP4 (#53533): scenario 4 plus four Multi-Token-Prediction levels after the
    #    trunk's last layer. What this exercises that scenario 4 does not, end to end through the
    #    real serving path:
    #      * the H2D socket carries CHUNK_SIZE + K tokens per chunk, in rows that OVERLAP by K, so
    #        MTP level k's window is the same local slice on every SP chip (no cross-chip rotation);
    #      * the runner slices the trunk's own CHUNK_SIZE tokens back out ON DEVICE and the trunk
    #        forward is bit-identical to scenario 4's -- which is exactly what the 78-layer KVPE +
    #        21-rank index PCC below re-proves;
    #      * the last rank embeds each level's window on device and runs the four levels, writing
    #        KVPE slots 78..81 and sharing one indexer slot (index_share_for_mtp_iteration).
    #
    #    The golden trace carries no KV for the layers past the trunk, so the MTP-vs-golden PCC SKIPS
    #    itself and says which layers it wanted; that math is gated against a torch reference in
    #    deepseek_v3_d_p/tests/mtp_prefill/. The comparison is written and starts running by itself
    #    the moment a trace carrying kv_cache/layer_{78..81} is dropped in -- no edit here. What a
    #    serving run does gate is the pair of things only it can show: every level's slot was written,
    #    and no two levels share a slot (a collision is invisible in the model outputs).
    #
    #    Needs the MTP weight cache (layer 78 + the fused MTP weights); the runner will NOT build it
    #    inside the serving process. Populate it with tests/mtp_prefill/test_mtp_transformer_chunks.py.
    "glm52_mtp4": {
        "users": 1,
        "layers": 78,
        "max_seq_len": 56320,
        "env": {
            "PREFILL_MODEL": "glm_5_2",
            "PREFILL_TRACE_DIR": GLM52_TRACE,
            "PREFILL_KV_ONLY_LAST_LAYER": "0",  # MTP level 1 consumes the last layer's post-norm hidden
            "PREFILL_MTP_LEVELS": "4",
            # Read by BOTH roles: the runner sizes its sockets/caches from it and the producer builds
            # the overlapping H2D rows from it. A mismatch is caught by the payload-size assert.
            "TT_GLM52_MTP_TTNN_CACHE": GLM52_MTP_TTNN_CACHE,
            "PREFILL_HF_MODEL": GLM52_HF_MODEL,
        },
        # Scenario 4's budgets plus the MTP tail: 4 more blocks per chunk on the last rank, and 4 more
        # KVPE layers to read back (78 + 4 of 1760 block reads each).
        "ready_timeout_s": 3600,
        "producer_timeout_s": 7200,
        "producer": {"PREFILL_PRODUCER_CHUNKS": "11", "PREFILL_PRODUCER_MAX_REQUESTS": "1"},
    },
}

# Opt-in prompt-driven scenario: instead of a recorded golden trace, generate the reference KV from a
# user prompt on the host (device-less pre-step) and validate device KV against it. Enabled by pointing
# PREFILL_PROMPT_FILE at a prompt JSON. The host reference forward uses chunked-SDPA MLA, so its memory
# stays bounded and PREFILL_PROMPT_CHUNKS chunks (default 1) can be validated — the correctness gate for
# arbitrary prompts. Runtime is still O(seq^2) in the sequence length, so deeper runs take longer.
_PROMPT_FILE = os.environ.get("PREFILL_PROMPT_FILE")
if _PROMPT_FILE:
    _PROMPT_CHUNKS = int(os.environ.get("PREFILL_PROMPT_CHUNKS", "1"))
    SCENARIOS["prompt_single_user"] = {
        "users": 1,
        # Cache width must exceed the deepest single push (Q.seq == CHUNK_SIZE): the chunked-attention
        # SDPA gate requires Q.seq < cache-width, so a 1-chunk prompt still needs a 2-chunk cache. For
        # N>=2 the accumulated fill N*CHUNK_SIZE already exceeds one chunk, so N*CHUNK_SIZE suffices.
        "max_seq_len": max(2, _PROMPT_CHUNKS) * CHUNK_SIZE,
        "producer": {"PREFILL_PRODUCER_CHUNKS": str(_PROMPT_CHUNKS), "PREFILL_PRODUCER_MAX_REQUESTS": "1"},
        "prompt_file": _PROMPT_FILE,
        "isl": _PROMPT_CHUNKS * CHUNK_SIZE,
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
    a collapsible ::group:: when running under Actions. Post-mortem fallback for PREFILL_CI_STREAM_LOGS=0;
    with streaming on (the default) the whole log is already in the step log, so this is skipped."""
    if not os.path.exists(path):
        return
    in_gha = os.environ.get("GITHUB_ACTIONS") == "true"
    print(f"::group::{title} (tail {n} lines)" if in_gha else f"\n===== {title} (tail {n} lines) =====", flush=True)
    print(_tail(path, n), flush=True)
    if in_gha:
        print("::endgroup::", flush=True)


# --- live child output --------------------------------------------------------------------------
# Both children are launched detached with their output going into a file, so in CI the runner used
# to be invisible until the post-mortem tail in `finally` -- and a step killed by the workflow's
# `timeout-minutes` never reaches that tail, nor is generated/test_reports/ uploaded anywhere, so a
# timed-out run yielded NOTHING about what the runner was doing. Each child's output is now pumped
# onto this process's stdout line by line as it is produced (pytest.ini's addopts carry -s, so
# nothing here is captured or deferred), which puts it in the CI step log while the test is still
# running: the runner and the producer show up interleaved and tagged, and whatever was printed
# before a kill survives the kill. Set PREFILL_CI_STREAM_LOGS=0 for the old file-only behaviour.
def _elapsed() -> str:
    return f"+{time.monotonic() - _T0:7.1f}s"


class _ChildStream:
    """One child's merged stdout+stderr, pumped by a daemon thread into `log_path` AND onto our own
    stdout, tagged with `tag` and the elapsed test time.

    The pump being a thread is what makes the runner visible while the main thread sits blocked in
    the producer's wait: both children stream concurrently, and the tags tell them apart.
    """

    def __init__(self, tag: str, log_path: str):
        self.tag = tag
        self.log_path = log_path
        self.lines = 0
        self.last_output = time.monotonic()
        self._thread = None

    def start(self, proc: subprocess.Popen) -> "_ChildStream":
        self._thread = threading.Thread(target=self._pump, args=(proc,), name=f"stream-{self.tag}", daemon=True)
        self._thread.start()
        return self

    def _pump(self, proc: subprocess.Popen) -> None:
        try:
            log = open(self.log_path, "w", buffering=1)  # line-buffered, so the tail survives a SIGKILL
        except OSError:
            log = None
        try:
            for line in proc.stdout:  # text mode, line at a time; ends at child EOF
                self.lines += 1
                self.last_output = time.monotonic()
                # File and stdout fail INDEPENDENTLY on purpose. Sharing one try meant that once
                # log.write() started failing (a full report volume, say), it raised before every
                # later print() and silently took the live stream with it -- and the step log is
                # exactly the view you still have when storage is the thing that broke. Neither may
                # stop the drain either: the child blocks as soon as the 64K pipe fills.
                if log is not None:
                    try:
                        log.write(line)
                    except Exception:
                        with contextlib.suppress(Exception):
                            log.close()
                        log = None  # a broken handle is dropped, not retried once per remaining line
                        with contextlib.suppress(Exception):
                            print(f"[{self.tag}] file log stopped ({self.log_path}); stream continues", flush=True)
                if _STREAM_LOGS:
                    with contextlib.suppress(Exception):
                        # Single write: a streamed line can't tear against the main thread's output.
                        print(f"[{self.tag} {_elapsed()}] {line.rstrip()}\n", end="", flush=True)
        except Exception:
            # Same reason: whatever went wrong reading the pipe, keep emptying it so that a child
            # holding the mesh can never wedge on a full pipe. The log/stream just stops here.
            with contextlib.suppress(Exception):
                while proc.stdout.read(65536):
                    pass
        finally:
            with contextlib.suppress(Exception):
                proc.stdout.close()
            if log is not None:
                with contextlib.suppress(Exception):
                    log.close()

    def status(self) -> str:
        return f"{self.tag}: {self.lines} lines, last output {time.monotonic() - self.last_output:.0f}s ago"

    def finish(self, timeout: float = 30.0) -> None:
        """Let the pump drain what is left in the pipe after the child exits, so the log is whole."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def _spawn(module: str, env: dict, stream: _ChildStream) -> subprocess.Popen:
    """Launch a child with its output piped into `stream`. `-u` stops the child block-buffering 8K at
    a time, which is what makes the stream live rather than arriving in bursts."""
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", module],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",  # device logs are not guaranteed UTF-8; a decode error must not stop the pump
        bufsize=1,
    )
    stream.start(proc)
    return proc


@contextlib.contextmanager
def _heartbeat(note):
    """Print `note()` every _HEARTBEAT_S for as long as the body runs, so a long wait can be told
    apart from a hang. The runner logs only twice between launch and readiness (its banner, then the
    descriptor), so the readiness wait is otherwise minutes of silence with the model load in it."""
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(_HEARTBEAT_S):
            print(f"[e2e {_elapsed()}] {note()}", flush=True)

    thread = threading.Thread(target=_beat, name="e2e-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def _readiness_gates() -> str:
    """Which of the three files the runner publishes to signal 'serving' are there yet."""
    return " ".join(
        f"{name}={'ok' if os.path.exists(path) else 'pending'}"
        for name, path in (("descriptor", _DESCRIPTOR), ("kv_table", TABLE_PATH), ("device_map", DEVMAP_PATH))
    )


@contextlib.contextmanager
def _running_runner(tag: str, sc: dict, **extra):
    """Spin up ONE runner (mock-migration, request mode) for a scenario and tear it down. Yields the
    live _ChildStream once it has published the H2D descriptor + KV table + device map (i.e. it is
    serving). `extra` layers additional env on top of the scenario's own (e.g. a generated prompt trace
    dir) -- same role as `_scenario_env`'s `extra`."""
    os.makedirs(_REPORT_DIR, exist_ok=True)
    log_path = os.path.join(_REPORT_DIR, f"ci_runner_{tag}.log")
    _cleanup_ipc()  # a stale table/descriptor from a prior scenario would make the readiness poll pass early
    env = _scenario_env(
        sc, PREFILL_MOCK_MIGRATION="1", PREFILL_ENABLE_LAYER_ACK="1", PREFILL_LAYER_ACK_D2H="1", **extra
    )
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
    stream = _ChildStream("runner", log_path)
    proc = _spawn(_RUNNER_MODULE, env, stream)
    print(f"[e2e {_elapsed()}] runner [{tag}] launched (pid={proc.pid}); its output follows, tagged", flush=True)
    try:
        deadline = time.monotonic() + ready_timeout_s
        with _heartbeat(lambda: f"runner [{tag}] not ready yet: {_readiness_gates()} | {stream.status()}"):
            while not (os.path.exists(_DESCRIPTOR) and os.path.exists(TABLE_PATH) and os.path.exists(DEVMAP_PATH)):
                if proc.poll() is not None:
                    stream.finish()
                    raise RuntimeError(
                        f"runner [{tag}] exited early (rc={proc.returncode}) during startup:\n{_tail(log_path)}"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"runner [{tag}] not ready within {ready_timeout_s}s:\n{_tail(log_path)}")
                time.sleep(2.0)
        print(f"[e2e {_elapsed()}] runner [{tag}] ready ({_readiness_gates()}); starting producer", flush=True)
        yield stream
    finally:
        died_rc = proc.poll()  # not None => the runner exited on its OWN, before our teardown signal
        if died_rc is None:
            proc.send_signal(signal.SIGINT)  # graceful; SIGKILL is the hard fallback
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=30)
        stream.finish()
        if not _STREAM_LOGS:  # otherwise it was already streamed live, line by line
            _emit_log_group(f"runner log [{tag}]", log_path)
        _cleanup_ipc()
        # A runner that died mid-run is a broken run, not a passing one, and nothing else makes the test
        # red: the producer reads KV that is already in DRAM, PCCs it, and exits 0. Only a NONZERO
        # self-exit counts, since with PREFILL_SEND_SHUTDOWN=1 the runner is *meant* to drain and exit 0;
        # and only with no exception in flight, so a real failure from the body is never masked by this.
        if died_rc not in (None, 0) and sys.exc_info()[0] is None:
            raise RuntimeError(f"runner [{tag}] died mid-run (rc={died_rc}); results are not trustworthy")


def _generate_prompt_trace(out_dir: str, isl: int, prompt_file: str, model: str) -> None:
    """Host-only pre-step: build a reference-KV trace dir from a prompt. The generator is device-free,
    so it runs in-process — no subprocess / OS command. A bad prompt or model surfaces as the
    generator's own exception, so a bad reference is never mistaken for a device PCC failure."""
    # Lazy import: the generator pulls in torch / transformers / the reference model, which must not
    # load at test-collection time.
    from models.demos.deepseek_v3_d_p.tt.runners.generate_prompt_trace import _load_prompt_text, generate

    generate(model, _load_prompt_text(None, prompt_file), isl, NUM_LAYERS, out_dir)


def _scenario_params():
    """One pytest param per scenario, each carrying a pytest-timeout budget matching its own limits.

    pytest.ini sets a blanket ``timeout = 300``, which is below what this test declares even for the
    small scenarios (_READY_TIMEOUT_S + _PRODUCER_TIMEOUT_S = 2100s) and nowhere near a full-depth
    model, which spends longer than that just loading weights. pytest-timeout reads the marker at
    setup time, so the bound has to be attached at collection rather than inside the test body. The
    real per-phase enforcement stays in _running_runner/producer.wait(); this only stops SIGALRM from
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
def test_producer_runner_pcc(scenario, tmp_path):
    """Spin up a fresh runner for the scenario, drive it with the producer, and require the per-slot
    KV PCC gate to pass (the producer exits non-zero if any resident slot is below threshold)."""
    sc = SCENARIOS[scenario]
    prod_log = os.path.join(_REPORT_DIR, f"ci_producer_{scenario}.log")
    trace_env = {}
    if "prompt_file" in sc:
        model = os.environ.get("PREFILL_MODEL", "kimi_k2_7")
        trace_env["PREFILL_MODEL"] = model
        reuse_dir = os.environ.get("PREFILL_REUSE_TRACE_DIR")
        if reuse_dir and os.path.exists(os.path.join(reuse_dir, "metadata.json")):
            trace_dir = reuse_dir
        else:
            trace_dir = str(tmp_path / "prompt_trace")
            _generate_prompt_trace(trace_dir, sc["isl"], sc["prompt_file"], model)
        trace_env["PREFILL_TRACE_DIR"] = trace_dir
    with _running_runner(scenario, sc, **trace_env) as runner_stream:
        env = _scenario_env(sc, PREFILL_PRODUCER_CHECK_PCC="1", **trace_env, **sc["producer"])
        producer_stream = _ChildStream("producer", prod_log)

        def _both_running() -> str:
            return f"producer [{scenario}] running | {producer_stream.status()} | {runner_stream.status()}"

        producer = None  # so the finally can tell "never spawned" from "spawned and still alive"
        try:
            producer = _spawn(_PRODUCER_MODULE, env, producer_stream)
            print(f"[e2e {_elapsed()}] producer [{scenario}] launched (pid={producer.pid})", flush=True)
            with _heartbeat(_both_running):
                returncode = producer.wait(
                    timeout=int(sc.get("producer_timeout_s", _PRODUCER_TIMEOUT_S))
                )  # raises TimeoutExpired
        finally:
            # Reap on EVERY exit where the producer is still alive, not just our own timeout: a
            # pytest-timeout signal or a Ctrl-C would otherwise leave it running, holding the transport
            # into the next scenario. subprocess.run() guaranteed this (bare `except: kill`) and the
            # rewrite has to as well. Kill BEFORE draining -- with a live pipe the pump is still blocked
            # in its read, so finish() would burn its whole join timeout before we got here.
            if producer is not None and producer.poll() is None:
                producer.kill()
                with contextlib.suppress(Exception):
                    producer.wait(timeout=30)
            producer_stream.finish()
            if not _STREAM_LOGS:  # otherwise it was already streamed live, line by line
                _emit_log_group(f"producer log [{scenario}]", prod_log)
        assert returncode == 0, (
            f"producer scenario {scenario!r} failed (rc={returncode}; PCC below threshold or error). "
            f"See the [producer]-tagged output above in this step log."
            # The runner tail is appended ONLY when the live stream is off. pytest renders an assertion
            # message three times on failure -- the ::error annotation, the FAILURES section, and (via
            # -rA in pytest.ini) the short test summary -- so with streaming on, a 200-line tail that is
            # already in the log verbatim would land in it four times over, ~800 lines of pure noise.
            + ("" if _STREAM_LOGS else f" Runner tail:\n{_tail(runner_stream.log_path)}")
        )
