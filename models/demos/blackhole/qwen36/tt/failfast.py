"""Fail fast when the device dies under a blocking event wait (serving robustness, rc1d).

The fused deferred serving path waits for each verify step with ``ttnn.event_synchronize`` on the reader thread
(spec_engine._def_data, called from the vLLM worker's async-output thread). tt-metal's EventSynchronize spins on the
host-side completed-event counter with no timeout. When the device hangs, tt-metal's completion-queue reader thread
detects it (TT_METAL_OPERATION_TIMEOUT_SECONDS), logs ``TT_THROW: TIMEOUT: device timeout, potential hang detected, the
device is unrecoverable`` and stores the exception for the next ``finish`` (synchronize_device). Nothing calls finish, so
the event never completes, the worker never answers sample_tokens, and EngineCore waits out the full RPC timeout
(rc1b d27u08 hang 2026-11-11 14:29Z: 5 min to EngineDeadError).

``watch(mesh, tag)`` wraps such a wait. A daemon watchdog thread (one per process, started on first use, wakes every
POLL_S) looks at the waits in progress; normal waits (milliseconds) are never touched, so the wait itself is unchanged.
When a wait has lasted STALL_S it probes the device once, on a separate thread, with ``ttnn.synchronize_device``: that
rethrows the stored TIMEOUT immediately on a dead device, or simply returns when the device is only busy.
- probe raises                    -> fatal: the device failed
- probe still blocked after HARD_S -> fatal: the device is unresponsive
- probe returned but the wait is still stuck GRACE_S later -> fatal (event never completes)
Fatal = log CRITICAL, then ``os._exit``: the stuck wait cannot be interrupted from Python (the C++ spin holds no GIL but
never returns), so the worker process exits. vLLM's executor monitor sees the worker die and fails EngineCore in seconds
(EngineDeadError) instead of after the RPC timeout. The device needs a reset either way.

Gates (environment):
  QWEN36_FAILFAST          1 (default) | report (log CRITICAL once, no probe, never exit) | 0 (off: no watchdog thread)
  QWEN36_FAILFAST_STALL_S  wait age that triggers the probe; default max(10, 2 * TT_METAL_OPERATION_TIMEOUT_SECONDS)
  QWEN36_FAILFAST_HARD_S   probe deadline; default 120 (tt-metal runs the timeout command, e.g. tt-triage, before it
                           stores the TIMEOUT, so the probe can legitimately wait that long)
Without TT_METAL_OPERATION_TIMEOUT_SECONDS (> 0) tt-metal never declares a hang, so mode 1 degrades to report: the hang-
capture harnesses (G's report-only watchdog, op timeout 0) keep the hung process alive and untouched for triage.
"""

import os
import threading
import time
from contextlib import contextmanager

POLL_S = 0.5
GRACE_S = 2.0
EXIT_CODE = 70  # EX_SOFTWARE


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return float(default)


def _op_timeout_s():
    return _env_float("TT_METAL_OPERATION_TIMEOUT_SECONDS", 0.0)


def mode():
    m = os.environ.get("QWEN36_FAILFAST", "1").strip().lower()
    if m in ("0", "off", "false", ""):
        return "off"
    if m == "report" or _op_timeout_s() <= 0:
        return "report"
    return "exit"


def _log(level, msg):
    try:
        from loguru import logger

        getattr(logger, level)(msg)
    except Exception:  # noqa: BLE001
        print(msg, flush=True)


class _Watchdog:
    def __init__(self):
        self.lock = threading.Lock()
        self.waits = {}  # token -> [t0, tag, mesh, probed]
        self.next = 0
        self.thread = None
        self.fired = False
        # test hooks: the probe and the fatal action (os._exit) are attributes so a test can observe them
        self.probe_fn = None
        self.exit_fn = os._exit

    def begin(self, mesh, tag):
        with self.lock:
            self.next += 1
            tok = self.next
            self.waits[tok] = [time.monotonic(), tag, mesh, False]
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name="qwen36-failfast", daemon=True)
                self.thread.start()
        return tok

    def end(self, tok):
        with self.lock:
            self.waits.pop(tok, None)

    def _run(self):
        while True:
            time.sleep(POLL_S)
            stall = _env_float("QWEN36_FAILFAST_STALL_S", max(10.0, 2.0 * _op_timeout_s()))
            now = time.monotonic()
            with self.lock:
                due = [(tok, w) for tok, w in self.waits.items() if not w[3] and now - w[0] >= stall]
                for _, w in due:
                    w[3] = True
            for tok, w in due:
                self._probe(tok, w, now - w[0])

    def _alive(self, tok):
        with self.lock:
            return tok in self.waits

    def _fatal(self, msg):
        m = mode()
        _log("critical", f"[failfast] {msg}" + ("; exiting the worker process (device needs a reset)" if m == "exit" else
                                                "; report only (QWEN36_FAILFAST=report or no operation timeout)"))
        if m == "exit" and not self.fired:
            self.fired = True
            try:
                import sys

                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:  # noqa: BLE001
                pass
            self.exit_fn(EXIT_CODE)

    def _probe(self, tok, w, age):
        t0, tag, mesh, _ = w
        if mode() != "exit":  # report: no device interaction (keeps a hung process untouched for triage)
            _log("critical", f"[failfast] '{tag}' event wait running {age:.1f} s: device may be hung (report only: "
                             "QWEN36_FAILFAST=report or no TT_METAL_OPERATION_TIMEOUT_SECONDS)")
            return
        _log("error", f"[failfast] '{tag}' event wait running {age:.1f} s: probing the device (synchronize_device)")
        res = {}

        def _p():
            try:
                if self.probe_fn is not None:
                    self.probe_fn(mesh)
                else:
                    import ttnn

                    ttnn.synchronize_device(mesh)
                res["ok"] = True
            except BaseException as e:  # noqa: BLE001 -- TT_THROW arrives as RuntimeError
                res["err"] = e

        th = threading.Thread(target=_p, name="qwen36-failfast-probe", daemon=True)
        th.start()
        th.join(_env_float("QWEN36_FAILFAST_HARD_S", 120.0))
        if "err" in res:
            e = res["err"]
            self._fatal(f"device error under '{tag}' (event wait {time.monotonic() - t0:.1f} s): {type(e).__name__}: "
                        f"{str(e).splitlines()[0] if str(e) else ''}")
            return
        if th.is_alive():
            self._fatal(f"device unresponsive under '{tag}': event wait {time.monotonic() - t0:.1f} s, synchronize_device "
                        f"still blocked after {_env_float('QWEN36_FAILFAST_HARD_S', 120.0):.0f} s")
            return
        time.sleep(GRACE_S)
        if self._alive(tok):
            self._fatal(f"'{tag}' event wait still blocked {time.monotonic() - t0:.1f} s although synchronize_device "
                        "returned (event can never complete)")
        else:
            _log("warning", f"[failfast] '{tag}' wait finished after the probe (slow device, no failure)")


_WD = _Watchdog()


@contextmanager
def watch(mesh, tag):
    """Wrap one blocking device wait (event_synchronize) on ``mesh``; ~1 us, no device work, unless it stalls."""
    if mode() == "off":
        yield
        return
    tok = _WD.begin(mesh, tag)
    try:
        yield
    finally:
        _WD.end(tok)
