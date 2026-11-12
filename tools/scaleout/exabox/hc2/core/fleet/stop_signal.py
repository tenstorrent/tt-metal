# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""stop_signal.py: SIGTERM/SIGINT end a device stage through its own finally blocks, which close the mesh
(as h3/synth_gen.py's _StopSignal). The tt-device-mcp broker TERMs a job's process group and KILLs it 10 s
later; run.sh's rs_dying TERMs each pg_run group and waits up to RS_DIE_S-1 s before its KILL. Without a handler Python
dies at the TERM with the device still open.
  install(tag)  before the open: a signal is only recorded (an open in progress finishes, so its mesh gets closed)
  arm()         first line inside the try whose finally closes the mesh: raises at once for a recorded signal, then
                each signal raises StopSignal there (handlers set again: an open may have replaced them)
  closing()     context manager around the close itself: a signal there is only recorded, so the close
                is not cut; once the close is done it raises StopSignal for it (exit 143/130, not the normal exit code);
                a close that raises after a signal is logged and still exits 143/130
The first signal raises once; later ones are ignored, so the close is not cut. stderr gets '<tag> SIGNAL <name> <epoch>'
at the signal and '<tag> SIGNAL-EXIT <name> <s> after the signal' at exit; an uncaught StopSignal exits 143 (TERM) or
130 (INT) after the finally blocks ran, not a Python traceback."""
import atexit
import os
import signal
import sys
import time
import traceback

SIGS = (signal.SIGTERM, signal.SIGINT)
_st = {"tag": "STAGE", "armed": False, "got": None, "raised": False, "phase": "open"}


class StopSignal(KeyboardInterrupt):
    """KeyboardInterrupt, so no 'except Exception' in a stage takes the stop for an op error; finally blocks still run."""


def _raise(name):
    _st["raised"] = name
    for s in SIGS:
        signal.signal(s, signal.SIG_IGN)
    t0 = time.time()
    print(f"{_st['tag']} SIGNAL {name} {t0:.0f} closing the device", file=sys.stderr, flush=True)
    atexit.register(lambda: print(f"{_st['tag']} SIGNAL-EXIT {name} {time.time() - t0:.1f} after the signal",
                                  file=sys.stderr, flush=True))
    raise StopSignal(name)


def _on_signal(signum, _frame):
    name = signal.Signals(signum).name
    if not _st["armed"]:
        if _st["got"] is None:
            _st["got"] = name
            msg = "closing once it is open" if _st["phase"] == "open" else "exiting once it is closed"
            print(f"{_st['tag']} SIGNAL {name} {time.time():.0f} during the {_st['phase']}: {msg}",
                  file=sys.stderr, flush=True)
        return
    _raise(name)


def _hook(tp, e, tb):
    if issubclass(tp, StopSignal):
        try:
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass
        atexit._run_exitfuncs()
        os._exit(143 if str(e) == "SIGTERM" else 130)
    sys.__excepthook__(tp, e, tb)


def install(tag):
    _st["tag"] = tag
    sys.excepthook = _hook
    for s in SIGS:
        signal.signal(s, _on_signal)


def arm():
    _st["armed"] = True
    for s in SIGS:
        signal.signal(s, _on_signal)
    if _st["got"] is not None:
        _raise(_st["got"])


class closing:
    """with ss.closing(): close the device. Disarmed inside, so a TERM/INT during the close is only recorded; at the end
    of the block a recorded, not yet raised signal raises StopSignal (a TERM mid-close would cut the close)."""

    def __enter__(self):
        self.prev = _st["armed"], _st["phase"]
        _st["armed"], _st["phase"] = False, "close"
        return self

    def __exit__(self, tp, e, tb):
        _st["armed"], _st["phase"] = self.prev   # after the close a later signal raises as before (fabric_repro reads on)
        sig = _st["raised"] or _st["got"]
        if tp is not None and not issubclass(tp, StopSignal) and sig:   # the close failed after a signal
            print(f"{_st['tag']} SIGNAL {sig}: the close failed: {tp.__name__}: {e}", file=sys.stderr, flush=True)
            traceback.print_exception(tp, e, tb, file=sys.stderr)
            if _st["raised"]:
                raise StopSignal(sig) from None   # exit 143/130 for the signal, not 1 for the close error
        if _st["got"] is not None and not _st["raised"] and (tp is None or not issubclass(tp, StopSignal)):
            _raise(_st["got"])
        return False
