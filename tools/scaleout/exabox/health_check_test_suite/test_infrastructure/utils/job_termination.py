# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Stop the whole run, diag suite included, when the job is cancelled.

The runner is PID 1 in its container, where the kernel drops signals left at
their default action, so an unhandled scancel is ignored and the suite keeps the
chips busy long after Slurm has released the node. Slurm can't reach the
container's processes itself; only docker's forwarded SIGTERM gets here.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

WATCHED_SIGNALS = (signal.SIGTERM, signal.SIGINT)

# Stopping must finish inside Slurm's KillWait (30s), after which the docker
# client is SIGKILLed and the container is left running.
DIAG_KILL_GRACE_SECONDS = 3

_terminating = threading.Event()


def terminating() -> bool:
    return _terminating.is_set()


def await_exit() -> None:
    """Block until the watcher has stopped the run and exited the process."""
    threading.Event().wait()


def _wait_and_stop(stop: Callable[[], object]) -> None:
    signo = signal.sigwait(set(WATCHED_SIGNALS))
    _terminating.set()
    log.warning("Received %s; stopping the run", signal.Signals(signo).name)
    try:
        stop()
    except Exception:
        log.exception("Could not stop the diag suite")
    os._exit(128 + signo)


def watch(stop: Callable[[], object]) -> None:
    """Call ``stop`` on SIGTERM/SIGINT, then exit.

    Must run on the main thread before other threads start: the signals are
    blocked process-wide so the watcher thread's ``sigwait`` receives them.
    """
    signal.pthread_sigmask(signal.SIG_BLOCK, set(WATCHED_SIGNALS))
    threading.Thread(target=_wait_and_stop, args=(stop,), name="termination-watch", daemon=True).start()
