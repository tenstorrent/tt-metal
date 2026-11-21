# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Keep a copy of the run's console output beside the run's other artifacts.

WHY THIS EXISTS. Everything a run produces is already filed per-model and per-run -- the report, the
profiles, the events, the state -- except the console output, which goes wherever the caller happened
to point stdout. So the one artifact that says WHY a run stopped is the one nothing owns: it lands in
a home directory, or a terminal, or nowhere. Voxtral run 41 exited after its baseline having tried no
lever, and the reason was unrecoverable, because the only copy of the log had been a shell redirect.

The run directory already answers "which model, which run" -- it is named for both -- so the log
belongs in it. Nothing here decides a path or a name: `Run.dir` does, exactly as it does for
RUN_REPORT.md.

BEST-EFFORT, ALWAYS. A log that cannot be opened must never cost the run it was recording, so every
failure falls back to the original stream untouched. The tee writes THROUGH to the real stream first,
so console behaviour is unchanged even if the file write fails mid-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

FILENAME = "console.log"


class _Tee:
    """Write to the real stream, then to the file. Never raises through to the caller."""

    def __init__(self, stream, fh):
        self._stream = stream
        self._fh = fh

    def write(self, data):
        n = self._stream.write(data)
        try:
            self._fh.write(data)
            self._fh.flush()
        except Exception:  # noqa: BLE001 -- the copy is a convenience; the stream is the contract
            pass
        return n

    def flush(self):
        try:
            self._stream.flush()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._fh.flush()
        except Exception:  # noqa: BLE001
            pass

    def __getattr__(self, name):
        # isatty, fileno, encoding, errors -- anything the wrapped stream exposes and something asks
        # for. Delegated rather than reimplemented so a caller cannot tell it is talking to a tee.
        return getattr(self._stream, name)


def install(run_dir) -> "callable | None":
    """Tee stdout and stderr into `run_dir`/console.log. Returns a restore callable, or None.

    Appends: the supervisor restarts the orchestrator in the same run directory, and a restart's
    output is part of the same story as the attempt that provoked it.
    """
    try:
        p = Path(run_dir) / FILENAME
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = open(p, "a", buffering=1, encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 -- no log is bad; a run that will not start over it is worse
        return None
    _out, _err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(_out, fh), _Tee(_err, fh)

    def restore():
        sys.stdout, sys.stderr = _out, _err
        try:
            fh.close()
        except Exception:  # noqa: BLE001
            pass

    return restore
