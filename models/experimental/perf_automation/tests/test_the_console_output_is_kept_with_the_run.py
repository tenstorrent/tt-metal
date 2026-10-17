# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The log that says WHY a run stopped belongs with the run, not in a shell redirect.

Every other artifact is filed in run.dir -- named for the model and the run -- except the console
output, which went wherever the caller pointed stdout. Voxtral run 41 exited after its baseline
having tried no lever, and the reason was unrecoverable: the only copy was a redirect that had been
deleted. A log nothing owns is a log that is not there when it is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
for _p in (str(_PA), str(_PA.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent.console_log import FILENAME, install  # noqa: E402


def test_both_streams_are_kept(tmp_path):
    restore = install(tmp_path)
    assert restore is not None
    try:
        print("on stdout")
        print("on stderr", file=sys.stderr)
    finally:
        restore()
    txt = (tmp_path / FILENAME).read_text()
    assert "on stdout" in txt and "on stderr" in txt, txt


def test_the_real_stream_still_gets_everything(tmp_path, capsys):
    """The file is a COPY. Teeing must not swallow the console -- the terminal is what a human is
    watching, and the run's own supervisor parses this output."""
    restore = install(tmp_path)
    try:
        print("visible")
    finally:
        restore()
    assert "visible" in capsys.readouterr().out


def test_a_log_that_cannot_be_opened_does_not_touch_the_streams(tmp_path):
    """No log is bad; a run that will not start because of one is worse."""
    before_out, before_err = sys.stdout, sys.stderr
    assert install(tmp_path / "nope" / "\0bad") is None
    assert sys.stdout is before_out and sys.stderr is before_err


def test_a_restart_appends_rather_than_truncates(tmp_path):
    """The supervisor restarts the orchestrator in the SAME run directory, and a restart's output is
    part of the same story as the attempt that provoked it."""
    r = install(tmp_path)
    print("first attempt")
    r()
    r = install(tmp_path)
    print("second attempt")
    r()
    txt = (tmp_path / FILENAME).read_text()
    assert "first attempt" in txt and "second attempt" in txt, txt


def test_it_is_installed_where_the_run_directory_is_known():
    """Pinned on the source: the tee has to go in before the first banner, or the setup output --
    which is where a refusal is explained -- is written before anyone is recording."""
    src = (_PA / "agent" / "before_loop.py").read_text()
    i = src.index("run = Run.create(")
    j = src.index("_install_console_log(run.dir)", i)
    k = src.index('print(f"run: {run.run_id}', i)
    assert i < j < k, "the console log is installed after output has already been written"
