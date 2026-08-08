#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
tt-run-triage — multi-rank wrapper around tools/triage/triage.py.

Usage:
    tt-run-triage TT_RUN_OPTIONS... [-- TRIAGE_FLAGS...]

Everything before `--` is forwarded verbatim to `tt-run`. Everything after `--`
is forwarded verbatim to `triage.py` (one invocation per rank).

Run `tt-run-triage --help` for wrapper options, or `tt-run-triage -- --help` for
triage flags.

For single-rank usage, call `tt-triage` directly — `tt-run-triage` only wraps
multi-rank runs (just like `tt-run` itself requires a binding flag).

The rank count is discovered at runtime with a
`tt-run ... printenv OMPI_COMM_WORLD_SIZE`, so every tt-run mode works with no
binding-specific parsing.

Examples:
    tt-run-triage --rank-binding=foo.yaml -- --run=check_arc
    tt-run-triage --mesh-graph-descriptor=mesh.textproto --hosts=h0,h1 -- --llm-output
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from tools.triage import utils

TRIAGE_PY = Path(__file__).resolve().parent / "triage" / "triage.py"

# tt-run runs triage under `mpirun --tag-output`, prefixing each line: `[<jobid>,<rank>]<stream>: <payload>`
_TAG_RE = re.compile(r"^\[\d+,(\d+)\]<(stdout|stderr)>:\s?(.*)$")
# Triage script-section header: `script_name.py:` or `script_name.py [0.42s]:` (group 1 = filename).
_HEADER_LINE_RE = re.compile(r"^([A-Za-z_]\w*\.py)(?:\s+\[[\d.]+s\])?\s*:\s*$")
# SGR escapes — stripped before header matching (per-rank output is colored).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _probe_rank_count(tt_run_args: list[str]) -> int:
    """Discover the rank count by launching a `tt-run ... printenv OMPI_COMM_WORLD_SIZE`"""
    cmd = ["tt-run", *tt_run_args, "printenv", "OMPI_COMM_WORLD_SIZE"]
    print("[tt-run-triage] counting ranks...", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for raw in proc.stdout.splitlines():
        m = _TAG_RE.match(raw.strip())
        val = (m.group(3) if m else raw).strip()
        if val.isdigit():
            return int(val)
    sys.stderr.write(
        f"[tt-run-triage] rank count probe failed (exit {proc.returncode}); output:\n{proc.stdout}{proc.stderr}\n"
    )
    raise SystemExit(2)


class TextStreamingRenderer:
    """Renders each triage script's per-rank output in canonical execution order."""

    def __init__(self, expected_ranks: int, scripts: dict):
        from rich.console import Console
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
        import triage

        self.N = expected_ranks
        self.order = [s.name for s in triage.resolve_execution_order(scripts)]
        self.order_index = {name: i for i, name in enumerate(self.order)}
        self.is_provider = {s.name: bool(s.config.data_provider) for s in scripts.values()}

        # `lines[script][rank]` is a rank's output for a script; `current[rank]` is
        # the script it is mid-emitting; `next_row` streams down the fixed order.
        self.lines: dict[str, dict[int, list[str]]] = {name: {} for name in self.order}
        self.current: dict[int, str] = {}
        self.next_row = 0

        out_width = None if sys.stdout.isatty() else 10000
        self.console = Console(
            theme=utils.create_console_theme(False),
            highlight=False,
            width=out_width,
            file=sys.stdout,
        )
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} ranks"),
            console=self.console,
            transient=True,
        )
        self.progress.start()
        self.ranks_task = self.progress.add_task("(waiting for first record)", total=self.N)

    def on_line(self, rank: int, payload: str) -> None:
        header = _HEADER_LINE_RE.match(_ANSI_RE.sub("", payload).strip())
        if header and header.group(1) in self.lines:
            self.current[rank] = header.group(1)
            self.lines[header.group(1)][rank] = []
            self._advance()
        elif rank in self.current:
            self.lines[self.current[rank]][rank].append(payload)

    def on_eof(self) -> None:
        self._advance(final=True)

    def _past(self, rank: int, row: int) -> bool:
        # The rank is emitting a script after `row`, so it is done with `row`.
        cur = self.current.get(rank)
        return cur is not None and self.order_index[cur] > row

    def _advance(self, final: bool = False) -> None:
        while self.next_row < len(self.order):
            done = sum(self._past(rank, self.next_row) for rank in range(self.N))
            if done < self.N and not final:
                self._progress(self.order[self.next_row], done)
                return
            self._render(self.order[self.next_row])
            self.next_row += 1
        self._progress(None, self.N)

    def _render(self, script: str) -> None:
        reported = self.lines[script]
        if self.is_provider.get(script):
            if not reported:
                return  # provider succeeded on every rank; nothing to report
            self.console.print()
            self.console.print(f"{script}:", markup=False, highlight=False)
            # Providers print only on failure/skip, so reported ranks are the ones with output.
            for rank in sorted(reported):
                self.console.print(f"  [rank {rank}]", markup=False, highlight=False)
                self._print_lines(reported[rank])
            rest = self.N - len(reported)
            if rest:
                self.console.print(f"  ({rest} rank(s): no failure reported)", markup=False, highlight=False)
            return
        self.console.print()
        self.console.print(f"{script}:", markup=False, highlight=False)
        for rank in range(self.N):
            self.console.print(f"  [rank {rank}]", markup=False, highlight=False)
            if rank in reported:
                self._print_lines(reported[rank])
            else:
                self.console.print("    (no output - rank stopped before this script)", markup=False, highlight=False)

    def _print_lines(self, lines: list[str]) -> None:
        from rich.text import Text

        for line in lines:
            self.console.print(Text.from_ansi(line), highlight=False)  # pre-colored by the subprocess

    def _progress(self, script: Optional[str], done: int) -> None:
        if script is not None:
            # reset() re-arms the spinner; otherwise completed==total marks it finished.
            self.progress.reset(self.ranks_task, total=self.N, description=script)
            self.progress.update(self.ranks_task, completed=done)
        else:
            self.progress.update(self.ranks_task, description="(done)", completed=self.N)

    def finalize(self) -> None:
        self.progress.stop()


def _run_multi_rank(passthrough: list[str], tt_run_args: list[str]) -> int:
    sys.path.insert(0, str(TRIAGE_PY.parent))
    import triage

    scripts = triage.TriageScript.discover_all_in_directory(str(TRIAGE_PY.parent))
    triage.parse_arguments(scripts, argv=["--disable-progress", *passthrough])

    rank_count = _probe_rank_count(tt_run_args)
    print(f"[tt-run-triage] found {rank_count} ranks", file=sys.stderr)

    triage_cmd = [
        sys.executable,
        str(TRIAGE_PY),
        "--disable-progress",
    ] + passthrough

    cmd = ["tt-run"] + tt_run_args + triage_cmd
    print(f"[tt-run-triage] running triage on {rank_count} ranks", file=sys.stderr)
    print(f"[tt-run-triage] launching: {' '.join(cmd)}", file=sys.stderr)

    renderer = TextStreamingRenderer(expected_ranks=rank_count, scripts=scripts)

    interactive = sys.stdout.isatty()
    cols = shutil.get_terminal_size().columns if interactive else 10000
    # ASCII table borders: multi-byte box-drawing chars corrupt mpirun's line buffer in multi-rank runs.
    env = {
        **os.environ,
        "COLUMNS": str(cols),
        "TT_TRIAGE_COLOR": "1" if interactive else "0",
        "TT_TRIAGE_SIMPLE_TABLE": "1",
    }

    # Buffer the stderr/stdout lines and flush after progress stops so they don't fight the Live display.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
    assert proc.stdout is not None

    stderr_lines: list[str] = []
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            m = _TAG_RE.match(line)
            if m is None or m.group(2) == "stderr":
                stderr_lines.append(line)
                continue
            renderer.on_line(int(m.group(1)), m.group(3))
        proc.wait()
        renderer.on_eof()
    finally:
        renderer.finalize()
        if stderr_lines and proc.returncode != 0:
            sys.stderr.write("\n".join(stderr_lines) + "\n")

    return proc.returncode


def main() -> int:
    # Everything before `--` goes to tt-run; everything after goes to triage.py.
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        wrapper_argv, passthrough = argv[:i], argv[i + 1 :]
    else:
        wrapper_argv, passthrough = argv, []

    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-h", "--help", action="store_true")
    ns, tt_run_args = p.parse_known_args(wrapper_argv)

    if ns.help:
        print(__doc__)
        return 0

    return _run_multi_rank(passthrough, tt_run_args)


if __name__ == "__main__":
    sys.exit(main())
