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

For single-rank usage, call `tt-triage` directly — `tt-run-triage` only wraps
multi-rank runs (just like `tt-run` itself requires a binding flag).

v1 supports the two legacy-mode tt-run binding flags:
    * `--rank-binding=<yaml>` (single rank-bindings file)
    * `--rank-bindings-mapping=<yaml>` (sub-context overlays merged into one
      global rank list)

New mode (`--mesh-graph-descriptor`) is deferred to v2. As a workaround, run
tt-run once with `--mesh-graph-descriptor` to populate the Phase 1 cache, then
pass `--rank-binding=generated/ttrun/<fingerprint>/rank_bindings.yaml` to
tt-run-triage.

Examples:
    tt-run-triage --rank-binding=foo.yaml -- --run=check_arc
    tt-run-triage --rank-bindings-mapping=mapping.yaml -- --llm-output
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

TRIAGE_PY = Path(__file__).resolve().parent / "triage" / "triage.py"

# prterun --tag-output line format: `[<jobid>,<rank>]<stream>: <payload>`
_TAG_RE = re.compile(r"^\[\d+,(\d+)\]<(stdout|stderr)>:\s?(.*)$")
# Triage script-section header: `script_name.py:` or `script_name.py [0.42s]:`
_SCRIPT_HEADER_RE = re.compile(r"^[a-zA-Z_]\w*\.py(?:\s+\[[\d.]+s\])?\s*:\s*$")


def _split_at_dashdash(argv: list[str]) -> tuple[list[str], list[str]]:
    if "--" in argv:
        i = argv.index("--")
        return argv[:i], argv[i + 1 :]
    return list(argv), []


def _parse_wrapper_flags(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("-h", "--help", action="store_true")
    return p.parse_known_args(argv)


def _extract_flag_value(tt_run_args: list[str], flag: str) -> Optional[str]:
    """Return the value of `--flag VALUE` or `--flag=VALUE` from `tt_run_args`, else None."""
    i = 0
    while i < len(tt_run_args):
        a = tt_run_args[i]
        if a == flag and i + 1 < len(tt_run_args):
            return tt_run_args[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
        i += 1
    return None


def _count_rank_binding_yaml(path: str) -> int:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)
    return len(data.get("rank_bindings", []))


def _count_rank_bindings_mapping_yaml(path: str) -> int:
    import yaml

    p = Path(path).resolve()
    with open(p) as f:
        data = yaml.safe_load(f)
    raw_map = (data or {}).get("subcontext_id_to_rank_bindings")
    if not raw_map:
        raise ValueError(f"{path}: missing or empty 'subcontext_id_to_rank_bindings'")
    total = 0
    for overlay_value in raw_map.values():
        overlay_path = Path(overlay_value)
        if not overlay_path.is_absolute():
            candidate = (p.parent / overlay_path).resolve()
            overlay_path = candidate if candidate.is_file() else overlay_path
        total += _count_rank_binding_yaml(str(overlay_path))
    return total


def _discover_rank_count(tt_run_args: list[str]) -> int:
    rb = _extract_flag_value(tt_run_args, "--rank-binding")
    if rb is not None:
        return _count_rank_binding_yaml(rb)
    rbm = _extract_flag_value(tt_run_args, "--rank-bindings-mapping")
    if rbm is not None:
        return _count_rank_bindings_mapping_yaml(rbm)
    if _extract_flag_value(tt_run_args, "--mesh-graph-descriptor") is not None:
        print(
            "Error: --mesh-graph-descriptor (tt-run new mode) is not supported in v1 of tt-run-triage.\n"
            "Run `tt-run --mesh-graph-descriptor=...` once to populate the Phase 1 cache,\n"
            "then point tt-run-triage at the generated bindings file:\n"
            "    tt-run-triage --rank-binding=generated/ttrun/<fingerprint>/rank_bindings.yaml -- ...",
            file=sys.stderr,
        )
        raise SystemExit(2)
    print(
        "Error: tt-run-triage requires one of --rank-binding=<yaml> or --rank-bindings-mapping=<yaml>.\n"
        "For single-rank triage, call `tt-triage` directly.",
        file=sys.stderr,
    )
    raise SystemExit(2)


class TextStreamingRenderer:
    """Buffers per-rank lines between script headers; renders each script's
    section once all N ranks have moved past it."""

    def __init__(self, expected_ranks: int):
        from rich.console import Console
        from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn

        self.N = expected_ranks

        out_width = None if sys.stdout.isatty() else 10000
        self.console = Console(
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

        self.rank_current_script: dict[int, str] = {}
        self.rank_current_lines: dict[int, list[str]] = {}
        self.seen_scripts: list[str] = []
        self.script_lines: dict[str, dict[int, list[str]]] = {}
        self.rendered_scripts: set[str] = set()

    def on_line(self, rank: int, payload: str) -> None:
        if _SCRIPT_HEADER_RE.match(payload.strip()):
            new_script = payload.strip().rstrip(":")
            self._flush_rank(rank)
            if new_script not in self.script_lines:
                self.seen_scripts.append(new_script)
                self.script_lines[new_script] = {}
            self.rank_current_script[rank] = new_script
            self.rank_current_lines[rank] = []
            self._pump()
        elif rank in self.rank_current_script:
            self.rank_current_lines[rank].append(payload)

    def _flush_rank(self, rank: int) -> None:
        if rank not in self.rank_current_script:
            return
        script = self.rank_current_script.pop(rank)
        lines = self.rank_current_lines.pop(rank)
        self.script_lines[script][rank] = lines

    def on_eof(self) -> None:
        # Final script of each rank has no "next header" — flush at exit.
        for rank in list(self.rank_current_script.keys()):
            self._flush_rank(rank)
        self._pump()

    def _pump(self) -> None:
        for script in self.seen_scripts:
            if script in self.rendered_scripts:
                continue
            finished = len(self.script_lines.get(script, {}))
            if finished < self.N:
                self._update_progress(script, finished)
                return
            self._render_script(script)
            self.rendered_scripts.add(script)
        self._update_progress(None, 0)

    def _render_script(self, script: str) -> None:
        self.console.print()
        self.console.print(f"{script}:", markup=False, highlight=False)
        for rank in sorted(self.script_lines[script].keys()):
            self.console.print(f"  [rank {rank}]", markup=False, highlight=False)
            for line in self.script_lines[script][rank]:
                self.console.print(line, markup=False, highlight=False)

    def _update_progress(self, script: Optional[str], finished: int) -> None:
        if script is not None:
            # reset() re-arms the spinner; otherwise completed==total marks it finished.
            self.progress.reset(self.ranks_task, total=self.N, description=script)
            self.progress.update(self.ranks_task, completed=finished)
        else:
            self.progress.update(self.ranks_task, description="(done)", completed=self.N)

    def finalize(self) -> None:
        self.progress.stop()


def _run_multi_rank(passthrough: list[str], tt_run_args: list[str]) -> int:
    rank_count = _discover_rank_count(tt_run_args)

    # Wrapper owns rendering / Live progress, so force-disable per-rank's.
    triage_cmd = [
        sys.executable,
        str(TRIAGE_PY),
        "--disable-progress",
        "--disable-colors",
    ] + passthrough

    cmd = ["tt-run"] + tt_run_args + triage_cmd
    print(f"[tt-run-triage] launching: {' '.join(cmd)}", file=sys.stderr)
    print(f"[tt-run-triage] expecting {rank_count} ranks", file=sys.stderr)

    renderer = TextStreamingRenderer(expected_ranks=rank_count)

    # Merge subprocess stderr into stdout so it doesn't fight the Live progress.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None

    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if not line:
                continue
            m = _TAG_RE.match(line)
            if m is None or m.group(2) == "stderr":
                continue
            renderer.on_line(int(m.group(1)), m.group(3))
        proc.wait()
        renderer.on_eof()
    finally:
        renderer.finalize()

    return proc.returncode


def _print_help() -> None:
    print(__doc__)


def main() -> int:
    wrapper_argv, passthrough = _split_at_dashdash(sys.argv[1:])
    ns, tt_run_args = _parse_wrapper_flags(wrapper_argv)

    if ns.help:
        _print_help()
        return 0

    return _run_multi_rank(passthrough, tt_run_args)


if __name__ == "__main__":
    sys.exit(main())
