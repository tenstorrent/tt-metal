"""Run a model/test command and capture its compiled kernels for analysis.

The most reliable way to know exactly which kernels a run produced is to control
where they are written. :class:`ModelRunner` executes the user's command with:

* ``TT_METAL_CACHE`` pointed at a dedicated (by default temporary) directory, so
  only this run's kernels live there; and
* ``TT_METAL_RISCV_DEBUG_INFO=1``, so the compute ELFs are compiled with the
  DWARF debug info the analyzer needs.

It then returns that cache directory for :class:`~.analyzer.LlkAnalyzer` to scan.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# tt-metal appends this subdirectory to whatever TT_METAL_CACHE points at.
_CACHE_SUBDIR = "tt-metal-cache"


@dataclass
class RunResult:
    """Outcome of running a model command."""

    cache_dir: Path  # directory to analyze (the TT_METAL_CACHE root we set)
    returncode: int
    is_temporary: bool


class ModelRunner:
    """Runs a model command in an environment instrumented for analysis."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        working_directory: str | Path | None = None,
        enable_debug_info: bool = True,
    ):
        self._explicit_cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._working_directory = Path(working_directory).expanduser() if working_directory else None
        self._enable_debug_info = enable_debug_info

    def run(self, command: str | list[str]) -> RunResult:
        """Execute ``command`` and return where its kernels were written.

        ``command`` may be a shell string (run via the shell) or an argv list.
        Output is streamed to this process's stdout/stderr so the user sees the
        model run live.
        """
        cache_dir, is_temporary = self._prepare_cache_dir()
        env = self._build_environment(cache_dir)

        print(f"[llk-analyzer] running model with TT_METAL_CACHE={cache_dir}", file=sys.stderr)
        if self._enable_debug_info:
            print("[llk-analyzer] TT_METAL_RISCV_DEBUG_INFO=1 (DWARF debug info enabled)", file=sys.stderr)
        print(f"[llk-analyzer] command: {command}", file=sys.stderr)

        completed = subprocess.run(
            command,
            shell=isinstance(command, str),
            env=env,
            cwd=str(self._working_directory) if self._working_directory else None,
            check=False,
        )

        if completed.returncode != 0:
            print(
                f"[llk-analyzer] warning: command exited with code {completed.returncode}; "
                "analyzing whatever kernels were produced",
                file=sys.stderr,
            )

        return RunResult(cache_dir=cache_dir, returncode=completed.returncode, is_temporary=is_temporary)

    def _prepare_cache_dir(self) -> tuple[Path, bool]:
        if self._explicit_cache_dir is not None:
            self._explicit_cache_dir.mkdir(parents=True, exist_ok=True)
            return self._explicit_cache_dir, False
        return Path(tempfile.mkdtemp(prefix="llk_analyzer_cache_")), True

    def _build_environment(self, cache_dir: Path) -> dict[str, str]:
        env = dict(os.environ)
        env["TT_METAL_CACHE"] = str(cache_dir)
        if self._enable_debug_info:
            env["TT_METAL_RISCV_DEBUG_INFO"] = "1"
        return env

    @staticmethod
    def cleanup(result: RunResult) -> None:
        """Remove a temporary cache directory created for a run."""
        if result.is_temporary and result.cache_dir.exists():
            shutil.rmtree(result.cache_dir, ignore_errors=True)
