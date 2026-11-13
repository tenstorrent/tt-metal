# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The public launcher tells the private tool which tt-llk checkout it was started from."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

TT_LLK = Path(__file__).resolve().parents[2]
LAUNCHER = TT_LLK / ".claude" / "scripts" / "llk_wave_debug.py"


def _fake_private_root(tmp_path: Path) -> Path:
    entry = (
        tmp_path
        / "tools"
        / "llk_wave_debug"
        / "codegen"
        / "scripts"
        / "llk_wave_debug.py"
    )
    entry.parent.mkdir(parents=True)
    entry.write_text(
        "import os, sys\n"
        "print(os.environ.get('LLK_CODEGEN_TT_LLK', '<unset>'))\n"
        "print(' '.join(sys.argv[1:]))\n"
    )
    return tmp_path


def _launch(tmp_path: Path, *args: str, **env: str) -> list[str]:
    full = {k: v for k, v in os.environ.items() if k != "LLK_CODEGEN_TT_LLK"}
    full["LLK_CODEGEN_PRIVATE_ROOT"] = str(_fake_private_root(tmp_path))
    full.update(env)
    out = subprocess.run(
        [sys.executable, str(LAUNCHER), *args],
        env=full,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return out.splitlines()


def test_launcher_exports_its_tt_llk_dir(tmp_path):
    lines = _launch(tmp_path, "perf-capture", "--help")
    assert Path(lines[0]) == TT_LLK
    assert lines[1] == "perf-capture --help"


def test_explicit_tt_llk_dir_is_kept(tmp_path):
    lines = _launch(tmp_path, "perf", LLK_CODEGEN_TT_LLK="/somewhere/else/tt-llk")
    assert lines[0] == "/somewhere/else/tt-llk"
