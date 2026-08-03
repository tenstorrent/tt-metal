#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Launch the private LLK waveform debugger from the shared CodeGen checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_PRIVATE_ROOT = Path("/proj_sw/user_dev/llk_code_gen")
PRIVATE_ROOT_ENV = "LLK_CODEGEN_PRIVATE_ROOT"
# Preferred name first, pre-rename name second (llk_code_gen#97). A checkout can
# be at either revision, so accept both rather than coupling the two repositories
# to a lockstep merge — a mismatch would otherwise surface only as a fail-open
# "not available", which is the easiest failure to overlook.
PRIVATE_ENTRYPOINTS = (
    Path("tools/llk_wave_debug/codegen/scripts/llk_wave_debug.py"),
    Path("tools/llk_wave_debug/codegen/scripts/llk_debug.py"),
)


def main() -> int:
    private_root = Path(os.environ.get(PRIVATE_ROOT_ENV, DEFAULT_PRIVATE_ROOT))
    candidates = [private_root / relative for relative in PRIVATE_ENTRYPOINTS]
    entrypoint = next((path for path in candidates if path.is_file()), None)
    if entrypoint is None:
        expected = "\n".join(f"  {path}" for path in candidates)
        print(
            "LLK waveform debugger is not available.\n"
            f"Tried these private entry points:\n{expected}\n"
            f"Set {PRIVATE_ROOT_ENV} to a private llk_code_gen checkout.",
            file=sys.stderr,
        )
        return 2

    os.execv(
        sys.executable,
        [sys.executable, str(entrypoint), *sys.argv[1:]],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
