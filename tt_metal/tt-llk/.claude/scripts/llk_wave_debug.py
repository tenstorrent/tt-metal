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
PRIVATE_ENTRYPOINT = Path("tools/llk_wave_debug/codegen/scripts/llk_wave_debug.py")


def main() -> int:
    private_root = Path(os.environ.get(PRIVATE_ROOT_ENV, DEFAULT_PRIVATE_ROOT))
    entrypoint = private_root / PRIVATE_ENTRYPOINT
    if not entrypoint.is_file():
        print(
            "LLK waveform debugger is not available.\n"
            f"Expected private entry point: {entrypoint}\n"
            f"Set {PRIVATE_ROOT_ENV} to a private llk_code_gen checkout.",
            file=sys.stderr,
        )
        return 2

    # Forward arguments as an argv vector, never through a shell or generated code.
    os.execv(
        sys.executable,
        [sys.executable, str(entrypoint), *sys.argv[1:]],
    )
    return 127


if __name__ == "__main__":
    raise SystemExit(main())
