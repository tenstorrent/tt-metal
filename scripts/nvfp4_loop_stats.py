#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entry point for scripts/quant_loop_stats.py (format-agnostic)."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.quant_loop_stats import main, summarize  # noqa: F401

if __name__ == "__main__":
    main()
