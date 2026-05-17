#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Legacy entry point for tt_hw_planner.

This script's logic was refactored into the `scripts/tt_hw_planner/` package
in v0.2.  This file is kept for backward compatibility — invoking it now
delegates to `python -m scripts.tt_hw_planner` with the same arguments.

The canonical invocation going forward is:
    python -m scripts.tt_hw_planner <model_id> [options]
"""

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    # Make `scripts/` importable so `tt_hw_planner` resolves.
    HERE = Path(__file__).resolve().parent
    sys.path.insert(0, str(HERE))

    from tt_hw_planner.cli import main

    sys.exit(main())
