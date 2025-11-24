#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Runner script for the Specification Validator Tool.

Usage:
    python run_validator.py list
    python run_validator.py validate dropout
    python run_validator.py show-spec dropout
"""

import sys
from pathlib import Path

# Add the _dev_TODO.now directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from validator.cli import main

if __name__ == "__main__":
    main()
