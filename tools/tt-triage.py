#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import os

# Add the triage directory to Python path so local modules can be imported
script_dir = os.path.dirname(os.path.abspath(__file__))
triage_dir = os.path.join(script_dir, "triage")
if triage_dir not in sys.path:
    sys.path.insert(0, triage_dir)

from triage import main

if __name__ == "__main__":
    main()
