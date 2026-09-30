#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render a MiniMax-M3 zone profile: text summary + self-contained HTML report.

    python3 models/demos/minimax_m3/tests/perf/visualize_zones.py <ops_perf_results_*.csv> [-o report.html] [--open]

Shim over models/demos/common/prefill/profiling/visualize_zones.py with the M3 ZoneSpec.
"""

import sys

from models.demos.common.prefill.profiling.visualize_zones import main
from models.demos.minimax_m3.utils.profiler_utils import SPEC

if __name__ == "__main__":
    sys.exit(main(SPEC))
