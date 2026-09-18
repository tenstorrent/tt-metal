#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Roll a MiniMax-M3 tracy ops CSV up into prefill zones — text report + optional JSON.

    python3 models/demos/minimax_m3/tests/perf/parse_zone_perf.py <ops_perf_results_*.csv> [--json out.json] [--top 5]

Shim over models/demos/common/prefill/profiling/parse_zone_perf.py with the M3 ZoneSpec.
"""

import sys

from models.demos.common.prefill.profiling.parse_zone_perf import main
from models.demos.minimax_m3.utils.profiler_utils import SPEC

if __name__ == "__main__":
    sys.exit(main(SPEC))
