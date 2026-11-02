# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Print per-zone cycle statistics from a device-profiler CSV produced on the simulator (calibration runs)."""
import json
import sys

from sim_profile import read_new_rows, reduce_all_zones, summarize_zones

rows, _ = read_new_rows(sys.argv[1], 0)
for s in summarize_zones(reduce_all_zones(rows)):
    print(json.dumps(s))
