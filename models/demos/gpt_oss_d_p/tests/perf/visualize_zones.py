#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Render a GPT-OSS zone profile: text summary + self-contained HTML report.

    python3 models/demos/gpt_oss_d_p/tests/perf/visualize_zones.py <ops_perf_results_*.csv> [-o report.html] [--open]

Shim over models/demos/common/prefill/profiling/visualize_zones.py with the GPT-OSS ZoneSpec.
"""

import sys

from models.demos.common.prefill.profiling.visualize_zones import main
from models.demos.gpt_oss_d_p.utils.profiler_utils import SPEC

if __name__ == "__main__":
    sys.exit(main(SPEC))
