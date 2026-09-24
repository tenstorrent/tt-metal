# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Gate P2.2: the sharding plan fits per-chip DRAM at 55k context."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from models.demos.ernie45_d_p.bringup import metrics  # noqa: E402
from models.demos.ernie45_d_p.bringup.plan import plan  # noqa: E402

TASK = os.environ.get("ERNIE_BRINGUP_TASK", "P2.2")

p = plan()
for k, v in p["per_chip_bytes"].items():
    print(f"{k:28s} {v / 2**30:7.2f} GB")
print(f"{'TOTAL per chip':28s} {p['per_chip_total_gb']:7.2f} GB of {p['chip_dram_gb']:.0f} GB")
metrics.record(TASK, "plan_fits_dram", int(p["fits"]))
metrics.record(TASK, "per_chip_total_gb", round(p["per_chip_total_gb"], 2))
