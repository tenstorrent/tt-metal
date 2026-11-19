# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Add source-checked stage-5 traffic accounting to the standard perf report."""

import argparse
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tests.optimized_multichip_capacity import decode_lower_bound, layer_inventory

p = argparse.ArgumentParser()
p.add_argument("--result", type=Path, required=True)
p.add_argument("--perf", type=Path, required=True)
a = p.parse_args()
r = json.loads(a.result.read_text())
path = a.perf / "accounting.json"
report = json.loads(path.read_text())
kind = "linear_attention" if r["layer"] == 0 else "full_attention"
policy = r["selected_policy"]
inventory = layer_inventory(kind, policy)
for name, matrix in inventory["matrices"].items():
    actual = r["matrix_allocations"][name]
    assert actual["padded_shape"] == matrix["physical_shape"], (name, actual, matrix)
    assert actual["dtype"].split(".")[-1].lower() == matrix["dtype"], (name, actual, matrix)
traffic = decode_lower_bound(kind, r["batch"], [r["length"] + 1] * r["batch"], policy)
report.setdefault("legacy_weight_only_roofline_ms", report["roofline_ms"])
report["traffic_lower_bound"] = traffic
report["matrix_inventory_verified_against_run"] = inventory
report["weight_read_bytes_per_device"] = traffic["matrix_read_once_bytes"]
report["kv_read_bytes_per_device"] = traffic["visible_kv_tile_read_bytes"]
report["recurrent_read_write_bytes_per_device"] = traffic["recurrent_read_write_bytes"]
report["roofline_ms"] = traffic["roofline_ms"]
report["roofline_fraction_of_host"] = traffic["roofline_ms"] / report["same_run_host_ms"]
report["limitations"] = [v for v in report["limitations"] if "omits intermediate" not in v]
report["limitations"] = [v for v in report["limitations"] if not v.startswith("Compulsory traffic")]
report["limitations"].append(
    "Compulsory traffic lower bound includes physical matrix tiles once and unavoidable recurrent state read/write or visible KV reads; excludes intermediate movement, repeated reads, allocator slack, unused resident copies and protocol overhead."
)
path.write_text(json.dumps(report, indent=2) + "\n")
print(
    json.dumps(
        {"kind": kind, "traffic_lower_bound": traffic, "roofline_fraction_of_host": report["roofline_fraction_of_host"]}
    )
)
