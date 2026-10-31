"""Device timing for isolated reader trials, explicitly separate from layer rooflines."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from models.autoports.google_gemma_4_31b_it.tests.summarize_multichip_profile import extract, window

p = argparse.ArgumentParser()
p.add_argument("csv")
p.add_argument("host")
p.add_argument("output")
a = p.parse_args()
rows = list(csv.DictReader(Path(a.csv).open()))
data = json.loads(Path(a.host).read_text())
for record in data["records"]:
    samples = []
    for repeat in range(4):
        label = f'READER_{record["role"]}_{record["readers"]}_{repeat}'
        groups = defaultdict(list)
        for row in extract(rows, label, label + "_END"):
            if row.get("DEVICE FW START CYCLE"):
                groups[(row["METAL TRACE ID"], row["METAL TRACE REPLAY SESSION ID"])].append(row)
        assert len(groups) == 8, (label, len(groups))
        assert all(len(group) == 4 for group in groups.values())
        samples.append(statistics.mean(window(group)[0] for group in groups.values()))
    duration = statistics.mean(samples)
    record.update(
        device_us=duration,
        alternating_repeat_device_us=samples,
        repeat_spread_pct=100 * (max(samples) - min(samples)) / duration,
        physical_weight_gb_per_s=record["physical_weight_bytes"] / duration / 1000,
        logical_weight_gb_per_s=record["logical_weight_bytes"] / duration / 1000,
        physical_weight_peak_pct=100
        * record["physical_weight_bytes"]
        / (duration / 1e6 * record["peak_dram_bytes_per_s"]),
    )
data["evidence"] = a.csv
data[
    "denominator"
] = "Isolated one-op device FW window, maximum across four devices, 32 traced repetitions; per-device weight-only bandwidth, not layer roofline."
Path(a.output).write_text(json.dumps(data, indent=2) + "\n")
lines = [
    "# Isolated alternating reader trials",
    "",
    data["denominator"],
    "",
    "| Role | Readers | Device us | Physical GB/s | Peak % | Spread % | PCC |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for r in data["records"]:
    lines.append(
        f'| {r["role"]} | {r["readers"]} | {r["device_us"]:.3f} | {r["physical_weight_gb_per_s"]:.2f} | {r["physical_weight_peak_pct"]:.2f} | {r["repeat_spread_pct"]:.2f} | {r["pcc"]:.8f} |'
    )
Path(a.output).with_suffix(".md").write_text("\n".join(lines) + "\n")
