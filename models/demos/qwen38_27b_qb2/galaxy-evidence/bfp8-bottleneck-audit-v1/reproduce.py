import argparse
import csv
import gzip
import hashlib
import io
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = args.model
out = args.output
out.mkdir(exist_ok=False)
profiles = root / "galaxy-evidence/bfp8-priority-profiles-v2"
fields = [
    "DEVICE COMPUTE CB WAIT FRONT [ns]",
    "DEVICE COMPUTE CB RESERVE BACK [ns]",
    "NOC UTIL (%)",
    "DRAM BW UTIL (%)",
    "NPE CONG IMPACT (%)",
    "DEVICE KERNEL DURATION PER CORE MIN [ns]",
    "DEVICE KERNEL DURATION PER CORE MAX [ns]",
]
report = dict(
    scope="Reanalysis of completed warm eager two-layer BFP8 Tracy profiles; not traced full-model attribution or measured NoC utilization",
    sources={},
    profiles=[],
    bandwidth_model=[],
)
for path in sorted(profiles.rglob("profile-summary.json")):
    summary = json.loads(path.read_text())
    report["sources"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    p = next(path.parent.parent.rglob("ops_perf_results*.csv.gz"))
    raw = gzip.decompress(p.read_bytes())
    expected = next(v["sha256"] for k, v in summary["sources"].items() if k.startswith("ops_perf_results"))
    assert hashlib.sha256(raw).hexdigest() == expected
    report["sources"][str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    selected = []
    stack = []
    for row in csv.DictReader(io.StringIO(raw.decode())):
        name = row.get("OP CODE", "")
        if row.get("OP TYPE") == "signpost" and name.startswith("P0_"):
            if name.endswith("_BEGIN"):
                stack.append(name[:-6])
            elif name.endswith("_END"):
                assert stack.pop() == name[:-4]
        elif stack and row.get("DEVICE ID") not in ("", "-", None):
            selected.append(dict(row, stage=stack[-1]))
    assert not stack
    for device in summary["device_totals"]:
        rows = [r for r in selected if int(r["DEVICE ID"]) == device["device"]]
        assert len(rows) == device["device_op_rows"]
        assert abs(sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in rows) - device["kernel_ns"]) < 1
    stages = []
    for stage in sorted({r["stage"] for r in summary["inclusive_stages"]}):
        rows = [r for r in summary["inclusive_stages"] if r["stage"] == stage]
        assert len(rows) == 4
        stages.append(
            dict(
                stage=stage,
                kernel_us_median_rank=statistics.median(r["kernel_ns"] for r in rows) / 1000,
                device_op_rows=rows[0]["device_op_rows"],
            )
        )
    recurrence = []
    for device in summary["device_totals"]:
        rows = [
            r
            for r in selected
            if int(r["DEVICE ID"]) == device["device"]
            and r["stage"].endswith("_delta_recurrence")
            and r["OP CODE"] == "GenericOpDeviceOperation"
        ]
        recurrence.append(
            dict(device=device["device"], generic_op_us=[float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in rows])
        )
    report["profiles"].append(
        dict(
            name=path.parent.parent.name,
            op_rows=len(selected),
            missing_attribution_columns={
                field: sum(r.get(field) not in ("", "-", None, "N/A") for r in selected) for field in fields
            },
            inclusive_stages=stages,
            recurrence_generic_ops=recurrence,
        )
    )
projection = json.loads((profiles / "projection.json").read_text())
p = root / "galaxy-evidence/precision-perf-v1/bandwidth-estimate.json"
traffic = json.loads(p.read_text())
report["sources"][str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
report["assumptions"] = [
    "512 GB/s/chip assumed peak; one padded weight read per step; BFP8 KV; FP32 recurrent read/write; excludes compute, CCL, extra traffic, and launch costs",
    "Effective bandwidth is modeled useful bytes times measured TSU, not a DRAM counter reading",
    "Per-RISC timers include waits; phase and inclusive stage intervals must not be summed as independent costs",
]
for row in projection["rows"]:
    b = row["batch_per_tp4"]
    weights = traffic["streamed_weights_per_chip_bytes"]["bfloat8_b"]
    state = b * traffic["recurrent_read_write_per_user_chip_bytes"]
    kv = b * 32768 * traffic["kv_read_per_context_token_per_user_chip_bytes"]
    total = weights + state + kv
    tsu = row["baseline_measured_tsu"]
    candidate = row["shared_qk_plus_epilogue"]["projected_tsu"]
    report["bandwidth_model"].append(
        dict(
            batch_per_tp4=b,
            context=32768,
            weights_bytes=weights,
            state_bytes=state,
            kv_bytes=kv,
            total_bytes_per_chip_step=total,
            measured_native_tsu=tsu,
            ideal_512GBps_tsu=512e9 / total,
            modeled_effective_GBps=tsu * total / 1e9,
            modeled_fraction_of_ideal=tsu * total / 512e9,
            projected_fused_tsu=candidate,
            projected_fused_fraction_of_ideal=candidate * total / 512e9,
        )
    )
(out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report["bandwidth_model"], indent=2))
print("Verified", len(report["profiles"]), "profiles and raw CSV hashes")
