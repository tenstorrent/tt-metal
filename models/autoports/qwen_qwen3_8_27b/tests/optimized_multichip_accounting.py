# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile signposted per-device tables with their own host measurement."""

import argparse
import csv
import gzip
import json
import re
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/optimized_multichip_decoder"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="+")
    args = parser.parse_args()
    records = []
    matmuls = []
    for name in args.profiles:
        report = json.loads((DOC / f"{name}.json").read_text())
        assert not report["baseline"] and report["mesh_shape"] == [1, 4]
        assert report["trace_bitwise_equal"] and report["post_decode_state_bitwise_equal"]
        policies = [(report["layer"], report["effective_policy"])]
        if report["stack"]:
            policies.append((3, report["next_layer_effective_policy"]))
        weights_per_device = kv_per_device = 0
        for layer, policy in policies:
            for role, k, n in (
                ("attention", 5120, 3584 if layer == 3 else 4160),
                ("output", 1536, 5120),
                ("gate", 5120, 8704),
                ("down", 4352, 5120),
            ):
                assert policy[role + "_dtype"] == "bfloat4_b"
                readers = policy[role + "_readers"]
                bank_tiles = (n + 256 * readers - 1) // (256 * readers) * readers
                weights_per_device += (k // 32) * bank_tiles * 8 * 576
            if layer == 3:
                assert policy["kv_dtype"] == "bfloat8_b"
                # Both K/V, one local KV head, 256 channels. Count the tiles
                # covering all logical positions consumed by this decode.
                kv_per_device += ((report["length"] + 1 + 31) // 32) * 8 * 1088 * 2 * report["batch"]
        devices = []
        raw_attributes = {}
        profile_dir = DOC / "tracy" / name
        raw_paths = list(profile_dir.glob("device*_ops.csv"))
        raw_paths += [path for path in profile_dir.glob("device*_ops.csv.gz") if not path.with_suffix("").exists()]
        for path in raw_paths:
            with gzip.open(path, "rt") if path.suffix == ".gz" else path.open() as stream:
                for row in csv.DictReader(stream):
                    if row.get("OP CODE") == "MatmulDeviceOperation":
                        raw_attributes[int(float(row["GLOBAL CALL COUNT"]))] = row["ATTRIBUTES"]
        for path in sorted((DOC / "tracy" / name).glob("device*_decode_perf_report.csv")):
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            kernel_us = sum(float(row["Device Time"] or 0) for row in rows)
            gaps_us = sum(float(row["Op-to-Op Gap"] or 0) for row in rows)
            devices.append(
                {
                    "device": int(rows[0]["Device"]),
                    "table": str(path.relative_to(DOC)),
                    "ops": len(rows),
                    "kernel_ms": kernel_us / 1000,
                    "gap_ms": gaps_us / 1000,
                    "span_ms": (kernel_us + gaps_us) / 1000,
                }
            )
            for row in rows:
                if row["OP Code"].startswith("MatmulDeviceOperation"):
                    assert row["Input 0 Datatype"] == "BFLOAT16", row
                    assert row["Input 1 Datatype"] == "BFLOAT4_B", row
                    assert "LoFi" in row["Math Fidelity"], row
                    attributes = raw_attributes[int(float(row["Global Call Count"]))]
                    readers = int(re.search(r"num_workers_per_dram_bank=(\d+)", attributes).group(1))
                    matmuls.append(
                        {
                            "profile": name,
                            **row,
                            "Runtime readers per bank": readers,
                            "Native compute worker count": 8 * readers,
                            "FLOPs percent with native worker count": float(row["FLOPs %"]) / readers,
                            "Runtime attributes": attributes,
                        }
                    )
        assert len(devices) == 4
        critical = max(devices, key=lambda device: device["span_ms"])
        host_ms = report["signposted_decode_host_ms"]
        assert 0 < critical["span_ms"] < host_ms * 1.2
        records.append(
            {
                "profile": name,
                "source_sha256": report["source_sha256"],
                "mesh": [1, 4],
                "batch": report["batch"],
                "prefill_length": report["length"],
                "stored_projection_read_bytes_per_device": weights_per_device,
                "kv_read_tile_bytes_per_device": kv_per_device,
                "aggregate_read_bytes": 4 * (weights_per_device + kv_per_device),
                "aggregate_dram_bandwidth_bytes_per_second": 4 * 512e9,
                "read_roofline_ms": (weights_per_device + kv_per_device) / 512e9 * 1000,
                "perf_tool_core_count_caveat": "Installed tt-perf-report fixes DRAM matmul compute count to8; native multi-reader uses16/24. Original tables retained; final_matmul_rows.csv adds runtime reader count and corrected compute utilization. Logical-bandwidth estimates omit bank/tile padding.",
                "roofline_scope": "Stored projection tiles including bank padding plus K/V read tiles; excludes recurrent state, constants, activations and writes. This is a lower bound, not measured bandwidth.",
                "devices": devices,
                "critical_device": critical["device"],
                "device_span_ms": critical["span_ms"],
                "signposted_host_ms": host_ms,
                "host_minus_device_span_ms": host_ms - critical["span_ms"],
                "same_run_warmed_decode_ms": report["decode_ms"],
                "same_run_warmed_prefill_ms": report["prefill_ms"],
            }
        )
    (DOC / "performance_accounting.json").write_text(json.dumps(records, indent=2) + "\n")
    if matmuls:
        with (DOC / "final_matmul_rows.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(matmuls[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(matmuls)
    boundary_records = []
    for name in args.profiles:
        report = json.loads((DOC / f"{name}.json").read_text())
        if not report["stack"]:
            continue
        for device in range(4):
            path = DOC / "tracy" / name / f"device{device}_decode_perf_report.csv"
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            # Full attention additionally normalizes Q and K on their head grids.
            # The four residual-stream norms use the selected 40-core contract.
            norms = [
                i
                for i, row in enumerate(rows)
                if row["OP Code"] == "LayerNormDeviceOperation" and int(row["Cores"]) == 40
            ]
            assert len(norms) == 4, (name, device, norms)
            start = norms[2]
            assert rows[start - 1]["OP Code"] == "BinaryNgDeviceOperation", (name, device)
            collectives = [row for row in rows if "AllReduceAsyncDeviceOperation" in row["OP Code"]]
            assert len(collectives) == 4, (name, device)
            boundary_records.append(
                {
                    "profile": name,
                    "source_sha256": report["source_sha256"],
                    "device": device,
                    "collectives": [{"op": row["OP Code"], "us": row["Device Time"]} for row in collectives],
                    "inter_layer_window": [
                        {"id": row["ID"], "op": row["OP Code"], "input_memory": row["Input 0 Memory"]}
                        for row in rows[start - 1 : start + 3]
                    ],
                    "verdict": "Final residual add feeds next input norm directly; no intervening collective or reshard.",
                }
            )
    if boundary_records:
        (DOC / "inter_layer_profile_audit.json").write_text(json.dumps(boundary_records, indent=2) + "\n")
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
