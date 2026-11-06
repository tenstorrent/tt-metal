# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Offline RTL lower-bound triage; no device imports.

python quasar/dest_write_analysis.py path/to/dest_write_path_bank_stamps.jsonl
Bounds are based on the user's 2026-11-04 RTL answers, not independent RTL verification.
A large excess flags investigation; it does not establish a hardware discrepancy.
"""

import argparse
import json
from pathlib import Path


def hash_state(registers):
    if not registers:
        return "unknown"
    if not registers["ctrl"] & 1:
        return "disabled"
    if registers["mask"] == 0 and registers["match"] != 0:
        return "enabled_no_match"
    if registers["mask"] or registers["match"]:
        return "conditional_mapping"
    table = [registers[f"addr{i}"] for i in range(4, 12)]
    minimal = [1 << i for i in range(4, 12)]
    minimal[2] = 0x140
    dv = [0xA290, 0x5560, 0x5540, 0xAA80, 0x112110, 0x221220, 0x48480, 0x84840]
    if table == minimal:
        return "minimal_xor"
    if table == dv:
        return "dv_default"
    return "unverified_mapping"


def assess(record):
    state = hash_state(record.get("hash_registers"))
    srcs = record["path"].startswith("B-")
    input_bytes = 2048 if record["format"] == "Float16_b" else 4096
    output = record.get("out_format", record["format"])
    output_bytes = {"Float32": 4096, "Int32": 4096, "Float16_b": 2048}.get(output)
    # FP32 Dest bus stays 128 B/clk even when L1 input is narrower.
    bounds = {
        "unpack_transfer": input_bytes / 64 if srcs else max(32, input_bytes / 128)
    }
    if srcs:
        bounds["sfpu_copy_issue"] = 64
    if output_bytes is not None:
        bounds["pack_transfer"] = output_bytes / (64 if record.get("pack1") else 128)
        if record.get("writer_only"):
            bounds["pack_transfer"] /= 4 * record["tiles_per_bank"]
    floor = max(bounds.values())
    measured = record.get(
        "completed_cycles_per_tile",
        record["total_cycles"] / (record["num_banks"] * record["tiles_per_bank"]),
    )
    notes = []
    if state not in ("minimal_xor", "dv_default"):
        notes.append("Full-rate hashing not established; verify host/NoC mapping too")
    if not srcs and state in ("disabled", "enabled_no_match"):
        notes.append("Unhashed U0 FP32 transfer is expected near 64 cycles, not 32")
    if record["kernel"] != "Copy":
        notes.append("Transfer floor only: arithmetic issue/latency is additional")
    if record.get("producer_gap") or record.get("consumer_gap"):
        notes.append("Intentional injection gap; not eligible for no-stall comparison")
    elif record["kernel"] == "Copy" and measured >= 1.5 * floor:
        notes.append(
            "INVESTIGATE excess: separate bank/issue overhead, L1 conflicts and instrumentation"
        )
    if record.get("num_banks", 0) >= 16 and measured < floor * 0.95:
        notes.append(
            "INVESTIGATE below bound: verify clock window, tile count and completion"
        )
    if record.get("num_banks", 0) <= 2:
        notes.append("Startup sample; not steady state")
    return {
        "path": record["path"],
        "kernel": record["kernel"],
        "hash": state,
        "completed_cycles_per_tile": measured,
        "stage_transfer_floors": bounds,
        "pipeline_transfer_floor": floor,
        "notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stamps", type=Path)
    args = parser.parse_args()
    # Emit every observation; repeats and differing hash configurations are never overwritten.
    for line in args.stamps.read_text().splitlines():
        record = json.loads(line)
        if record["kernel"] == "Matmul":
            print(
                json.dumps(
                    {
                        "path": record["path"],
                        "note": "Matmul includes FPU/reconfiguration; inspect counter CSV",
                        "record": record,
                    }
                )
            )
        else:
            print(json.dumps({"assessment": assess(record), "record": record}))


if __name__ == "__main__":
    main()
