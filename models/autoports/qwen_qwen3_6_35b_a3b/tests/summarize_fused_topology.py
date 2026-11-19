# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Export the measured fused operation sequence, including layout transitions."""

import csv
import os
from pathlib import Path

ROOT = Path(__file__).parents[1]
DOC = Path(os.environ.get("QWEN_FUSED_ARTIFACT_DIR", ROOT / "doc/fused_decoder"))


def tensor(row, prefix):
    shape = "x".join(row.get(f"{prefix}_{dim}_PAD[LOGICAL]", "") for dim in "WZYX")
    return " ".join([shape] + [row.get(f"{prefix}_{key}", "") for key in ("DATATYPE", "LAYOUT", "MEMORY")])


def main():
    lines = [
        "# Measured fused operation topology",
        "",
        "Each sequence is one warmed prefill or the first measured decode replay. "
        "Shapes preserve the profiler's padded[logical] notation. "
        "When profiler operation-info caching is enabled, tensor metadata may describe the first call "
        "of a shape-independent program, not this execution. Do not infer runtime bytes or FLOPs from "
        "cached shapes. Call, program-hash and replay identifiers preserve the raw record association. "
        "Authoritative per-call shapes require a collection with TT_METAL_PROFILER_NO_CACHE_OP_INFO=1.",
        "",
    ]
    for layer in (0, 3):
        directory = DOC / f"tracy/layer_{layer}"
        raw = list(csv.DictReader((directory / "ops.csv").open()))
        for phase in ("prefill", "decode"):
            signpost = f"PERF_{phase.upper()}"
            start = next(i for i, r in enumerate(raw) if r["OP CODE"] == signpost)
            end = next(i for i, r in enumerate(raw) if r["OP CODE"] == signpost + "_END")
            rows = [r for r in raw[start + 1 : end] if r["DEVICE ID"]]
            if phase == "decode":
                session = rows[0]["METAL TRACE REPLAY SESSION ID"]
                rows = [r for r in rows if r["METAL TRACE REPLAY SESSION ID"] == session]
            assert rows
            records = [
                dict(
                    step=i + 1,
                    op=r["OP CODE"],
                    global_call_count=r["GLOBAL CALL COUNT"],
                    program_hash=r["PROGRAM HASH"],
                    trace_replay_session=r["METAL TRACE REPLAY SESSION ID"],
                    program_cache_hit=r["PROGRAM CACHE HIT"],
                    input_0=tensor(r, "INPUT_0"),
                    input_1=tensor(r, "INPUT_1"),
                    output_0=tensor(r, "OUTPUT_0"),
                    device_us=float(r["DEVICE KERNEL DURATION [ns]"]) / 1000,
                )
                for i, r in enumerate(rows)
            ]
            path = directory / f"{phase}_op_sequence.csv"
            with path.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=records[0])
                writer.writeheader()
                writer.writerows(records)
            lines += [
                f"## Layer {layer}, {phase}",
                "",
                f"Full operands: [{path.name}]({path.relative_to(DOC)}).",
                "",
                "| Step | Device operation | Input 0 | Output 0 | μs |",
                "|---:|---|---|---|---:|",
            ]
            lines += [
                f"| {r['step']} | {r['op']} | {r['input_0']} | {r['output_0']} | {r['device_us']:.3f} |"
                for r in records
            ]
            lines.append("")
    (DOC / "op_topology.md").write_text("\n".join(lines))
    print("FUSED_TOPOLOGY_WRITTEN")


if __name__ == "__main__":
    main()
