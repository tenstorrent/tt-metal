# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from collections import defaultdict
from pathlib import Path


def category(sources):
    names = {Path(source).name for source in sources}
    if "ring_joint_sdpa.cpp" in names:
        return "sliding_sdpa" if any("halo" in name for name in names) else "global_sdpa"
    if any("/matmul/" in source for source in sources):
        return "matmuls"
    if "reader_update_padded_kv_cache.cpp" in names:
        return "kv_update"
    if "eltwise_typecast.cpp" in names:
        return "typecast"
    if names == {"reader_unary_interleaved_start_id_metal2.cpp", "writer_unary_interleaved_start_id_metal2.cpp"}:
        return "copies"
    if any("reduce_scatter" in name or name.startswith("unicast_") for name in names):
        return "other_collectives"
    return "other"


def summarize(path):
    report = json.loads(path.read_text())
    assert report["complete"], path
    result = {key: value for key, value in report.items() if key != "results"}
    result["source"] = str(path)
    result["results"] = []
    for row in report["results"]:
        compact = {key: value for key, value in row.items() if key not in ("captures", "program_profile")}
        if "program_profile" in row:
            capture = row["program_profile"]
            profile = capture["profile"]
            chip = capture["chips"][str(profile["chip_id"])]
            durations = defaultdict(float)
            counts = defaultdict(int)
            for start, end, source_id in profile["programs"]:
                name = category(profile["kernel_sources"][str(source_id)])
                durations[name] += (end - start) / chip["frequency_ghz"] / 1e6
                counts[name] += 1
            compact["profile"] = dict(
                chip_id=profile["chip_id"],
                span_ms=capture["device_us"] / 1000,
                summed_program_ms=sum(durations.values()),
                program_counts=dict(counts),
                category_ms=dict(durations),
            )
        result["results"].append(compact)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps([summarize(path) for path in args.inputs], indent=2) + "\n")
