# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bind final default evidence to current production sources after all jobs finish."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "doc/optimized_multichip_decoder"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="final_v2")
    collection = parser.parse_args().collection
    sources = [
        "tt/multichip_decoder.py",
        "tt/flat_ordered_post_loop.py",
        "tt/history_row_write.py",
        "tt/tiled_conv.py",
        "tt/partial_norm_rope.py",
        "tt/conv_gates.py",
        "tt/flat_ordered_conv_gates.py",
    ]
    sources += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tt/kernels").rglob("*")) if p.is_file()]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    completed = {row["label"]: row for row in map(json.loads, (D / "device_jobs.jsonl").read_text().splitlines())}
    artifacts = []
    lists = ["gates", "perf", "prefill", "profile", "recorded", "kernel"]
    optional = D / f"{collection}_context128k_answer_jobs.json"
    if optional.exists() and all(job["label"] in completed for job in json.loads(optional.read_text())):
        lists.append("context128k_answer")
    direct = []
    for name in lists:
        for job in json.loads((D / f"{collection}_{name}_jobs.json").read_text()):
            label = job["label"]
            actual = completed[label]
            assert actual["exit_code"] == 0, label
            assert all(actual["source_sha256"][key] == value for key, value in hashes.items()), label
            output = Path(job["command"][job["command"].index("--output") + 1])
            assert output.exists(), output
            if name == "kernel":
                direct.append({"job": label, "path": output.name})
                continue
            if "watcher" in label:
                category = "watcher"
            elif "large_boundaries" in label:
                category = "prefill_performance"
            elif "_boundary_" in label:
                category = "batch_matrix"
            elif "_contracts_" in label:
                category = "contracts"
            elif "_stress_" in label:
                category = "stress"
            elif "_max_context_" in label:
                category = "max_context"
            elif "_pool_" in label:
                category = "pool"
            elif label.startswith(collection + "_answer1024_32"):
                category = "long_answer"
            else:
                category = {
                    "perf": "performance",
                    "prefill": "prefill_performance",
                    "profile": "profile",
                    "recorded": "recorded_hf",
                }.get(name, "extended")
            artifacts.append(
                {
                    "job": label,
                    "path": output.name,
                    "category": category,
                    "tracking": job.get("env", {}).get("TT_METAL_TRACE_ALLOC_TRACKING") == "1",
                }
            )
    manifest = {
        "collection": collection,
        "production_sources": sources,
        "source_sha256": hashes,
        "artifacts": artifacts,
        "direct_kernel_artifacts": direct,
        "supplementary_boundary_evidence": "retained_short_prefill_evidence.json",
        "separate_timing_map": "review_timing_map.json",
        "profile_provenance": [
            f"tracy_{collection}/{kind}/{phase}_provenance.json" for kind in (0, 3) for phase in ("prefill", "decode")
        ]
        + [f"tracy_{collection}_target/stack32/decode_provenance.json"],
    }
    (D / "final_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("FINAL_MANIFEST_BOUND", len(artifacts), len(direct))


if __name__ == "__main__":
    main()
