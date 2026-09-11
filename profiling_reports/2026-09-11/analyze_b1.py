"""Reproduce historical PP B1 evidence with strict eight-device op grouping; no hardware.
Usage: python3 analyze_b1.py --source-root /path/to/original/tt-metal --output-dir DIR
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

MOE = {
    "CombineDeviceOperation",
    "DispatchDeviceOperation",
    "UnifiedRoutedExpertFfnDeviceOperation",
    "PostCombineReduceDeviceOperation",
    "MaskedBincountDeviceOperation",
    "TopKDeviceOperation",
    "OffsetCumsumDeviceOperation",
}
MLA = {
    "RingJointSDPADeviceOperation",
    "NlpCreateHeadsDeviceOperation",
    "NLPConcatHeadsDeviceOperation",
    "RotaryEmbeddingIndexedDeviceOperation",
    "UpdatePaddedKvCacheDeviceOperation",
    "ZeroPaddedKvCacheDeviceOperation",
    "SoftmaxDeviceOperation",
}


def analyze(path, rank):
    seen = collections.Counter()
    totals = collections.defaultdict(lambda: collections.defaultdict(float))
    layer = None
    batch = []

    def flush():
        if not batch:
            return
        assert len(batch) == 8 and len({r["DEVICE ID"] for r in batch}) == 8, (path, layer, len(batch))
        counts = [int(r["GLOBAL CALL COUNT"]) for r in batch]
        assert sorted(counts) == list(range(min(counts), min(counts) + 8))
        totals[(layer, seen[layer] - 1)][batch[0]["OP CODE"]] += (
            max(float(r["DEVICE KERNEL DURATION [ns]"]) for r in batch) / 1e6
        )
        batch.clear()

    with path.open() as fh:
        for row in csv.DictReader(fh):
            op = row["OP CODE"]
            start = re.fullmatch(r"forward_layer_(\d+)_start", op)
            if start:
                flush()
                assert layer is None
                layer = int(start[1]) + rank * 9
                seen[layer] += 1
            elif re.fullmatch(r"forward_layer_\d+_end", op):
                flush()
                layer = None
            elif layer is not None and op in MOE | MLA and row["DEVICE ID"]:
                if batch and (batch[0]["OP CODE"] != op or row["DEVICE ID"] in {r["DEVICE ID"] for r in batch}):
                    flush()
                batch.append(row)
            else:
                flush()
    flush()
    assert len(seen) == 9 and set(seen.values()) == {10}, seen
    result = []
    for layer in sorted(seen):
        chunks = [totals[layer, c] for c in range(1, 10)]
        assert all(MOE <= set(c) for c in chunks)
        values = {op: statistics.mean(c[op] for c in chunks) for op in MOE}
        result.append(
            {
                "layer": layer,
                "rank": rank,
                "kept_passes": 9,
                "moe_ms": sum(values.values()),
                "mla_ms": statistics.mean(sum(v for op, v in c.items() if op in MLA) for c in chunks),
                **{op + "_ms": v for op, v in sorted(values.items())},
            }
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    sources, layers = [], []
    for rank in range(4):
        matches = list(
            (args.source_root / f"mistral4_perf_profile/pp4_deep36/rank{rank}/reports").glob("*/ops_perf_results_*.csv")
        )
        assert len(matches) == 1, matches
        path = matches[0]
        sources.append(
            {"path": str(path.resolve()), "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest()}
            if hasattr(hashlib, "file_digest")
            else {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
        layers.extend(analyze(path, rank))
    low, high = min(layers, key=lambda x: x["moe_ms"]), max(layers, key=lambda x: x["moe_ms"])
    by_layer = {r["layer"]: r for r in layers}
    delta = {op: by_layer[18][op + "_ms"] - by_layer[23][op + "_ms"] for op in MOE}
    summary = {
        "metric": "sum of per-operation maximum device kernel durations; mean of nine retained passes; not layer elapsed time",
        "sources": sources,
        "min_layer": low["layer"],
        "min_moe_ms": low["moe_ms"],
        "max_layer": high["layer"],
        "max_moe_ms": high["moe_ms"],
        "spread_percent": 100 * (high["moe_ms"] / low["moe_ms"] - 1),
        "same_rank_L18_minus_L23_ms": delta,
        "dispatch_combine_share_of_moe_gap": (delta["DispatchDeviceOperation"] + delta["CombineDeviceOperation"])
        / sum(delta.values()),
        "layers": layers,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b1_evidence.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.output_dir / "b1_per_layer.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(layers[0]))
        writer.writeheader()
        writer.writerows(layers)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"sources", "layers"}}, indent=2))


if __name__ == "__main__":
    main()
