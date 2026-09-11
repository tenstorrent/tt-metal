"""Prepare counterfactual routing replays; no accelerator/torch imports."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import struct


def prepare(source, output):
    raw = source.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size])
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"source": str(source), "sha256": hashlib.sha256(raw).hexdigest(), "cases": []}
    for layer in (18, 23):
        entry = header[f"expert_ids_layer_{layer}"]
        assert entry["dtype"] == "I32" and entry["shape"] == [8, 2560], entry
        start, end = entry["data_offsets"]
        original = list(struct.unpack("<20480i", raw[8 + header_size + start : 8 + header_size + end]))
        counts = Counter(original)
        assert all(0 <= e < 128 for e in original)
        # Exactly 16 logical expert slots per device; greedy placement preserves each
        # original expert's count, all token selections, and total assignments.
        bins, loads = [[] for _ in range(8)], [0] * 8
        for e in sorted(range(128), key=lambda e: (-counts[e], e)):
            d = min((d for d in range(8) if len(bins[d]) < 16), key=lambda d: (loads[d], d))
            bins[d].append(e)
            loads[d] += counts[e]
        mapping = {e: d * 16 + slot for d, experts in enumerate(bins) for slot, e in enumerate(experts)}
        assert sorted(mapping.values()) == list(range(128))
        shuffled = [original[i : i + 4] for i in range(0, 20480, 4)]
        random.Random(20260911).shuffle(shuffled)
        variants = {
            "captured": original,
            "placement_balanced": [mapping[e] for e in original],
            "source_shuffled": [e for token in shuffled for e in token],
            "uniform": [(i * 4 + k) % 128 for i in range(5120) for k in range(4)],
        }
        for mode, ids in variants.items():
            assert all(len(set(ids[i : i + 4])) == 4 for i in range(0, 20480, 4))
            c = Counter(ids)
            device = [sum(c[e] for e in range(d * 16, (d + 1) * 16)) for d in range(8)]
            traffic = [[0] * 8 for _ in range(8)]
            fanout = Counter()
            for t in range(5120):
                picks = ids[t * 4 : t * 4 + 4]
                fanout[len({e // 16 for e in picks})] += 1
                for e in picks:
                    traffic[t // 640][e // 16] += 1
            case = {
                "layer": layer,
                "mode": mode,
                "indices_shape": [8, 640, 4],
                "expert_counts": [c[e] for e in range(128)],
                "destination_assignments": device,
                "destination_max_mean": max(device) / 2560,
                "source_destination_assignments": traffic,
                "token_destination_fanout_histogram": dict(fanout),
                "old_to_new_expert_id": [mapping[e] for e in range(128)] if mode == "placement_balanced" else None,
                "indices": ids,
            }
            path = output / f"layer{layer}-{mode}.json"
            path.write_text(json.dumps(case) + "\n")
            evidence["cases"].append({k: v for k, v in case.items() if k != "indices"})
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
