"""Offline, standard-library-only descriptive routing analysis; no timing join."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import struct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = args.routing.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + n])
    keys = {k for k in header if k != "__metadata__"}
    assert keys == {f"expert_ids_layer_{i}" for i in range(35)}
    layers = []
    for layer in range(35):
        entry = header[f"expert_ids_layer_{layer}"]
        assert entry["dtype"] == "I32" and entry["shape"] == [8, 2560]
        start, end = entry["data_offsets"]
        assert end - start == 81920
        ids = struct.unpack("<20480i", raw[8 + n + start : 8 + n + end])
        assert min(ids) >= 0 and max(ids) < 128
        assert all(len(set(ids[t : t + 4])) == 4 for t in range(0, len(ids), 4))
        counts = [ids.count(e) for e in range(128)]
        pp_load = [sum(counts[d * 16 : (d + 1) * 16]) for d in range(8)]
        tp_load = [sum(counts[d * 4 : (d + 1) * 4]) for d in range(32)]
        columns = [sum(counts[c * 32 : (c + 1) * 32]) for c in range(4)]
        pp_traffic = [[0] * 8 for _ in range(8)]
        tp_traffic = [[[0] * 8 for _ in range(8)] for _ in range(4)]
        for i, expert in enumerate(ids):
            source = i // 2560
            pp_traffic[source][expert // 16] += 1
            tp_traffic[expert // 32][source][(expert % 32) // 4] += 1
        assert sum(map(sum, pp_traffic)) == 20480
        assert sum(sum(map(sum, col)) for col in tp_traffic) == 20480
        layers.append(
            dict(
                layer=layer,
                expert_counts=counts,
                expert_max_mean=max(counts) / 160,
                pp_device_picks=pp_load,
                pp_max_mean=max(pp_load) / 2560,
                tp_device_picks_column_major=tp_load,
                tp_max_mean=max(tp_load) / 640,
                tp_column_picks=columns,
                tp_column_max_share=max(columns) / 20480,
                pp_source_destination_picks=pp_traffic,
                tp_column_source_destination_picks=tp_traffic,
                pp_remote_pick_share=1 - sum(pp_traffic[d][d] for d in range(8)) / 20480,
                tp_remote_pick_share=1 - sum(tp_traffic[c][d][d] for c in range(4) for d in range(8)) / 20480,
            )
        )
    result = dict(
        source=str(args.routing.resolve()),
        sha256=hashlib.sha256(raw).hexdigest(),
        interpretation="Descriptive Sept10 TP-captured routing; PP figures are counterfactual contiguous placement of same IDs, not measured PP routing. No historical timing correlation due to unmatched capture provenance.",
        mapping="ExpertMapping column-major: PP destination=e//16; TP column=e//32, destination=(e%32)//4. Source row=i//2560 assumes documented raw source-major capture order.",
        layers=layers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "b1_routing_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        "layer",
        "expert_max_mean",
        "pp_max_mean",
        "tp_max_mean",
        "tp_column_max_share",
        "pp_remote_pick_share",
        "tp_remote_pick_share",
    ]
    with (args.output_dir / "b1_routing_per_layer.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in layers)
    for i in (18, 23):
        r = layers[i]
        print({k: r[k] for k in fields})
        print("PP device picks:", r["pp_device_picks"], "TP column picks:", r["tp_column_picks"])
    for k in fields[1:]:
        lo = min(layers, key=lambda r: r[k])
        hi = max(layers, key=lambda r: r[k])
        print(k, "range", lo[k], hi[k], "layers", lo["layer"], hi["layer"])


if __name__ == "__main__":
    main()
