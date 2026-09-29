"""Same-policy layer-stack estimate from a two-layer full-path profile."""

import argparse
import csv
import json
import lzma
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    opener = lzma.open if args.csv.suffix == ".xz" else open
    windows = {}
    active = None
    with opener(args.csv, "rt") as f:
        for row in csv.DictReader(f):
            if row["OP TYPE"] == "signpost":
                active = None if row["OP CODE"].endswith("_END") else row["OP CODE"].removeprefix("PERF_").lower()
            elif active and row.get("DEVICE FW START CYCLE"):
                windows.setdefault(active, {}).setdefault(row["DEVICE ID"], []).append(row)

    def span(rows):
        return (
            max(int(r["DEVICE FW END CYCLE"]) for r in rows) - min(int(r["DEVICE FW START CYCLE"]) for r in rows)
        ) / 1350

    ranks = {}
    for rank, rows in windows["model"].items():
        norms = [i for i, r in enumerate(rows) if r["OP CODE"] == "LayerNormDeviceOperation"]
        assert len(norms) == 5, "Expected two real decoder layers and one final norm"
        ranks[rank] = {
            "layer0_bfp8_mlp_us": span(rows[norms[0] : norms[2]]),
            "layer1_bfp4_mlp_us": span(rows[norms[2] : norms[4]]),
            "embedding_rope_us": span(rows[: norms[0]]),
            "terminal_and_position_us": span(rows[norms[4] :]),
            "sampling_us": span(windows["sampling"][rank]),
        }
    maxima = {name: max(r[name] for r in ranks.values()) for name in next(iter(ranks.values()))}
    stack = (9 * maxima["layer0_bfp8_mlp_us"] + 27 * maxima["layer1_bfp4_mlp_us"]) / 1000
    terminal = sum(maxima[k] for k in ["embedding_rope_us", "terminal_and_position_us", "sampling_us"]) / 1000
    result = {
        "source": str(args.csv),
        "per_rank": ranks,
        "maximum_rank_us": maxima,
        "layer_class_counts": {"bfp8_mlp": 9, "bfp4_mlp": 27},
        "selected_policy_stack_ms": stack,
        "selected_policy_stack_tps": 1000 / stack,
        "terminal_ms": terminal,
        "stack_plus_terminal_ms": stack + terminal,
        "historical_stage5_stack_ms": 36 * 0.238776,
        "scope": "Estimate from real layers0/1 at the profile context; complete per-rank FW spans including gaps. No full36 profile or full36 device-time measurement is claimed. Historical stage5 timing used a rejected full-model precision policy.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
