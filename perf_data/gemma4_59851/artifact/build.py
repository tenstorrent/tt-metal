"""Build the #59851 results page (claude.ai artifact) from sweep results.

    python perf_data/gemma4_59851/artifact/build.py <results_dir> [<second_machine_summary.csv>] [--out page.html]

<results_dir> is a run_matmul_chunk_sweep.sh output directory (summary.csv, fallback/summary.csv, layer_shares.csv).
Writes data.json next to this script and the page with the data embedded (template.html, __DATA__ placeholder).
"""
import argparse
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def config_kind(pc):
    if pc.startswith("MatmulMultiCoreReuseMultiCast1D"):
        return "1D"
    return "2D" if pc.startswith("MatmulMultiCoreReuseMultiCast") else "default"


def sweep_rows(path):
    out = []
    for r in csv.DictReader(open(path)):
        if not r.get("median_ns"):
            continue
        out.append(
            [
                r["op"],
                int(r["chunk"]),
                r["attn_fidelity"],
                r["math_fidelity"],
                int(r["cores"]),
                round(float(r["median_ns"]) / 1e3, 1),
                round(float(r["p25_ns"]) / 1e3, 1),
                round(float(r["p75_ns"]) / 1e3, 1),
                round(float(r["ns_per_token"]), 1),
                round(float(r["util_full_grid_pct"]), 1),
                round(float(r["weight_gbps"])),
                round(float(r["pcc"]), 5),
                config_kind(r["program_config"]),
                r["fp32_dest_acc"],
            ]
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results")
    p.add_argument("second", nargs="?", help="other machine's summary.csv, for the comparison columns")
    p.add_argument("--out", default=os.path.join(HERE, "gemma4_matmul_sweep.html"))
    args = p.parse_args()
    keys = [
        "layer",
        "chunk",
        "chunk_idx",
        "span_us",
        "matmul_us",
        "sdpa_us",
        "ccl_us",
        "qkv_us",
        "o_us",
        "gate_us",
        "up_us",
        "down_us",
    ]
    layers = [
        {k: (r[k] if k == "layer" else round(float(r[k]), 1)) for k in keys}
        for r in csv.DictReader(open(f"{args.results}/layer_shares.csv"))
    ]
    old = {}
    if args.second:
        for r in csv.DictReader(open(args.second)):
            if r["attn_fidelity"] == "prod" and r.get("median_ns"):
                old[f"{r['op']}|{r['chunk']}"] = round(float(r["median_ns"]) / 1e3, 1)
    data = dict(
        cols=["op", "C", "mode", "mf", "cores", "us", "p25", "p75", "nspt", "util", "gbps", "pcc", "cfg", "fp32"],
        sweep=sweep_rows(f"{args.results}/summary.csv"),
        fallback=sweep_rows(f"{args.results}/fallback/summary.csv"),
        layers=layers,
        old=old,
    )
    text = json.dumps(data, separators=(",", ":"))
    open(os.path.join(HERE, "data.json"), "w").write(text)
    open(args.out, "w").write(open(os.path.join(HERE, "template.html")).read().replace("__DATA__", text))
    print(
        f"{args.out}: {len(data['sweep'])} sweep rows, {len(data['fallback'])} fallback rows, {len(layers)} layer cells"
    )


if __name__ == "__main__":
    main()
