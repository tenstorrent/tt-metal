# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer KV PCC and indexer top-k agreement against a Hy4 golden trace, recorded so drift over the
78 layers is visible (one row per layer, JSON + CSV), not just the min the producer gates on.

Three inputs, one report format::

    # golden vs another trace in the same layout (BF16 vs MXFP8 goldens, or a device capture written
    # with GoldenTraceWriter): KVPE halves, index keys, and top-k agreement per full-indexer layer
    python -m models.demos.deepseek_v3_d_p.reference.hy4.compare traces GOLDEN OTHER --out report

    # golden vs the producer's KV dumps (PREFILL_DUMP_KV_DIR=... PREFILL_PRODUCER_CHECK_PCC=1)
    python -m models.demos.deepseek_v3_d_p.reference.hy4.compare dump GOLDEN DUMP_DIR --slot 0 --out report

    # the producer's own per-layer log lines (KV nope/pe and index PCC), tabulated
    python -m models.demos.deepseek_v3_d_p.reference.hy4.compare log producer.log --out report

``--fail-under 0.99`` (default) makes the exit status non-zero when any layer's PCC is below it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.reference.hy4.golden import (
    golden_pe_rope,
    index_k_device_to_hf,
    kvpe_half_pccs,
    load_rows,
    pcc,
    read_metadata,
    topk_agreement,
)

PCC_KEYS = ("kvpe_nope", "kvpe_pe", "index_k")


def _kv_lora(meta) -> int:
    m = re.search(r"latent\) (\d+)", meta["frames"]["kvpe"]["layout"])
    return int(m.group(1)) if m else 512


def _rope_dim(meta) -> int:
    m = re.search(r"k_pe (\d+)", meta["frames"]["kvpe"]["layout"])
    return int(m.group(1)) if m else 64


def _index_k_hf(trace, meta, layer, n):
    k = load_rows(trace, f"dsa/indexer_k_layer_{layer}", f"indexer_k_layer_{layer}", 0, n).float()
    if meta["frames"]["index_k"]["frame"] != "device":
        return k
    return index_k_device_to_hf(k, _rope_dim(meta), golden_pe_rope(meta))


def compare_traces(golden, other, max_rows: int | None = None) -> list[dict]:
    gm, om = read_metadata(golden), read_metadata(other)
    n = min(gm["n_prompt_tokens"], om["n_prompt_tokens"], max_rows or 1 << 62)
    if gm["token_ids"][:n] != om["token_ids"][:n]:
        raise SystemExit(f"{golden} and {other} were captured from different prompts")
    if golden_pe_rope(gm) != golden_pe_rope(om):
        raise SystemExit(
            f"{golden} ({golden_pe_rope(gm)}) and {other} ({golden_pe_rope(om)}) were roped with different pairings"
        )
    kv_lora = _kv_lora(gm)
    rows = []
    layers = sorted(set(gm.get("layers_written", [])) & set(om.get("layers_written", [])))
    for layer in layers:
        key = f"kv_post_transform_layer_{layer}"
        g = load_rows(golden, f"kv_cache/layer_{layer}", key, 0, n).float()
        o = load_rows(other, f"kv_cache/layer_{layer}", key, 0, n).float()
        nope, pe = kvpe_half_pccs(g, o, kv_lora, rebase_golden_pe=False)
        row = {"layer": layer, "kvpe_nope": nope, "kvpe_pe": pe}
        if layer in gm["full_indexer_layers"] and layer in om["full_indexer_layers"]:
            if (Path(golden) / "dsa" / f"indexer_k_layer_{layer}").is_dir() and (
                Path(other) / "dsa" / f"indexer_k_layer_{layer}"
            ).is_dir():
                row["index_k"] = pcc(_index_k_hf(golden, gm, layer, n), _index_k_hf(other, om, layer, n))
            tk = f"dsa_topk_indices_layer_{layer}"
            if (Path(golden) / "dsa" / tk).is_dir() and (Path(other) / "dsa" / tk).is_dir():
                agree = topk_agreement(
                    load_rows(golden, f"dsa/{tk}", tk, 0, n), load_rows(other, f"dsa/{tk}", tk, 0, n)
                )
                row.update(
                    topk_agree_mean=float(agree.mean()),
                    topk_agree_min=float(agree.min()),
                    topk_agree_p01=float(agree.quantile(0.01)),
                )
        rows.append(row)
    return rows


def compare_dump(golden, dump_dir, slot: int = 0, golden_offset: int = 0) -> list[dict]:
    """Producer dumps (``device_kv_layer{L}_slot{S}.pt``): the compare window's rows, device frame
    (interleaved pe). ``golden_offset`` is the golden row the window starts at (PREFILL_PCC_GOLDEN_OFFSET). A current
    golden is interleaved too and compares directly; one written before the interleaved fix is re-based."""
    gm = read_metadata(golden)
    kv_lora = _kv_lora(gm)
    rebase = golden_pe_rope(gm) == "half_split"
    rows = []
    for f in sorted(
        Path(dump_dir).glob(f"device_kv_layer*_slot{slot}.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0])
    ):
        layer = int(re.match(r"device_kv_layer(\d+)_slot", f.name).group(1))
        dev = torch.load(f, weights_only=True).float()
        g = load_rows(
            golden, f"kv_cache/layer_{layer}", f"kv_post_transform_layer_{layer}", golden_offset, dev.shape[0]
        )
        nope, pe = kvpe_half_pccs(g.float(), dev, kv_lora, rebase_golden_pe=rebase)
        rows.append({"layer": layer, "kvpe_nope": nope, "kvpe_pe": pe, "rows": dev.shape[0]})
    return rows


_KV_RE = re.compile(r"slot (\d+) layer\s+(\d+) KV PCC: nope=([\d.eE+-]+) pe=([\d.eE+-]+)")
_IX_RE = re.compile(r"slot (\d+) layer\s+(\d+) index PCC: ([\d.eE+-]+)")


def parse_producer_log(path, slot: int | None = None) -> list[dict]:
    by_layer: dict[int, dict] = {}
    for line in Path(path).read_text(errors="replace").splitlines():
        if m := _KV_RE.search(line):
            s, layer = int(m.group(1)), int(m.group(2))
            if slot is None or s == slot:
                r = by_layer.setdefault(layer, {"layer": layer})
                r["kvpe_nope"] = min(r.get("kvpe_nope", 1.0), float(m.group(3)))
                r["kvpe_pe"] = min(r.get("kvpe_pe", 1.0), float(m.group(4)))
        elif m := _IX_RE.search(line):
            s, layer = int(m.group(1)), int(m.group(2))
            if slot is None or s == slot:
                r = by_layer.setdefault(layer, {"layer": layer})
                r["index_k"] = min(r.get("index_k", 1.0), float(m.group(3)))
    return [by_layer[k] for k in sorted(by_layer)]


def summarize(rows: list[dict], threshold: float) -> dict:
    below = [{"layer": r["layer"], k: r[k]} for r in rows for k in PCC_KEYS if k in r and r[k] < threshold]
    s = {"layers": len(rows), "threshold": threshold, "below_threshold": below}
    for k in PCC_KEYS + ("topk_agree_mean", "topk_agree_min"):
        vals = [(r[k], r["layer"]) for r in rows if k in r]
        if vals:
            v, layer = min(vals)
            s[f"min_{k}"] = {"value": v, "layer": layer}
    return s


def write_report(rows: list[dict], summary: dict, out: str | None) -> None:
    cols = ["layer"] + sorted({k for r in rows for k in r} - {"layer"}, key=lambda k: (k not in PCC_KEYS, k))
    print(" ".join(f"{c:>15}" for c in cols))
    for r in rows:
        print(" ".join(f"{r[c]:>15.6f}" if isinstance(r.get(c), float) else f"{str(r.get(c, '')):>15}" for c in cols))
    print(json.dumps(summary, indent=1))
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(f"{out}.json").write_text(json.dumps({"summary": summary, "layers": rows}, indent=1))
        with open(f"{out}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("traces")
    t.add_argument("golden")
    t.add_argument("other")
    t.add_argument("--max-rows", type=int, default=None)
    d = sub.add_parser("dump")
    d.add_argument("golden")
    d.add_argument("dump_dir")
    d.add_argument("--slot", type=int, default=0)
    d.add_argument("--golden-offset", type=int, default=0)
    lg = sub.add_parser("log")
    lg.add_argument("logfile")
    lg.add_argument("--slot", type=int, default=None)
    for sp in (t, d, lg):
        sp.add_argument("--out", default=None, help="report path prefix (.json and .csv are appended)")
        sp.add_argument("--fail-under", type=float, default=0.99)
    args = p.parse_args(argv)

    if args.mode == "traces":
        rows = compare_traces(args.golden, args.other, args.max_rows)
    elif args.mode == "dump":
        rows = compare_dump(args.golden, args.dump_dir, args.slot, args.golden_offset)
    else:
        rows = parse_producer_log(args.logfile, args.slot)
    if not rows:
        print("no layers to compare", file=sys.stderr)
        return 2
    summary = summarize(rows, args.fail_under)
    write_report(rows, summary, args.out)
    return 1 if summary["below_threshold"] else 0


if __name__ == "__main__":
    sys.exit(main())
