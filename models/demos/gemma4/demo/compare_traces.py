#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""SCRIPT 2 — compare device dumps (script 1) against the bit_sculpt GPU traces.

Both sides use the identical layout, so a comparison is a path join:

    decoder_io/decoder_input_layer_0/rows_{s:08d}_{e:08d}.safetensors
    decoder_io/decoder_output_layer_{i}/rows_*.safetensors
    kv_cache/layer_{i}/rows_*.safetensors

Metrics, per tensor:
  PCC    Pearson r, computed entirely in float64. DEVICE_USAGE.md warns that fp32 on a
         [8192, 5376] tensor can return values ABOVE 1.0 (they measured 1.0185) because
         the centred dot product overruns fp32's 24-bit significand.
  relL2  ||a-b|| / ||b||. PCC is scale-invariant; relL2 is not. Reported together because
         "PCC 0.998 and relL2 6%" describe the same data.

For KV the combined K||V stream is also split so K and V are scored separately --
K is post-k_norm and post-RoPE, V is post-v_norm and never RoPE'd, so they can fail
independently.

Usage:
    compare_traces.py --device <dump_root> --golden <trace_root> [--out report.md]
                      [--layers all|0,1,5] [--chunks all|0,1,2] [--skip-kv] [--skip-decoder]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch
from safetensors import safe_open

CHUNK = 8192
GEOMETRY = {"sliding_attention": (16, 256), "full_attention": (4, 512)}


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    a = a - a.mean()
    b = b - b.mean()
    d = a.norm() * b.norm()
    if d == 0:
        return float("nan")
    return float(torch.clamp((a @ b) / d, -1.0, 1.0))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.to(torch.float64).flatten()
    b = b.to(torch.float64).flatten()
    n = b.norm()
    return float("nan") if n == 0 else float((a - b).norm() / n)


def load(path: pathlib.Path, key: str):
    if not path.exists():
        return None
    with safe_open(str(path), "pt") as f:
        keys = list(f.keys())
        return f.get_tensor(key if key in keys else keys[0])


def rows_name(c: int) -> str:
    return f"rows_{c * CHUNK:08d}_{(c + 1) * CHUNK:08d}.safetensors"


def sel(raw: str, n: int):
    raw = (raw or "all").strip()
    return list(range(n)) if raw in ("all", "") else sorted({int(x) for x in raw.split(",") if x.strip()})


def summarize(rows):
    """rows: list of (pcc, relL2) -> (n, mean_pcc, min_pcc, max_pcc, mean_rel, max_rel)."""
    ok = [r for r in rows if r[0] == r[0]]  # drop NaN
    if not ok:
        return None
    p = torch.tensor([r[0] for r in ok], dtype=torch.float64)
    l = torch.tensor([r[1] for r in ok], dtype=torch.float64)
    return len(ok), p.mean().item(), p.min().item(), p.max().item(), l.mean().item(), l.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--out", default="/data/amilovanovic/TRACE_PCC_REPORT.md")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--chunks", default="all")
    ap.add_argument("--n-layers", type=int, default=60)
    ap.add_argument("--n-chunks", type=int, default=32)
    ap.add_argument("--skip-kv", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    a = ap.parse_args()

    dev, gold = pathlib.Path(a.device), pathlib.Path(a.golden)
    layers, chunks = sel(a.layers, a.n_layers), sel(a.chunks, a.n_chunks)

    # layer_types drives the KV geometry (sliding 16x256 vs global 4x512)
    lt = None
    for cand in list(gold.glob("*metadata.json")) + [gold / "metadata.json"]:
        if cand.exists():
            lt = json.loads(cand.read_text()).get("layer_types")
            break
    if lt is None:
        print("WARNING: no metadata.json; assuming every 6th layer (5,11,...) is global", file=sys.stderr)
        lt = ["full_attention" if (i % 6) == 5 else "sliding_attention" for i in range(a.n_layers)]

    per_layer, kv_per_layer, missing = {}, {}, []
    inp_rows = []

    # ── decoder_input_layer_0 ────────────────────────────────────────────────
    if not a.skip_decoder:
        for c in chunks:
            d = load(dev / "decoder_io" / "decoder_input_layer_0" / rows_name(c), "decoder_input_layer_0")
            g = load(gold / "decoder_io" / "decoder_input_layer_0" / rows_name(c), "decoder_input_layer_0")
            if d is None or g is None:
                missing.append(f"decoder_input_layer_0 chunk {c}")
                continue
            if d.shape != g.shape:
                missing.append(f"decoder_input_layer_0 chunk {c}: shape {tuple(d.shape)} vs {tuple(g.shape)}")
                continue
            inp_rows.append((pcc(d, g), rel_l2(d, g)))
        print(f"decoder_input_layer_0: {len(inp_rows)} chunks compared")

    # ── decoder_output_layer_{i} ─────────────────────────────────────────────
    if not a.skip_decoder:
        for i in layers:
            rows = []
            for c in chunks:
                key = f"decoder_output_layer_{i}"
                d = load(dev / "decoder_io" / key / rows_name(c), key)
                g = load(gold / "decoder_io" / key / rows_name(c), key)
                if d is None or g is None:
                    missing.append(f"{key} chunk {c}")
                    continue
                if d.shape != g.shape:
                    missing.append(f"{key} chunk {c}: shape {tuple(d.shape)} vs {tuple(g.shape)}")
                    continue
                rows.append((pcc(d, g), rel_l2(d, g)))
            if rows:
                per_layer[i] = rows
                s = summarize(rows)
                print(
                    f"decoder_output_layer_{i:2d}: n={s[0]:2d} pcc mean={s[1]:.6f} min={s[2]:.6f}  relL2 mean={s[4]:.4f}"
                )

    # ── kv_post_transform_layer_{i}, split into K and V ──────────────────────
    if not a.skip_kv:
        for i in layers:
            H, D = GEOMETRY[lt[i]] if i < len(lt) else GEOMETRY["sliding_attention"]
            krows, vrows = [], []
            for c in chunks:
                key = f"kv_post_transform_layer_{i}"
                d = load(dev / "kv_cache" / f"layer_{i}" / rows_name(c), key)
                g = load(gold / "kv_cache" / f"layer_{i}" / rows_name(c), key)
                if d is None or g is None:
                    missing.append(f"{key} chunk {c}")
                    continue
                if d.shape != g.shape:
                    missing.append(f"{key} chunk {c}: shape {tuple(d.shape)} vs {tuple(g.shape)}")
                    continue
                half = H * D
                krows.append((pcc(d[:, :half], g[:, :half]), rel_l2(d[:, :half], g[:, :half])))
                vrows.append((pcc(d[:, half:], g[:, half:]), rel_l2(d[:, half:], g[:, half:])))
            if krows:
                kv_per_layer[i] = (krows, vrows)
                sk, sv = summarize(krows), summarize(vrows)
                print(
                    f"kv layer {i:2d} ({lt[i][:8]}): K pcc={sk[1]:.6f} relL2={sk[4]:.4f} | "
                    f"V pcc={sv[1]:.6f} relL2={sv[4]:.4f}"
                )

    # ── report ───────────────────────────────────────────────────────────────
    out = pathlib.Path(a.out)
    L = []
    L.append("# Device vs GPU trace PCC report\n")
    L.append(f"- device dump: `{dev}`\n- golden: `{gold}`\n")
    L.append(f"- layers compared: {len(per_layer) or len(kv_per_layer)} | chunks: {len(chunks)}\n")
    L.append("\nPCC and relL2 both computed in float64. relL2 = ||dev-gold||/||gold||.\n")
    L.append("The golden is bf16 and self-agrees to only ~0.998 PCC / ~6% relL2, so ~0.998 is the ceiling.\n")

    if inp_rows:
        s = summarize(inp_rows)
        L.append("\n## decoder_input_layer_0 (embedding output)\n\n")
        L.append("| n | pcc mean | pcc min | pcc max | relL2 mean | relL2 max |\n|---|---|---|---|---|---|\n")
        L.append(f"| {s[0]} | {s[1]:.6f} | {s[2]:.6f} | {s[3]:.6f} | {s[4]:.4f} | {s[5]:.4f} |\n")

    if per_layer:
        L.append("\n## decoder_output_layer_{i} — per layer, aggregated over chunks\n\n")
        L.append("| layer | type | n | pcc mean | pcc min | pcc max | relL2 mean | relL2 max |\n")
        L.append("|---|---|---|---|---|---|---|---|\n")
        for i in sorted(per_layer):
            s = summarize(per_layer[i])
            L.append(
                f"| {i} | {lt[i] if i < len(lt) else '?'} | {s[0]} | {s[1]:.6f} | {s[2]:.6f} | "
                f"{s[3]:.6f} | {s[4]:.4f} | {s[5]:.4f} |\n"
            )

    if kv_per_layer:
        # mean AND min/max: a mean over 32 chunks hides a single bad chunk, which is
        # exactly the kind of thing worth seeing (chunk 1 was an outlier end-to-end).
        L.append("\n## kv_post_transform_layer_{i} — K and V scored separately\n\n")
        L.append("Aggregated over chunks. `pcc min`/`relL2 max` expose per-chunk outliers a mean would bury.\n\n")
        L.append(
            "| layer | type | n | K pcc mean | K pcc min | K pcc max | K relL2 mean | K relL2 max "
            "| V pcc mean | V pcc min | V pcc max | V relL2 mean | V relL2 max |\n"
        )
        L.append("|---" * 13 + "|\n")
        for i in sorted(kv_per_layer):
            k, v = kv_per_layer[i]
            sk, sv = summarize(k), summarize(v)
            L.append(
                f"| {i} | {lt[i] if i < len(lt) else '?'} | {sk[0]} "
                f"| {sk[1]:.6f} | {sk[2]:.6f} | {sk[3]:.6f} | {sk[4]:.4f} | {sk[5]:.4f} "
                f"| {sv[1]:.6f} | {sv[2]:.6f} | {sv[3]:.6f} | {sv[4]:.4f} | {sv[5]:.4f} |\n"
            )

        # per-chunk matrices, so a bad (layer, chunk) cell is locatable
        ls = sorted(kv_per_layer)
        for which, label in ((0, "K"), (1, "V")):
            L.append(f"\n## Per-chunk detail, KV {label} (pcc)\n\n")
            L.append("| chunk | " + " | ".join(f"L{i}" for i in ls) + " |\n")
            L.append("|---" * (len(ls) + 1) + "|\n")
            for ci, c in enumerate(chunks):
                cells = []
                for i in ls:
                    r = kv_per_layer[i][which]
                    cells.append(f"{r[ci][0]:.5f}" if ci < len(r) else "-")
                L.append(f"| {c} | " + " | ".join(cells) + " |\n")

    if per_layer:
        L.append("\n## Per-chunk detail, decoder output (pcc)\n\n")
        ls = sorted(per_layer)
        L.append("| chunk | " + " | ".join(f"L{i}" for i in ls) + " |\n")
        L.append("|---" * (len(ls) + 1) + "|\n")
        for ci, c in enumerate(chunks):
            cells = []
            for i in ls:
                r = per_layer[i]
                cells.append(f"{r[ci][0]:.5f}" if ci < len(r) else "-")
            L.append(f"| {c} | " + " | ".join(cells) + " |\n")

    if missing:
        L.append(f"\n## Missing / mismatched ({len(missing)})\n\n")
        for m in missing[:60]:
            L.append(f"- {m}\n")
        if len(missing) > 60:
            L.append(f"- ... and {len(missing) - 60} more\n")

    out.write_text("".join(L))
    print(f"\nreport -> {out}   ({len(missing)} missing/mismatched)")


if __name__ == "__main__":
    main()
