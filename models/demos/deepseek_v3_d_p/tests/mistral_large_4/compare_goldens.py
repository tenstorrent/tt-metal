# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PCC between two cpu_golden.py outputs of the same prompt (e.g. unrounded vs --experts-dtype bfloat4_b), CPU only:

    python -m models.demos.deepseek_v3_d_p.tests.mistral_large_4.compare_goldens A B [--json out.json]

Per layer: kv_cache/layer_<i>.safetensors (kv_latent, k_rope, index_key; the --state golden), then every tensor both
dirs dumped in chunk_<c>.safetensors (L<i> hidden states, final_norm, logits). Prints PCC and the relative L2 error
||b - a|| / ||a||; tensors with no elements are skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

KV = ("kv_latent", "k_rope", "index_key")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = (a.norm() * b.norm()).item()
    return float("nan") if den == 0 else (a @ b).item() / den


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    n = a.double().norm().item()
    return float("nan") if n == 0 else (b.double() - a.double()).norm().item() / n


def _get(path: Path, key: str):
    with safe_open(str(path), "pt") as f:
        return f.get_tensor(key) if key in f.keys() else None


def compare(a: Path, b: Path) -> dict:
    out = {"kv": {}, "dumps": {}}
    layers = sorted(int(p.stem.split("_")[1]) for p in (a / "kv_cache").glob("layer_*.safetensors"))
    for i in layers:
        pa, pb = a / "kv_cache" / f"layer_{i}.safetensors", b / "kv_cache" / f"layer_{i}.safetensors"
        if not pb.is_file():
            continue
        row = {}
        for n in KV:
            ta, tb = _get(pa, f"{n}_cache_layer_{i}"), _get(pb, f"{n}_cache_layer_{i}")
            if ta is None or tb is None or ta.numel() == 0:
                continue
            assert ta.shape == tb.shape, (i, n, ta.shape, tb.shape)
            row[n] = {"pcc": pcc(ta, tb), "rel": rel(ta, tb)}
        if row:
            out["kv"][i] = row
    for ca in sorted(a.glob("chunk_*.safetensors")):
        cb = b / ca.name
        if not cb.is_file():
            continue
        with safe_open(str(ca), "pt") as fa, safe_open(str(cb), "pt") as fb:
            common = [k for k in fa.keys() if k in set(fb.keys())]
            for k in sorted(common, key=lambda k: (not k.startswith("L"), int(k[1:]) if k[1:].isdigit() else 0, k)):
                ta, tb = fa.get_tensor(k), fb.get_tensor(k)
                if ta.numel() == 0:
                    continue
                out["dumps"][f"{ca.stem}.{k}"] = {"pcc": pcc(ta, tb), "rel": rel(ta, tb)}
                if k == "logits":
                    out["dumps"][f"{ca.stem}.{k}"]["top1_agree"] = int((ta.argmax(-1) == tb.argmax(-1)).sum())
                    out["dumps"][f"{ca.stem}.{k}"]["rows"] = int(ta.shape[0])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("a", help="baseline golden dir")
    p.add_argument("b", help="golden dir compared with it")
    p.add_argument("--json", help="also write the numbers here")
    args = p.parse_args(argv)
    res = compare(Path(args.a), Path(args.b))
    print(f"A={args.a}\nB={args.b}")
    print(f"{'layer':>5}  " + "  ".join(f"{n + ' pcc':>15} {'rel':>8}" for n in KV))
    for i, row in res["kv"].items():
        cells = [f"{row[n]['pcc']:15.6f} {row[n]['rel']:8.4f}" if n in row else f"{'-':>15} {'-':>8}" for n in KV]
        print(f"{i:5d}  " + "  ".join(cells))
    print(f"{'dump':>24} {'pcc':>10} {'rel':>8}")
    for k, v in res["dumps"].items():
        extra = f"  top1 {v['top1_agree']}/{v['rows']}" if "top1_agree" in v else ""
        print(f"{k:>24} {v['pcc']:10.6f} {v['rel']:8.4f}{extra}")
    kv_lat = [(i, r["kv_latent"]["pcc"]) for i, r in res["kv"].items() if "kv_latent" in r]
    if kv_lat:
        i, v = min(kv_lat, key=lambda t: t[1])
        print(f"COMPARE kv_latent min pcc {v:.6f} at layer {i}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
