# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3.1 DSA layer 3: the FP64 attention weight the keys a device top-k missed carry, on the FP64 chain's sampled
rows. CPU only; never imports ttnn.

A device run teacher-fed from the FP64 chain (``dsa_teacher_feed.py``, input from ``fp64_chain.py``'s ``dsa_input_3``)
selects its own top-k; its recall of the chain's FP64 top-2048 falls with context while its attention output error
stays flat. This asks whether the keys whose membership changes carry attention weight. For each sampled row ``r``,
with the chain's FP64 query (``input_layernorm`` of ``dsa_input_3``) and FP64 keys, ``A`` the chain's selection and
``D`` the device's:

* ``missed_mass``: the weight on ``A \\ D`` of the FP64 softmax over ``A``;
* ``bottom_mass``: the weight, of the same softmax, on the ``|A \\ D|`` keys of ``A`` the FP64 indexer ranks lowest
  (the tail of ``A``, which the chain stores best first; what a pure near-tie miss would drop);
* ``intruder_mass``: the weight on ``D \\ A`` of the FP64 softmax over ``D``;
* ``missed_key_max``: the largest weight of one missed key in one head;
* ``selection_nrmse``: the FP64 attention output over ``D`` against over ``A``, i.e. the output error the device's
  selection alone causes, with the device score's NRMSE floor;
* ``rest_nrmse``: the device's attention output against the FP64 output over ``D``, the error its selection does not
  explain (cache format and arithmetic);
* ``selection_rest_cos``, ``selection_share``, ``rest_share``: the split of the device's error against the FP64 output
  over ``A`` into those two parts (``accuracy.dsa.error_split``); the shares sum to one with the cross term
  ``1 - both``.

The masses come from ``accuracy.dsa.selection_masses``; this module computes the FP64 attention weights they need.

Masses are per head, reported as the mean over heads and the worst head. ``check_nrmse`` compares the recomputed
FP64 output over ``A`` with the chain's stored ``attn_output_3`` (the chain's query came from its FP64 read, this
one from that read stored fp32).

    python -m models.demos.deepseek_v3_d_p.tests.kimi_k3.dsa_missed_weight --device RUN --out OUT
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from models.demos.deepseek_v3_d_p.tests.accuracy.dsa import error_split, selection_masses
from models.demos.deepseek_v3_d_p.tests.accuracy.windowed import quantile, row_nrmse, row_sums
from models.demos.deepseek_v3_d_p.tests.kda.long_context_golden import MANIFEST
from models.demos.deepseek_v3_d_p.tests.kimi_k3 import dsa_fp64_golden as dsa
from models.demos.deepseek_v3_d_p.tests.kimi_k3.fp64_chain import (
    DSA_LAYER,
    load_layer,
    load_products,
    load_sampled,
    open_chain,
)
from models.demos.deepseek_v3_d_p.tests.kimi_k3.teacher_feed import chunk_path

ROW_METRICS = (
    "missed",
    "intruders",
    "missed_mass_mean",
    "missed_mass_max",
    "bottom_mass_mean",
    "bottom_mass_max",
    "intruder_mass_mean",
    "intruder_mass_max",
    "missed_key_max",
    "selection_nrmse",
    "rest_nrmse",
    "selection_rest_cos",
    "selection_share",
    "rest_share",
    "check_nrmse",
    "device_nrmse",
)


def row_weights(
    q: dict,
    i: int,
    fp64_sel: torch.Tensor,
    device_sel: torch.Tensor,
    latent: torch.Tensor,
    kpe: torch.Tensor,
    w: dict,
    dims: dsa.Dims,
) -> dict:
    """The missed-key weights of query row ``i`` of ``q``: ``fp64_sel`` the FP64 selection (valid keys only, best first
    by the FP64 indexer score), ``device_sel`` the device's (valid, distinct keys only); ``latent`` / ``kpe`` the
    FP64 keys of every causal row. Returns the per-row metrics (masses as ``[heads]`` tensors) and both outputs."""
    wa = dsa.attention_weights(q, i, latent[fp64_sel], kpe[fp64_sel], w, dims)
    wd = dsa.attention_weights(q, i, latent[device_sel], kpe[device_sel], w, dims)
    return {
        **selection_masses(wa, wd, fp64_sel, device_sel),
        "out_fp64": dsa.attend(q, i, latent[fp64_sel], kpe[fp64_sel], w, dims),
        "out_device_sel": dsa.attend(q, i, latent[device_sel], kpe[device_sel], w, dims),
    }


def device_row(device: Path, chunk: int, row: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The device's top-k of absolute row ``row`` (valid: causal, not the no-key sentinel; distinct keys) and its
    attention output there (FP64)."""
    payload = torch.load(chunk_path(device, row // chunk), weights_only=True, mmap=True)
    sel = payload["topk"][row - payload["start"]].long()
    return torch.unique(sel[(sel >= 0) & (sel <= row)]), payload["attn_out"][row - payload["start"]].double()


def nrmse(want: torch.Tensor, got: torch.Tensor, floor: float) -> float:
    return float(row_nrmse(row_sums(want.unsqueeze(0), got.unsqueeze(0)), floor)[0])


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ra, rb = a.argsort().argsort().double(), b.argsort().argsort().double()
    return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.corrcoef(torch.stack([a.double(), b.double()]))[0, 1])


def summarize(rows: dict[str, torch.Tensor], window: int, count: int, topk: int) -> dict:
    """Per ``window`` of context positions, median / p90 / max of each row metric over rows past ``topk`` (rows below
    it select every causal key on both sides), and over all such rows the correlations with the device's NRMSE. A
    window without such a row has only its ``rows`` count (0), no statistics.

    Windowed by hand, not by ``windowed.RowSeries``: the study holds sampled rows only, so there is no per-row column
    that spans the window, and ``rows`` here counts the sampled rows in a window."""
    past = rows["row"] >= topk
    windows = []
    for lo in range(0, count, window):
        inside = past & (rows["row"] >= lo) & (rows["row"] < lo + window)
        entry = {"start": lo, "end": min(lo + window, count), "rows": int(inside.sum())}
        for m in ROW_METRICS if inside.any() else ():
            v = rows[m][inside].double()
            entry[f"{m}_median"] = quantile(v, 0.5)
            entry[f"{m}_p90"] = quantile(v, 0.9)
            entry[f"{m}_max"] = float(v.max())
        windows.append(entry)
    dev = rows["device_nrmse"][past]
    correlation = {
        f"{m}_vs_device_nrmse": {"pearson": pearson(rows[m][past], dev), "spearman": spearman(rows[m][past], dev)}
        for m in (
            "missed_mass_mean",
            "missed_mass_max",
            "missed_key_max",
            "intruder_mass_max",
            "selection_nrmse",
            "rest_nrmse",
            "missed",
        )
    }
    correlation["row_vs_missed_mass_mean"] = {
        "pearson": pearson(rows["row"][past].double(), rows["missed_mass_mean"][past]),
        "spearman": spearman(rows["row"][past].double(), rows["missed_mass_mean"][past]),
    }
    overall = {
        m: {
            "median": quantile(rows[m][past], 0.5),
            "p90": quantile(rows[m][past], 0.9),
            "max": float(rows[m][past].max()),
        }
        for m in ROW_METRICS
    }
    return {"rows": int(past.sum()), "overall": overall, "correlation": correlation, "windows": windows}


def run(*, device: Path, out: Path, window: int, log) -> dict:
    """Score every sampled row of the device run ``device``'s FP64 chain; writes ``rows.pt``, ``metrics.json`` and
    ``windows.tsv`` to ``out``."""
    manifest = json.loads((device / MANIFEST).read_text())
    input_manifest = json.loads((Path(manifest["input"]) / MANIFEST).read_text())
    golden, count, chunk = open_chain(input_manifest["chain"]), manifest["count"], manifest["chunk"]
    chain, chain_manifest = golden.path, golden.manifest
    checkpoint = Path(chain_manifest["identity"]["checkpoint"])
    # layer 3 as the chain loaded it, so its digest is the one the chain recorded
    L, digests = load_layer(checkpoint, DSA_LAYER, kda=False, dense=False, moe=False, dsa_attn=True, threads=1, log=log)
    open_chain(chain, weights=digests)
    dims = dsa.Dims(**chain_manifest["model"]["dsa_dims"])
    score = json.loads((device / "metrics.json").read_text())
    floor = score["attn_nrmse_floor"]
    device_rows = torch.load(device / "rows.pt", weights_only=True)
    device_nrmse = dict(zip(device_rows["attn_rows"].tolist(), device_rows["attn_nrmse"].tolist()))

    w, norm = L["dsa"], L["input_layernorm"]
    sampled = load_sampled(chain)
    keep = sampled["sampled_rows"] < count
    sampled_rows, stored = sampled["sampled_rows"][keep].tolist(), sampled["attn_output_3"][keep]
    log(f"{len(sampled_rows)} sampled rows < {count}; loading FP64 keys")
    keys = load_products(chain, ["latent", "kpe"], 0, max(sampled_rows) + 1)
    latent, kpe = keys["latent"].double(), keys["kpe"].double()
    log(f"keys {tuple(latent.shape)} {tuple(kpe.shape)} loaded")

    records = {m: [] for m in ("row", *ROW_METRICS)}
    masses = {m: [] for m in ("missed_mass", "bottom_mass", "intruder_mass")}
    for n, (r, want) in enumerate(zip(sampled_rows, stored)):
        p = load_products(chain, ["dsa_input_3", "topk_indices_3"], r, r + 1)
        read = p["dsa_input_3"].double()
        hidden = dsa.rms64(read, norm, dims.eps)
        q = dsa.layer_queries(hidden, w, dims)
        sel = p["topk_indices_3"][0].long()
        device_sel, device_out = device_row(device, chunk, r)
        res = row_weights(q, 0, sel[sel >= 0], device_sel, latent, kpe, w, dims)
        records["row"].append(r)
        records["missed"].append(res["missed"])
        records["intruders"].append(res["intruders"])
        for m in masses:
            masses[m].append(res[m])
            records[f"{m}_mean"].append(float(res[m].mean()))
            records[f"{m}_max"].append(float(res[m].max()))
        records["missed_key_max"].append(res["missed_key_max"])
        records["selection_nrmse"].append(nrmse(res["out_fp64"], res["out_device_sel"], floor))
        records["rest_nrmse"].append(nrmse(res["out_device_sel"], device_out, floor))
        for k, v in error_split(res["out_fp64"], res["out_device_sel"], device_out).items():
            records[k].append(v)
        records["check_nrmse"].append(nrmse(want, res["out_fp64"], floor))
        records["device_nrmse"].append(device_nrmse.get(r, float("nan")))
        if n % 32 == 0 or n == len(sampled_rows) - 1:
            log(
                f"row {n + 1}/{len(sampled_rows)} r={r} missed={res['missed']} "
                f"missed_mass_mean={records['missed_mass_mean'][-1]:.3e} bottom_mass_mean={records['bottom_mass_mean'][-1]:.3e} "
                f"intruder_mass_mean={records['intruder_mass_mean'][-1]:.3e} sel_nrmse={records['selection_nrmse'][-1]:.4f} "
                f"rest_nrmse={records['rest_nrmse'][-1]:.4f} sel_share={records['selection_share'][-1]:.3f} "
                f"check={records['check_nrmse'][-1]:.2e} device={records['device_nrmse'][-1]:.4f}"
            )
    rows = {
        k: torch.tensor(v, dtype=torch.int64 if k in ("row", "missed", "intruders") else torch.float64)
        for k, v in records.items()
    }
    rows.update({f"{m}_heads": torch.stack(v) for m, v in masses.items()})
    out.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out / "rows.pt")
    result = {
        "device": str(device),
        "chain": str(chain),
        "layer": DSA_LAYER,
        "count": count,
        "window": window,
        "attn_nrmse_floor": floor,
        **summarize(rows, window, count, dims.topk),
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=1))
    columns = list(result["windows"][0])
    (out / "windows.tsv").write_text(
        "\t".join(columns) + "\n" + "".join("\t".join(f"{x[c]:.6g}" for c in columns) + "\n" for x in result["windows"])
    )
    log(f"wrote {out}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", type=Path, required=True, help="complete device run teacher-fed from the chain")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=int, default=65536)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(
        device=args.device,
        out=args.out,
        window=args.window,
        log=lambda m: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {m}", flush=True),
    )


if __name__ == "__main__":
    main()
