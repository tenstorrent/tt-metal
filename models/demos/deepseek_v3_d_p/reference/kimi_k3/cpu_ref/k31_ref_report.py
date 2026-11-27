# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tables and end-to-end scores for the K3.1 CPU references written by k31_cpu_ref.py (CPU only).

usage: python k31_ref_report.py <out dir> [--kld ref:test ...]

Reads <out>/<track>/metrics.jsonl (last entry per layer) and, for every --kld pair, the two tracks'
logits.pt + final.pt: per-row KL(ref || test) in fp64 over the full vocabulary, top-1 agreement,
and the logprob each assigns to the other's top-1. Writes <out>/report.json and prints markdown tables.
Pairs are track names, plus "trace" for the tf track's logits (the trace's own final stream through
the fp32 head). --trace-topk compares a track's logprobs with the trace's recorded top-100 (row 0 of
the trace topk streams = the prefill's last prompt row).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

TRACE = "/mnt/weka/model-weights/llm/ref-data/debug_traces/kimi_k31_pro_10k_v2/vllm-d06690a4-10240tok"


def metrics(path):
    rows = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                rows[m["layer"]] = m
    return rows


def g(d, *ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def fmt(v, nd=2):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def logits_of(out, name):
    """(logits [N, V] fp32 mmap, lse [N] fp64) of a track; "trace" = tf track; "<track>:fp32head" = fp32 lm_head variant."""
    base, _, variant = name.partition(":")
    d = out / ("tf" if base == "trace" else base)
    fin = torch.load(d / "final.pt")
    if variant == "fp32head":
        return torch.load(d / "logits_fp32head.pt", mmap=True), fin["lse_fp32head"]
    return torch.load(d / "logits.pt", mmap=True), fin["lse"]


def kld(out, ref, test, chunk=256):
    L_r, lse_r = logits_of(out, ref)
    L_t, lse_t = logits_of(out, test)
    N = L_r.shape[0]
    kl = torch.empty(N, dtype=torch.float64)
    top_r, top_t = torch.empty(N, dtype=torch.long), torch.empty(N, dtype=torch.long)
    lp_t_at_r = torch.empty(N, dtype=torch.float64)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        lr = L_r[a:b].double() - lse_r[a:b].unsqueeze(1)
        lt = L_t[a:b].double() - lse_t[a:b].unsqueeze(1)
        kl[a:b] = (lr.exp() * (lr - lt)).sum(1)
        top_r[a:b], top_t[a:b] = lr.argmax(1), lt.argmax(1)
        lp_t_at_r[a:b] = lt.gather(1, top_r[a:b].unsqueeze(1)).squeeze(1)
    agree = (top_r == top_t).double()
    segs = {"all": (0, N), "c0": (0, min(N, 4096)), "c1": (4096, N)} if N > 4096 else {"all": (0, N)}
    res = {}
    for nm, (a, b) in segs.items():
        res[nm] = {
            "kld_mean": float(kl[a:b].mean()),
            "kld_median": float(kl[a:b].median()),
            "kld_p99": float(kl[a:b].quantile(0.99)),
            "kld_max": float(kl[a:b].max()),
            "top1_agree": float(agree[a:b].mean()),
            "rows": b - a,
        }
    return res, kl


def trace_topk_check(out, track):
    from safetensors import safe_open

    root = Path(TRACE)
    ix = json.loads((root / "index.json").read_text())
    st = ix["tensor_streams"]

    def row0(key):
        ch = st[key]["chunks"][0]
        with safe_open(str(root / ch["path"]), framework="pt") as fh:
            return fh.get_tensor(list(fh.keys())[0])[:2]

    idx, lp = row0("topk_indices").long(), row0("topk_logprobs").float()
    L, lse = logits_of(out, track)
    res = {}
    for tr_row in range(idx.shape[0]):
        best = None
        for pos in (4095, 8191):
            if pos >= L.shape[0]:
                continue
            ours = (L[pos].double() - lse[pos])[idx[tr_row]]
            d = (ours - lp[tr_row].double()).abs()
            cand = {
                "pos": pos,
                "maxabs_logprob_diff_top100": float(d.max()),
                "mean_abs": float(d.mean()),
                "top1_ours": int(L[pos].argmax()),
                "top1_trace": int(idx[tr_row, 0]),
            }
            if best is None or cand["mean_abs"] < best["mean_abs"]:
                best = cand
        res[f"trace_topk_row{tr_row}"] = best
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--kld", nargs="*", default=[])
    ap.add_argument("--trace-topk", nargs="*", default=[])
    args = ap.parse_args()
    out = Path(args.out)
    rep = {}
    tf = metrics(out / "tf" / "metrics.jsonl")
    fr = metrics(out / "fp32" / "metrics.jsonl")
    bfm = metrics(out / "bf16" / "metrics.jsonl")
    layers = sorted(set(tf) | set(fr) | set(bfm))
    rep["layers"] = {l: {"tf": tf.get(l), "fp32": fr.get(l), "bf16": bfm.get(l)} for l in layers}
    if tf:
        print("\n## C3: fp32 teacher-forced on the trace's inputs vs the trace's outputs (rows 0-8191, dB)\n")
        print(
            "| L | attn | type | attn dB | attn bf16 floor | DSA recall | KV latent dB | FFN dB (own route) | routing identical | FFN dB same-route rows (n) | FFN dB trace route |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for l in layers:
            m = tf.get(l)
            if not m:
                continue
            kind = "MLA" if "kv_vs_trace" in m else "KDA"
            ffn = "dense" if l == 0 else "MoE"
            sr = g(m, "ffn_same_route_vs_trace")
            print(
                f"| {l} | {kind} | {ffn} | {fmt(g(m, 'attn_vs_trace', 'all'))} | {fmt(g(m, 'attn_bf16_floor'))} | "
                f"{fmt(g(m, 'dsa_recall_vs_trace', 'mean'), 4)} | {fmt(g(m, 'kv_vs_trace', 'latent'))} | {fmt(g(m, 'ffn_vs_trace', 'all'))} | "
                f"{fmt(g(m, 'route_vs_trace', 'identical'), 4)} | {fmt(g(sr, 'snr'))} ({g(sr, 'rows') or ''}) | {fmt(g(m, 'ffn_trace_route_vs_trace', 'all'))} |"
            )
    for name, mm in (("fp32", fr), ("bf16", bfm)):
        if not mm:
            continue
        print(f"\n## free-running {name} vs the trace, per layer (drift, dB)\n")
        print("| L | attn dB | FFN dB | routing identical | DSA recall | stream rms |")
        print("|---|---|---|---|---|---|")
        for l in layers:
            m = mm.get(l)
            if m:
                print(
                    f"| {l} | {fmt(g(m, 'attn_vs_trace', 'all'))} | {fmt(g(m, 'ffn_vs_trace', 'all'))} | {fmt(g(m, 'route_vs_trace', 'identical'), 4)} | "
                    f"{fmt(g(m, 'dsa_recall_vs_trace', 'mean'), 4)} | {fmt(g(m, 'run_rms'), 4)} |"
                )
    rep["kld"] = {}
    for pair in args.kld:
        ref, test = pair.split(":", 1)
        res, kl = kld(out, ref, test)
        rep["kld"][pair] = res
        torch.save(kl, out / f"kld_{ref}_vs_{test.replace(':', '_')}.pt")
        print(f"\nKL({ref} || {test}): " + json.dumps(res))
    for t in args.trace_topk:
        rep[f"trace_topk_{t}"] = trace_topk_check(out, t)
        print(f"\n{t} vs trace top-100: " + json.dumps(rep[f"trace_topk_{t}"]))
    p = out / "report.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(rep)
    p.write_text(json.dumps(old, indent=1, default=str))


if __name__ == "__main__":
    main()
