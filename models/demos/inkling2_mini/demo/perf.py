# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini perf harness: golden gate, traced decode tok/s, prefill tok/s + TTFT. One JSON line per run.

    python -m models.demos.inkling2_mini.demo.perf --label my-experiment [--long code1500 prose8k] [--set-baseline]

Every card part runs in its own worker process under `flock /tmp/tt-card.lock` (other tasks interleave between parts).
The line is printed and appended to /work/inkling-bringup/perf/results.jsonl (with branch, commit, label).

Gate (review-r5 D3, design §7), short golden prompts prompt0/prompt1/chat/math, eager:
  forced:     per-app PCC >= baseline - 0.0005, logits PCC >= 0.98, top-1 >= 90 %, top-5 contains ref top-1 >= 98 %
  token-only: logits PCC >= 0.98, top-1 hits >= baseline - 1
  long (code1500, prose8k; traced prefill, ALL positions vs golden v1, last position vs v2): regression gate =
              top-1, top-5, logits PCC (full for code1500, ref top-32 entries for prose8k) >= baseline - 0.01.
              Design §7 absolute bars are reported per prompt as "bars_7" (v1 baseline does not meet them).
              No per-app PCC on long prompts: two valid CPU refs differ to 0.958 there (golden-v2.md).
Baseline = perf/baseline.json (written by --set-baseline). Without it only the absolute bars apply.

Decode: traced 48-app step, median of 5 x 10 steps, at (max_ctx, pos) = (4K, 2K), (32K, 30K), (128K, 120K).
v1 decode cost depends on max_ctx; pos is set directly (no real 120K sequence).

Prefill: whatever the branch has. Default = traced decode token by token. If a module
models/demos/inkling2_mini/tt/prefill_*.py defines `prefill(model, ids) -> dict` it is used with --prefill auto
(contract: same result dict as Inkling2Model.step for the last position; optional "tokens": host list of the
argmax at every position, used by the long gate). TTFT = first prompt token in -> first new token on host.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import importlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

GOLDEN = Path("/work/inkling-bringup/golden")
PERF = Path("/work/inkling-bringup/perf")
SHORT = ["prompt0", "prompt1", "chat", "math"]
DECODE = [(4096, 2048), (32768, 30720), (131072, 122880)]
LONG_CTX = {"code1500": 2048, "prose8k": 8704}
LOCK = "/tmp/tt-card.lock"
HERE = Path(__file__).resolve()


# ---------------------------------------------------------------- workers (run on the card)
def _open(trace_mib=512):
    import ttnn

    return ttnn.open_device(device_id=0, trace_region_size=trace_mib << 20)


def worker_gate(out):
    import torch  # noqa: F401

    import ttnn
    from models.demos.inkling2_mini.demo.golden_check import run_prompt
    from models.demos.inkling2_mini.tt.model import Inkling2Model

    dev = _open()
    res = {}
    try:
        model = Inkling2Model(dev, max_ctx=2048, log=lambda s: None)
        for name in SHORT:
            res[name] = {
                "forced": run_prompt(model, name, True, gdir=GOLDEN),
                "tokens": run_prompt(model, name, False, gdir=GOLDEN),
            }
    finally:
        ttnn.close_device(dev)
    Path(out).write_text(json.dumps(res))


def _prefill_impl(mode):
    if mode == "decode":
        return None, "decode"
    for f in sorted(glob.glob(str(HERE.parent.parent / "tt" / "prefill_*.py"))):
        mod = importlib.import_module(f"models.demos.inkling2_mini.tt.{Path(f).stem}")
        if hasattr(mod, "prefill"):
            return mod.prefill, Path(f).stem
    if mode == "chunked":
        raise RuntimeError("no tt/prefill_*.py with prefill()")
    return None, "decode"


def worker_decode(out, max_ctx, pos, steps=10, reps=5):
    import ttnn
    from models.demos.inkling2_mini.tt.generator import Generator
    from models.demos.inkling2_mini.tt.model import Inkling2Model

    dev = _open()
    try:
        model = Inkling2Model(dev, max_ctx=max_ctx, log=lambda s: None)
        gen = Generator(model)
        gen.capture()
        gen.decode(1, pos)
        ttnn.synchronize_device(dev)
        ms = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for i in range(steps):
                gen.decode(1, pos + i)
            ttnn.synchronize_device(dev)
            ms.append((time.perf_counter() - t0) / steps * 1e3)
        t0 = time.perf_counter()
        for i in range(steps):
            gen.read_token(gen.decode(1, pos + i))
        ms_read = (time.perf_counter() - t0) / steps * 1e3
        tv = ttnn.get_memory_view(dev, ttnn.BufferType.TRACE)
        dv = ttnn.get_memory_view(dev, ttnn.BufferType.DRAM)
        r = {
            "max_ctx": max_ctx,
            "pos": pos,
            "ms": statistics.median(ms),
            "ms_runs": [round(x, 2) for x in ms],
            "tok_s": 1e3 / statistics.median(ms),
            "ms_with_token_read": ms_read,
            "trace_mib": tv.num_banks * tv.total_bytes_allocated_per_bank / 2**20,
            "dram_free_gib": dv.num_banks * dv.total_bytes_free_per_bank / 2**30,
        }
        gen.release()
    finally:
        ttnn.close_device(dev)
    Path(out).write_text(json.dumps(r))


def worker_prefill(out, name, mode):
    """Long-prompt prefill + gate over ALL positions (golden v1 full-position files; v2 last position).

    TT logits are read back at every position for the gate; that readback is timed separately and is NOT in ttft_s
    (ttft_s = per-step decode + argmax read, i.e. what a real prefill would cost)."""
    import torch

    import ttnn
    from models.demos.inkling2_mini.tt.generator import Generator
    from models.demos.inkling2_mini.tt.model import VOCAB_REAL, Inkling2Model

    d = GOLDEN / name
    ids = torch.load(d / "input_ids.pt")[0].tolist()
    top32 = torch.load(d / "logits_top32.pt")
    ref_idx, ref_val = top32["indices"].long(), top32["values"].float()
    full = torch.load(d / "logits_bf16.pt")[0] if (d / "logits_bf16.pt").exists() else None  # [T, V] bf16 (T <= 2048)
    v2 = GOLDEN / "v2" / name / "last.pt"
    v2_last = torch.load(v2)["logits_last"].float() if v2.exists() else None
    T = len(ids)
    fn, impl = _prefill_impl(mode)

    def pcc(a, b):
        a, b = a.double() - a.double().mean(), b.double() - b.double().mean()
        return float((a @ b) / (a.norm() * b.norm() + 1e-30))

    toks, top5, pcc_pos, pcc32 = [], [], [], []
    last = None
    readback = 0.0
    dev = _open()
    try:
        model = Inkling2Model(dev, max_ctx=LONG_CTX.get(name, T + 576), log=lambda s: None)
        gen = Generator(model)
        gen.capture()
        model.reset()
        t0 = time.perf_counter()
        if fn is not None:
            r = fn(model, ids)
            first = int(ttnn.to_torch(r["token"]).reshape(-1)[0])
            toks = list(r.get("tokens", []))
            ttft = time.perf_counter() - t0
            last = ttnn.to_torch(r["logits"]).float().reshape(-1)[:VOCAB_REAL]
        else:
            for t, tok in enumerate(ids):
                r = gen.decode(tok, t)
                toks.append(gen.read_token(r))
                t1 = time.perf_counter()
                lg = ttnn.to_torch(r["logits"]).float().reshape(-1)[:VOCAB_REAL]
                top5.append(int(lg.topk(5).indices.eq(ref_idx[t, 0]).any()))
                pcc32.append(pcc(lg[ref_idx[t]], ref_val[t]))
                if full is not None:
                    pcc_pos.append(pcc(lg, full[t, :VOCAB_REAL].float()))
                last = lg
                readback += time.perf_counter() - t1
            first = toks[-1]
            ttft = time.perf_counter() - t0 - readback
        gen.release()
    finally:
        ttnn.close_device(dev)

    res = {"name": name, "T": T, "impl": impl, "max_ctx": model.max_ctx, "ttft_s": ttft, "prefill_tok_s": T / ttft}
    res["gate_readback_s"] = readback
    res["last_argmax_eq_v1"] = bool(first == int(ref_idx[T - 1, 0]))
    if v2_last is not None and last is not None:
        res["last_argmax_eq_v2"] = bool(int(last.argmax()) == int(v2_last.argmax()))
        res["last_logits_pcc_v2"] = pcc(last, v2_last)
    if toks:
        hit = torch.tensor(toks[:T]).eq(ref_idx[: len(toks), 0]).float()
        res["top1"] = float(hit.mean())
        edges = [0, 512, 1024, 2048, 4096, 8192, T]
        res["top1_by_pos"] = {
            f"{a}-{b}": round(float(hit[a:b].mean()), 3) for a, b in zip(edges, edges[1:]) if a < min(b, len(hit))
        }
    if top5:
        res["top5"] = statistics.mean(top5)
        res["top32_pcc_mean"] = statistics.mean(pcc32)
    if pcc_pos:
        res["logits_pcc_mean"] = statistics.mean(pcc_pos)
        res["logits_pcc_min"] = min(pcc_pos)
    Path(out).write_text(json.dumps(res))


# ---------------------------------------------------------------- driver (no card)
def _run_worker(args, tmp):
    limit = _timeout_s(args)
    cmd = ["timeout", str(limit), "flock", LOCK, sys.executable, "-m", "models.demos.inkling2_mini.demo.perf"]
    cmd += ["--worker", *args, "--out", tmp]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode == 124:  # timed out: reset the card inside the lock before anything else runs
        subprocess.run(["flock", LOCK, "bash", "-c", "(. ~/.tenstorrent-venv/bin/activate && tt-smi -r)"])
        raise RuntimeError(f"worker {args} timed out after {limit}s; card reset with tt-smi -r")
    if p.returncode != 0 or not Path(tmp).exists():
        tail = "\n".join(l for l in (p.stdout + p.stderr).splitlines() if "| info" not in l and "| DEBUG" not in l)[
            -3000:
        ]
        raise RuntimeError(f"worker {args} failed rc={p.returncode}:\n{tail}")
    r = json.loads(Path(tmp).read_text())
    os.remove(tmp)
    print(f"  {' '.join(args)}: {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return r


def _timeout_s(args):
    """<= 3x the expected time of each worker (lock wait included)."""
    kind = args[0]
    if kind == "prefill":
        return {"code1500": 450, "prose8k": 2400}.get(args[1], 3600)
    return {"gate": 450, "decode": 360}.get(kind, 900)


def _git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=HERE.parent).stdout.strip()


def summarize_gate(g, base):
    apps = [min(g[n]["forced"]["app_pcc"][a] for n in SHORT) for a in range(48)]
    tot = sum(g[n]["forced"]["T"] for n in SHORT)
    f_top1 = sum(g[n]["forced"]["top1_agree"] * g[n]["forced"]["T"] for n in SHORT) / tot
    f_top5 = sum(g[n]["forced"]["top5_has_ref_top1"] * g[n]["forced"]["T"] for n in SHORT) / tot
    t_hits = round(sum(g[n]["tokens"]["top1_agree"] * g[n]["tokens"]["T"] for n in SHORT))
    s = {
        "forced_app_pcc_min": min(apps),
        "forced_logits_pcc_min": min(g[n]["forced"]["logits_pcc"] for n in SHORT),
        "forced_top1": f_top1,
        "forced_top5": f_top5,
        "tokens_logits_pcc_min": min(g[n]["tokens"]["logits_pcc"] for n in SHORT),
        "tokens_top1_hits": t_hits,
        "tokens_positions": tot,
        "tokens_kl_mean": sum(g[n]["tokens"]["kl_mean"] for n in SHORT) / len(SHORT),
        "app_pcc": [round(x, 5) for x in apps],
    }
    fails = []
    if s["forced_logits_pcc_min"] < 0.98:
        fails.append("forced logits pcc")
    if f_top1 < 0.90:
        fails.append("forced top1")
    if f_top5 < 0.98:
        fails.append("forced top5")
    if s["tokens_logits_pcc_min"] < 0.98:
        fails.append("token-only logits pcc")
    if base:
        bad = [a for a in range(48) if apps[a] < base["app_pcc"][a] - 0.0005]
        if bad:
            fails.append(f"app pcc < baseline-0.0005 at apps {bad}")
        if t_hits < base["tokens_top1_hits"] - 1:
            fails.append("token-only top1 < baseline-1")
    return s, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--long", nargs="*", default=["code1500", "prose8k"])
    ap.add_argument("--prefill", choices=["auto", "decode", "chunked"], default="auto")
    ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--decode", nargs="*", default=None, help="subset of max_ctx values, e.g. 4096")
    ap.add_argument("--set-baseline", action="store_true")
    ap.add_argument("--no-append", action="store_true")
    ap.add_argument("--worker", nargs="*")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.worker:
        kind, *rest = a.worker
        if kind == "gate":
            worker_gate(a.out)
        elif kind == "decode":
            worker_decode(a.out, int(rest[0]), int(rest[1]))
        elif kind == "prefill":
            worker_prefill(a.out, rest[0], rest[1])
        return

    PERF.mkdir(parents=True, exist_ok=True)
    tmp = str(PERF / f".w{os.getpid()}.json")
    base_f = PERF / "baseline.json"
    base = json.loads(base_f.read_text()) if base_f.exists() and not a.set_baseline else None
    line = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "label": a.label,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
    }
    fails = []
    if not a.skip_gate:
        g = _run_worker(["gate"], tmp)
        line["gate"], fails = summarize_gate(g, base)
    line["decode"] = {}
    for mc, pos in DECODE:
        if a.decode is None or str(mc) in a.decode:
            r = _run_worker(["decode", str(mc), str(pos)], tmp)
            line["decode"][f"{mc // 1024}K"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
    line["prefill"] = {}
    for name in a.long:
        r = _run_worker(["prefill", name, a.prefill], tmp)
        line["prefill"][name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
        bars = {
            "top1>=0.90": r.get("top1", 0) >= 0.90,
            "top5>=0.98": r.get("top5", 0) >= 0.98,
            "logits_pcc>=0.98": r.get("logits_pcc_mean", r.get("top32_pcc_mean", 0)) >= 0.98,
            "last_argmax": r.get("last_argmax_eq_v1", False),
        }
        line["prefill"][name]["bars_7"] = bars  # absolute design §7 bars (reported; baseline does not meet them)
        b = (base or {}).get("long", {}).get(name)
        if b:  # regression gate vs baseline
            for k in ("top1", "top5", "top32_pcc_mean", "logits_pcc_mean"):
                if k in b and k in r and r[k] < b[k] - 0.01:
                    fails.append(f"{name} {k} {r[k]:.3f} < baseline {b[k]:.3f} - 0.01")
    line["gate_pass"] = (not fails) if not a.skip_gate else None
    line["gate_fails"] = fails
    if a.set_baseline and "gate" in line:
        b = {
            "commit": line["commit"],
            "app_pcc": line["gate"]["app_pcc"],
            "tokens_top1_hits": line["gate"]["tokens_top1_hits"],
        }
        keys = ("top1", "top5", "top32_pcc_mean", "logits_pcc_mean")
        b["long"] = {n: {k: v[k] for k in keys if k in v} for n, v in line["prefill"].items()}
        base_f.write_text(json.dumps(b, indent=1))
    s = json.dumps(line, sort_keys=True)
    print(s)
    if not a.no_append:
        with open(PERF / "results.jsonl", "a") as f:
            f.write(s + "\n")


if __name__ == "__main__":
    main()
