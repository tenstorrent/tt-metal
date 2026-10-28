# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Run one bench-golden case through KairosGenerator and score it: the gate records of CONTRACT §3.

Every call starts from the golden's bias0 (RouterState restored). Records:
- `<case>`: the generate() path. Head metrics over every device row of the comparable forwards (all forwards for
  H <= 128; the first forward of the x half for a free-running rollout), data-space metrics on the final quantiles,
  routing keys over every forward (report-only from free-running step 2 on: `free_step`).
- `<case>_tf` (rollouts only): teacher-forced, every step fed the golden's step context and bias; gated per step.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import torch

from models.experimental.kairos_forecast.tests.helpers import BENCH_GOLDENS
from models.experimental.kairos_forecast.tests.pcc import metrics as M
from models.experimental.kairos_forecast.tt.generator import KairosGenerator
from models.experimental.kairos_forecast.tt.host_frontend import front_end
from models.experimental.kairos_forecast.tt.router_state import RouterState, routing_record

GIFT_CASES = ["paper", "long_horizon", "ragged", "single", "single_plain", "single_short", "single_long_horizon"]


def load(name):
    return torch.load(BENCH_GOLDENS / f"{name}.pt", weights_only=False)


def _routing(log, rec, free_from=None):
    """Sum of the routing keys over forwards (free_from: index of the first forward whose context came from device
    output; from there on the keys are report-only and kept separately)."""
    out = {"route_mismatch_rows": 0, "route_mismatch_patches": 0, "bias_max_abs_diff": 0.0, "min_top3_margin": 1.0}
    free = dict(out)
    for k, entry in enumerate(log):
        ref_w, ref_i = rec["gate"][k]
        rows = entry["context"].shape[0]
        rr = routing_record(
            ref_i.view(rows, -1, 3),
            entry["routing"]["indices"].view(rows, -1, 3),
            rec["bias_after"][k],
            entry["routing"]["bias_after"],
            rec["scores"][k],
        )
        tgt = free if free_from is not None and k in free_from else out
        for key in ("route_mismatch_rows", "route_mismatch_patches"):
            tgt[key] += rr[key]
        tgt["bias_max_abs_diff"] = max(tgt["bias_max_abs_diff"], rr["bias_max_abs_diff"])
        tgt["min_top3_margin"] = min(tgt["min_top3_margin"], rr["min_top3_margin"])
    if free_from:
        out.update({f"free_{k}": v for k, v in free.items()})
    return out


def _post(rec):
    """Raw golden-layout head of the case's single forward pair -> data-space forecast (H <= 128 only)."""
    S, h, gift = rec["context"].shape[0], rec["horizon"], rec["gift"]
    ls = [tuple(torch.cat(x) for x in zip(*rec["loc_scale"]))] if gift else rec["loc_scale"]
    loc, scale = ls[0]
    pos = (~torch.any(rec["context"] < 0, dim=1)).view(-1, 1, 1)

    def post(raw):
        from models.experimental.kairos_forecast.reference.dataflow import head_to_quantiles

        q = head_to_quantiles(raw, loc, scale)[..., :h]
        if gift:
            q = (q[:S] - torch.flip(q[S:], dims=[1])) / 2
            q = torch.where(pos, q.clamp_min(0.0), q)
        return q

    return post


def save_heads(name, precision, dev_raw, q):
    """v1.32: with $KAIROS_SAVE_HEADS set, keep the record's device head and final quantiles (default off)."""
    root = os.environ.get("KAIROS_SAVE_HEADS")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        torch.save({"head": dev_raw.cpu(), "q": q.cpu()}, Path(root) / f"{name}_{precision}.pt")


def score(dev_raw, gold_raw, q, g_q, scale, halves=1) -> dict:
    """halves: 2 when the rows are the x half then the -x half (flip cases), for the per-half row bias."""
    m = M.gate_metrics(M.head_view(dev_raw), M.head_view(gold_raw), q, g_q, scale, halves)
    m["pcc_normalized_full"] = M.pearson(M.head_view(gold_raw, 128), M.head_view(dev_raw, 128))
    return m


def run_case(model, name, precision, *, collapsed=False):
    rec = load(name)
    S, h, gift = rec["context"].shape[0], rec["horizon"], rec["gift"]
    steps = -(-h // 128)
    gen = KairosGenerator(model, RouterState(rec["bias0"]))
    t0 = time.perf_counter()
    q = gen.generate(rec["context"], h, force_flip_invariance=gift, infer_is_positive=gift)
    e2e_s = time.perf_counter() - t0
    g_q = rec["prediction_outputs"]
    scale = rec["loc_scale"][0][1].reshape(-1)
    base = {
        "name": name,
        "precision": precision,
        "rows": S * (2 if gift else 1),
        "forwards": len(gen.log),
        "e2e_s": e2e_s,
    }
    if steps == 1:
        dev = torch.cat([e["head"] for e in gen.log])
        gold = torch.cat(rec["head"])
        save_heads(name, precision, dev, q)
        r = {**base, **score(dev, gold, q, g_q, scale, 2 if gift else 1), **_routing(gen.log, rec)}
        if collapsed:
            r["collapsed"] = M.collapsed_metrics(gold, g_q, _post(rec), scale)
        return [r], gen
    # rollout: forwards [x step 1..steps, -x step 1..steps]; free-running from x step 2 on (and the whole -x half,
    # whose bias follows the x half's device-fed contexts)
    free = set(range(1, len(gen.log)))
    first = gen.log[0]["head"]
    save_heads(name, precision, first, q)
    r = {**base, **score(first, rec["head"][0], q, g_q, scale), **_routing(gen.log, rec, free), "free_step": 1}
    return [r, teacher_forced(model, name, precision, rec)], gen


def teacher_forced(model, name, precision, rec):
    """Every rollout step on the golden's own context and bias: head metrics over all steps' rows, routing gated."""
    log, heads = [], []
    for k, ctx in enumerate(rec["step_context"]):
        router = RouterState(rec["bias0"] if k == 0 else rec["bias_after"][k - 1])
        front, routing = front_end(ctx, router, model.W)
        heads.append(model.forward(front))
        log.append({"head": heads[-1], "routing": routing, "context": ctx})
    from models.experimental.kairos_forecast.reference.dataflow import head_to_quantiles

    dev, gold = torch.cat(heads), torch.cat(rec["head"])
    # data space per step: each step's head de-normalized with that step's golden loc / scale
    dq = torch.cat([head_to_quantiles(heads[k], *rec["loc_scale"][k]) for k in range(len(heads))])
    gq = torch.cat([head_to_quantiles(rec["head"][k], *rec["loc_scale"][k]) for k in range(len(heads))])
    scale = torch.cat([rec["loc_scale"][k][1].reshape(-1) for k in range(len(heads))])
    save_heads(f"{name}_tf", precision, dev, dq)
    m = score(dev, gold, dq, gq, scale, 2 if rec["gift"] else 1)  # steps of the x half, then of the -x half
    step_rows = rec["head"][0].shape[0]
    m["pcc_normalized_step_min"] = min(
        M.pearson(M.head_view(rec["head"][k]), M.head_view(heads[k])) for k in range(len(heads))
    )
    return {
        "name": f"{name}_tf",
        "precision": precision,
        "rows": step_rows * len(heads),
        "forwards": len(heads),
        **m,
        **_routing(log, rec),
    }
