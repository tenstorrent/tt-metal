# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for optimizer runs on Ternary-Bonsai-2-27B (dense conversion), one P150, batch 1, all 64 layers.

Runs the paths the perf test times, both traced: the 4096-token chunk-trace prefill, then 99 teacher-forced decode
steps on the decode trace (primed as the demo primes it). Each step reads its logits back. The 100 positions (the
prefill's last position + 99 decode steps) are compared with a CPU HF bf16 forward of the same checkpoint
(gen_reference.py), kept under the repo's gitignored generated/optimizer_reference/.

Checks:
1. Top-1 / top-5 agreement with the reference argmax, mean logits correlation and mean correlation over the
   reference's top-100 tokens, RELATIVE to a baseline pinned from the unmodified tree
   (generated/optimizer_accuracy_baseline_bonsai2-27b-isl4096.json; one position is 1.0 point). The first run, or
   PCC_GATE_PIN_BASELINE=1, pins it.
2. The mean logits correlation is at least PCC_THRESHOLD.
Exit status is the verdict.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

from models.demos.blackhole.qwen36.tests.optimizer_bonsai.bonsai_support import (
    BASELINE_PATH,
    CHUNK,
    DEVICE_PARAMS,
    after_prefill,
    append_history,
    build_model,
    decode_step,
    git_head,
    load_reference,
    make_generator,
    prefill_traced,
    prime_decode,
)

PCC_THRESHOLD = 0.90
TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TOP1_DROP_PTS", "1.0"))
TOP5_DROP_PTS = float(os.environ.get("PCC_GATE_TOP5_DROP_PTS", "1.0"))
MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_MEAN_PCC_DROP", "0.002"))
TOP100_MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_TOP100_MEAN_PCC_DROP", "0.005"))


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.float().reshape(-1), expected.float().reshape(-1)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _pcc_top(actual: torch.Tensor, expected: torch.Tensor, k: int = 100) -> float:
    ids = expected.reshape(-1).float().topk(k).indices
    return _pcc(actual.reshape(-1)[ids], expected.reshape(-1)[ids])


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, bool]:
    ref = int(expected.reshape(-1).argmax())
    top5 = actual.reshape(-1).float().topk(5).indices.tolist()
    return top5[0] == ref, ref in top5


def _load_baseline() -> dict | None:
    try:
        doc = json.loads(BASELINE_PATH.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) and doc.get("top1_pct") is not None else None


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_bonsai_pcc(mesh_device, device_params):
    print("", flush=True)  # under -s the first print lands on the node-id line
    device = mesh_device
    device.enable_program_cache()
    ref = load_reference()
    prompt, cont, hf = ref["prompt_ids"], ref["continuation_ids"], ref["logits"]  # hf row i predicts cont[i]
    P, G, V = len(prompt), len(cont), ref["logits"].shape[-1]

    model, page_table = build_model(device)
    assert model.args.n_layers == 64, model.args.n_layers
    t0 = time.perf_counter()
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=CHUNK, warmup_masked_buckets=False)
    rows = [prefill_traced(model, prompt, page_table)[:V]]
    after_prefill(model)
    generator = make_generator(model, device)
    prime_decode(generator, model, cont[0], P, page_table)
    for i in range(G - 1):
        rows.append(decode_step(generator, cont[i], P + i, page_table, trace=True, read=True)[:V])
    seconds = time.perf_counter() - t0

    pccs = [_pcc(r, hf[i]) for i, r in enumerate(rows)]
    top_pccs = [_pcc_top(r, hf[i]) for i, r in enumerate(rows)]
    hits = [_agreement(r, hf[i]) for i, r in enumerate(rows)]
    positions = len(rows)
    top1_pct = 100.0 * sum(h[0] for h in hits) / positions
    top5_pct = 100.0 * sum(h[1] for h in hits) / positions
    mean_corr = sum(pccs) / positions
    worst_corr, worst_pos = min(pccs), min(range(positions), key=pccs.__getitem__)
    top100_mean, top100_worst = sum(top_pccs) / positions, min(top_pccs)
    print(
        f"ACCURACY positions={positions} top1_pct={top1_pct:.2f} top5_pct={top5_pct:.2f} mean_corr={mean_corr:.6f} "
        f"worst_corr={worst_corr:.6f} worst_pos={worst_pos} prefill_corr={pccs[0]:.6f} "
        f"top100_mean_corr={top100_mean:.6f} top100_worst_corr={top100_worst:.6f} seconds={seconds:.1f} "
        f"tree={git_head()}",
        flush=True,
    )

    metrics = {
        "top1_pct": top1_pct,
        "top5_pct": top5_pct,
        "mean_corr": mean_corr,
        "worst_corr": worst_corr,
        "top100_mean_corr": top100_mean,
        "top100_worst_corr": top100_worst,
        "positions": positions,
    }
    baseline = _load_baseline()
    failures = []
    if os.environ.get("PCC_GATE_PIN_BASELINE") == "1" or baseline is None:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        pinned = {**metrics, "tree": git_head(), "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        BASELINE_PATH.write_text(json.dumps(pinned, indent=1))
        print(f"ACCURACY_BASELINE pinned to {BASELINE_PATH} from tree {git_head() or '?'}", flush=True)
    else:
        if top1_pct < baseline["top1_pct"] - TOP1_DROP_PTS:
            failures.append(f"top-1 {top1_pct:.2f}% < baseline {baseline['top1_pct']:.2f}% - {TOP1_DROP_PTS}")
        if top5_pct < baseline["top5_pct"] - TOP5_DROP_PTS:
            failures.append(f"top-5 {top5_pct:.2f}% < baseline {baseline['top5_pct']:.2f}% - {TOP5_DROP_PTS}")
        if top100_mean < baseline["top100_mean_corr"] - TOP100_MEAN_PCC_DROP:
            failures.append(
                f"top-100 mean correlation {top100_mean:.6f} < baseline {baseline['top100_mean_corr']:.6f} "
                f"- {TOP100_MEAN_PCC_DROP}"
            )
        if mean_corr < baseline["mean_corr"] - MEAN_PCC_DROP:
            failures.append(
                f"mean correlation {mean_corr:.6f} < baseline {baseline['mean_corr']:.6f} - {MEAN_PCC_DROP}"
            )
        print(
            f"ACCURACY_VS_BASELINE tree={baseline.get('tree') or '?'} "
            f"top1_delta_pts={top1_pct - baseline['top1_pct']:+.2f} top5_delta_pts={top5_pct - baseline['top5_pct']:+.2f} "
            f"top100_mean_corr_delta={top100_mean - baseline['top100_mean_corr']:+.6f} "
            f"mean_corr_delta={mean_corr - baseline['mean_corr']:+.6f}",
            flush=True,
        )
    if mean_corr < PCC_THRESHOLD:
        failures.append(f"mean correlation {mean_corr:.6f} < {PCC_THRESHOLD}")
    append_history(
        "OPTIMIZER_ACCURACY_HISTORY", {**metrics, "passed": not failures, "failures": failures, "seconds": seconds}
    )
    if failures:
        print("ACCURACY GATE FAILED: " + "; ".join(failures), flush=True)
    assert not failures, "; ".join(failures)
    print(f"GATE_SCORE top1={top1_pct / 100:.4f} top5={top5_pct / 100:.4f} | PCC: {mean_corr:.6f}", flush=True)
