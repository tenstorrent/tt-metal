# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model accuracy gate for optimizer runs on gpt-oss-20b (QuietBox 2, 1x4 mesh, batch 1, all 24 layers).

Sequence: the demo's first prefill_128 prompt in the chat template, and a fixed 100-token continuation. The
reference is HF fp32 with eager attention (gen_reference.py), kept under the repo's gitignored
generated/optimizer_reference/.

Checks over the same 100 positions (one prefill position + 99 teacher-forced eager decode steps):

1. The score the optimizer parses ("PCC: x") is the mean logits PCC against HF fp32 over the 100 positions,
   held to the absolute floor PCC_THRESHOLD. (Token agreement is not the score: the unmodified tree agrees on
   77/100 top-1 positions, because 38 reference positions have an HF top-1 probability under 0.6, and the
   optimizer only reads thresholds of the form 0.9x.)
2. Top-1 / top-5 agreement, mean logits correlation and mean top-100 correlation, RELATIVE to a baseline pinned
   from the unmodified tree (generated/optimizer_accuracy_baseline_gpt-oss-20b.json). One position is 1.0 point.
3. Top-1 of the TRACED token-out path (decode trace + on-device greedy sampling, the path the perf gate times),
   teacher forced over the same continuation, also relative to the pinned baseline.

The optimizer keeps or reverts a change on the parsed PCC (the WORST "pcc ... <float>" in the output), not on
the pytest exit code, so a failed agreement check also prints "PCC: 0.000000".
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_MODEL", "openai/gpt-oss-20b")
# Weight upload from the ttnn cache through the pinned host-memory path is ~10x slower on this QB2 (#57763);
# the copy path is used instead. Host load time only; device timing is unaffected.
os.environ.setdefault("TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES", "0")
os.environ.setdefault(
    "TT_CACHE_PATH", str(Path(__file__).resolve().parents[5] / "generated" / "gpt_oss_20b_tt_cache")
)

import pytest  # noqa: E402
import torch  # noqa: E402

import ttnn  # noqa: E402
from models.demos.gpt_oss.tests.optimizer.gate_support import (  # noqa: E402
    MESH_SHAPE,
    REPO_ROOT,
    append_history,
    build_generator,
    device_params,
    git_head,
    greedy_sampling_params,
)

# The checkpoint this gate runs; stated as a literal so the optimizer's roofline can resolve the model.
HF_MODEL_ID = "openai/gpt-oss-20b"

# Absolute floor for the gate score: the mean logits PCC against HF fp32 over the 100 positions (0.967729 on
# the unmodified tree). The optimizer lifts this constant from the file text as its pass/fail threshold, and
# only reads thresholds of the form 0.9x. Token agreement is held relative to the pinned baseline (check 2).
PCC_THRESHOLD = 0.95

GENERATED = REPO_ROOT / "generated"
REFERENCE_LOGITS = GENERATED / "optimizer_reference" / "gpt-oss-20b-logits.pt"
BASELINE_PATH = GENERATED / "optimizer_accuracy_baseline_gpt-oss-20b.json"

# Relative floors against the pinned baseline: one position is 1.0 point, so 1.0 allows one flip.
TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TOP1_DROP_PTS", "1.0"))
TOP5_DROP_PTS = float(os.environ.get("PCC_GATE_TOP5_DROP_PTS", "1.0"))
TRACED_TOP1_DROP_PTS = float(os.environ.get("PCC_GATE_TRACED_TOP1_DROP_PTS", "1.0"))
MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_MEAN_PCC_DROP", "0.002"))
TOP100_MEAN_PCC_DROP = float(os.environ.get("PCC_GATE_TOP100_MEAN_PCC_DROP", "0.005"))


def _pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    return float(torch.corrcoef(torch.stack((actual, expected)))[0, 1])


def _pcc_top(actual: torch.Tensor, expected: torch.Tensor, k: int = 100) -> float:
    """Correlation over the reference's top-k token ids at this position (the tokens that matter)."""
    ids = expected.reshape(-1).float().topk(k).indices
    return _pcc(actual.reshape(-1)[ids], expected.reshape(-1)[ids])


def _agreement(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, bool]:
    """(top-1 match, reference argmax within the model's top 5) for one position."""
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
@pytest.mark.parametrize("mesh_device, device_params", [(MESH_SHAPE, device_params())], ids=["1x4"], indirect=True)
def test_optimizer_full_model_pcc(mesh_device, device_params):
    """Prefill + 99 teacher-forced eager decode steps against HF fp32, every layer; plus the traced path."""
    # Under `pytest -s` the first print lands on the node-id line, which contains "pcc"; the optimizer's
    # parser reads any "pcc ... <float>" on a line as a measurement. End that line first.
    print("", flush=True)
    assert REFERENCE_LOGITS.is_file(), f"missing {REFERENCE_LOGITS}; produce it with tests/optimizer/gen_reference.py"
    ref = torch.load(REFERENCE_LOGITS)
    prompt, cont, hf = ref["prompt_ids"], ref["continuation_ids"], ref["logits"]  # hf row i predicts cont[i]
    P, G = len(prompt), len(cont)
    assert hf.shape[0] == G

    generator, model_args, model, page_table, kv_cache, tokenizer = build_generator(mesh_device)
    assert model[0].args.n_layers == 24 and mesh_device.get_num_devices() == 4

    def prefill():
        return generator.prefill_forward_text(
            torch.tensor([prompt]), page_table=page_table, kv_cache=kv_cache, prompt_lens=[P], enable_trace=False
        )

    t0 = time.perf_counter()
    tt_prefill = prefill().reshape(-1)[: hf.shape[-1]]
    pccs = [_pcc(tt_prefill, hf[0])]
    top_pccs = [_pcc_top(tt_prefill, hf[0])]
    hits = [_agreement(tt_prefill, hf[0])]
    eager_argmax = [int(tt_prefill.argmax())]
    for i in range(G - 1):
        logits, _ = generator.decode_forward(
            torch.tensor([[cont[i]]]),
            torch.tensor([P + i]),
            enable_trace=False,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=None,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        logits = logits.reshape(-1)[: hf.shape[-1]]
        pccs.append(_pcc(logits, hf[i + 1]))
        top_pccs.append(_pcc_top(logits, hf[i + 1]))
        hits.append(_agreement(logits, hf[i + 1]))
        eager_argmax.append(int(logits.argmax()))
    eager_seconds = time.perf_counter() - t0

    # Traced token-out path: decode trace with on-device greedy sampling, teacher forced (host inputs reloaded
    # every step, so the device-fed token is replaced by the reference token).
    model[0].clear_kv_caches()
    t1 = time.perf_counter()
    first = int(prefill().reshape(-1)[: hf.shape[-1]].argmax())
    traced = [first]
    sampling = greedy_sampling_params()
    for i in range(G - 1):
        out_tok, _ = generator.decode_forward(
            torch.tensor([[cont[i]]]),
            torch.tensor([P + i]),
            enable_trace=True,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling,
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=i == 0,
            reset_sampling_state=i == 0,
        )
        traced.append(int(out_tok.reshape(-1)[0]))
    traced_seconds = time.perf_counter() - t1
    hf_argmax = hf.argmax(dim=-1).tolist()
    traced_top1 = 100.0 * sum(int(a == b) for a, b in zip(traced, hf_argmax)) / G
    traced_vs_eager = sum(int(a == b) for a, b in zip(traced, eager_argmax))

    positions = len(pccs)
    top1_pct = 100.0 * sum(h[0] for h in hits) / positions
    top5_pct = 100.0 * sum(h[1] for h in hits) / positions
    mean_corr = sum(pccs) / positions
    worst_corr = min(pccs)
    worst_pos = min(range(positions), key=pccs.__getitem__)
    worst_top = min(top_pccs)
    top100_mean = sum(top_pccs) / positions
    score = mean_corr
    print(
        f"ACCURACY positions={positions} top1_pct={top1_pct:.2f} top5_pct={top5_pct:.2f} "
        f"mean_corr={mean_corr:.6f} worst_corr={worst_corr:.6f} worst_pos={worst_pos} "
        f"prefill_corr={pccs[0]:.6f} top100_mean_corr={top100_mean:.6f} top100_worst_corr={worst_top:.6f} "
        f"traced_top1_pct={traced_top1:.2f} traced_vs_eager={traced_vs_eager}/{G} "
        f"eager_seconds={eager_seconds:.1f} traced_seconds={traced_seconds:.1f} tree={git_head()}",
        flush=True,
    )

    baseline = _load_baseline()
    failures = []
    if os.environ.get("PCC_GATE_PIN_BASELINE") == "1" or baseline is None:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "top1_pct": top1_pct,
                    "top5_pct": top5_pct,
                    "traced_top1_pct": traced_top1,
                    "mean_corr": mean_corr,
                    "worst_corr": worst_corr,
                    "top100_worst_corr": worst_top,
                    "top100_mean_corr": top100_mean,
                    "positions": positions,
                    "tree": git_head(),
                    "pinned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=1,
            )
        )
        print(
            f"ACCURACY_BASELINE pinned to {BASELINE_PATH} from tree {git_head() or '?'}: top1={top1_pct:.2f} "
            f"top5={top5_pct:.2f} traced_top1={traced_top1:.2f} mean_corr={mean_corr:.6f}.",
            flush=True,
        )
    else:
        if top1_pct < baseline["top1_pct"] - TOP1_DROP_PTS:
            failures.append(f"top-1 {top1_pct:.2f}% < baseline {baseline['top1_pct']:.2f}% - {TOP1_DROP_PTS}")
        if top5_pct < baseline["top5_pct"] - TOP5_DROP_PTS:
            failures.append(f"top-5 {top5_pct:.2f}% < baseline {baseline['top5_pct']:.2f}% - {TOP5_DROP_PTS}")
        if traced_top1 < baseline["traced_top1_pct"] - TRACED_TOP1_DROP_PTS:
            failures.append(
                f"traced top-1 {traced_top1:.2f}% < baseline {baseline['traced_top1_pct']:.2f}% - {TRACED_TOP1_DROP_PTS}"
            )
        if top100_mean < baseline["top100_mean_corr"] - TOP100_MEAN_PCC_DROP:
            failures.append(
                f"top-100 mean correlation {top100_mean:.6f} < baseline {baseline['top100_mean_corr']:.6f} - {TOP100_MEAN_PCC_DROP}"
            )
        if mean_corr < baseline["mean_corr"] - MEAN_PCC_DROP:
            failures.append(
                f"mean logits correlation {mean_corr:.6f} < baseline {baseline['mean_corr']:.6f} - {MEAN_PCC_DROP}"
            )
        print(
            f"ACCURACY_VS_BASELINE tree={baseline.get('tree') or '?'} "
            f"top1_delta_pts={top1_pct - baseline['top1_pct']:+.2f} "
            f"top5_delta_pts={top5_pct - baseline['top5_pct']:+.2f} "
            f"traced_top1_delta_pts={traced_top1 - baseline['traced_top1_pct']:+.2f} "
            f"top100_mean_corr_delta={top100_mean - baseline['top100_mean_corr']:+.6f} "
            f"mean_corr_delta={mean_corr - baseline['mean_corr']:+.6f}",
            flush=True,
        )
    append_history(
        "OPTIMIZER_ACCURACY_HISTORY",
        {
            "top1_pct": top1_pct,
            "top5_pct": top5_pct,
            "traced_top1_pct": traced_top1,
            "mean_corr": mean_corr,
            "worst_corr": worst_corr,
            "top100_mean_corr": top100_mean,
            "top100_worst_corr": worst_top,
            "score": score if not failures else 0.0,
            "passed": not failures and score >= PCC_THRESHOLD,
            "failures": failures,
            "eager_seconds": round(eager_seconds, 1),
            "traced_seconds": round(traced_seconds, 1),
        },
    )
    if failures:
        print("ACCURACY GATE FAILED: " + "; ".join(failures), flush=True)
        print("PCC: 0.000000 (accuracy gate failed, see ACCURACY GATE FAILED above)", flush=True)
        assert not failures, "; ".join(failures)

    print(
        f"GATE_SCORE eager_top1={top1_pct / 100:.4f} traced_top1={traced_top1 / 100:.4f} "
        f"(top-100 corr worst {worst_top:.4f} mean {top100_mean:.4f}) | PCC: {score:.6f}",
        flush=True,
    )
    assert score >= PCC_THRESHOLD
