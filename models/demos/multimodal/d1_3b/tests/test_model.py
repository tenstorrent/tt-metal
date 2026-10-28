# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""d1-3B on device against the CPU fp32 HF reference: answer-slot hidden state, logits and decisions.

    MESH_DEVICE=N150 pytest models/demos/multimodal/d1_3b/tests/test_model.py -s

PCC, top-1 token and the decision (argmax / yes-no side / score level) are checked against the fp32 reference,
the ground truth. Probability closeness is gated against the bf16 reference when present: the device runs bf16,
so a bf16 HF run is the fair baseline and isolates kernel differences from the fp32->bf16 rounding floor
(~0.04 on some high-confidence choices; see test_decision.test_bf16_reference_drift). Without it, the fp32
reference is used at the looser FP32_FALLBACK_MAX.
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.multimodal.d1_3b.tests.common import (
    REFERENCE_DIR,
    SIDECAR_DIR,
    ids_sha256,
    load_examples,
    load_state,
)
from models.demos.multimodal.d1_3b.tt.decision import D1Prompt
from models.demos.multimodal.d1_3b.tt.model import L1_SMALL_SIZE, TRACE_REGION_SIZE, TtD1Model

HIDDEN_PCC = 0.99
LOGITS_PCC = 0.99
# Per-question max |dp| against the bf16 reference. The floor has two measured parts: HF's own fp32->bf16
# rounding reaches 0.028 (test_decision.test_bf16_reference_drift), and the device's bf16 kernels differ from
# host bf16 by a little more on high-confidence multi-way choices (logits PCC still ~0.9996). 0.04 covers both;
# the mean stays an order of magnitude tighter.
MAX_ABS_PROB = 0.04
MEAN_ABS_PROB = 0.01
FP32_FALLBACK_MAX = 0.06
MAX_SCORE_LEVELS = 0.05
NEAR_TIE = 0.05
RESULTS_DIR = Path(os.environ.get("D1_RESULTS_DIR", "generated/d1_3b"))
DEVICE_PARAMS = {"l1_small_size": L1_SMALL_SIZE, "trace_region_size": TRACE_REGION_SIZE}


def pcc(a, b):
    return float(torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1])


def expected_level(p):
    return sum(i * x for i, x in enumerate(p))


def _load_refs():
    ref_path, side_path = REFERENCE_DIR / "decision_reference.json", SIDECAR_DIR / "decision_reference.pt"
    if not ref_path.is_file() or not side_path.is_file():
        pytest.skip("run tests/generate_reference.py --dtype float32 first")
    reference = {e["name"]: e for e in json.loads(ref_path.read_text())["examples"]}
    sidecar = torch.load(side_path)
    bf16_path = REFERENCE_DIR / "decision_reference_bfloat16.json"
    bf16_ref = {e["name"]: e for e in json.loads(bf16_path.read_text())["examples"]} if bf16_path.is_file() else None
    gate = MAX_ABS_PROB if bf16_ref is not None else FP32_FALLBACK_MAX
    logger.info(f"probability closeness gated against the {'bf16' if bf16_ref else 'fp32'} reference (<= {gate})")
    return reference, sidecar, bf16_ref, gate


def _compare(key, n_tokens, hidden, logits, probs, ref, side, prob_baseline):
    ref_probs = ref["probabilities"]
    top_ref = sorted(ref_probs, reverse=True)
    row = {
        "key": key,
        "type": ref["type"],
        "tokens": n_tokens,
        "hidden_pcc": pcc(hidden, side["hidden"]),
        "logits_pcc": pcc(logits, side["logits"]),
        "top1_match": int(logits.argmax()) == int(side["logits"].argmax()),
        "max_abs_prob": max(abs(a - b) for a, b in zip(probs, prob_baseline)),
        "max_abs_prob_fp32": max(abs(a - b) for a, b in zip(probs, ref_probs)),
        "argmax_match": probs.index(max(probs)) == ref_probs.index(max(ref_probs)),
        "ref_gap": top_ref[0] - top_ref[1],
        "probabilities": probs,
        "ref_probabilities": ref_probs,
    }
    if ref["type"] == "score":
        row["score_diff"] = abs(expected_level(probs) - expected_level(ref_probs))
    logger.info(
        f"{key:<34}{n_tokens:>5} tok  hidden {row['hidden_pcc']:.5f}  logits {row['logits_pcc']:.5f}  "
        f"|dp| {row['max_abs_prob']:.4f}  argmax {'ok' if row['argmax_match'] else 'FLIP'}"
    )
    return row


def _failures(row, gate, bf16_ref):
    out = []
    if row["hidden_pcc"] < HIDDEN_PCC:
        out.append(f"{row['key']}: hidden PCC {row['hidden_pcc']:.5f}")
    if row["logits_pcc"] < LOGITS_PCC:
        out.append(f"{row['key']}: logits PCC {row['logits_pcc']:.5f}")
    if row["max_abs_prob"] > gate:
        out.append(f"{row['key']}: |dp| {row['max_abs_prob']:.4f} vs {'bf16' if bf16_ref else 'fp32'} ref")
    if not row["argmax_match"] and row["ref_gap"] >= NEAR_TIE:
        out.append(f"{row['key']}: decision flipped (reference gap {row['ref_gap']:.3f})")
    if row.get("score_diff", 0.0) > MAX_SCORE_LEVELS:
        out.append(f"{row['key']}: expected level off by {row['score_diff']:.3f}")
    return out


def _finish(rows, failures, bf16_ref, results_name):
    mean_abs = sum(r["max_abs_prob"] for r in rows) / len(rows)
    summary = {
        "questions": len(rows),
        "min_hidden_pcc": min(r["hidden_pcc"] for r in rows),
        "min_logits_pcc": min(r["logits_pcc"] for r in rows),
        "top1_match_rate": sum(r["top1_match"] for r in rows) / len(rows),
        "argmax_match_rate": sum(r["argmax_match"] for r in rows) / len(rows),
        "prob_baseline": "bfloat16" if bf16_ref else "float32",
        "max_abs_prob": max(r["max_abs_prob"] for r in rows),
        "max_abs_prob_fp32": max(r["max_abs_prob_fp32"] for r in rows),
        "mean_abs_prob": mean_abs,
    }
    logger.info(f"summary: {summary}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / results_name).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    if mean_abs > MEAN_ABS_PROB:
        failures.append(f"mean |dp| {mean_abs:.4f} > {MEAN_ABS_PROB}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_text_decisions(mesh_device):
    reference, sidecar, bf16_ref, gate = _load_refs()
    model = TtD1Model(mesh_device)
    model.capture_traces()
    d1 = D1Prompt(tokenizer=model.args.tokenizer)

    rows, failures = [], []
    for example in load_examples():
        if example.get("images"):
            continue
        state = load_state(example)
        for qname, q in example["questions"].items():
            key = f"{example['name']}/{qname}"
            ref, side = reference[example["name"]]["questions"][qname], sidecar[key]
            ids = d1.input_ids(state, q)
            assert ids_sha256(ids) == ref["input_ids_sha256"], f"{key}: prompt tokens differ from the reference"
            hidden = model.hidden(ids)
            probs = d1.probabilities(q, model.option_logits(hidden, d1.option_token_ids(q)))
            baseline = (
                bf16_ref[example["name"]]["questions"][qname]["probabilities"] if bf16_ref else ref["probabilities"]
            )
            row = _compare(key, len(ids), hidden, model.logits(hidden), probs, ref, side, baseline)
            rows.append(row)
            failures += _failures(row, gate, bf16_ref)
    model.release_traces()
    _finish(rows, failures, bf16_ref, "accuracy_text.json")


@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_image_decisions(mesh_device):
    """Image states: the SigLIP2 tower + projector run on device, their embeddings spliced at the image tokens.

    The processor's pixel inputs come from the fp32 sidecar, so this isolates the device vision+text path from
    host image preprocessing (validated separately by the byte-exact prompt test).
    """
    reference, sidecar, bf16_ref, gate = _load_refs()
    model = TtD1Model(mesh_device)
    d1 = D1Prompt(tokenizer=model.args.tokenizer)

    rows, failures = [], []
    for example in load_examples():
        if not example.get("images"):
            continue
        for qname, q in example["questions"].items():
            key = f"{example['name']}/{qname}"
            ref, side = reference[example["name"]]["questions"][qname], sidecar[key]
            pix = {k: v for k, v in side["pixel_inputs"].items()}
            vision = model.vision_embeddings(
                pix["pixel_values"],
                spatial_shapes=pix.get("spatial_shapes"),
                pixel_attention_mask=pix.get("pixel_attention_mask"),
            )
            ids = side["input_ids"]
            hidden = model.hidden(ids, vision_embeddings=vision)
            probs = d1.probabilities(q, model.option_logits(hidden, d1.option_token_ids(q)))
            baseline = (
                bf16_ref[example["name"]]["questions"][qname]["probabilities"] if bf16_ref else ref["probabilities"]
            )
            row = _compare(key, len(ids), hidden, model.logits(hidden), probs, ref, side, baseline)
            rows.append(row)
            failures += _failures(row, gate, bf16_ref)
    _finish(rows, failures, bf16_ref, "accuracy_image.json")
