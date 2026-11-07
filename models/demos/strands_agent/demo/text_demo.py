# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""strands-decider text demo on one Blackhole chip.

Answers every question in tests/examples.json in one batched call, then a batch of 32 questions, and checks the
device against the CPU fp32 reference in tests/reference/ (made by tests/generate_reference.py)."""

import json
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors.torch import load_file

from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.strands_agent.tests.common import ids_sha256, load_example_state, parametrize_mesh
from models.demos.strands_agent.tt.answers import build_answer, option_log_probs
from models.demos.strands_agent.tt.model import StrandsDecider, plan_batches
from models.demos.strands_agent.tt.prompting import encode_request

TESTS_DIR = Path(__file__).parent.parent / "tests"
REFERENCE_PATH = TESTS_DIR / "reference" / "decider_reference.json"
HIDDEN_PATH = TESTS_DIR / "reference" / "decider_hidden.safetensors"
EXAMPLES_PATH = TESTS_DIR / "examples.json"

BATCH = 32
MIN_HIDDEN_PCC = 0.9995  # min over all rows of all questions; measured >= 0.9998
MAX_PROB_DIFF = 0.06  # per question; measured max 0.042
MEAN_PROB_DIFF = 0.025  # over all questions; measured 0.012
TIE_MARGIN = (
    0.02  # top-1 may differ when the reference top1-top2 probability margin is below this (order_json/priority: 0.001)
)


def _compare(q, logits, rows, ref, ref_rows, cfg):
    """(min hidden PCC, logit diff, prob diff, reference top1-top2 margin, top-1 match) against the reference."""
    min_pcc = min(float(comp_pcc(ref_rows[i], rows[i], MIN_HIDDEN_PCC)[1]) for i in range(rows.shape[0]))
    n = len(q.labels)
    t = float(cfg.temperature_by_kind.get(q.kind, cfg.temperature))
    probs = option_log_probs(logits, n, t).exp()[:n]
    ref_probs = torch.tensor(ref["probabilities"], dtype=torch.float32)[:n]
    ref_logits = torch.tensor(ref["logits"], dtype=torch.float32)
    logit_diff = float((logits.reshape(-1)[:n] - ref_logits.reshape(-1)[:n]).abs().max())
    prob_diff = float((probs - ref_probs).abs().max())
    top2 = torch.topk(ref_probs, min(2, n)).values
    margin = float(top2[0] - top2[1]) if n > 1 else 1.0
    top1 = int(torch.argmax(probs)) == int(torch.argmax(ref_probs))
    return min_pcc, logit_diff, prob_diff, margin, top1


def _check(label, table):
    """Log the accuracy table and return the failed limits."""
    logger.info(
        f"[{label}] {'example/question':<32} {'T':>6} {'minPCC':>8} {'logit diff':>11} {'prob diff':>10} {'margin':>8} {'top1':>5}"
    )
    for name, tokens, pcc, ld, pd, margin, top1 in table:
        logger.info(f"[{label}] {name:<32} {tokens:>6} {pcc:>8.5f} {ld:>11.5f} {pd:>10.5f} {margin:>8.4f} {top1!s:>5}")
    mean_pd = sum(r[4] for r in table) / len(table)
    logger.info(
        f"[{label}] min PCC {min(r[2] for r in table):.5f}, max prob diff {max(r[4] for r in table):.5f}, "
        f"mean prob diff {mean_pd:.5f}"
    )
    failures = []
    for name, _, pcc, _, pd, margin, top1 in table:
        if pcc < MIN_HIDDEN_PCC:
            failures.append(f"[{label}] {name}: min hidden PCC {pcc:.5f} < {MIN_HIDDEN_PCC}")
        if pd > MAX_PROB_DIFF:
            failures.append(f"[{label}] {name}: prob diff {pd:.5f} > {MAX_PROB_DIFF}")
        if not top1 and margin >= TIE_MARGIN:
            failures.append(f"[{label}] {name}: top-1 differs (reference margin {margin:.4f} >= {TIE_MARGIN})")
    if mean_pd > MEAN_PROB_DIFF:
        failures.append(f"[{label}] mean prob diff {mean_pd:.5f} > {MEAN_PROB_DIFF}")
    return failures


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.skipif(
    not (REFERENCE_PATH.exists() and HIDDEN_PATH.exists()), reason="reference missing; run generate_reference.py"
)
@parametrize_mesh()
def test_strands_decider_text_demo(mesh_device):
    reference = json.loads(REFERENCE_PATH.read_text())
    refs = {e["name"]: e["questions"] for e in reference["examples"]}
    ref_hidden = load_file(str(HIDDEN_PATH))
    examples = json.loads(EXAMPLES_PATH.read_text())

    start = time.perf_counter()
    decider = StrandsDecider.from_pretrained(mesh_device)
    logger.info(f"Model load time: {time.perf_counter() - start:.1f} s")
    cfg = decider.config

    requests, qs, names = [], [], []
    for example in examples:
        state = load_example_state(example, decider.args.adapter_dir)
        requests.append((state, example["questions"]))
        for q in encode_request(decider.tokenizer, state, example["questions"], cfg.max_length):
            name = f"{example['name']}/{q.name}"
            assert ids_sha256(q.input_ids) == refs[example["name"]][q.name]["input_ids_sha256"], f"{name}: ids differ"
            qs.append(q)
            names.append(name)

    def run(idx, label):
        batch = [qs[i] for i in idx]
        groups = plan_batches([len(q.input_ids) for q in batch], decider.max_batch, decider.max_batch_tokens)
        logger.info(f"[{label}] {len(batch)} questions in {len(groups)} forwards: {[len(g) for g in groups]}")
        start = time.perf_counter()
        out = decider.question_logits_batch(batch)
        logger.info(f"[{label}] batched call: {(time.perf_counter() - start) * 1000:.0f} ms (includes any compile)")
        table = []
        for i, (logits, rows) in zip(idx, out):
            ref_rows = ref_hidden[names[i]].float()
            assert ref_rows.shape == rows.shape, f"{names[i]}: rows {tuple(rows.shape)} vs {tuple(ref_rows.shape)}"
            ref = refs[names[i].split("/")[0]][qs[i].name]
            table.append((names[i], len(qs[i].input_ids), *_compare(qs[i], logits, rows, ref, ref_rows, cfg)))
        return out, groups, table

    # All example questions in one batched call.
    first, _, table = run(list(range(len(qs))), "all")
    answers = {}
    for name, q, (logits, _) in zip(names, qs, first):
        answers[name] = build_answer(q, logits, cfg.temperature_by_kind, cfg.temperature, cfg.ordinal_smoothing)
        logger.info(f"{name} ({len(q.input_ids)} tokens): {json.dumps(answers[name])}")
    failures = _check("all", table)

    # Batch of 32: the questions repeated. Copies in one forward must be bit-identical.
    idx = [k % len(qs) for k in range(BATCH)]
    out, groups, table = run(idx, f"b{BATCH}")
    failures += _check(f"b{BATCH}", table)
    for group in groups:
        seen = {}
        for k in group:
            if idx[k] in seen and not torch.equal(out[k][1], out[seen[idx[k]]][1]):
                failures.append(f"[b{BATCH}] {names[idx[k]]}: copies in one forward differ")
            seen.setdefault(idx[k], k)

    # The public evaluate_batch() call must give the first run's answers bit for bit (same prompts, same forwards).
    for example, result in zip(examples, decider.evaluate_batch(requests)):
        for qname, answer in result["answers"].items():
            if answer != answers[f"{example['name']}/{qname}"]:
                failures.append(f"{example['name']}/{qname}: evaluate_batch() answer differs from the first run")
    assert not failures, "accuracy check failed:\n" + "\n".join(failures)
