# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""strands-decider text demo on one Blackhole chip.

Answers every question in tests/examples.json and checks the device against the CPU fp32 reference
in tests/reference/ (made by tests/generate_reference.py)."""

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
from models.demos.strands_agent.tt.model import StrandsDecider
from models.demos.strands_agent.tt.prompting import encode_request

TESTS_DIR = Path(__file__).parent.parent / "tests"
REFERENCE_PATH = TESTS_DIR / "reference" / "decider_reference.json"
HIDDEN_PATH = TESTS_DIR / "reference" / "decider_hidden.safetensors"
EXAMPLES_PATH = TESTS_DIR / "examples.json"

MIN_HIDDEN_PCC = 0.9995  # min over all rows of all questions; measured >= 0.9998
MAX_PROB_DIFF = 0.06  # per question; measured max 0.046
MEAN_PROB_DIFF = 0.025  # over all questions; measured 0.015
TIE_MARGIN = (
    0.02  # top-1 may differ when the reference top1-top2 probability margin is below this (order_json/priority: 0.001)
)


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

    table, failures, answers = [], [], {}
    for example in examples:
        state = load_example_state(example, decider.args.adapter_dir)
        for q in encode_request(decider.tokenizer, state, example["questions"], cfg.max_length):
            name = f"{example['name']}/{q.name}"
            ref = refs[example["name"]][q.name]
            assert ids_sha256(q.input_ids) == ref["input_ids_sha256"], f"{name}: ids differ"
            t0 = time.perf_counter()
            logits, rows = decider.question_logits(q)
            ms = (time.perf_counter() - t0) * 1000
            answers[name] = build_answer(q, logits, cfg.temperature_by_kind, cfg.temperature, cfg.ordinal_smoothing)
            logger.info(f"{name} ({len(q.input_ids)} tokens, {ms:.0f} ms): {json.dumps(answers[name])}")

            ref_rows = ref_hidden[name].float()
            if ref_rows.shape != rows.shape:
                failures.append(f"{name}: row shape {tuple(rows.shape)} vs {tuple(ref_rows.shape)}")
                continue
            min_pcc = min(float(comp_pcc(ref_rows[i], rows[i], MIN_HIDDEN_PCC)[1]) for i in range(rows.shape[0]))
            ref_logits = torch.tensor(ref["logits"], dtype=torch.float32)
            n = len(q.labels)
            t = float(cfg.temperature_by_kind.get(q.kind, cfg.temperature))
            probs = option_log_probs(logits, n, t).exp()[:n]
            ref_probs = torch.tensor(ref["probabilities"], dtype=torch.float32)[:n]
            logit_diff = float((logits.reshape(-1)[:n] - ref_logits.reshape(-1)[:n]).abs().max())
            prob_diff = float((probs - ref_probs).abs().max())
            top2 = torch.topk(ref_probs, min(2, n)).values
            margin = float(top2[0] - top2[1]) if n > 1 else 1.0
            top1 = int(torch.argmax(probs)) == int(torch.argmax(ref_probs))
            table.append((name, len(q.input_ids), min_pcc, logit_diff, prob_diff, margin, top1, ms))

    logger.info(
        f"{'example/question':<32} {'T':>6} {'minPCC':>8} {'logit diff':>11} {'prob diff':>10} {'margin':>8} {'top1':>5} {'ms':>8}"
    )
    for name, tokens, pcc, ld, pd, margin, top1, ms in table:
        logger.info(f"{name:<32} {tokens:>6} {pcc:>8.5f} {ld:>11.5f} {pd:>10.5f} {margin:>8.4f} {top1!s:>5} {ms:>8.1f}")
    if table:
        logger.info(
            f"min PCC {min(r[2] for r in table):.5f}, max prob diff {max(r[4] for r in table):.5f}, "
            f"mean prob diff {sum(r[4] for r in table) / len(table):.5f}"
        )

    for name, _, pcc, _, pd, margin, top1, _ in table:
        if pcc < MIN_HIDDEN_PCC:
            failures.append(f"{name}: min hidden PCC {pcc:.5f} < {MIN_HIDDEN_PCC}")
        if pd > MAX_PROB_DIFF:
            failures.append(f"{name}: prob diff {pd:.5f} > {MAX_PROB_DIFF}")
        if not top1 and margin >= TIE_MARGIN:
            failures.append(f"{name}: top-1 differs (reference margin {margin:.4f} >= {TIE_MARGIN})")
    if table:
        mean_pd = sum(r[4] for r in table) / len(table)
        if mean_pd > MEAN_PROB_DIFF:
            failures.append(f"mean prob diff {mean_pd:.5f} > {MEAN_PROB_DIFF}")

    # The public evaluate() call must give the same answers bit for bit, after all other prompts ran (determinism).
    example = examples[0]
    result = decider.evaluate(load_example_state(example, decider.args.adapter_dir), example["questions"])
    for qname, answer in result["answers"].items():
        if answer != answers[f"{example['name']}/{qname}"]:
            failures.append(f"{example['name']}/{qname}: evaluate() answer differs from the first run")
    assert not failures, "accuracy check failed:\n" + "\n".join(failures)
