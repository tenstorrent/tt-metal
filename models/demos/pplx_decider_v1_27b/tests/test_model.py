# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device accuracy of PplxDecider against the CPU fp32 reference."""

import json
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import comp_pcc, run_for_blackhole
from models.demos.pplx_decider_v1_27b.tests.common import ids_sha256, load_example_state, parametrize_mesh_traced
from models.demos.pplx_decider_v1_27b.tt.decision import decision_probabilities
from models.demos.pplx_decider_v1_27b.tt.model import PplxDecider

TESTS_DIR = Path(__file__).parent
REFERENCE_PATH = TESTS_DIR / "reference" / "decision_reference.json"
EXAMPLES_PATH = TESTS_DIR / "examples.json"

# End to end through 64 bfp8-weight layers; measured PCC is 0.998-0.999, so 0.999 would be too tight.
PCC_THRESHOLD = 0.99
PROB_MAX_DIFF = 0.02


def _oracle_logits(decider, input_ids):
    length = len(input_ids)
    padded_len = max(128, -(-length // 128) * 128)
    padded = torch.zeros(1, padded_len, dtype=torch.long)
    padded[0, :length] = torch.tensor(input_ids, dtype=torch.long)
    decider.model.reset_tp()
    logits = decider.model.prefill_tp(padded, valid_len=length)
    return logits.float()[list(decider.decision_config.token_ids)]


def _compare(tt_logits, ref, temperature):
    ref_logits = torch.tensor(ref["readout_logits"], dtype=torch.float32)
    tt_probs = decision_probabilities(tt_logits, len(ref["probabilities"]), temperature)
    ref_probs = torch.tensor(ref["probabilities"], dtype=torch.float32)
    _, pcc = comp_pcc(ref_logits, tt_logits, PCC_THRESHOLD)
    prob_diff = float((tt_probs - ref_probs).abs().max())
    return pcc, prob_diff, int(torch.argmax(tt_probs)) == int(torch.argmax(ref_probs))


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.skipif(not REFERENCE_PATH.exists(), reason=f"{REFERENCE_PATH} is missing; run generate_reference.py")
@parametrize_mesh_traced()
def test_pplx_decider_matches_reference(mesh_device):
    references = {e["name"]: e for e in json.loads(REFERENCE_PATH.read_text())["examples"]}
    examples = json.loads(EXAMPLES_PATH.read_text())

    decider = PplxDecider.from_pretrained(mesh_device)
    temperature = decider.decision_config.temperature

    prompts = []
    for example in examples:
        state = load_example_state(example, decider.args.CKPT_DIR)
        input_ids = decider.input_ids(state, example["question"])
        ref = references[example["name"]]
        assert ids_sha256(input_ids) == ref["input_ids_sha256"], (
            f"{example['name']}: rendered prompt differs from the reference prompt "
            f"({len(input_ids)} tokens vs {ref['num_tokens']})"
        )
        prompts.append((example["name"], input_ids, ref))

    rows, failures = [], []

    def run_paths(paths):
        for path, run in paths:
            for name, input_ids, ref in prompts:
                start = time.perf_counter()
                tt_logits = run(input_ids)
                elapsed_ms = (time.perf_counter() - start) * 1000
                pcc, prob_diff, top1_ok = _compare(tt_logits, ref, temperature)
                rows.append((name, path, len(input_ids), top1_ok, pcc, prob_diff, elapsed_ms))
                if not (top1_ok and pcc >= PCC_THRESHOLD and prob_diff < PROB_MAX_DIFF):
                    failures.append(f"{name} [{path}]: top1_ok={top1_ok} pcc={pcc} prob_diff={prob_diff}")

    run_paths((("chunked", decider.readout_logits), ("oracle", lambda ids: _oracle_logits(decider, ids))))

    first_ids = prompts[0][1]
    first = decider.readout_logits(first_ids)
    second = decider.readout_logits(first_ids)
    chunked_repeat_ok = torch.equal(first, second)

    decider.capture_prefill_trace()
    run_paths((("traced", decider.readout_logits),))

    # long_license spans one traced chunk plus an eager tail.
    long_ids = next(ids for name, ids, _ in prompts if name == "long_license")
    first = decider.readout_logits(long_ids)
    second = decider.readout_logits(long_ids)
    traced_repeat_ok = torch.equal(first, second)

    logger.info(f"{'example':<14} {'path':<8} {'T':>6} {'top1':>6} {'PCC':>8} {'prob diff':>10} {'ms':>9}")
    for name, path, num_tokens, top1_ok, pcc, prob_diff, elapsed_ms in rows:
        logger.info(
            f"{name:<14} {path:<8} {num_tokens:>6} {top1_ok!s:>6} {pcc:>8.5f} {prob_diff:>10.5f} {elapsed_ms:>9.1f}"
        )
    assert not failures, "; ".join(failures)
    assert chunked_repeat_ok, f"{prompts[0][0]}: repeated chunked prefill is not bit-identical"
    assert traced_repeat_ok, "long_license: repeated traced prefill is not bit-identical"
