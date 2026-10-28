# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side checks of d1-3B's prompt and readout against the HF reference. No device needed."""

import json

import pytest
import torch
from loguru import logger

from models.demos.multimodal.d1_3b.tests.common import (
    REFERENCE_DIR,
    SIDECAR_DIR,
    ids_sha256,
    load_examples,
    load_state,
    resolve_checkpoint,
    text_sha256,
)
from models.demos.multimodal.d1_3b.tt.decision import D1Prompt

EXAMPLES = {e["name"]: e for e in load_examples()}
TEXT_EXAMPLES = [n for n, e in EXAMPLES.items() if not e.get("images")]
NEAR_TIE = 0.05


def _load_reference(dtype="float32"):
    suffix = "" if dtype == "float32" else f"_{dtype}"
    path = REFERENCE_DIR / f"decision_reference{suffix}.json"
    if not path.is_file():
        pytest.skip(f"{path} is missing; run tests/generate_reference.py --dtype {dtype}")
    return {e["name"]: e for e in json.loads(path.read_text())["examples"]}


@pytest.fixture(scope="module")
def d1():
    try:
        return D1Prompt(resolve_checkpoint())
    except Exception as e:
        pytest.skip(f"d1-3B checkpoint is not available: {e}")


@pytest.fixture(scope="module")
def reference():
    return _load_reference()


def assert_answer_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_answer_close(actual[key], value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-9)
    else:
        assert actual == expected


@pytest.mark.parametrize("name", TEXT_EXAMPLES)
def test_render_matches_reference(name, d1, reference):
    example = EXAMPLES[name]
    state = load_state(example)
    for qname, q in example["questions"].items():
        ref = reference[name]["questions"][qname]
        text = d1.render(state, q)
        ids = d1.input_ids(state, q)
        assert text_sha256(text) == ref["text_sha256"]
        assert len(ids) == ref["num_tokens"]
        assert ids_sha256(ids) == ref["input_ids_sha256"]
        assert d1.readout_ids(q) == ref["readout_ids"]


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_readout_reproduces_reference(name, d1, reference):
    for qname, q in EXAMPLES[name]["questions"].items():
        ref = reference[name]["questions"][qname]
        scores = {i: v for group, values in zip(ref["readout_ids"], ref["option_logz"]) for i, v in zip(group, values)}
        probabilities = d1.probabilities(q, scores)
        assert probabilities == pytest.approx(ref["probabilities"], abs=1e-9)
        assert_answer_close(d1.answer(q, probabilities), ref["answer"])


def test_option_logits_suffice(d1, reference):
    """Raw logits of the option tokens alone give the same decision as the full-vocab log-softmax."""
    path = SIDECAR_DIR / "decision_reference.pt"
    if not path.is_file():
        pytest.skip(f"{path} is missing; run tests/generate_reference.py")
    sidecar = torch.load(path)
    for key, side in sidecar.items():
        name, qname = key.split("/")
        q = EXAMPLES[name]["questions"][qname]
        logits = side["logits"].double()
        raw = {i: float(logits[i]) for i in d1.option_token_ids(q)}
        assert d1.probabilities(q, raw) == pytest.approx(reference[name]["questions"][qname]["probabilities"], abs=1e-6)


def test_bf16_reference_drift(reference):
    """How far HF itself moves between fp32 and bf16 on CPU: the noise floor for the TT tolerances."""
    bf16 = _load_reference("bfloat16")
    diffs, flips = [], []
    for name, entry in reference.items():
        for qname, ref in entry["questions"].items():
            p32, p16 = ref["probabilities"], bf16[name]["questions"][qname]["probabilities"]
            diffs.append(max(abs(a - b) for a, b in zip(p32, p16)))
            top = sorted(p32, reverse=True)
            if p32.index(max(p32)) != p16.index(max(p16)):
                flips.append((f"{name}/{qname}", (top[0] - top[1]) if len(top) > 1 else 1.0))
    diffs = torch.tensor(diffs)
    logger.info(
        f"HF bf16 vs fp32 over {len(diffs)} questions: max |dp| {diffs.max():.4f}, mean {diffs.mean():.4f}, "
        f"p95 {diffs.quantile(0.95):.4f}, argmax flips {flips}"
    )
    assert all(gap < NEAR_TIE for _, gap in flips), f"bf16 flips a decision that is not a near-tie: {flips}"
