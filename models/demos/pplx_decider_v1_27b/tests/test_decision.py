# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from models.demos.pplx_decider_v1_27b.tests.common import ids_sha256, load_example_state
from models.demos.pplx_decider_v1_27b.tt.decision import (
    DecisionConfig,
    answer,
    decision_probabilities,
    options,
    render_input_ids,
)
from models.demos.pplx_decider_v1_27b.tt.weight_mapping import resolve_checkpoint

HERE = Path(__file__).parent
REFERENCE_PATH = HERE / "reference" / "decision_reference.json"
EXAMPLES = {e["name"]: e for e in json.loads((HERE / "examples.json").read_text())}


@pytest.fixture(scope="module")
def ckpt_dir():
    try:
        return resolve_checkpoint()
    except Exception as e:
        pytest.skip(f"pplx-decider checkpoint is not available: {e}")


@pytest.fixture(scope="module")
def config(ckpt_dir):
    return DecisionConfig.from_checkpoint(ckpt_dir)


@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_PATH.is_file():
        pytest.skip(f"{REFERENCE_PATH} is missing; run tests/generate_reference.py")
    return json.loads(REFERENCE_PATH.read_text())


def assert_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            assert_close(actual[key], value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-5)
    else:
        assert actual == expected


def reference_entry(reference, name):
    return next(e for e in reference["examples"] if e["name"] == name)


def test_config_loads(config):
    assert len(config.codes) == 255
    assert len(set(config.token_ids)) == 255
    assert config.temperature > 0


@pytest.mark.parametrize("name", EXAMPLES)
def test_render_input_ids(name, ckpt_dir, config, reference):
    example = EXAMPLES[name]
    ref = reference_entry(reference, name)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    ids = render_input_ids(tokenizer, load_example_state(example, ckpt_dir), example["question"], config.codes)
    assert len(ids) == ref["num_tokens"]
    assert ids_sha256(ids) == ref["input_ids_sha256"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_reference_probabilities_and_answer(name, reference):
    ref = reference_entry(reference, name)
    question = EXAMPLES[name]["question"]
    n = len(options(question)[0])
    probabilities = decision_probabilities(torch.tensor(ref["readout_logits"]), n, reference["temperature"])
    assert torch.allclose(probabilities, torch.tensor(ref["probabilities"]), atol=1e-6)
    assert_close(answer(question, probabilities.tolist()), ref["answer"])


def test_answer_choice():
    question = {"type": "choice", "criteria": {"a": "first", "b": None, "c": "third"}}
    result = answer(question, [0.1, 0.7, 0.2])
    assert result["choice"] == "b"
    assert result["probabilities"] == pytest.approx({"a": 0.1, "b": 0.7, "c": 0.2})
    assert result["confidence"] == pytest.approx((0.7 - 1 / 3) / (1 - 1 / 3))


def test_answer_noul():
    result = answer({"type": "noul"}, [0.25, 0.75])
    assert result == {"type": "noul", "noul": pytest.approx(0.75)}


def test_answer_score():
    question = {"type": "score", "criteria": ["low", "mid", "high"]}
    result = answer(question, [0.0, 0.5, 0.5])
    assert result["score"] == pytest.approx(1.5)
    assert result["legend"] == {"0": "low", "1": "mid", "2": "high"}
    assert result["confidence"] == pytest.approx(1.0 - 0.5 / (2 / 3))


def test_answer_rejects_wrong_length(expect_error):
    with expect_error(ValueError, "Each option must have a probability"):
        answer({"type": "noul"}, [1.0])
