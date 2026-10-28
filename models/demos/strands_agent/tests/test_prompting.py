# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only: prompt encoding and answer building match the reference JSON."""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from models.demos.strands_agent.tests.common import ids_sha256, load_example_state
from models.demos.strands_agent.tt.answers import build_answer
from models.demos.strands_agent.tt.head import load_config
from models.demos.strands_agent.tt.prompting import encode_request
from models.demos.strands_agent.tt.weight_mapping import resolve_checkpoint

TESTS_DIR = Path(__file__).parent
REFERENCE_PATH = TESTS_DIR / "reference" / "decider_reference.json"
EXAMPLES_PATH = TESTS_DIR / "examples.json"


@pytest.mark.skipif(not REFERENCE_PATH.exists(), reason=f"{REFERENCE_PATH} is missing")
def test_prompting_matches_reference():
    ckpt = resolve_checkpoint()
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    cfg = load_config(ckpt)
    refs = {e["name"]: e["questions"] for e in json.loads(REFERENCE_PATH.read_text())["examples"]}
    for example in json.loads(EXAMPLES_PATH.read_text()):
        state = load_example_state(example, ckpt)
        encoded = encode_request(tokenizer, state, example["questions"], cfg.max_length)
        assert [q.name for q in encoded] == list(refs[example["name"]])
        for q in encoded:
            ref = refs[example["name"]][q.name]
            tag = f"{example['name']}/{q.name}"
            assert ids_sha256(q.input_ids) == ref["input_ids_sha256"], tag
            assert q.query_position == ref["query_position"], tag
            assert q.option_positions == ref["option_positions"], tag
            assert q.labels == ref["labels"], tag
            assert q.kind == ref["kind"], tag
            answer = build_answer(
                q,
                torch.tensor(ref["logits"], dtype=torch.float32),
                cfg.temperature_by_kind,
                cfg.temperature,
                cfg.ordinal_smoothing,
            )
            assert answer == ref["answer"], tag


@pytest.mark.parametrize(
    "kind, spec",
    [
        ("noul", {"instructions": "Is it raining?"}),
        ("noul", {"instructions": "Is it raining?", "criteria": {"true": "wet", "false": "dry"}}),
        ("choice", {"instructions": "Pick.", "criteria": {"a": "first", "b": None, "c": "third"}}),
        ("score", {"instructions": "Rate.", "criteria": ["low", "mid", "high"]}),
    ],
)
def test_missing_type_resolves_like_upstream_union(kind, spec):
    from models.demos.strands_agent.tt.prompting import render_question, resolve_kind

    assert resolve_kind(spec) == kind
    assert render_question("q", spec) == render_question("q", {"type": kind, **spec})
