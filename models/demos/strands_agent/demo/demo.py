# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""strands-decider demo: answers every question in tests/examples.json on one Blackhole chip."""

import json
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.common.utility_functions import run_for_blackhole
from models.demos.strands_agent.tests.common import load_example_state, parametrize_mesh
from models.demos.strands_agent.tt.model import StrandsDecider
from models.demos.strands_agent.tt.prompting import encode_request

EXAMPLES_PATH = Path(__file__).parent.parent / "tests" / "examples.json"


@torch.no_grad()
@run_for_blackhole()
@pytest.mark.timeout(1800)
@parametrize_mesh()
def test_strands_decider_demo(mesh_device):
    examples = json.loads(EXAMPLES_PATH.read_text())
    start = time.perf_counter()
    decider = StrandsDecider.from_pretrained(mesh_device)
    logger.info(f"Model load time: {time.perf_counter() - start:.1f} s")

    for example in examples:
        state = load_example_state(example, decider.args.adapter_dir)
        for q in encode_request(decider.tokenizer, state, example["questions"], decider.config.max_length):
            # Same path as decider.evaluate, one question at a time for per-question timing.
            start = time.perf_counter()
            result = decider.evaluate(state, {q.name: example["questions"][q.name]})
            ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"{example['name']}/{q.name} ({len(q.input_ids)} tokens, {ms:.0f} ms): {json.dumps(result['answers'][q.name])}"
            )
