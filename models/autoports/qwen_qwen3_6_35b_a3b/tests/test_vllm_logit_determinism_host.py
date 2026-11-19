# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for strict HTTP token/logprob comparison and boundary ties."""

from copy import deepcopy

import pytest
import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_logit_determinism import (
    ATOL,
    compact_logits,
    compare_steps,
    response_steps,
)


def fixture():
    expected = [compact_logits(torch.arange(64, dtype=torch.float32)) for _ in range(2)]
    logprobs = dict(
        tokens=[f"token_id:{step['token_id']}" for step in expected],
        token_logprobs=[step["token_logprob"] for step in expected],
        top_logprobs=[{f"token_id:{key}": value for key, value in step["top_logprobs"].items()} for step in expected],
    )
    body = dict(choices=[dict(logprobs=logprobs, finish_reason="length", token_ids=[63, 63])])
    return expected, body


def test_http_scores_match_full_logits_and_fail_beyond_absolute_tolerance():
    expected, body = fixture()
    actual = response_steps(body)
    assert compare_steps(expected, actual, "test") == 0
    actual[1]["top_logprobs"]["63"] += ATOL * 2
    with pytest.raises(AssertionError):  # allow-pytest.raises: CPU-only validation, no device error state
        compare_steps(expected, actual, "test")


def test_tokens_and_unique_id_keys_are_required():
    expected, body = fixture()
    invalid = deepcopy(body)
    invalid["choices"][0]["logprobs"]["tokens"][0] = "ordinary token text"
    with pytest.raises(AssertionError, match="unambiguous"):  # allow-pytest.raises: CPU-only validation
        response_steps(invalid)
    actual = response_steps(body)
    actual[0]["token_id"] = 62
    with pytest.raises(AssertionError):  # allow-pytest.raises: CPU-only validation, no device error state
        compare_steps(expected, actual, "test")


def test_rank20_boundary_tie_accepts_only_tied_replacement():
    logits = torch.arange(64, dtype=torch.float32)
    logits[43] = logits[44]  # Ranks20/21 tie; either is a legitimate top20 tail.
    expected = [compact_logits(logits) for _ in range(2)]
    actual = [{key: deepcopy(step[key]) for key in ("token_id", "token_logprob", "top_logprobs")} for step in expected]
    selected = "43" if "43" in actual[0]["top_logprobs"] else "44"
    alternate = "44" if selected == "43" else "43"
    score = actual[0]["top_logprobs"].pop(selected)
    actual[0]["top_logprobs"][alternate] = score
    assert compare_steps(expected, actual, "tie") == 0
    actual[0]["top_logprobs"]["42"] = actual[0]["top_logprobs"].pop(alternate)
    with pytest.raises(AssertionError, match="unexpected"):  # allow-pytest.raises: CPU-only validation
        compare_steps(expected, actual, "bad")


def test_missing_second_token_or_top20_is_not_a_pass():
    _, body = fixture()
    body["choices"][0]["logprobs"]["top_logprobs"].pop()
    with pytest.raises(AssertionError, match="two-token"):  # allow-pytest.raises: CPU-only validation
        response_steps(body)
