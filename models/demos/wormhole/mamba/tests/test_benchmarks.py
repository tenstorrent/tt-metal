# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.demos.wormhole.mamba.reference.decode_model import MambaDecode, MambaPretrainedModelName
from models.demos.wormhole.mamba.benchmarks.loglikelihood import (
    compute_loglikelihood,
    compute_loglikelihood_given_prompt_and_target,
)
from models.utility_functions import skip_for_grayskull

from transformers import AutoTokenizer


@skip_for_grayskull()
def test_loglikelihood():
    logits = torch.tensor([[0.5, 1.0], [1.5, 2.5]]).reshape(1, 2, 2)
    labels = torch.tensor([[0, 1]]).reshape(1, 2, 1)
    probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B x L x VOCAB)
    assert compute_loglikelihood(logits, labels) == probs[0, 0, 0] + probs[0, 1, 1]


def get_compute(model_version: MambaPretrainedModelName):
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = MambaDecode.from_pretrained(model_version)
    model.eval()

    def compute(context, target):
        with torch.no_grad():
            context_ids = tokenizer(context, return_tensors="pt").input_ids  # (1 x CONTEXT_LEN)
            assert len(context_ids.shape) == 2 and context_ids.shape[1] > 0, "Expected at least one context token"

            target_ids = tokenizer(target, return_tensors="pt").input_ids  # (1 x TARGET_LEN)
            assert len(target_ids.shape) == 2 and target_ids.shape[1] > 0, "Expected at least one target token"

            return compute_loglikelihood_given_prompt_and_target(context_ids, target_ids, model, tokenizer.vocab_size)

    return compute


@skip_for_grayskull()
@pytest.mark.parametrize(
    "model_version",
    (("state-spaces/mamba-370m"), ("state-spaces/mamba-2.8b")),
)
def test_loglikelihood_from_prompt(model_version: MambaPretrainedModelName):
    compute = get_compute(model_version)

    llh1, greedy1 = compute("Mamba is the ", "x x x x x")
    llh2, greedy2 = compute("Mamba is the ", "something something something something something")
    llh3, greedy3 = compute("Mamba is the ", "this is really really wrong")
    llh4, greedy4 = compute("Mamba is the ", "first game to be released")
    llh5, greedy5 = compute("Mamba is the ", "first game to be released")

    assert llh1 < llh4, f"Expected {llh1} < {llh4}"
    assert llh2 < llh4, f"Expected {llh2} < {llh4}"
    assert llh3 < llh4, f"Expected {llh3} < {llh4}"
    assert llh4 == llh5, "Identical queries should match"
    assert greedy1 == greedy2 == greedy3
    assert greedy4 == greedy5
