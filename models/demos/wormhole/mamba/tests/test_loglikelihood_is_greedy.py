# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: is_greedy follows the lm-eval-harness definition (argmax prediction equals the next target token)."""

import torch

from models.demos.wormhole.mamba.benchmarks.loglikelihood import compute_loglikelihood_given_prompt_and_target

VOCAB = 10


class _NextTokenModel:
    """Fake decode model: one-hot logits that predict `(token + shift) % VOCAB` as the next token."""

    def __init__(self, shift):
        self.shift = shift

    def initialize_states(self):
        pass

    def __call__(self, tokens):  # (B x 1) => (B x 1 x VOCAB)
        return torch.nn.functional.one_hot((tokens + self.shift) % VOCAB, VOCAB).float() * 10.0


def test_is_greedy_true_when_model_predicts_every_target_token():
    context_ids = torch.tensor([[7, 1, 2]])
    target_ids = torch.tensor([[3, 4]])
    _, is_greedy = compute_loglikelihood_given_prompt_and_target(context_ids, target_ids, _NextTokenModel(1), VOCAB)
    assert is_greedy is True


def test_is_greedy_false_when_a_target_token_is_missed():
    context_ids = torch.tensor([[7, 1, 2]])
    target_ids = torch.tensor([[3, 5]])
    _, is_greedy = compute_loglikelihood_given_prompt_and_target(context_ids, target_ids, _NextTokenModel(1), VOCAB)
    assert is_greedy is False


def test_is_greedy_ignores_context_positions():
    # Context transitions (7 -> 1) do not follow the model's rule, but only the target span counts.
    context_ids = torch.tensor([[7, 1, 2]])
    target_ids = torch.tensor([[3]])
    _, is_greedy = compute_loglikelihood_given_prompt_and_target(context_ids, target_ids, _NextTokenModel(1), VOCAB)
    assert is_greedy is True
