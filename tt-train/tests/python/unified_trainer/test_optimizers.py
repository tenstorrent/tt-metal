# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Which optimizer trains which weight, and stepping them as one. Device-free."""

import pytest

from ttml.unified_trainer import AdamWConfig, UnsupportedTrainingConfig
from ttml.unified_trainer.utils import CombinedOptimizer, assign_parameters

QKV_A = "llama/blocks/0/attention/qkv_linear/lora_A"
QKV_B = "llama/blocks/0/attention/qkv_linear/lora_B"
OUT_A = "llama/blocks/0/attention/out_linear/lora_A"
NORM = "llama/blocks/0/attention_norm/gamma"
NAMES = [QKV_A, QKV_B, OUT_A, NORM]


def adamw(modules=None):
    return AdamWConfig(lr=1e-4, modules=modules)


def test_one_optimizer_without_modules_trains_every_weight():
    assert assign_parameters(NAMES, [adamw()]) == [NAMES]


def test_an_optimizer_trains_every_weight_its_modules_match():
    assert assign_parameters(NAMES, [adamw(["qkv_linear"]), adamw()]) == [[QKV_A, QKV_B], [OUT_A, NORM]]


def test_the_optimizer_without_modules_takes_the_rest_wherever_it_is_listed():
    assert assign_parameters(NAMES, [adamw(), adamw(["gamma"])]) == [[QKV_A, QKV_B, OUT_A], [NORM]]


def test_a_weight_goes_to_the_first_optimizer_that_matches_it():
    assigned = assign_parameters(NAMES, [adamw(["attention"]), adamw(["qkv_linear", "gamma"])])
    assert assigned == [[QKV_A, QKV_B, OUT_A], [NORM]]


def test_modules_match_whole_name_parts(expect_error):
    # "attention" is not a part of "attention_norm/gamma".
    with expect_error(UnsupportedTrainingConfig, f"no optimizer trains {NORM}"):
        assign_parameters(NAMES, [adamw(["attention"])])


def test_every_trainable_weight_needs_an_optimizer(expect_error):
    with expect_error(UnsupportedTrainingConfig, "no optimizer trains .*out_linear.* and 1 more"):
        assign_parameters(NAMES + [NORM + "2", NORM + "3"], [adamw(["qkv_linear"])])


@pytest.mark.parametrize(
    "optimizers, error",
    [
        ([adamw(["tok_emb"]), adamw()], r"optimizer\[0\]: modules \['tok_emb'\] match no trainable weight"),
        ([adamw(["attention"]), adamw(["gamma"]), adamw()], r"optimizer\[2\]: the other optimizers train every"),
    ],
)
def test_every_optimizer_needs_a_weight(optimizers, error, expect_error):
    with expect_error(UnsupportedTrainingConfig, error):
        assign_parameters(NAMES, optimizers)


class FakeOptimizer:
    def __init__(self, lrs):
        self.lrs, self.steps, self.zero_grads = lrs, 0, 0

    def step(self):
        self.steps += 1

    def zero_grad(self):
        self.zero_grads += 1

    def get_group_lrs(self):
        return self.lrs


def test_the_combined_optimizer_steps_and_zeroes_every_optimizer():
    optimizers = [FakeOptimizer([1e-3]), FakeOptimizer([1e-4, 1e-5])]
    combined = CombinedOptimizer(optimizers)
    combined.step()
    combined.zero_grad()
    assert [(o.steps, o.zero_grads) for o in optimizers] == [(1, 1), (1, 1)]
    assert combined.get_group_lrs() == [1e-3, 1e-4, 1e-5]
