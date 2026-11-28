# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A run's optimizers: which trainable weights each trains (``assign_parameters``), the config's rule that at most
one of them takes the remaining weights, and ``CombinedOptimizer``, which steps them as one. Device-free."""

import re

from pydantic import ValidationError

from ttml.experimental.errors import UnsupportedTrainingConfig
from ttml.experimental.trainer import AdamWConfig, TrainerConfig
from ttml.experimental.trainer.utils import CombinedOptimizer, assign_parameters

# The trainable weights of a LoRA Llama of one block, adapting qkv and gate/up, with the LM head trained too.
QKV_A, QKV_B = "Llama/blocks/0/attention/qkv_linear/lora_A", "Llama/blocks/0/attention/qkv_linear/lora_B"
GATE_UP_A, GATE_UP_B = "Llama/blocks/0/mlp/w_gate_up/lora_A", "Llama/blocks/0/mlp/w_gate_up/lora_B"
HEAD = "Llama/fc/weight"
NAMES = [QKV_A, QKV_B, GATE_UP_A, GATE_UP_B, HEAD]


def _optimizers(*modules):
    return [AdamWConfig(lr=1e-3, modules=m) for m in modules]


def test_one_optimizer_without_modules_trains_every_weight():
    assert assign_parameters(NAMES, _optimizers(None)) == [NAMES]


def test_a_weight_goes_to_the_first_optimizer_that_lists_one_of_its_modules():
    # qkv matches both of the first two optimizers and goes to the first; the rest go to the one without modules.
    optimizers = _optimizers(["attention"], ["qkv_linear", "mlp"], None)
    assert assign_parameters(NAMES, optimizers) == [[QKV_A, QKV_B], [GATE_UP_A, GATE_UP_B], [HEAD]]


def test_the_optimizer_without_modules_may_come_first():
    assert assign_parameters(NAMES, _optimizers(None, ["lora_B"])) == [[QKV_A, GATE_UP_A, HEAD], [QKV_B, GATE_UP_B]]


def test_modules_match_whole_parts_of_a_name_not_substrings(expect_error):
    # "lora" is a prefix of "lora_A" and "lora_B", and "w_gate" of "w_gate_up": neither is a part of any name.
    for modules in (["lora"], ["w_gate"]):
        message = re.escape(f"optimizer[0]: modules {modules} match no trainable weight")
        with expect_error(UnsupportedTrainingConfig, message):
            assign_parameters(NAMES, _optimizers(modules, None))


def test_a_weight_no_optimizer_trains_is_rejected(expect_error):
    with expect_error(UnsupportedTrainingConfig, f"^optimizer: no optimizer trains {QKV_B}, {GATE_UP_B}, {HEAD};"):
        assign_parameters(NAMES, _optimizers(["lora_A"]))
    # Past three, the rest are counted.
    with expect_error(UnsupportedTrainingConfig, f"no optimizer trains {QKV_A}, {QKV_B}, {GATE_UP_A} and 2 more;"):
        assign_parameters(NAMES, _optimizers(["nothing"]))


def test_the_optimizer_without_modules_must_have_weights_left_to_train(expect_error):
    with expect_error(UnsupportedTrainingConfig, r"optimizer\[1\]: the other optimizers train every weight"):
        assign_parameters(NAMES, _optimizers(["blocks", "fc"], None))


def _config(optimizer):
    return TrainerConfig(model={"name": "stub"}, dataset={"id": "stub"}, optimizer=optimizer, batch_size=1)


def test_at_most_one_optimizer_may_leave_modules_unset(expect_error):
    with expect_error(ValidationError, "at most one optimizer may leave modules unset"):
        _config([{"lr": 1e-3}, {"lr": 1e-4}])
    with expect_error(ValidationError, "at most one optimizer may leave modules unset"):
        _config([{"lr": 1e-3}, {"lr": 1e-4, "modules": ["fc"]}, {"lr": 1e-5}])

    config = _config([{"lr": 1e-3, "modules": ["attention"]}, {"lr": 1e-4, "modules": ["mlp"]}, {"lr": 1e-5}])
    assert [optimizer.modules for optimizer in config.optimizer] == [["attention"], ["mlp"], None]


def test_a_single_optimizer_is_a_list_of_one():
    (optimizer,) = _config({"lr": 1e-3}).optimizer
    assert isinstance(optimizer, AdamWConfig) and optimizer.modules is None


class _Optimizer:
    """Records the calls it gets into ``calls``, with its name."""

    def __init__(self, name, lrs, calls):
        self.name = name
        self.lrs = lrs
        self.calls = calls
        self.loaded = None

    def step(self):
        self.calls.append((self.name, "step"))

    def zero_grad(self):
        self.calls.append((self.name, "zero_grad"))

    def get_group_lrs(self):
        return list(self.lrs)

    def get_state_dict(self):
        return {"name": self.name}

    def set_state_dict(self, state):
        self.loaded = state


def test_combined_optimizer_steps_and_zeroes_every_optimizer_in_order():
    calls = []
    combined = CombinedOptimizer([_Optimizer("first", [1e-3], calls), _Optimizer("second", [1e-4], calls)])
    combined.step()
    combined.zero_grad()
    assert calls == [("first", "step"), ("second", "step"), ("first", "zero_grad"), ("second", "zero_grad")]


def test_combined_optimizer_lists_every_optimizers_group_lrs_in_order():
    combined = CombinedOptimizer([_Optimizer("first", [1e-3, 2e-3], []), _Optimizer("second", [1e-4], [])])
    assert combined.get_group_lrs() == [1e-3, 2e-3, 1e-4]


def test_combined_optimizer_state_dict_nests_each_optimizers_under_its_index():
    saved = CombinedOptimizer([_Optimizer("first", [], []), _Optimizer("second", [], [])]).get_state_dict()
    assert saved == {"0": {"name": "first"}, "1": {"name": "second"}}

    first, second = _Optimizer("a", [], []), _Optimizer("b", [], [])
    CombinedOptimizer([first, second]).set_state_dict(saved)
    assert (first.loaded, second.loaded) == ({"name": "first"}, {"name": "second"})
