# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-Python helpers from the ``examples/train`` entry point; no device needed."""

from __future__ import annotations

import os
import sys
import types

import pytest

# Not part of the installed ``ttml`` wheel. examples/qwen3 also ships a top-level ``train`` module and
# other tests put that dir on sys.path, so force ours to the front and evict a stale cache hit.
_TRAIN_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sources", "examples", "train")
)
if _TRAIN_EXAMPLE_DIR in sys.path:
    sys.path.remove(_TRAIN_EXAMPLE_DIR)
sys.path.insert(0, _TRAIN_EXAMPLE_DIR)
for _name in ("train", "callbacks"):
    _cached = sys.modules.get(_name)
    if _cached is not None and not os.path.realpath(getattr(_cached, "__file__", "") or "").startswith(
        _TRAIN_EXAMPLE_DIR
    ):
        del sys.modules[_name]

from callbacks import EpochCallback  # noqa: E402
from train import TrainingConfig, resolve_effective_max_steps, resolve_warmup_steps  # noqa: E402


def _config(**training_config) -> TrainingConfig:
    return TrainingConfig({"training_config": training_config})


# ── resolve_warmup_steps ──────────────────────────────────────────────────────


def test_warmup_defaults_to_ratio_of_schedule():
    assert resolve_warmup_steps(_config(), 1000) == 100


def test_warmup_ratio_is_configurable():
    assert resolve_warmup_steps(_config(warmup_ratio=0.25), 1000) == 250


def test_explicit_warmup_steps_overrides_ratio():
    assert resolve_warmup_steps(_config(warmup_ratio=0.25, warmup_steps=7), 1000) == 7


def test_warmup_steps_zero_means_no_warmup():
    # Distinct from unset: 0 must not fall through to warmup_ratio.
    assert resolve_warmup_steps(_config(warmup_ratio=0.25, warmup_steps=0), 1000) == 0


def test_warmup_is_clamped_to_the_schedule_length():
    assert resolve_warmup_steps(_config(warmup_steps=5000), 1000) == 1000


def test_negative_warmup_steps_clamps_to_zero():
    assert resolve_warmup_steps(_config(warmup_steps=-5), 1000) == 0


# ── resolve_effective_max_steps ───────────────────────────────────────────────


def test_step_cap_alone_sets_the_run_length():
    assert resolve_effective_max_steps(_config(max_steps=1000), steps_per_epoch=10.0) == 1000


def test_epoch_cap_alone_sets_the_run_length():
    assert resolve_effective_max_steps(_config(max_steps=0, num_epochs=3), steps_per_epoch=10.0) == 30


def test_epoch_cap_rounds_up_to_cover_the_last_partial_step():
    assert resolve_effective_max_steps(_config(max_steps=0, num_epochs=1), steps_per_epoch=10.9) == 11


def test_epoch_cap_is_at_least_one_step():
    assert resolve_effective_max_steps(_config(max_steps=0, num_epochs=1), steps_per_epoch=0.2) == 1


@pytest.mark.parametrize(
    ("max_steps", "num_epochs", "expected"),
    [(1000, 3, 30), (10, 3, 10)],
)
def test_the_first_of_the_two_caps_wins(max_steps, num_epochs, expected):
    cfg = _config(max_steps=max_steps, num_epochs=num_epochs)
    assert resolve_effective_max_steps(cfg, steps_per_epoch=10.0) == expected


def test_no_cap_at_all_is_rejected(expect_error):
    with expect_error(ValueError, "No stop condition"):
        resolve_effective_max_steps(_config(max_steps=0, num_epochs=0), steps_per_epoch=10.0)


# ── EpochCallback ─────────────────────────────────────────────────────────────


def _run_steps(callback: EpochCallback, steps, start_step: int = 0) -> None:
    trainer = types.SimpleNamespace(step=start_step)
    callback.on_train_begin(trainer)
    for step in steps:
        callback.on_step_end(trainer, step, 0.0, 1e-3)


def test_epoch_printed_once_per_crossing(capsys):
    callback = EpochCallback(steps_per_epoch=4.0)
    _run_steps(callback, range(1, 9))
    assert capsys.readouterr().out.splitlines() == ["Epoch 1 completed", "Epoch 2 completed"]


def test_resuming_does_not_replay_completed_epochs(capsys):
    callback = EpochCallback(steps_per_epoch=4.0)
    _run_steps(callback, range(9, 13), start_step=8)
    assert capsys.readouterr().out.splitlines() == ["Epoch 3 completed"]


def test_multiple_epochs_in_one_step_are_all_reported(capsys):
    # Corpus smaller than a single step's token budget.
    callback = EpochCallback(steps_per_epoch=0.5)
    _run_steps(callback, [1])
    assert capsys.readouterr().out.splitlines() == ["Epoch 1 completed", "Epoch 2 completed"]
