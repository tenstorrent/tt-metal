# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The scheduler, dataloader and RNG block of a checkpoint, without a device.

Every generator comes back exactly; the ttml generator is restored only from this rank's own entry; scheduler and
dataloader state round-trip through header.json and must be JSON; what was asked for but not stored is named.
"""

import json
import random

import numpy as np
import pytest
import torch

import ttml
from ttml.common.schedulers import CosineAnnealingScheduler
from ttml.distributed_checkpointing._errors import CheckpointError
from ttml.distributed_checkpointing._index import read_header, write_header
from ttml.distributed_checkpointing._train_state import (
    TRAIN_STATE_KEY,
    capture_rng,
    capture_train_state,
    missing_train_state,
    restore_rng,
    restore_train_state,
)


class _Opt:
    """The three LR accessors a scheduler needs, with no device behind them."""

    def __init__(self, lrs):
        self._initial = list(lrs)
        self._lrs = list(lrs)

    def get_initial_group_lrs(self):
        return list(self._initial)

    def get_group_lrs(self):
        return list(self._lrs)

    def set_group_lrs(self, lrs):
        self._lrs = list(lrs)


class _Loader:
    def __init__(self) -> None:
        self.position = 0

    def advance(self) -> None:
        self.position += 1

    def get_state_dict(self) -> dict:
        return {"position": self.position}

    def set_state_dict(self, state: dict) -> None:
        self.position = int(state["position"])


def test_every_generator_comes_back_exactly():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(11)
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    state = capture_rng(world_rank=0)
    assert json.loads(json.dumps(state)) == state, "plain JSON, nothing numpy leaks into it"
    assert set(state) == {"format", "ranks"} and list(state["ranks"]) == ["0"]
    own = state["ranks"]["0"]
    assert set(own) == {"ttml", "python", "numpy", "torch"}
    assert len(own["python"]["state"]) == 625 * 8 and len(own["numpy"]["state"]) == 624 * 8  # hex of u32
    expected = (random.random(), float(np.random.random()), torch.rand(3).tolist(), ctx.get_generator_state())

    ctx.set_seed(99)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    assert ctx.get_generator_state() != own["ttml"]
    assert restore_rng(state, world_rank=0) == ["rng.ttml", "rng.python", "rng.numpy", "rng.torch"]
    assert ctx.get_generator_state() == own["ttml"]
    # The same draws again, and the ttml generator reaches the same state through them (nothing drew from it).
    assert (random.random(), float(np.random.random()), torch.rand(3).tolist(), ctx.get_generator_state()) == expected


def test_a_rank_restores_only_its_own_entry():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(5)
    random.seed(5)
    state = capture_rng(world_rank=1)  # saved by world rank 1
    ctx.set_seed(6)
    random.seed(6)
    current = (ctx.get_generator_state(), random.getstate())
    with pytest.warns(UserWarning, match="no RNG state for world rank 0"):
        restored = restore_rng(state, world_rank=0)
    assert restored == [] and (ctx.get_generator_state(), random.getstate()) == current
    assert restore_rng(state, world_rank=1)[:2] == ["rng.ttml", "rng.python"]
    assert ctx.get_generator_state() == state["ranks"]["1"]["ttml"]


def test_scheduler_and_dataloader_round_trip_through_the_header(tmp_path):
    opt = _Opt([1e-3, 2e-3])
    sched = CosineAnnealingScheduler(opt, T_max=10)
    loader = _Loader()
    for _ in range(3):
        sched.step()
        loader.advance()
    state = capture_train_state(scheduler=sched, dataloader=loader, rng=False)
    assert set(state) == {"scheduler", "dataloader"} and state["dataloader"] == {"position": 3}
    write_header(str(tmp_path), {"user": {"step": 3}, TRAIN_STATE_KEY: state})
    back = read_header(str(tmp_path))[TRAIN_STATE_KEY]
    assert back == state

    opt2 = _Opt([1e-3, 2e-3])
    sched2 = CosineAnnealingScheduler(opt2, T_max=10)
    loader2 = _Loader()
    assert missing_train_state(back, scheduler=sched2, dataloader=loader2, rng=True) == ["train_state/rng"]
    assert restore_train_state(back, scheduler=sched2, dataloader=loader2, rng=True) == ["dataloader", "scheduler"]
    assert sched2.get_last_lrs() == sched.get_last_lrs() and opt2.get_group_lrs() == opt.get_group_lrs()
    sched.step()
    sched2.step()
    assert sched2.get_last_lrs() == sched.get_last_lrs()
    assert loader2.get_state_dict() == loader.get_state_dict()
    assert missing_train_state({}, scheduler=sched2) == ["train_state/scheduler"]
    assert restore_train_state({}, scheduler=sched2, dataloader=loader2, rng=True) == []


def test_state_that_is_not_json_is_refused(expect_error):
    class _Bad:
        def get_state_dict(self):
            return {"position": np.int64(3)}

    with expect_error(CheckpointError, "dataloader state is not JSON"):
        capture_train_state(dataloader=_Bad(), rng=False)
