# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The scheduler, dataloader and RNG block of a checkpoint, without a device.

Every generator comes back exactly; the ttml generator is restored only from this rank's own entry; scheduler and
dataloader state round-trip through header.json and must be JSON; what was asked for but not stored is named.
"""

import json
import random
import sys

import numpy as np
import pytest
import torch

from dcp_oracle import Loader

import ttml
from ttml.common.schedulers import CosineAnnealingScheduler
from ttml.distributed_checkpointing._errors import CheckpointError, VersionError
from ttml.distributed_checkpointing._index import read_header_json, write_header
from ttml.distributed_checkpointing._train_state import (
    TRAIN_STATE_KEY,
    capture_rng,
    capture_train_state,
    check_train_state,
    missing_train_state,
    restore_rng,
    restore_train_state,
    this_world_rank,
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


def test_every_generator_comes_back_exactly():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(11)
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    ttml.init.manual_seed(15)
    state = capture_rng(world_rank=0)
    assert json.loads(json.dumps(state)) == state, "plain JSON, nothing numpy leaks into it"
    assert set(state) == {"format", "ranks"} and list(state["ranks"]) == ["0"]
    own = state["ranks"]["0"]
    assert set(own) == {"ttml", "python", "numpy", "ttml_init", "torch"}
    assert len(own["python"]["state"]) == 625 * 8 and len(own["numpy"]["state"]) == 624 * 8  # hex of u32
    expected = (
        random.random(),
        float(np.random.random()),
        torch.rand(3).tolist(),
        ttml.init._rng.random(2).tolist(),  # what the next layer's initial weights would be drawn from
        ctx.get_generator_state(),
    )

    ctx.set_seed(99)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ttml.init.manual_seed(0)
    assert ctx.get_generator_state() != own["ttml"]
    assert restore_rng(state, world_rank=0) == ["rng.ttml", "rng.python", "rng.numpy", "rng.ttml_init", "rng.torch"]
    assert ctx.get_generator_state() == own["ttml"]
    # The same draws again, and the ttml generator reaches the same state through them (nothing drew from it).
    assert (
        random.random(),
        float(np.random.random()),
        torch.rand(3).tolist(),
        ttml.init._rng.random(2).tolist(),
        ctx.get_generator_state(),
    ) == expected


def test_a_generator_without_stored_state_keeps_its_state_with_a_warning(monkeypatch):
    block = capture_rng(world_rank=0)
    without_numpy = json.loads(json.dumps(block))
    del without_numpy["ranks"]["0"]["numpy"]  # saved while numpy's global generator was not MT19937
    np.random.seed(7)
    numpy_before = np.random.get_state()[1].copy()
    with pytest.warns(UserWarning, match=r"no state for these generators, which keep their current state: \['numpy'\]"):
        restored = restore_rng(without_numpy, world_rank=0)
    assert "rng.numpy" not in restored and np.array_equal(np.random.get_state()[1], numpy_before)

    monkeypatch.setitem(sys.modules, "torch", None)  # torch's state stored, torch not imported in this process
    with pytest.warns(UserWarning, match="holds torch's generator state, but torch is not imported: not restored"):
        restored = restore_rng(block, world_rank=0)
    assert restored == ["rng.ttml", "rng.python", "rng.numpy", "rng.ttml_init"]


def test_capture_and_restore_default_to_this_process_world_rank():
    state = capture_rng()  # one process: world rank 0
    assert list(state["ranks"]) == [str(this_world_rank())] == ["0"]
    assert restore_rng(state)[0] == "rng.ttml"


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
    loader = Loader()
    for _ in range(3):
        sched.step()
        loader.advance()
    state = capture_train_state(scheduler=sched, dataloader=loader, rng=False)
    assert set(state) == {"scheduler", "dataloader"} and state["dataloader"] == {"position": 3}
    write_header(str(tmp_path), {"user": {"step": 3}, TRAIN_STATE_KEY: state})
    back = read_header_json(str(tmp_path))[TRAIN_STATE_KEY]
    assert back == state

    opt2 = _Opt([1e-3, 2e-3])
    sched2 = CosineAnnealingScheduler(opt2, T_max=10)
    loader2 = Loader()
    assert missing_train_state(back, scheduler=sched2, dataloader=loader2, rng=True) == ["train_state/rng"]
    assert restore_train_state(back, scheduler=sched2, dataloader=loader2, rng=True) == ["scheduler", "dataloader"]
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

    with expect_error(CheckpointError, "dataloader state at /position: a int64 has no JSON form"):
        capture_train_state(dataloader=_Bad(), rng=False)


class _RefusingScheduler:
    def get_state_dict(self) -> dict:
        return {}

    def set_state_dict(self, state: dict) -> None:
        raise KeyError("m_step_size")  # what a StepScheduler raises for a cosine scheduler's state


def test_a_refused_state_stops_the_restore_before_anything_else_moves(expect_error):
    loader = Loader()
    stored = {"scheduler": {"m_last_step": 3}, "dataloader": {"position": 5}}
    with expect_error(CheckpointError, "the live scheduler refused the stored scheduler state"):
        restore_train_state(stored, scheduler=_RefusingScheduler(), dataloader=loader)
    assert loader.position == 0, "the scheduler is restored first, so the loader has not moved"


def test_a_component_that_cannot_take_a_state_is_refused_before_anything_is_placed(expect_error):
    stored = {"scheduler": {"m_last_step": 3}, "dataloader": {"position": 5}}
    with expect_error(TypeError, r"the dataloader passed \(list_iterator\) has no set_state_dict"):
        check_train_state(stored, dataloader=iter([]))
    check_train_state(stored, scheduler=_RefusingScheduler(), dataloader=Loader())  # both can take a state
    check_train_state({}, dataloader=iter([]))  # nothing stored for it: nothing to restore it with


def test_rng_state_is_checked_whole_before_any_generator_is_set(expect_error):
    random.seed(21)
    block = capture_rng(world_rank=0)
    before = random.getstate()
    broken = json.loads(json.dumps(block))
    broken["ranks"]["0"]["numpy"]["state"] = "not hex"
    with expect_error(CheckpointError, "the stored RNG state is malformed"):
        check_train_state({"rng": broken}, rng=True, world_rank=0)
    with expect_error(CheckpointError, "the stored RNG state is malformed"):
        restore_rng(broken, world_rank=0)
    assert random.getstate() == before, "python's generator was not set before numpy's failed to decode"
    newer = dict(block, format=2)
    with expect_error(VersionError, "newer than this reader"):
        check_train_state({"rng": newer}, rng=True, world_rank=0)
    # The whole train state passed where its RNG block belongs, or a block with a bad format, restores nothing
    # silently no more.
    with expect_error(CheckpointError, r"not an RNG block \(found \['dataloader', 'rng'\]\)"):
        restore_rng({"dataloader": {"position": 1}, "rng": block}, world_rank=0)
    with expect_error(CheckpointError, "RNG state format 'x' is not a version number"):
        restore_rng(dict(block, format="x"), world_rank=0)
    check_train_state({"rng": block}, rng=False)  # not asked for: nothing to check


def _shorten_hex(text: str, nbytes: int) -> str:
    return text[: -2 * nbytes]


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param(
            lambda own: own["numpy"].update(state=_shorten_hex(own["numpy"]["state"], 4)), id="numpy_623_words"
        ),
        pytest.param(
            lambda own: own["python"].update(state=_shorten_hex(own["python"]["state"], 4)), id="python_624_words"
        ),
        pytest.param(lambda own: own.update(torch=_shorten_hex(own["torch"], 1)), id="torch_one_byte_short"),
        pytest.param(lambda own: own.update(ttml=" ".join(own["ttml"].split()[:600])), id="ttml_600_tokens"),
    ],
)
def test_rng_state_of_the_wrong_length_changes_no_generator(damage, expect_error):
    # Valid JSON and hex, the wrong length: what each setter, not the decoder, used to find out, after the generators
    # before it in the restore order were already set.
    ctx = ttml.autograd.AutoContext.get_instance()
    block = capture_rng(world_rank=0)
    broken = json.loads(json.dumps(block))
    damage(broken["ranks"]["0"])
    before = (ctx.get_generator_state(), random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    with expect_error(CheckpointError, "the stored RNG state is malformed"):
        restore_rng(broken, world_rank=0)
    after = (ctx.get_generator_state(), random.getstate(), np.random.get_state()[1], torch.get_rng_state())
    assert after[0] == before[0] and after[1] == before[1], "ttml and python untouched"
    assert np.array_equal(after[2], before[2]) and torch.equal(after[3], before[3]), "numpy and torch untouched"


class _IntKeyLoader(Loader):
    def get_state_dict(self) -> dict:
        return {"per_shard": {0: 7}}


def test_state_that_would_not_read_back_as_written_is_refused(expect_error):
    with expect_error(CheckpointError, "dataloader state at /per_shard: key 0 is a int"):
        capture_train_state(dataloader=_IntKeyLoader(), rng=False)
