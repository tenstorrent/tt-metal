# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The scheduler, dataloader and RNG block of a checkpoint, without a device.

Every generator comes back exactly; the ttml generator is restored only from this rank's own entry; scheduler and
dataloader state round-trip through header.json and must be JSON; what was asked for but not stored is named.
"""

import json
import os
import random
import sys

import numpy as np
import pytest
import torch

from dcp_oracle import MISREAD_RNG, WRONG_LENGTH_RNG, Loader, advance_generators

import ttml
from ttml.common.schedulers import CosineAnnealingScheduler, LinearScheduler, SequentialScheduler, StepScheduler
from ttml.distributed_checkpointing import _train_state
from ttml.distributed_checkpointing._errors import CheckpointError, IncompleteCheckpointError, VersionError
from ttml.distributed_checkpointing._index import read_header_json, write_header
from ttml.distributed_checkpointing._train_state import (
    TRAIN_STATE_KEY,
    capture_rng,
    capture_train_state,
    check_train_state,
    missing_train_state,
    read_train_state,
    restore_rng,
    restore_train_state,
    this_world_rank,
)


def _adamw(lrs):
    """A device-free AdamW with one param group per LR and no parameters, as test_schedulers.py builds it."""
    cfg = ttml.optimizers.AdamWConfig.make(lr=lrs[0], beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    groups = [{"params": [], "lr": lr} for lr in lrs[1:]]
    return ttml.optimizers.AdamW(ttml.NamedParameters(), cfg, param_groups=groups)


def test_every_generator_comes_back_exactly():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.set_seed(11)
    random.seed(12)
    np.random.seed(13)
    torch.manual_seed(14)
    ttml.init.manual_seed(15)
    advance_generators()
    state = capture_rng(world_rank=0)
    assert json.loads(json.dumps(state)) == state, "plain JSON, nothing numpy leaks into it"
    assert set(state) == {"format", "ranks"} and list(state["ranks"]) == ["0"]
    own = state["ranks"]["0"]
    assert set(own) == {"ttml", "python", "numpy", "ttml_init", "torch"}
    # Mid-stream, so the parts of each state that seeding leaves at their defaults must come back too.
    assert own["python"]["gauss_next"] is not None and own["numpy"]["has_gauss"] == 1 and own["numpy"]["pos"] != 624
    assert own["ttml_init"]["has_uint32"] == 1
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


def test_generator_without_stored_state_warns(monkeypatch):
    """A generator the checkpoint holds no state for keeps its state, with a warning."""
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
    opt = _adamw([1e-3, 2e-3])
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

    opt2 = _adamw([1e-3, 2e-3])
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


def test_a_header_without_a_train_state_object_is_refused(tmp_path, expect_error):
    ckpt = str(tmp_path / "step_0000001")
    os.makedirs(ckpt)
    envelope = {"format": "ttml-dcp", "version": 1, "user": {}}
    write_header(ckpt, {**envelope, TRAIN_STATE_KEY: {}})
    assert read_train_state(ckpt) == {}
    for header, found in ((envelope, "NoneType"), ({**envelope, TRAIN_STATE_KEY: ["rng"]}, "list")):
        write_header(ckpt, header)
        with expect_error(IncompleteCheckpointError, rf"has no train_state object \(found {found}\)"):
            read_train_state(ckpt)


@pytest.mark.parametrize("steps", [2, 7], ids=["mid_warmup", "into_cosine"])
def test_a_warmup_then_cosine_schedule_resumes_where_it_stopped(steps):
    """The common schedule: a linear warmup chained to a cosine decay, saved inside either child."""

    def schedule(opt):
        warmup = LinearScheduler(opt, start_factor=0.1, end_factor=1.0, total_steps=4)
        cosine = CosineAnnealingScheduler(opt, T_max=10)
        return SequentialScheduler(opt, [warmup, cosine], [4, 10])

    opt = _adamw([1e-3, 2e-3])
    sched = schedule(opt)
    for _ in range(steps):
        sched.step()
    stored = json.loads(json.dumps(capture_train_state(scheduler=sched, rng=False)))

    opt2 = _adamw([1e-3, 2e-3])
    sched2 = schedule(opt2)
    assert restore_train_state(stored, scheduler=sched2) == ["scheduler"]
    assert sched2.get_last_lrs() == sched.get_last_lrs() and opt2.get_group_lrs() == opt.get_group_lrs()
    for _ in range(6):  # across the milestone for the run saved mid-warmup
        sched.step()
        sched2.step()
        assert sched2.get_last_lrs() == sched.get_last_lrs()


def test_state_that_is_not_json_is_refused(expect_error):
    class _Bad:
        def get_state_dict(self):
            return {"position": np.int64(3)}

    with expect_error(CheckpointError, "dataloader state at /position: a int64 has no JSON form"):
        capture_train_state(dataloader=_Bad(), rng=False)


def _cosine_state() -> dict:
    sched = CosineAnnealingScheduler(_adamw([1e-3]), T_max=4)
    sched.step()
    return json.loads(json.dumps(sched.get_state_dict()))


def test_refused_state_stops_the_restore_first(expect_error):
    """A refused state stops the restore before anything else moves."""
    # A StepScheduler given a cosine scheduler's state: the run was configured differently.
    loader = Loader()
    stepper = StepScheduler(_adamw([1e-3]), step_size=2)
    before = stepper.get_state_dict()
    stored = {"scheduler": _cosine_state(), "dataloader": {"position": 5}}
    with expect_error(CheckpointError, r"the live scheduler refused the stored scheduler state \(KeyError"):
        restore_train_state(stored, scheduler=stepper, dataloader=loader)
    assert stepper.get_state_dict() == before, "a refusing scheduler keeps its state"
    assert loader.position == 0, "the scheduler is restored first, so the loader has not moved"


def test_component_without_state_setter_is_refused(expect_error):
    """A component that cannot take a state is refused before anything is placed."""
    stored = {"scheduler": {"m_last_step": 3}, "dataloader": {"position": 5}}
    with expect_error(TypeError, r"the dataloader passed \(list_iterator\) has no set_state_dict"):
        check_train_state(stored, dataloader=iter([]))
    check_train_state(stored, scheduler=StepScheduler(_adamw([1e-3]), step_size=2), dataloader=Loader())
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
    # The whole train state passed where its RNG block belongs, and a block whose format is not a number, are refused.
    with expect_error(CheckpointError, r"not an RNG block \(found \['dataloader', 'rng'\]\)"):
        restore_rng({"dataloader": {"position": 1}, "rng": block}, world_rank=0)
    with expect_error(CheckpointError, "RNG state format 'x' is not a version number"):
        restore_rng(dict(block, format="x"), world_rank=0)
    check_train_state({"rng": block}, rng=False)  # not asked for: nothing to check


@pytest.mark.parametrize("damage", list(WRONG_LENGTH_RNG.values()), ids=list(WRONG_LENGTH_RNG))
def test_rng_state_of_the_wrong_length_changes_no_generator(damage, expect_error):
    # Valid JSON and hex of the wrong length for its generator: refused while decoding, before any generator is set.
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


@pytest.mark.parametrize("damage", list(MISREAD_RNG.values()), ids=list(MISREAD_RNG))
def test_misreadable_rng_state_is_refused(damage, expect_error):
    """RNG state a setter would misread is refused, and no generator changes."""
    ctx = ttml.autograd.AutoContext.get_instance()
    block = capture_rng(world_rank=0)
    broken = json.loads(json.dumps(block))
    damage(broken["ranks"]["0"])
    before = (ctx.get_generator_state(), random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
    with expect_error(CheckpointError, "the stored RNG state is malformed .*pass rng=False to load"):
        check_train_state({"rng": broken}, rng=True, world_rank=0)
    with expect_error(CheckpointError, "the stored RNG state is malformed"):
        restore_rng(broken, world_rank=0)
    after = (ctx.get_generator_state(), random.getstate(), np.random.get_state()[1], torch.get_rng_state())
    assert after[0] == before[0] and after[1] == before[1], "ttml and python untouched"
    assert np.array_equal(after[2], before[2]) and torch.equal(after[3], before[3]), "numpy and torch untouched"
    check_train_state({"rng": broken}, rng=False)  # not asked for: not decoded, so a load without rng goes ahead


def test_the_ttml_text_must_have_the_form_this_build_writes(monkeypatch, expect_error):
    """libstdc++ writes an mt19937 as its 624 words and its position, libc++ as the 624 words alone, and each reads
    the other's text wrongly. The decoder expects the form the running build writes, so a build takes its own
    checkpoints and refuses the other's."""
    block = capture_rng(world_rank=0)
    tokens = block["ranks"]["0"]["ttml"].split()
    this_form = len(tokens)
    assert this_form in (624, 625), f"an mt19937 text of {this_form} tokens"
    other_form = 624 if this_form == 625 else 625
    other = json.loads(json.dumps(block))
    other["ranks"]["0"]["ttml"] = " ".join(tokens[:624] + ["0"] * (other_form - 624))
    with expect_error(CheckpointError, f"is not the {this_form} tokens this build writes .*it has {other_form}"):
        check_train_state({"rng": other}, rng=True, world_rank=0)

    monkeypatch.setattr(_train_state, "_mt19937_text_tokens", lambda: other_form)  # a build with the other library
    check_train_state({"rng": other}, rng=True, world_rank=0)  # its own form decodes
    with expect_error(CheckpointError, f"is not the {other_form} tokens this build writes .*it has {this_form}"):
        check_train_state({"rng": block}, rng=True, world_rank=0)


class _TupleLoader(Loader):
    """Stores a tuple, as random.Random().getstate() returns, and wants it back as one."""

    def get_state_dict(self) -> dict:
        return {"position": self.position, "shuffle": (1, 2)}

    def set_state_dict(self, state: dict) -> None:
        if not isinstance(state["shuffle"], tuple):
            raise TypeError("state vector must be a tuple")
        super().set_state_dict({"position": state["position"]})


def test_a_tuple_comes_back_as_a_list_and_the_refusal_says_so(expect_error):
    stored = capture_train_state(dataloader=_TupleLoader(), rng=False)
    assert stored["dataloader"]["shuffle"] == [1, 2], "JSON has no tuples"
    with expect_error(CheckpointError, r"does it take its state back as JSON types \(a tuple comes back as a list\)"):
        restore_train_state(stored, dataloader=_TupleLoader())


class _IntKeyLoader(Loader):
    def get_state_dict(self) -> dict:
        return {"per_shard": {0: 7}}


def test_state_that_would_not_read_back_as_written_is_refused(expect_error):
    with expect_error(CheckpointError, "dataloader state at /per_shard: key 0 is a int"):
        capture_train_state(dataloader=_IntKeyLoader(), rng=False)
