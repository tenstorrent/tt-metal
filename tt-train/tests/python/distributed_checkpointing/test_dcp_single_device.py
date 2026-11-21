# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Save and load on one device with no mesh open: the paths the mesh fixtures of ``test_dcp_device.py`` do not reach.

The device is the one ``AutoContext`` opens on first use, so ``ttml.maybe_mesh()`` is None and the index records the
device's own shape. The cases: a FLOAT32 parameter (every other test is BF16), the optimizers beyond plain AdamW
(AdamW with amsgrad, SGD with momentum, Muon), optimizer state not keyed by a parameter of the model passed, the byte
budget that bounds the host buffers in flight, a missing data file and a restored tensor whose spec does not match the
live one.
"""

import errno
import os
import threading
import weakref

import numpy as np
import pytest
import ttnn
from dcp_shards import assert_same_shards, assert_some_differ, model_shards, optimizer_shards
from tp_block import make_adamw, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.distributed_checkpointing import _load, _save
from ttml.distributed_checkpointing._coordinator import Election, PlanPart, ResultPart
from ttml.modules import AbstractModuleBase, LinearLayer, Parameter

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(900)]

DIM = 64
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


@pytest.fixture(scope="module", autouse=True)
def single_device(release_metal_env):
    """No mesh: the first tensor opens AutoContext's default device; closed again for the modules after this one."""
    release_metal_env()
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.close_device()
    yield
    ctx.close_device()


class _Pair(AbstractModuleBase):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.a = LinearLayer(dim, dim, has_bias=False)
        self.b = LinearLayer(dim, dim, has_bias=False)

    def forward(self, x):
        return self.b(self.a(x))


class _F32(AbstractModuleBase):
    """A FLOAT32 parameter, which save untilizes and load tilizes in FLOAT32."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        values = np.random.default_rng(seed).standard_normal((1, 1, 48, DIM)).astype(np.float32)
        self.w = Parameter(ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.FLOAT32))


def test_float32_round_trip_records_the_device(tmp_path):
    """A float32 parameter round trips, and the index records the device."""
    assert ttml.maybe_mesh() is None
    model = _F32(seed=1)
    assert model.w.tensor.get_value(NATIVE).dtype == ttnn.DataType.FLOAT32, "precondition"
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model)
    index = dcp.read_index(path)
    assert index.entries["model/_F32/w"].dtype == "F32"
    (rank,) = index.world["ranks"]
    assert rank["shape"] == [1, 1] and rank["axis_names"] == ["_0", "_1"]

    fresh = _F32(seed=2)
    assert_some_differ(model_shards(fresh.parameters()), model_shards(model.parameters()))
    dcp.load(path, model=fresh)
    assert_same_shards(model_shards(fresh.parameters()), model_shards(model.parameters()))


def _adamw_amsgrad(params):
    cfg = ttml.optimizers.AdamWConfig.make(
        lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0, amsgrad=True
    )
    return ttml.optimizers.AdamW(params, cfg)


def _sgd_momentum(params):
    cfg = ttml.optimizers.SGDConfig.make(lr=1e-2, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False)
    return ttml.optimizers.SGD(params, cfg)


def _muon(params):
    return ttml.optimizers.MuonComposite(params, ttml.optimizers.MuonConfig.make(lr=1e-2, momentum=0.95, ns_steps=5))


@pytest.mark.parametrize("make_optimizer", [_adamw_amsgrad, _sgd_momentum, _muon], ids=["adamw_amsgrad", "sgd", "muon"])
def test_optimizer_state_round_trips_and_the_next_step_matches(tmp_path, make_optimizer):
    model = _Pair(DIM)
    opt = make_optimizer(model.parameters())
    train_step(model, opt, DIM)
    train_step(model, opt, DIM, input_scale=0.5)
    path = str(tmp_path / "step_0000002")
    dcp.save(path, model=model, optimizer=opt)
    state = optimizer_shards(opt)[0]
    assert state, "the optimizer keeps per-parameter state"

    fresh = _Pair(DIM)
    fresh_opt = make_optimizer(fresh.parameters())
    dcp.load(path, model=fresh, optimizer=fresh_opt)
    assert_same_shards(model_shards(fresh.parameters()), model_shards(model.parameters()))
    assert_same_shards(optimizer_shards(fresh_opt)[0], state)
    train_step(model, opt, DIM, input_scale=0.25)
    train_step(fresh, fresh_opt, DIM, input_scale=0.25)
    assert_same_shards(
        model_shards(fresh.parameters()), model_shards(model.parameters()), "the restored state steps alike"
    )


def test_state_of_other_parameters_is_saved_as_is(tmp_path):
    """Optimizer state not keyed by a parameter of the model passed is saved as it is."""
    # The optimizer covers the whole pair, the model passed is one half: none of the state's names is one of its
    # parameters, so each state tensor is saved under its own key with an identity layout, and loads back the same way.
    pair = _Pair(DIM)
    opt = make_adamw(pair.parameters())
    train_step(pair, opt, DIM)
    path = str(tmp_path / "step_0000001")
    with pytest.warns(UserWarning, match="is not keyed by a parameter of the model; saved with an identity layout"):
        dcp.save(path, model=pair.a, optimizer=opt)
    fresh = _Pair(DIM)
    fresh_opt = make_adamw(fresh.parameters())
    train_step(fresh, fresh_opt, DIM, input_scale=0.5)
    with pytest.warns(UserWarning, match="is not keyed by a parameter of the model"):
        dcp.load(path, model=fresh.a, optimizer=fresh_opt)
    assert_same_shards(optimizer_shards(fresh_opt)[0], optimizer_shards(opt)[0])


def test_one_byte_budget(tmp_path, monkeypatch):
    """A one-byte in-flight budget writes one key at a time."""
    model = _Pair(DIM)
    opt = make_adamw(model.parameters())
    train_step(model, opt, DIM)
    in_flight = []

    class _Recording(_save._ByteBudget):
        def acquire(self, nbytes: int) -> None:
            super().acquire(nbytes)
            in_flight.append((self.used, nbytes))

    monkeypatch.setattr(_save, "_ByteBudget", _Recording)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model, optimizer=opt, max_inflight_bytes=1)
    assert len(in_flight) > 1 and all(used == n for used, n in in_flight), "each key waited for the one before"

    fresh = _Pair(DIM)
    fresh_opt = make_adamw(fresh.parameters())
    dcp.load(path, model=fresh, optimizer=fresh_opt)
    assert_same_shards(model_shards(fresh.parameters()), model_shards(model.parameters()))
    assert_same_shards(optimizer_shards(fresh_opt)[0], optimizer_shards(opt)[0])


class _Interrupt(BaseException):
    """Stands in for a KeyboardInterrupt, which is a BaseException and not an Exception, without being one pytest
    would act on."""


def _trained_pair():
    model = _Pair(DIM)
    opt = make_adamw(model.parameters())
    train_step(model, opt, DIM)
    return model, opt


def _join_writers() -> None:
    for thread in threading.enumerate():
        if thread.name.startswith("dcp-writer-"):
            thread.join()


def test_an_interrupted_save_writes_only_into_its_own_files(tmp_path, monkeypatch, expect_error):
    """An interrupt while save waits for its writers (a ^C in a join) leaves no writer writing through a descriptor
    that was closed under it, and so reused by the next file opened: each writer closes its own files after its last
    write."""
    model, opt = _trained_pair()
    gate = threading.Event()
    real_pwrite_all, real_join = _save.pwrite_all, threading.Thread.join
    interrupted, lost = [], []

    def held_pwrite_all(fd, view, offset):  # each writer's first write is still going when the interrupt lands
        inode = os.fstat(fd).st_ino
        gate.wait()
        try:
            if os.fstat(fd).st_ino != inode:
                lost.append(fd)  # closed, and reused by another file
        except OSError:
            lost.append(fd)  # closed
        real_pwrite_all(fd, view, offset)

    def interrupted_join(thread, timeout=None):
        if thread.name.startswith("dcp-writer-") and not interrupted:
            interrupted.append(thread.name)
            threading.Timer(0.5, gate.set).start()  # the writes that were going on go on after the interrupt
            raise _Interrupt()
        return real_join(thread, timeout)

    monkeypatch.setattr(_save, "pwrite_all", held_pwrite_all)
    monkeypatch.setattr(threading.Thread, "join", interrupted_join)
    open_fds = len(os.listdir("/proc/self/fd"))
    with expect_error(_Interrupt, ""):
        dcp.save(str(tmp_path / "step_0000001"), model=model, optimizer=opt)
    monkeypatch.undo()
    # The next file opened takes the lowest free descriptor: the save's own, had it been closed under its writer.
    bystander = tmp_path / "bystander"
    with open(bystander, "wb"):
        gate.wait()
        _join_writers()
    assert interrupted, "the save never waited for a writer"
    assert not lost, f"descriptors {lost} were closed while their writer was still writing through them"
    assert os.path.getsize(bystander) == 0, "an interrupted save's writer wrote into a file it did not open"
    assert len(os.listdir("/proc/self/fd")) == open_fds, "every descriptor the save opened is closed"
    assert os.listdir(tmp_path) == ["bystander"], "no checkpoint and no staging directory left"


def test_a_shard_is_freed_before_its_budget_is_released(tmp_path, monkeypatch):
    """The writer lets go of a shard's host buffers before it releases the shard's budget: otherwise the producer is
    told the bytes are free while they are still held, and the host buffers in flight outgrow max_inflight_bytes."""
    model, opt = _trained_pair()
    views, alive_at_release = [], []
    real_logical_bytes = _save._logical_bytes

    def recorded_logical_bytes(view, sizes, itemsize):
        out = real_logical_bytes(view, sizes, itemsize)
        views.append(weakref.ref(out))
        return out

    class _Checked(_save._ByteBudget):
        def release(self, nbytes: int) -> None:
            alive_at_release.append(sum(ref() is not None for ref in views))
            super().release(nbytes)

    monkeypatch.setattr(_save, "_logical_bytes", recorded_logical_bytes)
    monkeypatch.setattr(_save, "_ByteBudget", _Checked)
    dcp.save(str(tmp_path / "step_0000001"), model=model, optimizer=opt, threads=1)  # one writer, one shard at a time
    assert views and len(alive_at_release) == len(views), "every shard written released its budget"
    assert not any(alive_at_release), f"shards still held when their budget was released: {alive_at_release}"


@pytest.mark.timeout(120)  # a writer that kept the failed write's budget would leave save waiting for it forever
def test_a_failed_write_stops_the_save(tmp_path, monkeypatch, expect_error):
    """The second write fails. Save raises that error, not a later one; writes nothing after it; does not wait for the
    budget the failed write held; and leaves no descriptor open and no directory behind. With a one-byte budget one
    key is in flight at a time, so the producer is waiting for the failed write's budget when it fails."""
    model, opt = _trained_pair()
    calls = []
    real_pwrite_all = _save.pwrite_all

    def failing_second_write(fd, view, offset):
        calls.append(offset)
        if len(calls) == 2:
            raise OSError(errno.ENOSPC, "injected: no space left on device")
        real_pwrite_all(fd, view, offset)

    monkeypatch.setattr(_save, "pwrite_all", failing_second_write)
    open_fds = len(os.listdir("/proc/self/fd"))
    with expect_error(dcp.CheckpointError, r"writing the shards failed: OSError: .*injected: no space left") as raised:
        dcp.save(str(tmp_path / "step_0000001"), model=model, optimizer=opt, max_inflight_bytes=1)
    assert raised.value.__cause__.errno == errno.ENOSPC
    assert len(calls) == 2, f"{len(calls)} writes: nothing may be written after the failed one"
    assert len(os.listdir("/proc/self/fd")) == open_fds, "every descriptor the save opened is closed"
    assert os.listdir(tmp_path) == [], "no checkpoint and no staging directory left"


def test_strict_refuses_a_momentum_free_optimizer_lacking_saved_params(tmp_path, expect_error):
    """SGD without momentum keeps no per-parameter state, so its member lists are the only record of what it trains.
    A strict load into one whose group lacks a param the saved group held is refused before anything is placed (that
    param would stop being trained without a word); a non-strict load is allowed."""
    cfg = ttml.optimizers.SGDConfig.make(lr=1e-2, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=False)
    model = _Pair(DIM)
    opt = ttml.optimizers.SGD(model.parameters(), cfg)
    train_step(model, opt, DIM)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model, optimizer=opt)
    fresh = _Pair(DIM)
    params = fresh.parameters()
    without_b = {name: p for name, p in params.items() if "/b/" not in name}
    assert len(without_b) == 1
    before = model_shards(params)
    with expect_error(dcp.CheckpointError, r"optimizer param group 0: the checkpoint's group holds params no rank's"):
        dcp.load(path, model=fresh, optimizer=ttml.optimizers.SGD(without_b, cfg))
    assert_same_shards(model_shards(params), before, "nothing was placed")
    dcp.load(path, model=fresh, optimizer=ttml.optimizers.SGD(without_b, cfg), strict=False)
    assert_same_shards(model_shards(params), model_shards(model.parameters()))


def test_one_process_keeps_its_parts_as_objects(tmp_path, monkeypatch):
    """The group of one has no peer to send its plan part, election or result part to, so it keeps them as objects:
    encoding and decoding them would only cost time and a copy of every tensor's description."""
    model, opt = _trained_pair()
    for cls in (PlanPart, Election, ResultPart):
        monkeypatch.setattr(
            cls, "to_bytes", lambda self: pytest.fail(f"the group of one encoded a {type(self).__name__}")
        )
    path = str(tmp_path / "step_0000001")
    stats = dcp.save(path, model=model, optimizer=opt).stats
    monkeypatch.undo()
    assert dcp.load(path, model=model, optimizer=opt).missing == []
    # Each step records its own time, the planning and the merge included.
    steps = ("collect_s", "election_s", "plan_s", "readback_s", "write_s", "finalize_s")
    assert all(0 <= stats[s] <= stats["total_s"] for s in steps), stats
    assert sum(stats[s] for s in steps if s != "readback_s") <= stats["total_s"], stats


def test_missing_file_or_bad_spec_is_refused_first(tmp_path, monkeypatch, expect_error):
    """A missing data file and a mismatched spec are refused before anything is placed."""
    model = _Pair(DIM)
    path = str(tmp_path / "step_0000001")
    dcp.save(path, model=model)
    fresh = _Pair(DIM)
    before = model_shards(fresh.parameters())

    monkeypatch.setattr(_load, "_same_spec", lambda restored, live: False)
    with expect_error(dcp.CheckpointError, r"model/_Pair/\w/weight: the restored tensor .* does not match the live"):
        dcp.load(path, model=fresh)
    monkeypatch.undo()

    (data,) = os.listdir(os.path.join(path, "data"))
    os.remove(os.path.join(path, "data", data))
    with expect_error(dcp.CheckpointError, f"data file .*{data} listed in the index is missing"):
        dcp.load(path, model=fresh)
    assert_same_shards(model_shards(fresh.parameters()), before, "nothing was placed")
