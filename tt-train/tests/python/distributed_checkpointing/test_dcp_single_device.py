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

import os

import numpy as np
import pytest
import ttnn
from dcp_shards import assert_same_shards, assert_some_differ, model_shards, optimizer_shards
from tp_block import make_adamw, train_step

import ttml
import ttml.distributed_checkpointing as dcp
from ttml.distributed_checkpointing import _load, _save
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
