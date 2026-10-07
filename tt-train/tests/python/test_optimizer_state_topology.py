# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizer state must carry its parameter's mesh topology, or TP / FSDP checkpoints truncate it.

AdamW / SGD moments are allocated with ``core::zeros_like(param)``; ``ttml.checkpointing`` gathers every tensor by
its own topology label, so a moment that reports ``Replicate`` where its parameter is sharded is saved as one
device's shard (386 of 518 AdamW moments in a Llama-8B TP=8 run). Both groups below pin the fix on a ``[1, 2]``
mesh: every moment reports its parameter's placements and distribution shape, a checkpoint restores every tensor
device-for-device into a fresh model, and (TP) an optimizer step cannot relabel a parameter through a mislabelled
gradient.

TP: ``TPBlock`` mixes replicated params with weights sharded on dims 2 and 3 and a sharded bias.
FSDP: ``ttml.fsdp.fully_shard`` installs ``Shard(dim)`` on the ``"fsdp"`` axis of every managed parameter -- by a
host roundtrip in the eager path, by rewriting the lazy mapper before ``materialize_module`` allocates in the lazy
path -- and the optimizer is created afterwards. The FSDP hooks swap the gathered weight in for forward/backward and
the cached shard back out afterwards, so after a step the parameter is the shard again; a manual ``unshard()`` /
``reshard()`` cycle must likewise leave moments and parameters in agreement. FSDP+TP needs a distinct sharded axis
for each, i.e. at least a ``[2, 2]`` mesh, and is not covered here.

``tp_mesh`` and ``fsdp_mesh`` (conftest) are both module-scoped and each reopens the device with its own axis names,
so the two groups must not interleave: every TP test lives in ``TestTP``, every FSDP test in ``TestFSDP``, in that
order.
"""

from __future__ import annotations

import numpy as np
import pytest

import ttml
import ttnn
from ttml import checkpointing
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, RowParallelLinear

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

DIM = 64  # tile-aligned per device after a 2-way split (32 rows / cols per shard)
NATIVE = ttml.autograd.PreferredPrecision.NATIVE

# Every Python-bound optimizer with per-parameter state that accepts sharded params (Muon rejects them).
OPTIMIZERS = ["AdamW", "AdamWFullPrecision", "SGD"]


# --- shared helpers -------------------------------------------------------------------------------------------


def _make_optimizer(name: str, params: "ttml.NamedParameters") -> "ttml.optimizers.OptimizerBase":
    if name == "AdamW":
        cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
        return ttml.optimizers.AdamW(params, cfg)
    if name == "AdamWFullPrecision":
        cfg = ttml.optimizers.AdamWFullPrecisionConfig.make(
            lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0
        )
        return ttml.optimizers.AdamWFullPrecision(params, cfg)
    if name == "SGD":
        cfg = ttml.optimizers.SGDConfig.make(lr=1e-3, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False)
        return ttml.optimizers.SGD(params, cfg)
    raise ValueError(f"Unknown optimizer name: {name!r}")


def _train_step(model, opt, input_scale: float = 1.0) -> None:
    """One forward/backward/optimizer step on a ``(1, 1, DIM, DIM)`` input, so every moment exists and has been
    touched by the update kernel (and, for FSDP, so the gather -> compute -> reduce-scatter -> reshard hooks ran)."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x = ttml.autograd.Tensor.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, DIM, DIM), dtype=np.float32) * input_scale,
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )
    opt.zero_grad()
    loss = ttml.ops.unary.mean(model(x))
    loss.backward(False)
    opt.step()
    ctx.reset_graph()


def _placement_key(p) -> tuple:
    return ("shard", p.dim) if isinstance(p, ttnn.PlacementShard) else ("replicate",)


def _layout(tensor: ttml.autograd.Tensor) -> tuple:
    """(placements, dist_shape) of a tensor's live topology, in a form that compares with ``==``."""
    sharding = ttml.Sharding.from_tensor(tensor)
    assert sharding.placements is not None, "a tensor on a [1, 2] mesh must report a topology"
    return tuple(_placement_key(p) for p in sharding.placements), tuple(sharding.dist_shape)


def _expected_full_shape(name: str) -> tuple:
    """Gathered shape of a parameter of the ``DIM x DIM`` linear stacks these tests build: ``(1, 1, DIM, DIM)`` for a
    weight, ``(1, 1, 1, DIM)`` for a bias. Stated absolutely, so a parameter whose own topology label is wrong cannot
    vouch for the shape it was saved at."""
    leaf = name.rsplit("/", 1)[-1]
    if leaf == "weight":
        return (1, 1, DIM, DIM)
    if leaf == "bias":
        return (1, 1, 1, DIM)
    raise ValueError(f"unexpected parameter name {name!r}")


def _state_tensors(state, prefix: tuple = ()):
    """(path, param_name, tensor) for every NamedParameters entry of an optimizer state dict, DFS in dict order."""
    if isinstance(state, ttml.NamedParameters):
        for name, tensor in state.items():
            yield prefix, name, tensor
    elif isinstance(state, dict):
        for key, sub in state.items():
            yield from _state_tensors(sub, prefix + (key,))


def _moment_mismatches(opt, expected: dict) -> list:
    """Optimizer state tensors whose layout differs from ``expected[param_name]``, as readable strings."""
    mismatches = []
    for path, name, moment in _state_tensors(opt.get_state_dict()):
        got = _layout(moment)
        if got != expected[name]:
            mismatches.append(f"{'/'.join(path)}[{name}]: moment {got} != param {expected[name]}")
    return mismatches


def _device_slices(tensor: ttml.autograd.Tensor) -> list:
    """Each device's slice of ``tensor`` as a numpy array, in mesh order. No composer is involved, so nothing here
    trusts the tensor's topology label."""
    value = tensor.get_value(NATIVE)
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(value)]


def _checkpointed_tensors(params, opt):
    yield from ((("model", name), t) for name, t in params.items())
    yield from ((("optimizer", *path, name), m) for path, name, m in _state_tensors(opt.get_state_dict()))


def _snapshot(params: "ttml.NamedParameters", opt) -> dict:
    """{("model", name) | ("optimizer", *path, name): per-device slices} for everything a checkpoint of them holds."""
    return {key: _device_slices(t) for key, t in _checkpointed_tensors(params, opt)}


def _assert_checkpoint_round_trips(path: str, params: "ttml.NamedParameters", opt, fresh) -> None:
    """The checkpoint at ``path`` (saved from ``params`` / ``opt``) restores every parameter and optimizer tensor
    device-for-device into a fresh model + optimizer of the same layout (``fresh()`` -> ``(params, opt)``).

    Public API only. ``load_checkpoint`` redistributes each record by the live tensor's topology, so a moment that
    was saved as a single device's shard (the bug) comes back re-sharded or broadcast and the other devices' slices
    no longer match the original -- provided the shards differ, which is asserted first. The gathered shapes are
    also held to ``_expected_full_shape``, so a mislabelled tensor cannot vouch for itself."""
    before = _snapshot(params, opt)
    distinct = [k for k, s in before.items() if len(s) > 1 and not all(np.array_equal(s[0], x) for x in s[1:])]
    assert distinct, "precondition: no tensor differs across devices, so a truncated record could not be detected"

    wrong = [
        f"{'/'.join(key)}: gathers to {ttml.Sharding.from_tensor(t).gather(t).shape}, expected {_expected_full_shape(key[-1])}"
        for key, t in _checkpointed_tensors(params, opt)
        if ttml.Sharding.from_tensor(t).gather(t).shape != _expected_full_shape(key[-1])
    ]
    assert not wrong, "tensors do not gather to their full shape:\n  " + "\n  ".join(wrong)

    fresh_params, fresh_opt = fresh()
    checkpointing.load_checkpoint(path, model_params=fresh_params, optimizer=fresh_opt)
    after = _snapshot(fresh_params, fresh_opt)
    assert (
        after.keys() == before.keys()
    ), f"restored state holds different tensors: {sorted(after.keys() ^ before.keys())}"
    mismatches = []
    for key, slices in before.items():
        for dev, (a, b) in enumerate(zip(slices, after[key])):
            if a.shape != b.shape:
                mismatches.append(f"{'/'.join(key)} device {dev}: saved back as {b.shape}, was {a.shape}")
            elif not np.array_equal(a, b):
                mismatches.append(f"{'/'.join(key)} device {dev}: values differ")
    assert not mismatches, (
        "checkpoint did not round-trip device-for-device (a record saved as one shard comes back broadcast):\n  "
        + "\n  ".join(mismatches)
    )


# --- TP -------------------------------------------------------------------------------------------------------


class TPBlock(AbstractModuleBase):
    """Replicated linear -> column-parallel -> row-parallel: sharded weights on dims 2 and 3, a sharded bias
    (column-parallel, dim 3) and replicated params (the first linear, the row-parallel bias)."""

    def __init__(self) -> None:
        super().__init__()
        self.inp = LinearLayer(DIM, DIM)
        self.up = ColumnParallelLinear(DIM, DIM, has_bias=True)
        self.down = RowParallelLinear(DIM, DIM, has_bias=True, input_is_parallel=True)

    def forward(self, x):
        return self.down(self.up(self.inp(x)))


def _fresh_tp(name: str):
    """An untrained TPBlock and its optimizer: (model, params, opt)."""
    model = TPBlock()
    params = model.parameters()
    return model, params, _make_optimizer(name, params)


def _trained_tp(name: str):
    model, params, opt = _fresh_tp(name)
    _train_step(model, opt)
    return params, opt


class TestTP:
    @pytest.mark.parametrize("opt_name", OPTIMIZERS)
    def test_moments_carry_parameter_topology(self, tp_mesh, opt_name):
        params, opt = _trained_tp(opt_name)
        expected = {name: _layout(t) for name, t in params.items()}

        # The module must really mix sharded and replicated params, or the check below proves nothing.
        sharded = {name for name, (placements, _) in expected.items() if any(p[0] == "shard" for p in placements)}
        assert sharded and sharded != set(expected), f"expected a TP mix of sharded/replicated params, got {expected}"

        checked = sum(1 for _ in _state_tensors(opt.get_state_dict()))
        assert checked >= len(sharded), "optimizer state dict holds no per-parameter tensors"
        mismatches = _moment_mismatches(opt, expected)
        assert not mismatches, "optimizer state does not carry its parameter's mesh topology:\n  " + "\n  ".join(
            mismatches
        )

    def test_checkpoint_round_trips_at_full_shape(self, tp_mesh, tmp_path):
        params, opt = _trained_tp("AdamW")
        path = str(tmp_path / "tp_adamw.ckpt")
        checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)
        _assert_checkpoint_round_trips(path, params, opt, fresh=lambda: _fresh_tp("AdamW")[1:])

    @pytest.mark.parametrize("opt_name", OPTIMIZERS)
    def test_step_keeps_parameter_topology_despite_mislabelled_grad(self, tp_mesh, opt_name):
        """The AdamW/SGD update kernels write the parameter in place and hand it back as the op output, and ttnn
        derives an output's topology from the union of the op's inputs. A gradient carrying a wrong label -- a CCL
        output that kept a stale ``Shard`` on the axis it reduced, say -- must not be able to relabel the parameter
        or its moments, because the checkpointer gathers by that label. Every parameter here is replicated; every
        gradient has the parameter's per-device shape but is labelled ``Shard(3)`` on the tp axis."""
        model = LinearLayer(DIM, DIM)
        params = model.parameters()
        opt = _make_optimizer(opt_name, params)
        before = {name: _layout(t) for name, t in params.items()}
        assert all(all(p == ("replicate",) for p in placements) for placements, _ in before.values()), before
        moments_before = {(path, name): _layout(m) for path, name, m in _state_tensors(opt.get_state_dict())}

        tp_axis = tp_mesh.axis_index("tp")
        device = ttml.autograd.AutoContext.get_instance().get_device()
        mislabel = ttml.core.distributed.shard_tensor_to_mesh_mapper(device, 3, tp_axis)
        for name, t in params.items():
            local_shape = list(t.get_value(NATIVE).shape)
            wide = local_shape[:-1] + [local_shape[-1] * tp_mesh.axis_size("tp")]  # sharded on dim 3 -> local_shape
            grad = ttml.autograd.Tensor.from_numpy(
                np.full(wide, 0.01, dtype=np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mislabel
            )
            assert list(grad.get_value(NATIVE).shape) == local_shape
            assert _layout(grad)[0][tp_axis] == (
                "shard",
                3,
            ), f"precondition: gradient labelled Shard(3), got {_layout(grad)}"
            t.set_grad(grad.get_value(NATIVE))

        opt.step()

        after = {name: _layout(t) for name, t in params.items()}
        assert after == before, f"an optimizer step relabelled parameters:\n  before {before}\n  after  {after}"
        moments_after = {(path, name): _layout(m) for path, name, m in _state_tensors(opt.get_state_dict())}
        assert moments_after == moments_before, "an optimizer step relabelled optimizer state"


# --- FSDP -----------------------------------------------------------------------------------------------------

FSDP_INPUT_SCALE = 0.1


class Block(AbstractModuleBase):
    """The ``fully_shard`` target: two biased linears -> weights sharded on dim 2, biases on dim 3 (auto)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = LinearLayer(DIM, DIM)
        self.fc2 = LinearLayer(DIM, DIM)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class Model(AbstractModuleBase):
    """``block`` is FSDP-wrapped; ``head`` is left alone so the module mixes sharded and replicated params."""

    def __init__(self) -> None:
        super().__init__()
        self.block = Block()
        self.head = LinearLayer(DIM, DIM, has_bias=False)

    def forward(self, x):
        return self.head(self.block(x))


def _build_fsdp_model(init: str) -> Model:
    """``"eager"``: shard materialized params (host roundtrip). ``"lazy"``: rewrite the mappers of not-yet-allocated
    params so ``materialize_module`` allocates them already sharded."""
    if init == "eager":
        model = Model()
        ttml.fsdp.fully_shard(model.block)
        return model
    with ttml.lazy_init():
        model = Model()
    ttml.fsdp.fully_shard(model.block)
    ttml.materialize_module(model)
    return model


def _assert_fsdp_layout(model: Model, mesh) -> dict:
    """Check the parameters themselves are laid out as FSDP intends, and return {name: layout}.

    Managed params must report ``Shard(_fsdp_shard_dim)`` on the ``"fsdp"`` axis; the unwrapped head must not be
    sharded anywhere. Without this the moment comparison could pass vacuously with both sides replicated."""
    fsdp_axis = mesh.axis_index("fsdp")
    layouts = {}
    for name, t in model.parameters().items():
        placements, _ = layouts.setdefault(name, _layout(t))
        if ttml.fsdp.is_fsdp_managed(t):
            assert placements[fsdp_axis] == ("shard", int(t._fsdp_shard_dim)), f"{name}: {placements}"
        else:
            assert all(p == ("replicate",) for p in placements), f"{name}: {placements}"
    assert any(ttml.fsdp.is_fsdp_managed(t) for t in model.parameters().values())
    assert not all(ttml.fsdp.is_fsdp_managed(t) for t in model.parameters().values())
    return layouts


class TestFSDP:
    @pytest.mark.parametrize("init", ["eager", "lazy"])
    @pytest.mark.parametrize("opt_name", OPTIMIZERS)
    def test_moments_carry_parameter_topology(self, fsdp_mesh, init, opt_name):
        model = _build_fsdp_model(init)
        params = model.parameters()
        opt = _make_optimizer(opt_name, params)
        _train_step(model, opt, input_scale=FSDP_INPUT_SCALE)

        expected = _assert_fsdp_layout(model, fsdp_mesh)
        mismatches = _moment_mismatches(opt, expected)
        assert not mismatches, "optimizer state does not carry its parameter's FSDP topology:\n  " + "\n  ".join(
            mismatches
        )

    @pytest.mark.parametrize("init", ["eager", "lazy"])
    def test_checkpoint_round_trips_at_full_shape(self, fsdp_mesh, tmp_path, init):
        model = _build_fsdp_model(init)
        params = model.parameters()
        opt = _make_optimizer("AdamW", params)
        _train_step(model, opt, input_scale=FSDP_INPUT_SCALE)
        _assert_fsdp_layout(model, fsdp_mesh)  # steady state: the parameters are the shards again

        path = str(tmp_path / f"fsdp_adamw_{init}.ckpt")
        checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)

        def fresh():
            fresh_params = _build_fsdp_model(init).parameters()
            return fresh_params, _make_optimizer("AdamW", fresh_params)

        _assert_checkpoint_round_trips(path, params, opt, fresh)

    def test_unshard_reshard_cycle_keeps_moments_aligned(self, fsdp_mesh):
        """``unshard()`` swaps the gathered weight into the parameter; ``reshard()`` swaps the cached shard back.
        The moments are untouched by either, so once resharded they must agree with the parameters again."""
        model = _build_fsdp_model("eager")
        params = model.parameters()
        opt = _make_optimizer("AdamW", params)
        _train_step(model, opt, input_scale=FSDP_INPUT_SCALE)
        expected = _assert_fsdp_layout(model, fsdp_mesh)

        model.block.unshard()
        gathered = params["Model/block/fc1/weight"]
        assert tuple(gathered.get_value(NATIVE).shape) == (1, 1, DIM, DIM), "unshard() did not gather the weight"
        assert not _moment_mismatches(opt, expected), "unshard() must not touch optimizer state"

        model.block.reshard()
        assert {name: _layout(t) for name, t in params.items()} == expected, "reshard() did not restore the shards"
        assert not _moment_mismatches(opt, expected)
