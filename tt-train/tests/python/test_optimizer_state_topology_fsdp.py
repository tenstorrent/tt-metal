# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""FSDP counterpart of ``test_optimizer_state_topology``: moments must carry their parameter's FSDP placements.

``ttml.fsdp.fully_shard`` installs ``Shard(dim)`` on the ``"fsdp"`` axis of every managed parameter -- by a host
roundtrip in the eager path, by rewriting the lazy mapper before ``materialize_module`` allocates in the lazy path.
The optimizer is created afterwards, so each moment is ``zeros_like(shard)`` and must report the shard's topology,
otherwise ``ttml.checkpointing`` saves one device's slice of it (see the TP module for the underlying bug).

The FSDP hooks swap the gathered weight into the parameter for forward/backward and the cached shard back out
afterwards, so after an optimizer step the parameter is the shard again; a manual ``unshard()`` / ``reshard()``
cycle must likewise leave moments and parameters in agreement. FSDP+TP needs a distinct sharded axis for each, i.e.
at least a ``[2, 2]`` mesh, and is not covered here.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import ttml
import ttnn
from ttml import checkpointing
from ttml.modules import AbstractModuleBase, LinearLayer

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

DIM = 64  # tile-aligned per device after the FSDP=2 split (32 rows / cols per shard)
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


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


def _build_model(init: str) -> Model:
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


def _make_optimizer(name: str, params: "ttml.NamedParameters") -> "ttml.optimizers.OptimizerBase":
    """Python-bound optimizers with per-parameter state that accept FSDP-managed params (Muon rejects them)."""
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


def _train_step(model: Model, opt) -> None:
    """One forward/backward/step through the FSDP hooks (gather -> compute -> reduce-scatter -> reshard)."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x = ttml.autograd.Tensor.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, DIM, DIM), dtype=np.float32) * 0.1,
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


def _global_shape(tensor: ttml.autograd.Tensor) -> tuple:
    """Full (gathered) shape implied by a tensor's per-device shape and its live placements."""
    shape = list(tensor.get_value(NATIVE).shape)
    placements, dist_shape = _layout(tensor)
    for axis, placement in enumerate(placements):
        if placement[0] == "shard":
            shape[placement[1]] *= dist_shape[axis]
    return tuple(shape)


def _state_tensors(state, prefix: tuple = ()):
    """(path, param_name, tensor) for every NamedParameters entry of an optimizer state dict, DFS in dict order."""
    if isinstance(state, ttml.NamedParameters):
        for name, tensor in state.items():
            yield prefix, name, tensor
    elif isinstance(state, dict):
        for key, sub in state.items():
            yield from _state_tensors(sub, prefix + (key,))


def _manifest_tensor_paths(node, prefix: tuple = ()):
    """Record order of the tensors a checkpoint manifest describes (mirrors ``checkpointing._walk``)."""
    if not isinstance(node, dict):
        return
    if "named_parameters" in node:
        for name in node["named_parameters"]:
            yield prefix + (name,)
        return
    for key, sub in node.items():
        yield from _manifest_tensor_paths(sub, prefix + (key,))


def _read_records(path: str) -> list:
    """Every pickle record in a checkpoint: the manifest dict, then one numpy array per tensor."""
    records = []
    with open(path, "rb") as f:
        while True:
            try:
                records.append(pickle.load(f))
            except EOFError:
                return records


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


def _moment_mismatches(opt, expected: dict) -> list:
    mismatches = []
    for path, name, moment in _state_tensors(opt.get_state_dict()):
        got = _layout(moment)
        if got != expected[name]:
            mismatches.append(f"{'/'.join(path)}[{name}]: moment {got} != param {expected[name]}")
    return mismatches


@pytest.mark.parametrize("init", ["eager", "lazy"])
@pytest.mark.parametrize("opt_name", ["AdamW", "AdamWFullPrecision", "SGD"])
def test_moments_carry_fsdp_parameter_topology(fsdp_mesh, init, opt_name):
    model = _build_model(init)
    params = model.parameters()
    opt = _make_optimizer(opt_name, params)
    _train_step(model, opt)

    expected = _assert_fsdp_layout(model, fsdp_mesh)
    mismatches = _moment_mismatches(opt, expected)
    assert not mismatches, "optimizer state does not carry its parameter's FSDP topology:\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("init", ["eager", "lazy"])
def test_checkpoint_saves_fsdp_moments_at_full_shape(fsdp_mesh, tmp_path, init):
    model = _build_model(init)
    params = model.parameters()
    opt = _make_optimizer("AdamW", params)
    _train_step(model, opt)
    _assert_fsdp_layout(model, fsdp_mesh)

    path = str(tmp_path / f"fsdp_adamw_{init}.ckpt")
    checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)

    records = _read_records(path)
    paths = list(_manifest_tensor_paths(records[0]["manifest"]))
    arrays = records[1:]
    assert len(paths) == len(arrays), f"manifest lists {len(paths)} tensors but {len(arrays)} records follow"
    saved = {p: tuple(a.shape) for p, a in zip(paths, arrays)}

    # Params gather to the full weight (the FSDP shard is not saved as a per-device slice); this is the reference
    # the moment records are held to.
    for name, tensor in params.items():
        assert saved[("model", name)] == _global_shape(tensor), f"param {name} saved at the wrong shape"
    assert saved[("model", "Model/block/fc1/weight")] == (1, 1, DIM, DIM)

    truncated = [
        f"{'/'.join(p)}: saved {shape} != param {saved[('model', p[-1])]}"
        for p, shape in saved.items()
        if p[0] == "optimizer" and shape != saved[("model", p[-1])]
    ]
    assert len([p for p in saved if p[0] == "optimizer"]) == 2 * len(params)  # exp_avg + exp_avg_sq per param
    assert not truncated, "checkpoint saved FSDP optimizer moments as a single shard:\n  " + "\n  ".join(truncated)


def test_unshard_reshard_cycle_keeps_moments_aligned(fsdp_mesh):
    """``unshard()`` swaps the gathered weight into the parameter; ``reshard()`` swaps the cached shard back. The
    moments are untouched by either, so once resharded they must agree with the parameters again."""
    model = _build_model("eager")
    params = model.parameters()
    opt = _make_optimizer("AdamW", params)
    _train_step(model, opt)
    expected = _assert_fsdp_layout(model, fsdp_mesh)

    model.block.unshard()
    gathered = params["Model/block/fc1/weight"]
    assert tuple(gathered.get_value(NATIVE).shape) == (1, 1, DIM, DIM), "unshard() did not gather the weight"
    assert not _moment_mismatches(opt, expected), "unshard() must not touch optimizer state"

    model.block.reshard()
    assert {name: _layout(t) for name, t in params.items()} == expected, "reshard() did not restore the shards"
    assert not _moment_mismatches(opt, expected)
