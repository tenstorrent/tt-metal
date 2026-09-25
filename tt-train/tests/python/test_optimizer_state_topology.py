# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Optimizer state must carry its parameter's mesh topology, or TP checkpoints truncate it.

ttnn allocates every op output fully replicated across the mesh, and ``core::zeros_like`` -- which
allocates the AdamW/SGD moments -- is a creation op that never sees the parameter's placements. A moment
of a TP-sharded parameter therefore reported ``[Replicate, Replicate]``; ``ttml.checkpointing`` gathers
by the live topology, so it kept one device's shard and dropped the rest (386 of 518 AdamW moments in a
Llama-8B TP=8 run), and on resume would have broadcast that shard to every TP rank.

These tests pin the fix on a ``[1, 2]`` TP mesh: every moment reports its parameter's placements and
distribution shape, and every moment record in a saved checkpoint has the parameter's full shape.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import ttml
import ttnn
from ttml import checkpointing
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, RowParallelLinear

pytestmark = [pytest.mark.requires_device, pytest.mark.timeout(1800)]

DIM = 64  # tile-aligned per device after the TP=2 split (32 rows / cols per shard)
NATIVE = ttml.autograd.PreferredPrecision.NATIVE


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


def _make_optimizer(name: str, params: "ttml.NamedParameters") -> "ttml.optimizers.OptimizerBase":
    """Every Python-bound optimizer with per-parameter state that accepts sharded params (Muon rejects them)."""
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


def _train_step(model: TPBlock, opt) -> None:
    """One backward + optimizer step so every moment exists and has been touched by the update kernel."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x = ttml.autograd.Tensor.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, DIM, DIM), dtype=np.float32),
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
    """Record order of the tensors a checkpoint manifest describes (mirrors ``checkpointing._walk``): a
    ``{"named_parameters": {name: meta}}`` leaf streams one record per name; dicts recurse in insertion
    order; scalars stream nothing."""
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


def _build(name: str):
    model = TPBlock()
    params = model.parameters()
    opt = _make_optimizer(name, params)
    _train_step(model, opt)
    return params, opt


@pytest.mark.parametrize("opt_name", ["AdamW", "AdamWFullPrecision", "SGD"])
def test_moments_carry_parameter_topology(tp_mesh, opt_name):
    params, opt = _build(opt_name)
    expected = {name: _layout(t) for name, t in params.items()}

    # The module must really mix sharded and replicated params, or the check below proves nothing.
    sharded = {name for name, (placements, _) in expected.items() if any(p[0] == "shard" for p in placements)}
    assert sharded and sharded != set(expected), f"expected a TP mix of sharded/replicated params, got {expected}"

    mismatches = []
    checked = 0
    for path, name, moment in _state_tensors(opt.get_state_dict()):
        checked += 1
        got = _layout(moment)
        if got != expected[name]:
            mismatches.append(f"{'/'.join(path)}[{name}]: moment {got} != param {expected[name]}")
    assert checked >= len(sharded), "optimizer state dict holds no per-parameter tensors"
    assert not mismatches, "optimizer state does not carry its parameter's mesh topology:\n  " + "\n  ".join(mismatches)


def test_checkpoint_saves_moments_at_full_shape(tp_mesh, tmp_path):
    params, opt = _build("AdamW")
    path = str(tmp_path / "tp_adamw.ckpt")
    checkpointing.save_checkpoint(path, header={}, model_params=params, optimizer=opt)

    records = _read_records(path)
    paths = list(_manifest_tensor_paths(records[0]["manifest"]))
    arrays = records[1:]
    assert len(paths) == len(arrays), f"manifest lists {len(paths)} tensors but {len(arrays)} records follow"
    saved = {p: tuple(a.shape) for p, a in zip(paths, arrays)}

    # Params gather to their full shape (the sharded weight is not saved as a per-device shard): this is the
    # reference the moment records are held to, so it must be right before the comparison means anything.
    for name, tensor in params.items():
        assert saved[("model", name)] == _global_shape(tensor), f"param {name} saved at the wrong shape"

    truncated = [
        f"{'/'.join(p)}: saved {shape} != param {saved[('model', p[-1])]}"
        for p, shape in saved.items()
        if p[0] == "optimizer" and shape != saved[("model", p[-1])]
    ]
    assert len([p for p in saved if p[0] == "optimizer"]) == 2 * len(params)  # exp_avg + exp_avg_sq per param
    assert not truncated, "checkpoint saved optimizer moments as a single shard:\n  " + "\n  ".join(truncated)
