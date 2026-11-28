# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Models and helpers shared by test_sync_gradients_placements.py and test_sync_gradients_placements_2x2.py."""

from __future__ import annotations

import itertools
import re

import numpy as np

import ttnn
import ttml
from gradient_sync_helpers import upload
from ttml.common.utils import build_causal_mask
from ttml.models.llama import Llama, LlamaConfig
from ttml.modules import AbstractModuleBase, LinearLayer, LoraConfig, LoraModel

DIM = 64  # tile-aligned per device after a 2-way split
NATIVE = ttml.autograd.PreferredPrecision.NATIVE

is_sharded_on_axis = ttml.core.distributed.is_sharded_on_axis


# The parameters of tp_block.TPBlock that are sharded on the tp axis.
TP_SHARDED_SUFFIXES = ("up/weight", "up/bias", "down/weight")


class Block(AbstractModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = LinearLayer(DIM, DIM)
        self.fc2 = LinearLayer(DIM, DIM)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class Model(AbstractModuleBase):
    """``block`` is the FSDP target where FSDP is used; ``head`` is never FSDP-managed."""

    def __init__(self) -> None:
        super().__init__()
        self.block = Block()
        self.head = LinearLayer(DIM, DIM, has_bias=False)

    def forward(self, x):
        return self.head(self.block(x))


def _build_fsdp_model(init: str) -> Model:
    if init == "eager":
        model = Model()
        ttml.fsdp.fully_shard(model.block)
        return model
    with ttml.lazy_init():
        model = Model()
    ttml.fsdp.fully_shard(model.block)
    ttml.materialize_module(model)
    return model


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _input(shape, mapper=None, *, seed: int = 0, scale: float = 0.1):
    data = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) * scale
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _backward(model, x) -> None:
    ctx = ttml.autograd.AutoContext.get_instance()
    loss = ttml.ops.unary.mean(model(x))
    loss.backward(False)
    ctx.reset_graph()


def _placements(value) -> list:
    return list(value.tensor_topology().placements())


def _shards(value) -> list:
    """Per-device arrays in mesh row-major order (every label built here lists its coordinates in that order)."""
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(value)]


def _reduce_set(value, data_axes) -> set:
    """The axes synchronize_gradients reduces a gradient along: the data axes the parameter is not sharded on."""
    return {axis for axis in data_axes if not is_sharded_on_axis(value, axis)}


def _assert_mean_over_axis(before: list, after: list, mesh_shape, axis: int, what: str) -> None:
    """``after`` holds, on every device, the mean of ``before`` over mesh axis ``axis`` (bf16 sum then exact scale)."""
    shape = tuple(mesh_shape)
    b = np.stack(before).reshape(shape + before[0].shape)
    a = np.stack(after).reshape(shape + after[0].shape)
    expected = np.broadcast_to(b.mean(axis=axis, keepdims=True), b.shape)
    np.testing.assert_allclose(
        a, expected, rtol=2e-2, atol=1e-4, err_msg=f"{what}: after vs mean of before over mesh axis {axis}"
    )


def _assert_per_axis_labels(params, n_axes: int) -> None:
    """Every parameter carries one placement per mesh axis: none of them is a collapsed 1-D label."""
    wrong = {name: _placements(t.get_value(NATIVE)) for name, t in params.items()}
    wrong = {name: p for name, p in wrong.items() if len(p) != n_axes}
    assert not wrong, f"parameters without one placement per mesh axis: {wrong}"


# --- A one-block Llama, to check which gradients ttml.sync_gradients reduces ---------------------------------------

LLAMA_SEQ_LEN = 128  # two tiles per rank under 2-way SP
LLAMA_VOCAB = 128

# Parameter name patterns, matched with re.search.
EVERY = r""
NORMS = r"(attention_norm|mlp_norm|ln_fc)/gamma$"
COLUMN_LORA_A = r"(qkv_linear|w_gate_up)/lora_A$"
ROW_LORA_B = r"(out_linear|w2)/lora_B$"
# The same norms as FSDP replicate= patterns, which match dotted names relative to the wrapped module.
FSDP_REPLICATED_NORMS = [r"_norm\.", r"ln_fc\."]

# The right reduction lands within 0.003 of the exact one (bf16 rounding); any other is at least 0.5 off.
SYNC_TOLERANCE = 0.01


def tiny_llama(tp_strategy, *, lora: bool = False, fsdp_replicate=None) -> AbstractModuleBase:
    """A one-block Llama. Pass ``fsdp_replicate`` (a list of patterns, possibly empty) to wrap it in FSDP as train.py
    does."""
    config = LlamaConfig(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        intermediate_size=128,
        vocab_size=LLAMA_VOCAB,
        max_position_embeddings=LLAMA_SEQ_LEN,
        tp_strategy=tp_strategy,
    )
    ttml.manual_seed(0)
    if fsdp_replicate is None:
        model = Llama(config)
    else:
        with ttml.lazy_init():
            model = Llama(config)
        for block in model.blocks:
            ttml.fsdp.fully_shard(block, replicate=fsdp_replicate)
        ttml.fsdp.fully_shard(model, replicate=fsdp_replicate)
        ttml.materialize_module(model)
    if not lora:
        return model
    np.random.seed(0)  # lora_A is drawn from numpy's global RNG
    model = LoraModel(
        model,
        LoraConfig(
            rank=8,
            target_modules=["qkv_linear", "out_linear", "w_gate_up", "w2"],
            trainable_modules=["_norm", "ln_fc"],
        ),
    )
    # lora_B starts at zero, which would leave lora_A without a gradient.
    for name, param in model.parameters().items():
        if name.endswith("lora_B"):
            sharding = ttml.Sharding.from_tensor(param)
            values = (np.random.default_rng(1).standard_normal(sharding.global_shape) * 0.05).astype(np.float32)
            ttml.autograd.assign_numpy(param, values, layout=ttnn.Layout.TILE, mapper=sharding.derive_mapper())
    return model


def llama_backward(model, data_axes: tuple[str, ...]) -> None:
    """One forward and backward pass, with one sequence per device along ``data_axes``, batched as in train.py."""
    mesh = ttml.mesh()
    batch = int(np.prod([mesh.axis_size(axis) for axis in data_axes]))
    if not data_axes:
        mapper = None
    elif len(data_axes) == 1:
        mapper = mesh.axis_mapper(data_axes[0], tdim=0)
    else:
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(_device(), 0, None)
    ids = np.random.default_rng(0).integers(0, LLAMA_VOCAB, (batch, 1, 1, LLAMA_SEQ_LEN)).astype(np.uint32)
    ids = ttml.autograd.Tensor.from_numpy(ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper)
    model.train()
    model(ids, build_causal_mask(LLAMA_SEQ_LEN, device=True)).backward(retain_graph=False)
    ttml.autograd.AutoContext.get_instance().reset_graph()


def reductions_by_rule(params, rules) -> dict:
    """Map each parameter with a gradient to ``{axis: "mean" | "sum"}``, from ``(axis, op, pattern)`` rules."""
    expected = {}
    for name, tensor in params.items():
        if tensor.is_grad_initialized():
            expected[name] = {axis: op for axis, op, pattern in rules if re.search(pattern, name)}
    unmatched = [rule for rule in rules if not any(re.search(rule[2], name) for name in expected)]
    assert not unmatched, f"rules that match no parameter with a gradient: {unmatched}"
    return expected


def _per_device(value, mesh_shape) -> np.ndarray:
    shards = _shards(value)
    return np.stack(shards).reshape(tuple(mesh_shape) + shards[0].shape).astype(np.float64)


def _reduce(before: np.ndarray, plan: dict) -> np.ndarray:
    """Apply ``plan`` (``{mesh axis index: "mean" | "sum"}``) to per-device values shaped ``(*mesh, *local)``."""
    after = before
    for axis, op in plan.items():
        total = after.sum(axis=axis, keepdims=True)
        after = np.broadcast_to(total / before.shape[axis] if op == "mean" else total, before.shape)
    return after


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def _closest_reduction(before: np.ndarray, after: np.ndarray, mesh) -> str:
    axes = [axis for axis, size in enumerate(mesh.shape) if size > 1]
    errors = []
    for ops in itertools.product((None, "mean", "sum"), repeat=len(axes)):
        plan = {axis: op for axis, op in zip(axes, ops) if op}
        errors.append((_relative_error(after, _reduce(before, plan)), plan))
    error, plan = min(errors, key=lambda e: e[0])
    named = {mesh.axis_names[axis]: op for axis, op in plan.items()}
    return f"{named} (relative error {error:.3g})"


def assert_sync_reduces_exactly(params, expected: dict) -> None:
    """Check that ``ttml.sync_gradients`` reduces each gradient as ``expected`` says and nothing else, whatever the
    gradients' labels: as computed, all Replicate, or all Shard."""
    mesh = ttml.mesh()
    names = list(expected)
    computed = {name: ttml.core.distributed.placements_per_axis(params[name].get_grad()) for name in names}
    rng = np.random.default_rng(0)
    for label in ("computed", "Replicate", "Shard"):
        before = {}
        for name in names:
            tensor = params[name]
            local_shape = tuple(tensor.get_grad().shape)
            assert len(local_shape) == 4 and len(mesh.shape) == 2, (name, local_shape, mesh.shape)
            rows, cols = mesh.shape
            values = rng.standard_normal(tuple(mesh.shape) + local_shape).astype(np.float32)
            # Tile the blocks along dims 0 and 1 so that device (r, c) gets block (r, c).
            tiled = values.transpose(0, 2, 1, 3, 4, 5).reshape(
                rows * local_shape[0], cols * local_shape[1], *local_shape[2:]
            )
            grad = upload(tiled, [ttnn.PlacementShard(0), ttnn.PlacementShard(1)]).get_value(NATIVE)
            if label == "computed":
                ttml.core.distributed.set_placements_per_axis(grad, computed[name])
            elif label == "Replicate":
                ttml.core.distributed.set_placements_per_axis(grad, [ttnn.PlacementReplicate()] * len(mesh.shape))
            tensor.set_grad(grad)
            before[name] = _per_device(grad, mesh.shape)

        ttml.sync_gradients(params)

        wrong = []
        for name in names:
            after = _per_device(params[name].get_grad(), mesh.shape)
            plan = {mesh.axis_index(axis): op for axis, op in expected[name].items()}
            if _relative_error(after, _reduce(before[name], plan)) > SYNC_TOLERANCE:
                wrong.append(f"{name}: expected {expected[name]}, got {_closest_reduction(before[name], after, mesh)}")
        assert not wrong, f"gradients labelled {label}:\n  " + "\n  ".join(wrong)
