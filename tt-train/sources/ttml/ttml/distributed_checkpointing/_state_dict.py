# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The live state: every tensor a model and optimizer hold, with its label, its layout declaration and its names.

Keys are ``model/<fqn>`` for parameters, with ``<fqn>`` the name ``module.parameters()`` surfaces (slash-joined from
the root module's name), and ``optimizer/<state path>/<fqn>`` for optimizer state tensors. Scalars in the optimizer
state (steps, learning rates, flags) are collected separately and stored in the index.

Topology labels are read from the live tensors and trusted. The one fact a label cannot carry, how a fused
projection orders its segments, comes from the declaration the fusing layer put on its ``Parameter``
(``checkpoint_layout``); the vocab padding likewise. Tied weights are detected by object identity: every Python
attribute path that reaches the same tensor is recorded as a live key, so a destination that surfaces the other name
still resolves the entry.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import ttnn

import ttml
from ttml.modules.parameter import Parameter

from ._errors import CheckpointError, FSDPStateError, LazyParameterError, TopologyError
from ._layouts import Identity, declared_transform
from ._planner import TensorInfo
from ._topology import Label, Placement, global_shape

NATIVE = ttml.autograd.PreferredPrecision.NATIVE

DTYPE_NAMES = {
    ttnn.DataType.BFLOAT16: "BF16",
    ttnn.DataType.FLOAT32: "F32",
    ttnn.DataType.UINT32: "U32",
    ttnn.DataType.INT32: "I32",
    ttnn.DataType.UINT16: "U16",
    ttnn.DataType.UINT8: "U8",
    ttnn.DataType.INT8: "I8",
}
TTNN_DTYPES = {name: dtype for dtype, name in DTYPE_NAMES.items()}
# numpy carriers of the raw bytes (bfloat16 travels as uint16; nothing is ever converted).
NUMPY_CARRIERS = {
    "BF16": np.uint16,
    "F32": np.float32,
    "U32": np.uint32,
    "I32": np.int32,
    "U16": np.uint16,
    "U8": np.uint8,
    "I8": np.int8,
}

_FUSION_HINT = re.compile(r"(qkv|kv_|_kv\b|gate_up|fused)", re.IGNORECASE)


@dataclass
class LiveTensor:
    """One live tensor: its key, the autograd tensor that holds it, and what the planner needs to know."""

    key: str
    tensor: Any  # ttml.autograd.Tensor
    info: TensorInfo
    declaration: dict = field(default_factory=dict)


def label_of(value) -> Label:
    """The topology label of a ttnn tensor as a ``Label``, shard dims normalized against the tensor's rank."""
    topology = value.tensor_topology()
    placements = [
        Placement.shard(p.dim) if isinstance(p, ttnn.PlacementShard) else Placement.replicate()
        for p in topology.placements()
    ]
    return Label.normalized(
        placements,
        list(topology.distribution_shape()),
        [tuple(c) for c in topology.mesh_coords()],
        rank=len(value.shape),
    )


def dtype_name(value) -> str:
    try:
        return DTYPE_NAMES[value.dtype]
    except KeyError:
        raise CheckpointError(
            f"tensor dtype {value.dtype} cannot be checkpointed (supported: {sorted(TTNN_DTYPES)})"
        ) from None


def layout_name(value) -> str:
    return "TILE" if value.layout == ttnn.TILE_LAYOUT else "ROW_MAJOR"


def _root_name(model, params) -> str:
    for name in params:
        return name.split("/", 1)[0]
    return type(model).__name__


def _python_parameters(model, params) -> tuple[dict[int, list[str]], dict[int, dict]]:
    """Walk the Python module tree. For every materialized ``Parameter`` reachable from ``model``, the fully
    qualified names it is reachable under (the C++ naming, ``Root/path/attr``) and its ``checkpoint_layout``
    declaration, both keyed by the identity of the autograd tensor."""
    root = _root_name(model, params)
    names: dict[int, list[str]] = {}
    declarations: dict[int, dict] = {}
    for path, module in model.named_modules():
        for attr, value in vars(module).items():
            if not isinstance(value, Parameter):
                continue
            if value.is_lazy:
                raise LazyParameterError(
                    f"{root}/{path}/{attr} has not been materialized; call ttml.materialize_module first"
                )
            tensor = value.tensor
            fqn = "/".join([root] + ([path.replace(".", "/")] if path else []) + [attr])
            names.setdefault(id(tensor), []).append(fqn)
            declaration = getattr(value, "checkpoint_layout", None)
            if declaration:
                declarations[id(tensor)] = dict(declaration)
    return names, declarations


def _lint_fusion(key: str, label: Label, declaration: dict) -> None:
    """A TP-sharded weight that looks fused by name but declares no segments loads silently wrong across TP sizes."""
    if "fused" in declaration or label.shard_count(2) <= 1:
        return
    module_name = key.rsplit("/", 2)[-2] if key.count("/") >= 2 else key
    if _FUSION_HINT.search(module_name):
        warnings.warn(
            f"{key} is sharded on dim 2 across {label.shard_count(2)} devices and its name suggests a fused projection, "
            f"but it declares no fused_segments; a checkpoint of it reshards correctly only between equal TP sizes",
            stacklevel=3,
        )


def collect_model(model) -> list[LiveTensor]:
    """Every parameter of ``model`` as a ``LiveTensor``, in ``parameters()`` order."""
    params = model.parameters()
    names, declarations = _python_parameters(model, params)
    live: list[LiveTensor] = []
    for fqn, tensor in params.items():
        value = tensor.get_value(NATIVE)
        key = f"model/{fqn}"
        label = label_of(value)
        local_shape = tuple(int(n) for n in value.shape)
        declaration = declarations.get(id(tensor), {})
        transform = declared_transform(
            global_shape(label, local_shape),
            label,
            fused=declaration.get("fused"),
            logical_extent=declaration.get("logical_extent"),
            key=key,
        )
        _lint_fusion(key, label, declaration)
        live_keys = tuple(sorted({key, *(f"model/{n}" for n in names.get(id(tensor), []))}))
        info = TensorInfo(key, "param", dtype_name(value), layout_name(value), label, local_shape, transform, live_keys)
        live.append(LiveTensor(key, tensor, info, declaration))
    return live


def collect_optimizer(optimizer, params: list[LiveTensor]) -> tuple[list[LiveTensor], dict[str, Any]]:
    """Every tensor of the optimizer's state dict as a ``LiveTensor``, and its scalars.

    A state tensor keyed by a parameter name inherits that parameter's transform and live names; its per-device
    shape must match the parameter's, the one invariant every optimizer keeps (state is created as zeros of its
    parameter), so a mismatch means the optimizer was created before the model was sharded.
    """
    by_fqn = {lt.key[len("model/") :]: lt for lt in params}
    live: list[LiveTensor] = []
    scalars: dict[str, Any] = {}

    def walk(node, path: tuple[str, ...]) -> None:
        if isinstance(node, ttml.NamedParameters):
            for fqn, tensor in node.items():
                value = tensor.get_value(NATIVE)
                key = "optimizer/" + "/".join(path + (fqn,))
                label = label_of(value)
                local_shape = tuple(int(n) for n in value.shape)
                param = by_fqn.get(fqn)
                if param is None:
                    warnings.warn(
                        f"{key} is not keyed by a parameter of the model; saved with an identity layout", stacklevel=3
                    )
                    transform, live_keys, ref = Identity(), (key,), None
                else:
                    if local_shape != param.info.local_shape:
                        raise CheckpointError(
                            f"{key} is {list(local_shape)} per device but its parameter {param.key} is "
                            f"{list(param.info.local_shape)}; was the optimizer created before the model was sharded?"
                        )
                    transform = param.info.transform
                    prefix = "optimizer/" + "/".join(path) + "/"
                    live_keys = tuple(sorted(prefix + n[len("model/") :] for n in param.info.live_keys))
                    ref = param.key
                info = TensorInfo(
                    key,
                    "optimizer_state",
                    dtype_name(value),
                    layout_name(value),
                    label,
                    local_shape,
                    transform,
                    live_keys,
                    ref,
                    path,
                )
                live.append(LiveTensor(key, tensor, info))
        elif isinstance(node, dict):
            for name, child in node.items():
                walk(child, path + (str(name),))
        elif isinstance(node, (bool, int, float, str)):
            scalars["optimizer/" + "/".join(path)] = node
        else:
            raise CheckpointError(f"optimizer state at {'/'.join(path)} has unsupported type {type(node).__name__}")

    walk(optimizer.get_state_dict(), ())
    return live, scalars


def restore_scalars(optimizer, scalars: dict[str, Any]) -> list[str]:
    """Put the checkpoint's scalars into the optimizer's state dict and re-install it; returns the keys applied."""
    if not scalars:
        return []
    state = optimizer.get_state_dict()
    applied: list[str] = []

    def walk(node: dict, path: tuple[str, ...]) -> None:
        for name in list(node):
            child = node[name]
            key = "optimizer/" + "/".join(path + (str(name),))
            if isinstance(child, dict):
                walk(child, path + (str(name),))
            elif isinstance(child, (bool, int, float, str)) and key in scalars:
                node[name] = bool(scalars[key]) if isinstance(child, bool) else type(child)(scalars[key])
                applied.append(key)

    walk(state, ())
    optimizer.set_state_dict(state)
    return applied


def fsdp_gate(model, reshard: bool) -> list[Any]:
    """Every FSDP state on ``model`` must be sharded: a module left gathered by a forward with
    ``reshard_after_forward=False`` holds the full weight while its moments are shards, and its cached shard would
    overwrite whatever is loaded into it at the next backward. Reshards when allowed, raises otherwise."""
    from ttml.fsdp import _ShardedState

    states = []
    for path, module in model.named_modules():
        state = getattr(module, "_fsdp_state", None)
        if state is None:
            continue
        states.append(state)
        if state.sharded_state == _ShardedState.SHARDED:
            continue
        if not reshard:
            raise FSDPStateError(f"FSDP module {path or '<root>'} is not in its sharded state; pass reshard_fsdp=True")
        state._reshard_all_params()
        state.sharded_state = _ShardedState.SHARDED
    return states


def mesh_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def world_record() -> dict:
    device = mesh_device()
    mesh = ttml.maybe_mesh()
    shape = list(mesh.shape) if mesh is not None else list(device.shape)
    axis_names = list(mesh.axis_names) if mesh is not None else [f"_{i}" for i in range(len(shape))]
    return {"group_size": 1, "ranks": [{"world_rank": 0, "mesh_id": 0, "shape": shape, "axis_names": axis_names}]}


def check_label_unchanged(live: LiveTensor, when: str) -> None:
    label = label_of(live.tensor.get_value(NATIVE))
    if label != live.info.label:
        raise TopologyError(f"{live.key}: topology label changed {when}: {live.info.label} -> {label}")
