# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The live state: every tensor a model and optimizer hold, with its label, its layout declaration and its names.

Keys are ``model/<fqn>`` for parameters, with ``<fqn>`` the name ``module.parameters()`` surfaces (slash-joined from
the root module's name), ``model/<fqn>`` for buffers likewise (``named_buffers()``, a DeepSeek MoE's expert bias say),
and ``optimizer/<state path>/<fqn>`` for optimizer state tensors. Scalars in the optimizer state (steps, learning
rates, flags) are collected separately and stored in the index.

Topology labels are read from the live tensors and trusted. The one fact a label cannot carry, how a fused
projection orders its segments, comes from the declaration the fusing layer put on its ``Parameter``
(``checkpoint_layout``); the vocab padding likewise. Tied weights are detected by object identity: every Python
attribute path that reaches the same tensor is recorded as a live key, so a destination that surfaces the other name
still resolves the entry.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import ttnn

import ttml
from ttml.fsdp import fsdp_axis_of
from ttml.modules.parameter import Parameter

from ._errors import CheckpointError, LayoutError, LazyParameterError, MissingKeyError, TopologyError
from ._index import Entry
from ._layouts import DECLARATIONS, Identity, Segments, Transform, declared_transform
from ._planner import TensorInfo
from ._topology import Label, Placement, global_shape, tp_split

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

# A module name with qkv, kv, gate_up or fused as a whole underscore-separated part (kv_linear, qkv_proj, w_qkv,
# gate_up_proj), but not DeepSeek's head-major wkv_a/wkv_b, which reshard correctly without a declaration.
_FUSION_HINT = re.compile(r"(?:^|_)(qkv|kv|gate_up|fused)(?:_|$)", re.IGNORECASE)


@dataclass
class LiveTensor:
    """One live tensor: its key, the autograd tensor that holds it, and what the planner needs to know."""

    key: str
    tensor: Any  # ttml.autograd.Tensor
    info: TensorInfo
    # The name suggests a fused projection (kv_linear, qkv_proj, ...), for the parameter or the optimizer state of one.
    fused_name: bool = False


def label_of(value) -> Label:
    """The topology label of a ttnn tensor as a ``Label``, shard dims normalized against the tensor's rank."""
    topology = value.tensor_topology()
    # TODO(awliu-TT): #62347 - map a partial-sum placement to a "partial" Placement once tensor placements carry one;
    # until then every placement that is not a shard is read as a replicate.
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


def _named_modules(model):
    """The Python module tree of ``model``; a C++-bound model (no ``named_modules``) exposes none."""
    walk = getattr(model, "named_modules", None)
    return walk() if walk is not None else ()


def _named_buffers(model):
    """The Python module tree's buffers; a C++-bound model exposes none."""
    walk = getattr(model, "named_buffers", None)
    return walk() if walk is not None else ()


def _cpp_bound_params(model, params) -> list[str]:
    """The names of the parameters held by C++-bound modules: the whole model when the model itself is C++-bound,
    else the parameters under a child that has no Python attributes. Such parameters cannot carry a Python layout
    declaration."""
    if getattr(model, "named_modules", None) is None:
        return list(params)
    root = _root_name(model, params)
    prefixes = [
        "/".join([root, path.replace(".", "/")])
        for path, module in model.named_modules()
        if path and getattr(module, "__dict__", None) is None
    ]
    return [name for name in params if any(name.startswith(prefix + "/") for prefix in prefixes)]


def _python_parameters(model, params) -> tuple[dict[int, list[str]], dict[int, dict]]:
    """Walk the Python module tree. For every materialized ``Parameter`` reachable from ``model``, the fully
    qualified names it is reachable under (the C++ naming, ``Root/path/attr``) and its ``checkpoint_layout``
    declaration, both keyed by the identity of the autograd tensor."""
    root = _root_name(model, params)
    names: dict[int, list[str]] = {}
    declarations: dict[int, dict] = {}
    for path, module in _named_modules(model):
        # A C++-bound child module has no Python attributes (and no __dict__); its parameters still reach the
        # checkpoint through model.parameters(), just without Python declarations.
        for attr, value in getattr(module, "__dict__", {}).items():
            if not isinstance(value, Parameter):
                continue
            fqn = "/".join([root] + ([path.replace(".", "/")] if path else []) + [attr])
            if value.is_lazy:
                raise LazyParameterError(f"{fqn} has not been materialized; call ttml.materialize_module first")
            tensor = value.tensor
            names.setdefault(id(tensor), []).append(fqn)
            declaration = getattr(value, "checkpoint_layout", None)
            if declaration:
                unknown = sorted(set(declaration) - set(DECLARATIONS))
                if unknown:
                    raise LayoutError(
                        f"{fqn}: checkpoint_layout declares {unknown}; the declarations are 'fused' (dim, segment "
                        f"sizes) and 'logical_extent' (dim, logical rows)"
                    )
                declarations[id(tensor)] = dict(declaration)
    return names, declarations


def _module_name(key: str) -> str:
    return key.rsplit("/", 2)[-2] if key.count("/") >= 2 else key


def _lint_fusion(key: str, label: Label, declaration: dict, tensor: Any = None) -> None:
    """A TP-sharded weight that looks fused by name but declares no segments loads silently wrong across TP sizes.

    Only axes other than the parameter's own FSDP axis count: FSDP splits rows that are already in canonical order,
    so an FSDP-only model's fused-looking weights are fine."""
    if "fused" in declaration:
        return
    fsdp_axis = fsdp_axis_of(tensor)
    axes = [a for a in label.shard_axes(2) if a != fsdp_axis]
    shards = int(np.prod([label.dist_shape[a] for a in axes])) if axes else 1
    if shards <= 1:
        return
    if _FUSION_HINT.search(_module_name(key)):
        warnings.warn(
            f"{key} is sharded on dim 2 across {shards} devices and its name suggests a fused projection, but it "
            f"declares no fused_segments; a checkpoint of it reshards correctly only between equal TP sizes",
            stacklevel=3,
        )


def collect_model(model) -> list[LiveTensor]:
    """Every parameter of ``model`` as a ``LiveTensor``, in ``parameters()`` order, then every buffer (role
    ``"buffer"``, identity layout) in ``named_buffers()`` order."""
    params = model.parameters()
    names, declarations = _python_parameters(model, params)
    cpp_bound = _cpp_bound_params(model, params)
    if cpp_bound:
        warnings.warn(
            f"{len(cpp_bound)} of {len(params)} parameters (such as {cpp_bound[0]}) belong to C++-bound modules, which "
            f"cannot declare fused or padded layouts: a checkpoint of them restores correctly only onto the same TP size "
            f"it was saved at",
            stacklevel=3,
        )
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
        _lint_fusion(key, label, declaration, tensor)
        live_keys = tuple(sorted({key, *(f"model/{n}" for n in names.get(id(tensor), []))}))
        info = TensorInfo(
            key,
            "param",
            dtype_name(value),
            layout_name(value),
            label,
            local_shape,
            transform,
            live_keys,
            tp_split=tp_split(label, fsdp_axis_of(tensor), len(local_shape)),
        )
        fused_name = bool(_FUSION_HINT.search(_module_name(key)))
        live.append(LiveTensor(key, tensor, info, fused_name=fused_name))
    live.extend(_collect_buffers(model, _root_name(model, params)))
    return live


def _collect_buffers(model, root: str) -> list[LiveTensor]:
    """Buffers are state a training step updates without the optimizer (a MoE's load-balancing bias): saved and
    loaded like parameters, under role ``"buffer"``. A buffer reachable under several names is one entry."""
    names: dict[int, list[str]] = {}
    tensors: dict[int, Any] = {}
    for name, tensor in _named_buffers(model):  # the autograd tensors the Buffer wrappers registered
        tensors[id(tensor)] = tensor
        names.setdefault(id(tensor), []).append(f"model/{root}/{name.replace('.', '/')}")
    live: list[LiveTensor] = []
    for ident, keys in names.items():
        tensor = tensors[ident]
        value = tensor.get_value(NATIVE)
        label = label_of(value)
        local_shape = tuple(int(n) for n in value.shape)
        live_keys = tuple(sorted(set(keys)))
        info = TensorInfo(
            keys[0],
            "buffer",
            dtype_name(value),
            layout_name(value),
            label,
            local_shape,
            Identity(),
            live_keys,
            tp_split=tp_split(label, None, len(local_shape)),
        )
        live.append(LiveTensor(keys[0], tensor, info))
    return live


def collect_optimizer(optimizer, params: list[LiveTensor]) -> tuple[list[LiveTensor], dict[str, Any]]:
    """Every tensor of the optimizer's state dict as a ``LiveTensor``, and its scalars.

    A state tensor keyed by a parameter name inherits that parameter's transform and live names; its per-device and
    global shapes must match the parameter's, the invariant every optimizer keeps (state is created as zeros of its
    parameter): a per-device mismatch means the optimizer was created before the model was sharded, a global one that
    the state is laid out over the mesh unlike its parameter.
    """
    by_fqn = {lt.key[len("model/") :]: lt for lt in params if lt.info.role == "param"}
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
                    transform, live_keys, ref, fsdp_axis, fused_name = Identity(), (key,), None, None, False
                else:
                    if local_shape != param.info.local_shape:
                        raise CheckpointError(
                            f"{key} is {list(local_shape)} per device but its parameter {param.key} is "
                            f"{list(param.info.local_shape)}; was the optimizer created before the model was sharded?"
                        )
                    # Equal per-device shapes still allow a moment labelled Replicate on the parameter's TP axis: save
                    # would write one device's piece as the whole moment and load would put it on every device.
                    if global_shape(label, local_shape) != param.info.global_shape:
                        raise CheckpointError(
                            f"{key} is {list(global_shape(label, local_shape))} over the mesh but its parameter "
                            f"{param.key} is {list(param.info.global_shape)}: it is not sharded as its parameter is"
                        )
                    transform = param.info.transform
                    prefix = "optimizer/" + "/".join(path) + "/"
                    live_keys = tuple(sorted(prefix + n[len("model/") :] for n in param.info.live_keys))
                    ref = param.key
                    fsdp_axis = fsdp_axis_of(param.tensor)  # the moments are sharded like their parameter
                    fused_name = param.fused_name
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
                    tp_split=tp_split(label, fsdp_axis, len(local_shape)),
                )
                live.append(LiveTensor(key, tensor, info, fused_name=fused_name))
        elif isinstance(node, dict):
            for name, child in node.items():
                walk(child, path + (str(name),))
        elif isinstance(node, (bool, int, float, str)):
            scalars["optimizer/" + "/".join(path)] = node
        else:
            raise CheckpointError(f"optimizer state at {'/'.join(path)} has unsupported type {type(node).__name__}")

    walk(optimizer.get_state_dict(), ())
    return live, scalars


def _undescribed_split(transform: Transform, split: tuple[int, ...]) -> tuple[int, ...]:
    """The TP split of a fused-looking weight that its layout does not describe: none when its transform reorders
    segments (a declared fusion), else the TP split itself."""
    return (1,) * len(split) if isinstance(transform, Segments) else tuple(split)


def check_undeclared_fusion(live: LiveTensor, entry: Entry) -> None:
    """Refuse to load a fused projection whose segments would be scrambled.

    Under TP a fused weight's rows are in a per-shard order ([Q_0 K_0 V_0 Q_1 K_1 V_1 ...]) that only a
    ``fused_segments`` declaration describes. Without one, on either side, the layout says nothing about that order,
    and the planner copies rows as if they were canonical: right when both sides leave the same TP split undescribed
    (the same TP size, both undeclared), scrambled otherwise. So for a weight whose name suggests a fused projection,
    the TP split left undescribed must be the same at save (recorded in the index) and at load. A declared side and a
    TP split of one leave nothing undescribed; FSDP's split never counts, since it splits rows already in canonical
    order. Python layers declare ``fused_segments``; C++-bound modules cannot yet."""
    if not live.fused_name:
        return
    before = _undescribed_split(entry.transform, entry.tp_split)
    after = _undescribed_split(live.info.transform, live.info.tp_split)
    if before != after:
        raise LayoutError(
            f"{live.key}: its name suggests a fused projection whose per-shard segment order is not declared on both "
            f"sides: the checkpoint leaves a TP split of {list(before)} per dim undescribed and the live model "
            f"{list(after)}, so its segments would be scrambled. Declare fused_segments where the layer is built "
            f"(C++-bound modules cannot yet: #65584), or load at the TP size and with the declarations it was saved "
            f"with"
        )


def check_param_groups(optimizer, scalars: dict[str, Any], strict: bool) -> None:
    """The live optimizer's param groups must fit the checkpoint's: the same number of groups, and every live member
    of a group among the checkpoint's members of that group. The checkpoint records each group's members over every
    rank of the saving world (a rank of a multi-mesh world holds only its own), so one rank's optimizer may hold fewer
    (under ``strict``, ``check_member_union`` then requires the ranks to hold every one between them), while a param
    in another group, or one the checkpoint never saw, is refused. Checked before any tensor is placed, so a mismatch
    leaves the model and optimizer untouched.

    A checkpoint without optimizer scalars was saved without an optimizer. A strict load refuses it when the live
    optimizer records param groups: an optimizer whose state is scalars only (SGD without momentum, say) would
    otherwise restore nothing without a word. A non-strict load has nothing to check."""
    state = optimizer.get_state_dict()
    live_count = state.get("num_param_groups")
    saved_count = scalars.get("optimizer/num_param_groups")
    if saved_count is None:
        if strict and live_count is not None:
            raise MissingKeyError(
                "the checkpoint holds no optimizer state (no optimizer/num_param_groups); load it with optimizer=None, "
                "or with strict=False to keep the live optimizer's state"
            )
        return
    if live_count is not None and int(saved_count) != int(live_count):
        raise CheckpointError(
            f"the checkpoint has {saved_count} optimizer param groups, the live optimizer {live_count}"
        )
    for group in range(int(saved_count)):
        name = f"param_groups.{group}.params"
        saved, live_members = scalars.get("optimizer/" + name), state.get(name)
        if saved is None or live_members is None:
            lacking = " and ".join(
                side
                for side, members in (("the checkpoint", saved), ("the live optimizer", live_members))
                if members is None
            )
            warnings.warn(
                f"optimizer param group {group}: {lacking} records no member list, so the group's params cannot be "
                f"checked; if they differ, the group's options (learning rate, weight decay, ...) apply to other params",
                stacklevel=3,
            )
            continue
        saved_set = {n for n in saved.split("\n") if n}
        live_set = {n for n in live_members.split("\n") if n}
        extra = live_set - saved_set
        if extra:
            raise CheckpointError(
                f"optimizer param group {group} holds params the checkpoint's group {group} does not: "
                f"{sorted(extra)[:3]}; build the optimizer with the same param groups"
            )


def live_group_members(optimizer) -> dict[str, list[str]]:
    """The live optimizer's members per param group (the group's index as a string), for ``check_member_union``; a
    group without a member list is left out."""
    state = optimizer.get_state_dict()
    members: dict[str, list[str]] = {}
    for group in range(int(state.get("num_param_groups") or 0)):
        names = state.get(f"param_groups.{group}.params")
        if names is not None:
            members[str(group)] = sorted(n for n in names.split("\n") if n)
    return members


def check_member_union(scalars: dict[str, Any], live_by_rank: list[dict[str, list[str]]]) -> None:
    """Under ``strict``: the ranks' live members of each param group, together, must be the checkpoint's members of
    that group. A param the checkpoint's group holds and no rank's optimizer holds would otherwise stop being trained
    without a word when the optimizer keeps no per-parameter state (SGD without momentum), since no state tensor of
    it is left over for ``strict`` to refuse. ``live_by_rank`` holds each rank's ``live_group_members``; on one process
    that is its own, and the members must equal the checkpoint's."""
    for group in range(int(scalars.get("optimizer/num_param_groups") or 0)):
        saved = scalars.get(f"optimizer/param_groups.{group}.params")
        lists = [live[str(group)] for live in live_by_rank if str(group) in live]
        if saved is None or not lists:  # no member list to compare: check_param_groups warned
            continue
        missing = {n for n in saved.split("\n") if n} - set().union(*lists)
        if missing:
            raise CheckpointError(
                f"optimizer param group {group}: the checkpoint's group holds params no rank's optimizer holds: "
                f"{sorted(missing)[:3]}; build the optimizer with the same param groups, or load with strict=False"
            )


def is_structural_scalar(key: str) -> bool:
    """A scalar that describes the optimizer's own structure rather than its progress: the parameter names of a
    param group. It is never written back (each rank's optimizer keeps its own list, which on a multi-mesh world
    holds only that rank's parameters); ``check_param_groups`` checks it against the checkpoint's list, the union over
    the saving ranks, before anything is placed."""
    return bool(_PARAM_GROUP_MEMBERS.search(key))


_PARAM_GROUP_MEMBERS = re.compile(r"(^|/)param_groups\.\d+\.params$")


def restore_scalars(optimizer, scalars: dict[str, Any]) -> list[str]:
    """Put the checkpoint's scalars into the optimizer's state dict and re-install it; returns the keys applied.
    Structural scalars (``is_structural_scalar``) keep the live optimizer's own value."""
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
            elif is_structural_scalar(key):
                continue
            elif isinstance(child, (bool, int, float, str)) and key in scalars:
                node[name] = bool(scalars[key]) if isinstance(child, bool) else type(child)(scalars[key])
                applied.append(key)

    walk(state, ())
    optimizer.set_state_dict(state)
    return applied


def reshard_fsdp_modules(model) -> list[Any]:
    """Reshard every FSDP module of ``model`` and return their states. A module left gathered by a forward with
    ``reshard_after_forward=False`` holds the full weight while its moments are shards, and its cached shard would
    overwrite whatever is loaded into it at the next backward. Resharding is always safe: ``reshard()`` does nothing to
    a sharded module, and the next forward or backward gathers it again as it does under
    ``reshard_after_forward=True``."""
    states = []
    for _, module in _named_modules(model):
        state = getattr(module, "_fsdp_state", None)
        if state is None:
            continue
        states.append(state)
        module.reshard()  # installed by fully_shard
    return states


def check_threads(threads: int | None) -> None:
    if threads is not None and int(threads) < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")


def mesh_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def local_coords() -> list[tuple[int, ...]]:
    """The mesh coordinates this process holds: every coordinate on a single-host mesh, this host's devices of a
    multi-host mesh. get_local_mesh_coord_range is their bounding box, which also holds other hosts' devices when this
    host's do not form a box, so each coordinate is checked."""
    view = mesh_device().get_view()
    return [tuple(int(x) for x in c) for c in view.get_local_mesh_coord_range() if view.is_local(c)]


def mesh_id() -> int:
    """The id the control plane gave this process's mesh; a plain single-host mesh reports 0."""
    return int(mesh_device().get_system_mesh_id())


def rank_record(rank: int, world_rank: int) -> dict:
    """This process's entry of the world record: its ranks, mesh id, the global mesh shape and axis names, and
    the coordinates it holds."""
    device = mesh_device()
    mesh = ttml.maybe_mesh()
    shape = list(mesh.shape) if mesh is not None else list(device.shape)
    axis_names = list(mesh.axis_names) if mesh is not None else [f"_{i}" for i in range(len(shape))]
    return {
        "rank": int(rank),
        "world_rank": int(world_rank),
        "mesh_id": mesh_id(),
        "shape": shape,
        "axis_names": axis_names,
        "local_coords": [list(c) for c in local_coords()],
    }


def check_label_unchanged(live: LiveTensor, when: str) -> None:
    label = label_of(live.tensor.get_value(NATIVE))
    if label != live.info.label:
        raise TopologyError(f"{live.key}: topology label changed {when}: {live.info.label} -> {label}")
