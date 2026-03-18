# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Training helpers: distribute_tensor, parallelize_module, sync_gradients.

These are the main user-facing entry points that wire the rule-based layout
system into the model initialization and gradient synchronization steps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Union

import numpy as np
import ml_dtypes
import ttnn

import ttml
from ttml.modules import AbstractModuleBase
from ttml.modules.module_base import ModuleList

from .layout import Layout, Shard, Replicate, get_layout, set_layout, replicated_layout
from .mesh_runtime import MeshRuntime, get_runtime, set_runtime
from .rules.registry import get_module_rule
from .style import ParallelStyle


# ---------------------------------------------------------------------------
# distribute_tensor
# ---------------------------------------------------------------------------


def distribute_tensor(
    tensor,
    mesh_device,
    layout: Layout,
    requires_grad: Optional[bool] = None,
) -> Any:
    """Distribute a single ttml autograd tensor to *mesh_device* with *layout*.

    The underlying ttnn tensor is round-tripped through NumPy for sharding.
    The TensorPtr wrapper preserves ``requires_grad`` status.
    Layout metadata is stamped on the result.

    Args:
        tensor: The tensor to distribute
        mesh_device: The mesh device to distribute to
        layout: The target layout
        requires_grad: If provided, override requires_grad on result.
                       If None, preserves original tensor's requires_grad status.
    """
    # Preserve requires_grad and dtype from original tensor (ttml.autograd.Tensor: get_requires_grad, dtype)
    orig_requires_grad = tensor.get_requires_grad()
    final_requires_grad = (
        requires_grad if requires_grad is not None else orig_requires_grad
    )
    orig_dtype = tensor.dtype()

    # Use composer to gather tensor from mesh (needed when mesh is open)
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)

    np_data = tensor.to_numpy(orig_dtype, composer)

    # Composer concatenates all devices along dim 0, take first slice for replicated data
    if np_data.shape[0] > 1:
        np_data = np_data[:1]

    shard_dim = None
    shard_axis = None
    for axis, p in enumerate(layout.placements):
        if isinstance(p, Shard):
            shard_dim = p.dim
            shard_axis = axis
            break

    mapper = None
    if shard_dim is not None:
        rank = len(np_data.shape)
        dim = shard_dim if shard_dim >= 0 else rank + shard_dim
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
            mesh_device, dim, shard_axis
        )
    else:
        # Use replicate mapper for fully replicated tensors to get correct 2D topology
        mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(mesh_device)

    result = ttml.autograd.Tensor.from_numpy(
        np_data,
        ttnn.Layout.TILE,
        orig_dtype,
        mapper,
    )
    # Restore requires_grad status
    result.set_requires_grad(final_requires_grad)
    set_layout(result, layout)
    return result


# ---------------------------------------------------------------------------
# parallelize_module (PyTorch-style ParallelStyle API)
# ---------------------------------------------------------------------------


def _match_plan(name: str, plan: Dict[str, ParallelStyle]) -> Optional[ParallelStyle]:
    """Match module name against plan patterns (exact first, then regex)."""
    if name in plan:
        return plan[name]
    for pattern, style in plan.items():
        try:
            if re.fullmatch(pattern, name):
                return style
        except re.error:
            continue
    return None


def _build_policy_for_composite(
    module: AbstractModuleBase,
    prefix: str,
    plan: Dict[str, ParallelStyle],
    mesh_device,
    tp_axis: int,
) -> Optional[Dict[str, Layout]]:
    """Build a Layout policy from parallelize_plan for a composite module (e.g. GQA)."""
    try:
        from ttml.models.llama.gqattn import GroupedQueryAttention
    except ImportError:
        return None
    if not isinstance(module, GroupedQueryAttention):
        return None
    policy: Dict[str, Layout] = {}
    for name in ("q_linear", "kv_linear", "out_linear"):
        module_path = f"{prefix}.{name}" if prefix else name
        style = _match_plan(module_path, plan)
        if style is not None and hasattr(style, "get_layout"):
            key = f"{prefix}.{name}.weight" if prefix else f"{name}.weight"
            policy[key] = style.get_layout(mesh_device, tp_axis)
    # Return policy (possibly empty) so rule is always invoked (e.g. for CP-only)
    return policy


def _apply_parallelize_plan(
    module: AbstractModuleBase,
    mesh_device,
    plan: Dict[str, ParallelStyle],
    tp_axis: int,
    cp_axis: Optional[int],
    prefix: str,
) -> None:
    """Recursively apply parallelize_plan to module tree."""
    # If this module has a registered module rule (e.g. GQA), run it then recurse into children.
    # The rule handles composite-only concerns (e.g. head count, CP/rope, ring_sdpa); child linears
    # get weight sharding + forward collectives from the plan when we recurse (Colwise/Rowwise).
    rule = get_module_rule(type(module))
    if rule is not None:
        composite_policy = _build_policy_for_composite(
            module, prefix, plan, mesh_device, tp_axis
        )
        if composite_policy is not None:
            from . import module_rules as _  # noqa: F401

            try:
                rule(module, mesh_device, composite_policy, prefix, cp_axis=cp_axis)
            except TypeError:
                rule(module, mesh_device, composite_policy, prefix)
            # Recurse so q_linear, kv_linear, out_linear get ColwiseParallel/RowwiseParallel
            for name, child in module.named_children():
                if isinstance(child, AbstractModuleBase):
                    child_prefix = f"{prefix}.{name}" if prefix else name
                    _apply_parallelize_plan(
                        child, mesh_device, plan, tp_axis, cp_axis, child_prefix
                    )
            return

    # Check if this module matches a style
    style = _match_plan(prefix, plan) if prefix else None
    if style is not None:
        style._apply(module, mesh_device, tp_axis)
        return

    # Recurse into children
    for name, child in module.named_children():
        if isinstance(child, AbstractModuleBase):
            child_prefix = f"{prefix}.{name}" if prefix else name
            _apply_parallelize_plan(
                child, mesh_device, plan, tp_axis, cp_axis, child_prefix
            )


def parallelize_module(
    module: AbstractModuleBase,
    mesh_device,
    parallelize_plan: Dict[str, ParallelStyle],
    *,
    tp_axis: int = 0,
    cp_axis: Optional[int] = None,
) -> AbstractModuleBase:
    """Apply tensor parallelism by assigning ParallelStyle to modules by name pattern.

    Args:
        module: The model to parallelize.
        mesh_device: The mesh device.
        parallelize_plan: Dict mapping module name patterns (exact or regex) to ParallelStyle.
        tp_axis: Mesh axis for tensor parallelism.
        cp_axis: Optional mesh axis for context parallelism.

    Example:
        parallelize_module(model, mesh, {
            r".*\\.(q_linear|kv_linear|w1|w3)": ColwiseParallel(),
            r".*\\.(out_linear|w2)": RowwiseParallel(),
            "fc": ColwiseParallel(output_gradient_replicated=True),
        })
    """
    runtime = MeshRuntime(mesh_device=mesh_device, tp_axis=tp_axis, cp_axis=cp_axis)
    set_runtime(runtime)

    _apply_parallelize_plan(
        module, mesh_device, parallelize_plan, tp_axis, cp_axis, prefix=""
    )

    return module


def _match_policy(param_key: str, policy: Dict[str, Layout]) -> Optional[Layout]:
    """Look up *param_key* in *policy*, supporting both exact and regex keys.

    Exact keys are tried first.  If no exact match, each key is compiled as a
    regex and tested with ``re.fullmatch``.  The first matching regex wins.
    """
    if param_key in policy:
        return policy[param_key]
    for pattern, layout in policy.items():
        try:
            if re.fullmatch(pattern, param_key):
                return layout
        except re.error:
            continue
    return None


# ---------------------------------------------------------------------------
# sync_gradients
# ---------------------------------------------------------------------------


def sync_gradients(
    model: AbstractModuleBase,
    runtime: Optional[MeshRuntime] = None,
    cluster_axes: Optional[list] = None,
) -> None:
    """Synchronize gradients across the specified cluster axes.

    Args:
        model: The model whose gradients to synchronize.
        runtime: MeshRuntime to infer DP/CP axes from (if cluster_axes not given).
        cluster_axes: Explicit list of mesh axes to all_reduce gradients across.
                      If None, inferred from runtime (DP axis + CP axis if enabled).

    For DP: all_reduce gradients across dp_axis.
    For TP: gradients stay sharded (optimizer updates local shard).
    """
    if cluster_axes is None:
        if runtime is None:
            runtime = get_runtime()
        if runtime is None:
            return
        cluster_axes = []
        if runtime.is_dp_enabled:
            cluster_axes.append(runtime.dp_axis)
        if runtime.is_cp_enabled:
            cluster_axes.append(runtime.cp_axis)

    if not cluster_axes:
        return

    ttml.core.distributed.synchronize_gradients(model.parameters(), cluster_axes)
