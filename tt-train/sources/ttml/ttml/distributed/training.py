# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Training helpers: distribute_tensor, distribute_module, sync_gradients.

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
    # Preserve requires_grad from original tensor if not explicitly specified
    orig_requires_grad = (
        tensor.get_requires_grad() if hasattr(tensor, "get_requires_grad") else False
    )
    final_requires_grad = (
        requires_grad if requires_grad is not None else orig_requires_grad
    )

    # Use composer to gather tensor from mesh (needed when mesh is open)
    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0)

    try:
        np_data = tensor.to_numpy(ttnn.DataType.FLOAT32, composer)
    except Exception:
        np_data = tensor.to_numpy(None, composer)

    # Composer concatenates all devices along dim 0, take first slice for replicated data
    if np_data.shape[0] > 1:
        np_data = np_data[:1]

    np_bf16 = np_data.astype(ml_dtypes.bfloat16)

    shard_dim = None
    shard_axis = None
    for axis, p in enumerate(layout.placements):
        if isinstance(p, Shard):
            shard_dim = p.dim
            shard_axis = axis
            break

    mapper = None
    if shard_dim is not None:
        rank = len(np_bf16.shape)
        dim = shard_dim if shard_dim >= 0 else rank + shard_dim
        mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
            mesh_device, dim, shard_axis
        )
    else:
        # Use replicate mapper for fully replicated tensors to get correct 2D topology
        mapper = ttml.core.distributed.replicate_tensor_to_mesh_mapper(mesh_device)

    result = ttml.autograd.Tensor.from_numpy(
        np_bf16,
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
        mapper,
    )
    # Restore requires_grad status
    result.set_requires_grad(final_requires_grad)
    set_layout(result, layout)
    return result


# ---------------------------------------------------------------------------
# distribute_batch
# ---------------------------------------------------------------------------


def distribute_batch(
    tensor,
    mesh_device,
    dp_axis: int,
) -> Any:
    """Distribute a batch tensor across the DP axis.

    Each DP rank gets a slice of the batch. The tensor is sharded
    along dimension 0 (batch dimension) across the dp_axis of the mesh.

    Args:
        tensor: Input batch tensor (batch dimension is dim 0)
        mesh_device: The mesh device
        dp_axis: The data parallel mesh axis

    Returns:
        Distributed tensor with batch sharded across dp_axis
    """
    mesh_shape = mesh_device.shape
    ndim = mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)

    # Build layout: Shard(0) on dp_axis, Replicate on others
    placements = tuple(Shard(0) if i == dp_axis else Replicate() for i in range(ndim))
    layout = Layout(placements=placements)

    return distribute_tensor(tensor, mesh_device, layout)


# ---------------------------------------------------------------------------
# distribute_module
# ---------------------------------------------------------------------------


def distribute_module(
    model: AbstractModuleBase,
    mesh_device,
    policy: Dict[str, Layout],
) -> AbstractModuleBase:
    """Distribute a model for parallel training.

    1. Walks the module tree.
    2. For each module, checks if a module-level rule is registered.
       If yes, applies the transform (the rule may distribute sub-parameters).
    3. Otherwise, distributes individual parameters that appear in *policy*.

    Args:
        model: The model to distribute
        mesh_device: The mesh device to distribute tensors to
        policy: Dict mapping parameter names (or regex patterns) to Layouts
    """
    from . import module_rules as _  # ensure module rules are registered  # noqa: F401

    # Set up a minimal runtime for dispatch to use
    runtime = MeshRuntime(mesh_device=mesh_device)
    set_runtime(runtime)

    _apply_module_rules(model, mesh_device, policy, prefix="")

    return model


def _apply_module_rules(
    module: AbstractModuleBase,
    mesh_device,
    policy: Dict[str, Layout],
    prefix: str,
) -> None:
    """Recursively apply module rules or direct parameter distribution."""
    rule = get_module_rule(type(module))

    if rule is not None:
        rule(module, mesh_device, policy, prefix)
        return

    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if isinstance(child, AbstractModuleBase):
            _apply_module_rules(child, mesh_device, policy, child_prefix)

    _distribute_unhandled_params(module, mesh_device, policy, prefix)


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


def _distribute_unhandled_params(
    module: AbstractModuleBase,
    mesh_device,
    policy: Dict[str, Layout],
    prefix: str,
) -> None:
    """Distribute individual parameters that haven't been handled by a module rule.

    IMPORTANT: We must call override_tensor() to update the C++ side's m_named_tensors
    map. This ensures the old tensor can be deallocated and the optimizer (which calls
    model.parameters()) will get the new distributed tensors.
    """
    for attr_name in list(vars(module).keys()):
        attr = getattr(module, attr_name, None)
        if attr is None:
            continue
        if not hasattr(attr, "tensor"):
            continue
        param_key = f"{prefix}.{attr_name}" if prefix else attr_name
        param_key_with_tensor = f"{param_key}.weight"
        matched = _match_policy(param_key, policy)
        if matched is None:
            matched = _match_policy(param_key_with_tensor, policy)
        if matched is not None:
            new_tensor = distribute_tensor(attr.tensor, mesh_device, matched)
            # Update both Python side (Parameter.tensor) and C++ side (m_named_tensors)
            attr.tensor = new_tensor
            # Try to update C++ side - attr_name may be "weight" for Parameter wrappers
            try:
                module.override_tensor(new_tensor, attr_name)
            except Exception:
                # If attr_name doesn't match C++ registration name, try common names
                pass


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
