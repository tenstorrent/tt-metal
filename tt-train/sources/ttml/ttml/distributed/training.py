# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Training helpers: parallelize_module."""

from __future__ import annotations

import re
from typing import Dict, Optional

from ttml.modules import AbstractModuleBase

from .mesh_runtime import MeshRuntime, set_runtime
from .rules.registry import get_module_rule
from .style import ParallelStyle
from ._register_ops import init_ops


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
        rule(module, mesh_device, tp_axis, cp_axis)
        # Recurse into children only (do not also apply a ParallelStyle to this
        # composite if the plan matches this prefix—rules own the parent node).
        for name, child in module.named_children():
            if isinstance(child, AbstractModuleBase):
                child_prefix = f"{prefix}.{name}" if prefix else name
                _apply_parallelize_plan(child, mesh_device, plan, tp_axis, cp_axis, child_prefix)
        return

    # Leaf / non-composite: match a style, or recurse to find children.
    style = _match_plan(prefix, plan)
    if style is not None:
        style._apply(module, mesh_device, tp_axis)
        return

    for name, child in module.named_children():
        if isinstance(child, AbstractModuleBase):
            child_prefix = f"{prefix}.{name}" if prefix else name
            _apply_parallelize_plan(child, mesh_device, plan, tp_axis, cp_axis, child_prefix)


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
            "fc": ColwiseParallel(gather_output=True),
        })
    """
    runtime = MeshRuntime(mesh_device=mesh_device, tp_axis=tp_axis, cp_axis=cp_axis)
    init_ops()
    set_runtime(runtime)

    _apply_parallelize_plan(module, mesh_device, parallelize_plan, tp_axis, cp_axis, prefix="")

    return module
