# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runtime arg handling, compile-time arg merging, and define management.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.cb_allocator import _is_cb_named_arg

logger = logging.getLogger(__name__)


# =============================================================================
# Runtime Arg Handling
# =============================================================================


def _get_core_coords_from_ranges(core_ranges: Any) -> List[Any]:
    """Extract ordered list of CoreCoords from a CoreRangeSet."""
    coords = []
    for cr in core_ranges.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                coords.append(ttnn.CoreCoord(x, y))
    return coords


def _compute_and_concatenate_runtime_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    target_core_range: Optional[Any] = None,
) -> Tuple[Dict[int, int], List[Tuple[Any, List[int]]]]:
    """Compute per-phase RT arg offsets and concatenate per-core args in one pass.

    Returns (offsets, per_core_args):
    - offsets: {phase_idx: cumulative_offset} for RT arg redirect wrappers
    - per_core_args: [(CoreCoord, args), ...] for descriptor runtime_args
    """
    # Determine core coordinates from target or first available kernel
    if target_core_range is not None:
        core_coords = _get_core_coords_from_ranges(target_core_range)
    else:
        core_coords = None
        for pk in phase_kernels:
            kernel = pk.get(kernel_type)
            if kernel is not None:
                core_coords = _get_core_coords_from_ranges(kernel.core_ranges)
                break

    offsets: Dict[int, int] = {}
    cumulative = 0

    if not core_coords:
        for i in range(len(phase_kernels)):
            offsets[i] = 0
        return offsets, []

    num_cols = len(core_coords)
    col_args: List[List[int]] = [[] for _ in range(num_cols)]

    for i, pk in enumerate(phase_kernels):
        offsets[i] = cumulative
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        # Compute max_args across all cores for this phase
        max_args = 0
        for core in core_coords:
            try:
                max_args = max(max_args, len(kernel.runtime_args[core.x][core.y]))
            except (IndexError, KeyError):
                pass

        # Concatenate per-core args, padded to max_args for offset alignment
        for col_idx, core in enumerate(core_coords):
            try:
                args = kernel.runtime_args[core.x][core.y]
                arg_list = list(args)
                col_args[col_idx].extend(arg_list)
                pad_count = max_args - len(arg_list)
                if pad_count > 0:
                    col_args[col_idx].extend([0] * pad_count)
            except (IndexError, KeyError):
                if max_args > 0:
                    col_args[col_idx].extend([0] * max_args)
                if target_core_range is not None:
                    logger.warning(
                        "Phase %d %s: no runtime args for core (%d,%d) "
                        "with target_core_range (stem op may not cover this core)",
                        i,
                        kernel_type,
                        core.x,
                        core.y,
                    )

        cumulative += max_args

    # When building a fused kernel (target_core_range provided), always include all
    # cores — even if no phase contributed per-core RT args (e.g. matmul compute has 0),
    # the barrier suffix still needs to be appended to every core.
    if target_core_range is not None:
        per_core = [(core_coords[i], col_args[i]) for i in range(num_cols)]
    else:
        per_core = [(core_coords[i], col_args[i]) for i in range(num_cols) if col_args[i]]
    return offsets, per_core


def _append_barrier_runtime_args(
    rt_args: List[Tuple[Any, List[int]]],
    barrier_addrs: List[int],
) -> Tuple[List[Tuple[Any, List[int]]], int]:
    """Append barrier L1 flag addresses to each core's runtime args.

    Returns (updated_rt_args, barrier_rt_offset) where barrier_rt_offset
    is the index in each core's args where the barrier addresses start.
    """
    if not rt_args:
        return rt_args, 0

    # Offset = length of first core's existing args (all cores should have same count)
    barrier_offset = len(rt_args[0][1])

    updated = []
    for core_coord, args in rt_args:
        updated.append((core_coord, args + barrier_addrs))

    return updated, barrier_offset


def _append_rebind_runtime_args(
    rt_args: List[Tuple[Any, List[int]]],
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
) -> Tuple[List[Tuple[Any, List[int]]], Optional[int]]:
    """Append rebind (addr, size) pairs to each core's runtime args.

    Flattens rebind_info in sorted phase order.  Each rebind entry
    contributes two RT args: [addr, size].  The same values are appended
    to ALL cores (sharded CB addresses are uniform L1 offsets).

    Returns (updated_rt_args, rebind_rt_offset) where rebind_rt_offset
    is the index in each core's args where the rebind data starts,
    or None if there are no rebinds.
    """
    if not rt_args or not rebind_info:
        return rt_args, None

    # Flatten (addr, size) pairs in deterministic sorted-phase order
    rebind_args: List[int] = []
    for phase_idx in sorted(rebind_info.keys()):
        for _slot_idx, addr, size in rebind_info[phase_idx]:
            rebind_args.extend([addr, size])

    if not rebind_args:
        return rt_args, None

    rebind_offset = len(rt_args[0][1])
    updated = []
    for core_coord, args in rt_args:
        updated.append((core_coord, args + rebind_args))

    return updated, rebind_offset


def _concatenate_common_runtime_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> List[int]:
    """Concatenate common runtime args from all phases."""
    common_args: List[int] = []
    for pk in phase_kernels:
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue
        try:
            common_args.extend(list(kernel.common_runtime_args))
        except (AttributeError, TypeError):
            pass
    return common_args


def _compute_common_rt_arg_offsets(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Optional[Dict[int, int]]:
    """Compute per-phase cumulative offsets for common runtime args.

    Returns None if no phase has common runtime args, otherwise a dict
    mapping phase_idx -> starting offset in the concatenated common args array.
    """
    offsets: Dict[int, int] = {}
    cumulative = 0
    any_common = False

    for i, pk in enumerate(phase_kernels):
        offsets[i] = cumulative
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue
        try:
            n = len(list(kernel.common_runtime_args))
            if n > 0:
                any_common = True
                cumulative += n
        except (AttributeError, TypeError):
            pass

    return offsets if any_common else None


# =============================================================================
# Named Compile-Time Arg Merging
# =============================================================================


def _merge_named_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    barrier_rt_offset: Optional[int] = None,
    phase_remaps: Optional[List[Dict[int, int]]] = None,
) -> List[Tuple[str, int]]:
    """Merge named compile-time args from all phases with phase prefixes.

    Phase 0 keeps original names. Phase N>0 gets "phaseN_" prefix.
    CB-reference args (names starting with "cb_") are remapped to pool slot indices.
    """
    merged = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        remap = phase_remaps[i] if phase_remaps else None

        for name, value in kernel.named_compile_time_args:
            actual_value = value
            # Remap CB-reference named args to pool slot indices
            if remap is not None and _is_cb_named_arg(name, value):
                actual_value = remap.get(value, value)

            if i == 0:
                merged.append((name, actual_value))
            else:
                merged.append((f"phase_{i}_{name}", actual_value))

    # Add barrier runtime arg offset
    if barrier_rt_offset is not None:
        merged.append(("barrier_rt_offset", barrier_rt_offset))

    return merged


def _merge_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Tuple[List[int], Dict[int, int]]:
    """Concatenate all phases' compile-time args and return (merged_args, offsets).

    Phase 0's args go first, then phase 1's, etc.  The offsets dict maps
    phase_idx -> starting index in the merged array.
    """
    merged: List[int] = []
    offsets: Dict[int, int] = {}

    for i, pk in enumerate(phase_kernels):
        offsets[i] = len(merged)
        kernel = pk.get(kernel_type)
        if kernel is not None:
            merged.extend(list(kernel.compile_time_args))

    return merged, offsets


# =============================================================================
# Define Handling
# =============================================================================

# These defines are referenced by LLK headers at include time and cannot
# vary per-phase.  They MUST have identical values across all fused phases.
_MUST_MATCH_DEFINES = frozenset({"REDUCE_OP", "REDUCE_DIM", "BCAST_LLKOP", "BCAST_DIM"})


def _collect_phase_defines(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Tuple[List[Tuple[str, str]], Dict[int, List[Tuple[str, str]]]]:
    """Collect defines: MUST_MATCH at file scope, everything else per-phase.

    Raises ValueError if a MUST_MATCH define has inconsistent values across phases.
    """
    # Collect per-phase define dicts: name -> value
    per_phase_defs: Dict[int, Dict[str, str]] = {}
    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            per_phase_defs[i] = {}
            continue
        defs = {}
        if hasattr(kernel, "defines"):
            for name, value in kernel.defines:
                defs[name] = value
        per_phase_defs[i] = defs

    # Validate MUST_MATCH defines and collect them for file scope
    must_match: List[Tuple[str, str]] = []
    must_match_seen: Dict[str, Tuple[str, int]] = {}  # name -> (value, first_phase)
    for idx, defs in per_phase_defs.items():
        for name, value in defs.items():
            if name not in _MUST_MATCH_DEFINES:
                continue
            if name in must_match_seen:
                prev_val, prev_phase = must_match_seen[name]
                if value != prev_val:
                    raise ValueError(
                        f"Define '{name}' has inconsistent values across phases: "
                        f"phase {prev_phase} has '{prev_val}', phase {idx} has '{value}'. "
                        f"These defines must have identical values in all fused phases "
                        f"because they are referenced by LLK headers at include time."
                    )
            else:
                must_match_seen[name] = (value, idx)
                must_match.append((name, value))

    # All non-MUST_MATCH defines go per-phase
    must_match_names = set(must_match_seen.keys())
    per_phase: Dict[int, List[Tuple[str, str]]] = {}
    for idx, defs in per_phase_defs.items():
        phase_defs = [(n, v) for n, v in sorted(defs.items()) if n not in must_match_names]
        if phase_defs:
            per_phase[idx] = phase_defs

    return must_match, per_phase


def _emit_define_lines(defines: List[Tuple[str, str]]) -> List[str]:
    """Generate ``#define NAME VALUE`` lines from a list of (name, value) pairs."""
    lines = []
    for name, value in defines:
        if value:
            lines.append(f"#define {name} {value}")
        else:
            lines.append(f"#define {name}")
    return lines


def _emit_undef_lines(defines: List[Tuple[str, str]]) -> List[str]:
    """Generate ``#undef NAME`` lines from a list of (name, value) pairs."""
    return [f"#undef {name}" for name, _ in defines]


# =============================================================================
# Validation
# =============================================================================


def _validate_fp32_consistency(op_descriptors: List[OpDescriptor]) -> None:
    """Validate fp32_dest_acc_en consistency across all phases.

    DST_ACCUM_MODE is a compile-time constant that cannot change mid-kernel.
    All fused phases must use the same fp32_dest_acc_en setting.
    """
    fp32_settings = []
    for i, desc in enumerate(op_descriptors):
        for kernel_desc in desc.descriptor.kernels:
            config = kernel_desc.config
            if hasattr(config, "fp32_dest_acc_en"):
                fp32_settings.append((i, config.fp32_dest_acc_en))
                break

    if not fp32_settings:
        return

    fp32_values = {v for _, v in fp32_settings}
    if len(fp32_values) <= 1:
        return

    phases_with = [i for i, v in fp32_settings if v]
    phases_without = [i for i, v in fp32_settings if not v]

    raise ValueError(
        f"fp32_dest_acc_en mismatch: phases {phases_with} use fp32=True, "
        f"phases {phases_without} use fp32=False. "
        f"DST_ACCUM_MODE is a kernel-level hardware setting that cannot be "
        f"changed mid-kernel. All phases must use the same fp32_dest_acc_en "
        f"setting. To fix: create all op descriptors with a consistent "
        f"compute_kernel_config."
    )
