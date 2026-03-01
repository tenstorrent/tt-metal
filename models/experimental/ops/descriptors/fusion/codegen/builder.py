# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fused descriptor builder: validation, barrier configuration, and build orchestration.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.common import (
    BarrierConfig,
    MultiBarrierSpec,
    _BuildResult,
    _get_role_key,
    _kernel_overlaps_core_range,
)
from models.experimental.ops.descriptors.fusion.cb_allocator import (
    PhaseInfo,
    CBPoolAllocator,
    extract_cb_info,
    _get_phantom_cb_indices,
    _compute_rebind_info,
    _extract_remote_cb_indices,
)
from models.experimental.ops.descriptors.fusion.codegen.source_gen import (
    _generate_fused_source,
)
from models.experimental.ops.descriptors.fusion.codegen.args import (
    _get_core_coords_from_ranges,
    _compute_and_concatenate_runtime_args,
    _append_barrier_runtime_args,
    _append_rebind_runtime_args,
    _concatenate_common_runtime_args,
    _compute_common_rt_arg_offsets,
    _merge_named_compile_time_args,
    _merge_compile_time_args,
    _collect_phase_defines,
    _validate_fp32_consistency,
)


# =============================================================================
# Compute Config Validation
# =============================================================================


def _validate_and_get_compute_config_for_role(
    phase_kernels: List[Dict[Any, Any]],
    role_key: Any,
) -> "ttnn.ComputeConfigDescriptor":
    """Validate compute config consistency for a specific role across phases."""
    base = None
    base_phase = -1

    for phase_idx, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue

        config = kernel.config
        if base is None:
            base = config
            base_phase = phase_idx
            continue

        mismatches = []
        for field in ("fp32_dest_acc_en", "math_approx_mode", "math_fidelity", "dst_full_sync_en", "bfp8_pack_precise"):
            base_val = getattr(base, field, None)
            this_val = getattr(config, field, None)
            if base_val != this_val:
                mismatches.append(f"  {field}: phase {base_phase}={base_val}, phase {phase_idx}={this_val}")

        if mismatches:
            raise ValueError(f"Compute config mismatch for role {role_key}.\n" + "\n".join(mismatches))

    if base is None:
        return ttnn.ComputeConfigDescriptor()

    return base


# =============================================================================
# Barrier Configuration
# =============================================================================


def _create_barrier_segment_config(device: Any, core_ranges: Any) -> BarrierConfig:
    """Create a lightweight barrier config for OpGraph segments.

    Only allocates ``global_arrive`` and ``global_release`` GlobalSemaphores
    (2 instead of 4).  The per-core ``compute_done`` / ``writer_done`` flags
    are shared across all segments and allocated separately in
    ``OpGraphBuilder.build()``, so per-segment copies would waste L1.
    """
    config = BarrierConfig()

    sem_global_arrive = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_release = ttnn.create_global_semaphore(device, core_ranges, 0)

    config._sem_refs = [sem_global_arrive, sem_global_release]
    config.global_arrive_addr = ttnn.get_global_semaphore_address(sem_global_arrive)
    config.global_release_addr = ttnn.get_global_semaphore_address(sem_global_release)

    logical_coords = _get_core_coords_from_ranges(core_ranges)
    config.num_cores = len(logical_coords)

    if config.num_cores > 0:
        phys_coords = [device.worker_core_from_logical_core(c) for c in logical_coords]
        config.core0_phys_x = phys_coords[0].x
        config.core0_phys_y = phys_coords[0].y
        config.mcast_start_x = min(c.x for c in phys_coords)
        config.mcast_start_y = min(c.y for c in phys_coords)
        config.mcast_end_x = max(c.x for c in phys_coords)
        config.mcast_end_y = max(c.y for c in phys_coords)

        if config.num_cores > 1:
            _validate_rectangular_grid(phys_coords, config)

    return config


def _validate_rectangular_grid(phys_coords: List[Any], config: BarrierConfig) -> None:
    """Validate that physical cores form a rectangle for safe NOC multicast.

    NOC multicast sends to ALL cores in the bounding box. If the actual core
    set is non-rectangular (e.g., L-shaped), the multicast would write to
    unintended cores, corrupting their L1 memory.
    """
    phys_set = set((c.x, c.y) for c in phys_coords)
    bbox_w = config.mcast_end_x - config.mcast_start_x + 1
    bbox_h = config.mcast_end_y - config.mcast_start_y + 1
    bbox_area = bbox_w * bbox_h
    if len(phys_set) != bbox_area:
        raise ValueError(
            f"Fused kernel global barrier requires rectangular core grid for "
            f"safe NOC multicast. Got {len(phys_set)} physical cores in "
            f"bounding box {bbox_w}x{bbox_h} ({bbox_area} cores). "
            f"Physical coords: {sorted(phys_set)}"
        )


# =============================================================================
# Fused Descriptor Builder
# =============================================================================


def _build_fused_descriptor(
    phases: List[PhaseInfo],
    device: Any,
    target_core_range: Optional[Any] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    cb_pool: Optional[CBPoolAllocator] = None,
) -> _BuildResult:
    """Build a fused ProgramDescriptor from multiple phases with barrier sync.

    Discovers kernel roles dynamically via ``(risc_type, core_ranges)`` keys.

    Args:
        cb_pool: If provided, use this pre-built pool for CB allocation
            (from global pool projection).  Otherwise, self-allocate
            (used by the linear-chain path).
    """
    # Validate fp32 consistency
    _validate_fp32_consistency([p.op_descriptor for p in phases])

    # Discover all kernel roles from ALL phases (not just phase 0).
    # Different ops may have different role sets — e.g. slice has no compute
    # kernel, while layer_norm has reader+compute+writer.  We need the union.
    # When target_core_range is set, skip kernels whose native core ranges
    # do not overlap the target — this prevents e.g. a block-sharded LN's
    # mcast receiver (row 1) from overwriting the sender (row 0) when
    # building a fused kernel for a row-0 branch.
    role_keys: List[Tuple[str, frozenset]] = []
    role_keys_set: Set[Tuple[str, frozenset]] = set()
    for phase in phases:
        for kernel_desc in phase.op_descriptor.descriptor.kernels:
            if not _kernel_overlaps_core_range(kernel_desc, target_core_range):
                continue
            rk = _get_role_key(kernel_desc, target_core_range)
            if rk not in role_keys_set:
                role_keys.append(rk)
                role_keys_set.add(rk)

    # Build phase_kernels as List[Dict[role_key, KernelDescriptor]]
    # and phase_kernel_indices as List[Dict[role_key, kernel_index]]
    phase_kernels: List[Dict[Any, Any]] = []
    phase_kernel_indices: List[Dict[Any, int]] = []
    for phase_idx, phase in enumerate(phases):
        role_map: Dict[Any, Any] = {}
        idx_map: Dict[Any, int] = {}
        for k_idx, kernel_desc in enumerate(phase.op_descriptor.descriptor.kernels):
            if not _kernel_overlaps_core_range(kernel_desc, target_core_range):
                continue
            rk = _get_role_key(kernel_desc, target_core_range)
            role_map[rk] = kernel_desc
            idx_map[rk] = k_idx
        phase_kernels.append(role_map)
        phase_kernel_indices.append(idx_map)

    # Pool-allocate CB slots based on compatibility keys.
    if cb_pool is not None:
        pool = cb_pool
    else:
        # Self-allocate (linear-chain path or single-group trees).
        # Pre-reserve remote CB indices from GlobalCBs — prevents collisions
        # without adding to remaps, so they are excluded from inter-phase CB reset.
        pool = CBPoolAllocator()
        for phase in phases:
            for remote_idx in _extract_remote_cb_indices(phase.op_descriptor.descriptor):
                pool.reserve_index(remote_idx)
        for phase_idx, phase in enumerate(phases):
            phantom_indices = _get_phantom_cb_indices(phase)
            pool.allocate_phase(phase_idx, phase.cb_info, phantom_indices)

    # Compute CB address rebinding info using remapped slot indices.
    rebind_info = _compute_rebind_info(phases, pool.phase_remaps)

    # Build merged CB descriptors from pool (uses stored source_cb/source_fmt references)
    merged_cbs = pool.build_merged_cb_descriptors(phases)

    # Set CB core_ranges to the target when building for a specific core group.
    # Skip GlobalCB-backed descriptors — their core_ranges must stay within
    # the GlobalCircularBuffer's all_cores().
    if target_core_range is not None:
        for cb_desc in merged_cbs:
            if not cb_desc.has_global_circular_buffer():
                cb_desc.core_ranges = target_core_range

    # Per-phase CB slot sets (for targeted reset between phases)
    per_phase_cb_slots: List[List[int]] = []
    for i in range(len(phases)):
        remap = pool.get_remap(i)
        slots = sorted(set(remap.values()))
        per_phase_cb_slots.append(slots)

    # Collect all unique op semaphore (id, initial_value) pairs used by any phase.
    op_semaphore_info: List[Tuple[int, int]] = []
    seen_sem_ids_for_reset: Set[int] = set()
    for phase in phases:
        for sem in phase.op_descriptor.descriptor.semaphores:
            if sem.id not in seen_sem_ids_for_reset:
                op_semaphore_info.append((sem.id, sem.initial_value))
                seen_sem_ids_for_reset.add(sem.id)
    op_semaphore_info.sort(key=lambda x: x[0])

    # Determine whether any phase has a compute kernel.
    # When no compute exists, BRISC barrier skips compute_done semaphore.
    has_compute = any(rk[0] == "compute" for rk in role_keys)

    fused_kernels = []
    kernel_labels = []

    for role_key in role_keys:
        risc_type, core_key = role_key

        # Get role-specific core_ranges
        if target_core_range is not None:
            role_core_ranges = target_core_range
        else:
            role_core_ranges = None
            for pk in phase_kernels:
                kernel = pk.get(role_key)
                if kernel is not None:
                    role_core_ranges = kernel.core_ranges
                    break

        if role_core_ranges is None:
            continue

        # Merge compile-time args, compute RT offsets + concatenate RT args
        ct_args, ct_offsets = _merge_compile_time_args(phase_kernels, role_key)

        rt_offsets, rt_args = _compute_and_concatenate_runtime_args(
            phase_kernels,
            role_key,
            target_core_range=target_core_range,
        )

        # Generate fused source (or reuse from cache)
        role_label = {"riscv_0": "reader", "riscv_1": "writer", "compute": "compute"}.get(risc_type)
        if role_label is None:
            continue

        common_rt_offsets = _compute_common_rt_arg_offsets(phase_kernels, role_key)

        # All phase indices (for dispatcher — ensures barrier participation
        # even for phases where this role has no kernel).
        all_phase_indices = list(range(len(phases)))

        fused_source = _generate_fused_source(
            phase_kernels,
            role_key,
            phases,
            ct_offsets,
            per_phase_cb_slots,
            risc_type=risc_type,
            role_label=role_label,
            rebind_info=rebind_info,
            op_semaphore_info=op_semaphore_info if risc_type == "riscv_0" else None,
            multi_barrier=multi_barrier,
            rt_arg_offsets=rt_offsets,
            common_rt_arg_offsets=common_rt_offsets,
            all_phase_indices=all_phase_indices,
            has_compute=has_compute,
        )

        # Determine barrier RT addresses per RISC type.
        # When no compute kernel exists, BRISC only tracks writer_done
        # (compute_done would never be signaled).
        barrier_addrs: List[int] = []
        if multi_barrier is not None:
            if risc_type == "riscv_0":
                if has_compute:
                    barrier_addrs = [multi_barrier.compute_done_addr, multi_barrier.writer_done_addr]
                else:
                    barrier_addrs = [multi_barrier.writer_done_addr]
                barrier_addrs.append(multi_barrier.reset_done_addr)
                for seg in multi_barrier.segments:
                    barrier_addrs.extend([seg.arrive_addr, seg.release_addr])
            elif risc_type == "riscv_1":
                barrier_addrs = [multi_barrier.writer_done_addr, multi_barrier.reset_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.append(seg.release_addr)
            elif risc_type == "compute":
                barrier_addrs = [multi_barrier.compute_done_addr, multi_barrier.reset_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.append(seg.release_addr)

        if fused_source is None:
            continue

        # Append barrier addresses to runtime args
        rt_args, barrier_offset = _append_barrier_runtime_args(rt_args, barrier_addrs)

        # Merge named compile-time args
        named_ct_args = _merge_named_compile_time_args(
            phase_kernels,
            role_key,
            barrier_rt_offset=barrier_offset if barrier_addrs else None,
            phase_remaps=pool.phase_remaps,
        )
        # Add per-segment named compile-time args (only riscv_0 needs them)
        if multi_barrier is not None and risc_type == "riscv_0":
            for seg_idx, seg in enumerate(multi_barrier.segments):
                s = f"seg{seg_idx}"
                named_ct_args.append((f"{s}_num_cores", seg.config.num_cores))
                named_ct_args.append((f"{s}_core0_phys_x", seg.config.core0_phys_x))
                named_ct_args.append((f"{s}_core0_phys_y", seg.config.core0_phys_y))
                named_ct_args.append((f"{s}_mcast_start_x", seg.config.mcast_start_x))
                named_ct_args.append((f"{s}_mcast_start_y", seg.config.mcast_start_y))
                named_ct_args.append((f"{s}_mcast_end_x", seg.config.mcast_end_x))
                named_ct_args.append((f"{s}_mcast_end_y", seg.config.mcast_end_y))

        # Append rebind addresses as runtime args (not CT to avoid JIT cache busting)
        rt_args, rebind_offset = _append_rebind_runtime_args(rt_args, rebind_info)
        if rebind_offset is not None:
            named_ct_args.append(("rebind_rt_offset", rebind_offset))

        # Get config from first available kernel for this role
        role_config = None
        for pk in phase_kernels:
            kernel = pk.get(role_key)
            if kernel is not None:
                role_config = kernel.config
                break

        # For compute roles, validate configs match across phases and
        # rebuild unpack_to_dest_mode from pool-allocated slot indices
        if risc_type == "compute":
            role_config = _validate_and_get_compute_config_for_role(phase_kernels, role_key)
            role_config.unpack_to_dest_mode = pool.build_unpack_to_dest_mode()

        # Build fused kernel descriptor
        desc = ttnn.KernelDescriptor()
        desc.kernel_source = fused_source
        desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
        desc.core_ranges = role_core_ranges
        desc.compile_time_args = ct_args
        desc.named_compile_time_args = named_ct_args
        # Only MUST_MATCH defines go to the compiler as -D flags.
        # All other defines are handled by #define/#undef in the generated source.
        must_match_defs, _ = _collect_phase_defines(phase_kernels, role_key)
        desc.defines = must_match_defs
        desc.runtime_args = rt_args
        desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, role_key)
        desc.config = role_config
        fused_kernels.append(desc)

        # Compute kernel label from op names (all-or-nothing: only if ALL phases are named)
        role_phase_names = []
        all_named = True
        for phase_idx, pk in enumerate(phase_kernels):
            if pk.get(role_key) is not None:
                name = phases[phase_idx].op_descriptor.name
                if name:
                    role_phase_names.append(name)
                else:
                    all_named = False
        kernel_labels.append("_".join(role_phase_names) if all_named and role_phase_names else "")

    # Merge semaphores (dedup by ID).
    # When building for a specific core group, restrict each semaphore's
    # core_ranges to the target.  Without this, the stem phase's semaphores
    # (covering the full 16-core range) would overlap across groups after merge.
    # We create new SemaphoreDescriptor objects to avoid mutating the originals
    # (no save/restore mechanism for semaphores like there is for CBs).
    all_semaphores = []
    seen_sem_ids: Set[int] = set()
    for phase in phases:
        for sem in phase.op_descriptor.descriptor.semaphores:
            if sem.id not in seen_sem_ids:
                if target_core_range is not None:
                    new_sem = ttnn.SemaphoreDescriptor(
                        id=sem.id,
                        core_type=sem.core_type,
                        core_ranges=target_core_range,
                        initial_value=sem.initial_value,
                    )
                    all_semaphores.append(new_sem)
                else:
                    all_semaphores.append(sem)
                seen_sem_ids.add(sem.id)

    # Collect input/output tensors (use id() for dedup because ttnn Tensor's
    # __eq__ returns an element-wise Tensor, making `in` unreliable)
    all_input_tensors = []
    seen_tensor_ids: Set[int] = set()
    for phase in phases:
        for tensor in phase.op_descriptor.input_tensors:
            tid = id(tensor)
            if tid not in seen_tensor_ids:
                all_input_tensors.append(tensor)
                seen_tensor_ids.add(tid)

    output_tensor = None
    if phases[-1].op_descriptor.output_tensors:
        output_tensor = phases[-1].op_descriptor.output_tensors[0]

    # Create the merged ProgramDescriptor
    merged_descriptor = ttnn.ProgramDescriptor()
    merged_descriptor.kernels = fused_kernels
    merged_descriptor.cbs = merged_cbs
    merged_descriptor.semaphores = all_semaphores

    # Collect semaphore references to prevent GC of GlobalSemaphores
    sem_refs = tuple(multi_barrier._sem_refs) if multi_barrier is not None else ()

    # Build kernel_phase_map: per fused kernel, list of (OpDescriptor, kernel_index)
    # identifying which source phase kernels' RT args were concatenated in.
    kernel_phase_map = []
    for role_key in role_keys:
        sources = []
        for phase_idx, pk in enumerate(phase_kernels):
            if role_key in pk:
                sources.append((phases[phase_idx].op_descriptor, phase_kernel_indices[phase_idx][role_key]))
        kernel_phase_map.append(sources)

    return _BuildResult(
        descriptor=merged_descriptor,
        input_tensors=all_input_tensors,
        output_tensors=[output_tensor] if output_tensor else [],
        semaphores=sem_refs,
        kernel_labels=tuple(kernel_labels),
        kernel_phase_map=tuple(kernel_phase_map),
    )


def _create_phase_info(op_descriptor: OpDescriptor, phase_idx: int) -> PhaseInfo:
    """Create a PhaseInfo from an OpDescriptor.

    Extracts CB info and unpack_to_dest_mode from the op's kernels.
    """
    utd_modes = None
    for kd in op_descriptor.descriptor.kernels:
        config = kd.config
        if hasattr(config, "unpack_to_dest_mode"):
            modes = config.unpack_to_dest_mode
            if modes is not None and len(modes) > 0:
                utd_modes = modes
                break
    cb_info = extract_cb_info(op_descriptor.descriptor, utd_modes)
    return PhaseInfo(phase_idx=phase_idx, op_descriptor=op_descriptor, cb_info=cb_info)
