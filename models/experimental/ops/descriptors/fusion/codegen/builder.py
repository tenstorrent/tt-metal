# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
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
    _NOOP_OP,
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
    _generate_coordinator_only_source,
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


def _create_barrier_segment_config(
    device: Any,
    core_ranges: Any,
    arrive_ranges: Any = None,
) -> BarrierConfig:
    """Create a lightweight barrier config for OpGraph segments.

    Only allocates ``global_arrive`` and ``global_release`` GlobalSemaphores
    (2 instead of 4).  The per-core ``compute_done`` / ``writer_done`` flags
    are shared across all segments and allocated separately in
    ``OpGraphBuilder.build()``, so per-segment copies would waste L1.

    Args:
        core_ranges: CoreRangeSet of ALL cores that receive release (and
            on which GlobalSemaphores are allocated).
        arrive_ranges: CoreRangeSet of cores that arrive at the barrier.
            If None, all release cores also arrive (symmetric barrier).
    """
    config = BarrierConfig()

    sem_global_arrive = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_release = ttnn.create_global_semaphore(device, core_ranges, 0)

    config._sem_refs = [sem_global_arrive, sem_global_release]
    config.global_arrive_addr = ttnn.get_global_semaphore_address(sem_global_arrive)
    config.global_release_addr = ttnn.get_global_semaphore_address(sem_global_release)

    release_coords = _get_core_coords_from_ranges(core_ranges)
    config.num_release_cores = len(release_coords)

    if config.num_release_cores > 0:
        if arrive_ranges is not None:
            arrive_coords = _get_core_coords_from_ranges(arrive_ranges)
            config.num_arrive_cores = len(arrive_coords)
            # Core0 must be an arrive core — use first arrive coord
            arrive_phys = [device.worker_core_from_logical_core(c) for c in arrive_coords]
            config.core0_phys_x = arrive_phys[0].x
            config.core0_phys_y = arrive_phys[0].y
            # Other cores = ALL release cores except core0
            all_phys = [device.worker_core_from_logical_core(c) for c in release_coords]
            config.other_core_phys_coords = [
                (c.x, c.y) for c in all_phys if not (c.x == config.core0_phys_x and c.y == config.core0_phys_y)
            ]
        else:
            # Symmetric: arrive = release (backward compat)
            config.num_arrive_cores = config.num_release_cores
            phys_coords = [device.worker_core_from_logical_core(c) for c in release_coords]
            config.core0_phys_x = phys_coords[0].x
            config.core0_phys_y = phys_coords[0].y
            config.other_core_phys_coords = [(c.x, c.y) for c in phys_coords[1:]]

    return config


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

    # Determine whether any phase has a compute / writer kernel.
    # When no compute exists, coordinator barrier skips compute_done semaphore.
    # When no writer exists, coordinator barrier skips writer_done semaphore.
    has_compute = any(rk[0] == "compute" for rk in role_keys)
    has_writer = any(rk[0] == "riscv_0" for rk in role_keys)

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
        # riscv_0 = BRISC = writer, riscv_1 = NCRISC = reader
        role_label = {"riscv_0": "writer", "riscv_1": "reader", "compute": "compute"}.get(risc_type)
        if role_label is None:
            continue

        is_coordinator = risc_type == "riscv_1"

        common_rt_offsets = _compute_common_rt_arg_offsets(phase_kernels, role_key)

        # All phase indices (for dispatcher — ensures barrier participation
        # even for phases where this role has no kernel).
        all_phase_indices = list(range(len(phases)))

        # Identify no-op phases (for asymmetric barrier — skip arrive)
        noop_phase_indices = frozenset(i for i, p in enumerate(phases) if p.op_descriptor is _NOOP_OP)

        fused_source = _generate_fused_source(
            phase_kernels,
            role_key,
            phases,
            ct_offsets,
            per_phase_cb_slots,
            risc_type=risc_type,
            role_label=role_label,
            rebind_info=rebind_info,
            op_semaphore_info=op_semaphore_info if is_coordinator else None,
            multi_barrier=multi_barrier,
            rt_arg_offsets=rt_offsets,
            common_rt_arg_offsets=common_rt_offsets,
            all_phase_indices=all_phase_indices,
            has_compute=has_compute,
            has_writer=has_writer,
            noop_phase_indices=noop_phase_indices,
        )

        # Determine barrier RT addresses per RISC type.
        # Coordinator (riscv_1/NCRISC) gets all semaphore addresses;
        # followers only get their own signal + reset_done + per-seg release.
        barrier_addrs: List[int] = []
        if multi_barrier is not None:
            if is_coordinator:
                barrier_addrs = []
                if has_compute:
                    barrier_addrs.append(multi_barrier.compute_done_addr)
                if has_writer:
                    barrier_addrs.append(multi_barrier.writer_done_addr)
                barrier_addrs.append(multi_barrier.reset_done_addr)
                for seg in multi_barrier.segments:
                    barrier_addrs.extend([seg.arrive_addr, seg.release_addr])
            elif risc_type == "riscv_0":
                # BRISC/writer follower: signals writer_done
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
        # Add per-segment named compile-time args (only coordinator needs them)
        if multi_barrier is not None and is_coordinator:
            for seg_idx, seg in enumerate(multi_barrier.segments):
                s = f"seg{seg_idx}"
                named_ct_args.append((f"{s}_num_release_cores", seg.config.num_release_cores))
                named_ct_args.append((f"{s}_num_arrive_cores", seg.config.num_arrive_cores))
                named_ct_args.append((f"{s}_core0_phys_x", seg.config.core0_phys_x))
                named_ct_args.append((f"{s}_core0_phys_y", seg.config.core0_phys_y))

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

    # Inject a synthetic coordinator kernel when no phase has a reader (riscv_1)
    # but a barrier is needed.  This ensures riscv_1 still gets a barrier-only
    # fused kernel (init + sync dispatch, no phase_N::run() calls).
    coordinator_generated = any(rk[0] == "riscv_1" for rk in role_keys)
    needs_barrier = multi_barrier is not None and len(multi_barrier.transition_map) > 0
    if needs_barrier and not coordinator_generated:
        # Derive core_ranges from the first available role key
        coord_core_ranges = target_core_range
        if coord_core_ranges is None:
            for pk in phase_kernels:
                for kernel in pk.values():
                    if kernel is not None:
                        coord_core_ranges = kernel.core_ranges
                        break
                if coord_core_ranges is not None:
                    break

        if coord_core_ranges is not None:
            coord_kernel = _build_coordinator_only_kernel(
                multi_barrier=multi_barrier,
                phases=phases,
                per_phase_cb_slots=per_phase_cb_slots,
                rebind_info=rebind_info,
                op_semaphore_info=op_semaphore_info,
                core_ranges=coord_core_ranges,
                has_compute=has_compute,
                has_writer=has_writer,
                noop_phase_indices=noop_phase_indices,
            )
            fused_kernels.append(coord_kernel)
            kernel_labels.append("")
            # Add empty phase map entry for the injected coordinator
            kernel_phase_map_extra = True
        else:
            kernel_phase_map_extra = False
    else:
        kernel_phase_map_extra = False

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
    if kernel_phase_map_extra:
        kernel_phase_map.append([])  # Injected coordinator has no source phase kernels

    return _BuildResult(
        descriptor=merged_descriptor,
        input_tensors=all_input_tensors,
        output_tensors=[output_tensor] if output_tensor else [],
        semaphores=sem_refs,
        kernel_labels=tuple(kernel_labels),
        kernel_phase_map=tuple(kernel_phase_map),
    )


def _build_coordinator_only_kernel(
    multi_barrier: MultiBarrierSpec,
    phases: List[PhaseInfo],
    per_phase_cb_slots: List[List[int]],
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    op_semaphore_info: List[Tuple[int, int]],
    core_ranges: Any,
    has_compute: bool,
    has_writer: bool,
    noop_phase_indices: frozenset = frozenset(),
) -> "ttnn.KernelDescriptor":
    """Build a barrier-only coordinator kernel for riscv_1.

    Called when no phase has a reader kernel but a barrier is needed.
    The generated kernel has no phase_N::run() calls — only barrier
    init, sync dispatch, and CB reset/rebind between phases.
    """
    all_phase_indices = list(range(len(phases)))
    phase_names = {i: phases[i].op_descriptor.name for i in range(len(phases))}

    source = _generate_coordinator_only_source(
        multi_barrier=multi_barrier,
        rebind_info=rebind_info,
        all_phase_indices=all_phase_indices,
        per_phase_cb_slots=per_phase_cb_slots,
        op_semaphore_info=op_semaphore_info,
        has_compute=has_compute,
        has_writer=has_writer,
        phase_names=phase_names,
        noop_phase_indices=noop_phase_indices,
    )

    # Barrier RT args (coordinator layout)
    barrier_addrs: List[int] = []
    if has_compute:
        barrier_addrs.append(multi_barrier.compute_done_addr)
    if has_writer:
        barrier_addrs.append(multi_barrier.writer_done_addr)
    barrier_addrs.append(multi_barrier.reset_done_addr)
    for seg in multi_barrier.segments:
        barrier_addrs.extend([seg.arrive_addr, seg.release_addr])

    # Build per-core RT args (barrier addrs only, uniform across all cores)
    coords = _get_core_coords_from_ranges(core_ranges)
    rt_args = [(coord, list(barrier_addrs)) for coord in coords]

    # Named CT args
    named_ct_args: List[Tuple[str, int]] = [("barrier_rt_offset", 0)]
    for seg_idx, seg in enumerate(multi_barrier.segments):
        s = f"seg{seg_idx}"
        named_ct_args.append((f"{s}_num_release_cores", seg.config.num_release_cores))
        named_ct_args.append((f"{s}_num_arrive_cores", seg.config.num_arrive_cores))
        named_ct_args.append((f"{s}_core0_phys_x", seg.config.core0_phys_x))
        named_ct_args.append((f"{s}_core0_phys_y", seg.config.core0_phys_y))

    # Rebind RT args
    rebind_rt_args, rebind_offset = _append_rebind_runtime_args(rt_args, rebind_info)
    if rebind_offset is not None:
        named_ct_args.append(("rebind_rt_offset", rebind_offset))

    desc = ttnn.KernelDescriptor()
    desc.kernel_source = source
    desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
    desc.core_ranges = core_ranges
    desc.compile_time_args = []
    desc.named_compile_time_args = named_ct_args
    desc.defines = []
    desc.runtime_args = rebind_rt_args
    desc.common_runtime_args = []
    desc.config = ttnn.ReaderConfigDescriptor()
    return desc


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
