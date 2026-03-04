# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Barrier C++ infrastructure generation for fused kernels.

Generates the ``namespace barrier { }`` block for each RISC type (BRISC,
NCRISC, compute).  Structure::

    namespace barrier {
        // State variables (done counter, semaphore pointers)
        namespace local  { void sync(); }   // per-core phase sync
        namespace group  { void sync(); }   // cross-core segment sync
        void sync() { local::sync(); group::sync(); }
        void init();
    }

The barrier coordinator is the reader RISC (NCRISC / riscv_1).  NCRISC
is chosen because ``ReaderDataMovementConfig`` maps to ``RISCV_1`` in the
hardware (``kernel_types.cpp``).  When no phase has a reader kernel, a
synthetic coordinator-only kernel is injected by the builder.

When ``has_compute`` is False, the coordinator barrier skips the
``compute_done`` semaphore wait.  When ``has_writer`` is False, the
coordinator skips the ``writer_done`` semaphore wait.

Coordinator ``local::sync()``: drain NOC, wait followers, reset op sems +
CBs, rebind, signal ``*reset_done = done``.

Follower ``local::sync()``: signal done, wait ``*reset_done >= done``.

Coordinator ``group::sync()``: per-transition segment unicast release.

Follower ``group::sync()``: per-transition segment spinwait + CB resync +
rebind.
"""

from typing import Any, Dict, List, Optional, Tuple

from models.experimental.ops.descriptors.fusion.common import MultiBarrierSpec

# =============================================================================
# Module Constants
# =============================================================================

_SECTION_SEP = "// " + "=" * 76
_BARRIER_RT_OFFSET_CT_ARG = "barrier_rt_offset"
_REBIND_RT_OFFSET_CT_ARG = "rebind_rt_offset"

# Short suffix per RISC type for unique Tracy zone names.  Each RISC compiles
# its own kernel_includes.hpp at different line offsets.  Without distinct zone
# names the profiler's 16-bit FNV hash can collide (only 65536 buckets).
# Tracy profiler hashes "name,__FILE__,__LINE__" into 16-bit keys (65536
# buckets).  Each fused kernel variant has DeviceZoneScopedN at different line
# numbers, and with 300+ total zones the birthday paradox guarantees collisions.
# Fix: emit ``#line N "fused_zones"`` before each DeviceZoneScopedN to pin
# __FILE__/__LINE__ to stable values so hashes are deterministic.
_ZONE_LINE = {"local-sync": 100, "group-sync": 200}


def _pinned_zone(zone_name: str, indent: str = "        ") -> list:
    """Emit DeviceZoneScopedN with stable __FILE__/__LINE__ via #line."""
    return [
        f'#line {_ZONE_LINE[zone_name]} "fused_zones"',
        f'{indent}DeviceZoneScopedN("{zone_name}");',
    ]


# =============================================================================
# Per-Role CB Reset/Resync C++ Templates
# =============================================================================

# Coordinator: equalize stream registers + reset FIFO pointers.
# Stream registers are shared hardware — any RISC can write to them.
# The coordinator equalizes once; followers resync their local state after.
_COORDINATOR_RESET_CBS = """\
// Equalize stream tile counters + reset FIFO pointers for the given CB indices.
// Skips write when counters already match (writing same value to a stream
// register may have hardware side-effects).
template <size_t N>
__attribute__((noinline)) void reset_cbs(const std::array<uint32_t, N>& cbs) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t cb = cbs[i];
        uint16_t received = (uint16_t)(*get_cb_tiles_received_ptr(cb));
        uint16_t acked = (uint16_t)(*get_cb_tiles_acked_ptr(cb));
        if (received != acked) {
            volatile tt_reg_ptr uint32_t* acked_ptr = (volatile tt_reg_ptr uint32_t*)
                ((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr(cb));
            *acked_ptr = (uint32_t)received;
        }
        uint32_t fifo_start = get_local_cb_interface(cb).fifo_limit
                            - get_local_cb_interface(cb).fifo_size;
        get_local_cb_interface(cb).fifo_rd_ptr = fifo_start;
        get_local_cb_interface(cb).fifo_wr_ptr = fifo_start;
    }
}"""

# Data-movement follower (BRISC or NCRISC): simple FIFO pointer reset.
_DM_FOLLOWER_RESYNC_CBS = """\
// Resync data-movement follower local CB pointers to CB start between phases.
template <size_t N>
__attribute__((noinline)) void resync_cbs(const std::array<uint32_t, N>& cbs) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t cb = cbs[i];
        uint32_t fifo_start = get_local_cb_interface(cb).fifo_limit
                            - get_local_cb_interface(cb).fifo_size;
        get_local_cb_interface(cb).fifo_rd_ptr = fifo_start;
        get_local_cb_interface(cb).fifo_wr_ptr = fifo_start;
    }
}"""

# Compute follower: TRISC0 syncs tiles_acked, TRISC2 syncs tiles_received.
_COMPUTE_RESYNC_CBS = """\
// Resync compute-side local CB state after coordinator reset.
// TRISC0: sync tiles_acked + reset fifo_rd_ptr to CB start.
// TRISC2: sync tiles_received + reset fifo_wr_ptr to CB start.
template <size_t N>
__attribute__((noinline)) void resync_cbs(const std::array<uint32_t, N>& cbs) {
#ifdef TRISC_UNPACK
    for (uint32_t i = 0; i < N; i++) {
        uint32_t cb = cbs[i];
        uint16_t stream_acked = (uint16_t)reg_read((uint32_t)get_cb_tiles_acked_ptr(cb));
        get_local_cb_interface(cb).tiles_acked = stream_acked;
        uint32_t fifo_start = get_local_cb_interface(cb).fifo_limit
                            - get_local_cb_interface(cb).fifo_size;
        get_local_cb_interface(cb).fifo_rd_ptr = fifo_start;
    }
#endif
#ifdef TRISC_PACK
    for (uint32_t i = 0; i < N; i++) {
        uint32_t cb = cbs[i];
        uint16_t stream_received = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr(cb));
        get_local_cb_interface(cb).tiles_received = stream_received;
        uint32_t fifo_start = get_local_cb_interface(cb).fifo_limit
                            - get_local_cb_interface(cb).fifo_size;
        get_local_cb_interface(cb).fifo_wr_ptr = fifo_start;
        get_local_cb_interface(cb).fifo_wr_tile_ptr = 0;
    }
#endif
}"""

# Rebind CB buffer addresses from runtime args.  Uses cb_addr_shift from
# circular_buffer_interface.h (4 on TRISC, 0 on BRISC/NCRISC) to convert
# byte addresses to the per-RISC CB interface units.
_REBIND_CBS = """\
template <size_t N>
__attribute__((noinline)) void rebind_cbs(
    const std::array<uint32_t, N>& slots, uint32_t rt_start) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t slot = slots[i];
        uint32_t addr = get_arg_val<uint32_t>(rebind_rt_offset + rt_start + i * 2);
        uint32_t size = get_arg_val<uint32_t>(rebind_rt_offset + rt_start + i * 2 + 1);
        get_local_cb_interface(slot).fifo_rd_ptr = addr >> cb_addr_shift;
        get_local_cb_interface(slot).fifo_wr_ptr = addr >> cb_addr_shift;
        get_local_cb_interface(slot).fifo_size = size >> cb_addr_shift;
        get_local_cb_interface(slot).fifo_limit = (addr + size) >> cb_addr_shift;
    }
}"""

# =============================================================================
# Segment C++ Templates (with .format() placeholders)
# =============================================================================


def _generate_unicast_segment(
    seg_idx: int,
    s: str,
    arrive_offset: int,
    release_offset: int,
    other_phys_coords: "List[Tuple[int, int]]",
) -> str:
    """Generate coordinator barrier segment code with unicast release.

    Instead of multicasting release to a rectangular bounding box, this
    writes to each non-coordinator core individually via ``noc_async_write``.
    This removes the requirement for a rectangular core grid (~300 ns for
    16 cores vs ~30 ns for multicast — negligible).

    ``sync()`` is templated on ``SyncMode``:
    - ``SyncMode::Full``: arrive + wait + release (normal barrier).
    - ``SyncMode::WaitOnly``: skip arrive, only spinwait on release.
      Used by no-op cores that have no real work to signal.
    """
    n_other = len(other_phys_coords)

    # Build constexpr array of other cores' physical coordinates.
    # Always declare the array (with dummy values when empty) because
    # `if constexpr` does NOT discard branches in non-template context —
    # the compiler still checks name lookup in the discarded branch.
    if n_other > 0:
        coords_flat = []
        for x, y in other_phys_coords:
            coords_flat.extend([str(x), str(y)])
        array_init = ", ".join(coords_flat)
        array_size = n_other * 2
    else:
        array_init = "0, 0"
        array_size = 2
    array_decl = f"    constexpr std::array<uint32_t, {array_size}> other_core_phys = {{{{{array_init}}}}};"

    # Unicast release loop: write local release value to each other core
    release_loop = (
        "            *release = call_count + 1;\n"
        "            for (uint32_t i = 0; i < {n_other} * 2; i += 2) {{\n"
        "                uint64_t dest = get_noc_addr(other_core_phys[i], other_core_phys[i+1], (uint32_t)release);\n"
        "                noc_async_write((uint32_t)release, dest, 4);\n"
        "            }}\n"
        "            noc_async_write_barrier();"
    ).format(n_other=n_other)

    return (
        f"namespace seg_{seg_idx} {{\n"
        f'    constexpr uint32_t num_release_cores = get_named_compile_time_arg_val("{s}_num_release_cores");\n'
        f'    constexpr uint32_t num_arrive_cores = get_named_compile_time_arg_val("{s}_num_arrive_cores");\n'
        f'    constexpr uint32_t core0_phys_x = get_named_compile_time_arg_val("{s}_core0_phys_x");\n'
        f'    constexpr uint32_t core0_phys_y = get_named_compile_time_arg_val("{s}_core0_phys_y");\n'
        f"{array_decl}\n"
        f"    uint32_t call_count;\n"
        f"    volatile tt_l1_ptr uint32_t* arrive;\n"
        f"    volatile tt_l1_ptr uint32_t* release;\n"
        f"\n"
        f"    __attribute__((noinline)) void init() {{\n"
        f"        call_count = 0;\n"
        f"        arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        f"            get_arg_val<uint32_t>(rt_offset + {arrive_offset}));\n"
        f"        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        f"            get_arg_val<uint32_t>(rt_offset + {release_offset}));\n"
        f"        // Reset L1 semaphores for re-execution\n"
        f"        *arrive = 0;\n"
        f"        *release = 0;\n"
        f"    }}\n"
        f"\n"
        f"    template <SyncMode mode>\n"
        f"    __attribute__((noinline)) void sync() {{\n"
        f"        if constexpr (num_release_cores > 1) {{\n"
        f"            if constexpr (mode != SyncMode::WaitOnly) {{\n"
        f"                uint64_t core0_arrive_noc_addr = get_noc_addr(core0_phys_x, core0_phys_y, (uint32_t)arrive);\n"
        f"                noc_semaphore_inc(core0_arrive_noc_addr, 1);\n"
        f"            }}\n"
        f"            bool is_core_0 = (my_x[0] == core0_phys_x && my_y[0] == core0_phys_y);\n"
        f"            if (is_core_0 && mode != SyncMode::WaitOnly) {{\n"
        f"                noc_semaphore_wait_min(arrive, num_arrive_cores * (call_count + 1));\n"
        f"{release_loop}\n"
        f"            }} else {{\n"
        f"                noc_semaphore_wait_min(release, call_count + 1);\n"
        f"            }}\n"
        f"        }} else if constexpr (mode != SyncMode::WaitOnly) {{\n"
        f"            *release = call_count + 1;\n"
        f"        }}\n"
        f"        call_count++;\n"
        f"    }}\n"
        f"}} // namespace seg_{seg_idx}"
    )


# Spinwait segment template (BRISC/compute): spin on release semaphore.
# Templated on SyncMode for uniform codegen — followers always spinwait,
# so ``WaitOnly`` and ``Full`` behave identically for them.
_SPINWAIT_SEGMENT_TEMPLATE = """\
namespace seg_{seg_idx} {{
    uint32_t call_count;
    volatile tt_l1_ptr uint32_t* release;

    __attribute__((noinline)) void init() {{
        call_count = 0;
        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_arg_val<uint32_t>(rt_offset + {release_offset}));
        // Reset L1 semaphore for re-execution
        *release = 0;
    }}

    template <SyncMode mode>
    __attribute__((noinline)) void sync() {{
        while (*release < call_count + 1) {{ }}
        call_count++;
    }}
}} // namespace seg_{seg_idx}"""


# =============================================================================
# Barrier Generation Utilities
# =============================================================================


def _build_barrier_dispatch(
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
    noop_phase_indices: frozenset = frozenset(),
) -> List[Dict[str, Any]]:
    """Build dispatch table for barrier transitions.

    Each entry maps a ``done`` counter value to the segment and rebinds
    needed for that transition.

    ``is_arrive`` is True when the completed phase had real work (not a
    no-op), meaning the core should participate in the arrive step.
    When the completed phase is a no-op, ``is_arrive`` is False and the
    barrier uses ``SyncMode::WaitOnly`` (skip arrive, only spinwait).
    """
    dispatch: List[Dict[str, Any]] = []
    cumulative_rebind_offset = 0
    for idx in range(len(sources) - 1):
        phase_idx = sources[idx][0]
        next_phase_idx = sources[idx + 1][0]
        done_val = idx + 1
        if phase_idx in multi_barrier.transition_map:
            seg_idx, _ = multi_barrier.transition_map[phase_idx]
            rebinds = rebind_info.get(next_phase_idx, [])
            dispatch.append(
                {
                    "done_val": done_val,
                    "seg_idx": seg_idx,
                    "next_phase_idx": next_phase_idx,
                    "rebinds": rebinds,
                    "rebind_entry_offset": cumulative_rebind_offset,
                    "is_arrive": phase_idx not in noop_phase_indices,
                }
            )
            cumulative_rebind_offset += len(rebinds) * 2
    # Trailing barrier (after last phase, e.g. for parent sync in OpGraph)
    last_phase_idx = sources[-1][0]
    if last_phase_idx in multi_barrier.transition_map:
        seg_idx, _ = multi_barrier.transition_map[last_phase_idx]
        dispatch.append(
            {
                "done_val": len(sources),
                "seg_idx": seg_idx,
                "next_phase_idx": None,
                "rebinds": [],
                "rebind_entry_offset": cumulative_rebind_offset,
                "is_arrive": last_phase_idx not in noop_phase_indices,
            }
        )
    return dispatch


def _generate_rebind_call(done_val: int, rebind_entry_offset: int, indent: str) -> str:
    """Generate a ``rebind_cbs(rebind_slots_K, offset)`` call."""
    return f"{indent}rebind_cbs(rebind_slots_{done_val}, {rebind_entry_offset});"


def _emit_rebind_slot_arrays(dispatch: List[Dict[str, Any]]) -> List[str]:
    """Emit ``constexpr std::array<uint32_t, N> rebind_slots_K = {...};`` per transition."""
    lines: List[str] = []
    for entry in dispatch:
        rebinds = entry["rebinds"]
        if not rebinds:
            continue
        done_val = entry["done_val"]
        slots = [str(slot_idx) for slot_idx, _, _ in rebinds]
        n = len(slots)
        slot_str = ", ".join(slots)
        lines.append(f"constexpr std::array<uint32_t, {n}> rebind_slots_{done_val} = {{{{{slot_str}}}}};")
    if lines:
        lines.append("")
    return lines


def _emit_phase_cb_arrays(
    per_phase_cb_slots: List[List[int]],
    num_transitions: int,
) -> List[str]:
    """Emit ``constexpr std::array<uint32_t, N> phase_K_cbs = {...};`` for each transition."""
    lines: List[str] = []
    for phase_idx, phase_slots in enumerate(per_phase_cb_slots):
        if phase_idx >= num_transitions:
            break
        slots = sorted(phase_slots)
        n = len(slots)
        slot_str = ", ".join(str(s) for s in slots)
        lines.append(f"constexpr std::array<uint32_t, {n}> phase_{phase_idx}_cbs = {{{{{slot_str}}}}};")
    lines.append("")
    return lines


# =============================================================================
# State Variables
# =============================================================================


def _emit_state_vars(risc_type: str, is_coordinator: bool, has_compute: bool, has_writer: bool = True) -> List[str]:
    """Emit state variables at barrier namespace level."""
    lines = ["uint32_t done;"]
    if is_coordinator:
        if has_compute:
            lines.append("volatile tt_l1_ptr uint32_t* compute_done;")
        if has_writer:
            lines.append("volatile tt_l1_ptr uint32_t* writer_done;")
    elif risc_type == "riscv_0":
        if has_writer:
            lines.append("volatile tt_l1_ptr uint32_t* writer_done;")
    elif risc_type == "compute":
        lines.append("volatile tt_l1_ptr uint32_t* compute_done;")
    lines.append("volatile tt_l1_ptr uint32_t* reset_done;")
    lines.append("")
    return lines


# =============================================================================
# local::sync()
# =============================================================================


def _emit_local_sync_coordinator(
    has_compute: bool,
    dispatch: List[Dict[str, Any]],
    op_semaphore_info: List[Tuple[int, int]],
    has_writer: bool = True,
) -> List[str]:
    """Emit ``namespace local { void sync(); }`` for the coordinator (NCRISC).

    Order: done++ -> drain NOC -> wait followers -> reset op sems ->
    per-done CB reset + rebind -> signal *reset_done = done.
    """
    lines = ["namespace local {"]
    lines.append("    __attribute__((noinline)) void sync() {")
    lines.extend(_pinned_zone("local-sync"))
    lines.append("        done++;")
    lines.append("        noc_async_full_barrier();")
    if has_compute:
        lines.append("        noc_semaphore_wait_min(compute_done, done);")
    if has_writer:
        lines.append("        noc_semaphore_wait_min(writer_done, done);")
    if op_semaphore_info:
        lines.append("        // Reset op semaphores")
        for sem_id, initial_value in op_semaphore_info:
            lines.append(
                f"        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = {initial_value};"
            )
    for entry in dispatch:
        done_val = entry["done_val"]
        rebinds = entry["rebinds"]
        next_phase_idx = entry["next_phase_idx"]
        completed_phase_idx = done_val - 1
        lines.append(f"        if (done == {done_val}) {{")
        lines.append(f"            reset_cbs(phase_{completed_phase_idx}_cbs);")
        if rebinds and next_phase_idx is not None:
            lines.append(_generate_rebind_call(done_val, entry["rebind_entry_offset"], "            "))
        lines.append("        }")
    lines.append("        // Signal followers that reset is complete on this core.")
    lines.append("        *reset_done = done;")
    lines.append("    }")
    lines.append("} // namespace local")
    return lines


def _emit_local_sync_follower(risc_type: str) -> List[str]:
    """Emit ``namespace local { void sync(); }`` for BRISC or compute.

    Order: done++ -> drain NOC (DM only) -> signal done -> wait reset_done.
    """
    lines = ["namespace local {"]
    lines.append("    __attribute__((noinline)) void sync() {")
    lines.extend(_pinned_zone("local-sync"))
    lines.append("        done++;")
    if risc_type != "compute":
        # DM follower: drain outstanding NOC writes before signaling done.
        # For a writer this flushes real writes; for a reader with no
        # pending writes this is a harmless no-op.
        lines.append("        noc_async_write_barrier();")
        lines.append("        *writer_done = done;")
    else:  # compute
        lines.append("        *compute_done = done;")
    lines.append("        // Wait for coordinator to complete reset (op sem + CB)")
    lines.append("        // before proceeding.  Without this, a fast core can start")
    lines.append("        // the next phase's multicast while a slow core's coordinator")
    lines.append("        // hasn't reset op semaphores yet, clobbering in-flight")
    lines.append("        // noc_semaphore_inc.")
    lines.append("        while (*reset_done < done) {}")
    lines.append("    }")
    lines.append("} // namespace local")
    return lines


# =============================================================================
# group::sync()
# =============================================================================


def _emit_group_sync_coordinator(dispatch: List[Dict[str, Any]]) -> List[str]:
    """Emit ``group::sync()`` function for the coordinator (NCRISC)."""
    lines = []
    lines.append("    __attribute__((noinline)) void sync() {")
    lines.extend(_pinned_zone("group-sync"))
    for entry in dispatch:
        done_val = entry["done_val"]
        seg_idx = entry["seg_idx"]
        is_arrive = entry.get("is_arrive", True)
        mode = "SyncMode::Full" if is_arrive else "SyncMode::WaitOnly"
        lines.append(f"        if (done == {done_val}) {{")
        lines.append(f"            seg_{seg_idx}::sync<{mode}>();")
        lines.append("        }")
    lines.append("    }")
    return lines


def _emit_group_sync_follower(
    dispatch: List[Dict[str, Any]],
    for_compute: bool,
    risc_type: str,
) -> List[str]:
    """Emit ``group::sync()`` function for BRISC or compute.

    Per-transition: segment spinwait -> resync_cbs -> rebind.
    Uses ``SyncMode`` template for uniform codegen with coordinator.
    """
    lines = []
    lines.append("    __attribute__((noinline)) void sync() {")
    lines.extend(_pinned_zone("group-sync"))
    for entry in dispatch:
        done_val = entry["done_val"]
        seg_idx = entry["seg_idx"]
        rebinds = entry["rebinds"]
        next_phase_idx = entry["next_phase_idx"]
        completed_phase_idx = done_val - 1
        is_arrive = entry.get("is_arrive", True)
        mode = "SyncMode::Full" if is_arrive else "SyncMode::WaitOnly"
        lines.append(f"        if (done == {done_val}) {{")
        lines.append(f"            seg_{seg_idx}::sync<{mode}>();")
        lines.append(f"            resync_cbs(phase_{completed_phase_idx}_cbs);")
        if rebinds and next_phase_idx is not None:
            if for_compute:
                lines.append("#ifndef TRISC_MATH")
            lines.append(_generate_rebind_call(done_val, entry["rebind_entry_offset"], "            "))
            if for_compute:
                lines.append("#endif")
        lines.append("        }")
    lines.append("    }")
    return lines


# =============================================================================
# init()
# =============================================================================


def _emit_init_coordinator(
    has_compute: bool,
    num_segments: int,
    op_semaphore_info: Optional[List[Tuple[int, int]]] = None,
    has_writer: bool = True,
) -> List[str]:
    """Emit ``init()`` for the coordinator (NCRISC)."""
    lines = []
    lines.append("__attribute__((noinline)) void init() {")
    lines.append("    done = 0;")
    offset = 0
    if has_compute:
        lines.append("    compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append(f"        get_arg_val<uint32_t>(rt_offset + {offset}));")
        offset += 1
    if has_writer:
        lines.append("    writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append(f"        get_arg_val<uint32_t>(rt_offset + {offset}));")
        offset += 1
    lines.append("    reset_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
    lines.append(f"        get_arg_val<uint32_t>(rt_offset + {offset}));")
    lines.append("    *reset_done = 0;")
    lines.append("    // NOTE: Do NOT reset *compute_done or *writer_done here.")
    lines.append("    // Each follower RISC resets its own semaphore in its own init().")
    lines.append("    // Resetting here races with fast compute/BRISC signaling")
    lines.append("    // (e.g. no-op phase 0 where compute signals immediately).")
    if op_semaphore_info:
        lines.append("    // Reset op semaphores so phase 0 doesn't see stale values")
        lines.append("    // from the previous execution's last phase.")
        for sem_id, initial_value in op_semaphore_info:
            lines.append(
                f"    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = {initial_value};"
            )
    for seg_idx in range(num_segments):
        lines.append(f"    group::seg_{seg_idx}::init();")
    lines.append("}")
    return lines


def _emit_init_follower(
    risc_type: str,
    num_segments: int,
) -> List[str]:
    """Emit ``init()`` for the DM follower (BRISC) or compute."""
    lines = []
    lines.append("__attribute__((noinline)) void init() {")
    lines.append("    done = 0;")
    if risc_type != "compute":
        # DM follower: signals writer_done, reads reset_done
        lines.append("    writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append("        get_arg_val<uint32_t>(rt_offset));")
        lines.append("    *writer_done = 0;")
        lines.append("    reset_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append("        get_arg_val<uint32_t>(rt_offset + 1));")
    else:  # compute
        lines.append("    compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append("        get_arg_val<uint32_t>(rt_offset));")
        lines.append("    *compute_done = 0;")
        lines.append("    reset_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append("        get_arg_val<uint32_t>(rt_offset + 1));")
    for seg_idx in range(num_segments):
        lines.append(f"    group::seg_{seg_idx}::init();")
    lines.append("}")
    return lines


# =============================================================================
# Main Generation Function
# =============================================================================


def _generate_barrier_namespace(
    risc_type: str,
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
    per_phase_cb_slots: List[List[int]],
    op_semaphore_info: Optional[List[Tuple[int, int]]] = None,
    has_compute: bool = True,
    has_writer: bool = True,
    noop_phase_indices: frozenset = frozenset(),
) -> List[str]:
    """Generate ``namespace barrier { }`` for any RISC type.

    Structure: preamble -> SyncMode enum -> CB reset/resync function ->
    phase CB arrays -> state variables -> local::sync() ->
    group { segments + sync() } -> top-level sync() -> init() -> close.

    The coordinator is ``riscv_1`` (NCRISC / reader).  When
    ``has_compute`` is False, the coordinator barrier skips the
    ``compute_done`` semaphore wait (no compute kernel to signal it).
    When ``has_writer`` is False, the coordinator skips the
    ``writer_done`` semaphore wait (no writer kernel to signal it).

    ``noop_phase_indices`` identifies phases that are no-ops for this
    group.  Transitions after a no-op phase use ``SyncMode::WaitOnly``
    (skip arrive, only spinwait on release).
    """
    is_coordinator = risc_type == "riscv_1"
    num_segments = len(multi_barrier.segments)
    dispatch = _build_barrier_dispatch(multi_barrier, rebind_info, sources, noop_phase_indices)

    has_rebinds = any(rebind_info.get(k) for k in rebind_info)

    lines: List[str] = []

    # Preamble
    lines.extend([_SECTION_SEP, "// Barrier infrastructure", _SECTION_SEP])
    lines.append("namespace barrier {")
    lines.append("")
    lines.append(f'constexpr uint32_t rt_offset = get_named_compile_time_arg_val("{_BARRIER_RT_OFFSET_CT_ARG}");')
    if has_rebinds:
        lines.append(
            f'constexpr uint32_t rebind_rt_offset = get_named_compile_time_arg_val("{_REBIND_RT_OFFSET_CT_ARG}");'
        )
    lines.append("")

    # SyncMode enum — used by segment sync() templates
    lines.append("enum class SyncMode { Full, WaitOnly };")
    lines.append("")

    # CB reset/resync function + per-phase CB index arrays.
    # The coordinator equalizes shared stream registers; followers only
    # resync their local CB interface pointers.
    if is_coordinator:
        cb_template = _COORDINATOR_RESET_CBS
    elif risc_type == "compute":
        cb_template = _COMPUTE_RESYNC_CBS
    else:
        cb_template = _DM_FOLLOWER_RESYNC_CBS
    lines.extend(cb_template.split("\n"))
    lines.append("")
    if has_rebinds:
        lines.extend(_REBIND_CBS.split("\n"))
        lines.append("")
    lines.extend(_emit_phase_cb_arrays(per_phase_cb_slots, len(dispatch)))
    if has_rebinds:
        lines.extend(_emit_rebind_slot_arrays(dispatch))

    # State variables at barrier namespace level
    lines.extend(_emit_state_vars(risc_type, is_coordinator, has_compute, has_writer))

    # namespace local { void sync(); }
    if is_coordinator:
        lines.extend(
            _emit_local_sync_coordinator(has_compute, dispatch, op_semaphore_info or [], has_writer=has_writer)
        )
    else:
        lines.extend(_emit_local_sync_follower(risc_type))
    lines.append("")

    # namespace group { seg namespaces + void sync(); }
    lines.append("namespace group {")
    lines.append("")
    # Barrier RT args = [compute_done? writer_done? reset_done, seg0_arrive, seg0_release, ...]
    num_phase_sems = int(has_compute) + int(has_writer) + 1  # [compute_done?] + [writer_done?] + reset_done
    seg_base_offset = num_phase_sems
    if is_coordinator:
        for seg_idx in range(num_segments):
            seg_config = multi_barrier.segments[seg_idx].config
            lines.extend(
                _generate_unicast_segment(
                    seg_idx=seg_idx,
                    s=f"seg{seg_idx}",
                    arrive_offset=seg_base_offset + seg_idx * 2,
                    release_offset=seg_base_offset + 1 + seg_idx * 2,
                    other_phys_coords=seg_config.other_core_phys_coords,
                ).split("\n")
            )
            lines.append("")
    else:
        for seg_idx in range(num_segments):
            lines.extend(
                _SPINWAIT_SEGMENT_TEMPLATE.format(
                    seg_idx=seg_idx,
                    release_offset=2 + seg_idx,  # +2: done_signal + reset_done
                ).split("\n")
            )
            lines.append("")

    # group::sync()
    if is_coordinator:
        lines.extend(_emit_group_sync_coordinator(dispatch))
    else:
        lines.extend(_emit_group_sync_follower(dispatch, for_compute=(risc_type == "compute"), risc_type=risc_type))
    lines.append("} // namespace group")
    lines.append("")

    # Top-level sync()
    lines.append("__attribute__((noinline)) void sync() {")
    lines.append("    local::sync();")
    lines.append("    group::sync();")
    lines.append("}")
    lines.append("")

    # init()
    if is_coordinator:
        lines.extend(_emit_init_coordinator(has_compute, num_segments, op_semaphore_info, has_writer=has_writer))
    else:
        lines.extend(_emit_init_follower(risc_type, num_segments))
    lines.append("")

    lines.append("} // namespace barrier")
    lines.append("")
    return lines
