# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Barrier C++ infrastructure generation for fused kernels.

Generates the ``namespace barrier { }`` block for each RISC type (BRISC,
NCRISC, compute). Uses per-RISC template strings and dict lookups to
produce identical output from a single unified generator.
"""

from typing import Any, Dict, List, Optional, Tuple

from models.experimental.ops.descriptors.fusion.common import MultiBarrierSpec

# =============================================================================
# Module Constants
# =============================================================================

_SECTION_SEP = "// " + "=" * 76
_BARRIER_RT_OFFSET_CT_ARG = "barrier_rt_offset"

# =============================================================================
# Per-RISC CB Reset/Resync C++ Templates
# =============================================================================

# BRISC: equalize stream registers + reset FIFO pointers
_RISCV0_RESET_CBS = """\
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

# NCRISC: simple FIFO pointer reset
_RISCV1_RESYNC_CBS = """\
// Resync NCRISC local CB pointers to CB start between phases.
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

# Compute: TRISC0 syncs tiles_acked, TRISC2 syncs tiles_received
_COMPUTE_RESYNC_CBS = """\
// Resync compute-side local CB state after BRISC reset.
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

_CB_RESET_TEMPLATES = {
    "riscv_0": _RISCV0_RESET_CBS,
    "riscv_1": _RISCV1_RESYNC_CBS,
    "compute": _COMPUTE_RESYNC_CBS,
}

# =============================================================================
# Per-RISC Phase State, Wait, and Init C++ Snippets
# =============================================================================

_PHASE_STATE = {
    "riscv_0": (
        "    uint32_t done;\n"
        "    volatile tt_l1_ptr uint32_t* compute_done;\n"
        "    volatile tt_l1_ptr uint32_t* writer_done;"
    ),
    "riscv_1": ("    uint32_t done;\n" "    volatile tt_l1_ptr uint32_t* writer_done;"),
    "compute": ("    uint32_t done;\n" "    volatile tt_l1_ptr uint32_t* compute_done;"),
}

_PHASE_WAIT = {
    "riscv_0": (
        "    FORCE_INLINE void wait() {\n"
        '        DeviceZoneScopedN("barrier-wait");\n'
        "        done++;\n"
        "        {\n"
        '            DeviceZoneScopedN("barrier-noc-drain");\n'
        "            noc_async_full_barrier();\n"
        "        }\n"
        "        {\n"
        '            DeviceZoneScopedN("barrier-sem-wait");\n'
        "            noc_semaphore_wait_min(compute_done, done);\n"
        "            noc_semaphore_wait_min(writer_done, done);\n"
        "        }\n"
        "    }"
    ),
    "riscv_1": (
        "    FORCE_INLINE void wait() {\n"
        '        DeviceZoneScopedN("barrier-wait");\n'
        "        done++;\n"
        "        {\n"
        '            DeviceZoneScopedN("barrier-noc-drain");\n'
        "            noc_async_write_barrier();\n"
        "        }\n"
        "        *writer_done = done;\n"
        "    }"
    ),
    "compute": (
        "    FORCE_INLINE void wait() {\n"
        '        DeviceZoneScopedN("barrier-wait");\n'
        "        done++;\n"
        "        *compute_done = done;\n"
        "    }"
    ),
}

_INIT_BODY = {
    "riscv_0": (
        "    phase::done = 0;\n"
        "    phase::compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "        get_arg_val<uint32_t>(rt_offset));\n"
        "    phase::writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "        get_arg_val<uint32_t>(rt_offset + 1));"
    ),
    "riscv_1": (
        "    phase::done = 0;\n"
        "    phase::writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "        get_arg_val<uint32_t>(rt_offset));"
    ),
    "compute": (
        "    phase::done = 0;\n"
        "    phase::compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(\n"
        "        get_arg_val<uint32_t>(rt_offset));"
    ),
}

# =============================================================================
# Segment C++ Templates (with .format() placeholders)
# =============================================================================

# Multicast segment template (BRISC only): leader gathers arrive, multicasts release.
_MULTICAST_SEGMENT_TEMPLATE = """\
namespace segment_{seg_idx} {{
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("{s}_num_cores");
    constexpr uint32_t core0_phys_x = get_named_compile_time_arg_val("{s}_core0_phys_x");
    constexpr uint32_t core0_phys_y = get_named_compile_time_arg_val("{s}_core0_phys_y");
    constexpr uint32_t mcast_start_x = get_named_compile_time_arg_val("{s}_mcast_start_x");
    constexpr uint32_t mcast_start_y = get_named_compile_time_arg_val("{s}_mcast_start_y");
    constexpr uint32_t mcast_end_x = get_named_compile_time_arg_val("{s}_mcast_end_x");
    constexpr uint32_t mcast_end_y = get_named_compile_time_arg_val("{s}_mcast_end_y");
    uint32_t call_count;
    volatile tt_l1_ptr uint32_t* arrive;
    volatile tt_l1_ptr uint32_t* release;

    FORCE_INLINE void init() {{
        call_count = 0;
        arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_arg_val<uint32_t>(rt_offset + {arrive_offset}));
        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_arg_val<uint32_t>(rt_offset + {release_offset}));
    }}

    FORCE_INLINE void sync() {{
        if constexpr (num_cores > 1) {{
            uint64_t core0_arrive_noc_addr = get_noc_addr(core0_phys_x, core0_phys_y, (uint32_t)arrive);
            noc_semaphore_inc(core0_arrive_noc_addr, 1);
            bool is_core_0 = (my_x[0] == core0_phys_x && my_y[0] == core0_phys_y);
            if (is_core_0) {{
                noc_semaphore_wait_min(arrive, num_cores * (call_count + 1));
                *release = call_count + 1;
                uint64_t mcast_addr = get_noc_multicast_addr(
                    mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, (uint32_t)release);
                noc_semaphore_set_multicast_loopback_src((uint32_t)release, mcast_addr, num_cores);
                noc_async_write_barrier();
            }} else {{
                noc_semaphore_wait_min(release, call_count + 1);
            }}
        }} else {{
            *release = call_count + 1;
        }}
        call_count++;
    }}
}} // namespace segment_{seg_idx}"""

# Spinwait segment template (NCRISC/compute): spin on release semaphore.
_SPINWAIT_SEGMENT_TEMPLATE = """\
namespace segment_{seg_idx} {{
    uint32_t call_count;
    volatile tt_l1_ptr uint32_t* release;

    FORCE_INLINE void init() {{
        call_count = 0;
        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
            get_arg_val<uint32_t>(rt_offset + {release_offset}));
    }}

    FORCE_INLINE void sync() {{
        while (*release < call_count + 1) {{ }}
        call_count++;
    }}
}} // namespace segment_{seg_idx}"""


# =============================================================================
# Barrier Generation Functions
# =============================================================================


def _build_barrier_dispatch(
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    """Build dispatch table for barrier transitions.

    Each entry maps a ``done`` counter value to the segment and rebinds
    needed for that transition.
    """
    dispatch: List[Dict[str, Any]] = []
    for idx in range(len(sources) - 1):
        phase_idx = sources[idx][0]
        next_phase_idx = sources[idx + 1][0]
        done_val = idx + 1
        if phase_idx in multi_barrier.transition_map:
            seg_idx, _ = multi_barrier.transition_map[phase_idx]
            dispatch.append(
                {
                    "done_val": done_val,
                    "seg_idx": seg_idx,
                    "next_phase_idx": next_phase_idx,
                    "rebinds": rebind_info.get(next_phase_idx, []),
                }
            )
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
            }
        )
    return dispatch


def _generate_rebind_lines(
    rebinds: List[Tuple[int, int, int]],
    next_phase_idx: int,
    indent: str,
    for_compute: bool = False,
) -> List[str]:
    """Generate C++ rebind code for use inside barrier::phase::reset()."""
    shift = " >> 4" if for_compute else ""
    lines: List[str] = []
    for slot_idx, _, _ in rebinds:
        prefix = f"phase_{next_phase_idx}_cb{slot_idx}"
        lines.append(f"{indent}{{")
        lines.append(
            f'{indent}    constexpr uint32_t new_addr = get_named_compile_time_arg_val("{prefix}_rebind_addr"){shift};'
        )
        lines.append(
            f'{indent}    constexpr uint32_t new_size = get_named_compile_time_arg_val("{prefix}_rebind_size"){shift};'
        )
        lines.append(f"{indent}    get_local_cb_interface({slot_idx}).fifo_rd_ptr = new_addr;")
        lines.append(f"{indent}    get_local_cb_interface({slot_idx}).fifo_wr_ptr = new_addr;")
        lines.append(f"{indent}    get_local_cb_interface({slot_idx}).fifo_size = new_size;")
        lines.append(f"{indent}    get_local_cb_interface({slot_idx}).fifo_limit = new_addr + new_size;")
        lines.append(f"{indent}}}")
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


def _emit_coordinator_reset(
    dispatch: List[Dict[str, Any]],
    op_semaphore_info: List[Tuple[int, int]],
) -> List[str]:
    """Emit ``phase::reset()`` body for BRISC (coordinator).

    Order: op semaphore reset -> reset_cbs -> rebind -> segment sync.
    """
    lines: List[str] = []
    lines.append("    FORCE_INLINE void reset() {")
    if op_semaphore_info:
        lines.append("        // Reset op semaphores")
        for sem_id, initial_value in op_semaphore_info:
            lines.append(
                f"        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = {initial_value};"
            )
    for entry in dispatch:
        done_val = entry["done_val"]
        seg_idx = entry["seg_idx"]
        rebinds = entry["rebinds"]
        next_phase_idx = entry["next_phase_idx"]
        completed_phase_idx = done_val - 1
        lines.append(f"        if (done == {done_val}) {{")
        lines.append(f"            {{")
        lines.append(f'                DeviceZoneScopedN("barrier-cb-reset");')
        lines.append(f"                reset_cbs(phase_{completed_phase_idx}_cbs);")
        lines.append(f"            }}")
        if rebinds and next_phase_idx is not None:
            lines.extend(_generate_rebind_lines(rebinds, next_phase_idx, "            "))
        lines.append(f"            {{")
        lines.append(f'                DeviceZoneScopedN("barrier-segment-sync");')
        lines.append(f"                segment_{seg_idx}::sync();")
        lines.append(f"            }}")
        lines.append(f"        }}")
    lines.append("    }")
    return lines


def _emit_follower_reset(
    dispatch: List[Dict[str, Any]],
    for_compute: bool = False,
) -> List[str]:
    """Emit ``phase::reset()`` body for NCRISC/compute (followers).

    Order: segment sync -> resync_cbs -> rebind.
    For compute, rebind is guarded by ``#ifndef TRISC_MATH`` and uses ``>> 4``.
    """
    lines: List[str] = []
    lines.append("    FORCE_INLINE void reset() {")
    for entry in dispatch:
        done_val = entry["done_val"]
        seg_idx = entry["seg_idx"]
        rebinds = entry["rebinds"]
        next_phase_idx = entry["next_phase_idx"]
        completed_phase_idx = done_val - 1
        lines.append(f"        if (done == {done_val}) {{")
        lines.append(f"            {{")
        lines.append(f'                DeviceZoneScopedN("barrier-segment-sync");')
        lines.append(f"                segment_{seg_idx}::sync();")
        lines.append(f"            }}")
        lines.append(f"            {{")
        lines.append(f'                DeviceZoneScopedN("barrier-cb-resync");')
        lines.append(f"                resync_cbs(phase_{completed_phase_idx}_cbs);")
        lines.append(f"            }}")
        if rebinds and next_phase_idx is not None:
            if for_compute:
                lines.append("#ifndef TRISC_MATH")
            lines.extend(_generate_rebind_lines(rebinds, next_phase_idx, "            ", for_compute=for_compute))
            if for_compute:
                lines.append("#endif")
        lines.append(f"        }}")
    lines.append("    }")
    return lines


def _generate_barrier_namespace(
    risc_type: str,
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
    per_phase_cb_slots: List[List[int]],
    op_semaphore_info: Optional[List[Tuple[int, int]]] = None,
) -> List[str]:
    """Generate ``namespace barrier { }`` for any RISC type.

    Produces identical output to the former per-RISC generators. Structure:
    preamble -> CB reset/resync function -> phase CB arrays -> segment
    namespaces -> phase namespace (state + wait + reset) -> init -> close.
    """
    is_coordinator = risc_type == "riscv_0"
    num_segments = len(multi_barrier.segments)
    dispatch = _build_barrier_dispatch(multi_barrier, rebind_info, sources)

    lines: List[str] = []

    # Preamble
    lines.extend([_SECTION_SEP, "// Barrier infrastructure", _SECTION_SEP])
    lines.append("namespace barrier {")
    lines.append("")
    lines.append(f'constexpr uint32_t rt_offset = get_named_compile_time_arg_val("{_BARRIER_RT_OFFSET_CT_ARG}");')
    lines.append("")

    # CB reset/resync function + per-phase CB index arrays
    lines.extend(_CB_RESET_TEMPLATES[risc_type].split("\n"))
    lines.append("")
    lines.extend(_emit_phase_cb_arrays(per_phase_cb_slots, len(dispatch)))

    # Segment namespaces
    if is_coordinator:
        for seg_idx in range(num_segments):
            lines.extend(
                _MULTICAST_SEGMENT_TEMPLATE.format(
                    seg_idx=seg_idx,
                    s=f"seg{seg_idx}",
                    arrive_offset=2 + seg_idx * 2,
                    release_offset=3 + seg_idx * 2,
                ).split("\n")
            )
            lines.append("")
    else:
        for seg_idx in range(num_segments):
            lines.extend(
                _SPINWAIT_SEGMENT_TEMPLATE.format(
                    seg_idx=seg_idx,
                    release_offset=1 + seg_idx,
                ).split("\n")
            )
            lines.append("")

    # Phase namespace: state variables + wait() + reset()
    lines.append("namespace phase {")
    lines.extend(_PHASE_STATE[risc_type].split("\n"))
    lines.append("")
    lines.extend(_PHASE_WAIT[risc_type].split("\n"))
    lines.append("")
    if is_coordinator:
        lines.extend(_emit_coordinator_reset(dispatch, op_semaphore_info or []))
    else:
        lines.extend(_emit_follower_reset(dispatch, for_compute=(risc_type == "compute")))
    lines.append("} // namespace phase")
    lines.append("")

    # init()
    lines.append("FORCE_INLINE void init() {")
    lines.extend(_INIT_BODY[risc_type].split("\n"))
    for seg_idx in range(num_segments):
        lines.append(f"    segment_{seg_idx}::init();")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace barrier")
    lines.append("")
    return lines
