# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Sequential Kernel Chaining Infrastructure

Fuses multiple operations to run sequentially on the SAME cores within a
single program.  All readers/writers run for every phase.  Data flows through
DRAM between phases (Writer_N -> DRAM -> Reader_{N+1}).

No CB remapping — each phase uses its native CB indices (0-31).
CB descriptors are merged as max(total_size) per index across phases.

Two-level barrier synchronization between phases:
  - Local barrier (per core): L1 flags allocated via GlobalSemaphore.
    Compute/writer signal done, reader waits then resets CBs.
  - Global barrier (across cores): Reader uses noc_semaphore_inc/wait
    on GlobalSemaphore L1 words, then sets global_release which also
    serves as the phase release signal for compute/writer.

Usage:
    >>> builder = SequentialChainBuilder()
    >>> builder.add_phase(op0_desc)
    >>> builder.add_phase(op1_desc)
    >>> fused = builder.build(device)
    >>> outputs = composite.launch([fused])
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Set, Any
import re
import os

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CBInfo:
    """Information about a circular buffer extracted from a CBDescriptor."""

    original_index: int
    total_size: int
    data_format: Any  # tt::DataFormat
    page_size: int
    core_ranges: Any  # CoreRangeSet


@dataclass
class PhaseInfo:
    """Information about a phase (op) in the sequential chain."""

    phase_idx: int
    op_descriptor: OpDescriptor
    cb_info: Dict[int, CBInfo] = field(default_factory=dict)


@dataclass
class BarrierConfig:
    """Configuration for the two-level barrier between phases.

    Holds GlobalSemaphore references (to prevent GC) and their L1 addresses,
    plus physical core coordinates for the global barrier.
    """

    # L1 addresses of per-core flags (from GlobalSemaphore.address())
    compute_done_addr: int = 0
    writer_done_addr: int = 0
    global_arrive_addr: int = 0
    global_release_addr: int = 0

    # Physical core coordinates for global barrier
    num_cores: int = 1
    core0_phys_x: int = 0
    core0_phys_y: int = 0
    mcast_start_x: int = 0
    mcast_start_y: int = 0
    mcast_end_x: int = 0
    mcast_end_y: int = 0

    # GlobalSemaphore references (prevent GC)
    _sem_refs: List[Any] = field(default_factory=list)


# =============================================================================
# Analysis Functions
# =============================================================================


def extract_cb_info(descriptor: "ttnn.ProgramDescriptor") -> Dict[int, CBInfo]:
    """Extract CB information from a ProgramDescriptor.

    Returns a dict mapping CB index -> CBInfo.
    """
    cb_info = {}
    for cb_desc in descriptor.cbs:
        for fmt_desc in cb_desc.format_descriptors:
            cb_idx = fmt_desc.buffer_index
            try:
                data_format = fmt_desc.data_format
            except (TypeError, AttributeError):
                data_format = None
            cb_info[cb_idx] = CBInfo(
                original_index=cb_idx,
                total_size=cb_desc.total_size,
                data_format=data_format,
                page_size=fmt_desc.page_size,
                core_ranges=cb_desc.core_ranges,
            )
    return cb_info


def extract_cb_names_from_kernel(kernel_desc: "ttnn.KernelDescriptor") -> Dict[str, int]:
    """Extract CB name -> index mapping from kernel's named compile-time args."""
    cb_names = {}
    if hasattr(kernel_desc, "named_compile_time_args"):
        for name, value in kernel_desc.named_compile_time_args:
            if name.startswith("cb_"):
                cb_names[name] = value
    return cb_names


# =============================================================================
# Kernel Classification
# =============================================================================


def _classify_kernel(kernel_desc: "ttnn.KernelDescriptor") -> str:
    """Classify a kernel as reader, writer, or compute based on its config."""
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return "compute"
    elif isinstance(config, ttnn.ReaderConfigDescriptor):
        return "reader"
    elif isinstance(config, ttnn.WriterConfigDescriptor):
        return "writer"
    elif isinstance(config, ttnn.DataMovementConfigDescriptor):
        if config.processor == ttnn.DataMovementProcessor.RISCV_1:
            return "reader"
        else:
            return "writer"
    return "unknown"


# =============================================================================
# Source Code Utilities
# =============================================================================


def _read_kernel_source(kernel_desc: "ttnn.KernelDescriptor") -> Tuple[str, Optional[str]]:
    """Read kernel source code from a kernel descriptor.

    Returns (source_code, kernel_file_dir) where kernel_file_dir is the
    directory of the source file (for resolving local includes), or None
    if the kernel is inline SOURCE_CODE.
    """
    if kernel_desc.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
        return kernel_desc.kernel_source, None

    base_paths = [
        os.environ.get("TT_METAL_HOME", ""),
        "",
    ]
    for base in base_paths:
        full_path = os.path.join(base, kernel_desc.kernel_source) if base else kernel_desc.kernel_source
        if os.path.exists(full_path):
            with open(full_path, "r") as f:
                return f.read(), os.path.dirname(full_path)
    return "", None


def _inline_local_includes(source: str, kernel_dir: Optional[str]) -> str:
    """Inline local includes (same-directory headers) into source.

    For generated SOURCE_CODE kernels, local includes (no path separator)
    won't resolve because the compiler doesn't know the original directory.
    We inline them directly into the source.
    """
    if kernel_dir is None:
        return source

    lines = source.split("\n")
    result = []
    inlined = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include "'):
            match = re.match(r'#include\s+"([^"]+)"', stripped)
            if match:
                inc_path = match.group(1)
                # Local include = no path separators
                if "/" not in inc_path and inc_path not in inlined:
                    full_inc = os.path.join(kernel_dir, inc_path)
                    if os.path.exists(full_inc):
                        with open(full_inc, "r") as f:
                            inc_content = f.read()
                        # Remove include guards from inlined content
                        inc_lines = inc_content.split("\n")
                        for il in inc_lines:
                            ils = il.strip()
                            if ils.startswith("#pragma once"):
                                continue
                            # Skip the same local include if it re-includes itself
                            if ils.startswith('#include "') and "/" not in ils:
                                continue
                            result.append(il)
                        inlined.add(inc_path)
                        continue
        result.append(line)

    return "\n".join(result)


# Defines that are resolved per-phase in source code (not passed to compiler)
_SOURCE_LEVEL_DEFINES = {"RMSNORM", "FUSE_PRE_ADD", "FUSED_PRE_ADD", "FUSE_GAMMA", "FUSE_BETA"}


def _resolve_ifdef_directives(source: str, active_defines: set) -> str:
    """Resolve preprocessor #ifdef/#ifndef/#if defined directives in source code.

    Only resolves directives involving known source-level defines (RMSNORM,
    FUSE_PRE_ADD, FUSED_PRE_ADD). Other directives are left untouched.
    """
    lines = source.split("\n")
    result = []
    stack: List[Tuple[bool, bool]] = []

    for line in lines:
        stripped = line.strip()
        directive_handled = False

        if stripped.startswith("#if defined"):
            names_found = re.findall(r"\bdefined\s+(\w+)", stripped)
            involved = [n for n in names_found if n in _SOURCE_LEVEL_DEFINES]
            if involved:
                result_val = _eval_ifdef_expression(stripped, active_defines)
                stack.append((result_val, True))
                directive_handled = True
            else:
                stack.append((True, False))

        elif stripped.startswith("#ifdef "):
            name = stripped[7:].strip()
            if name in _SOURCE_LEVEL_DEFINES:
                stack.append((name in active_defines, True))
                directive_handled = True
            else:
                stack.append((True, False))

        elif stripped.startswith("#ifndef "):
            name = stripped[8:].strip()
            if name in _SOURCE_LEVEL_DEFINES:
                stack.append((name not in active_defines, True))
                directive_handled = True
            else:
                stack.append((True, False))

        elif stripped == "#else":
            if stack and stack[-1][1]:
                incl, known = stack[-1]
                stack[-1] = (not incl, known)
                directive_handled = True

        elif stripped == "#endif":
            if stack:
                _, known = stack[-1]
                stack.pop()
                if known:
                    directive_handled = True

        if directive_handled:
            continue

        include = True
        for incl, known in stack:
            if known and not incl:
                include = False
                break

        if include:
            result.append(line)

    return "\n".join(result)


def _eval_ifdef_branch(branch: str, active_defines: set) -> bool:
    """Evaluate a single AND-branch of a #if defined expression."""
    clauses = re.findall(r"(not\s+)?defined\s+(\w+)", branch)
    if not clauses:
        return True
    result = True
    for negation, name in clauses:
        is_defined = name in active_defines
        if negation:
            result = result and (not is_defined)
        else:
            result = result and is_defined
    return result


def _eval_ifdef_expression(directive: str, active_defines: set) -> bool:
    """Evaluate a #if defined expression with || and && support.

    Splits on || first (lower precedence), then evaluates each branch
    with AND semantics. E.g. ``#if defined FUSE_GAMMA || defined FUSE_BETA``
    returns True if either is defined.
    """
    or_branches = re.split(r"\|\|", directive)
    for branch in or_branches:
        if _eval_ifdef_branch(branch.strip(), active_defines):
            return True
    return False


def _extract_kernel_body_for_fusion(source: str) -> str:
    """Extract the body of kernel_main() for fusion."""
    lines = source.split("\n")
    in_kernel_main = False
    brace_depth = 0
    body_lines = []

    for line in lines:
        stripped = line.strip()

        if "void kernel_main()" in line or "KERNEL_MAIN" in line:
            in_kernel_main = True
            brace_depth += stripped.count("{") - stripped.count("}")
            continue

        if in_kernel_main:
            open_braces = stripped.count("{")
            close_braces = stripped.count("}")
            new_brace_depth = brace_depth + open_braces - close_braces

            if new_brace_depth > 0 or (brace_depth > 0 and new_brace_depth > 0):
                body_lines.append(line)
            elif brace_depth > 0 and new_brace_depth == 0:
                if stripped != "}":
                    pass
                break

            brace_depth = new_brace_depth
            if brace_depth == 0:
                break

    return "\n".join(body_lines)


def _collect_includes(sources: List[str]) -> List[str]:
    """Collect unique #include lines from multiple source strings."""
    includes = set()
    for source in sources:
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#include"):
                includes.add(stripped)
    return sorted(includes)


def _collect_defines(sources: List[str]) -> List[str]:
    """Collect unique #define lines from multiple source strings (before kernel_main)."""
    defines = []
    seen = set()
    for source in sources:
        for line in source.split("\n"):
            stripped = line.strip()
            if "void kernel_main()" in line or "KERNEL_MAIN" in line:
                break
            if stripped.startswith("#define") and stripped not in seen:
                defines.append(line)
                seen.add(stripped)
    return defines


def _collect_pre_main_code(source: str) -> str:
    """Extract code before kernel_main (namespaces, macros, helpers)."""
    lines = source.split("\n")
    pre_main = []
    for line in lines:
        if "void kernel_main()" in line or "KERNEL_MAIN" in line:
            break
        stripped = line.strip()
        if (
            stripped
            and not stripped.startswith("#include")
            and not stripped.startswith("#define")
            and not stripped.startswith("//")
        ):
            pre_main.append(line)
    return "\n".join(pre_main)


# =============================================================================
# Source Transformations for Phase N>0
# =============================================================================


def _prefix_named_args_in_source(source: str, phase_idx: int) -> str:
    """Replace get_named_compile_time_arg_val("X") with phase-prefixed version."""
    if phase_idx == 0:
        return source

    def replace_named_arg(match):
        name = match.group(1)
        return f'get_named_compile_time_arg_val("phase{phase_idx}_{name}")'

    return re.sub(
        r'get_named_compile_time_arg_val\("([^"]+)"\)',
        replace_named_arg,
        source,
    )


def _offset_compile_time_args_in_source(source: str, phase_idx: int, ct_arg_offset: int) -> str:
    """Offset get_compile_time_arg_val(N) and TensorAccessorArgs<N> for phase N>0.

    Instead of substituting literal values, we offset the indices so that
    each phase reads from its own slice of the concatenated compile-time arg
    array.  This also handles TensorAccessorArgs<N> which internally calls
    get_compile_time_arg_val(N).
    """
    if phase_idx == 0 or ct_arg_offset == 0:
        return source

    def offset_ct_arg(match):
        arg_idx = int(match.group(1))
        return f"get_compile_time_arg_val({arg_idx + ct_arg_offset})"

    source = re.sub(
        r"get_compile_time_arg_val\((\d+)\)",
        offset_ct_arg,
        source,
    )

    def offset_tensor_accessor(match):
        arg_idx = int(match.group(1))
        return f"TensorAccessorArgs<{arg_idx + ct_arg_offset}>"

    source = re.sub(
        r"TensorAccessorArgs<(\d+)>",
        offset_tensor_accessor,
        source,
    )

    return source


def _offset_runtime_args_in_source(source: str, phase_idx: int) -> str:
    """Replace get_arg_val<T>(N) with offset version for phase N>0."""
    if phase_idx == 0:
        return source

    offset_name = f"__phase{phase_idx}_rt_offset"
    offset_decl = (
        f'    constexpr uint32_t {offset_name} = get_named_compile_time_arg_val("phase{phase_idx}_rt_arg_offset");\n'
    )

    def replace_rt_arg(match):
        type_name = match.group(1)
        arg_idx = match.group(2)
        return f"get_arg_val<{type_name}>({offset_name} + {arg_idx})"

    source = re.sub(
        r"get_arg_val<(\w+)>\((\d+)\)",
        replace_rt_arg,
        source,
    )

    # Handle incrementing variable pattern: uint32_t rt_args_idx = 0;
    source = re.sub(
        r"(uint32_t\s+\w*args?\w*\s*=\s*)0(\s*;)",
        rf"\g<1>{offset_name}\2",
        source,
    )

    return offset_decl + source


def _transform_phase_source(source: str, phase_idx: int, ct_arg_offset: int = 0) -> str:
    """Apply all transformations for a phase's kernel body."""
    source = _prefix_named_args_in_source(source, phase_idx)
    source = _offset_compile_time_args_in_source(source, phase_idx, ct_arg_offset)
    source = _offset_runtime_args_in_source(source, phase_idx)
    return source


# =============================================================================
# CB Descriptor Merging
# =============================================================================

NUM_CIRCULAR_BUFFERS = 32


def _merge_cb_descriptors(phases: List[PhaseInfo]) -> list:
    """Merge CB descriptors from all phases.

    For each CB index used by any phase, keeps the descriptor with the
    largest total_size so the CB can accommodate any phase's data.
    """
    cb_by_index: Dict[int, Any] = {}  # cb_index -> (largest_total_size, cb_desc)

    for phase in phases:
        desc = phase.op_descriptor.descriptor
        for cb_desc in desc.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                cb_idx = fmt_desc.buffer_index
                if cb_idx not in cb_by_index or cb_desc.total_size > cb_by_index[cb_idx][0]:
                    cb_by_index[cb_idx] = (cb_desc.total_size, cb_desc)

    return [cb_desc for _, (_, cb_desc) in sorted(cb_by_index.items())]


# =============================================================================
# Fused Kernel Source Generation
# =============================================================================


def _get_all_cb_descriptor_indices(phases: List[PhaseInfo]) -> Set[int]:
    """Get the union of all CB indices that have CBDescriptors across all phases."""
    indices: Set[int] = set()
    for phase in phases:
        for cb_desc in phase.op_descriptor.descriptor.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                indices.add(fmt_desc.buffer_index)
    return indices


def _generate_fused_reader_source(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    sweep_cb_indices: List[int],
    barrier_config: Optional[BarrierConfig] = None,
) -> Optional[str]:
    """Generate fused reader kernel source with two-level barrier sync.

    Between phases, the reader (dataflow RISC) acts as the barrier coordinator:
      1. Wait for local compute + writer to signal done (L1 flag spin)
      2. Reset residual tiles from ALL CBs on BRISC
      3. Global barrier across cores (sets global_release which also serves
         as the phase release signal for compute/writer)

    The BRISC reset updates stream register tiles_acked but NOT TRISC0's
    local copy.  Compute must resync after being released (see compute source).
    """
    reader_sources = []

    for i, pk in enumerate(phase_kernels):
        if pk["reader"] is None:
            continue
        source, kernel_dir = _read_kernel_source(pk["reader"])
        if not source:
            continue
        source = _inline_local_includes(source, kernel_dir)
        phase_defs = {name for name, _ in pk["reader"].defines} if hasattr(pk["reader"], "defines") else set()
        resolved = _resolve_ifdef_directives(source, phase_defs)
        reader_sources.append((i, resolved))

    if not reader_sources:
        return None

    all_sources = [s for _, s in reader_sources]
    includes = _collect_includes(all_sources)
    defines = _collect_defines(all_sources)
    pre_main = _collect_pre_main_code(all_sources[0])

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused reader kernel - {len(reader_sources)} phases",
        "",
    ]
    lines.extend(defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if pre_main.strip():
        lines.append(pre_main)
        lines.append("")

    is_multi_phase = len(reader_sources) > 1

    if is_multi_phase:
        # Barrier named compile-time args
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append('constexpr uint32_t __num_barrier_cores = get_named_compile_time_arg_val("num_barrier_cores");')
        lines.append("")

        # Global barrier compile-time args (only used if num_cores > 1)
        lines.append('constexpr uint32_t __core0_phys_x = get_named_compile_time_arg_val("core0_phys_x");')
        lines.append('constexpr uint32_t __core0_phys_y = get_named_compile_time_arg_val("core0_phys_y");')
        lines.append('constexpr uint32_t __mcast_start_x = get_named_compile_time_arg_val("mcast_start_x");')
        lines.append('constexpr uint32_t __mcast_start_y = get_named_compile_time_arg_val("mcast_start_y");')
        lines.append('constexpr uint32_t __mcast_end_x = get_named_compile_time_arg_val("mcast_end_x");')
        lines.append('constexpr uint32_t __mcast_end_y = get_named_compile_time_arg_val("mcast_end_y");')
        lines.append("")

        # BRISC-side CB reset: pop residual tiles between phases.
        # All CBs are reset (generic — any CB id can be input or output).
        # Safe because all RISCs have finished (local barrier complete).
        lines.append("// BRISC-side CB reset: pop residual tiles between phases.")
        lines.append("FORCE_INLINE void __cb_reset_to_empty() {")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(
                f"        uint16_t remaining = (uint16_t)(*get_cb_tiles_received_ptr({cb_idx})) - (uint16_t)(*get_cb_tiles_acked_ptr({cb_idx}));"
            )
            lines.append(f"        if (remaining > 0) {{")
            lines.append(f"            cb_pop_front({cb_idx}, remaining);")
            lines.append(f"        }}")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")

        # Global barrier helper (also serves as phase release for compute/writer)
        lines.append("// Global barrier across cores. Sets global_release which compute/writer spin on.")
        lines.append(
            "FORCE_INLINE void __global_barrier(uint32_t phase, volatile tt_l1_ptr uint32_t* global_arrive, volatile tt_l1_ptr uint32_t* global_release) {"
        )
        lines.append("    if constexpr (__num_barrier_cores > 1) {")
        lines.append("        // Arrive: all cores send atomic inc to core 0's arrive semaphore")
        lines.append(
            "        uint64_t core0_arrive_noc_addr = get_noc_addr(__core0_phys_x, __core0_phys_y, (uint32_t)global_arrive);"
        )
        lines.append("        noc_semaphore_inc(core0_arrive_noc_addr, 1);")
        lines.append("")
        lines.append("        bool is_core_0 = (my_x[0] == __core0_phys_x && my_y[0] == __core0_phys_y);")
        lines.append("        if (is_core_0) {")
        lines.append("            // Core 0: wait for all cores to arrive")
        lines.append("            noc_semaphore_wait_min(global_arrive, __num_barrier_cores * (phase + 1));")
        lines.append("            // Multicast release to all cores (including self via loopback)")
        lines.append("            *global_release = phase + 1;")
        lines.append(
            "            uint64_t mcast_addr = get_noc_multicast_addr(__mcast_start_x, __mcast_start_y, __mcast_end_x, __mcast_end_y, (uint32_t)global_release);"
        )
        lines.append(
            "            noc_semaphore_set_multicast_loopback_src((uint32_t)global_release, mcast_addr, __num_barrier_cores);"
        )
        lines.append("            noc_async_write_barrier();")
        lines.append("        } else {")
        lines.append("            // Other cores: wait for release from core 0")
        lines.append("            noc_semaphore_wait_min(global_release, phase + 1);")
        lines.append("        }")
        lines.append("    } else {")
        lines.append("        // Single core: set release directly (no NOC ops needed)")
        lines.append("        *global_release = phase + 1;")
        lines.append("    }")
        lines.append("}")
        lines.append("")

    # Generate phase functions
    for phase_idx, resolved_source in reader_sources:
        body = _extract_kernel_body_for_fusion(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        transformed = _transform_phase_source(body, phase_idx, offset)

        lines.append(f"// Phase {phase_idx} reader")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_reader() {{")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        lines.append("}")
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    if is_multi_phase:
        # Read barrier flag addresses from runtime args
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __compute_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append("    const uint32_t __writer_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 1);")
        lines.append("    const uint32_t __global_arrive_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 2);")
        lines.append("    const uint32_t __global_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 3);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__compute_done_addr);"
        )
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__writer_done_addr);"
        )
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __global_arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_arrive_addr);"
        )
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __global_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_release_addr);"
        )
        lines.append("")

    first = True
    for phase_idx, _ in reader_sources:
        if not first and is_multi_phase:
            lines.append("")
            lines.append(f"    // === Barrier: Phase {phase_idx - 1} -> Phase {phase_idx} ===")
            lines.append("    noc_async_full_barrier();")
            lines.append("")
            lines.append(f"    // Wait for local compute + writer to finish Phase {phase_idx - 1}")
            lines.append(f"    noc_semaphore_wait_min(__compute_done, {phase_idx});")
            lines.append(f"    noc_semaphore_wait_min(__writer_done, {phase_idx});")
            lines.append("")
            lines.append("    // Reset residual tiles from ALL CBs")
            lines.append("    __cb_reset_to_empty();")
            lines.append("")
            lines.append("    // Global barrier (sets global_release, releasing compute/writer)")
            lines.append(f"    __global_barrier({phase_idx - 1}, __global_arrive, __global_release);")
            lines.append("")
        lines.append(f"    phase{phase_idx}_reader();")
        first = False
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_writer_source(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    barrier_config: Optional[BarrierConfig] = None,
) -> Optional[str]:
    """Generate fused writer kernel source with L1 flag barrier sync.

    Between phases, the writer:
      1. Signals done by writing phase+1 to writer_done L1 flag
      2. Spins on global_release L1 flag (plain volatile read, no NOC APIs)
    """
    writer_sources = []

    for i, pk in enumerate(phase_kernels):
        if pk["writer"] is None:
            continue
        source, kernel_dir = _read_kernel_source(pk["writer"])
        if not source:
            continue
        source = _inline_local_includes(source, kernel_dir)
        phase_defs = {name for name, _ in pk["writer"].defines} if hasattr(pk["writer"], "defines") else set()
        resolved = _resolve_ifdef_directives(source, phase_defs)
        writer_sources.append((i, resolved))

    if not writer_sources:
        return None

    all_sources = [s for _, s in writer_sources]
    includes = _collect_includes(all_sources)
    defines = _collect_defines(all_sources)
    pre_main = _collect_pre_main_code(all_sources[0])

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused writer kernel - {len(writer_sources)} phases",
        "",
    ]
    lines.extend(defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if pre_main.strip():
        lines.append(pre_main)
        lines.append("")

    is_multi_phase = len(writer_sources) > 1

    if is_multi_phase:
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append("")

    # Generate phase functions
    for phase_idx, resolved_source in writer_sources:
        body = _extract_kernel_body_for_fusion(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        transformed = _transform_phase_source(body, phase_idx, offset)

        lines.append(f"// Phase {phase_idx} writer")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_writer() {{")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        lines.append("}")
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    if is_multi_phase:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __writer_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append("    const uint32_t __global_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 1);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__writer_done_addr);"
        )
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __global_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_release_addr);"
        )
        lines.append("")

    num_writers = len(writer_sources)
    for count, (phase_idx, _) in enumerate(writer_sources):
        lines.append(f"    phase{phase_idx}_writer();")
        if count < num_writers - 1 and is_multi_phase:
            lines.append("")
            lines.append(f"    // Signal done for Phase {phase_idx}")
            lines.append(f"    *__writer_done = {phase_idx + 1};")
            lines.append("")
            lines.append(f"    // Wait for global release (Phase {phase_idx + 1})")
            lines.append(f"    while (*__global_release < {phase_idx + 1}) {{ }}")
            lines.append("")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_compute_source(
    phase_kernels: List[Dict[str, Any]],
    phases: List[PhaseInfo],
    ct_arg_offsets: Optional[Dict[int, int]] = None,
    sweep_cb_indices: Optional[List[int]] = None,
    barrier_config: Optional[BarrierConfig] = None,
) -> Optional[str]:
    """Generate fused compute kernel with L1 flag barrier sync.

    Between phases, compute:
      1. Signals done by writing phase+1 to compute_done L1 flag
      2. Spins on global_release L1 flag (plain volatile read, no NOC APIs)
      3. Resyncs TRISC0 local CB state with stream registers

    Step 3 is critical: BRISC reset (in reader) updates the hardware stream
    register tiles_acked but NOT TRISC0's local copy.  Without resync,
    compute sees stale tiles_acked and reads garbage.
    """
    if ct_arg_offsets is None:
        ct_arg_offsets = {}

    compute_sources = []

    for i, pk in enumerate(phase_kernels):
        if pk["compute"] is None:
            continue
        source, kernel_dir = _read_kernel_source(pk["compute"])
        if not source:
            continue
        source = _inline_local_includes(source, kernel_dir)
        phase_defs = {name for name, _ in pk["compute"].defines}
        resolved = _resolve_ifdef_directives(source, phase_defs)
        compute_sources.append((i, resolved))

    if not compute_sources:
        return None

    all_sources = [s for _, s in compute_sources]
    includes = _collect_includes(all_sources)
    defines = _collect_defines(all_sources)
    pre_main = _collect_pre_main_code(all_sources[0])

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused compute kernel - {len(compute_sources)} phases",
        "",
    ]
    lines.extend(defines)
    lines.append("")
    lines.extend(includes)
    lines.append("")
    if pre_main.strip():
        lines.append(pre_main)
        lines.append("")

    is_multi_phase = len(compute_sources) > 1

    if is_multi_phase:
        lines.append('constexpr uint32_t __barrier_rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
        lines.append("")

    # Generate compute-side CB state resync function.
    # After BRISC sweep, TRISC0's local tiles_acked and fifo_rd_ptr are stale.
    # This function reads the stream register tiles_acked (updated by BRISC)
    # and directly updates the local CB interface to match.
    # Guarded by TRISC_UNPACK because cb_interface only exists on TRISC0.
    if sweep_cb_indices and is_multi_phase:
        lines.append("// Resync TRISC0 local CB state with stream registers after BRISC sweep.")
        lines.append("FORCE_INLINE void __resync_cb_state_after_sweep() {")
        lines.append("#ifdef TRISC_UNPACK")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(f"        volatile tt_l1_ptr uint32_t* acked_ptr = get_cb_tiles_acked_ptr({cb_idx});")
            lines.append(f"        uint16_t stream_acked = (uint16_t)reg_read((uint32_t)acked_ptr);")
            lines.append(f"        uint16_t local_acked = get_local_cb_interface({cb_idx}).tiles_acked;")
            lines.append(f"        uint16_t swept = stream_acked - local_acked;")
            lines.append(f"        if (swept > 0) {{")
            lines.append(f"            get_local_cb_interface({cb_idx}).tiles_acked = stream_acked;")
            lines.append(f"            uint32_t advance = swept * get_local_cb_interface({cb_idx}).fifo_page_size;")
            lines.append(f"            get_local_cb_interface({cb_idx}).fifo_rd_ptr += advance;")
            lines.append(
                f"            if (get_local_cb_interface({cb_idx}).fifo_rd_ptr >= get_local_cb_interface({cb_idx}).fifo_limit) {{"
            )
            lines.append(
                f"                get_local_cb_interface({cb_idx}).fifo_rd_ptr -= get_local_cb_interface({cb_idx}).fifo_size;"
            )
            lines.append(f"            }}")
            lines.append(f"        }}")
            lines.append(f"    }}")
        lines.append("#endif")
        lines.append("}")
        lines.append("")

    # Generate phase functions
    for phase_idx, resolved_source in compute_sources:
        body = _extract_kernel_body_for_fusion(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        transformed = _transform_phase_source(body, phase_idx, offset)

        lines.append(f"// Phase {phase_idx} compute")
        lines.append(f"FORCE_INLINE void phase{phase_idx}_compute() {{")
        for line in transformed.split("\n"):
            lines.append(f"    {line}")
        lines.append("}")
        lines.append("")

    # Generate kernel_main
    lines.append("void kernel_main() {")

    if is_multi_phase:
        lines.append("    // Read barrier L1 flag addresses from runtime args")
        lines.append("    const uint32_t __compute_done_addr = get_arg_val<uint32_t>(__barrier_rt_offset);")
        lines.append("    const uint32_t __global_release_addr = get_arg_val<uint32_t>(__barrier_rt_offset + 1);")
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__compute_done_addr);"
        )
        lines.append(
            "    volatile tt_l1_ptr uint32_t* __global_release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(__global_release_addr);"
        )
        lines.append("")

    first = True
    for count, (phase_idx, _) in enumerate(compute_sources):
        if not first and is_multi_phase:
            lines.append("")
            lines.append(f"    // Signal done for Phase {phase_idx - 1}")
            lines.append(f"    *__compute_done = {phase_idx};")
            lines.append("")
            lines.append(f"    // Wait for global release (Phase {phase_idx})")
            lines.append(f"    while (*__global_release < {phase_idx}) {{ }}")
            lines.append("")
            if sweep_cb_indices:
                lines.append("    // Resync TRISC0 local CB state (tiles_acked, fifo_rd_ptr)")
                lines.append("    __resync_cb_state_after_sweep();")
                lines.append("")
        lines.append(f"    phase{phase_idx}_compute();")
        first = False
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Runtime Arg Handling
# =============================================================================


def _compute_runtime_arg_offsets(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Dict[int, int]:
    """Compute per-phase runtime arg offsets.

    Returns {phase_idx: offset} where offset is the cumulative count of
    runtime args from all prior phases (max across cores).

    RuntimeArgsView API: runtime_args[col_idx] -> RuntimeArgsColProxy,
    runtime_args[col_idx][0] -> VectorUInt32 of args for that core.
    """
    offsets: Dict[int, int] = {}
    cumulative = 0

    for i, pk in enumerate(phase_kernels):
        offsets[i] = cumulative
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        # Count runtime args for this phase (max across cores)
        max_args = 0
        try:
            for col_idx in range(len(kernel.runtime_args)):
                col = kernel.runtime_args[col_idx]
                args = col[0]  # VectorUInt32 of args for this core
                max_args = max(max_args, len(args))
        except Exception:
            pass

        cumulative += max_args

    return offsets


def _get_core_coords_from_ranges(core_ranges: Any) -> List[Any]:
    """Extract ordered list of CoreCoords from a CoreRangeSet."""
    coords = []
    for cr in core_ranges.ranges():
        for y in range(cr.start.y, cr.end.y + 1):
            for x in range(cr.start.x, cr.end.x + 1):
                coords.append(ttnn.CoreCoord(x, y))
    return coords


def _concatenate_runtime_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> List[Tuple[Any, List[int]]]:
    """Concatenate per-core runtime args from all phases.

    Returns list of (CoreCoord, concatenated_args) pairs.

    RuntimeArgsView API: runtime_args[col_idx] -> RuntimeArgsColProxy,
    runtime_args[col_idx][0] -> VectorUInt32 of args for that core.
    Column order matches the core order in core_ranges.
    """
    # Find core_ranges from first available kernel
    core_coords = None
    for pk in phase_kernels:
        kernel = pk.get(kernel_type)
        if kernel is not None:
            core_coords = _get_core_coords_from_ranges(kernel.core_ranges)
            break
    if not core_coords:
        return []

    num_cols = len(core_coords)
    col_args: List[List[int]] = [[] for _ in range(num_cols)]

    for pk in phase_kernels:
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue
        try:
            for col_idx in range(min(len(kernel.runtime_args), num_cols)):
                col = kernel.runtime_args[col_idx]
                args = col[0]  # VectorUInt32 of args for this core
                col_args[col_idx].extend(list(args))
        except Exception:
            pass

    return [(core_coords[i], col_args[i]) for i in range(num_cols) if col_args[i]]


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
        except Exception:
            pass
    return common_args


# =============================================================================
# Named Compile-Time Arg Merging
# =============================================================================


def _merge_named_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
    barrier_rt_offset: Optional[int] = None,
    barrier_config: Optional[BarrierConfig] = None,
) -> List[Tuple[str, int]]:
    """Merge named compile-time args from all phases with phase prefixes.

    Phase 0 keeps original names. Phase N>0 gets "phaseN_" prefix.
    Runtime arg offsets and barrier config are added as named args.
    """
    merged = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        for name, value in kernel.named_compile_time_args:
            if i == 0:
                merged.append((name, value))
            else:
                merged.append((f"phase{i}_{name}", value))

        # Add runtime arg offset for phase 1+
        if i > 0 and rt_arg_offsets is not None and i in rt_arg_offsets:
            merged.append((f"phase{i}_rt_arg_offset", rt_arg_offsets[i]))

    # Add barrier named args
    if barrier_rt_offset is not None:
        merged.append(("barrier_rt_offset", barrier_rt_offset))
    if barrier_config is not None:
        merged.append(("num_barrier_cores", barrier_config.num_cores))
        merged.append(("core0_phys_x", barrier_config.core0_phys_x))
        merged.append(("core0_phys_y", barrier_config.core0_phys_y))
        merged.append(("mcast_start_x", barrier_config.mcast_start_x))
        merged.append(("mcast_start_y", barrier_config.mcast_start_y))
        merged.append(("mcast_end_x", barrier_config.mcast_end_x))
        merged.append(("mcast_end_y", barrier_config.mcast_end_y))

    return merged


def _merge_compile_time_args(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> Tuple[List[int], Dict[int, int]]:
    """Concatenate all phases' compile-time args and return (merged_args, offsets).

    Phase 0's args go first, then phase 1's, etc.  The offsets dict maps
    phase_idx -> starting index in the merged array so that phase N's source
    can reference get_compile_time_arg_val(original_idx + offset).
    """
    merged: List[int] = []
    offsets: Dict[int, int] = {}

    for i, pk in enumerate(phase_kernels):
        offsets[i] = len(merged)
        kernel = pk.get(kernel_type)
        if kernel is not None:
            merged.extend(list(kernel.compile_time_args))

    return merged, offsets


def _merge_defines(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
) -> List[Tuple[str, str]]:
    """Merge defines from all phases' kernels.

    Source-level defines (RMSNORM etc.) are resolved per-phase into source.
    Common defines (REDUCE_OP etc.) are kept as-is. Others get phase-prefixed.
    """
    merged = []
    seen_common = set()
    common_defines = {"REDUCE_OP", "REDUCE_DIM", "BCAST_LLKOP", "BCAST_DIM"}

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        for name, value in kernel.defines:
            if name in common_defines:
                if name not in seen_common:
                    merged.append((name, value))
                    seen_common.add(name)
            elif name in _SOURCE_LEVEL_DEFINES:
                continue
            else:
                if i == 0:
                    merged.append((name, value))
                else:
                    merged.append((f"PHASE{i}_{name}", value))

    return merged


# =============================================================================
# Compute Config Validation
# =============================================================================


def _validate_and_get_compute_config(
    phase_kernels: List[Dict[str, Any]],
) -> "ttnn.ComputeConfigDescriptor":
    """Validate that all phases have identical compute configs and return it.

    Compute kernel configs (fp32_dest_acc_en, math_fidelity, math_approx_mode,
    etc.) are hardware settings that cannot be reconfigured mid-kernel.  All
    phases must use exactly the same config.
    """
    base = None
    base_phase = -1

    for phase_idx, pk in enumerate(phase_kernels):
        compute = pk.get("compute")
        if compute is None:
            continue

        config = compute.config
        if base is None:
            base = config
            base_phase = phase_idx
            continue

        # Validate all fields match
        mismatches = []
        for field in ("fp32_dest_acc_en", "math_approx_mode", "math_fidelity", "dst_full_sync_en", "bfp8_pack_precise"):
            base_val = getattr(base, field, None)
            this_val = getattr(config, field, None)
            if base_val != this_val:
                mismatches.append(f"  {field}: phase {base_phase}={base_val}, phase {phase_idx}={this_val}")

        if mismatches:
            raise ValueError(
                f"Compute config mismatch between phases. These are hardware "
                f"settings that cannot change mid-kernel — all phases must use "
                f"identical compute configs.\n" + "\n".join(mismatches)
            )

    if base is None:
        return ttnn.ComputeConfigDescriptor()

    return base


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
        f"setting. To fix: create all descriptors with the same "
        f"compute_kernel_config. For example:\n"
        f"  config = ttnn.layernorm_default_compute_config(device.arch())\n"
        f"  rms = rms_norm.rms_norm(input, ..., compute_kernel_config=config)\n"
        f"  ln  = layer_norm.layer_norm(input, ..., compute_kernel_config=config)"
    )


# =============================================================================
# Barrier Configuration
# =============================================================================


def _create_barrier_config(device: Any, core_ranges: Any) -> BarrierConfig:
    """Create barrier configuration with GlobalSemaphore L1 flags.

    Allocates 4 GlobalSemaphores (one 4-byte L1 word per core each):
      - compute_done: compute signals phase completion
      - writer_done: writer signals phase completion
      - global_arrive: cross-core barrier arrive counter
      - global_release: cross-core barrier release flag (also serves as
        phase release — compute/writer spin on this directly)

    Also computes physical core coordinates for NOC addressing.
    """
    config = BarrierConfig()

    # Create GlobalSemaphores for per-core L1 flags
    sem_compute_done = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_writer_done = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_arrive = ttnn.create_global_semaphore(device, core_ranges, 0)
    sem_global_release = ttnn.create_global_semaphore(device, core_ranges, 0)

    # Store references to prevent GC
    config._sem_refs = [sem_compute_done, sem_writer_done, sem_global_arrive, sem_global_release]

    # Get L1 addresses
    config.compute_done_addr = ttnn.get_global_semaphore_address(sem_compute_done)
    config.writer_done_addr = ttnn.get_global_semaphore_address(sem_writer_done)
    config.global_arrive_addr = ttnn.get_global_semaphore_address(sem_global_arrive)
    config.global_release_addr = ttnn.get_global_semaphore_address(sem_global_release)

    # Compute physical core coordinates for global barrier
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

    return config


# =============================================================================
# Sequential Chain Builder
# =============================================================================


class SequentialChainBuilder:
    """Builds a fused ProgramDescriptor from a sequence of OpDescriptors.

    All readers/writers run for every phase.  Data flows through DRAM between
    phases.  No CB remapping — each phase uses native CB indices (0-31).

    Uses two-level barrier synchronization:
      - Local: L1 flags (via GlobalSemaphore) for per-core RISC coordination
      - Global: NOC semaphore ops for cross-core barrier (dataflow RISC only)
    """

    def __init__(self):
        self.phases: List[PhaseInfo] = []
        self._built = False
        self._barrier_config: Optional[BarrierConfig] = None

    def add_phase(self, op_descriptor: OpDescriptor) -> "SequentialChainBuilder":
        """Add a phase to the sequential chain.

        Args:
            op_descriptor: The OpDescriptor for this phase.  For phases 1+,
                the input tensor should be the previous phase's output tensor
                so the reader reads from the correct DRAM address.

        Returns:
            self for method chaining
        """
        phase_idx = len(self.phases)
        cb_info = extract_cb_info(op_descriptor.descriptor)
        phase = PhaseInfo(
            phase_idx=phase_idx,
            op_descriptor=op_descriptor,
            cb_info=cb_info,
        )
        self.phases.append(phase)
        return self

    def build(self, device: Any) -> OpDescriptor:
        """Build the fused OpDescriptor from the chain.

        Args:
            device: The device (MeshDevice or IDevice) for GlobalSemaphore
                allocation and coordinate conversion.

        Returns:
            Fused OpDescriptor that executes all phases sequentially.
        """
        if self._built:
            raise ValueError("Chain has already been built")
        if not self.phases:
            raise ValueError("Chain has no phases")

        self._built = True

        if len(self.phases) == 1:
            return self.phases[0].op_descriptor

        return self._build_fused_descriptor(device)

    def _build_fused_descriptor(self, device: Any) -> OpDescriptor:
        """Build the fused descriptor with two-level barrier sync."""
        # Validate fp32 consistency
        _validate_fp32_consistency([p.op_descriptor for p in self.phases])

        # Classify kernels by type for each phase
        phase_kernels: List[Dict[str, Any]] = []
        for phase in self.phases:
            kernels_by_type: Dict[str, Any] = {"reader": None, "writer": None, "compute": None}
            for kernel_desc in phase.op_descriptor.descriptor.kernels:
                kernel_type = _classify_kernel(kernel_desc)
                kernels_by_type[kernel_type] = kernel_desc
            phase_kernels.append(kernels_by_type)

        # Merge CB descriptors (max size per index across phases)
        merged_cbs = _merge_cb_descriptors(self.phases)

        core_ranges = self.phases[0].op_descriptor.descriptor.kernels[0].core_ranges

        # Create barrier configuration (GlobalSemaphore L1 flags + core coords)
        self._barrier_config = _create_barrier_config(device, core_ranges)

        # Compute runtime arg offsets for each kernel type
        reader_rt_offsets = _compute_runtime_arg_offsets(phase_kernels, "reader")
        writer_rt_offsets = _compute_runtime_arg_offsets(phase_kernels, "writer")
        compute_rt_offsets = _compute_runtime_arg_offsets(phase_kernels, "compute")

        # Merge compile-time args (concatenate all phases, get offsets)
        reader_ct_args, reader_ct_offsets = _merge_compile_time_args(phase_kernels, "reader")
        writer_ct_args, writer_ct_offsets = _merge_compile_time_args(phase_kernels, "writer")
        compute_ct_args, compute_ct_offsets = _merge_compile_time_args(phase_kernels, "compute")

        # Compute sweep CB indices: ALL CBs with descriptors across phases (generic).
        valid_cb_indices = _get_all_cb_descriptor_indices(self.phases)
        sweep_cb_indices = sorted(valid_cb_indices)

        # Generate fused kernel sources
        fused_reader_source = _generate_fused_reader_source(
            phase_kernels,
            self.phases,
            reader_ct_offsets,
            sweep_cb_indices,
            self._barrier_config,
        )
        fused_writer_source = _generate_fused_writer_source(
            phase_kernels,
            self.phases,
            writer_ct_offsets,
            self._barrier_config,
        )
        fused_compute_source = _generate_fused_compute_source(
            phase_kernels,
            self.phases,
            compute_ct_offsets,
            sweep_cb_indices,
            self._barrier_config,
        )

        # DEBUG: dump generated sources to /tmp for inspection
        for name, src in [
            ("reader", fused_reader_source),
            ("writer", fused_writer_source),
            ("compute", fused_compute_source),
        ]:
            if src:
                with open(f"/tmp/fused_{name}_debug.cpp", "w") as f:
                    f.write(src)

        # Concatenate runtime args and append barrier addresses
        reader_rt_args = _concatenate_runtime_args(phase_kernels, "reader")
        writer_rt_args = _concatenate_runtime_args(phase_kernels, "writer")
        compute_rt_args = _concatenate_runtime_args(phase_kernels, "compute")

        bc = self._barrier_config

        # Reader gets 4 barrier addresses
        reader_barrier_addrs = [
            bc.compute_done_addr,
            bc.writer_done_addr,
            bc.global_arrive_addr,
            bc.global_release_addr,
        ]
        reader_rt_args, reader_barrier_offset = _append_barrier_runtime_args(
            reader_rt_args,
            reader_barrier_addrs,
        )

        # Writer gets 2 barrier addresses (writer_done, global_release)
        writer_barrier_addrs = [bc.writer_done_addr, bc.global_release_addr]
        writer_rt_args, writer_barrier_offset = _append_barrier_runtime_args(
            writer_rt_args,
            writer_barrier_addrs,
        )

        # Compute gets 2 barrier addresses (compute_done, global_release)
        compute_barrier_addrs = [bc.compute_done_addr, bc.global_release_addr]
        compute_rt_args, compute_barrier_offset = _append_barrier_runtime_args(
            compute_rt_args,
            compute_barrier_addrs,
        )

        # Build fused kernel descriptors
        fused_kernels = []

        # Fused reader
        if fused_reader_source is not None:
            reader_desc = ttnn.KernelDescriptor()
            reader_desc.kernel_source = fused_reader_source
            reader_desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            reader_desc.core_ranges = core_ranges
            reader_desc.compile_time_args = reader_ct_args
            reader_desc.named_compile_time_args = _merge_named_compile_time_args(
                phase_kernels,
                "reader",
                reader_rt_offsets,
                barrier_rt_offset=reader_barrier_offset,
                barrier_config=self._barrier_config,
            )
            reader_desc.defines = _merge_defines(phase_kernels, "reader")
            reader_desc.runtime_args = reader_rt_args
            reader_desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, "reader")
            reader_desc.config = phase_kernels[0]["reader"].config
            fused_kernels.append(reader_desc)

        # Fused writer
        if fused_writer_source is not None:
            writer_desc = ttnn.KernelDescriptor()
            writer_desc.kernel_source = fused_writer_source
            writer_desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            writer_desc.core_ranges = core_ranges
            writer_desc.compile_time_args = writer_ct_args
            writer_desc.named_compile_time_args = _merge_named_compile_time_args(
                phase_kernels,
                "writer",
                writer_rt_offsets,
                barrier_rt_offset=writer_barrier_offset,
            )
            writer_desc.defines = _merge_defines(phase_kernels, "writer")
            writer_desc.runtime_args = writer_rt_args
            writer_desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, "writer")
            # Use first available writer config
            for pk in phase_kernels:
                if pk["writer"] is not None:
                    writer_desc.config = pk["writer"].config
                    break
            fused_kernels.append(writer_desc)

        # Fused compute
        if fused_compute_source is not None:
            compute_desc = ttnn.KernelDescriptor()
            compute_desc.kernel_source = fused_compute_source
            compute_desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            compute_desc.core_ranges = core_ranges
            compute_desc.compile_time_args = compute_ct_args
            compute_desc.named_compile_time_args = _merge_named_compile_time_args(
                phase_kernels,
                "compute",
                compute_rt_offsets,
                barrier_rt_offset=compute_barrier_offset,
            )
            compute_desc.defines = _merge_defines(phase_kernels, "compute")
            compute_desc.runtime_args = compute_rt_args
            compute_desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, "compute")
            compute_desc.config = _validate_and_get_compute_config(phase_kernels)
            fused_kernels.append(compute_desc)

        # Merge semaphores (dedup by ID)
        all_semaphores = []
        seen_sem_ids: Set[int] = set()
        for phase in self.phases:
            for sem in phase.op_descriptor.descriptor.semaphores:
                if sem.id not in seen_sem_ids:
                    all_semaphores.append(sem)
                    seen_sem_ids.add(sem.id)

        # Collect input/output tensors
        all_input_tensors = []
        for phase in self.phases:
            for tensor in phase.op_descriptor.input_tensors:
                if tensor not in all_input_tensors:
                    all_input_tensors.append(tensor)

        output_tensor = None
        if self.phases[-1].op_descriptor.output_tensors:
            output_tensor = self.phases[-1].op_descriptor.output_tensors[0]

        # Create the merged ProgramDescriptor
        merged_descriptor = ttnn.ProgramDescriptor()
        merged_descriptor.kernels = fused_kernels
        merged_descriptor.cbs = merged_cbs
        merged_descriptor.semaphores = all_semaphores

        return OpDescriptor(
            descriptor=merged_descriptor,
            input_tensors=all_input_tensors,
            output_tensors=[output_tensor] if output_tensor else [],
            keepalive=tuple(self._barrier_config._sem_refs),
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def chain_descriptors(descriptors: List[OpDescriptor], device: Any) -> OpDescriptor:
    """Chain multiple OpDescriptors sequentially.

    For phases 1+, the input tensor should be the previous phase's output
    tensor.  This ensures each reader reads from the correct DRAM address.

    Args:
        descriptors: List of OpDescriptors to chain sequentially.
        device: The device for GlobalSemaphore allocation.

    Returns:
        Fused OpDescriptor.
    """
    builder = SequentialChainBuilder()
    for desc in descriptors:
        builder.add_phase(desc)
    return builder.build(device)


def create_parallel_chain_descriptors(
    chains: List[List[OpDescriptor]],
    device: Any,
) -> List[OpDescriptor]:
    """Create fused descriptors for multiple parallel chains.

    Each chain is fused sequentially, and the resulting fused ops can be
    run in parallel using composite.launch().

    Args:
        chains: List of chains, where each chain is a list of OpDescriptors.
        device: The device for GlobalSemaphore allocation.

    Returns:
        List of fused OpDescriptors, one per chain.
    """
    fused_descriptors = []
    for chain in chains:
        if not chain:
            continue
        if len(chain) == 1:
            fused_descriptors.append(chain[0])
        else:
            fused_descriptors.append(chain_descriptors(chain, device))
    return fused_descriptors


__all__ = [
    # Core classes
    "SequentialChainBuilder",
    "PhaseInfo",
    "CBInfo",
    "BarrierConfig",
    # Functions
    "chain_descriptors",
    "create_parallel_chain_descriptors",
    "extract_cb_info",
    "extract_cb_names_from_kernel",
]
