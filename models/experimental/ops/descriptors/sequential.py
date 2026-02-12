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
    """Classify a kernel as reader, writer, or compute based on its config.

    DEPRECATED: Use _get_risc_type() + _get_role_key() instead.
    """
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return "compute"
    elif isinstance(config, ttnn.ReaderConfigDescriptor):
        return "reader"
    elif isinstance(config, ttnn.WriterConfigDescriptor):
        return "writer"
    elif isinstance(config, ttnn.DataMovementConfigDescriptor):
        if config.processor == ttnn.DataMovementProcessor.RISCV_0:
            return "reader"
        else:
            return "writer"
    return "unknown"


def _get_risc_type(kernel_desc: "ttnn.KernelDescriptor") -> str:
    """Return the RISC processor type: 'riscv_0', 'riscv_1', or 'compute'."""
    config = kernel_desc.config
    if isinstance(config, ttnn.ComputeConfigDescriptor):
        return "compute"
    elif isinstance(config, ttnn.ReaderConfigDescriptor):
        return "riscv_0"
    elif isinstance(config, ttnn.WriterConfigDescriptor):
        return "riscv_1"
    elif isinstance(config, ttnn.DataMovementConfigDescriptor):
        if config.processor == ttnn.DataMovementProcessor.RISCV_0:
            return "riscv_0"
        else:
            return "riscv_1"
    return "unknown"


def _core_ranges_key(core_ranges: Any) -> frozenset:
    """Create a hashable key from a CoreRangeSet for grouping."""
    return frozenset((cr.start.x, cr.start.y, cr.end.x, cr.end.y) for cr in core_ranges.ranges())


def _get_role_key(kernel_desc: "ttnn.KernelDescriptor") -> Tuple[str, frozenset]:
    """Return (risc_type, core_ranges_key) identifying this kernel's role."""
    return (_get_risc_type(kernel_desc), _core_ranges_key(kernel_desc.core_ranges))


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
    """Resolve preprocessor #ifdef/#ifndef/#if defined/#elif directives in source code.

    Only resolves directives involving known source-level defines (RMSNORM,
    FUSE_PRE_ADD, FUSED_PRE_ADD). Other directives are left untouched.

    Stack entries are (current_branch_included, known, any_branch_taken) tuples.
    ``any_branch_taken`` tracks whether any prior branch in a #if/#elif/#else
    chain was included, so that #elif and #else can skip when a prior branch
    was already selected.
    """
    lines = source.split("\n")
    result = []
    # Stack of (current_included, known_directive, any_prior_branch_taken)
    stack: List[Tuple[bool, bool, bool]] = []

    for line in lines:
        stripped = line.strip()
        directive_handled = False

        if stripped.startswith("#if defined"):
            names_found = re.findall(r"\bdefined\s+(\w+)", stripped)
            involved = [n for n in names_found if n in _SOURCE_LEVEL_DEFINES]
            if involved:
                result_val = _eval_ifdef_expression(stripped, active_defines)
                stack.append((result_val, True, result_val))
                directive_handled = True
            else:
                stack.append((True, False, False))

        elif stripped.startswith("#ifdef "):
            name = stripped[7:].strip()
            if name in _SOURCE_LEVEL_DEFINES:
                result_val = name in active_defines
                stack.append((result_val, True, result_val))
                directive_handled = True
            else:
                stack.append((True, False, False))

        elif stripped.startswith("#ifndef "):
            name = stripped[8:].strip()
            if name in _SOURCE_LEVEL_DEFINES:
                result_val = name not in active_defines
                stack.append((result_val, True, result_val))
                directive_handled = True
            else:
                stack.append((True, False, False))

        elif stripped.startswith("#elif "):
            if stack and stack[-1][1]:
                _, known, any_taken = stack[-1]
                if any_taken:
                    # A prior branch was already taken — exclude this branch
                    stack[-1] = (False, known, True)
                else:
                    # No prior branch taken — evaluate this condition
                    names_found = re.findall(r"\bdefined\s+(\w+)", stripped)
                    involved = [n for n in names_found if n in _SOURCE_LEVEL_DEFINES]
                    if involved:
                        result_val = _eval_ifdef_expression(stripped, active_defines)
                        stack[-1] = (result_val, known, result_val)
                    else:
                        # Unknown define in #elif — conservatively include
                        stack[-1] = (True, known, True)
                directive_handled = True

        elif stripped == "#else" or stripped.startswith("#else ") or stripped.startswith("#else\t"):
            if stack and stack[-1][1]:
                _, known, any_taken = stack[-1]
                # #else is included only if no prior branch was taken
                stack[-1] = (not any_taken, known, True)
                directive_handled = True

        elif (
            stripped == "#endif"
            or stripped.startswith("#endif ")
            or stripped.startswith("#endif\t")
            or stripped.startswith("#endif//")
            or stripped.startswith("#endif /")
        ):
            if stack:
                _, known, _ = stack[-1]
                stack.pop()
                if known:
                    directive_handled = True

        if directive_handled:
            continue

        include = True
        for incl, known, _ in stack:
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
    """Extract code before kernel_main (namespaces, helpers, globals).

    Strips all preprocessor directives (``#include``, ``#define``,
    ``#pragma``, etc.), single-line comments (``//``), and block
    comments (``/* ... */``).  Block comment stripping is important
    for safe dedup across phases — without it, line-by-line dedup
    can break a block comment by deduplicating the opening ``/*``
    while keeping unique body lines (e.g. ``* @brief``), leaving
    orphaned Doxygen lines as invalid C++.

    Preprocessor directives are collected separately by
    ``_collect_includes`` and ``_collect_defines``; any remaining
    directives (``#pragma once``, resolved ``#if`` remnants) are
    not needed in the generated fused kernel source.
    """
    lines = source.split("\n")
    pre_main = []
    in_block_comment = False
    for line in lines:
        if "void kernel_main()" in line or "KERNEL_MAIN" in line:
            break
        stripped = line.strip()

        # Track block comments (/* ... */)
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block_comment = True
            continue

        if stripped and not stripped.startswith("#") and not stripped.startswith("//"):
            pre_main.append(line)
    return "\n".join(pre_main)


# Regex pattern matching C/C++ global/static variable declarations.
# Captures: optional 'static', type, variable name.
# Excludes: functions (parens), namespaces, typedefs, using, ALWI/FORCE_INLINE.
_GLOBAL_VAR_RE = re.compile(
    r"^(?:static\s+)?"  # optional static
    r"(?:volatile\s+)?"  # optional volatile
    r"(?:constexpr\s+)?"  # optional constexpr
    r"(?:const\s+)?"  # optional const
    r"(?:(?:u?int(?:8|16|32|64)_t|float|double|bool|char|uint|size_t|"
    r"tt_l1_ptr\s+\w+)\s+)"  # type (common C/C++ types)
    r"(\w+)"  # variable name (capture group 1)
    r"\s*(?:=|;|\[)"  # followed by = or ; or [
)


def _extract_global_var_names(pre_main: str) -> List[str]:
    """Find global/static variable names in pre-main code.

    Returns variable names that should be prefixed per phase to avoid
    collisions in fused kernels. Namespace aliases, inline functions,
    and type definitions are excluded.
    """
    names = []
    for line in pre_main.split("\n"):
        stripped = line.strip()
        # Skip namespace aliases, function definitions, typedefs
        if any(
            kw in stripped
            for kw in ["namespace ", "ALWI ", "FORCE_INLINE ", "inline ", "typedef ", "using ", "void ", "template"]
        ):
            continue
        # Skip lines that look like function declarations (contain parens)
        if "(" in stripped and "=" not in stripped.split("(")[0]:
            continue
        m = _GLOBAL_VAR_RE.match(stripped)
        if m:
            names.append(m.group(1))
    return names


def _prefix_globals_in_source(source: str, phase_idx: int, global_names: List[str]) -> str:
    """Prefix global variable names with phase prefix in source code.

    Applied to both pre-main declarations and phase body to ensure
    consistent renaming of global/static variables per phase.
    """
    if phase_idx == 0 or not global_names:
        return source
    for name in global_names:
        source = re.sub(rf"\b{re.escape(name)}\b", f"phase{phase_idx}_{name}", source)
    return source


def _split_into_top_level_blocks(pre_main: str) -> List[str]:
    """Split pre-main code into top-level C++ declarations.

    Uses brace counting to keep each top-level construct (function,
    namespace, struct, class) as a single unit.  Single-line declarations
    (namespace aliases, ``using``, ``typedef``) are individual blocks.

    Block boundaries are detected when brace depth returns to 0 at a
    statement boundary (line ending with ``;``, ``}``, or ``};``), or
    when an empty line appears between depth-0 constructs.
    """
    blocks: List[str] = []
    current_lines: List[str] = []
    brace_depth = 0

    for line in pre_main.split("\n"):
        stripped = line.strip()

        # Empty lines separate top-level blocks at depth 0
        if not stripped:
            if current_lines and brace_depth == 0:
                blocks.append("\n".join(current_lines))
                current_lines = []
            continue

        current_lines.append(line)
        brace_depth += stripped.count("{") - stripped.count("}")
        brace_depth = max(0, brace_depth)  # Defensive

        # Emit block when brace depth returns to 0 at a statement boundary.
        # Strip trailing C++ comments (// ...) before checking line ending,
        # so that "} // namespace my_ns" is recognized as ending with '}'.
        if brace_depth == 0 and current_lines:
            check = stripped.split("//")[0].rstrip() if "//" in stripped else stripped
            if check.endswith(";") or check.endswith("}") or check.endswith("};"):
                blocks.append("\n".join(current_lines))
                current_lines = []

    # Flush any remaining lines
    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


def _normalize_block(block: str) -> str:
    """Normalize a code block for deduplication comparison.

    Strips leading/trailing whitespace on each line, collapses
    multiple spaces to single space, and removes empty lines.
    """
    lines = []
    for line in block.split("\n"):
        normalized = " ".join(line.split())
        if normalized:
            lines.append(normalized)
    return "\n".join(lines)


def _extract_block_signature(block: str) -> str:
    """Extract the declaration signature of a top-level block.

    For braced blocks (functions, namespaces, structs): returns the
    declaration line(s) before the opening ``{``, normalized.
    For single-line declarations: returns the full line, normalized.

    Used for signature-based deduplication — blocks with the same
    signature are considered the same declaration (first wins).
    """
    lines = block.strip().split("\n")
    sig_parts: List[str] = []
    for line in lines:
        stripped = line.strip()
        if "{" in stripped:
            before = stripped.split("{")[0].strip()
            if before:
                sig_parts.append(before)
            break
        sig_parts.append(stripped)

    if not sig_parts:
        return _normalize_block(block)

    sig = " ".join(sig_parts)
    return " ".join(sig.split())


def _is_global_var_block(block: str) -> bool:
    """Check if a top-level block is a global/static variable declaration."""
    stripped = block.strip()
    # Global variables are single-line declarations (no multi-line bodies)
    if "\n" in stripped:
        return False
    # Exclude namespace aliases, function definitions, typedefs, etc.
    if any(
        kw in stripped
        for kw in [
            "namespace ",
            "ALWI ",
            "FORCE_INLINE ",
            "inline ",
            "typedef ",
            "using ",
            "void ",
            "template",
        ]
    ):
        return False
    if "(" in stripped and "=" not in stripped.split("(")[0]:
        return False
    return bool(_GLOBAL_VAR_RE.match(stripped))


def _collect_all_pre_main_code(sources_with_indices: List[Tuple[int, str]]) -> str:
    """Merge pre-main code from all phases, deduplicating top-level declarations.

    Extracts pre-main from each phase, splits into top-level C++ blocks
    (brace-balanced), and merges using a two-tier dedup strategy:

    1. **Braced blocks** (functions, namespaces, structs): dedup by
       *signature* — the declaration part before ``{``.  First occurrence
       wins.  This correctly handles the case where the same template
       function appears in multiple phases with slightly different bodies
       due to ``#ifdef`` resolution (e.g., LN vs RMS variants of the
       same utility header).

    2. **Single-line declarations** (namespace aliases, ``using``,
       ``typedef``): dedup by exact normalized content.

    3. **Global variables** from phase N>0: included with a
       ``phaseN_`` prefix on the variable name to avoid collisions.
       Phase 0's globals are included as-is.

    This ensures:
      - Same helper from same base kernel → deduped (first copy kept)
      - Different helpers from different ops → both included
      - Global variables → phase-prefixed for N>0
    """
    if not sources_with_indices:
        return ""

    all_blocks: List[str] = []
    seen_signatures: Set[str] = set()
    seen_content: Set[str] = set()

    for phase_idx, source in sources_with_indices:
        pre_main = _collect_pre_main_code(source)
        blocks = _split_into_top_level_blocks(pre_main)

        for block in blocks:
            normalized = _normalize_block(block)
            if not normalized:
                continue

            # Global variables: phase N>0 gets prefixed
            if _is_global_var_block(block):
                if phase_idx == 0:
                    if normalized not in seen_content:
                        seen_content.add(normalized)
                        all_blocks.append(block)
                else:
                    names = _extract_global_var_names(block)
                    prefixed = block
                    for name in names:
                        prefixed = re.sub(
                            rf"\b{re.escape(name)}\b",
                            f"phase{phase_idx}_{name}",
                            prefixed,
                        )
                    prefixed_norm = _normalize_block(prefixed)
                    if prefixed_norm not in seen_content:
                        seen_content.add(prefixed_norm)
                        all_blocks.append(prefixed)
                continue

            # Braced blocks: dedup by signature (first wins)
            has_braces = "{" in normalized and "}" in normalized
            if has_braces:
                sig = _extract_block_signature(block)
                if sig not in seen_signatures:
                    seen_signatures.add(sig)
                    all_blocks.append(block)
            else:
                # Single-line declarations: dedup by content
                if normalized not in seen_content:
                    seen_content.add(normalized)
                    all_blocks.append(block)

    return "\n\n".join(all_blocks)


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


def _transform_phase_source(
    source: str,
    phase_idx: int,
    ct_arg_offset: int = 0,
    global_names: Optional[List[str]] = None,
) -> str:
    """Apply all transformations for a phase's kernel body."""
    source = _prefix_named_args_in_source(source, phase_idx)
    source = _offset_compile_time_args_in_source(source, phase_idx, ct_arg_offset)
    source = _offset_runtime_args_in_source(source, phase_idx)
    if global_names:
        source = _prefix_globals_in_source(source, phase_idx, global_names)
    return source


# =============================================================================
# CB Descriptor Merging
# =============================================================================


def _validate_cb_page_sizes(phases: List[PhaseInfo]) -> None:
    """Validate that shared CB indices have compatible page sizes across phases.

    When two phases share the same CB index, the page_size must match because
    BRISC/NCRISC use fifo_page_size to advance pointers (cb_push_back/cb_pop_front).
    A mismatch would cause pointer arithmetic errors and data corruption.
    """
    # Collect page_size per (cb_index, phase_idx)
    cb_page_sizes: Dict[int, List[Tuple[int, int]]] = {}  # cb_idx -> [(phase_idx, page_size), ...]
    for phase_idx, phase in enumerate(phases):
        desc = phase.op_descriptor.descriptor
        for cb_desc in desc.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                cb_idx = fmt_desc.buffer_index
                if cb_idx not in cb_page_sizes:
                    cb_page_sizes[cb_idx] = []
                cb_page_sizes[cb_idx].append((phase_idx, fmt_desc.page_size))

    for cb_idx, entries in cb_page_sizes.items():
        if len(entries) <= 1:
            continue
        sizes = {ps for _, ps in entries}
        if len(sizes) > 1:
            detail = ", ".join(f"phase {pi}: {ps}" for pi, ps in entries)
            raise ValueError(
                f"CB[{cb_idx}] has mismatched page sizes across phases ({detail}). "
                f"All phases sharing a CB must use the same page size. "
                f"This typically indicates a data format mismatch (e.g., BF16 vs FP32). "
                f"Ensure compute_kernel_config is consistent across fused ops."
            )


def _merge_cb_descriptors(phases: List[PhaseInfo]) -> list:
    """Merge CB descriptors from all phases.

    For each CB index used by any phase, keeps the descriptor with the
    largest total_size so the CB can accommodate any phase's data.

    When multiple phases have buffer-backed CBs at the same index, the merge
    always keeps phase 0's buffer. This guarantees phase 0 is correct without
    rebinding; only phases 1+ need mid-kernel CB address rebinding.
    """
    _validate_cb_page_sizes(phases)
    cb_by_index: Dict[int, Any] = {}  # cb_index -> (largest_total_size, cb_desc, phase_idx)

    for phase_idx, phase in enumerate(phases):
        desc = phase.op_descriptor.descriptor
        for cb_desc in desc.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                cb_idx = fmt_desc.buffer_index
                if cb_idx not in cb_by_index:
                    cb_by_index[cb_idx] = (cb_desc.total_size, cb_desc, phase_idx)
                elif cb_desc.total_size > cb_by_index[cb_idx][0]:
                    _, old_desc, old_phase = cb_by_index[cb_idx]
                    if old_desc.has_buffer() and old_phase == 0:
                        # Phase 0 had the buffer — keep its descriptor (for correct
                        # initial setup) even though a later phase has larger total_size
                        cb_by_index[cb_idx] = (cb_desc.total_size, old_desc, old_phase)
                    else:
                        cb_by_index[cb_idx] = (cb_desc.total_size, cb_desc, phase_idx)

    return [cb_desc for _, (_, cb_desc, _) in sorted(cb_by_index.items())]


# =============================================================================
# CB Address Rebinding
# =============================================================================


def _compute_rebind_info(
    phases: List[PhaseInfo],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """Compute which CBs need address rebinding at each phase transition.

    For each phase 1+, identifies CB indices where the buffer address differs
    from what was set in the previous phase. Phase 0 never needs rebinding
    because _merge_cb_descriptors always keeps phase 0's buffer.

    Returns:
        Dict mapping phase_idx -> list of (cb_idx, new_addr, new_size) tuples.
    """
    # Collect per-phase buffer addresses
    phase_buffer_addrs: List[Dict[int, Tuple[int, int]]] = []
    for phase in phases:
        addrs: Dict[int, Tuple[int, int]] = {}
        for cb_desc in phase.op_descriptor.descriptor.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                if cb_desc.has_buffer():
                    addr = cb_desc.buffer_address()
                    if addr is not None:
                        addrs[fmt_desc.buffer_index] = (addr, cb_desc.total_size)
        phase_buffer_addrs.append(addrs)

    if not phase_buffer_addrs:
        return {}

    # Start with phase 0's addresses as baseline
    rebind_info: Dict[int, List[Tuple[int, int, int]]] = {}
    current_addrs = dict(phase_buffer_addrs[0])

    for phase_idx in range(1, len(phases)):
        rebinds: List[Tuple[int, int, int]] = []
        for cb_idx, (phase_addr, phase_size) in phase_buffer_addrs[phase_idx].items():
            current = current_addrs.get(cb_idx)
            if current is None or current[0] != phase_addr:
                rebinds.append((cb_idx, phase_addr, phase_size))
                current_addrs[cb_idx] = (phase_addr, phase_size)
        rebind_info[phase_idx] = rebinds

    return rebind_info


def _generate_rebind_code(
    rebinds: List[Tuple[int, int, int]],
    phase_idx: int,
    for_compute: bool = False,
) -> List[str]:
    """Generate C++ code to rebind CB addresses for a phase.

    Args:
        rebinds: List of (cb_idx, addr, size) tuples for this phase.
        phase_idx: Which phase these rebinds are for.
        for_compute: If True, shift addresses by >> 4 for TRISC and guard
            with #ifndef TRISC_MATH (TRISC1 has no cb_interface).

    Returns:
        List of C++ source lines (indented with 4 spaces).
    """
    if not rebinds:
        return []
    lines = [f"    // Rebind CB addresses for phase {phase_idx}"]
    if for_compute:
        # TRISC1 (math) doesn't have cb_interface linked in — skip it
        lines.append("#ifndef TRISC_MATH")
    for cb_idx, _, _ in rebinds:
        prefix = f"phase{phase_idx}_cb{cb_idx}"
        if for_compute:
            lines.append(f"    {{")
            lines.append(
                f'        constexpr uint32_t new_addr = get_named_compile_time_arg_val("{prefix}_rebind_addr") >> 4;'
            )
            lines.append(
                f'        constexpr uint32_t new_size = get_named_compile_time_arg_val("{prefix}_rebind_size") >> 4;'
            )
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = new_addr;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = new_addr;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_size = new_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_limit = new_addr + new_size;")
            lines.append(f"    }}")
        else:
            lines.append(f"    {{")
            lines.append(
                f'        constexpr uint32_t new_addr = get_named_compile_time_arg_val("{prefix}_rebind_addr");'
            )
            lines.append(
                f'        constexpr uint32_t new_size = get_named_compile_time_arg_val("{prefix}_rebind_size");'
            )
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = new_addr;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = new_addr;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_size = new_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_limit = new_addr + new_size;")
            lines.append(f"    }}")
    if for_compute:
        lines.append("#endif")
    return lines


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


def _generate_fused_riscv0_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    sweep_cb_indices: List[int],
    barrier_config: Optional[BarrierConfig] = None,
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    op_semaphore_ids: Optional[List[int]] = None,
) -> Optional[str]:
    """Generate fused RISCV_0 (reader/BRISC) kernel source with two-level barrier sync.

    Between phases, the RISCV_0 processor acts as the barrier coordinator:
      1. Wait for local compute + writer to signal done (L1 flag spin)
      2. Reset residual tiles from ALL CBs on BRISC
      3. Global barrier across cores (sets global_release which also serves
         as the phase release signal for compute/writer)

    The BRISC reset updates stream register tiles_acked but NOT TRISC0's
    local copy.  Compute must resync after being released (see compute source).
    """
    reader_sources = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        source = _inline_local_includes(source, kernel_dir)
        phase_defs = {name for name, _ in kernel.defines} if hasattr(kernel, "defines") else set()
        resolved = _resolve_ifdef_directives(source, phase_defs)
        reader_sources.append((i, resolved))

    if not reader_sources:
        return None

    all_sources = [s for _, s in reader_sources]
    includes = _collect_includes(all_sources)
    defines = _collect_defines(all_sources)
    # Merge pre-main code from all phases (namespace aliases, helpers, globals).
    # Block comments are stripped by _collect_pre_main_code so line-level dedup is safe.
    pre_main = _collect_all_pre_main_code(reader_sources)

    # Extract per-phase global variable names for body prefixing
    phase_globals: Dict[int, List[str]] = {}
    for phase_idx, source in reader_sources:
        pm = _collect_pre_main_code(source)
        names = _extract_global_var_names(pm)
        if names:
            phase_globals[phase_idx] = names

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

        # BRISC-side CB reset: equalize stream registers + reset pointers to CB start.
        # Uses direct tt_reg_ptr stream register increment (no cb_pop_front dependency).
        # The stream controller requires per-tile increments — bulk acked += N hangs.
        lines.append("// BRISC-side CB reset: equalize stream registers + reset pointers to CB start.")
        lines.append("FORCE_INLINE void __cb_reset_to_empty() {")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(f"        uint16_t remaining = (uint16_t)(*get_cb_tiles_received_ptr({cb_idx}))")
            lines.append(f"                          - (uint16_t)(*get_cb_tiles_acked_ptr({cb_idx}));")
            lines.append(f"        volatile tt_reg_ptr uint32_t* acked_ptr = (volatile tt_reg_ptr uint32_t*)")
            lines.append(f"            ((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr({cb_idx}));")
            lines.append(f"        for (uint16_t i = 0; i < remaining; i++) {{")
            lines.append(f"            acked_ptr[0] += 1;")
            lines.append(f"        }}")
            lines.append(f"        // Reset BRISC local pointers to CB start")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
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
        gnames = phase_globals.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, global_names=gnames)

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

    if rebind_info is None:
        rebind_info = {}

    first = True
    for phase_idx, _ in reader_sources:
        if not first and is_multi_phase:
            lines.append("")
            lines.append(f"    // === Barrier: Phase {phase_idx - 1} -> Phase {phase_idx} ===")
            lines.append("    // Invariant: BRISC (reader) coordinates all inter-phase cleanup.")
            lines.append("    // Order: noc_barrier -> wait compute/writer done -> reset CBs ->")
            lines.append("    //         reset semaphores -> rebind CB addrs -> global barrier.")
            lines.append("    // Compute/writer must NOT touch CBs until global_release is set.")
            lines.append("    noc_async_full_barrier();")
            lines.append("")
            lines.append(f"    // Wait for local compute + writer to finish Phase {phase_idx - 1}")
            lines.append(f"    noc_semaphore_wait_min(__compute_done, {phase_idx});")
            lines.append(f"    noc_semaphore_wait_min(__writer_done, {phase_idx});")
            lines.append("")
            lines.append("    // Reset residual tiles from ALL CBs")
            lines.append("    __cb_reset_to_empty();")
            lines.append("")
            # Reset op semaphores to initial value (0) so next phase starts clean
            if op_semaphore_ids:
                lines.append("    // Reset op semaphores to 0 (as if each phase runs standalone)")
                for sem_id in op_semaphore_ids:
                    lines.append(f"    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = 0;")
                lines.append("")
            # Rebind CB addresses before global barrier (so BRISC has correct state)
            rebind_lines = _generate_rebind_code(rebind_info.get(phase_idx, []), phase_idx, for_compute=False)
            if rebind_lines:
                lines.extend(rebind_lines)
                lines.append("")
            lines.append("    // Global barrier (sets global_release, releasing compute/writer)")
            lines.append(f"    __global_barrier({phase_idx - 1}, __global_arrive, __global_release);")
            lines.append("")
        lines.append(f"    phase{phase_idx}_reader();")
        first = False
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_riscv1_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    sweep_cb_indices: List[int],
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    barrier_config: Optional[BarrierConfig] = None,
) -> Optional[str]:
    """Generate fused RISCV_1 (writer/NCRISC) kernel source with L1 flag barrier sync.

    Between phases, the writer:
      1. Signals done by writing phase+1 to writer_done L1 flag
      2. Spins on global_release L1 flag (plain volatile read, no NOC APIs)
      3. Resyncs NCRISC local CB pointers to CB start
    """
    writer_sources = []

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        source = _inline_local_includes(source, kernel_dir)
        phase_defs = {name for name, _ in kernel.defines} if hasattr(kernel, "defines") else set()
        resolved = _resolve_ifdef_directives(source, phase_defs)
        writer_sources.append((i, resolved))

    if not writer_sources:
        return None

    all_sources = [s for _, s in writer_sources]
    includes = _collect_includes(all_sources)
    defines = _collect_defines(all_sources)
    pre_main = _collect_all_pre_main_code(writer_sources)

    # Extract per-phase global variable names for body prefixing
    phase_globals: Dict[int, List[str]] = {}
    for phase_idx, source in writer_sources:
        pm = _collect_pre_main_code(source)
        names = _extract_global_var_names(pm)
        if names:
            phase_globals[phase_idx] = names

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

    # Generate NCRISC CB state resync function (resets local pointers to CB start).
    if sweep_cb_indices and is_multi_phase:
        lines.append("// Resync NCRISC local CB pointers to CB start between phases.")
        lines.append("FORCE_INLINE void __resync_ncrisc_cb_state() {")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")

    # Generate phase functions
    for phase_idx, resolved_source in writer_sources:
        body = _extract_kernel_body_for_fusion(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        gnames = phase_globals.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, global_names=gnames)

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

    if rebind_info is None:
        rebind_info = {}

    num_writers = len(writer_sources)
    for count, (phase_idx, _) in enumerate(writer_sources):
        lines.append(f"    phase{phase_idx}_writer();")
        if count < num_writers - 1 and is_multi_phase:
            next_phase_idx = writer_sources[count + 1][0]
            lines.append("")
            lines.append(f"    // Ensure all async NOC writes from Phase {phase_idx} are complete")
            lines.append("    noc_async_write_barrier();")
            lines.append(f"    // Signal done for Phase {phase_idx}")
            lines.append(f"    *__writer_done = {phase_idx + 1};")
            lines.append("")
            lines.append(f"    // Wait for global release (Phase {phase_idx + 1})")
            lines.append(f"    while (*__global_release < {phase_idx + 1}) {{ }}")
            lines.append("")
            # Resync NCRISC local CB pointers to CB start
            if sweep_cb_indices:
                lines.append("    // Resync NCRISC CB pointers to start")
                lines.append("    __resync_ncrisc_cb_state();")
                lines.append("")
            # Rebind CB addresses after barrier wait
            rebind_lines = _generate_rebind_code(rebind_info.get(next_phase_idx, []), next_phase_idx, for_compute=False)
            if rebind_lines:
                lines.extend(rebind_lines)
                lines.append("")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _generate_fused_compute_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Optional[Dict[int, int]] = None,
    sweep_cb_indices: Optional[List[int]] = None,
    barrier_config: Optional[BarrierConfig] = None,
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
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
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        source = _inline_local_includes(source, kernel_dir)
        phase_defs = {name for name, _ in kernel.defines}
        resolved = _resolve_ifdef_directives(source, phase_defs)
        compute_sources.append((i, resolved))

    if not compute_sources:
        return None

    all_sources = [s for _, s in compute_sources]
    includes = _collect_includes(all_sources)
    defines = _collect_defines(all_sources)
    pre_main = _collect_all_pre_main_code(compute_sources)

    # Extract per-phase global variable names for body prefixing
    phase_globals: Dict[int, List[str]] = {}
    for phase_idx, source in compute_sources:
        pm = _collect_pre_main_code(source)
        names = _extract_global_var_names(pm)
        if names:
            phase_globals[phase_idx] = names

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
    # After BRISC's __cb_reset_to_empty(), stream registers are equalized and
    # BRISC pointers are at CB start. TRISC0 and TRISC2 need to sync their
    # local state and reset pointers to CB start as well.
    if sweep_cb_indices and is_multi_phase:
        lines.append("// Resync compute-side local CB state after BRISC reset.")
        lines.append("// TRISC0: sync tiles_acked + reset fifo_rd_ptr to CB start.")
        lines.append("// TRISC2: sync tiles_received + reset fifo_wr_ptr to CB start.")
        lines.append("FORCE_INLINE void __resync_cb_state_after_sweep() {")
        lines.append("#ifdef TRISC_UNPACK")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(
                f"        uint16_t stream_acked = (uint16_t)reg_read((uint32_t)get_cb_tiles_acked_ptr({cb_idx}));"
            )
            lines.append(f"        get_local_cb_interface({cb_idx}).tiles_acked = stream_acked;")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"    }}")
        lines.append("#endif")
        lines.append("#ifdef TRISC_PACK")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(
                f"        uint16_t stream_received = (uint16_t)reg_read((uint32_t)get_cb_tiles_received_ptr({cb_idx}));"
            )
            lines.append(f"        get_local_cb_interface({cb_idx}).tiles_received = stream_received;")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_tile_ptr = 0;")
            lines.append(f"    }}")
        lines.append("#endif")
        lines.append("}")
        lines.append("")

    # Generate phase functions
    for phase_idx, resolved_source in compute_sources:
        body = _extract_kernel_body_for_fusion(resolved_source)
        offset = ct_arg_offsets.get(phase_idx, 0)
        gnames = phase_globals.get(phase_idx, [])
        transformed = _transform_phase_source(body, phase_idx, offset, global_names=gnames)

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

    if rebind_info is None:
        rebind_info = {}

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
            # Rebind CB addresses after resync (all TRISC instances, with >> 4 shift)
            rebind_lines = _generate_rebind_code(rebind_info.get(phase_idx, []), phase_idx, for_compute=True)
            if rebind_lines:
                lines.extend(rebind_lines)
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

        # Count runtime args for this phase (max across cores).
        # RuntimeArgsView uses coordinate-based 2D indexing: [x][y] -> CoreCoord(x,y).
        max_args = 0
        core_coords = _get_core_coords_from_ranges(kernel.core_ranges)
        for core in core_coords:
            try:
                args = kernel.runtime_args[core.x][core.y]
                max_args = max(max_args, len(args))
            except (IndexError, KeyError):
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

    RuntimeArgsView uses coordinate-based 2D indexing: runtime_args[x][y]
    maps to CoreCoord(x, y). We must use actual core coordinates, not
    sequential indices.
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
        for col_idx, core in enumerate(core_coords):
            try:
                args = kernel.runtime_args[core.x][core.y]
                col_args[col_idx].extend(list(args))
            except (IndexError, KeyError):
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
        for field in (
            "fp32_dest_acc_en",
            "math_approx_mode",
            "math_fidelity",
            "dst_full_sync_en",
            "bfp8_pack_precise",
            "unpack_to_dest_mode",
        ):
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
        for fld in ("fp32_dest_acc_en", "math_approx_mode", "math_fidelity", "dst_full_sync_en", "bfp8_pack_precise"):
            base_val = getattr(base, fld, None)
            this_val = getattr(config, fld, None)
            if base_val != this_val:
                mismatches.append(f"  {fld}: phase {base_phase}={base_val}, phase {phase_idx}={this_val}")

        if mismatches:
            raise ValueError(f"Compute config mismatch for role {role_key}.\n" + "\n".join(mismatches))

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

        # Validate rectangular grid for safe NOC multicast
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


def _create_role_barrier_config(
    device: Any,
    role_core_ranges: Any,
    shared_config: BarrierConfig,
) -> BarrierConfig:
    """Create a role-specific barrier config sharing semaphore addresses.

    Uses the shared GlobalSemaphore L1 addresses but computes role-specific
    core counts and physical coordinates for the multicast barrier.
    """
    cfg = BarrierConfig()
    cfg.compute_done_addr = shared_config.compute_done_addr
    cfg.writer_done_addr = shared_config.writer_done_addr
    cfg.global_arrive_addr = shared_config.global_arrive_addr
    cfg.global_release_addr = shared_config.global_release_addr

    logical_coords = _get_core_coords_from_ranges(role_core_ranges)
    cfg.num_cores = len(logical_coords)

    if cfg.num_cores > 0:
        phys_coords = [device.worker_core_from_logical_core(c) for c in logical_coords]
        cfg.core0_phys_x = phys_coords[0].x
        cfg.core0_phys_y = phys_coords[0].y
        cfg.mcast_start_x = min(c.x for c in phys_coords)
        cfg.mcast_start_y = min(c.y for c in phys_coords)
        cfg.mcast_end_x = max(c.x for c in phys_coords)
        cfg.mcast_end_y = max(c.y for c in phys_coords)

        # Validate rectangular grid for safe NOC multicast
        if cfg.num_cores > 1:
            _validate_rectangular_grid(phys_coords, cfg)

    return cfg


def _compute_union_core_ranges(phases: List[PhaseInfo]) -> Any:
    """Compute the union of all core ranges across all kernels in phase 0.

    Returns a CoreRangeSet covering all cores used by any kernel.
    Handles overlapping core ranges by collecting individual cores and
    creating a bounding box CoreRange that covers all of them.
    """
    # Collect all unique logical core coordinates
    all_coords = set()
    for kernel_desc in phases[0].op_descriptor.descriptor.kernels:
        for cr in kernel_desc.core_ranges.ranges():
            for y in range(cr.start.y, cr.end.y + 1):
                for x in range(cr.start.x, cr.end.x + 1):
                    all_coords.add((x, y))

    if not all_coords:
        return phases[0].op_descriptor.descriptor.kernels[0].core_ranges

    # Create a bounding box CoreRange covering all cores
    min_x = min(x for x, y in all_coords)
    max_x = max(x for x, y in all_coords)
    min_y = min(y for x, y in all_coords)
    max_y = max(y for x, y in all_coords)

    bounding_range = ttnn.CoreRange(
        ttnn.CoreCoord(min_x, min_y),
        ttnn.CoreCoord(max_x, max_y),
    )
    return ttnn.CoreRangeSet([bounding_range])


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
        """Build the fused descriptor with two-level barrier sync.

        Dynamically discovers kernel roles from the ProgramDescriptor using
        (risc_type, core_ranges) as a unique key. This supports any op type
        (interleaved with 3 kernels, sharded with up to 7 kernels, etc.).
        """
        # Validate fp32 consistency
        _validate_fp32_consistency([p.op_descriptor for p in self.phases])

        # Discover all kernel roles from phase 0
        # Role key = (risc_type, frozenset of core range tuples)
        role_keys: List[Tuple[str, frozenset]] = []
        role_keys_set: Set[Tuple[str, frozenset]] = set()
        for kernel_desc in self.phases[0].op_descriptor.descriptor.kernels:
            rk = _get_role_key(kernel_desc)
            if rk not in role_keys_set:
                role_keys.append(rk)
                role_keys_set.add(rk)

        # Build phase_kernels as List[Dict[role_key, KernelDescriptor]]
        phase_kernels: List[Dict[Any, Any]] = []
        for phase_idx, phase in enumerate(self.phases):
            role_map: Dict[Any, Any] = {}
            for kernel_desc in phase.op_descriptor.descriptor.kernels:
                rk = _get_role_key(kernel_desc)
                role_map[rk] = kernel_desc
            phase_kernels.append(role_map)

        # Merge CB descriptors (max size per index, phase 0's buffer preferred)
        merged_cbs = _merge_cb_descriptors(self.phases)

        # Compute CB address rebinding info for buffer-backed CBs
        rebind_info = _compute_rebind_info(self.phases)

        # DEBUG: print rebind info
        if rebind_info:
            print(f"\n=== REBIND INFO ({len(self.phases)} phases) ===")
            for phase_idx, rebinds in sorted(rebind_info.items()):
                if rebinds:
                    for cb_idx, addr, size in rebinds:
                        print(f"  Phase {phase_idx}: CB[{cb_idx}] -> addr=0x{addr:x}, size={size}")
                else:
                    print(f"  Phase {phase_idx}: no rebinds")

        # Create barrier config with GlobalSemaphores on union of all core ranges
        union_core_ranges = _compute_union_core_ranges(self.phases)
        self._barrier_config = _create_barrier_config(device, union_core_ranges)

        # Compute sweep CB indices: ALL CBs with descriptors across phases (generic).
        valid_cb_indices = _get_all_cb_descriptor_indices(self.phases)
        sweep_cb_indices = sorted(valid_cb_indices)

        bc = self._barrier_config

        # Collect all unique op semaphore IDs (0-15) used by any phase.
        # These need to be reset to 0 between phases so each phase starts clean.
        op_semaphore_ids: List[int] = []
        seen_sem_ids_for_reset: Set[int] = set()
        for phase in self.phases:
            for sem in phase.op_descriptor.descriptor.semaphores:
                if sem.id not in seen_sem_ids_for_reset:
                    op_semaphore_ids.append(sem.id)
                    seen_sem_ids_for_reset.add(sem.id)
        op_semaphore_ids.sort()

        fused_kernels = []

        # For each discovered role: generate fused source, merge args, build descriptor
        for role_key in role_keys:
            risc_type, core_key = role_key

            # Get role-specific core_ranges from first available phase
            role_core_ranges = None
            for pk in phase_kernels:
                kernel = pk.get(role_key)
                if kernel is not None:
                    role_core_ranges = kernel.core_ranges
                    break

            if role_core_ranges is None:
                continue

            # Merge compile-time args and compute offsets
            ct_args, ct_offsets = _merge_compile_time_args(phase_kernels, role_key)
            rt_offsets = _compute_runtime_arg_offsets(phase_kernels, role_key)

            # Generate fused source and determine barrier addresses per RISC type
            # IMPORTANT: riscv_0 must use the GLOBAL barrier config (bc) so that
            # ALL riscv_0 cores across ALL roles synchronize via a single barrier.
            # Per-role barriers would cause sender/receiver readers to proceed
            # independently, leading to data races between phases.
            if risc_type == "riscv_0":
                fused_source = _generate_fused_riscv0_source(
                    phase_kernels,
                    role_key,
                    self.phases,
                    ct_offsets,
                    sweep_cb_indices,
                    bc,
                    rebind_info,
                    op_semaphore_ids=op_semaphore_ids,
                )
                barrier_addrs = [
                    bc.compute_done_addr,
                    bc.writer_done_addr,
                    bc.global_arrive_addr,
                    bc.global_release_addr,
                ]
            elif risc_type == "riscv_1":
                fused_source = _generate_fused_riscv1_source(
                    phase_kernels,
                    role_key,
                    self.phases,
                    ct_offsets,
                    sweep_cb_indices,
                    rebind_info,
                    bc,
                )
                barrier_addrs = [bc.writer_done_addr, bc.global_release_addr]
            elif risc_type == "compute":
                fused_source = _generate_fused_compute_source(
                    phase_kernels,
                    role_key,
                    self.phases,
                    ct_offsets,
                    sweep_cb_indices,
                    bc,
                    rebind_info,
                )
                barrier_addrs = [bc.compute_done_addr, bc.global_release_addr]
            else:
                continue

            if fused_source is None:
                continue

            # Concatenate runtime args and append barrier addresses
            rt_args = _concatenate_runtime_args(phase_kernels, role_key)
            rt_args, barrier_offset = _append_barrier_runtime_args(rt_args, barrier_addrs)

            # Merge named compile-time args (only riscv_0 gets full barrier config for global barrier)
            # Use global bc (not per-role) so all riscv_0 roles share one unified barrier
            barrier_cfg_for_named = bc if risc_type == "riscv_0" else None
            named_ct_args = _merge_named_compile_time_args(
                phase_kernels,
                role_key,
                rt_offsets,
                barrier_rt_offset=barrier_offset,
                barrier_config=barrier_cfg_for_named,
            )

            # Add rebind named compile-time args (addr + size for each CB that changes)
            for phase_idx, rebinds in rebind_info.items():
                for cb_idx, addr, size in rebinds:
                    prefix = f"phase{phase_idx}_cb{cb_idx}"
                    named_ct_args.append((f"{prefix}_rebind_addr", addr))
                    named_ct_args.append((f"{prefix}_rebind_size", size))

            # Get config from first available kernel for this role
            role_config = None
            for pk in phase_kernels:
                kernel = pk.get(role_key)
                if kernel is not None:
                    role_config = kernel.config
                    break

            # For compute roles, validate configs match across phases
            if risc_type == "compute":
                role_config = _validate_and_get_compute_config_for_role(phase_kernels, role_key)

            # Build fused kernel descriptor
            desc = ttnn.KernelDescriptor()
            desc.kernel_source = fused_source
            desc.source_type = ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            desc.core_ranges = role_core_ranges
            desc.compile_time_args = ct_args
            desc.named_compile_time_args = named_ct_args
            desc.defines = _merge_defines(phase_kernels, role_key)
            desc.runtime_args = rt_args
            desc.common_runtime_args = _concatenate_common_runtime_args(phase_kernels, role_key)
            desc.config = role_config
            fused_kernels.append(desc)

        # Merge semaphores (dedup by ID)
        all_semaphores = []
        seen_sem_ids: Set[int] = set()
        for phase in self.phases:
            for sem in phase.op_descriptor.descriptor.semaphores:
                if sem.id not in seen_sem_ids:
                    all_semaphores.append(sem)
                    seen_sem_ids.add(sem.id)

        # Collect input/output tensors (use id() for dedup because ttnn Tensor's
        # __eq__ returns an element-wise Tensor, making `in` unreliable)
        all_input_tensors = []
        seen_tensor_ids: Set[int] = set()
        for phase in self.phases:
            for tensor in phase.op_descriptor.input_tensors:
                tid = id(tensor)
                if tid not in seen_tensor_ids:
                    all_input_tensors.append(tensor)
                    seen_tensor_ids.add(tid)

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
