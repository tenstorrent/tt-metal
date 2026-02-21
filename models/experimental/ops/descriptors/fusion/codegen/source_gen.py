# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fused kernel source generation: source utilities, phase namespace generation,
and the unified fused source generator.
"""

import datetime
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.fusion.cb_allocator import PhaseInfo
from models.experimental.ops.descriptors.fusion.common import MultiBarrierSpec
from models.experimental.ops.descriptors.fusion.codegen.cpp_parser import (
    extract_kernel_body,
    inline_local_includes,
    collect_includes,
    collect_defines,
)
from models.experimental.ops.descriptors.fusion.codegen.barrier import (
    _generate_barrier_namespace,
)
from models.experimental.ops.descriptors.fusion.codegen.args import (
    _collect_phase_defines,
    _emit_define_lines,
    _emit_undef_lines,
)


# =============================================================================
# Module Constants
# =============================================================================

_SECTION_SEP = "// " + "=" * 76
_PROFILER_INCLUDE = '#include "tools/profiler/kernel_profiler.hpp"'
_ARRAY_INCLUDE = "#include <array>"


def _spdx_header() -> List[str]:
    """Return SPDX license header lines."""
    return [
        f"// SPDX-FileCopyrightText: \u00a9 {datetime.date.today().year} Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
    ]


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


_KERNEL_MAIN_SEARCH_RE = re.compile(r"\b(?:ALWI\s+)?void\s+kernel_main\s*\(")
_SKIP_LINE_PREFIXES = ("#include", "#define", "#pragma", "#undef")
_SKIP_COMMENT_SUBSTRINGS = ("SPDX-FileCopyrightText", "SPDX-License-Identifier")


def _extract_pre_main_text(source: str) -> str:
    """Extract pre-main code from source, excluding preprocessor directives.

    Returns everything before ``kernel_main()`` that is not a preprocessor
    directive line (``#include``, ``#define``, ``#pragma``, ``#undef``).
    Bare comment-only lines (just ``//``) are also stripped.
    """
    match = _KERNEL_MAIN_SEARCH_RE.search(source)
    pre_main_text = source[: match.start()] if match else source

    lines = []
    for line in pre_main_text.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _SKIP_LINE_PREFIXES):
            continue
        if any(sub in stripped for sub in _SKIP_COMMENT_SUBSTRINGS):
            continue
        # Skip bare comment lines (just "//" with no content)
        if stripped == "//":
            continue
        lines.append(line)

    # Collapse consecutive blank lines into a single blank line
    result = "\n".join(lines).strip()
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


def _dedent_ignoring_column_zero(text: str) -> str:
    """Dedent text ignoring preprocessor directives and comment lines at column 0."""
    lines = text.split("\n")

    def _is_non_code(stripped: str) -> bool:
        """Return True for lines that should not contribute to min indent."""
        # Preprocessor directives, comments, and block-comment continuations
        return stripped.startswith(("#", "//", "/*", "*"))

    # Find minimum indent of code lines
    min_indent = None
    for line in lines:
        stripped = line.lstrip()
        if not stripped or _is_non_code(stripped):
            continue
        indent = len(line) - len(stripped)
        if min_indent is None or indent < min_indent:
            min_indent = indent

    if not min_indent:
        return text

    # Strip min_indent spaces from indented lines, leave others untouched
    result = []
    for line in lines:
        if not line.strip():
            result.append(line)
        else:
            indent = len(line) - len(line.lstrip())
            if indent >= min_indent:
                result.append(line[min_indent:])
            else:
                result.append(line)
    return "\n".join(result)


def _extract_phase_pre_main(
    sources_with_indices: List[Tuple[int, str]],
    phase_headers: Dict[int, List[Tuple[str, str]]],
) -> Tuple[List[str], Dict[int, str]]:
    """Extract pre-main text: header content -> file scope, original source -> phase scope."""
    # File scope: deduplicated header content by resolved path
    header_path_seen: Set[str] = set()
    file_scope_blocks: List[str] = []
    for phase_idx, _ in sources_with_indices:
        for resolved_path, content in phase_headers.get(phase_idx, []):
            if resolved_path not in header_path_seen:
                header_path_seen.add(resolved_path)
                content_stripped = content.strip()
                if content_stripped:
                    file_scope_blocks.append(content_stripped)

    # Phase scope: original source pre-main minus preprocessor directives
    phase_pre_main: Dict[int, str] = {}
    for phase_idx, source in sources_with_indices:
        phase_pre_main[phase_idx] = _extract_pre_main_text(source)

    return file_scope_blocks, phase_pre_main


# =============================================================================
# Source Transformations for Phase N>0
# =============================================================================


def _prefix_named_args_in_source(source: str, phase_idx: int) -> str:
    """Replace get_named_compile_time_arg_val("X") with phase-prefixed version."""
    if phase_idx == 0:
        return source

    def replace_named_arg(match):
        name = match.group(1)
        return f'get_named_compile_time_arg_val("phase_{phase_idx}_{name}")'

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


def _emit_rt_arg_wrapper(phase_idx: int, rt_offset: int) -> List[str]:
    """Emit the RT arg wrapper function definition for a phase.

    Emitted once at file scope (before any #define redirect) so the
    wrapper body references the real ``get_arg_val``.  All phases
    (including phase 0) get a wrapper for uniform treatment.
    """
    wrapper_name = f"phase_{phase_idx}_get_arg_val"
    return [
        f"template <typename T>",
        f"FORCE_INLINE T {wrapper_name}(int arg_idx) {{",
        f"    return get_arg_val<T>(arg_idx + {rt_offset});",
        f"}}",
    ]


def _emit_rt_arg_define(phase_idx: int) -> str:
    """Emit #define to redirect get_arg_val to the phase wrapper."""
    return f"#define get_arg_val phase_{phase_idx}_get_arg_val"


def _emit_rt_arg_undef() -> str:
    """Emit #undef to restore get_arg_val after a phase."""
    return "#undef get_arg_val"


def _transform_phase_source(
    source: str,
    phase_idx: int,
    ct_arg_offset: int = 0,
) -> str:
    """Apply compile-time arg transformations to a phase's source.

    Transforms applied:
      1. Named CT arg prefixing (``get_named_compile_time_arg_val("X")``
         -> ``get_named_compile_time_arg_val("phase_N_X")``).
      2. Positional CT arg offsetting (``get_compile_time_arg_val(N)``
         -> ``get_compile_time_arg_val(N + offset)``).

    Runtime arg offsetting is handled by ``#define``/``#undef`` redirect
    of ``get_arg_val`` (see ``_emit_rt_arg_define``).  Name isolation
    is handled by C++ namespace wrapping — no prefixing needed.
    """
    source = _prefix_named_args_in_source(source, phase_idx)
    source = _offset_compile_time_args_in_source(source, phase_idx, ct_arg_offset)
    return source


def _generate_phase_namespace(
    phase_idx: int,
    pre_main: str,
    kernel_source: str,
    defines: List[Tuple[str, str]],
    ct_arg_offset: int,
    phase_name: str = "",
) -> List[str]:
    """Generate ``#define`` -> ``namespace phase_N { pre_main + void run() { body } }`` -> ``#undef``.

    Extracts the original ``kernel_main()`` body into ``run()`` and applies
    compile-time arg transformations (positional offsetting + named prefixing).
    """
    ns_name = f"phase_{phase_idx}"
    lines: List[str] = []

    label = f"Phase {phase_idx}: {phase_name}" if phase_name else f"Phase {phase_idx}"
    lines.append("// " + "=" * 76)
    lines.append(f"// {label}")
    lines.append("// " + "=" * 76)

    # Per-phase defines (outside namespace — preprocessor is namespace-unaware)
    if defines:
        lines.extend(_emit_define_lines(defines))

    # RT arg redirect (all phases, including phase 0)
    lines.append(_emit_rt_arg_define(phase_idx))

    # Open namespace
    lines.append(f"namespace {ns_name} {{")
    lines.append("")

    # Pre-main code (transformed for CT arg offsets + named arg prefixes)
    if pre_main.strip():
        transformed_pre_main = _transform_phase_source(pre_main, phase_idx, ct_arg_offset)
        lines.append(_dedent_ignoring_column_zero(transformed_pre_main).strip())
        lines.append("")

    # Transform the kernel body source (CT arg offsets + named arg prefixes)
    body = extract_kernel_body(kernel_source)
    transformed = _transform_phase_source(body, phase_idx, ct_arg_offset)
    dedented = _dedent_ignoring_column_zero(transformed)

    lines.append("void run() {")
    if phase_name:
        lines.append(f'    DeviceZoneScopedN("{phase_name}");')
    for line in dedented.split("\n"):
        if line.strip():
            lines.append(f"    {line}")
        else:
            lines.append("")
    lines.append("}")

    # Close namespace
    lines.append("")
    lines.append(f"}} // namespace {ns_name}")

    # Undef RT arg redirect
    lines.append(_emit_rt_arg_undef())

    # Undef per-phase defines
    if defines:
        lines.extend(_emit_undef_lines(defines))

    lines.append("")
    return lines


# =============================================================================
# Fused Kernel Source Generation
# =============================================================================


def _generate_fused_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    per_phase_cb_slots: List[List[int]],
    risc_type: str,
    role_label: str,
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    op_semaphore_info: Optional[List[Tuple[int, int]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused kernel source for any RISC type.

    Each phase's source goes into ``namespace phase_N { ... }`` with its
    body in ``run()``.  The outer ``kernel_main()`` calls
    ``phase_N::run()`` with barrier wait/reset between phases.
    """
    # Read and inline kernel sources for this role
    role_sources: List[Tuple[int, str]] = []
    phase_headers: Dict[int, List[Tuple[str, str]]] = {}

    for i, pk in enumerate(phase_kernels):
        kernel = pk.get(role_key)
        if kernel is None:
            continue
        source, kernel_dir = _read_kernel_source(kernel)
        if not source:
            continue
        headers, source = inline_local_includes(source, kernel_dir)
        phase_headers[i] = headers
        role_sources.append((i, source))

    if not role_sources:
        return None

    # Collect includes, defines, pre-main blocks
    all_combined = ["\n".join(c for _, c in phase_headers.get(i, [])) + "\n" + s for i, s in role_sources]
    includes = collect_includes(all_combined)
    source_defines = collect_defines(all_combined)
    must_match_defines, per_phase_defines = _collect_phase_defines(phase_kernels, role_key)
    file_scope_blocks, pre_mains = _extract_phase_pre_main(role_sources, phase_headers)

    # File preamble
    lines = _spdx_header() + [
        "",
        f"// Auto-generated fused {role_label} kernel - {len(role_sources)} phases",
        "",
    ]

    # File-scope: MUST_MATCH defines + source defines + includes
    lines.extend(_emit_define_lines(must_match_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append(_PROFILER_INCLUDE)
    lines.append(_ARRAY_INCLUDE)
    lines.append("")

    phase_names = {p.phase_idx: p.op_descriptor.name for p in phases}

    # File-scope: namespace blocks from inlined headers
    if file_scope_blocks:
        lines.extend([_SECTION_SEP, "// Inlined headers", _SECTION_SEP, ""])
        for block in file_scope_blocks:
            lines.append(block)
            lines.append("")
        lines.extend([_SECTION_SEP, "// End inlined headers", _SECTION_SEP, ""])

    # RT arg wrappers at file scope
    if rt_arg_offsets:
        for phase_idx, _ in role_sources:
            if phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    # Phase namespaces
    for phase_idx, raw_source in role_sources:
        pre_main = pre_mains.get(phase_idx, "")
        defines = per_phase_defines.get(phase_idx, [])
        ct_offset = ct_arg_offsets.get(phase_idx, 0)
        lines.extend(
            _generate_phase_namespace(
                phase_idx,
                pre_main,
                raw_source,
                defines,
                ct_offset,
                phase_name=phase_names.get(phase_idx, ""),
            )
        )

    needs_barrier = multi_barrier is not None and len(multi_barrier.transition_map) > 0

    # Barrier namespace
    if needs_barrier:
        lines.extend(
            _generate_barrier_namespace(
                risc_type,
                multi_barrier,
                rebind_info or {},
                role_sources,
                per_phase_cb_slots,
                op_semaphore_info=op_semaphore_info,
            )
        )

    # kernel_main
    lines.append("void kernel_main() {")
    if needs_barrier:
        lines.append("    barrier::init();")
        lines.append("")

    has_trailing = False
    if needs_barrier:
        last_phase_idx = role_sources[-1][0]
        has_trailing = last_phase_idx in multi_barrier.transition_map

    for count, (phase_idx, _) in enumerate(role_sources):
        pname = phase_names.get(phase_idx, "")
        label = f"Phase {phase_idx}: {pname}" if pname else f"Phase {phase_idx}"
        lines.append(f"    // {label}")
        lines.append(f"    phase_{phase_idx}::run();")
        is_last = count == len(role_sources) - 1
        if needs_barrier and (not is_last or has_trailing):
            lines.append("    barrier::phase::wait();")
            lines.append("    barrier::phase::reset();")
            if not is_last:
                lines.append("")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)
