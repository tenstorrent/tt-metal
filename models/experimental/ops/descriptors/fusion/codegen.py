# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
C++ Parsing and Source Code Generation for Kernel Fusion.

Combines C++ kernel source parsing (body extraction, include inlining) with
fused kernel source generation (phase namespaces, barrier infrastructure,
compile-time and runtime arg management).
"""

import os
import re
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import ttnn

from models.experimental.ops.descriptors.op_descriptor import OpDescriptor
from models.experimental.ops.descriptors.fusion.common import (
    BarrierConfig,
    MultiBarrierSpec,
    _BuildResult,
    _get_role_key,
)
from models.experimental.ops.descriptors.fusion.cb_allocator import (
    PhaseInfo,
    CBPoolAllocator,
    extract_cb_info,
    _is_cb_named_arg,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Kernel Body Extraction (from cpp_parser.py)
# =============================================================================

# Matches `void kernel_main() {` with optional ALWI prefix
_KERNEL_MAIN_RE = re.compile(r"\b(?:ALWI\s+)?void\s+kernel_main\s*\(\s*\)\s*\{")


def _is_raw_string_prefix(source: str, quote_pos: int) -> bool:
    """Check if the ``"`` at *quote_pos* opens a C++ raw string literal.

    Raw strings are: ``R"delim(...)delim"`` with optional encoding prefix
    ``L``, ``u``, ``U``, or ``u8`` before ``R``.  The character before the
    entire prefix must not be an identifier character, otherwise ``R`` is
    just part of an identifier like ``myR"..."``.
    """
    # Character immediately before " must be R
    r = quote_pos - 1
    if r < 0 or source[r] != "R":
        return False

    # Check for optional encoding prefix before R: L, u, U, u8
    # Start of the full prefix token (inclusive)
    start = r
    before_r = r - 1
    if before_r >= 0:
        ch = source[before_r]
        if ch in "LU":
            start = before_r  # LR" or UR"
        elif ch == "u":
            start = before_r  # uR"
        elif ch == "8" and before_r >= 1 and source[before_r - 1] == "u":
            start = before_r - 1  # u8R"

    # The character before the prefix must not be an identifier char,
    # otherwise this is something like `someVarR"..."` not a raw string.
    before_prefix = start - 1
    if before_prefix >= 0 and (source[before_prefix].isalnum() or source[before_prefix] == "_"):
        return False
    return True


def _skip_raw_string(source: str, quote_pos: int) -> int:
    """Skip past a raw string literal starting at the ``"`` at *quote_pos*.

    Raw string syntax: ``R"delim(content)delim"`` where *delim* can be
    empty or up to 16 characters (no spaces, backslashes, or parens).

    Returns the index just past the closing ``"``, or ``len(source)`` if
    unterminated (malformed source — treat the rest as consumed).
    """
    n = len(source)
    # Find the opening '(' that ends the delimiter
    paren_pos = source.find("(", quote_pos + 1)
    if paren_pos == -1 or paren_pos - (quote_pos + 1) > 16:
        # Not a valid raw string — fall back to treating as regular string
        return _skip_regular_string(source, quote_pos)
    delim = source[quote_pos + 1 : paren_pos]
    # Find the closing sequence: )delim"
    close_seq = f'){delim}"'
    end = source.find(close_seq, paren_pos + 1)
    if end == -1:
        return n  # unterminated — consume rest
    return end + len(close_seq)


def _skip_regular_string(source: str, quote_pos: int) -> int:
    """Skip past a regular string literal starting at ``"`` at *quote_pos*.

    Handles ``\\"`` escapes.  Returns the index just past the closing
    ``"``, or ``len(source)`` if unterminated.
    """
    i = quote_pos + 1
    n = len(source)
    while i < n:
        if source[i] == "\\":
            i += 2  # skip escape sequence
            continue
        if source[i] == '"':
            return i + 1  # past the closing quote
        i += 1
    return n  # unterminated


def _skip_char_literal(source: str, quote_pos: int) -> int:
    """Skip past a character literal starting at ``'`` at *quote_pos*.

    Handles ``\\'`` escapes.  Returns the index just past the closing
    ``'``, or ``len(source)`` if unterminated.
    """
    i = quote_pos + 1
    n = len(source)
    while i < n:
        if source[i] == "\\":
            i += 2
            continue
        if source[i] == "'":
            return i + 1
        i += 1
    return n


def _find_matching_brace(source: str, open_pos: int) -> Optional[int]:
    """Find the closing brace matching the opening brace at *open_pos*.

    Full C++ lexer-level scanner that tracks brace depth starting at 1,
    correctly skipping braces inside:

    - Line comments (``//`` to end of line)
    - Block comments (``/* ... */``)
    - Regular string literals (``"..."`` with backslash escapes)
    - Raw string literals (``R"delim(...)delim"`` and prefixed variants
      ``LR"``, ``uR"``, ``UR"``, ``u8R"``)
    - Character literals (``'...'`` with backslash escapes)

    Returns the index of the closing ``}`` or ``None`` if not found.
    """
    depth = 1
    i = open_pos + 1
    n = len(source)

    while i < n:
        c = source[i]

        # Line comment: // to end of line
        if c == "/" and i + 1 < n and source[i + 1] == "/":
            end = source.find("\n", i + 2)
            i = n if end == -1 else end + 1
            continue

        # Block comment: /* ... */
        if c == "/" and i + 1 < n and source[i + 1] == "*":
            end = source.find("*/", i + 2)
            if end == -1:
                return None  # unterminated block comment
            i = end + 2
            continue

        # String literal (regular or raw)
        if c == '"':
            if _is_raw_string_prefix(source, i):
                i = _skip_raw_string(source, i)
            else:
                i = _skip_regular_string(source, i)
            continue

        # Character literal
        if c == "'":
            i = _skip_char_literal(source, i)
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return None


def extract_kernel_body(source: str) -> str:
    """Extract the body of ``kernel_main()`` using regex + brace matching.

    Returns the inner body (without outer braces) of the kernel_main
    function definition.  Returns empty string if not found.
    """
    match = _KERNEL_MAIN_RE.search(source)
    if not match:
        return ""
    # The '{' is the last character of the match
    open_brace_pos = match.end() - 1
    close_pos = _find_matching_brace(source, open_brace_pos)
    if close_pos is None:
        return ""
    return source[open_brace_pos + 1 : close_pos]


# =============================================================================
# Include Inlining (from cpp_parser.py)
# =============================================================================


def inline_local_includes(source: str, kernel_dir: Optional[str]) -> Tuple[List[Tuple[str, str]], str]:
    """Inline local includes, returning header content separately.

    For generated SOURCE_CODE kernels, local includes won't resolve because
    the compiler doesn't know the original directory.  This function reads
    them and returns their content separately so callers can place header
    content at file scope while keeping original source in phase namespaces.

    Supports both local-only includes (no path separator) and relative path
    includes (e.g. ``"subdir/header.h"``), resolving them relative to
    *kernel_dir*.

    Returns:
        ``(headers, remaining_source)`` where *headers* is a list of
        ``(resolved_path, content)`` tuples for each inlined local header,
        and *remaining_source* is the original source with those
        ``#include "..."`` lines removed.
    """
    if kernel_dir is None:
        return [], source

    lines = source.split("\n")
    result: List[str] = []
    headers: List[Tuple[str, str]] = []
    inlined: Set[str] = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#include "'):
            match = re.match(r'#include\s+"([^"]+)"', stripped)
            if match:
                inc_path = match.group(1)
                if inc_path not in inlined:
                    full_inc = os.path.normpath(os.path.join(kernel_dir, inc_path))
                    if os.path.exists(full_inc):
                        with open(full_inc, "r") as f:
                            inc_content = f.read()
                        # Strip #pragma once and nested local includes
                        header_lines: List[str] = []
                        for inc_line in inc_content.split("\n"):
                            stripped_inc = inc_line.strip()
                            if stripped_inc.startswith("#pragma once"):
                                continue
                            if stripped_inc.startswith('#include "'):
                                nested_match = re.match(r'#include\s+"([^"]+)"', stripped_inc)
                                if nested_match:
                                    nested = nested_match.group(1)
                                    nested_full = os.path.normpath(os.path.join(os.path.dirname(full_inc), nested))
                                    # Only skip nested includes that resolve to
                                    # local files.  Non-existent paths are kept
                                    # as-is — they may be system/SDK includes.
                                    if os.path.exists(nested_full):
                                        continue
                            header_lines.append(inc_line)
                        headers.append((full_inc, "\n".join(header_lines)))
                        inlined.add(inc_path)
                        continue  # Remove the #include line from remaining source
        result.append(line)

    return headers, "\n".join(result)


# =============================================================================
# Collection Helpers (from cpp_parser.py)
# =============================================================================


def collect_includes(sources: List[str]) -> List[str]:
    """Collect unique #include lines from multiple source strings."""
    includes = set()
    for source in sources:
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#include"):
                includes.add(stripped)
    return sorted(includes)


def collect_defines(sources: List[str]) -> List[str]:
    """Collect unique #define lines from multiple source strings (before kernel_main)."""
    defines: List[str] = []
    seen: Set[str] = set()
    for source in sources:
        # Find where kernel_main starts (line number)
        match = _KERNEL_MAIN_RE.search(source)
        kernel_main_line = None
        if match:
            kernel_main_line = source[: match.start()].count("\n")

        for line_no, line in enumerate(source.split("\n")):
            if kernel_main_line is not None and line_no >= kernel_main_line:
                break
            stripped = line.strip()
            if stripped.startswith("#define") and stripped not in seen:
                defines.append(line)
                seen.add(stripped)
    return defines


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


_KERNEL_MAIN_SEARCH_RE = re.compile(r"\bvoid\s+kernel_main\s*\(")
_SKIP_LINE_PREFIXES = ("#include", "#define", "#pragma", "#undef")


def _extract_pre_main_text(source: str) -> str:
    """Extract pre-main code from source, excluding preprocessor directives.

    Returns everything before ``kernel_main()`` that is not a preprocessor
    directive line (``#include``, ``#define``, ``#pragma``, ``#undef``).
    """
    match = _KERNEL_MAIN_SEARCH_RE.search(source)
    pre_main_text = source[: match.start()] if match else source

    lines = []
    for line in pre_main_text.split("\n"):
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _SKIP_LINE_PREFIXES):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


def _extract_phase_pre_main(
    sources_with_indices: List[Tuple[int, str]],
    phase_headers: Dict[int, List[Tuple[str, str]]],
) -> Tuple[List[str], Dict[int, str]]:
    """Extract pre-main text for each phase, split by scope.

    Content from inlined headers goes to **file scope** (deduplicated by
    resolved path across phases).  Original source pre-main goes to **phase
    scope** inside ``namespace phase_N``.

    Returns:
        file_scope_blocks: Deduplicated header content for file scope.
        phase_pre_main: Per-phase pre-main text from original source.
    """
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
         → ``get_named_compile_time_arg_val("phase_N_X")``).
      2. Positional CT arg offsetting (``get_compile_time_arg_val(N)``
         → ``get_compile_time_arg_val(N + offset)``).

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
    """Generate a complete namespace block for one phase.

    All phases (including phase 0) are treated uniformly.  Emits::

        #define ... (per-phase defines)
        #define get_arg_val phase_N_get_arg_val
        namespace phase_N {
            <pre_main transformed>
            void run() {
                DeviceZoneScopedN("op_name");  // if phase_name set
                <transformed kernel body>
            }
        } // namespace phase_N
        #undef get_arg_val
        #undef ... (per-phase defines)

    The original ``kernel_main()`` body is extracted and placed in
    ``run()`` to avoid the JIT build system's multiple-kernel_main
    check.  The outer ``kernel_main()`` calls ``phase_N::run()``.
    Both pre-main code and kernel body receive compile-time arg
    transformations (positional offsetting + named arg prefixing).
    Name isolation is handled by the C++ namespace.
    """
    ns_name = f"phase_{phase_idx}"
    lines: List[str] = []

    label = f"Phase {phase_idx}: {phase_name}" if phase_name else f"Phase {phase_idx}"
    lines.append(f"// ==== {label} ====")

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
        lines.append(transformed_pre_main)
        lines.append("")

    # Transform the kernel body source (CT arg offsets + named arg prefixes)
    body = extract_kernel_body(kernel_source)
    transformed = _transform_phase_source(body, phase_idx, ct_arg_offset)

    lines.append("void run() {")
    if phase_name:
        lines.append(f'    DeviceZoneScopedN("{phase_name}");')
    for line in transformed.split("\n"):
        lines.append(f"    {line}")
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
# Barrier Infrastructure
# =============================================================================


def _build_barrier_dispatch(
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
) -> List[Dict[str, Any]]:
    """Build dispatch table for barrier transitions.

    Each entry maps a ``done`` counter value to the segment and rebinds
    needed for that transition.  The ``done`` counter is incremented by
    ``barrier::phase::wait()`` before ``barrier::phase::reset()`` runs.

    Returns list of dicts with keys:
        done_val: value of ``done`` when this transition fires
        seg_idx: which segment to sync
        next_phase_idx: phase whose rebinds to apply (None for trailing)
        rebinds: list of (slot_idx, addr, size) for CB rebinding
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
    """Generate C++ rebind code for use inside barrier::phase::reset().

    Args:
        rebinds: List of (slot_idx, addr, size) tuples.
        next_phase_idx: Phase whose CTA names to use for rebind addresses.
        indent: Indentation prefix for each generated line.
        for_compute: If True, apply ``>> 4`` shift for TRISC addresses.
    """
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


def _generate_barrier_namespace_riscv0(
    sweep_cb_indices: List[int],
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    op_semaphore_info: List[Tuple[int, int]],
    sources: List[Tuple[int, str]],
) -> List[str]:
    """Generate ``namespace barrier {{ }}`` for RISCV0 (reader/BRISC).

    BRISC coordinates the barrier:
      - ``phase::wait()``: noc_async_full_barrier, wait for compute+writer done
      - ``phase::reset()``: reset CBs, reset op sems, rebind, segment sync
      - ``segment_N::sync()``: multicast arrive/release barrier across cores
    """
    lines: List[str] = []
    num_segments = len(multi_barrier.segments)
    dispatch = _build_barrier_dispatch(multi_barrier, rebind_info, sources)

    lines.append("// ---- Barrier infrastructure ----")
    lines.append("namespace barrier {")
    lines.append("")
    lines.append('constexpr uint32_t rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
    lines.append("")

    # 1. reset_cbs() — BRISC-side CB reset
    lines.append("// BRISC-side CB reset: equalize stream registers + reset pointers to CB start.")
    lines.append("FORCE_INLINE void reset_cbs() {")
    for cb_idx in sweep_cb_indices:
        lines.append(f"    {{")
        lines.append(f"        uint16_t remaining = (uint16_t)(*get_cb_tiles_received_ptr({cb_idx}))")
        lines.append(f"                          - (uint16_t)(*get_cb_tiles_acked_ptr({cb_idx}));")
        lines.append(f"        volatile tt_reg_ptr uint32_t* acked_ptr = (volatile tt_reg_ptr uint32_t*)")
        lines.append(f"            ((uint32_t)(uintptr_t)get_cb_tiles_acked_ptr({cb_idx}));")
        lines.append(f"        for (uint16_t i = 0; i < remaining; i++) {{")
        lines.append(f"            acked_ptr[0] += 1;")
        lines.append(f"        }}")
        lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
        lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
        lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
        lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
        lines.append(f"    }}")
    lines.append("}")
    lines.append("")

    # 2. Segment namespaces (multicast barrier)
    # RT arg layout: [compute_done, writer_done, seg0_arrive, seg0_release, seg1_arrive, ...]
    for seg_idx in range(num_segments):
        s = f"seg{seg_idx}"
        arrive_offset = 2 + seg_idx * 2
        release_offset = 3 + seg_idx * 2
        lines.append(f"namespace segment_{seg_idx} {{")
        lines.append(f'    constexpr uint32_t num_cores = get_named_compile_time_arg_val("{s}_num_cores");')
        lines.append(f'    constexpr uint32_t core0_phys_x = get_named_compile_time_arg_val("{s}_core0_phys_x");')
        lines.append(f'    constexpr uint32_t core0_phys_y = get_named_compile_time_arg_val("{s}_core0_phys_y");')
        lines.append(f'    constexpr uint32_t mcast_start_x = get_named_compile_time_arg_val("{s}_mcast_start_x");')
        lines.append(f'    constexpr uint32_t mcast_start_y = get_named_compile_time_arg_val("{s}_mcast_start_y");')
        lines.append(f'    constexpr uint32_t mcast_end_x = get_named_compile_time_arg_val("{s}_mcast_end_x");')
        lines.append(f'    constexpr uint32_t mcast_end_y = get_named_compile_time_arg_val("{s}_mcast_end_y");')
        lines.append(f"    uint32_t call_count;")
        lines.append(f"    volatile tt_l1_ptr uint32_t* arrive;")
        lines.append(f"    volatile tt_l1_ptr uint32_t* release;")
        lines.append(f"")
        lines.append(f"    FORCE_INLINE void init() {{")
        lines.append(f"        call_count = 0;")
        lines.append(f"        arrive = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append(f"            get_arg_val<uint32_t>(rt_offset + {arrive_offset}));")
        lines.append(f"        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append(f"            get_arg_val<uint32_t>(rt_offset + {release_offset}));")
        lines.append(f"    }}")
        lines.append(f"")
        lines.append(f"    FORCE_INLINE void sync() {{")
        lines.append(f"        if constexpr (num_cores > 1) {{")
        lines.append(
            f"            uint64_t core0_arrive_noc_addr = get_noc_addr(core0_phys_x, core0_phys_y, (uint32_t)arrive);"
        )
        lines.append(f"            noc_semaphore_inc(core0_arrive_noc_addr, 1);")
        lines.append(f"            bool is_core_0 = (my_x[0] == core0_phys_x && my_y[0] == core0_phys_y);")
        lines.append(f"            if (is_core_0) {{")
        lines.append(f"                noc_semaphore_wait_min(arrive, num_cores * (call_count + 1));")
        lines.append(f"                *release = call_count + 1;")
        lines.append(f"                uint64_t mcast_addr = get_noc_multicast_addr(")
        lines.append(f"                    mcast_start_x, mcast_start_y, mcast_end_x, mcast_end_y, (uint32_t)release);")
        lines.append(
            f"                noc_semaphore_set_multicast_loopback_src((uint32_t)release, mcast_addr, num_cores);"
        )
        lines.append(f"                noc_async_write_barrier();")
        lines.append(f"            }} else {{")
        lines.append(f"                noc_semaphore_wait_min(release, call_count + 1);")
        lines.append(f"            }}")
        lines.append(f"        }} else {{")
        lines.append(f"            *release = call_count + 1;")
        lines.append(f"        }}")
        lines.append(f"        call_count++;")
        lines.append(f"    }}")
        lines.append(f"}} // namespace segment_{seg_idx}")
        lines.append("")

    # 3. Phase namespace (wait + reset)
    lines.append("namespace phase {")
    lines.append("    uint32_t done;")
    lines.append("    volatile tt_l1_ptr uint32_t* compute_done;")
    lines.append("    volatile tt_l1_ptr uint32_t* writer_done;")
    lines.append("")
    lines.append("    FORCE_INLINE void wait() {")
    lines.append("        done++;")
    lines.append("        noc_async_full_barrier();")
    lines.append("        noc_semaphore_wait_min(compute_done, done);")
    lines.append("        noc_semaphore_wait_min(writer_done, done);")
    lines.append("    }")
    lines.append("")
    lines.append("    FORCE_INLINE void reset() {")
    lines.append("        reset_cbs();")
    # Op semaphore reset (all transitions)
    if op_semaphore_info:
        lines.append("        // Reset op semaphores")
        for sem_id, initial_value in op_semaphore_info:
            lines.append(
                f"        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore({sem_id})) = {initial_value};"
            )
    # Dispatch: rebind + segment sync
    for entry in dispatch:
        done_val = entry["done_val"]
        seg_idx = entry["seg_idx"]
        rebinds = entry["rebinds"]
        next_phase_idx = entry["next_phase_idx"]
        lines.append(f"        if (done == {done_val}) {{")
        if rebinds and next_phase_idx is not None:
            lines.extend(_generate_rebind_lines(rebinds, next_phase_idx, "            "))
        lines.append(f"            segment_{seg_idx}::sync();")
        lines.append(f"        }}")
    lines.append("    }")
    lines.append("} // namespace phase")
    lines.append("")

    # 4. init()
    lines.append("FORCE_INLINE void init() {")
    lines.append("    phase::done = 0;")
    lines.append("    phase::compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
    lines.append("        get_arg_val<uint32_t>(rt_offset));")
    lines.append("    phase::writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
    lines.append("        get_arg_val<uint32_t>(rt_offset + 1));")
    for seg_idx in range(num_segments):
        lines.append(f"    segment_{seg_idx}::init();")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace barrier")
    lines.append("")
    return lines


def _generate_barrier_namespace_riscv1(
    sweep_cb_indices: List[int],
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
) -> List[str]:
    """Generate ``namespace barrier {{ }}`` for RISCV1 (writer/NCRISC).

    Writer signals done and spins on release:
      - ``phase::wait()``: noc_async_write_barrier, signal writer_done
      - ``phase::reset()``: segment sync (spin on release), resync CBs, rebind
      - ``segment_N::sync()``: spin-wait on release semaphore
    """
    lines: List[str] = []
    num_segments = len(multi_barrier.segments)
    dispatch = _build_barrier_dispatch(multi_barrier, rebind_info, sources)

    lines.append("// ---- Barrier infrastructure ----")
    lines.append("namespace barrier {")
    lines.append("")
    lines.append('constexpr uint32_t rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
    lines.append("")

    # 1. resync_cbs() — NCRISC CB pointer reset
    if sweep_cb_indices:
        lines.append("// Resync NCRISC local CB pointers to CB start between phases.")
        lines.append("FORCE_INLINE void resync_cbs() {")
        for cb_idx in sweep_cb_indices:
            lines.append(f"    {{")
            lines.append(f"        uint32_t fifo_start = get_local_cb_interface({cb_idx}).fifo_limit")
            lines.append(f"                            - get_local_cb_interface({cb_idx}).fifo_size;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_rd_ptr = fifo_start;")
            lines.append(f"        get_local_cb_interface({cb_idx}).fifo_wr_ptr = fifo_start;")
            lines.append(f"    }}")
        lines.append("}")
        lines.append("")

    # 2. Segment namespaces (spin-wait on release)
    # RT arg layout: [writer_done, seg0_release, seg1_release, ...]
    for seg_idx in range(num_segments):
        release_offset = 1 + seg_idx
        lines.append(f"namespace segment_{seg_idx} {{")
        lines.append(f"    uint32_t call_count;")
        lines.append(f"    volatile tt_l1_ptr uint32_t* release;")
        lines.append(f"")
        lines.append(f"    FORCE_INLINE void init() {{")
        lines.append(f"        call_count = 0;")
        lines.append(f"        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append(f"            get_arg_val<uint32_t>(rt_offset + {release_offset}));")
        lines.append(f"    }}")
        lines.append(f"")
        lines.append(f"    FORCE_INLINE void sync() {{")
        lines.append(f"        while (*release < call_count + 1) {{ }}")
        lines.append(f"        call_count++;")
        lines.append(f"    }}")
        lines.append(f"}} // namespace segment_{seg_idx}")
        lines.append("")

    # 3. Phase namespace
    lines.append("namespace phase {")
    lines.append("    uint32_t done;")
    lines.append("    volatile tt_l1_ptr uint32_t* writer_done;")
    lines.append("")
    lines.append("    FORCE_INLINE void wait() {")
    lines.append("        done++;")
    lines.append("        noc_async_write_barrier();")
    lines.append("        *writer_done = done;")
    lines.append("    }")
    lines.append("")
    lines.append("    FORCE_INLINE void reset() {")
    # Segment sync dispatch
    for entry in dispatch:
        lines.append(f"        if (done == {entry['done_val']}) {{")
        lines.append(f"            segment_{entry['seg_idx']}::sync();")
        lines.append(f"        }}")
    # Resync CBs
    if sweep_cb_indices:
        lines.append("        resync_cbs();")
    # Rebind dispatch
    has_rebinds = any(entry["rebinds"] for entry in dispatch)
    if has_rebinds:
        for entry in dispatch:
            if entry["rebinds"]:
                lines.append(f"        if (done == {entry['done_val']}) {{")
                lines.extend(_generate_rebind_lines(entry["rebinds"], entry["next_phase_idx"], "            "))
                lines.append(f"        }}")
    lines.append("    }")
    lines.append("} // namespace phase")
    lines.append("")

    # 4. init()
    lines.append("FORCE_INLINE void init() {")
    lines.append("    phase::done = 0;")
    lines.append("    phase::writer_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
    lines.append("        get_arg_val<uint32_t>(rt_offset));")
    for seg_idx in range(num_segments):
        lines.append(f"    segment_{seg_idx}::init();")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace barrier")
    lines.append("")
    return lines


def _generate_barrier_namespace_compute(
    sweep_cb_indices: List[int],
    multi_barrier: MultiBarrierSpec,
    rebind_info: Dict[int, List[Tuple[int, int, int]]],
    sources: List[Tuple[int, str]],
) -> List[str]:
    """Generate ``namespace barrier {{ }}`` for compute (TRISC0/TRISC2).

    Compute signals done and spins on release:
      - ``phase::wait()``: signal compute_done
      - ``phase::reset()``: segment sync (spin on release), resync CBs, rebind (>> 4)
      - ``segment_N::sync()``: spin-wait on release semaphore
    """
    lines: List[str] = []
    num_segments = len(multi_barrier.segments)
    dispatch = _build_barrier_dispatch(multi_barrier, rebind_info, sources)

    lines.append("// ---- Barrier infrastructure ----")
    lines.append("namespace barrier {")
    lines.append("")
    lines.append('constexpr uint32_t rt_offset = get_named_compile_time_arg_val("barrier_rt_offset");')
    lines.append("")

    # 1. resync_cbs() — compute-side CB state resync
    if sweep_cb_indices:
        lines.append("// Resync compute-side local CB state after BRISC reset.")
        lines.append("// TRISC0: sync tiles_acked + reset fifo_rd_ptr to CB start.")
        lines.append("// TRISC2: sync tiles_received + reset fifo_wr_ptr to CB start.")
        lines.append("FORCE_INLINE void resync_cbs() {")
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

    # 2. Segment namespaces (spin-wait on release)
    # RT arg layout: [compute_done, seg0_release, seg1_release, ...]
    for seg_idx in range(num_segments):
        release_offset = 1 + seg_idx
        lines.append(f"namespace segment_{seg_idx} {{")
        lines.append(f"    uint32_t call_count;")
        lines.append(f"    volatile tt_l1_ptr uint32_t* release;")
        lines.append(f"")
        lines.append(f"    FORCE_INLINE void init() {{")
        lines.append(f"        call_count = 0;")
        lines.append(f"        release = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
        lines.append(f"            get_arg_val<uint32_t>(rt_offset + {release_offset}));")
        lines.append(f"    }}")
        lines.append(f"")
        lines.append(f"    FORCE_INLINE void sync() {{")
        lines.append(f"        while (*release < call_count + 1) {{ }}")
        lines.append(f"        call_count++;")
        lines.append(f"    }}")
        lines.append(f"}} // namespace segment_{seg_idx}")
        lines.append("")

    # 3. Phase namespace
    lines.append("namespace phase {")
    lines.append("    uint32_t done;")
    lines.append("    volatile tt_l1_ptr uint32_t* compute_done;")
    lines.append("")
    lines.append("    FORCE_INLINE void wait() {")
    lines.append("        done++;")
    lines.append("        *compute_done = done;")
    lines.append("    }")
    lines.append("")
    lines.append("    FORCE_INLINE void reset() {")
    # Segment sync dispatch
    for entry in dispatch:
        lines.append(f"        if (done == {entry['done_val']}) {{")
        lines.append(f"            segment_{entry['seg_idx']}::sync();")
        lines.append(f"        }}")
    # Resync CBs
    if sweep_cb_indices:
        lines.append("        resync_cbs();")
    # Rebind dispatch (with >> 4 shift for compute)
    has_rebinds = any(entry["rebinds"] for entry in dispatch)
    if has_rebinds:
        lines.append("#ifndef TRISC_MATH")
        for entry in dispatch:
            if entry["rebinds"]:
                lines.append(f"        if (done == {entry['done_val']}) {{")
                lines.extend(
                    _generate_rebind_lines(entry["rebinds"], entry["next_phase_idx"], "            ", for_compute=True)
                )
                lines.append(f"        }}")
        lines.append("#endif")
    lines.append("    }")
    lines.append("} // namespace phase")
    lines.append("")

    # 4. init()
    lines.append("FORCE_INLINE void init() {")
    lines.append("    phase::done = 0;")
    lines.append("    phase::compute_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(")
    lines.append("        get_arg_val<uint32_t>(rt_offset));")
    for seg_idx in range(num_segments):
        lines.append(f"    segment_{seg_idx}::init();")
    lines.append("}")
    lines.append("")

    lines.append("} // namespace barrier")
    lines.append("")
    return lines


# =============================================================================
# CB Descriptor Merging
# =============================================================================


def _get_phantom_cb_indices(phase: PhaseInfo) -> Set[int]:
    """Get CB indices referenced in named compile-time args but without CBDescriptors.

    These "phantom" CBs need identity-mapped reservations in the pool to prevent
    real CBs from being allocated at conflicting indices.
    """
    # Collect all CB indices that have actual descriptors
    real_cb_indices = set(phase.cb_info.keys())

    # Collect all CB indices referenced in named compile-time args
    phantom = set()
    for kernel_desc in phase.op_descriptor.descriptor.kernels:
        for name, value in kernel_desc.named_compile_time_args:
            if _is_cb_named_arg(name, value) and value not in real_cb_indices:
                phantom.add(value)

    return phantom


def _compute_rebind_info(
    phases: List[PhaseInfo],
    phase_remaps: List[Dict[int, int]],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """Compute which CB slots need address rebinding at each phase transition.

    For each phase 1+, identifies remapped slot indices where the buffer address
    differs from what was set in the previous phase.  Phase 0 never needs
    rebinding because build_merged_cb_descriptors prefers phase 0's buffer.

    Args:
        phases: All PhaseInfo objects.
        phase_remaps: Per-phase {orig_cb_idx: slot_idx} from the pool allocator.

    Returns:
        Dict mapping phase_idx -> list of (slot_idx, new_addr, new_size) tuples.
    """
    # Collect per-phase buffer addresses, mapped to slot indices
    phase_slot_addrs: List[Dict[int, Tuple[int, int]]] = []
    for phase_idx, phase in enumerate(phases):
        remap = phase_remaps[phase_idx] if phase_idx < len(phase_remaps) else {}
        addrs: Dict[int, Tuple[int, int]] = {}
        for cb_desc in phase.op_descriptor.descriptor.cbs:
            for fmt_desc in cb_desc.format_descriptors:
                orig_idx = fmt_desc.buffer_index
                slot_idx = remap.get(orig_idx, orig_idx)
                if cb_desc.has_buffer():
                    addr = cb_desc.buffer_address()
                    if addr is not None:
                        addrs[slot_idx] = (addr, cb_desc.total_size)
        phase_slot_addrs.append(addrs)

    if not phase_slot_addrs:
        return {}

    # Start with phase 0's addresses as baseline
    rebind_info: Dict[int, List[Tuple[int, int, int]]] = {}
    current_addrs = dict(phase_slot_addrs[0])

    for phase_idx in range(1, len(phases)):
        rebinds: List[Tuple[int, int, int]] = []
        for slot_idx, (phase_addr, phase_size) in phase_slot_addrs[phase_idx].items():
            current = current_addrs.get(slot_idx)
            if current is None or current[0] != phase_addr:
                rebinds.append((slot_idx, phase_addr, phase_size))
                current_addrs[slot_idx] = (phase_addr, phase_size)
        rebind_info[phase_idx] = rebinds

    return rebind_info


# =============================================================================
# Fused Kernel Source Generation
# =============================================================================


def _generate_fused_riscv0_source(
    phase_kernels: List[Dict[str, Any]],
    role_key: Any,
    phases: List[PhaseInfo],
    ct_arg_offsets: Dict[int, int],
    sweep_cb_indices: List[int],
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    op_semaphore_info: Optional[List[Tuple[int, int]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused RISCV_0 (reader/BRISC) kernel source.

    Uses C++ namespace wrapping for phase isolation and barrier infrastructure.
    Each phase's source goes into ``namespace phase_N { ... }`` with its
    body in ``run()``.  The outer ``kernel_main()`` calls
    ``phase_N::run()`` with ``barrier::phase::wait()`` /
    ``barrier::phase::reset()`` between phases.
    """
    reader_sources = []
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
        reader_sources.append((i, source))

    if not reader_sources:
        return None

    all_combined = ["\n".join(c for _, c in phase_headers.get(i, [])) + "\n" + s for i, s in reader_sources]
    includes = collect_includes(all_combined)
    source_defines = collect_defines(all_combined)
    must_match_defines, per_phase_defines = _collect_phase_defines(phase_kernels, role_key)
    file_scope_blocks, pre_mains = _extract_phase_pre_main(reader_sources, phase_headers)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused reader kernel - {len(reader_sources)} phases",
        "",
    ]

    # File-scope: MUST_MATCH defines + source defines + includes
    lines.extend(_emit_define_lines(must_match_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append('#include "tools/profiler/kernel_profiler.hpp"')
    lines.append("")

    # Build phase name lookup
    phase_names = {p.phase_idx: p.op_descriptor.name for p in phases}

    # File-scope: namespace blocks from inlined headers (must stay at global scope)
    if file_scope_blocks:
        for block in file_scope_blocks:
            lines.append(block)
            lines.append("")

    # RT arg wrappers at file scope (all phases, uniform treatment)
    if rt_arg_offsets:
        for phase_idx, _ in reader_sources:
            if phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    # Phase namespaces
    for phase_idx, raw_source in reader_sources:
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
            _generate_barrier_namespace_riscv0(
                sweep_cb_indices, multi_barrier, rebind_info or {}, op_semaphore_info or [], reader_sources
            )
        )

    # kernel_main
    lines.append("void kernel_main() {")
    if needs_barrier:
        lines.append("    barrier::init();")
        lines.append("")

    has_trailing = False
    if needs_barrier:
        last_phase_idx = reader_sources[-1][0]
        has_trailing = last_phase_idx in multi_barrier.transition_map

    for count, (phase_idx, _) in enumerate(reader_sources):
        pname = phase_names.get(phase_idx, "")
        label = f"Phase {phase_idx}: {pname}" if pname else f"Phase {phase_idx}"
        lines.append(f"    // {label}")
        lines.append(f"    phase_{phase_idx}::run();")
        is_last = count == len(reader_sources) - 1
        if needs_barrier and (not is_last or has_trailing):
            lines.append("    barrier::phase::wait();")
            lines.append("    barrier::phase::reset();")
            if not is_last:
                lines.append("")

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
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused RISCV_1 (writer/NCRISC) kernel source.

    Uses C++ namespace wrapping for phase isolation and barrier infrastructure.
    Writer signals done via ``barrier::phase::wait()`` and spins on release
    via ``barrier::phase::reset()`` -> ``barrier::segment_N::sync()``.
    """
    writer_sources = []
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
        writer_sources.append((i, source))

    if not writer_sources:
        return None

    all_combined = ["\n".join(c for _, c in phase_headers.get(i, [])) + "\n" + s for i, s in writer_sources]
    includes = collect_includes(all_combined)
    source_defines = collect_defines(all_combined)
    must_match_defines, per_phase_defines = _collect_phase_defines(phase_kernels, role_key)
    file_scope_blocks, pre_mains = _extract_phase_pre_main(writer_sources, phase_headers)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused writer kernel - {len(writer_sources)} phases",
        "",
    ]

    lines.extend(_emit_define_lines(must_match_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append('#include "tools/profiler/kernel_profiler.hpp"')
    lines.append("")

    # Build phase name lookup
    phase_names = {p.phase_idx: p.op_descriptor.name for p in phases}

    # File-scope: namespace blocks from inlined headers
    if file_scope_blocks:
        for block in file_scope_blocks:
            lines.append(block)
            lines.append("")

    if rt_arg_offsets:
        for phase_idx, _ in writer_sources:
            if phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    for phase_idx, raw_source in writer_sources:
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

    if needs_barrier:
        lines.extend(
            _generate_barrier_namespace_riscv1(sweep_cb_indices, multi_barrier, rebind_info or {}, writer_sources)
        )

    lines.append("void kernel_main() {")
    if needs_barrier:
        lines.append("    barrier::init();")
        lines.append("")

    has_trailing = False
    if needs_barrier:
        last_phase_idx = writer_sources[-1][0]
        has_trailing = last_phase_idx in multi_barrier.transition_map

    for count, (phase_idx, _) in enumerate(writer_sources):
        pname = phase_names.get(phase_idx, "")
        label = f"Phase {phase_idx}: {pname}" if pname else f"Phase {phase_idx}"
        lines.append(f"    // {label}")
        lines.append(f"    phase_{phase_idx}::run();")
        is_last = count == len(writer_sources) - 1
        if needs_barrier and (not is_last or has_trailing):
            lines.append("    barrier::phase::wait();")
            lines.append("    barrier::phase::reset();")
            if not is_last:
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
    rebind_info: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    multi_barrier: Optional[MultiBarrierSpec] = None,
    rt_arg_offsets: Optional[Dict[int, int]] = None,
) -> Optional[str]:
    """Generate fused compute kernel source.

    Uses C++ namespace wrapping for phase isolation and barrier infrastructure.
    Compute signals done via ``barrier::phase::wait()`` and spins on release
    via ``barrier::phase::reset()`` -> ``barrier::segment_N::sync()``.
    TRISC0/TRISC2 resync their local CB state after BRISC reset.
    """
    if ct_arg_offsets is None:
        ct_arg_offsets = {}

    compute_sources = []
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
        compute_sources.append((i, source))

    if not compute_sources:
        return None

    all_combined = ["\n".join(c for _, c in phase_headers.get(i, [])) + "\n" + s for i, s in compute_sources]
    includes = collect_includes(all_combined)
    source_defines = collect_defines(all_combined)
    must_match_defines, per_phase_defines = _collect_phase_defines(phase_kernels, role_key)
    file_scope_blocks, pre_mains = _extract_phase_pre_main(compute_sources, phase_headers)

    lines = [
        "// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC",
        "//",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        f"// Auto-generated fused compute kernel - {len(compute_sources)} phases",
        "",
    ]

    lines.extend(_emit_define_lines(must_match_defines))
    lines.extend(source_defines)
    lines.append("")
    lines.extend(includes)
    lines.append('#include "tools/profiler/kernel_profiler.hpp"')
    lines.append("")

    # Build phase name lookup
    phase_names = {p.phase_idx: p.op_descriptor.name for p in phases}

    # File-scope: namespace blocks from inlined headers
    if file_scope_blocks:
        for block in file_scope_blocks:
            lines.append(block)
            lines.append("")

    if rt_arg_offsets:
        for phase_idx, _ in compute_sources:
            if phase_idx in rt_arg_offsets:
                lines.extend(_emit_rt_arg_wrapper(phase_idx, rt_arg_offsets[phase_idx]))
        lines.append("")

    for phase_idx, raw_source in compute_sources:
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

    if needs_barrier:
        lines.extend(
            _generate_barrier_namespace_compute(
                sweep_cb_indices or [], multi_barrier, rebind_info or {}, compute_sources
            )
        )

    lines.append("void kernel_main() {")
    if needs_barrier:
        lines.append("    barrier::init();")
        lines.append("")

    has_trailing = False
    if needs_barrier:
        last_phase_idx = compute_sources[-1][0]
        has_trailing = last_phase_idx in multi_barrier.transition_map

    for count, (phase_idx, _) in enumerate(compute_sources):
        pname = phase_names.get(phase_idx, "")
        label = f"Phase {phase_idx}: {pname}" if pname else f"Phase {phase_idx}"
        lines.append(f"    // {label}")
        lines.append(f"    phase_{phase_idx}::run();")
        is_last = count == len(compute_sources) - 1
        if needs_barrier and (not is_last or has_trailing):
            lines.append("    barrier::phase::wait();")
            lines.append("    barrier::phase::reset();")
            if not is_last:
                lines.append("")

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# Runtime Arg Handling
# =============================================================================


def _compute_runtime_arg_offsets(
    phase_kernels: List[Dict[str, Any]],
    kernel_type: str,
    target_core_range: Optional[Any] = None,
) -> Dict[int, int]:
    """Compute per-phase runtime arg offsets.

    Returns {phase_idx: offset} where offset is the cumulative count of
    runtime args from all prior phases (max across cores).

    RuntimeArgsView API: runtime_args[col_idx] -> RuntimeArgsColProxy,
    runtime_args[col_idx][0] -> VectorUInt32 of args for that core.

    If target_core_range is set, use those core ranges to determine which
    cores to count args for (needed when building a fused kernel for a
    core group where stem ops cover all cores but only a subset runs
    this group's kernel).
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
        if target_core_range is not None:
            core_coords = _get_core_coords_from_ranges(target_core_range)
        else:
            core_coords = _get_core_coords_from_ranges(kernel.core_ranges)
        for core in core_coords:
            try:
                args = kernel.runtime_args[core.x][core.y]
                max_args = max(max_args, len(args))
            except (IndexError, KeyError):
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
    target_core_range: Optional[Any] = None,
) -> List[Tuple[Any, List[int]]]:
    """Concatenate per-core runtime args from all phases.

    Returns list of (CoreCoord, concatenated_args) pairs.

    RuntimeArgsView uses coordinate-based 2D indexing: runtime_args[x][y]
    maps to CoreCoord(x, y). We must use actual core coordinates, not
    sequential indices.

    If target_core_range is set, use those core ranges instead of the
    kernel's native ranges.  This extracts per-core args for the specific
    cores in this group's range (OpGraph support).
    """
    if target_core_range is not None:
        core_coords = _get_core_coords_from_ranges(target_core_range)
    else:
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

    for phase_idx, pk in enumerate(phase_kernels):
        kernel = pk.get(kernel_type)
        if kernel is None:
            continue

        # First pass: compute max_args for this phase (must match _compute_runtime_arg_offsets)
        phase_max_args = 0
        for core in core_coords:
            try:
                args = kernel.runtime_args[core.x][core.y]
                phase_max_args = max(phase_max_args, len(args))
            except (IndexError, KeyError):
                pass

        # Second pass: append args + pad to phase_max_args so offsets align
        for col_idx, core in enumerate(core_coords):
            try:
                args = kernel.runtime_args[core.x][core.y]
                arg_list = list(args)
                col_args[col_idx].extend(arg_list)
                pad_count = phase_max_args - len(arg_list)
                if pad_count > 0:
                    col_args[col_idx].extend([0] * pad_count)
            except (IndexError, KeyError):
                # No args for this core — pad entire phase width
                if phase_max_args > 0:
                    col_args[col_idx].extend([0] * phase_max_args)
                if target_core_range is not None:
                    logger.warning(
                        "Phase %d %s: no runtime args for core (%d,%d) "
                        "with target_core_range (stem op may not cover this core)",
                        phase_idx,
                        kernel_type,
                        core.x,
                        core.y,
                    )

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
        except (AttributeError, TypeError):
            pass
    return common_args


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
    Per-segment barrier constants are added externally by the caller.
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

    MUST_MATCH defines (``REDUCE_OP``, ``REDUCE_DIM``, etc.) are validated
    for consistency and returned for file-scope emission.  ALL other defines
    are returned per-phase for ``#define``/``#undef`` wrapping around each
    phase's namespace.  No uniform/varying optimization — simplicity over
    minimal ``#define`` count.

    Returns:
        (must_match, per_phase) where:
        - must_match: list of (name, value) for defines emitted at file scope
        - per_phase: dict mapping phase_index -> list of (name, value)

    Raises:
        ValueError: If a MUST_MATCH define has inconsistent values across phases.
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
) -> _BuildResult:
    """Build a fused ProgramDescriptor with multi-segment barrier sync.

    Dynamically discovers kernel roles from the ProgramDescriptor using
    (risc_type, core_ranges) as a unique key. This supports any op type
    (interleaved with 3 kernels, sharded with up to 7 kernels, etc.).

    Args:
        phases: List of PhaseInfo objects for each phase.
        device: The device for GlobalSemaphore allocation.
        target_core_range: If set, the fused kernel binary will run on
            this core range. Used by OpGraphBuilder when building a kernel
            for a core group where phases may have different native ranges
            (e.g. stem covers 16 cores but this group's cores are 8).
        multi_barrier: Multi-segment barrier spec. Required for multi-phase
            chains. Provides barrier segment configs and transition map.
    """
    # Validate fp32 consistency
    _validate_fp32_consistency([p.op_descriptor for p in phases])

    # Discover all kernel roles from phase 0
    role_keys: List[Tuple[str, frozenset]] = []
    role_keys_set: Set[Tuple[str, frozenset]] = set()
    for kernel_desc in phases[0].op_descriptor.descriptor.kernels:
        rk = _get_role_key(kernel_desc, target_core_range)
        if rk not in role_keys_set:
            role_keys.append(rk)
            role_keys_set.add(rk)

    # Build phase_kernels as List[Dict[role_key, KernelDescriptor]]
    phase_kernels: List[Dict[Any, Any]] = []
    for phase_idx, phase in enumerate(phases):
        role_map: Dict[Any, Any] = {}
        for kernel_desc in phase.op_descriptor.descriptor.kernels:
            rk = _get_role_key(kernel_desc, target_core_range)
            role_map[rk] = kernel_desc
        phase_kernels.append(role_map)

    # Pool-allocate CB slots based on compatibility keys
    pool = CBPoolAllocator(max_slots=32)
    for phase_idx, phase in enumerate(phases):
        phantom_indices = _get_phantom_cb_indices(phase)
        pool.allocate_phase(phase_idx, phase.cb_info, phantom_indices)

    # Compute CB address rebinding info using remapped slot indices.
    rebind_info = _compute_rebind_info(phases, pool.phase_remaps)

    # Build merged CB descriptors from pool (modifies buffer_index in-place)
    merged_cbs = pool.build_merged_cb_descriptors(phases)

    # Set CB core_ranges to the target when building for a specific core group.
    if target_core_range is not None:
        for cb_desc in merged_cbs:
            cb_desc.core_ranges = target_core_range

    # Sweep indices = all allocated CB pool slots
    sweep_cb_indices = sorted(pool.get_all_slot_indices())

    # Collect all unique op semaphore (id, initial_value) pairs used by any phase.
    op_semaphore_info: List[Tuple[int, int]] = []
    seen_sem_ids_for_reset: Set[int] = set()
    for phase in phases:
        for sem in phase.op_descriptor.descriptor.semaphores:
            if sem.id not in seen_sem_ids_for_reset:
                op_semaphore_info.append((sem.id, sem.initial_value))
                seen_sem_ids_for_reset.add(sem.id)
    op_semaphore_info.sort(key=lambda x: x[0])

    fused_kernels = []

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

        # Merge compile-time args and compute offsets
        ct_args, ct_offsets = _merge_compile_time_args(phase_kernels, role_key)
        rt_offsets = _compute_runtime_arg_offsets(phase_kernels, role_key, target_core_range=target_core_range)

        # Generate fused source and determine barrier addresses per RISC type
        if risc_type == "riscv_0":
            fused_source = _generate_fused_riscv0_source(
                phase_kernels,
                role_key,
                phases,
                ct_offsets,
                sweep_cb_indices,
                rebind_info=rebind_info,
                op_semaphore_info=op_semaphore_info,
                multi_barrier=multi_barrier,
                rt_arg_offsets=rt_offsets,
            )
            barrier_addrs = []
            if multi_barrier is not None:
                barrier_addrs = [multi_barrier.compute_done_addr, multi_barrier.writer_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.extend([seg.arrive_addr, seg.release_addr])
        elif risc_type == "riscv_1":
            fused_source = _generate_fused_riscv1_source(
                phase_kernels,
                role_key,
                phases,
                ct_offsets,
                sweep_cb_indices,
                rebind_info=rebind_info,
                multi_barrier=multi_barrier,
                rt_arg_offsets=rt_offsets,
            )
            barrier_addrs = []
            if multi_barrier is not None:
                barrier_addrs = [multi_barrier.writer_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.append(seg.release_addr)
        elif risc_type == "compute":
            fused_source = _generate_fused_compute_source(
                phase_kernels,
                role_key,
                phases,
                ct_offsets,
                sweep_cb_indices,
                rebind_info=rebind_info,
                multi_barrier=multi_barrier,
                rt_arg_offsets=rt_offsets,
            )
            barrier_addrs = []
            if multi_barrier is not None:
                barrier_addrs = [multi_barrier.compute_done_addr]
                for seg in multi_barrier.segments:
                    barrier_addrs.append(seg.release_addr)
        else:
            continue

        if fused_source is None:
            continue

        # Concatenate runtime args and append barrier addresses
        rt_args = _concatenate_runtime_args(phase_kernels, role_key, target_core_range=target_core_range)
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

        # Add rebind named compile-time args (addr + size for each CB that changes)
        for phase_idx, rebinds in rebind_info.items():
            for slot_idx, addr, size in rebinds:
                prefix = f"phase_{phase_idx}_cb{slot_idx}"
                named_ct_args.append((f"{prefix}_rebind_addr", addr))
                named_ct_args.append((f"{prefix}_rebind_size", size))

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

    # Merge semaphores (dedup by ID)
    all_semaphores = []
    seen_sem_ids: Set[int] = set()
    for phase in phases:
        for sem in phase.op_descriptor.descriptor.semaphores:
            if sem.id not in seen_sem_ids:
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

    return _BuildResult(
        descriptor=merged_descriptor,
        input_tensors=all_input_tensors,
        output_tensors=[output_tensor] if output_tensor else [],
        semaphores=sem_refs,
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


__all__ = [
    # C++ parsing (from cpp_parser.py)
    "extract_kernel_body",
    "inline_local_includes",
    "collect_includes",
    "collect_defines",
    # Build orchestration
    "_build_fused_descriptor",
    "_create_phase_info",
    "_create_barrier_segment_config",
]
