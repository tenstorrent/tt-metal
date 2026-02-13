# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Code generation: Exo procs -> complete TT-Metal C++ kernel source strings.

Takes compiled Exo procedures and wraps them with:
    - TT-Metal #include headers
    - void kernel_main() { ... } entry point
    - Preamble code (runtime/compile-time arg reads, hardware init)
    - Epilogue code (write barriers, etc.)

The generated source strings can be passed directly to KernelDescriptor
with SourceType.SOURCE_CODE.
"""

from __future__ import annotations

import re
import textwrap

from exo import Procedure, compile_procs_to_strings


# ---------------------------------------------------------------------------
# Kernel templates
# ---------------------------------------------------------------------------

READER_INCLUDES = '#include "api/dataflow/dataflow_api.h"'

READER_PREAMBLE = textwrap.dedent(
    """\
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr auto src_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = 0;
    const uint32_t page_bytes = get_local_cb_interface(cb_id_in0).fifo_page_size;
    const auto s = TensorAccessor(src_args, src_addr, page_bytes);
    uint32_t l1_write_addr;

    uint32_t end_id = start_id + num_pages;"""
)

COMPUTE_INCLUDES_BASE = (
    "#include <cstdint>\n"
    '#include "api/compute/common.h"\n'
    '#include "api/compute/tile_move_copy.h"\n'
    '#include "api/compute/eltwise_unary/eltwise_unary.h"'
)

# Op-specific includes appended to the base
COMPUTE_OP_INCLUDES = {
    "identity": "",
    "relu": '\n#include "api/compute/eltwise_unary/relu.h"',
}

COMPUTE_PREAMBLE = textwrap.dedent(
    """\
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    copy_tile_init(tt::CBIndex::c_0);"""
)

WRITER_INCLUDES = '#include "api/dataflow/dataflow_api.h"'

WRITER_PREAMBLE = textwrap.dedent(
    """\
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;
    const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);
    uint32_t l1_read_addr;

    uint32_t end_id = start_id + num_pages;"""
)

WRITER_EPILOGUE = "    noc_async_write_barrier();"


# ---------------------------------------------------------------------------
# Exo -> C extraction
# ---------------------------------------------------------------------------


def _extract_loop_body(c_code: str) -> str:
    """Extract the loop body from Exo-generated C code.

    Exo generates a complete C function. We extract just the loop(s)
    from the function body, which contain our @instr-generated code.
    """
    # Find the function body (after the opening brace of the function)
    # Pattern: void func_name(...) {  <body>  }
    match = re.search(r"void \w+\s*\([^)]*\)\s*\{", c_code)
    if not match:
        raise ValueError("Could not find function body in Exo output")

    # Find the matching closing brace
    start = match.end()
    depth = 1
    pos = start
    while pos < len(c_code) and depth > 0:
        if c_code[pos] == "{":
            depth += 1
        elif c_code[pos] == "}":
            depth -= 1
        pos += 1

    body = c_code[start : pos - 1].strip()
    return body


def _rewrite_loop_var(body: str, kernel_type: str) -> str:
    """Rewrite Exo's loop variable to use TT-Metal conventions.

    Exo generates: for (int_fast32_t i = 0; i < N; i++)
    Reader/Writer need: for (uint32_t i = start_id; i < end_id; ++i)
    Compute uses its own block structure.
    """
    if kernel_type == "reader" or kernel_type == "writer":
        # Replace Exo's flat loop with TT-Metal's start_id-based loop
        body = re.sub(
            r"for\s*\(\s*int_fast32_t\s+(\w+)\s*=\s*0\s*;\s*\w+\s*<\s*\(\(N\)\s*/\s*\(1\)\)\s*;\s*\w+\+\+\s*\)",
            r"for (uint32_t \1 = start_id; \1 < end_id; ++\1)",
            body,
        )
        body = re.sub(
            r"for\s*\(\s*int_fast32_t\s+(\w+)\s*=\s*0\s*;\s*\w+\s*<\s*N\s*;\s*\w+\+\+\s*\)",
            r"for (uint32_t \1 = start_id; \1 < end_id; ++\1)",
            body,
        )
        # The @instr already uses start_id + i, but now i starts from start_id
        # We need the @instr to use just i (since i already includes start_id)
        body = body.replace("start_id + (i)", "i")
        body = body.replace("start_id + i", "i")
    elif kernel_type == "compute":
        # Replace Exo's N-based loop with compile-time args
        # Flat loop: for (int_fast32_t i = 0; i < N; i++)
        body = re.sub(
            r"for\s*\(\s*int_fast32_t\s+(\w+)\s*=\s*0\s*;\s*\w+\s*<\s*N\s*;\s*\w+\+\+\s*\)",
            r"for (uint32_t \1 = 0; \1 < per_core_block_cnt * per_core_block_dim; ++\1)",
            body,
        )
        # Block-tiled outer loop: for (int_fast32_t block_idx = 0; block_idx < ((N) / (K)); ...)
        body = re.sub(
            r"for\s*\(\s*int_fast32_t\s+block_idx\s*=\s*0\s*;\s*block_idx\s*<\s*\(\(N\)\s*/\s*\(\d+\)\)\s*;\s*block_idx\+\+\s*\)",
            "for (uint32_t block_idx = 0; block_idx < per_core_block_cnt; ++block_idx)",
            body,
        )
        # Block-tiled inner loop: for (int_fast32_t tile_idx = 0; tile_idx < K; ...)
        body = re.sub(
            r"for\s*\(\s*int_fast32_t\s+tile_idx\s*=\s*0\s*;\s*tile_idx\s*<\s*(\d+)\s*;\s*tile_idx\+\+\s*\)",
            r"for (uint32_t tile_idx = 0; tile_idx < per_core_block_dim; ++tile_idx)",
            body,
        )
        # Remove the remainder loop (N % K) — TT-Metal handles this via
        # per-core work splitting, so remainder is always 0 when block_dim
        # evenly divides the per-core tile count.
        body = re.sub(
            r"\nfor\s*\(\s*int_fast32_t\s+tile_idx\s*=\s*0\s*;\s*tile_idx\s*<\s*N\s*%\s*\d+\s*;.*?(?=\n[^\s]|\Z)",
            "",
            body,
            flags=re.DOTALL,
        )

    return body


def _normalize_indentation(body: str) -> str:
    """Normalize the indentation of Exo-generated code.

    Exo's multi-line @instr templates only indent the first line (2 spaces
    inside a for loop); subsequent lines are at column 0. This normalizer
    tracks brace depth and indents all lines according to their nesting.
    """
    lines = body.split("\n")
    result = []
    depth = 0
    for line in lines:
        stripped = line.lstrip()
        if not stripped:
            result.append("")
            continue

        # Closing brace decreases depth before indenting
        if stripped.startswith("}"):
            depth = max(0, depth - 1)

        result.append("    " * depth + stripped)

        # Opening brace increases depth after indenting
        if stripped.endswith("{"):
            depth += 1

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_kernel_source(
    exo_proc: Procedure,
    kernel_type: str,
    op: str = "identity",
) -> str:
    """Generate a complete TT-Metal kernel source string from an Exo proc.

    Args:
        exo_proc: Compiled Exo procedure.
        kernel_type: One of "reader", "compute", "writer".
        op: Operation name (used to select op-specific compute includes).

    Returns:
        Complete C++ source string suitable for KernelDescriptor(SOURCE_CODE).
    """
    if kernel_type not in ("reader", "compute", "writer"):
        raise ValueError(f"kernel_type must be reader/compute/writer, got {kernel_type}")

    # Get raw C from Exo
    c_code, _ = compile_procs_to_strings([exo_proc], "exo_kernel.h")

    # Extract the function body (loop structure with @instr calls)
    body = _extract_loop_body(c_code)

    # Rewrite loop variables to match TT-Metal conventions
    body = _rewrite_loop_var(body, kernel_type)

    # Normalize indentation
    body = _normalize_indentation(body)

    # Select kernel template components
    if kernel_type == "reader":
        includes = READER_INCLUDES
        preamble = READER_PREAMBLE
        epilogue = ""
    elif kernel_type == "compute":
        includes = COMPUTE_INCLUDES_BASE + COMPUTE_OP_INCLUDES.get(op, "")
        preamble = COMPUTE_PREAMBLE
        epilogue = ""
    elif kernel_type == "writer":
        includes = WRITER_INCLUDES
        preamble = WRITER_PREAMBLE
        epilogue = WRITER_EPILOGUE

    # Indent body
    indented_body = textwrap.indent(body, "    ")

    # Assemble
    parts = [includes, "", "void kernel_main() {"]
    parts.append(textwrap.indent(preamble, "    "))
    parts.append("")
    parts.append(indented_body)
    if epilogue:
        parts.append(epilogue)
    parts.append("}")

    return "\n".join(parts) + "\n"


def generate_eltwise_kernels(
    op: str = "identity",
    block_dim: int = 1,
) -> tuple[str, str, str]:
    """Generate all 3 kernel sources for an eltwise unary op.

    Args:
        op: "identity" or "relu"
        block_dim: Compute loop block dimension (1 = flat loop).

    Returns:
        (reader_source, compute_source, writer_source) tuple of C++ strings.
    """
    from models.experimental.ops.exo_codegen.eltwise_unary import get_procs

    reader_proc, compute_proc, writer_proc = get_procs(op, block_dim)

    reader_src = generate_kernel_source(reader_proc, "reader", op=op)
    compute_src = generate_kernel_source(compute_proc, "compute", op=op)
    writer_src = generate_kernel_source(writer_proc, "writer", op=op)

    return reader_src, compute_src, writer_src
