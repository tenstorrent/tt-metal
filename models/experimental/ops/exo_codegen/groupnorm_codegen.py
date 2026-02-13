# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Code generation for GroupNorm: Exo procs -> complete TT-Metal C++ kernel sources.

Generates 3 kernel source strings:
    1. Reader: Loads input tiles from DRAM into cb_in0
    2. Writer: Generates scaler/eps/mask tiles, writes output to DRAM
    3. Compute: 3-pass GroupNorm algorithm (Exo-generated inner loops)

The compute kernel uses Exo-generated C code for inner tile operations,
wrapped with CB protocol, init calls, and named compile-time args.
"""

from __future__ import annotations

import re
import textwrap

from exo import compile_procs_to_strings

from models.experimental.ops.exo_codegen.groupnorm import get_procs


# ---------------------------------------------------------------------------
# Exo -> C extraction (shared with eltwise codegen)
# ---------------------------------------------------------------------------


def _extract_loop_body(c_code: str) -> str:
    """Extract the for-loop body from Exo-generated C code."""
    match = re.search(r"void \w+\s*\([^)]*\)\s*\{", c_code)
    if not match:
        raise ValueError("Could not find function body in Exo output")
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


def _compile_and_extract(proc, name: str) -> str:
    """Compile an Exo proc and extract the loop body."""
    c_code, _ = compile_procs_to_strings([proc], f"gn_{name}.h")
    return _extract_loop_body(c_code)


def _rewrite_loop(body: str, loop_bound: str) -> str:
    """Rewrite Exo's loop variable to use TT-Metal conventions."""
    # Replace int_fast32_t with uint32_t and N with the actual bound
    body = re.sub(
        r"for\s*\(\s*int_fast32_t\s+(\w+)\s*=\s*0\s*;\s*\w+\s*<\s*N\s*;\s*\w+\+\+\s*\)",
        rf"for (uint32_t \1 = 0; \1 < {loop_bound}; ++\1)",
        body,
    )
    return body


def _replace_cb_names(body: str, replacements: dict[str, str]) -> str:
    """Replace placeholder CB names in Exo-generated code."""
    for placeholder, actual in replacements.items():
        body = body.replace(placeholder, actual)
    return body


# ---------------------------------------------------------------------------
# Generate Exo loop bodies for each GroupNorm step
# ---------------------------------------------------------------------------


def _generate_exo_loops() -> dict[str, str]:
    """Compile all GroupNorm Exo procs and extract loop bodies."""
    procs = get_procs()
    loops = {}
    for name, proc in procs.items():
        loops[name] = _compile_and_extract(proc, name)
    return loops


# ---------------------------------------------------------------------------
# Compute kernel template
# ---------------------------------------------------------------------------

COMPUTE_TEMPLATE = textwrap.dedent(
    """\
    #include <cstdint>

    #define REDUCE_OP PoolType::SUM
    #define REDUCE_DIM ReduceDim::REDUCE_SCALAR

    #define BCAST_LLKOP EltwiseBinaryType::ELWMUL
    #define BCAST_DIM BroadcastType::COL

    #include "api/compute/reduce.h"
    #include "api/compute/bcast.h"
    #include "api/compute/eltwise_binary.h"
    #include "api/compute/layernorm.h"
    #include "api/compute/tile_move_copy.h"

    void kernel_main() {{
        // Compile-time args
        constexpr uint32_t block_hw = get_compile_time_arg_val(0);

        // CB indices
        constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
        constexpr uint32_t cb_scaler = tt::CBIndex::c_2;
        constexpr uint32_t cb_eps = tt::CBIndex::c_3;
        constexpr uint32_t cb_scaler_global = tt::CBIndex::c_4;
        constexpr uint32_t cb_ex_partial = tt::CBIndex::c_8;
        constexpr uint32_t cb_ex_global = tt::CBIndex::c_15;
        constexpr uint32_t cb_ex2_partial = tt::CBIndex::c_21;
        constexpr uint32_t cb_ex2_global = tt::CBIndex::c_14;
        constexpr uint32_t cb_ex2pe = tt::CBIndex::c_27;
        constexpr uint32_t cb_x = tt::CBIndex::c_24;
        constexpr uint32_t cb_xmm = tt::CBIndex::c_25;
        constexpr uint32_t cb_input_mask = tt::CBIndex::c_28;
        constexpr uint32_t cb_out0 = tt::CBIndex::c_16;

        constexpr uint32_t dst0 = 0;
        constexpr uint32_t scaler0 = 0;
        constexpr bool FP32_DEST_ACC = false;

        // Init
        binary_op_init_common(cb_in0, cb_input_mask, cb_x);

        // Wait for all persistent data
        cb_wait_front(cb_in0, block_hw);
        cb_wait_front(cb_input_mask, 1);
        cb_wait_front(cb_scaler, 1);
        cb_wait_front(cb_scaler_global, 1);
        cb_wait_front(cb_eps, 1);

        // =====================================================================
        // Pass 1: Mean — E[x]
        // =====================================================================

        // Step 1.1: Mask input (input * mask -> cb_x)
        mul_tiles_init(cb_in0, cb_input_mask);
        cb_reserve_back(cb_x, block_hw);
    {mask_p1}
        cb_push_back(cb_x, block_hw);

        // Step 1.2: Local reduce (sum of masked tiles -> cb_ex_partial)
        reconfig_data_format_srcb(cb_input_mask, cb_scaler);
        reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_x, cb_scaler, cb_ex_partial);
        cb_reserve_back(cb_ex_partial, 1);
        tile_regs_acquire();
        cb_wait_front(cb_x, block_hw);
    {reduce_p1}
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial);
        tile_regs_release();
        cb_pop_front(cb_x, block_hw);
        cb_push_back(cb_ex_partial, 1);
        reduce_uninit<FP32_DEST_ACC>();

        // Step 1.3: Global reduce (partial * 1/N -> mean in cb_ex_global)
        reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex_partial, cb_scaler_global, cb_ex_global);
        cb_reserve_back(cb_ex_global, 1);
        tile_regs_acquire();
        cb_wait_front(cb_ex_partial, 1);
        reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex_partial, cb_scaler_global, 0, scaler0, dst0);
        cb_pop_front(cb_ex_partial, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_global);
        tile_regs_release();
        cb_push_back(cb_ex_global, 1);
        reduce_uninit<FP32_DEST_ACC>();

        // =====================================================================
        // Pass 2: Variance — E[(x - E[x])^2]
        // =====================================================================

        // Step 2.1: Subtract mean (input - mean -> cb_xmm)
        sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
        cb_reserve_back(cb_xmm, block_hw);
        cb_wait_front(cb_ex_global, 1);
    {sub_mean_p2}
        cb_push_back(cb_xmm, block_hw);

        // Step 2.2: Mask residual (residual * mask -> cb_x)
        reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
        mul_tiles_init(cb_xmm, cb_input_mask);
        cb_reserve_back(cb_x, block_hw);
        cb_wait_front(cb_xmm, block_hw);
    {mask_p2}
        cb_pop_front(cb_xmm, block_hw);
        cb_push_back(cb_x, block_hw);

        // Step 2.3: Square ((x - E[x])^2 -> cb_xmm)
        reconfig_data_format_srcb(cb_input_mask, cb_x);
        mul_tiles_init(cb_x, cb_x);
        cb_reserve_back(cb_xmm, block_hw);
        cb_wait_front(cb_x, block_hw);
    {square_p2}
        cb_pop_front(cb_x, block_hw);
        cb_push_back(cb_xmm, block_hw);

        // Step 2.4: Local reduce (sum of squares -> cb_ex2_partial)
        reconfig_data_format_srcb(cb_x, cb_scaler);
        reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_xmm, cb_scaler, cb_ex2_partial);
        cb_reserve_back(cb_ex2_partial, 1);
        tile_regs_acquire();
        cb_wait_front(cb_xmm, block_hw);
    {reduce_p2}
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2_partial);
        tile_regs_release();
        cb_pop_front(cb_xmm, block_hw);
        cb_push_back(cb_ex2_partial, 1);
        reduce_uninit<FP32_DEST_ACC>();

        // Step 2.5: Global reduce (partial variance * 1/N -> variance in cb_ex2_global)
        reduce_init<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2_partial, cb_scaler_global, cb_ex2_global);
        cb_reserve_back(cb_ex2_global, 1);
        tile_regs_acquire();
        cb_wait_front(cb_ex2_partial, 1);
        reduce_tile<REDUCE_OP, REDUCE_DIM, FP32_DEST_ACC>(cb_ex2_partial, cb_scaler_global, 0, scaler0, dst0);
        cb_pop_front(cb_ex2_partial, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2_global);
        tile_regs_release();
        cb_push_back(cb_ex2_global, 1);
        reduce_uninit<FP32_DEST_ACC>();

        // Step 2.6: inv_std = 1/sqrt(var + eps)
        cb_wait_front(cb_ex2_global, 1);
        cb_reserve_back(cb_ex2pe, 1);
        tile_regs_acquire();
        add_tiles_init(cb_ex2_global, cb_eps);
        add_tiles(cb_ex2_global, cb_eps, 0, 0, dst0);
        tile_regs_wait();
        rsqrt_tile_init<true>();
        rsqrt_tile<true>(dst0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex2pe);
        tile_regs_release();
        cb_push_back(cb_ex2pe, 1);
        cb_pop_front(cb_ex2_global, 1);

        // =====================================================================
        // Pass 3: Normalize — (x - E[x]) * inv_std
        // =====================================================================

        // Step 3.1: Subtract mean (input - mean -> cb_xmm)
        sub_tiles_bcast_scalar_init_short(cb_in0, cb_ex_global);
        cb_reserve_back(cb_xmm, block_hw);
    {sub_mean_p3}
        cb_push_back(cb_xmm, block_hw);

        // Step 3.2: Mask residual (residual * mask -> cb_x)
        reconfig_data_format_srcb(cb_ex_global, cb_input_mask);
        mul_tiles_init(cb_xmm, cb_input_mask);
        cb_reserve_back(cb_x, block_hw);
        cb_wait_front(cb_xmm, block_hw);
    {mask_p3}
        cb_pop_front(cb_xmm, block_hw);
        cb_push_back(cb_x, block_hw);

        // Step 3.3: Multiply by inv_std (masked_residual * inv_std -> cb_out0)
        reconfig_data_format_srcb(cb_input_mask, cb_ex2pe);
        mul_tiles_bcast_scalar_init_short(cb_x, cb_ex2pe);
        cb_reserve_back(cb_out0, block_hw);
        cb_wait_front(cb_x, block_hw);
        cb_wait_front(cb_ex2pe, 1);
    {mul_invstd_p3}
        cb_pop_front(cb_x, block_hw);
        cb_push_back(cb_out0, block_hw);

        // Cleanup: pop persistent CBs
        cb_pop_front(cb_in0, block_hw);
        cb_pop_front(cb_input_mask, 1);
        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_ex2pe, 1);
    }}
"""
)


# ---------------------------------------------------------------------------
# Reader kernel template (simple: load input from DRAM)
# ---------------------------------------------------------------------------

READER_SOURCE = textwrap.dedent(
    """\
    #include "api/dataflow/dataflow_api.h"

    void kernel_main() {
        const uint32_t src_addr = get_arg_val<uint32_t>(0);
        const uint32_t block_hw = get_arg_val<uint32_t>(1);
        const uint32_t start_id = get_arg_val<uint32_t>(2);

        constexpr auto src_args = TensorAccessorArgs<0>();
        constexpr uint32_t cb_in0 = 0;
        const uint32_t page_bytes = get_local_cb_interface(cb_in0).fifo_page_size;
        const auto s = TensorAccessor(src_args, src_addr, page_bytes);

        // Load all input tiles at once — compute reads 3x without popping
        for (uint32_t i = 0; i < block_hw; ++i) {
            cb_reserve_back(cb_in0, 1);
            uint32_t l1_write_addr = get_write_ptr(cb_in0);
            noc_async_read_page(start_id + i, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);
        }
    }
"""
)


# ---------------------------------------------------------------------------
# Writer kernel template (generate scaler/eps/mask tiles, write output)
# ---------------------------------------------------------------------------

WRITER_SOURCE = textwrap.dedent(
    """\
    #include <stdint.h>
    #include "api/dataflow/dataflow_api.h"
    #include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
    #include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"

    void kernel_main() {
        const uint32_t packed_scaler = get_arg_val<uint32_t>(0);
        const uint32_t packed_scaler_global = get_arg_val<uint32_t>(1);
        const uint32_t packed_eps = get_arg_val<uint32_t>(2);
        const uint32_t dst_addr = get_arg_val<uint32_t>(3);
        const uint32_t block_hw = get_arg_val<uint32_t>(4);
        const uint32_t start_id = get_arg_val<uint32_t>(5);

        constexpr auto dst_args = TensorAccessorArgs<0>();
        constexpr uint32_t cb_scaler = 2;
        constexpr uint32_t cb_eps = 3;
        constexpr uint32_t cb_scaler_global = 4;
        constexpr uint32_t cb_input_mask = 28;
        constexpr uint32_t cb_out0 = 16;

        // Generate reduce scaler (1.0) — used for local reduction
        generate_reduce_scaler(cb_scaler, packed_scaler);

        // Generate global reduce scaler (1/N) — used for mean/variance
        generate_reduce_scaler(cb_scaler_global, packed_scaler_global);

        // Generate epsilon tile
        generate_bcast_col_scalar(cb_eps, packed_eps);

        // Generate input mask tile (all 1.0 for aligned groups)
        {
            constexpr uint32_t packed_one = 0x3F803F80;  // bfloat16 1.0 x2
            cb_reserve_back(cb_input_mask, 1);
            uint32_t mask_addr = get_write_ptr(cb_input_mask);
            volatile tt_l1_ptr uint32_t* ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mask_addr);
            for (uint32_t i = 0; i < 512; ++i) {
                ptr[i] = packed_one;
            }
            cb_push_back(cb_input_mask, 1);
        }

        // Write output tiles to DRAM
        const uint32_t page_bytes = get_local_cb_interface(cb_out0).fifo_page_size;
        const auto s = TensorAccessor(dst_args, dst_addr, page_bytes);

        for (uint32_t i = 0; i < block_hw; ++i) {
            cb_wait_front(cb_out0, 1);
            uint32_t l1_read_addr = get_read_ptr(cb_out0);
            noc_async_write_page(start_id + i, s, l1_read_addr);
            noc_async_writes_flushed();
            cb_pop_front(cb_out0, 1);
        }
        noc_async_write_barrier();
    }
"""
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_groupnorm_compute(block_hw: int | None = None) -> str:
    """Generate the GroupNorm compute kernel C++ source from Exo.

    Args:
        block_hw: If provided, hardcode the loop bound. If None, use
                  compile-time arg (more flexible).

    Returns:
        Complete C++ source string for the compute kernel.
    """
    loops = _generate_exo_loops()
    bound = str(block_hw) if block_hw is not None else "block_hw"

    # CB replacement maps for each step
    # Pass 1: mask input
    mask_p1 = _rewrite_loop(loops["mask"], bound)
    mask_p1 = _replace_cb_names(
        mask_p1,
        {
            "SRC_A": "cb_in0",
            "SRC_B": "cb_input_mask",
            "DST_CB": "cb_x",
        },
    )

    # Pass 1: reduce
    reduce_p1 = _rewrite_loop(loops["reduce"], bound)
    reduce_p1 = _replace_cb_names(
        reduce_p1,
        {
            "SRC_A": "cb_x",
            "SRC_B": "cb_scaler",
        },
    )

    # Pass 2: sub mean
    sub_mean_p2 = _rewrite_loop(loops["sub_mean"], bound)
    sub_mean_p2 = _replace_cb_names(
        sub_mean_p2,
        {
            "SRC_A": "cb_in0",
            "SRC_B": "cb_ex_global",
            "DST_CB": "cb_xmm",
        },
    )

    # Pass 2: mask residual
    mask_p2 = _rewrite_loop(loops["mask"], bound)
    mask_p2 = _replace_cb_names(
        mask_p2,
        {
            "SRC_A": "cb_xmm",
            "SRC_B": "cb_input_mask",
            "DST_CB": "cb_x",
        },
    )

    # Pass 2: square
    square_p2 = _rewrite_loop(loops["square"], bound)
    square_p2 = _replace_cb_names(
        square_p2,
        {
            "SRC_A": "cb_x",
            "DST_CB": "cb_xmm",
        },
    )

    # Pass 2: reduce
    reduce_p2 = _rewrite_loop(loops["reduce"], bound)
    reduce_p2 = _replace_cb_names(
        reduce_p2,
        {
            "SRC_A": "cb_xmm",
            "SRC_B": "cb_scaler",
        },
    )

    # Pass 3: sub mean
    sub_mean_p3 = _rewrite_loop(loops["sub_mean"], bound)
    sub_mean_p3 = _replace_cb_names(
        sub_mean_p3,
        {
            "SRC_A": "cb_in0",
            "SRC_B": "cb_ex_global",
            "DST_CB": "cb_xmm",
        },
    )

    # Pass 3: mask residual
    mask_p3 = _rewrite_loop(loops["mask"], bound)
    mask_p3 = _replace_cb_names(
        mask_p3,
        {
            "SRC_A": "cb_xmm",
            "SRC_B": "cb_input_mask",
            "DST_CB": "cb_x",
        },
    )

    # Pass 3: multiply inv_std
    mul_invstd_p3 = _rewrite_loop(loops["mul_invstd"], bound)
    mul_invstd_p3 = _replace_cb_names(
        mul_invstd_p3,
        {
            "SRC_A": "cb_x",
            "SRC_B": "cb_ex2pe",
            "DST_CB": "cb_out0",
        },
    )

    # Indent all loops for insertion into template
    indent = "    " * 2  # 8 spaces

    def _indent(code: str) -> str:
        return textwrap.indent(code.strip(), indent)

    return COMPUTE_TEMPLATE.format(
        mask_p1=_indent(mask_p1),
        reduce_p1=_indent(reduce_p1),
        sub_mean_p2=_indent(sub_mean_p2),
        mask_p2=_indent(mask_p2),
        square_p2=_indent(square_p2),
        reduce_p2=_indent(reduce_p2),
        sub_mean_p3=_indent(sub_mean_p3),
        mask_p3=_indent(mask_p3),
        mul_invstd_p3=_indent(mul_invstd_p3),
    )


def generate_groupnorm_reader() -> str:
    """Generate the GroupNorm reader kernel C++ source."""
    return READER_SOURCE


def generate_groupnorm_writer() -> str:
    """Generate the GroupNorm writer kernel C++ source."""
    return WRITER_SOURCE


def generate_groupnorm_kernels(block_hw: int | None = None) -> tuple[str, str, str]:
    """Generate all 3 kernel sources for GroupNorm.

    Returns:
        (reader_source, compute_source, writer_source) tuple.
    """
    return (
        generate_groupnorm_reader(),
        generate_groupnorm_compute(block_hw),
        generate_groupnorm_writer(),
    )
