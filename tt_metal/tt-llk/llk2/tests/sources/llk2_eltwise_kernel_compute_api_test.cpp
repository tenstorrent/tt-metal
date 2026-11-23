// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: the LLK 2.0 tt-metal kernel kernels/eltwise_binary_no_bcast.cpp, the file itself, on one core in
// the tt-llk harness.
// NUM_TILES, the kernel's run-time arg 0, is a run-time parameter of the harness: one build serves every tile count.
//
// This file gives the kernel what tt-metal's build would (tests/helpers/tt_metal_kernel.h). What is emulated: the
// host wrote lhs and rhs to L1 and preload() marks them pushed (tt-metal: the reader kernel), and out holds every
// output tile and nothing pops it (tt-metal: the writer kernel).

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "params.h"

// ckernel_globals.h declares these extern; unp_cfg_context is the unpack context unpack.h flips.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

#include "tt_metal_kernel.h" // TEST ONLY: what tt-metal's build gives a kernel
#define LLK2_DEST_FP32     DST_ACCUM_MODE
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "kernel_settings.h"

namespace
{

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

// Ids as the tt-metal host names the CBs: c_0 lhs, c_1 rhs, c_2 out.
constexpr std::uint16_t lhs_id = 0, rhs_id = 1, out_id = 2;

/// The named compile-time args the kernel reads.
constexpr llk2_test::NamedArg named_args[] = {
    {"binary_op", OP},
    {"cb_lhs", lhs_id},
    {"cb_rhs", rhs_id},
    {"cb_out", out_id},
    {"cb_lhs_format", static_cast<std::uint32_t>(in_format)},
    {"cb_rhs_format", static_cast<std::uint32_t>(in_format)},
    {"cb_out_format", static_cast<std::uint32_t>(out_format)},
};

std::uint32_t num_tiles_arg = 0; ///< the kernel's run-time arg 0, set by run_kernel

} // namespace

/// tt-metal: the kernel's positional compile-time arg i. The kernel reads 0, tiles_per_cycle.
constexpr std::uint32_t get_compile_time_arg_val(std::uint32_t)
{
    return TILES_PER_CYCLE;
}

/// tt-metal: the kernel's named compile-time arg `name`.
constexpr std::uint32_t get_named_compile_time_arg_val(const char* name)
{
    return llk2_test::find_named_arg(named_args, name);
}

/// tt-metal: the kernel's run-time arg i. The kernel reads 0, its tile count.
template <class T>
inline T get_arg_val(int)
{
    return num_tiles_arg;
}

#include "eltwise_binary_no_bcast.cpp" // the kernel file, as tt-metal builds it

void run_kernel(RUNTIME_PARAMETERS params)
{
    using llk2_test::buffer_configs;
    num_tiles_arg = params.NUM_TILES;
    // TEST ONLY: the host's CB config; every buffer holds MAX_TILES pages.
    buffer_configs[lhs_id] = {params.buffer_A[0], llk2::tile_bytes(in_format, llk2::default_tile_datums), MAX_TILES};
    buffer_configs[rhs_id] = {params.buffer_B[0], llk2::tile_bytes(in_format, llk2::default_tile_datums), MAX_TILES};
    buffer_configs[out_id] = {params.buffer_Res[0], llk2::tile_bytes(out_format, llk2::default_tile_datums), MAX_TILES};

    // TEST ONLY: what the reader kernel and the firmware do in tt-metal (see the top of the file).
    DataflowBuffer(lhs_id).preload(params.NUM_TILES);
    DataflowBuffer(rhs_id).preload(params.NUM_TILES);
    DataflowBuffer(out_id).zero_counters();
    llk2_test::harness_barrier(); // every counter written before any thread reads one

    kernel_main();
}
