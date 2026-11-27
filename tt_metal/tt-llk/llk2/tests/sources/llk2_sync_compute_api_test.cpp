// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: synchronization through owners with Dest tensors at fixed rows, complete and migrate
// (compute_api/sync.h; tt-llk-lab llk-tensor, report 4 section 5). One kernel body for all three threads.
//
// Four rounds of y = copy(x), z = silu(y), pack(z), each packed to its own copy of the output. Two Dest tensors at
// fixed rows (A: row 0, B: the second half of Dest) take turns, and the SFPU moves between threads:
//   round  Dest  SFPU on   before it
//   0      A     math      -
//   1      B     pack      migrate<Sfpu, Math, Pack>
//   2      A     pack      complete<Math>(out0): A's pack is done, math may write A again
//   3      B     math      migrate<Sfpu, Pack, Math>, complete<Math>(out1)
// Every other hand-off is inside the ops: pack takes z from the SFPU on whichever thread drove it, and a pack-thread
// silu takes y from math.

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

#define LLK2_DEST_FP32 is_fp32_dest_acc_en
#include "compute_api/copy.h"
#include "compute_api/pack.h"
#include "compute_api/sync.h"
#include "compute_api/vec.h"

constexpr DataFormat in_format  = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);

inline constexpr auto& in_spec  = llk2::l1_spec<ROWS, COLS, in_format>;
inline constexpr auto& out_spec = llk2::l1_spec<ROWS, COLS, out_format>;

constexpr std::uint32_t row_a      = 0;                       // Dest tensor A: the first half of Dest
constexpr std::uint32_t row_b      = DEST_REGISTER_HALF_SIZE; // Dest tensor B: the second half
constexpr std::uint32_t out_tiles  = (ROWS / llk2::default_tile_dim) * (COLS / llk2::default_tile_dim);
constexpr llk2::vec_modifiers on_math {.thread = llk2::Thread::Math};
constexpr llk2::vec_modifiers on_pack {.thread = llk2::Thread::Pack};

/// The Dest tensor at row, nobody's: the first op on it waits for no one.
inline auto dest_at(std::uint32_t row)
{
    return llk2::dst_tensor<ROWS, COLS, llk2::kernel_dest_format>(row);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    using llk2::Exu;
    using llk2::Thread;
    const auto x = llk2::Tensor<in_spec>(params.buffer_A[0]);
    const auto out_at = [&](std::uint32_t round) { return llk2::Tensor<out_spec>(params.buffer_Res[round * out_tiles]); };

    const auto out0 = llk2::pack(llk2::vec_silu<on_math>(llk2::copy(x, dest_at(row_a))), out_at(0));

    llk2::migrate<Exu::Sfpu, Thread::Math, Thread::Pack>();
    const auto out1 = llk2::pack(llk2::vec_silu<on_pack>(llk2::copy(x, dest_at(row_b))), out_at(1));

    llk2::complete<Thread::Math>(out0); // A has been packed: math may write it again
    llk2::pack(llk2::vec_silu<on_pack>(llk2::copy(x, dest_at(row_a))), out_at(2));

    llk2::migrate<Exu::Sfpu, Thread::Pack, Thread::Math>();
    llk2::complete<Thread::Math>(out1); // B has been packed
    llk2::pack(llk2::vec_silu<on_math>(llk2::copy(x, dest_at(row_b))), out_at(3));
}
