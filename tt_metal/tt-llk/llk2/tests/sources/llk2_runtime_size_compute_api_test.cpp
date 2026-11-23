// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Compute API test: a tensor size known only at RUN TIME through an op, from the Tensors to
// the HAL (resolver/work_counts.h). One build serves every count up to the bound: COUNT is a run-time parameter of the
// harness, so every count runs the same three ELFs.
//
// KIND 0  add / sub / mul (OP: 0, 1, 2) of a and b of one tile row with a run-time number of tile columns:
//         [1, COUNT] of at most [1, CT_DIM]; the result is packed to an L1 tensor of the same run-time size
// KIND 1  the same with a run-time number of tile rows: [COUNT, CT_DIM] of at most [RT_DIM, CT_DIM]
// KIND 2  matmul with a run-time K: a [1, COUNT] x b [COUNT, CT_DIM], COUNT at most KT_DIM; the [1, CT_DIM] result
//         is packed as a compile-time tensor
// KIND 3  matmul with a run-time N through views: a [1, KT_DIM] x the first COUNT tile columns of b [KT_DIM, CT_DIM]
//         (llk2::cols_up_to<CT_DIM>(b, 0, COUNT): a view of several tile rows whose row step stays b's), packed to the
//         first COUNT tiles of the output row. The result is a Dest tensor that holds COUNT, and pack checks it
// KIND 4  a sum pack of a run-time number of tiles: a + b over the first COUNT tile columns of two [1, CT_DIM] rows,
//         every tile of the result added into one L1 tile (pack .sum_tiles)
// KIND 5  reduce over a run-time row length: the row max of the first COUNT tile columns of a [RT_DIM, CT_DIM]
//         (BF16; b is the scaler tile), one output tile per tile row
// KIND 6  sub with a column broadcast over a run-time number of tiles (SDPA's S - m): the first COUNT tile columns of a
//         [1, CT_DIM] row minus column 0 of the one tile b, packed to the first COUNT tiles of the output row
// KIND 7  an SFPU function over a run-time number of tiles (SDPA's P block): the fast exp, on the pack thread, of
//         a + b over the first COUNT tile columns of two [1, CT_DIM] rows (b is zeros), packed with ReLU. With
//         CT_DIM > 1 the exp walks the block in one call; with CT_DIM 1 (a bound of 1) tile by tile
// KIND 8  matmul with a run-time K through views, in0 of several tile rows (SDPA's P V): the first COUNT tile columns
//         of a [RT_DIM, KT_DIM] times the first COUNT tile rows of b [KT_DIM, CT_DIM]; a's tile rows stay KT_DIM tiles
//         apart. The [RT_DIM, CT_DIM] result is packed as a compile-time tensor
// The Dest tensor is the bound's size in every kind: a Dest section does not shrink.

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

#define LLK2_DEST_FP32     is_fp32_dest_acc_en
#define LLK2_MATH_FIDELITY static_cast<std::uint32_t>(MATH_FIDELITY)
#include "compute_api/dest.h"
#include "compute_api/eltwise_binary.h"
#include "compute_api/matmul.h"
#include "compute_api/pack.h"
#include "compute_api/reduce.h"
#include "compute_api/vec.h"

constexpr DataFormat in0_format = static_cast<DataFormat>(UNPACK_A_IN);
constexpr DataFormat in1_format = static_cast<DataFormat>(UNPACK_B_IN);
constexpr DataFormat out_format = static_cast<DataFormat>(PACK_OUT);
constexpr std::uint32_t tile    = llk2::default_tile_dim;

/// S with the size KIND makes run time: its tile rows (KIND 1) or its tile columns (KIND 0).
inline constexpr llk2::Dim runtime_dim = KIND == 1 ? llk2::TileY : llk2::TileX;
template <const auto& S>
inline constexpr const llk2::TensorSpec<llk2::tensor_rank>& runtime_grid = llk2::runtime_size_spec<S, runtime_dim>;

// KIND 0 and 1: [RT_DIM, CT_DIM] tiles at most, the rows or the columns a run-time count. Named here, outside any
// template: GCC takes a reference variable template as a template argument only through a plain reference variable
// (as a kernel names its specs) or in a dependent context.
inline constexpr const auto& a_spec   = runtime_grid<llk2::l1_spec<RT_DIM * tile, CT_DIM * tile, in0_format>>;
inline constexpr const auto& b_spec   = runtime_grid<llk2::l1_spec<RT_DIM * tile, CT_DIM * tile, in1_format>>;
inline constexpr const auto& out_spec = runtime_grid<llk2::l1_spec<RT_DIM * tile, CT_DIM * tile, out_format>>;

// One build for every count: the Plan holds the bound, and the Work names where the count comes from.
constexpr llk2::Work add_work = llk2::resolved<llk2::add_op, a_spec, b_spec, llk2::add_modifiers {}>.work;
static_assert(add_work.rows_from != llk2::CountSource::Fixed || add_work.cols_from != llk2::CountSource::Fixed);

/// KIND 0 and 1: y = a <OP> b for `count` tile rows or columns, packed to out. A, B and O are a_spec, b_spec and
/// out_spec (template parameters for the reason above).
template <const auto& A, const auto& B, const auto& O, std::uint32_t Op>
inline __attribute__((always_inline)) void eltwise(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    static_assert(KIND == 1 || RT_DIM == 1, "KIND 0: one tile row");
    const auto a   = llk2::Tensor<A>(a_address, llk2::NoRelease {}, count);
    const auto b   = llk2::Tensor<B>(b_address, llk2::NoRelease {}, count);
    const auto out = llk2::Tensor<O>(out_address, llk2::NoRelease {}, count);
    using AccRing  = llk2::DestRing<llk2::dst_spec<RT_DIM * tile, CT_DIM * tile, llk2::kernel_dest_format>>; // the bound's size
    const auto acc = llk2::reserve_back<AccRing>();
    if constexpr (Op == 2)
    {
        llk2::pack(llk2::mul(a, b, acc), out);
    }
    else if constexpr (Op == 1)
    {
        llk2::pack(llk2::sub(a, b, acc), out);
    }
    else
    {
        llk2::pack(llk2::add(a, b, acc), out);
    }
    llk2::release(acc);
}

/// KIND 2: y = a b over `count` K tiles, packed to out.
template <std::uint32_t Kt>
inline __attribute__((always_inline)) void matmul_k(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a   = llk2::Tensor<llk2::runtime_cols<llk2::l1_spec<tile, Kt * tile, in0_format>>>(a_address, llk2::NoRelease {}, count);          // [1, K]
    const auto b   = llk2::Tensor<llk2::runtime_rows<llk2::l1_spec<Kt * tile, CT_DIM * tile, in1_format>>>(b_address, llk2::NoRelease {}, count); // [K, N]
    const auto out = llk2::Tensor<llk2::l1_spec<tile, CT_DIM * tile, out_format>>(out_address);
    using AccRing  = llk2::DestRing<llk2::dst_spec<tile, CT_DIM * tile, llk2::kernel_dest_format>>;
    const auto y   = llk2::matmul(a, b, llk2::reserve_back<AccRing>());
    llk2::pack(y, out);
    llk2::release(y);
}

/// KIND 3: y = a (b's first `count` tile columns), packed to the output row's first `count` tiles.
template <std::uint32_t Kt>
inline __attribute__((always_inline)) void matmul_n(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a   = llk2::Tensor<llk2::l1_spec<tile, Kt * tile, in0_format>>(a_address);          // [1, K]
    const auto b   = llk2::Tensor<llk2::l1_spec<Kt * tile, CT_DIM * tile, in1_format>>(b_address); // [K, CT_DIM]
    const auto out = llk2::Tensor<llk2::l1_spec<tile, CT_DIM * tile, out_format>>(out_address);
    using AccRing  = llk2::DestRing<llk2::dst_spec<tile, CT_DIM * tile, llk2::kernel_dest_format>>;
    const auto y   = llk2::matmul(a, llk2::cols_up_to<CT_DIM>(b, 0, count), llk2::reserve_back<AccRing>());
    llk2::pack(y, llk2::cols_up_to<CT_DIM>(out, 0, count));
    llk2::release(y);
}

/// KIND 4: out (one tile) = the sum of the first `count` tiles of a + b.
template <std::uint32_t Ct>
inline __attribute__((always_inline)) void sum_of_tiles(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a   = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, in0_format>>(a_address);
    const auto b   = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, in1_format>>(b_address);
    const auto out = llk2::Tensor<llk2::l1_spec<tile, tile, out_format>>(out_address);
    using AccRing  = llk2::DestRing<llk2::dst_spec<tile, Ct * tile, llk2::kernel_dest_format>>;
    const auto y   = llk2::add(llk2::cols_up_to<Ct>(a, 0, count), llk2::cols_up_to<Ct>(b, 0, count), llk2::reserve_back<AccRing>());
    llk2::pack<llk2::pack_modifiers {.sum_tiles = true}>(y, out);
    llk2::release(y);
}

/// KIND 5: out [RT_DIM, 1] = the row max of the first `count` tile columns of a [RT_DIM, CT_DIM]; b is the scaler.
template <std::uint32_t Ct>
inline __attribute__((always_inline)) void row_max_of(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a      = llk2::Tensor<llk2::l1_spec<RT_DIM * tile, Ct * tile, in0_format>>(a_address);
    const auto scaler = llk2::Tensor<llk2::l1_spec<tile, tile, in1_format>>(b_address);
    const auto out    = llk2::Tensor<llk2::l1_spec<RT_DIM * tile, tile, out_format>>(out_address);
    using AccRing     = llk2::DestRing<llk2::dst_spec<RT_DIM * tile, tile, llk2::kernel_dest_format>>;
    const auto y =
        llk2::reduce<llk2::reduce_modifiers {.kind = llk2::ReduceKind::Max}>(llk2::cols_up_to<Ct>(a, 0, count), scaler, llk2::reserve_back<AccRing>());
    llk2::pack(y, out);
    llk2::release(y);
}

/// KIND 6: out's first `count` tiles = a's first `count` tiles minus column 0 of the tile b.
template <std::uint32_t Ct>
inline __attribute__((always_inline)) void minus_column(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a   = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, in0_format>>(a_address);
    const auto m   = llk2::Tensor<llk2::l1_spec<tile, tile, in1_format>>(b_address);
    const auto out = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, out_format>>(out_address);
    using AccRing  = llk2::DestRing<llk2::dst_spec<tile, Ct * tile, llk2::kernel_dest_format>>;
    const auto y   = llk2::sub<llk2::sub_modifiers {.bcast = llk2::Broadcast::Col}>(llk2::cols_up_to<Ct>(a, 0, count), m, llk2::reserve_back<AccRing>());
    llk2::pack(y, llk2::cols_up_to<Ct>(out, 0, count));
    llk2::release(y);
}

/// KIND 7: out's first `count` tiles = max(exp(a + b), 0) of the first `count` tiles, the exp on the pack thread.
template <std::uint32_t Ct>
inline __attribute__((always_inline)) void exp_of(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a   = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, in0_format>>(a_address);
    const auto b   = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, in1_format>>(b_address);
    const auto out = llk2::Tensor<llk2::l1_spec<tile, Ct * tile, out_format>>(out_address);
    using AccRing  = llk2::DestRing<llk2::dst_spec<tile, Ct * tile, llk2::kernel_dest_format>>;
    constexpr llk2::vec_modifiers fast_on_pack {.form = llk2::VecForm::Fast, .clamp_input = false, .thread = llk2::Thread::Pack};
    const auto y = llk2::vec_exp<fast_on_pack>(llk2::add(llk2::cols_up_to<Ct>(a, 0, count), llk2::cols_up_to<Ct>(b, 0, count), llk2::reserve_back<AccRing>()));
    llk2::pack<llk2::pack_modifiers {.relu = true}>(y, llk2::cols_up_to<Ct>(out, 0, count));
    llk2::release(y);
}

/// KIND 8: out [RT_DIM, CT_DIM] = (a's first `count` tile columns) (b's first `count` tile rows).
template <std::uint32_t Kt>
inline __attribute__((always_inline)) void matmul_k_rows(std::uint32_t a_address, std::uint32_t b_address, std::uint32_t out_address, std::uint32_t count)
{
    const auto a   = llk2::Tensor<llk2::l1_spec<RT_DIM * tile, Kt * tile, in0_format>>(a_address); // [RT_DIM, K]
    const auto b   = llk2::Tensor<llk2::l1_spec<Kt * tile, CT_DIM * tile, in1_format>>(b_address); // [K, CT_DIM]
    const auto out = llk2::Tensor<llk2::l1_spec<RT_DIM * tile, CT_DIM * tile, out_format>>(out_address);
    using AccRing  = llk2::DestRing<llk2::dst_spec<RT_DIM * tile, CT_DIM * tile, llk2::kernel_dest_format>>;
    const auto y   = llk2::matmul(llk2::cols_up_to<Kt>(a, 0, count), llk2::rows_up_to<Kt>(b, 0, count), llk2::reserve_back<AccRing>());
    llk2::pack(y, out);
    llk2::release(y);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const std::uint32_t count = params.COUNT;
    if constexpr (KIND == 8)
    {
        matmul_k_rows<KT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else if constexpr (KIND == 7)
    {
        exp_of<CT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else if constexpr (KIND == 6)
    {
        minus_column<CT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else if constexpr (KIND == 5)
    {
        row_max_of<CT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else if constexpr (KIND == 4)
    {
        sum_of_tiles<CT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else if constexpr (KIND == 3)
    {
        matmul_n<KT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else if constexpr (KIND == 2)
    {
        matmul_k<KT_DIM>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
    else
    {
        eltwise<a_spec, b_spec, out_spec, OP>(params.buffer_A[0], params.buffer_B[0], params.buffer_Res[0], count);
    }
}
