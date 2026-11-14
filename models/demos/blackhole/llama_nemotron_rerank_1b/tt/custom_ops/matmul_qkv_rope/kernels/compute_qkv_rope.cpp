// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// QKV projection + RoPE: minimal_matmul's compute.cpp (blocked matmul, in0 reuse across N
// blocks) with an output stage that rotates Q and K. Weights are in HF order, so each 64-wide
// head is two tiles (lo = dims 0-31, hi = dims 32-63) and rotate_half pairs dim j with j + 32:
//   lo' = lo * cos - hi * sin,   hi' = hi * cos + lo * sin
// cos / sin are one tile each per 32-token sequence tile (the 32 frequencies; CB 4 holds cos, sin
// and -sin for each of the M block's tile-rows). Output tiles at N >= rope_tiles (V)
// are copied. The writer scatters tiles into the Q / K / V head layouts.
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#ifdef ZONES
#include "api/debug/kernel_profiler.hpp"
#endif

namespace {
constexpr uint32_t in0_cb = tt::CBIndex::c_0;
constexpr uint32_t in1_cb = tt::CBIndex::c_1;
constexpr uint32_t out_cb = tt::CBIndex::c_2;
constexpr uint32_t acc_cb = tt::CBIndex::c_3;
constexpr uint32_t cs_cb = tt::CBIndex::c_4;
// The rotation's multiplies run at HiFi4 whatever the matmul's fidelity (mul_tiles takes the
// kernel's MATH_FIDELITY); these mirror mul_init / mul_tiles with the fidelity fixed.
#ifdef ROPE_FIDELITY
constexpr auto EPI_FIDELITY = MathFidelity::ROPE_FIDELITY;
#else
constexpr auto EPI_FIDELITY = MathFidelity::HiFi4;
#endif
}  // namespace

ALWI void mul_hifi_init(uint32_t icb0, uint32_t icb1) {
    state_configure(icb0, icb1, __builtin_LINE());
    MATH((
        llk_math_eltwise_binary_init<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, EPI_FIDELITY>(icb0, icb1, false)));
    UNPACK((llk_unpack_AB_init<BroadcastType::NONE>(icb0, icb1, Transpose::None)));
}

ALWI void mul_hifi_tile(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst, bool clear) {
    UNPACK((llk_unpack_AB(icb0, icb1, itile0, itile1)));
    MATH((llk_math_eltwise_binary<
          EltwiseBinaryType::ELWMUL,
          BroadcastType::NONE,
          DST_ACCUM_MODE,
          EPI_FIDELITY,
          EltwiseBinaryReuseDestType::NONE>(icb0, icb1, idst, clear)));
}

// From minimal_matmul compute.cpp.
void matmul_blocks(
    const uint32_t M_block_tiles,
    const uint32_t N_block_tiles,
    const uint32_t full_N_block_tiles,
    const uint32_t K_block_tiles,
    const uint32_t subblock_h,
    const uint32_t subblock_w) {
    uint32_t in0_index_offset = 0;
    for (uint32_t M_start = 0; M_start < M_block_tiles; M_start += subblock_h) {
        uint32_t in1_index_offset = 0;
        for (uint32_t N_start = 0; N_start < N_block_tiles; N_start += subblock_w) {
            tile_regs_acquire();
            uint32_t dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;
            for (uint32_t inner_dim = 0; inner_dim < K_block_tiles; inner_dim++) {
                matmul_block(
                    in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, K_block_tiles);
                in0_index++;
                in1_index += full_N_block_tiles;
            }
            tile_regs_commit();
            tile_regs_wait();
            uint32_t write_dst_index = 0;
            for (uint32_t h = 0; h < subblock_h; h++) {
                for (uint32_t w = 0; w < subblock_w; w++) {
                    pack_tile<true>(write_dst_index, acc_cb, (M_start + h) * full_N_block_tiles + N_start + w);
                    write_dst_index++;
                }
            }
            tile_regs_release();
            in1_index_offset += subblock_w;
        }
        in0_index_offset += subblock_h * K_block_tiles;
    }
}

// Rotate (or copy) one output block from the accumulator into the output CB, a tile-row at a time.
// CB 4 holds cos, sin, -sin per tile-row. Each rotated tile is two multiplies into one DST slot,
// the second accumulating (clear = false): lo' = lo * cos + hi * (-sin), hi' = hi * cos + lo * sin.
// A block lies entirely in Q / K or entirely in V (N blocks are whole heads, Q / K / V are whole
// heads), so the init happens once per block.
template <uint32_t M_block_tiles, uint32_t N_block_tiles>
void rope_block(uint32_t n_tile, uint32_t rope_tiles) {
    CircularBuffer cb_out(out_cb);
    pack_reconfig_data_format(out_cb);
#ifdef NO_ROPE
    const bool rotate = false;  // profiling: the matmul with a plain copy-out
#else
    const bool rotate = n_tile < rope_tiles;
#endif
    if (rotate) {
        reconfig_data_format(acc_cb, cs_cb);
        mul_hifi_init(acc_cb, cs_cb);
    } else {
        reconfig_data_format_srca(acc_cb);
        copy_init(acc_cb);
    }
    for (uint32_t m = 0; m < M_block_tiles; ++m) {
        cb_out.reserve_back(N_block_tiles);
        const uint32_t c = 3 * m, sn = c + 1, nsn = c + 2;
        for (uint32_t n = 0; n < N_block_tiles; n += 2) {
            const uint32_t lo = m * N_block_tiles + n;
            tile_regs_acquire();
            if (rotate) {
                mul_hifi_tile(acc_cb, cs_cb, lo, c, 0, true);         // lo * cos
                mul_hifi_tile(acc_cb, cs_cb, lo + 1, nsn, 0, false);  // + hi * (-sin)
                mul_hifi_tile(acc_cb, cs_cb, lo + 1, c, 1, true);     // hi * cos
                mul_hifi_tile(acc_cb, cs_cb, lo, sn, 1, false);       // + lo * sin
            } else {
                copy_tile(acc_cb, lo, 0);
                copy_tile(acc_cb, lo + 1, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, out_cb, n);
            pack_tile<true>(1, out_cb, n + 1);
            tile_regs_release();
        }
        cb_out.push_back(N_block_tiles);
    }
}

void kernel_main() {
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(0);
    constexpr uint32_t M_block_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t N_block_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
    constexpr uint32_t N_blocks_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t subblock_h = get_compile_time_arg_val(6);
    constexpr uint32_t subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t rope_tiles = get_compile_time_arg_val(8);
    static_assert(N_block_tiles % 2 == 0, "N block must hold whole heads (two tiles)");

    const uint32_t M_start_tile = get_arg_val<uint32_t>(0);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(1);
    const uint32_t N_start_tile = get_arg_val<uint32_t>(2);
    const uint32_t N_end_tile = get_arg_val<uint32_t>(3);

    CircularBuffer cb_in0(in0_cb), cb_in1(in1_cb), cb_acc(acc_cb), cb_cs(cs_cb);

    compute_kernel_hw_startup<SrcOrder::Reverse>(in0_cb, in1_cb, acc_cb);
    matmul_init(in0_cb, in1_cb);

    constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
    constexpr uint32_t in1_block_num_tiles = K_block_tiles * N_block_tiles;
    constexpr uint32_t out_block_num_tiles = M_block_tiles * N_block_tiles;

    bool reuse_in0_block = false;
    for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
        const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
        const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
        const uint32_t current_M_block_tiles = m_tile_end - m_tile;
        const uint32_t current_subblock_h = std::min(current_M_block_tiles, subblock_h);
        cb_cs.wait_front(3 * M_block_tiles);

        for (uint32_t n_block_iter = 0; n_block_iter < N_blocks_per_core; n_block_iter++) {
            const uint32_t n_tile = N_start_tile + n_block_iter * N_block_tiles;
            const uint32_t n_tile_end = std::min(n_tile + N_block_tiles, N_end_tile);
            const uint32_t current_N_block_tiles = n_tile_end - n_tile;
            const uint32_t current_subblock_w = std::min(current_N_block_tiles, subblock_w);

            reconfig_data_format(in1_cb, in0_cb);
            pack_reconfig_data_format(acc_cb);
            matmul_block_init(in0_cb, in1_cb, false, current_subblock_w, current_subblock_h, K_block_tiles);
            cb_acc.reserve_back(out_block_num_tiles);
            for (uint32_t k_block = 0; k_block < K_num_blocks; k_block++) {
                {
#ifdef ZONES
                    DeviceZoneScopedN("W0");
#endif
                    cb_in0.wait_front(in0_block_num_tiles);
                }
                {
#ifdef ZONES
                    DeviceZoneScopedN("W1");
#endif
                    cb_in1.wait_front(in1_block_num_tiles);
                }
                matmul_blocks(
                    current_M_block_tiles,
                    current_N_block_tiles,
                    N_block_tiles,
                    K_block_tiles,
                    current_subblock_h,
                    current_subblock_w);
                if (k_block == K_num_blocks - 1 && n_block_iter < N_blocks_per_core - 1) {
                    // the next N block starts with this in0 block again (the in0 kernel skips its read)
                    reuse_in0_block = true;
                }
                if (!reuse_in0_block) {
                    cb_in0.pop_front(in0_block_num_tiles);
                }
                cb_in1.pop_front(in1_block_num_tiles);
                reuse_in0_block = false;
                if (k_block == 0) {
                    PACK((llk_pack_reconfig_l1_acc(1)));
                }
            }
            cb_acc.push_back(out_block_num_tiles);
            PACK((llk_pack_reconfig_l1_acc(0)));

            cb_acc.wait_front(out_block_num_tiles);
#ifdef ZONES
            DeviceZoneScopedN("EP");
#endif
            rope_block<M_block_tiles, N_block_tiles>(n_tile, rope_tiles);
            cb_acc.pop_front(out_block_num_tiles);
        }
        cb_cs.pop_front(3 * M_block_tiles);
    }
}
