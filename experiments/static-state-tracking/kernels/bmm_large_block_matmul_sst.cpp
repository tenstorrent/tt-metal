// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// bmm_large_block matmul — Blackhole compute kernel in the static-state-tracking
// style. Port of tests/tt_metal/tt_metal/test_kernels/compute/matmul_large_block_zm.cpp:
// a blocked matmul out[M x N] = sum over `num_blocks` of in0_block[M x K] *
// in1_block[K x N], with TWO levels of tiling:
//   * K-blocks (num_blocks): the inner dim is accumulated one block at a time via
//     SPILL/RELOAD — the running partial for each output SUB-BLOCK is packed to the
//     intermediate CB (c_24, aliasing c_16) and reloaded into DST at the next K-block.
//   * output sub-blocks (out_subblock_h x out_subblock_w): the MxN output is produced
//     in DST-register-sized sub-blocks.
//
// The block>0 reload is an SST `branch`: the reload arm runs copy_tile (datacopy
// mode), the no-reload arm doesn't, so `phi` widens the engine mode at the join and
// the following matmul reconfigures — load-bearing. Tiled output (Remap=false).
//
// Launched by tests/tt_metal/tt_metal/llk/test_single_core_matmul_compute.cpp
// (TensixBmmLargeBlockStaticState), checked against the identity-matrix golden.

#include <cstddef>
#include <cstdint>

// Include first: brings common_globals.h (MATH/PACK/UNPACK) + get_compile_time_arg_val
// before our defs.h, so our #ifndef-guarded macros defer to the force-included ones.
#include "api/compute/compute_kernel_api.h"
#include "hostdevcommon/kernel_structs.h"  // tt::CBIndex

#include "experiments/static-state-tracking/compute/ops.h"
#include "experiments/static-state-tracking/inc/control.h"

void kernel_main() {
    using namespace sst;
    using namespace sst::compute;
    using namespace sst::tensor;

    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_partials = tt::CBIndex::c_24;

    using TileT = Tile32x32_Float16_b;

    // The ONE explicit setup. Tiled output => no DST remap.
    auto s0 = hw_startup<TileT, TileT, /*Remap=*/false>();

    auto s_final = loop(s0, static_cast<std::size_t>(num_blocks), [&](auto s_blk, std::size_t bi) {
        const uint32_t block = static_cast<uint32_t>(bi);
        const bool do_reload = block > 0;
        const bool last_out = block == (num_blocks - 1);

        auto in0 = Tensor<TileT, Cb>::wait_front(cb_in0, in0_block_num_tiles);
        auto in1 = Tensor<TileT, Cb>::wait_front(cb_in1, in1_block_num_tiles);

        // Sweep the output sub-blocks (in0_num_subblocks rows x in1_num_subblocks cols).
        auto s_sweep = loop(
            s_blk, static_cast<std::size_t>(in0_num_subblocks * in1_num_subblocks), [&](auto s_sb, std::size_t si) {
                const uint32_t sb_r = static_cast<uint32_t>(si) / in1_num_subblocks;
                const uint32_t sb_c = static_cast<uint32_t>(si) % in1_num_subblocks;
                const uint32_t in0_subblock_offset = sb_r * in0_subblock_num_tiles;
                const uint32_t in1_subblock_offset = sb_c * out_subblock_w;

                sst::compute::tile_regs_acquire();

                // Reload this sub-block's running partial (only on K-blocks after the
                // first). The reload arm leaves the engines in datacopy mode; phi widens
                // it at the join so the matmul below reconfigures.
                auto s_rl = branch(
                    s_sb,
                    when(
                        do_reload,
                        [&](auto s) {
                            auto p_in = Tensor<TileT, Cb>::wait_front(cb_partials, out_subblock_num_tiles);
                            auto s_c =
                                loop(s, static_cast<std::size_t>(out_subblock_num_tiles), [&](auto ss, std::size_t i) {
                                    return copy_tile(ss, p_in, static_cast<uint32_t>(i), static_cast<uint32_t>(i));
                                });
                            pop_front(p_in);
                            return s_c;
                        }),
                    otherwise([&](auto s) { return s; }));

                // Matmul-accumulate the sub-block: for each of its out_subblock_num_tiles
                // output tiles (h, w), sum over the in0_block_w inner tiles.
                auto s_mm =
                    loop(s_rl, static_cast<std::size_t>(out_subblock_num_tiles), [&](auto s_t, std::size_t dst_idx) {
                        const uint32_t h = static_cast<uint32_t>(dst_idx) / out_subblock_w;
                        const uint32_t w = static_cast<uint32_t>(dst_idx) % out_subblock_w;
                        return loop(s_t, static_cast<std::size_t>(in0_block_w), [&](auto s_k, std::size_t inner) {
                            const uint32_t in0_idx =
                                in0_subblock_offset + h * in0_block_w + static_cast<uint32_t>(inner);
                            const uint32_t in1_idx =
                                in1_subblock_offset + static_cast<uint32_t>(inner) * in1_per_core_w + w;
                            return matmul(s_k, in0, in1, in0_idx, in1_idx, static_cast<uint32_t>(dst_idx));
                        });
                    });
                sst::compute::tile_regs_commit();

                sst::compute::tile_regs_wait();
                // Last K-block writes the final output CB; earlier blocks spill to
                // partials. Same Default pack either way (runtime CB select).
                auto out_t = last_out ? Tensor<TileT, Cb>::reserve_back(cb_out, out_subblock_num_tiles)
                                      : Tensor<TileT, Cb>::reserve_back(cb_partials, out_subblock_num_tiles);
                auto s_pk = loop(s_mm, static_cast<std::size_t>(out_subblock_num_tiles), [&](auto s, std::size_t i) {
                    return pack_tile(s, out_t, static_cast<uint32_t>(i));
                });
                sst::compute::tile_regs_release();
                push_back(out_t);

                return s_pk;
            });

        pop_front(in0);
        pop_front(in1);
        return s_sweep;
    });
    (void)s_final;
}
