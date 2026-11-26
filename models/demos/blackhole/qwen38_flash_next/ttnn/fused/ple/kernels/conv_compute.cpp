// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE `_convolve` + the delta add, one core per column block of tiles_per_core tiles, 16-bit dest (every op's output
// is bf16).
// Per column tile of the [1,1,4,640] row-block, each step is the LLK the chain's op runs:
//   acc = multiply(conv[0], tap0)            binary_ng bf16 x bf16 -> SFPU mul_binary_tile, packed bf16
//   acc = mac(row_k, tap_k, acc), k = 1..3   ttnn.mac on bf16 -> SFPU mac_tile<Float16_b> = a*b + c (mac_form 0);
//                                            mac_form 1 is the FPU sequence mul_tiles + add_reuse_dest
//   conv = silu(acc)                         silu_tile<false>
//   delta = add(gated, conv)                 binary_ng bf16 add -> FPU add_tiles
// The rows: conv[0], conv[3], conv[6] (state) and normalized (tap 3).
// CBs: 0 conv0, 1 conv3, 2 conv6, 3 normalized, 4-7 taps 0-3, 8 gated (bf16, tiles_per_core each), 9 / 10 accumulator
// ping-pong (bf16, 1), 11 silu (bf16, 1), 16 delta (bf16, tiles_per_core).  With inject (one tile per core): 18 the
// delta's four rows as four row-0 tiles (built by the writer), 19 the four residual block tiles, 20 the four injected
// tiles = add_tiles(residual_b, delta_row_b).
// Compile-time args: 0 tiles_per_core (the builder's per_core), 1 mac_form, 2 debug_stage (0 = the delta; 1 = the
// tap-0 product; 2 = the accumulator after the three macs; 3 = the silu: packed to CB 16 instead), 3 inject.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/mac.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t tiles_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t mac_form = get_compile_time_arg_val(1);
    constexpr uint32_t debug_stage = get_compile_time_arg_val(2);
    constexpr uint32_t inject = get_compile_time_arg_val(3);
    constexpr uint32_t cb_delta_rows = 18, cb_residual = 19, cb_injected = 20;
    constexpr uint32_t cb_row[4] = {0, 1, 2, 3};
    constexpr uint32_t cb_tap[4] = {4, 5, 6, 7};
    constexpr uint32_t cb_gated = 8, cb_accumulator0 = 9, cb_accumulator1 = 10, cb_silu = 11, cb_out = 16;
    compute_kernel_hw_startup(cb_row[0], cb_tap[0], cb_accumulator0);
    DataflowBuffer rows[4] = {DataflowBuffer(0), DataflowBuffer(1), DataflowBuffer(2), DataflowBuffer(3)};
    DataflowBuffer taps[4] = {DataflowBuffer(4), DataflowBuffer(5), DataflowBuffer(6), DataflowBuffer(7)};
    DataflowBuffer gated(cb_gated), accumulator0(cb_accumulator0), accumulator1(cb_accumulator1), silu(cb_silu),
        out(cb_out);
    for (uint32_t i = 0; i < 4; ++i) {
        rows[i].wait_front(tiles_per_core);
        taps[i].wait_front(tiles_per_core);
    }
    gated.wait_front(tiles_per_core);
    for (uint32_t t = 0; t < tiles_per_core; ++t) {
        // tap 0: the SFPU multiply of binary_ng (bf16 operands in the dest, fp32 product, bf16 pack)
        tile_regs_acquire();
        reconfig_data_format_srca(cb_tap[0], cb_row[0]);
        copy_init(cb_row[0]);
        copy_tile(cb_row[0], t, 0);
        reconfig_data_format_srca(cb_row[0], cb_tap[0]);
        copy_init(cb_tap[0]);
        copy_tile(cb_tap[0], t, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        accumulator0.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_accumulator0);
        pack_tile(0, cb_accumulator0);
        if constexpr (debug_stage == 1) {
            out.reserve_back(1);
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            out.push_back(1);
        }
        tile_regs_release();
        accumulator0.push_back(1);
        // taps 1..3: acc = row_k * tap_k + acc
        for (uint32_t k = 1; k < 4; ++k) {
            DataflowBuffer& accumulator_in = (k & 1) ? accumulator0 : accumulator1;
            DataflowBuffer& accumulator_out = (k & 1) ? accumulator1 : accumulator0;
            const uint32_t cb_accumulator_in = (k & 1) ? cb_accumulator0 : cb_accumulator1,
                           cb_accumulator_out = (k & 1) ? cb_accumulator1 : cb_accumulator0;
            accumulator_in.wait_front(1);
            tile_regs_acquire();
            if constexpr (mac_form == 0) {
                // the chain's SFPU mac: a = row_k, b = tap_k, c = the accumulator
                reconfig_data_format_srca(cb_accumulator_in, cb_row[k]);
                copy_init(cb_row[k]);
                copy_tile(cb_row[k], t, 0);
                reconfig_data_format_srca(cb_row[k], cb_tap[k]);
                copy_init(cb_tap[k]);
                copy_tile(cb_tap[k], t, 1);
                reconfig_data_format_srca(cb_tap[k], cb_accumulator_in);
                copy_init(cb_accumulator_in);
                copy_tile(cb_accumulator_in, 0, 2);
                mac_tile_init<DataFormat::Float16_b>();
                mac_tile<DataFormat::Float16_b>(0, 1, 2, 0);
            } else {
                mul_init(cb_row[k], cb_tap[k]);
                mul_tiles(cb_row[k], cb_tap[k], t, t, 0);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_accumulator_in);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_accumulator_in, 0, 0);
            }
            tile_regs_commit();
            accumulator_out.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_accumulator_out);
            pack_tile(0, cb_accumulator_out);
            if constexpr (debug_stage == 2) {
                if (k == 3) {
                    out.reserve_back(1);
                    pack_reconfig_data_format(cb_out);
                    pack_tile(0, cb_out);
                    out.push_back(1);
                }
            }
            tile_regs_release();
            accumulator_out.push_back(1);
            accumulator_in.pop_front(1);
        }
        // silu: taps 1, 2, 3 ping-pong accumulator0 -> 1 -> 0 -> 1, so k = 3 leaves the accumulator in accumulator1;
        // then delta = gated + conv
        accumulator1.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(cb_tap[3], cb_accumulator1);
        copy_init(cb_accumulator1);
        copy_tile(cb_accumulator1, 0, 0);
        silu_tile_init();
        silu_tile<false>(0);
        tile_regs_commit();
        silu.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(cb_silu);
        pack_tile(0, cb_silu);
        if constexpr (debug_stage == 3) {
            out.reserve_back(1);
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            out.push_back(1);
        }
        tile_regs_release();
        silu.push_back(1);
        accumulator1.pop_front(1);
        silu.wait_front(1);
        if constexpr (debug_stage == 0) {
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_gated, cb_silu);
            tile_regs_acquire();
            add_tiles(cb_gated, cb_silu, t, 0, 0);
            tile_regs_commit();
            out.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_out);
            pack_tile(0, cb_out);
            tile_regs_release();
            out.push_back(1);
        }
        silu.pop_front(1);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        rows[i].pop_front(tiles_per_core);
        taps[i].pop_front(tiles_per_core);
    }
    gated.pop_front(tiles_per_core);
    if constexpr (inject && debug_stage == 0) {
        // the layer's add(residual, delta): per branch b the residual block tile + the delta's row b as a row-0 tile
        DataflowBuffer delta_rows(cb_delta_rows), residual(cb_residual), injected(cb_injected);
        delta_rows.wait_front(4);
        residual.wait_front(4);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(cb_residual, cb_delta_rows);
        for (uint32_t b = 0; b < 4; ++b) {
            tile_regs_acquire();
            add_tiles(cb_residual, cb_delta_rows, b, b, 0);
            tile_regs_commit();
            injected.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(cb_injected);
            pack_tile(0, cb_injected);
            tile_regs_release();
            injected.push_back(1);
        }
        delta_rows.pop_front(4);
        residual.pop_front(4);
    }
}
