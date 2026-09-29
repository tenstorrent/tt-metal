// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// PLE stage 5 (`_convolve` + the delta add), one core per column block of T tiles (20 cores x 1 tile), 16-bit dest
// (every op's output is bf16, so the chain's programs run without fp32_dest_acc): per column tile t of the [1,1,4,640]
// row-block
//   acc = multiply(conv[0], tap0)            binary_ng bf16 x bf16 -> the SFPU mul_binary_tile, packed bf16
//   acc = mac(row_k, tap_k, acc), k = 1..3   ttnn.mac on bf16 is the SFPU ternary kernel (ternary_addc_ops_sfpu.cpp
//   with
//                                            the MAC defines; the factory's is_fpu is computed for ADDCMUL/ADDCDIV
//                                            only): a -> dst 0, b -> dst 1, c -> dst 2, mac_tile_init<Float16_b>(),
//                                            mac_tile<Float16_b>(0, 1, 2, 0) = a*b + c in the 16-bit dest, packed bf16
//                                            -- MAC_FORM 0.  MAC_FORM 1 keeps the FPU addc sequence (mul_tiles +
//                                            add_reuse_dest) for reference; it differs on ~18% of lanes.
//   conv = silu(acc)                         unary silu on bf16 (silu_tile<false>)
//   delta = add(gated, conv)                 binary_ng bf16 add (fast_and_approximate default) -> FPU add_tiles
// The rows: conv[0], conv[3], conv[6] (state) and normalized (tap 3).
// CBs: 0 conv0, 1 conv3, 2 conv6, 3 normalized, 4-7 taps 0-3, 8 gated (bf16, T each), 9 / 10 acc ping-pong (bf16, 1),
// 11 silu (bf16, 1), 16 delta (bf16, T).  With INJECT (the layer glue, one tile per core): 18 the delta's four rows as
// four row-0 tiles (built by the writer RISC), 19 the four residual block tiles, 20 the four injected tiles =
// add_tiles(residual_b, delta_row_b) (the layer's add(residual, delta), FPU, 16-bit dest).
// Compile-time args: 0 T, 1 MAC_FORM, 2 DEBUG_STAGE (0 = the delta; 1 = the tap-0 product; 2 = the accumulator after
// the three macs; 3 = the silu: packed to CB 16 instead, for the stage test), 3 INJECT.
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/mac.h"
#include "api/dataflow/dataflow_buffer.h"
#include "../../kernels/zones.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t MAC_FORM = get_compile_time_arg_val(1);
    constexpr uint32_t DEBUG_STAGE = get_compile_time_arg_val(2);
    constexpr uint32_t INJECT = get_compile_time_arg_val(3);
    constexpr uint32_t c_rows = 18, c_resid = 19, c_inj = 20;
    constexpr uint32_t c_row[4] = {0, 1, 2, 3};
    constexpr uint32_t c_tap[4] = {4, 5, 6, 7};
    constexpr uint32_t c_gated = 8, c_acc0 = 9, c_acc1 = 10, c_silu = 11, c_out = 16;
    compute_kernel_hw_startup(c_row[0], c_tap[0], c_acc0);
    DataflowBuffer rows[4] = {DataflowBuffer(0), DataflowBuffer(1), DataflowBuffer(2), DataflowBuffer(3)};
    DataflowBuffer taps[4] = {DataflowBuffer(4), DataflowBuffer(5), DataflowBuffer(6), DataflowBuffer(7)};
    DataflowBuffer gated(c_gated), acc0(c_acc0), acc1(c_acc1), silu(c_silu), out(c_out);
    for (uint32_t i = 0; i < 4; ++i) {
        rows[i].wait_front(T);
        taps[i].wait_front(T);
    }
    gated.wait_front(T);
    for (uint32_t t = 0; t < T; ++t) {
        FUSED_ZONE("fz_pl_conv_c_tile");
        // tap 0: the SFPU multiply of binary_ng (bf16 operands in the dest, fp32 product, bf16 pack)
        tile_regs_acquire();
        reconfig_data_format_srca(c_tap[0], c_row[0]);
        copy_init(c_row[0]);
        copy_tile(c_row[0], t, 0);
        reconfig_data_format_srca(c_row[0], c_tap[0]);
        copy_init(c_tap[0]);
        copy_tile(c_tap[0], t, 1);
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        acc0.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(c_acc0);
        pack_tile(0, c_acc0);
        if constexpr (DEBUG_STAGE == 1) {
            out.reserve_back(1);
            pack_reconfig_data_format(c_out);
            pack_tile(0, c_out);
            out.push_back(1);
        }
        tile_regs_release();
        acc0.push_back(1);
        // taps 1..3: the ternary MAC FPU kernel
        for (uint32_t k = 1; k < 4; ++k) {
            DataflowBuffer& acc_in = (k & 1) ? acc0 : acc1;
            DataflowBuffer& acc_out = (k & 1) ? acc1 : acc0;
            const uint32_t c_in = (k & 1) ? c_acc0 : c_acc1, c_o = (k & 1) ? c_acc1 : c_acc0;
            acc_in.wait_front(1);
            tile_regs_acquire();
            if constexpr (MAC_FORM == 0) {
                // the chain's SFPU mac: a = row_k, b = tap_k, c = the accumulator
                reconfig_data_format_srca(c_in, c_row[k]);
                copy_init(c_row[k]);
                copy_tile(c_row[k], t, 0);
                reconfig_data_format_srca(c_row[k], c_tap[k]);
                copy_init(c_tap[k]);
                copy_tile(c_tap[k], t, 1);
                reconfig_data_format_srca(c_tap[k], c_in);
                copy_init(c_in);
                copy_tile(c_in, 0, 2);
                mac_tile_init<DataFormat::Float16_b>();
                mac_tile<DataFormat::Float16_b>(0, 1, 2, 0);
            } else {
                mul_init(c_row[k], c_tap[k]);
                mul_tiles(c_row[k], c_tap[k], t, t, 0);
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_in);
                add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(c_in, 0, 0);
            }
            tile_regs_commit();
            acc_out.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_o);
            pack_tile(0, c_o);
            if constexpr (DEBUG_STAGE == 2) {
                if (k == 3) {
                    out.reserve_back(1);
                    pack_reconfig_data_format(c_out);
                    pack_tile(0, c_out);
                    out.push_back(1);
                }
            }
            tile_regs_release();
            acc_out.push_back(1);
            acc_in.pop_front(1);
        }
        // silu: taps 1, 2, 3 ping-pong acc0 -> acc1 -> acc0 -> acc1, so k = 3 leaves the accumulator in acc1;
        // then delta = gated + conv
        acc1.wait_front(1);
        tile_regs_acquire();
        reconfig_data_format_srca(c_tap[3], c_acc1);
        copy_init(c_acc1);
        copy_tile(c_acc1, 0, 0);
        silu_tile_init();
        silu_tile<false>(0);
        tile_regs_commit();
        silu.reserve_back(1);
        tile_regs_wait();
        pack_reconfig_data_format(c_silu);
        pack_tile(0, c_silu);
        if constexpr (DEBUG_STAGE == 3) {
            out.reserve_back(1);
            pack_reconfig_data_format(c_out);
            pack_tile(0, c_out);
            out.push_back(1);
        }
        tile_regs_release();
        silu.push_back(1);
        acc1.pop_front(1);
        silu.wait_front(1);
        if constexpr (DEBUG_STAGE == 0) {
            binary_tiles_init<true, EltwiseBinaryType::ELWADD>(c_gated, c_silu);
            tile_regs_acquire();
            add_tiles(c_gated, c_silu, t, 0, 0);
            tile_regs_commit();
            out.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_out);
            pack_tile(0, c_out);
            tile_regs_release();
            out.push_back(1);
        }
        silu.pop_front(1);
    }
    for (uint32_t i = 0; i < 4; ++i) {
        rows[i].pop_front(T);
        taps[i].pop_front(T);
    }
    gated.pop_front(T);
    if constexpr (INJECT && DEBUG_STAGE == 0) {
        FUSED_ZONE("fz_pl_conv_c_inject");
        // the layer's add(residual, delta): per branch b the residual block tile + the delta's row b as a row-0 tile
        DataflowBuffer drows(c_rows), resid(c_resid), inj(c_inj);
        drows.wait_front(4);
        resid.wait_front(4);
        binary_tiles_init<true, EltwiseBinaryType::ELWADD>(c_resid, c_rows);
        for (uint32_t b = 0; b < 4; ++b) {
            tile_regs_acquire();
            add_tiles(c_resid, c_rows, b, b, 0);
            tile_regs_commit();
            inj.reserve_back(1);
            tile_regs_wait();
            pack_reconfig_data_format(c_inj);
            pack_tile(0, c_inj);
            tile_regs_release();
            inj.push_back(1);
        }
        drows.pop_front(4);
        resid.pop_front(4);
    }
}
