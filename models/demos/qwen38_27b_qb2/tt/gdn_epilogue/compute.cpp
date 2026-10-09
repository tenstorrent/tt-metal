// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"
// DFB_BINDINGS
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void square(uint32_t n, DataflowBuffer& tmp) {
    tmp.reserve_back(n);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::x, dfb::x);
    mul_init(dfb::x, dfb::x, false);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        mul_tiles(dfb::x, dfb::x, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(n);
}

void inverse_rms(DataflowBuffer& inv) {
    inv.reserve_back(1);
    pack_reconfig_data_format(dfb::inv);
    reconfig_data_format(dfb::stats, dfb::epsilon);
    add_init(dfb::stats, dfb::epsilon);
    tile_regs_acquire();
    add_tiles(dfb::stats, dfb::epsilon, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, dfb::inv, 0);
    tile_regs_release();
    inv.push_back(1);
}

void scale_by_inverse_rms(uint32_t Vt, DataflowBuffer& norm) {
    norm.reserve_back(Vt);
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format(dfb::x, dfb::inv);
    mul_bcast_cols_init(dfb::x, dfb::inv);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_cols(dfb::x, dfb::inv, i, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::norm, i);
        tile_regs_release();
    }
    norm.push_back(Vt);
}

void apply_weight(uint32_t Vt, DataflowBuffer& tmp) {
    tmp.reserve_back(Vt);
    pack_reconfig_data_format(dfb::tmp);
    reconfig_data_format(dfb::norm, dfb::weight);
    mul_bcast_rows_init(dfb::norm, dfb::weight);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles_bcast_rows(dfb::norm, dfb::weight, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::tmp, i);
        tile_regs_release();
    }
    tmp.push_back(Vt);
}

void activate_gate(uint32_t Vt, DataflowBuffer& norm) {
    norm.reserve_back(Vt);
    pack_reconfig_data_format(dfb::norm);
    reconfig_data_format_srca(dfb::gate);
    copy_init(dfb::gate);
    sigmoid_tile_init();
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        copy_tile(dfb::gate, i, 0);
        sigmoid_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::norm, i);
        tile_regs_release();
    }
    norm.push_back(Vt);
}

void multiply_rounded(uint32_t Vt, DataflowBuffer& rounded) {
    rounded.reserve_back(Vt);
    pack_reconfig_data_format(dfb::rounded);
    reconfig_data_format(dfb::tmp, dfb::norm);
    mul_init(dfb::tmp, dfb::norm);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        mul_tiles(dfb::tmp, dfb::norm, i, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::rounded, i);
        tile_regs_release();
    }
    rounded.push_back(Vt);
}

void multiply_output(uint32_t Vt, DataflowBuffer& out) {
    out.reserve_back(Vt);
    pack_reconfig_data_format(dfb::out);
    for (uint32_t i = 0; i < Vt; i++) {
        tile_regs_acquire();
        reconfig_data_format_srca(dfb::rounded);
        copy_init(dfb::rounded);
        copy_tile(dfb::rounded, i, 0);
        reconfig_data_format_srca(dfb::gate);
        copy_init(dfb::gate);
        copy_tile(dfb::gate, i, 1);
        // Match binary_ng's SFPU multiply and explicit BF16 RNE. Relying on
        // the FPU/packer alone changed the final product by one BF16 ULP in
        // the simulator despite a bit-identical pre-multiply norm result.
        mul_binary_tile_init();
        mul_binary_tile(0, 1, 0);
        typecast_tile_init<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>();
        typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out, i);
        tile_regs_release();
    }
    out.push_back(Vt);
}

void kernel_main() {
    constexpr uint32_t Vt = 4;
    constexpr bool multiply_z = get_compile_time_arg_val(0) != 0;
    const uint32_t wi_count = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(dfb::x, dfb::scaler, dfb::out);
    DataflowBuffer x(dfb::x);
    DataflowBuffer gate(dfb::gate);
    DataflowBuffer weight(dfb::weight);
    DataflowBuffer tmp(dfb::tmp);
    DataflowBuffer stats(dfb::stats);
    DataflowBuffer inv(dfb::inv);
    DataflowBuffer norm(dfb::norm);
    DataflowBuffer out(dfb::out);
    DataflowBuffer rounded(dfb::rounded);
    DataflowBuffer scaler(dfb::scaler);
    DataflowBuffer epsilon(dfb::epsilon);
    weight.wait_front(Vt);
    scaler.wait_front(1);
    epsilon.wait_front(1);
    for (uint32_t i = 0; i < wi_count; i++) {
        x.wait_front(Vt);
        gate.wait_front(Vt);
        square(Vt, tmp);
        compute_kernel_lib::
            reduce<ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, dfb::tmp, dfb::scaler, dfb::stats>(
                compute_kernel_lib::ReduceInputBlockShape::of(1, Vt));
        stats.wait_front(1);
        inverse_rms(inv);
        inv.wait_front(1);
        scale_by_inverse_rms(Vt, norm);
        norm.wait_front(Vt);
        x.pop_front(Vt);
        inv.pop_front(1);
        stats.pop_front(1);
        apply_weight(Vt, tmp);
        tmp.wait_front(Vt);
        norm.pop_front(Vt);
        activate_gate(Vt, norm);
        norm.wait_front(Vt);
        multiply_rounded(Vt, rounded);
        rounded.wait_front(Vt);
        if constexpr (multiply_z) {
            multiply_output(Vt, out);
        } else {
            out.reserve_back(Vt);
            pack_reconfig_data_format(dfb::out);
            reconfig_data_format_srca(dfb::rounded);
            copy_init(dfb::rounded);
            for (uint32_t tile = 0; tile < Vt; ++tile) {
                tile_regs_acquire();
                copy_tile(dfb::rounded, tile, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb::out, tile);
                tile_regs_release();
            }
            out.push_back(Vt);
        }
        rounded.pop_front(Vt);
        gate.pop_front(Vt);
        tmp.pop_front(Vt);
        norm.pop_front(Vt);
    }
}
