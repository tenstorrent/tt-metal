// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t in0 = get_compile_time_arg_val(0);
    constexpr uint32_t in1 = get_compile_time_arg_val(1);
    constexpr uint32_t in2 = get_compile_time_arg_val(2);
    constexpr uint32_t in3 = get_compile_time_arg_val(3);
    constexpr uint32_t in4 = get_compile_time_arg_val(4);
    constexpr uint32_t in5 = get_compile_time_arg_val(5);
    constexpr uint32_t out = get_compile_time_arg_val(6);

    DataflowBuffer d0(in0);
    DataflowBuffer d1(in1);
    DataflowBuffer d2(in2);
    DataflowBuffer d3(in3);
    DataflowBuffer d4(in4);
    DataflowBuffer d5(in5);
    DataflowBuffer dout(out);

    mm_init(d0.get_id(), d1.get_id(), dout.get_id());

    d0.wait_front(1);
    d1.wait_front(1);
    d2.wait_front(1);
    d3.wait_front(1);
    d4.wait_front(1);
    d5.wait_front(1);
    dout.reserve_back(3);

    // Three matmul_tiles ops with reconfig_data_format between operand pairs.
    // OP[1] uses Float32 operands; OP[0] and OP[2] use Float16_b.
    // Between ops we reconfig srcA/srcB formats and reprime the unpack MOP for
    // the new buf descriptors (mm_init_short is a stub on Quasar).
    //   OP[0]: matmul_tiles(d0, d1) -> dst[0]   [Float16_b srcA/B]
    //   OP[1]: matmul_tiles(d2, d3) -> dst[1]   [Float32   srcA/B, reconfig Float16_b -> Float32]
    //   OP[2]: matmul_tiles(d4, d5) -> dst[2]   [Float16_b srcA/B, reconfig Float32   -> Float16_b]
    tile_regs_acquire();

    matmul_tiles(d0.get_id(), d1.get_id(), 0, 0, 0);

    reconfig_data_format(d0.get_id(), d2.get_id(), d1.get_id(), d3.get_id());
    UNPACK((llk_unpack_AB_matmul_init(d2.get_id(), d3.get_id())));
    matmul_tiles(d2.get_id(), d3.get_id(), 0, 0, 1);

    reconfig_data_format(d2.get_id(), d4.get_id(), d3.get_id(), d5.get_id());
    UNPACK((llk_unpack_AB_matmul_init(d4.get_id(), d5.get_id())));
    matmul_tiles(d4.get_id(), d5.get_id(), 0, 0, 2);

    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dout.get_id());
    pack_tile(1, dout.get_id());
    pack_tile(2, dout.get_id());
    tile_regs_release();

    d0.pop_front(1);
    d1.pop_front(1);
    d2.pop_front(1);
    d3.pop_front(1);
    d4.pop_front(1);
    d5.pop_front(1);
    dout.push_back(3);
}
