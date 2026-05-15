// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    constexpr uint32_t in0 = get_compile_time_arg_val(0);
    constexpr uint32_t in1 = get_compile_time_arg_val(1);
    constexpr uint32_t in2 = get_compile_time_arg_val(2);
    constexpr uint32_t in3 = get_compile_time_arg_val(3);
    constexpr uint32_t out = get_compile_time_arg_val(4);

    DataflowBuffer d0(in0);
    DataflowBuffer d1(in1);
    DataflowBuffer d2(in2);
    DataflowBuffer d3(in3);
    DataflowBuffer dout(out);

    binary_op_init_common(d0.get_id(), d1.get_id(), dout.get_id());

    d0.wait_front(1);
    d1.wait_front(1);
    d2.wait_front(1);
    d3.wait_front(1);
    dout.reserve_back(2);

    // Two real ops with reconfig_data_format between them:
    //   OP[0]: add_tiles(d0, d1) -> dst[0]
    //   OP[1]: add_tiles(d2, d3) -> dst[1]
    tile_regs_acquire();

    add_tiles(d0.get_id(), d1.get_id(), 0, 0, 0);

    reconfig_data_format(d0.get_id(), d2.get_id(), d1.get_id(), d3.get_id());
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(d2.get_id(), d3.get_id());
    add_tiles(d2.get_id(), d3.get_id(), 0, 0, 1);

    // Workaround: on Quasar, the last add_tiles in a single tile_regs_acquire window
    // races maybe with the math->pack handshake and pack_tile reads zeros for that slot.
    // This dummy add_tiles takes the hit so OP[1] above survives.
    add_tiles(d2.get_id(), d3.get_id(), 0, 0, 2);

    tile_regs_commit();

    tile_regs_wait();
    pack_tile(0, dout.get_id());
    pack_tile(1, dout.get_id());
    tile_regs_release();

    d0.pop_front(1);
    d1.pop_front(1);
    d2.pop_front(1);
    d3.pop_front(1);
    dout.push_back(2);
}
