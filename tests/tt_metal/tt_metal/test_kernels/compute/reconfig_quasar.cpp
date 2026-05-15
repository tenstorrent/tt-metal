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
    // followed by a dummy trailing add_tiles to dst[2] whose result is never packed.
    //
    // Why the dummy: on Quasar the public Compute API has a math->pack drain race — the
    // LAST pack_tile in a tile_regs_acquire/commit/wait/release section reads zero because
    // the preceding math op hasn't finished writing dest by the time tile_regs_commit
    // signals math-done to pack. The trailing add_tiles gives OP[1] time to drain its MOP
    // before the math-pack handoff. TODO: remove once the LLK math->pack handshake is fixed.
    tile_regs_acquire();

    add_tiles(d0.get_id(), d1.get_id(), 0, 0, 0);

    reconfig_data_format(d0.get_id(), d2.get_id(), d1.get_id(), d3.get_id());
    binary_tiles_init<true, EltwiseBinaryType::ELWADD>(d2.get_id(), d3.get_id());
    add_tiles(d2.get_id(), d3.get_id(), 0, 0, 1);

    // Drain dummy: not packed, only here to absorb the lost-last-pack race.
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
