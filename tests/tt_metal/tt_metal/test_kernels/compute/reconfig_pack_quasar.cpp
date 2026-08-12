// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    DataflowBuffer d0(dfb::in0);
    DataflowBuffer d1(dfb::in1);
    DataflowBuffer d2(dfb::in2);
    DataflowBuffer d3(dfb::in3);
    DataflowBuffer d4(dfb::in4);
    DataflowBuffer d5(dfb::in5);
    DataflowBuffer out0(dfb::out0);
    DataflowBuffer out1(dfb::out1);
    DataflowBuffer out2(dfb::out2);

    compute_kernel_hw_startup<SrcOrder::Reverse>(d0.get_id(), d1.get_id(), out0.get_id());
    matmul_init(d0.get_id(), d1.get_id());

    d0.wait_front(1);
    d1.wait_front(1);
    d2.wait_front(1);
    d3.wait_front(1);
    d4.wait_front(1);
    d5.wait_front(1);
    out0.reserve_back(1);
    out1.reserve_back(1);
    out2.reserve_back(1);

    // OP[0]: d0×d1 (Float16_b); OP[1]: d2×d3 (Float32); OP[2]: d4×d5 (Float16_b).
    // Unpack reconfig + matmul init between ops (same as reconfig_unpack_quasar).
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
    pack_reconfig_data_format(out2.get_id(), out0.get_id());
    pack_init(out0.get_id());
    pack_tile(0, out0.get_id());
    pack_reconfig_data_format(out0.get_id(), out1.get_id());
    pack_init(out1.get_id());
    pack_tile(1, out1.get_id());
    pack_reconfig_data_format(out1.get_id(), out2.get_id());
    pack_init(out2.get_id());
    pack_tile(2, out2.get_id());
    tile_regs_release();

    d0.pop_front(1);
    d1.pop_front(1);
    d2.pop_front(1);
    d3.pop_front(1);
    d4.pop_front(1);
    d5.pop_front(1);
    out0.push_back(1);
    out1.push_back(1);
    out2.push_back(1);
}
