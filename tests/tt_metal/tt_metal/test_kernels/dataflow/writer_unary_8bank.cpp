// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/circular_buffer.h"
#include "experimental/noc.h"
#include "experimental/tensor.h"
#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#endif
#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t dst_addr  = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(2); // Index 2 to match with regular writer_unary

    constexpr uint32_t onetile = 1;
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer dfb_out(2);
    const uint32_t tile_bytes = dfb_out.get_entry_size();
#else
    constexpr uint32_t cb_id_out0 = 16;
    const uint32_t tile_bytes = get_tile_size(cb_id_out0);
    experimental::CircularBuffer cb(cb_id_out0);
#endif

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(dst_args, dst_addr, tile_bytes);

    experimental::Noc noc;

    DPRINT << "WR: start num_tiles=" << num_tiles << " tile_bytes=" << tile_bytes << " dst_addr=" << HEX() << dst_addr
           << DEC() << ENDL();

    for (uint32_t i = 0; i < num_tiles; i++) {
#ifdef ARCH_QUASAR
        // if (i == num_tiles - 1) {
        auto& dfb_intf = g_dfb_interface[dfb_out.get_id()];
        auto packed_tc = dfb_intf.tc_slots[dfb_intf.tc_idx].packed_tile_counter;
        uint8_t t_id = experimental::get_tensix_id(packed_tc);
        uint8_t c_id = experimental::get_counter_id(packed_tc);
        DPRINT << "WR: before wait front " << i << " tensix_id=" << (uint32_t)t_id << " tc_id=" << (uint32_t)c_id
               << " tc_idx=" << (uint32_t)dfb_intf.tc_idx << " occ=" << (uint32_t)llk_intf_get_occupancy(t_id, c_id)
               << " cap=" << (uint32_t)llk_intf_get_capacity(t_id, c_id)
               << " posted=" << (uint32_t)llk_intf_get_posted(t_id, c_id)
               << " acked=" << (uint32_t)llk_intf_get_acked(t_id, c_id) << ENDL();
        //}
        dfb_out.wait_front(onetile);
        DPRINT << "WR: after wait front " << i << " tensix_id=" << (uint32_t)t_id << " tc_id=" << (uint32_t)c_id
               << " tc_idx=" << (uint32_t)dfb_intf.tc_idx << " occ=" << (uint32_t)llk_intf_get_occupancy(t_id, c_id)
               << " cap=" << (uint32_t)llk_intf_get_capacity(t_id, c_id)
               << " posted=" << (uint32_t)llk_intf_get_posted(t_id, c_id)
               << " acked=" << (uint32_t)llk_intf_get_acked(t_id, c_id) << ENDL();
        if (i == 0) {
            auto* od = reinterpret_cast<volatile uint16_t*>(
                dfb_out.get_read_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR);
            DPRINT << "WR: out rd_ptr=" << HEX()
                   << dfb_out.get_read_ptr() + MEMORY_PORT_NONCACHEABLE_MEM_PORT_MEM_BASE_ADDR << DEC() << ENDL();
            DPRINT << "WR: out[0..7]=" << HEX() << (uint32_t)od[0] << " " << (uint32_t)od[1] << " " << (uint32_t)od[2]
                   << " " << (uint32_t)od[3] << " " << (uint32_t)od[4] << " " << (uint32_t)od[5] << " "
                   << (uint32_t)od[6] << " " << (uint32_t)od[7] << DEC() << ENDL();
        }
        // if (i == num_tiles - 1) {
        // DPRINT << "WR: after wait front " << i << ENDL();
        //}
        noc.async_write(dfb_out, s, tile_bytes, {}, {.page_id = i});
        // if (i == num_tiles - 2) {
        //     DPRINT << "WR: before write barrier " << i << ENDL();
        // }
        noc.async_write_barrier();
        // if (i == num_tiles - 2) {
        //     DPRINT << "WR: after write barrier " << i << ENDL();
        // }
        dfb_out.pop_front(onetile);
        // if (i == num_tiles - 2) {
        //     DPRINT << "WR: after pop front " << i << ENDL();
        // }
#else
        cb.wait_front(onetile);
        noc.async_write(cb, s, tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        cb.pop_front(onetile);
#endif
        // DPRINT << "WR: iteration i(num_tiles)=" << i << ENDL();
    }
    DPRINT << "WR: done" << ENDL();
}
