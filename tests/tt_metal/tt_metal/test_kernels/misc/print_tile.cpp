// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dprint.h"

// PACK trisc version of cb_wait_front just for this test
#if defined(UCK_CHLKC_PACK)
#include "compute_kernel_api/cb_api.h"
inline void cb_wait_front_pack(int operand, std::int32_t num_tiles) {
    std::uint32_t input = operand;
    volatile tt_l1_ptr std::uint32_t* tiles_received_ptr = get_cb_tiles_received_ptr(operand);
    std::uint16_t num_tiles_u = (std::uint16_t)num_tiles;

    std::uint16_t tiles_received;

    uint16_t num_tiles_recv;
    do {
        tiles_received = (std::uint16_t)reg_read((std::uint32_t)tiles_received_ptr);
        num_tiles_recv = tiles_received - cb_interface[input].tiles_acked;
    } while (num_tiles_recv < num_tiles_u);
}
#endif

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)
#include "dataflow_api.h"
void kernel_main() {
#else
#include "compute_kernel_api/common.h"
namespace NAMESPACE {
void MAIN {
#endif
    // Read out the tile we want to print using BRISC, put it in c_in0
    constexpr uint32_t cb_id = tt::CB::c_in0;
#if defined(COMPILE_FOR_BRISC)
    uint32_t src_addr  = get_arg_val<uint32_t>(0);
    uint32_t src_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_noc_y = get_arg_val<uint32_t>(2);

    uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, src_addr);
    cb_reserve_back(cb_id, 1);
    noc_async_read(src_noc_addr, get_write_ptr(cb_id), get_tile_size(cb_id));
    noc_async_read_barrier();
    cb_push_back(cb_id, 1);
#endif

	// PACK trisc doesn't have cb_wait_front implemented, have our own version just for this test.
#if defined(UCK_CHLKC_PACK)
	cb_wait_front_pack(cb_id, 1);
#else
    cb_wait_front(cb_id, 1);
#endif

    // Print the tile from each RISC, one after another
    DPRINT_DATA0(
        DPRINT << "Print tile from Data0:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false);
        DPRINT << RAISE{1};
    );
    DPRINT_UNPACK(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{1};
        DPRINT << "Print tile from Unpack:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, false);
        DPRINT << RAISE{2};
    );
    DPRINT_MATH(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{2};
        DPRINT << "Print tile from Math:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, false);
        DPRINT << RAISE{3};
    );
    DPRINT_PACK(
        // Wait for previous core (DATA0) to finish printing.
        DPRINT << WAIT{3};
        DPRINT << "Print tile from Pack:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), true, false);
        DPRINT << RAISE{4};
    );
    DPRINT_DATA1(
        // Wait for previous core (UNPACK) to finish printing.
        DPRINT << WAIT{4};
        DPRINT << "Print tile from Data1:" << ENDL();
        DPRINT << TSLICE(cb_id, 0, SliceRange::hw0_32_8(), TSLICE_INPUT_CB, TSLICE_RD_PTR, true, false);
    );

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC)
}
#else
}
}
#endif
