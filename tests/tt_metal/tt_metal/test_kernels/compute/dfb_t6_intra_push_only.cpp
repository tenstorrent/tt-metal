// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Pack posts `num_entries` to each of this Neo's eight intra-tensix DFBs (dfb::out0 .. dfb::out7) and unpack never
// pops them, deliberately leaving every Tensix-only tile counter of the Neo with posted != acked after the program
// completes. Pack calls the LLK wait/push directly rather than going through a DataflowBuffer, whose destructor would
// wait for the entries to be consumed.

#include <cstdint>
#include "api/compute/common.h"
#include "ckernel_trisc_common.h"
#include "experimental/kernel_args.h"
#ifdef UCK_CHLKC_PACK
#include "llk_io_pack.h"
#include "llk_pack_tile_api.h"
#endif

void kernel_main() {
    constexpr std::uint32_t num_entries = get_arg(args::num_entries);
    compute_kernel_hw_startup(dfb::out0, dfb::out0);
#ifdef UCK_CHLKC_PACK
    constexpr std::uint32_t dfb_ids[] = {
        dfb::out0, dfb::out1, dfb::out2, dfb::out3, dfb::out4, dfb::out5, dfb::out6, dfb::out7};
    for (std::uint32_t dfb_id : dfb_ids) {
        llk_wait_for_free_tiles(dfb_id, num_entries);
        llk_pack_dummy(dfb_id);
        llk_push_tiles(dfb_id, num_entries);
    }
    // Wait until each post has reached its counter, so it went through the packer's remapper pair before the
    // firmware tears that pair down.
    for (std::uint32_t dfb_id : dfb_ids) {
        const LocalDFBInterface& dfb = get_local_dfb_interface(dfb_id);
        const std::uint32_t tc = dfb::get_counter_id(dfb.tc_slots[0].packed_tile_counter);
        while ((ckernel::trisc::tile_counters[tc].f.posted & 0xFFFFu) != num_entries) {
        }
    }
#endif
}
