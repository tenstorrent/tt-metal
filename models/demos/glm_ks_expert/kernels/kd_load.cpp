// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Stage 4 down weight reader (NCRISC, NOC0) on a down core, the flat expert's down_weight_reader: NE experts x DNB
// blocks ([DKB K tiles][DC columns] bfp4) from this core's DRAM bank, DPIECES reads a block, DDEPTH in flight
// (transaction ids 8 ..), into the down compute's weight ring (WD_CB, this RISC its producer, two experts) as slots free
// up. (The DRAM region holds one expert's blocks: every expert re-reads it.)
// RT: 0 the region's bank address, 1 bank id
#include "api/dataflow/dataflow_api.h"
#include "kd_common.hpp"

using namespace kd;

void kernel_main() {
    const uint32_t src = get_arg_val<uint32_t>(0), bank = get_arg_val<uint32_t>(1);
    const uint32_t ring = get_write_ptr(WD_CB);
    constexpr uint32_t TOTAL = NE * DNB * DPIECES;
    constexpr uint32_t PIECE = DBLK_BYTES / DPIECES;
    uint32_t iss = 0, fin = 0, pushed = 0;
    while (fin < TOTAL) {
        invalidate_l1_cache();
        while (iss < TOTAL && iss - fin < DDEPTH && cb_pages_reservable_at_back(WD_CB, (iss / DPIECES - pushed + 1) * DBLK)) {
            const uint32_t blk = iss / DPIECES, pc = iss % DPIECES;
            noc_async_read_set_trid(8 + iss % DDEPTH);
            noc_async_read(
                get_noc_addr_from_bank_id<true>(bank, src + (blk % DNB) * DBLK_BYTES + pc * PIECE),
                ring + (blk % DWS) * DBLK_BYTES + pc * PIECE,
                PIECE);
            ++iss;
        }
        while (fin < iss && !noc_cmd_buf_ready(noc_index, read_cmd_buf)) {
        }
        while (fin < iss && ncrisc_noc_read_with_transaction_id_flushed(noc_index, 8 + fin % DDEPTH)) {
            ++fin;
            if (fin % DPIECES == 0) {
                cb_push_back(WD_CB, DBLK);
                ++pushed;
            }
        }
    }
    noc_async_read_set_trid(0);
    noc_async_read_barrier();
}
