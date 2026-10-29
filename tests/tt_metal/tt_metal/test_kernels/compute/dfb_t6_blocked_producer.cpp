// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) Tensix-side BLOCKED producer (TRISC -> DFB -> DM).
// The host pre-fills the ring; this kernel posts credits, one block per reserve/push, and the
// producing thread (PACK, or trisc3 when the host put the producer there) stamps word 0 of every
// entry it pushes with its thread id, so the host can see which thread pushed it.

#include "api/dataflow/dataflow_buffer.h"
#include "api/compute/common.h"  // for dummy_pack (TEN-4746 no-write pack ordering)
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t block_size = get_arg(args::block_size);
    constexpr bool producer_on_trisc3 = get_arg(args::producer_on_trisc3) != 0;
    constexpr uint32_t producer_stamp_base = get_arg(args::producer_stamp_base);
    constexpr bool i_produce = ckernel::TRISC_ID == (producer_on_trisc3 ? 3u : 2u);

    DataflowBuffer dfb(dfb::out);

    // dummy_pack's PACR_STRIDE validates a pack-partition bd_table entry; compute_kernel_hw_startup
    // is what runs llk_pack_init and programs that entry. copy_init is not needed: this kernel
    // never unpacks.
    compute_kernel_hw_startup(dfb::out, dfb::out);

    const uint32_t num_blocks = num_entries_per_producer / block_size;
    const uint32_t entry_bytes = dfb.get_entry_size();
    const uint32_t stride_tiles = dfb.get_producer_stride_tiles();
    for (uint32_t b = 0; b < num_blocks; ++b) {
        dfb.reserve_back(block_size);
        // TEN-4746: a real packer op must sit between reserve_back's WAIT_FREE and push_back's
        // PUSH_TILES. The host pre-fills the ring, so a no-write dummy pack supplies that op
        // without modifying the payload.
        ckernel::dummy_pack(dfb::out);
        if constexpr (i_produce) {
            // tensix_sync blocks this RISC until that pack retired, so the stores below cannot land
            // before the space was free. On trisc3 the reserve_back already blocked the RISC.
            ckernel::tensix_sync();
            const uint32_t block_base = dfb.get_write_ptr() << 4;
            for (uint32_t j = 0; j < block_size; ++j) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block_base + j * stride_tiles * entry_bytes) =
                    producer_stamp_base + ckernel::TRISC_ID;
            }
            asm volatile("fence" ::: "memory");  // the stamps land before push_back posts the credit
        }
        dfb.push_back(block_size);
    }
    dfb.finish();
}
