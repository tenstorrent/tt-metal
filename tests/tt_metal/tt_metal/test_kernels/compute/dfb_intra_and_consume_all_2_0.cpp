// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 (declarative API) variant of compute/dfb_intra_and_consume_all.cpp.
//
// Two DFBs live in the same program on one Tensix cluster (num_threads Neos):
//
//   dfb::consume - DM->Tensix, 1 strided producer x N blocked consumers (remapper).
//                  This kernel acts as the blocked consumer; data is produced by
//                  a concurrent DM kernel.
//
//   dfb::intra   - Intra-tensix, N packer-producer -> N unpacker-consumer pairs.
//                  This kernel binds the same DFB as PRODUCER + CONSUMER, which
//                  is how M2 infers tensix_scope=INTRA.
//
// Net result for dfb::intra L1 ring: every word = original_value + 2.

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t num_entries_consumer = get_arg(args::num_entries_consumer);
    constexpr uint32_t entries_per_neo = get_arg(args::entries_per_neo);
    constexpr uint32_t words_per_entry = get_arg(args::words_per_entry);

    // Phase 1: blocked consumer of the DM-produced strided x blocked DFB.
    // Remapper fans out 1 DM post to N UNPACK TC slots so every blocked UNPACK
    // TRISC sees the full set of entries. PACK/MATH TRISCs race through
    // (wait_front / pop_front are no-ops on those TRISCs).
    DataflowBuffer dfb_consumer(dfb::consume);
    // HW requires a copy (unpack) instruction between wait_front and pop_front.
    // Configure unpack + pack hw against the consumer DFB (no separate output DFB
    // exists for this drain; this also programs the buffer-descriptor table the
    // UNPACR reads) and discard the copied tile. The copy_tile / acquire_dst /
    // release_dst LLK macros self-gate per-TRISC, so this is correct even though
    // only the UNPACK TRISC actually waits/pops here (PACK/MATH race through).
    compute_kernel_hw_startup(dfb_consumer.get_id(), dfb_consumer.get_id());
    copy_init(dfb_consumer.get_id());
    for (uint32_t i = 0; i < num_entries_consumer; ++i) {
        acquire_dst();
        dfb_consumer.wait_front(1);
        copy_tile(dfb_consumer.get_id(), 0, 0);
        dfb_consumer.pop_front(1);
        release_dst();
    }
    dfb_consumer.finish();

    // Phase 2: intra-tensix credit flow (PACK -> UNPACK on the same Neo, hidden TC).
    DataflowBuffer dfb_intra(dfb::intra);

#ifdef UCK_CHLKC_UNPACK
    uint32_t trisc_id = ckernel::csr_read<ckernel::CSR::TRISC_ID>();
#endif

    // HW requires a copy (unpack) instruction between wait_front and pop_front (same
    // rule as Phase 1). Without an UNPACR in the consume, pop_front's TT_POP_TILES
    // read-done wait is only vacuously satisfied; with 4 Neos it races and deadlocks
    // some Neos in pop_front (their producer then wedges in finish()). Re-program
    // unpack + buffer-descriptor for dfb_intra and issue copy_tile in the consume so
    // read-done is always satisfied; acquire_dst/release_dst balance the MATH<->PACK
    // dest handshake (no pack_tile needed, the copied tile is discarded).
    // Mid-kernel reprogram (Phase 1 already ran compute_kernel_hw_startup): re-point unpack SrcA and
    // re-init the copy datapath for dfb_intra via reconfig + copy_init, not a mid-kernel hw startup.
    reconfig_data_format_srca(dfb_intra.get_id());
    copy_init(dfb_intra.get_id());

    for (uint32_t i = 0; i < entries_per_neo; ++i) {
        dfb_intra.reserve_back(1);
#ifdef UCK_CHLKC_PACK
        {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_write_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        dfb_intra.push_back(1);

        acquire_dst();
        dfb_intra.wait_front(1);
#ifdef UCK_CHLKC_UNPACK
        if (trisc_id == 0) {
            volatile uint32_t* entry = reinterpret_cast<volatile uint32_t*>(dfb_intra.get_read_ptr() << 4);
            for (uint32_t w = 0; w < words_per_entry; ++w) {
                entry[w] += 1;
            }
        }
#endif
        copy_tile(dfb_intra.get_id(), 0, 0);  // UNPACR -> satisfies pop_front read-done
        dfb_intra.pop_front(1);
        release_dst();
    }
    dfb_intra.finish();
}
