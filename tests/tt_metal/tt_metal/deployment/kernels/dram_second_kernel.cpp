#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "common_dram.hpp"
#include "dram_utils.hpp"
#include "patterns/sync_mailbox.hpp"

void kernel_main() {
    const uint32_t sync_mailbox_l1_addr = get_arg_val<uint32_t>(0);

    volatile uint32_t* mb = reinterpret_cast<volatile uint32_t*>(sync_mailbox_l1_addr);

    if (mb[MB_MAGIC] != DRAM_SYNC_MAILBOX_MAGIC) {
        return;
    }

    uint32_t seen_start_tag = 0u;

    while (true) {
        noc_async_read_barrier();

        if (mb[MB_STOP] != 0u) {
            break;
        }

        const uint32_t start_tag = mb[MB_NCRISC_START];

        if ((start_tag == 0u) || (start_tag == seen_start_tag)) {
            continue;
        }

        seen_start_tag = start_tag;

        const uint32_t bank_id = mb[MB_BANK_ID];
        const uint64_t bank_offset_base =
            (static_cast<uint64_t>(mb[MB_BANK_OFFSET_HI]) << 32) | static_cast<uint64_t>(mb[MB_BANK_OFFSET_LO]);

        const uint32_t offset = mb[MB_CURRENT_OFFSET_BYTES];
        const uint32_t transfer_bytes = mb[MB_CURRENT_TRANSFER_BYTES];

        const uint32_t expect_l1_addr = mb[MB_GEN_ACTIVE_L1_ADDR];
        const uint32_t observe_l1_addr = mb[MB_OBS_ACTIVE_L1_ADDR];

        const uint32_t write_noc = mb[MB_WRITE_NOC];
        const uint32_t read_noc = mb[MB_READ_NOC];

        const uint32_t skip_writes = mb[MB_SKIP_WRITES];
        const uint32_t skip_reads = mb[MB_SKIP_READS];

        mb[MB_NCRISC_ERROR] = MB_ERROR_NONE;
        mb[MB_NCRISC_ACTIVE_OFFSET_BYTES] = offset;
        mb[MB_NCRISC_ACTIVE_TRANSFER_BYTES] = transfer_bytes;

        if ((transfer_bytes == 0u) || ((transfer_bytes & 0x3u) != 0u)) {
            mb[MB_NCRISC_ERROR] = MB_ERROR_NCRISC_BAD_TRANSFER;
            mb[MB_ERROR] = MB_ERROR_NCRISC_BAD_TRANSFER;
            mb[MB_NCRISC_DONE] = start_tag;
            continue;
        }

        if (!skip_writes) {
            const uint64_t write_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                bank_id, static_cast<uint32_t>(bank_offset_base + static_cast<uint64_t>(offset)), write_noc);

            noc_async_write(expect_l1_addr, write_dram_noc_addr, transfer_bytes);
            noc_async_write_barrier();
        }

        if (!skip_reads) {
            const uint64_t read_dram_noc_addr = get_noc_addr_from_bank_id<true>(
                bank_id, static_cast<uint32_t>(bank_offset_base + static_cast<uint64_t>(offset)), read_noc);

            noc_async_read(read_dram_noc_addr, observe_l1_addr, transfer_bytes);
            noc_async_read_barrier();
        }

        mb[MB_NCRISC_DONE] = start_tag;
    }
}
