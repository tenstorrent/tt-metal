#include "tensix.h"

void read_trisc_debug_mailbox(tt_cluster* cluster, int chip_id, const CoreCoord core, uint16_t trisc_id, uint32_t index = 0) {
    std::uint32_t debug_mailbox_addr;
    debug_mailbox_addr = MEM_DEBUG_MAILBOX_ADDRESS + trisc_id * MEM_DEBUG_MAILBOX_SIZE;
    assert(trisc_id >= 0 && trisc_id <= 2);

    debug_mailbox_addr += (index * (sizeof(uint32_t)));

    std::vector<std::uint32_t> vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, debug_mailbox_addr, sizeof(uint32_t));
    log_info(tt::LogVerif, "TRISC{} debug mailbox value = {}", trisc_id, vec.at(0));
}
