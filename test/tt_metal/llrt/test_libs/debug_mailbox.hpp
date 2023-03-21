#include "tensix.h"

void read_trisc_debug_mailbox(tt_cluster* cluster, int chip_id, const tt_xy_pair core, uint16_t trisc_id, uint32_t index = 0) {
    std::uint32_t debug_mailbox_addr;
    switch (trisc_id) {
        case 0:
            debug_mailbox_addr = l1_mem::address_map::TRISC0_DEBUG_BUFFER_BASE;
            break;
        case 1:
            debug_mailbox_addr = l1_mem::address_map::TRISC1_DEBUG_BUFFER_BASE;
            break;
        case 2:
            debug_mailbox_addr = l1_mem::address_map::TRISC2_DEBUG_BUFFER_BASE;
            break;
        default:
            throw std::invalid_argument("'trisc_id' must be one of 0, 1, or 2");
    }

    debug_mailbox_addr += (index * (sizeof(uint32_t)));

    std::vector<std::uint32_t> vec = tt::llrt::read_hex_vec_from_core(cluster, chip_id, core, debug_mailbox_addr, sizeof(uint32_t));
    log_info(tt::LogVerif, "TRISC{} debug mailbox value = {}", trisc_id, vec.at(0));
}
