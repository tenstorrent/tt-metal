#include "firmware_common.h"
// #include "risc_common.h"
#include "risc_attribs.h"

uint8_t my_x[NUM_NOCS] __attribute__((used));
uint8_t my_y[NUM_NOCS] __attribute__((used));
uint8_t my_logical_x_ __attribute__((used));
uint8_t my_logical_y_ __attribute__((used));
uint8_t my_relative_x_ __attribute__((used));
uint8_t my_relative_y_ __attribute__((used));

uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_num_issued[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_writes_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_nonposted_atomics_acked[NUM_NOCS] __attribute__((used));
uint32_t noc_posted_writes_num_issued[NUM_NOCS] __attribute__((used));

// These arrays are stored in local memory of FW, but primarily used by the kernel which shares
// FW symbols. Hence mark these as 'used' so that FW compiler doesn't optimize it out.
uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used));
uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used));
int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used));
int32_t bank_to_l1_offset[NUM_L1_BANKS] __attribute__((used));

tt_l1_ptr mailboxes_t* const mailboxes = (tt_l1_ptr mailboxes_t*)(MEM_MAILBOX_BASE + MEM_L1_UNCACHED_BASE);

void device_setup() {
    // instn_buf
    // pc_buf
    // clock gating
    // NOC setup
    // set_deassert_addresses
    // wzeromem
    // invalidate_l1_cache
    // clear_destination_registers
    // enable_cc_stack
    // set_default_sfpu_constant_register_state
}

inline __attribute__((always_inline)) void signal_ncrisc_completion() {
    mailboxes->subordinate_sync.dm1 = RUN_SYNC_MSG_DONE;
}

inline void wait_subordinates() {
    // WAYPOINT("NTW");
    while (mailboxes->subordinate_sync.dm1 != RUN_SYNC_MSG_ALL_SUBORDINATES_DONE);
    // WAYPOINT("NTD");
}

int main() {
    configure_csr();
    std::uint64_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    // WAYPOINT("I");
    // clear bss
    // handle noc_tobank ???
    mailboxes->launch_msg_rd_ptr = 0;  // Initialize the rdptr to 0

    my_logical_x_ = mailboxes->core_info.absolute_logical_x;
    my_logical_y_ = mailboxes->core_info.absolute_logical_y;

    // risc_init();
    device_setup();
    if (hartid > 0) {
        signal_ncrisc_completion();
    } else {
        wait_subordinates();
        mailboxes->go_messages[0].signal = RUN_MSG_DONE;
    }
    while (1) {
        // WAYPOINT("GW");
    }

    return 0;
}
