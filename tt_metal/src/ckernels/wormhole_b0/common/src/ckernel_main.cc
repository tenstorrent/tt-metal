
// This c-file's purpose is:
// 1) include the generated list of kernels
//      The files hold run_kernel() definition and inline kernel_main functions for every ckernel
//      Need to make sure no other file includes these lists since it also include global parameter definitions
// 2) instantiate global variables

#include "ckernel_globals.h"

#include "chlkc_list.h"

#include "tools/profiler/kernel_profiler.hpp"

// Global vars
uint32_t unp_cfg_context = 0;
uint32_t pack_sync_tile_dst_ptr = 0;
uint32_t math_sync_tile_dst_index = 0;
uint32_t gl_alu_format_spec_reg = 0;

namespace ckernel
{
volatile uint * const regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile uint * const instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
volatile uint * const pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
volatile uint * const trisc_l1_mailbox = reinterpret_cast<volatile uint *>(MAILBOX_ADDR);
}

CBReadInterface cb_read_interface[NUM_CIRCULAR_BUFFERS];
CBWriteInterface cb_write_interface[NUM_CIRCULAR_BUFFERS];

void kernel_launch()
{
    // TODO(pgk): clean this up (into header shared with ckernel.cc)
    uint *local_l1_start_addr;
    if ((uint)__firmware_start == (uint)MEM_TRISC0_BASE) {
        local_l1_start_addr = (uint *)MEM_TRISC0_INIT_LOCAL_L1_BASE;
    } else if ((uint) __firmware_start == (uint)MEM_TRISC1_BASE) {
        local_l1_start_addr = (uint *)MEM_TRISC1_INIT_LOCAL_L1_BASE;
    } else {
        local_l1_start_addr = (uint *)MEM_TRISC2_INIT_LOCAL_L1_BASE;
    }
    firmware_kernel_common_init(local_l1_start_addr);

    run_kernel();
}
