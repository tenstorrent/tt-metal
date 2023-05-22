#include "ckernel.h"
#include "fw_debug.h"
#include "ckernel_main.h"
#include "ckernel_globals.h"
#include <tensix.h>

#include "tools/profiler/kernel_profiler.hpp"

namespace ckernel
{

enum class ttRiscCores : std::uint32_t { Unpack = 0, Math = 1, Pack = 2, Brisc = 3, Nrisc = 4};

volatile uint *reg_base = reinterpret_cast<volatile uint *>(0xFFB10000);
volatile uint *pc_buf_base = reinterpret_cast<volatile uint *>(PC_BUF_BASE);
volatile uint *regfile = reinterpret_cast<volatile uint *>(REGFILE_BASE);
volatile uint *instrn_buffer = reinterpret_cast<volatile uint *>(INSTRN_BUF_BASE);
volatile uint *mailbox_base[4] = {
    reinterpret_cast<volatile uint *>(TENSIX_MAILBOX0_BASE), reinterpret_cast<volatile uint *>(TENSIX_MAILBOX1_BASE),
    reinterpret_cast<volatile uint *>(TENSIX_MAILBOX2_BASE), reinterpret_cast<volatile uint *>(TENSIX_MAILBOX3_BASE)
};
volatile uint *dbg_event_scratch = nullptr;
uint *regmem = reinterpret_cast<uint *>(REGFILE_BASE);

uint32_t cfg_state_id __attribute__((section(".bss"))) = 0;  // Flip between 0 and 1 to keep state between kernel calls
uint32_t dest_offset_id __attribute__((section(".bss"))) = 0; // Flip between 0 and 1 to keep dest pointer between kernel calls

uint32_t dbg_event_index __attribute__((section(".bss"))) = 0;
uint32_t dbg_event_end __attribute__((section(".bss"))) = 0;
volatile uint16_t *debug_mailbox_base = nullptr;
uint8_t mailbox_index = 0;
uint8_t mailbox_end = 32;
uint32_t op_info_offset = 0;
volatile uint8_t *debug_buffer = nullptr;

uint8_t thread_id;

volatile uint local_mem_barrier;

volatile uint* trisc_l1_mailbox = reinterpret_cast<volatile uint *>(MAILBOX_ADDR);

void tensix_sync()
{
    volatile uint foo = 0xdeadbeef;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[1] = foo;

    // Now read -- this read will block until we're idle
    *fooptr = pc_buf_base[1];
}

void mop_sync()
{
    volatile uint foo = 0xdeadbeef;
    volatile uint *fooptr = &foo;
    // Write to pc buffer to push all writes ahead of us.. otherwise, the pc buffer read can bypass older writes
    pc_buf_base[2] = foo;

    // Now read -- this read will block until mops are done
    *fooptr = pc_buf_base[2];
}

inline bool ready_for_next_epoch() {         // place this through compiler into a section that is not going to overwritten
    return true;
    // mailbox_write(ttRiscCores::Nrisc);              // signal done epoch to NCRisc
    // mailbox_read(ttRiscCores::Nrisc);               // This is blocking read, until NCrisc signals epoch is ready
}

inline void set_thread_id_parameter() {
    if ((uint)__firmware_start == (uint)MEM_TRISC0_BASE) {
        thread_id = 0;
    } else if ((uint) __firmware_start == (uint)MEM_TRISC1_BASE) {
        thread_id = 1;
    } else {
        thread_id = 2;
    }
}

inline void allocate_debug_mailbox_buffer() {
   std::int32_t debug_mailbox_addr;
   if ((uint32_t)__firmware_start == (uint32_t)MEM_TRISC0_BASE) {
      debug_mailbox_addr = MEM_DEBUG_MAILBOX_ADDRESS + 0*MEM_DEBUG_MAILBOX_SIZE;
   } else if ((uint32_t) __firmware_start == (uint32_t)MEM_TRISC1_BASE) {
      debug_mailbox_addr = MEM_DEBUG_MAILBOX_ADDRESS + 1*MEM_DEBUG_MAILBOX_SIZE;
   } else {
      debug_mailbox_addr = MEM_DEBUG_MAILBOX_ADDRESS + 2*MEM_DEBUG_MAILBOX_SIZE;
   }
   debug_mailbox_base = reinterpret_cast<volatile uint16_t *>(debug_mailbox_addr);
   clear_mailbox_values();

}

inline void allocate_debug_buffer() {
    // TODO(pk) reimplement debug buffer
}

} // namespace ckernel

using namespace ckernel;

int main(int argc, char *argv[])
{
    kernel_profiler::init_profiler();

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_MAIN_START);
#endif
    FWEVENT("Launching proudction env kernels");

    // Initialize GPRs to all 0s
    for (int i = 0; i < 64; i++)
        regfile[i] = 0;

    // Init L1 buffer with 1.0f (used for reduce max)
    union {
        float f;
        uint32_t u;
    } f2u = {.f = 1.0f};

    for (uint i = 0; i < 16; i++) l1_buffer[i] = f2u.u;  // Load const into L1 buffer


    reset_cfg_state_id();

    trisc_l1_mailbox_write(RESET_VAL);

    uint *local_l1_start_addr;
    if ((uint)__firmware_start == (uint)MEM_TRISC0_BASE) {
        local_l1_start_addr = (uint *)MEM_TRISC0_INIT_LOCAL_L1_BASE;
    } else if ((uint) __firmware_start == (uint)MEM_TRISC1_BASE) {
        local_l1_start_addr = (uint *)MEM_TRISC1_INIT_LOCAL_L1_BASE;
    } else {
        local_l1_start_addr = (uint *)MEM_TRISC2_INIT_LOCAL_L1_BASE;
    }
    int32_t num_words = ((uint)__ldm_data_end - (uint)__ldm_data_start) >> 2;
    l1_to_local_mem_copy((uint*)__ldm_data_start, local_l1_start_addr, num_words);

    allocate_debug_mailbox_buffer();
    allocate_debug_buffer();

    //while (ready_for_next_epoch())
    {
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_KERNEL_MAIN_START);
#endif
        run_kernel();
#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_KERNEL_MAIN_END);
#endif
    }

    // Signal completion
    tensix_sync();
    trisc_l1_mailbox_write(KERNEL_COMPLETE);

#if defined(PROFILER_OPTIONS) && (PROFILER_OPTIONS & MAIN_FUNCT_MARKER)
  kernel_profiler::mark_time(CC_MAIN_END);
#endif
}
