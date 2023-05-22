
#include "ckernel.h"
#include "ckernel_addr_map.h"
#include "ckernel_pcbuf.h"
#include "fw_debug.h"
#include "ckernel_main.h"
#include "ckernel_globals.h"
#include <tensix.h>

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
    if ((uint)__firmware_start == (uint)l1_mem::address_map::TRISC0_BASE) {
        thread_id = 0;
    } else if ((uint) __firmware_start == (uint)l1_mem::address_map::TRISC1_BASE) {
        thread_id = 1;
    } else {
        thread_id = 2;
    }
}

inline void allocate_debug_mailbox_buffer() {
   std::int32_t debug_mailbox_addr;
   if ((uint32_t)__firmware_start == (uint32_t)l1_mem::address_map::TRISC0_BASE) {
      debug_mailbox_addr = DEBUG_MAILBOX_ADDRESS + 0*DEBUG_MAILBOX_SIZE;
   } else if ((uint32_t) __firmware_start == (uint32_t)l1_mem::address_map::TRISC1_BASE) {
      debug_mailbox_addr = DEBUG_MAILBOX_ADDRESS + 1*DEBUG_MAILBOX_SIZE;
   } else {
      debug_mailbox_addr = DEBUG_MAILBOX_ADDRESS + 2*DEBUG_MAILBOX_SIZE;
   }
   debug_mailbox_base = reinterpret_cast<volatile uint16_t *>(debug_mailbox_addr);
   clear_mailbox_values();
}

} // namespace ckernel

void local_mem_copy() {
   volatile uint *l1_local_mem_start_addr;
   volatile uint *local_mem_start_addr = (volatile uint*) LOCAL_MEM_BASE;

   if ((uint)__firmware_start == (uint)l1_mem::address_map::TRISC0_BASE) {
      l1_local_mem_start_addr = (volatile uint*)l1_mem::address_map::TRISC0_LOCAL_MEM_BASE;
   } else if ((uint) __firmware_start == (uint)l1_mem::address_map::TRISC1_BASE) {
      l1_local_mem_start_addr = (volatile uint*)l1_mem::address_map::TRISC1_LOCAL_MEM_BASE;
   } else {
      l1_local_mem_start_addr = (volatile uint*)l1_mem::address_map::TRISC2_LOCAL_MEM_BASE;
   }
   uint word_size = ((uint)__local_mem_rodata_end_addr - (uint)__local_mem_rodata_start_addr)>>2;

   if (word_size>0) {
      for (uint n=0;n<word_size;n++) {
         local_mem_start_addr[n] = l1_local_mem_start_addr[n];
      }
      ckernel::mem_barrier(l1_local_mem_start_addr[word_size-1]);
   }

}

using namespace ckernel;

int main(int argc, char *argv[])
{
    FWEVENT("Launching proudction env kernels");

    // Initialize GPRs to all 0s
    for (int i = 0; i < 64; i++)
        regfile[i] = 0;

    reset_cfg_state_id();

    trisc_l1_mailbox_write(RESET_VAL);

    if ((uint)LOCAL_MEM_BASE ==
            ((uint)__local_mem_rodata_end_addr&0xfff00000))
    {
       local_mem_copy();
    }

    allocate_debug_mailbox_buffer();

    //while (ready_for_next_epoch())
    {
        run_kernel();
    }

    // Signal completion
    tensix_sync();
    trisc_l1_mailbox_write(KERNEL_COMPLETE);

}
