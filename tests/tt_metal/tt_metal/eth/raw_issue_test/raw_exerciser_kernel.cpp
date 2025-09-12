#include <cstdint>
#include "dataflow_api.h"
#include <cassert>
#include "risc_common.h"

#include "debug/ring_buffer.h"

/**
 * @brief Kernel to reproduce a potential CPU hardware bug.
 *
 * This kernel sweeps over a specified L1 memory range, and for each base address,
 * performs multiple attempts to write different values to a specific offset (src_ch_id),
 * followed by a volatile read, memory fence, and validation of the read-back value.
 * If mismatch occurs, it invalidates L1 cache and retries the write multiple times.
 *
 * Runtime args:
 * - range_start: Start of memory range to sweep
 * - range_end: End of memory range
 * - addr_step: Step size between base addresses
 * - num_attempts: Number of write attempts per base address
 */
void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t range_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t range_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t addr_step = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_attempts = get_arg_val<uint32_t>(arg_idx++);

    volatile uint32_t* DUMMY = reinterpret_cast<volatile uint32_t*>(range_end) - 1;
    WATCHER_RING_BUFFER_PUSH((uint)range_end);
    DUMMY--;
    *DUMMY = 0xDEADBEEF;
    WATCHER_RING_BUFFER_PUSH((uint)*DUMMY);
    DUMMY--;
    *DUMMY = 0xDEADBEEF;
    WATCHER_RING_BUFFER_PUSH((uint)*DUMMY);
    DUMMY--;
    *DUMMY = 0xDEADBEEF;
    WATCHER_RING_BUFFER_PUSH((uint)*DUMMY);
    DUMMY--;
    *DUMMY = 0xDEADBEEF;
    WATCHER_RING_BUFFER_PUSH((uint)*DUMMY);
    DUMMY--;
    *DUMMY = 0xDEADBEEF;
    WATCHER_RING_BUFFER_PUSH((uint)*DUMMY);
    constexpr bool use_byte_writes = true;
    for (uint32_t attempt = 0; attempt < num_attempts; ++attempt) {
        for (uint32_t src_addr = range_start; src_addr < range_end; src_addr += addr_step) {
            uint8_t written_value = static_cast<uint8_t>(attempt % 256);

            constexpr uint32_t src_ch_id_offset = 27;
            const uint32_t src_ch_id_addr = src_addr + src_ch_id_offset;
            const uint32_t base_word_addr = src_ch_id_addr & ~0x3;
            const uint32_t src_ch_id_offset_bits = (src_ch_id_addr - base_word_addr) * 8;

            // INTERESTINGLY!!!! IF I REMOVE THE ADDRESS CALCULATION ABOVE AND SIMPLY USE SRC_ADDR DIRECTLY, THIS
            // SEEMS TO WORK!
            volatile uint8_t* ptr_to_word_with_src_ch_id =
                reinterpret_cast<volatile uint8_t*>(base_word_addr);  // if this is src_addr, it will pass

            // two writes: busted
            // one write: busted
            // write, fence, write: busted
            *ptr_to_word_with_src_ch_id = written_value;
            invalidate_l1_cache();
            *ptr_to_word_with_src_ch_id = written_value;

            asm volatile("fence" ::: "memory");

            invalidate_l1_cache();
            uint32_t ch_id = *ptr_to_word_with_src_ch_id;

            if (ch_id != written_value) {
                invalidate_l1_cache();
                WATCHER_RING_BUFFER_PUSH((uint)base_word_addr);
                WATCHER_RING_BUFFER_PUSH((uint)0xABAB0000 | attempt);
                WATCHER_RING_BUFFER_PUSH((uint)0xcdcd0000 | ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)0x10000000 | *ptr_to_word_with_src_ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)0x20000000 | *ptr_to_word_with_src_ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)0x30000000 | *ptr_to_word_with_src_ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)0x40000000 | *ptr_to_word_with_src_ch_id);

                // one extra write results in correct readback. Removing this additional write
                // will result in an incorrect readback. This can seemingly be moved around
                // and the next immediate write will be correct
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)0x50000000 | *ptr_to_word_with_src_ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)0x60000000 | *ptr_to_word_with_src_ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)0x70000000 | *ptr_to_word_with_src_ch_id);

                WATCHER_RING_BUFFER_PUSH((uint)0x80000000 | *ptr_to_word_with_src_ch_id);
                invalidate_l1_cache();

                WATCHER_RING_BUFFER_PUSH((uint)0xdeadbeef);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                *ptr_to_word_with_src_ch_id = written_value;
                WATCHER_RING_BUFFER_PUSH((uint)*ptr_to_word_with_src_ch_id);
                WATCHER_RING_BUFFER_PUSH((uint)ch_id);

                ch_id = *ptr_to_word_with_src_ch_id;  // this is the correct value
                ASSERT(false);
            }
        }
    }
}
