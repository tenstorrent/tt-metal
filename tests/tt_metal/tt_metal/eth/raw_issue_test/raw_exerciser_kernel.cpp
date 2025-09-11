#include <cstdint>
#include "dataflow_api.h"
#include <cassert>
#include "risc_common.h"

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
int kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t range_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t range_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t addr_step = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_attempts = get_arg_val<uint32_t>(arg_idx++);

    for (uint32_t src_addr = range_start; src_addr < range_end; src_addr += addr_step) {
        for (uint32_t attempt = 0; attempt < num_attempts; ++attempt) {
            uint8_t written_value = static_cast<uint8_t>(attempt % 256);

            constexpr uint32_t src_ch_id_offset = 27;
            const uint32_t src_ch_id_addr = src_addr + src_ch_id_offset;
            const uint32_t base_word_addr = src_ch_id_addr & ~0x3;
            const uint32_t src_ch_id_offset_bits = (src_ch_id_addr - base_word_addr) * 8;
            volatile uint32_t* ptr_to_word_with_src_ch_id = reinterpret_cast<volatile uint32_t*>(base_word_addr);

            uint32_t word_with_src_ch_id = *ptr_to_word_with_src_ch_id;
            word_with_src_ch_id = word_with_src_ch_id & ~(0xFFu << src_ch_id_offset_bits);
            word_with_src_ch_id = word_with_src_ch_id | (static_cast<uint32_t>(written_value) << src_ch_id_offset_bits);
            *ptr_to_word_with_src_ch_id = word_with_src_ch_id;

            asm volatile("fence" ::: "memory");

            uint32_t ch_id = (*ptr_to_word_with_src_ch_id >> src_ch_id_offset_bits) & 0xFF;

            if (ch_id != written_value) {
                invalidate_l1_cache();

                volatile uint8_t* src_ch_id_ptr = reinterpret_cast<volatile uint8_t*>(src_ch_id_addr);
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;
                *src_ch_id_ptr = written_value;

                ch_id = *src_ch_id_ptr;
                ASSERT(ch_id == written_value);
            }
        }
    }
}
