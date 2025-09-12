#include <cstdint>
#include "dataflow_api.h"

/**
 * @brief Kernel to emit NOC traffic for inducing L1 bank conflicts.
 *
 * This kernel repeatedly sends async NOC writes from a local buffer to a specified
 * range on a remote core's L1, wrapping around the range to generate continuous traffic.
 * This is intended to create noise and potential timing issues to expose hardware bugs
 * in conjunction with the exerciser kernel.
 *
 * Runtime args:
 * - num_iters (uint64_t): Number of write iterations
 * - valid_write_range_start: Start of remote write range
 * - valid_write_range_end: End of remote write range
 * - dest_noc_x: Destination NOC X coordinate
 * - dest_noc_y: Destination NOC Y coordinate
 * - noc_write_size: Size of each NOC write
 */
void kernel_main() {
    uint32_t arg_idx = 0;
    uint64_t num_iters =
        ((uint64_t)get_arg_val<uint32_t>(arg_idx + 1) << 32) | (uint64_t)get_arg_val<uint32_t>(arg_idx);
    arg_idx += 2;
    uint32_t valid_write_range_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t valid_write_range_end = get_arg_val<uint32_t>(arg_idx++);

    uint32_t dest_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t dest_noc_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t noc_write_size = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t buf_size = 1024;

    uint32_t current_dest_addr = valid_write_range_start;
    for (uint64_t iter = 0; iter < num_iters; ++iter) {
        uint64_t dest_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, current_dest_addr);
        uint32_t size = (noc_write_size < (valid_write_range_end - current_dest_addr))
                            ? noc_write_size
                            : (valid_write_range_end - current_dest_addr);
        // just arbitrarily copy from 0 since we only care about generating the traffic, not
        // the contents
        noc_async_write(0, dest_noc_addr, size);
        current_dest_addr += size;
        if (current_dest_addr >= valid_write_range_end) {
            current_dest_addr = valid_write_range_start;
        }
    }
    noc_async_write_barrier();
}
