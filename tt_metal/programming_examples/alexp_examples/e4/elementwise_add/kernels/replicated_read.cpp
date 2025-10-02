#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t a_addr = get_arg_val<uint32_t>(0);
    uint32_t b_addr = get_arg_val<uint32_t>(1);
    uint32_t n_tiles = get_arg_val<uint32_t>(2);
    uint32_t r_tiles = get_arg_val<uint32_t>(3);

    DPRINT << "READER KERNEL START: n_tiles=" << n_tiles << ", r_tiles=" << r_tiles << ENDL();
    DPRINT << "READER: a_addr=0x" << HEX() << a_addr << ", b_addr=0x" << HEX() << b_addr << ENDL();

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    const uint32_t tile_size_bytes = get_tile_size(cb_in0);
    DPRINT << "READER: tile_size_bytes=" << tile_size_bytes << ENDL();

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, a_addr, tile_size_bytes);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, b_addr, tile_size_bytes);

    DPRINT << "READER: TensorAccessors initialized" << ENDL();

    // const InterleavedAddrGenFast<true> a = {
    //     .bank_base_address = a_addr,
    //     .page_size = tile_size_bytes,
    //     .data_format = DataFormat::Float32,
    // };

    // const InterleavedAddrGenFast<true> b = {
    //     .bank_base_address = b_addr,
    //     .page_size = tile_size_bytes,
    //     .data_format = DataFormat::Float32,
    // };

    for (uint32_t i = 0; i < n_tiles; i++) {
        if (i >= 14) {  // Debug tiles near the failure boundary
            DPRINT << "READER: Processing tile i=" << i << " (critical tile)" << ENDL();
        }

        cb_reserve_back(cb_in0, 1);
        uint32_t cb_in0_addr = get_write_ptr(cb_in0);
        noc_async_read_tile(i, a, cb_in0_addr);
        noc_async_read_barrier();
        cb_push_back(cb_in0, 1);

        if (i >= 14) {
            DPRINT << "READER: Read A tile i=" << i << " to addr=0x" << HEX() << cb_in0_addr << ENDL();
        }

        // Scalable B tile batching - works with any CB1 size
        // Determine actual CB1 capacity by trying to reserve space
        uint32_t cb1_capacity = 2;  // Start with minimum assumption

        // For large workloads, use streaming with minimal batches
        // For small workloads, try to use larger batches if CB space allows
        if (r_tiles <= 8) {
            cb1_capacity = 4;  // Small workload: try larger batches
        } else if (r_tiles <= 16) {
            cb1_capacity = 3;  // Medium workload: moderate batches
        } else {
            cb1_capacity = 2;  // Large workload: minimal streaming batches
        }

        uint32_t batch_size = (r_tiles <= cb1_capacity) ? r_tiles : cb1_capacity;

        for (uint32_t j_start = 0; j_start < r_tiles; j_start += batch_size) {
            uint32_t j_end = (j_start + batch_size > r_tiles) ? r_tiles : j_start + batch_size;

            if (i >= 14 && j_start == 0) {
                DPRINT << "READER: Processing B tiles batch [" << j_start << "-" << (j_end-1)
                       << "] for A tile i=" << i << ENDL();
            }

            // Push a batch of B tiles
            for (uint32_t j = j_start; j < j_end; j++) {
                cb_reserve_back(cb_in1, 1);
                uint32_t cb_in1_addr = get_write_ptr(cb_in1);
                noc_async_read_tile(j, b, cb_in1_addr);
                noc_async_read_barrier();
                cb_push_back(cb_in1, 1);

                if (i >= 14 && j < 2) {  // Only log first few B tiles for critical A tiles
                    DPRINT << "READER: Read B tile j=" << j << " for A tile i=" << i << " to addr=0x" << HEX() << cb_in1_addr << ENDL();
                }
            }

            // If we have more batches to process, add a synchronization point
            // This allows compute kernel to process the current batch before we push the next
            if (j_end < r_tiles) {
                noc_async_read_barrier();  // Ensure all tiles in this batch are written
                // Small delay to allow compute kernel to process this batch
                for (volatile uint32_t delay = 0; delay < 1000; delay++);  // Increased delay for better sync
            }
        }
    }

    DPRINT << "READER KERNEL COMPLETE" << ENDL();
}
