#include "api/dataflow/dataflow_api.h"

#include <cstdint>
#include <utility>

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t num_pages_to_write_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_devices_rightside = get_compile_time_arg_val(5);
constexpr uint32_t num_devices_leftside = get_compile_time_arg_val(6);
constexpr uint32_t accessor_cta_base = 7;

void kernel_main() {
    uint32_t arg_idx = 0;
    uint32_t input_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t ouptut_address = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_address = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t direction = get_arg_val<uint32_t>(arg_idx++);  // 0 is forward, 1 is backward
    const auto input_page_id_start = get_arg_val<uint32_t>(arg_idx++);
    const auto input_page_id_end = get_arg_val<uint32_t>(arg_idx++);
    const auto pages_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_input_pages_per_device = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_input_pages_per_direction = get_arg_val<uint32_t>(arg_idx++);

    bool is_forward = (direction == 0);

    constexpr auto input_addrgen_args = TensorAccessorArgs<accessor_cta_base>();
    constexpr auto output_addrgen_args = TensorAccessorArgs<input_addrgen_args.next_compile_time_args_offset()>();

    const auto input_addrgen = TensorAccessor(input_addrgen_args, input_address, page_size);
    const auto output_addrgen = TensorAccessor(input_addrgen_args, ouptut_address, page_size);

    auto out_ready_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_address);

    // 1. Read local DRAM data to CB. Make it accessible in writer
    //  page id range : [start, end)
    uint32_t pages_read = 0;
    uint32_t num_total_pages_to_read = input_page_id_end - input_page_id_start;
    while (pages_read < num_total_pages_to_read) {
        uint32_t pages_remaining_to_read = num_total_pages_to_read - pages_read;
        uint32_t num_pages_to_read = std::min(pages_remaining_to_read, num_pages_to_write_per_packet);

        cb_reserve_back(cb_output_id, num_pages_to_write_per_packet);  // just reserve max num pages per packet
        size_t l1_write_addr = get_write_ptr(cb_output_id);
        // read num pages as much as it remained
        for (uint32_t j = 0; j < num_pages_to_read; ++j) {
            uint32_t page_id = input_page_id_start + pages_read;
            uint64_t noc_read_addr = get_noc_addr(page_id, input_addrgen);
            noc_async_read(noc_read_addr, l1_write_addr, page_size);

            l1_write_addr += page_size;
            pages_read++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_output_id, num_pages_to_write_per_packet);
    }

    // 2. Wait for data from previous neighbor and write it to CB again
    // so that, my writer can forward it to next neighbor
    // forward direction : [ ... , my_chip_id - 1 -> my_chip_id -> my_chip_id + 1 , ... ]
    // backward direction : [ ... , my_chip_id - 1 <- my_chip_id <- my_chip_id + 1 , ... ]
    uint32_t chunks_received = 0;
    uint32_t chunks_to_receive = 0;
    uint32_t writes_expected = 0;
    if (is_forward) {
        chunks_to_receive = num_devices_leftside;
        writes_expected = num_devices_leftside - 1;
    } else {
        chunks_to_receive = num_devices_rightside;
        writes_expected = num_devices_rightside - 1;
    }

    uint32_t page_sync_count = 0;
    uint32_t sem_target_val = 0;
    while (chunks_received < chunks_to_receive) {
        uint32_t sender_chip_id = UINT32_MAX;
        if (is_forward) {
            sender_chip_id = (my_chip_id - (chunks_received + 1) + ring_size) % ring_size;
        } else {
            sender_chip_id = (my_chip_id + (chunks_received + 1) + ring_size) % ring_size;
        }

        if (chunks_received < writes_expected) {
            uint32_t page_device_offset = sender_chip_id * num_input_pages_per_device;
            uint32_t page_start_offset = page_device_offset + input_page_id_start;

            DPRINT << "direction: " << direction << ", my_chip_id : " << my_chip_id
                   << ", sender_chip_id: " << sender_chip_id << ", page_start_offset: " << page_start_offset << ENDL();

            // Handle same amount of data as local read.
            page_sync_count = 0;
            uint32_t pages_read = 0;
            uint32_t num_total_pages_to_read = input_page_id_end - input_page_id_start;
            while (pages_read < num_total_pages_to_read) {
                if (page_sync_count % pages_per_sync == 0) {
                    // wait on semaphore until it becomes equal or greater than target value.
                    noc_semaphore_wait_min(out_ready_sem_ptr, sem_target_val + 1);
                    sem_target_val += 1;
                }
                page_sync_count += 1;

                uint32_t pages_remaining_to_read = num_total_pages_to_read - pages_read;
                uint32_t num_pages_to_read = std::min(pages_remaining_to_read, num_pages_to_write_per_packet);

                cb_reserve_back(cb_output_id, num_pages_to_write_per_packet);  // just reserve max num pages per packet
                size_t l1_write_addr = get_write_ptr(cb_output_id);
                // read num pages as much as it remained
                for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                    uint32_t page_id = page_start_offset + pages_read;
                    uint64_t noc_read_addr = get_noc_addr(page_id, output_addrgen);
                    noc_async_read(noc_read_addr, l1_write_addr, page_size);

                    l1_write_addr += page_size;
                    pages_read++;
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_id, num_pages_to_write_per_packet);
            }
        } else {
            uint32_t pages_read = input_page_id_start;
            uint32_t pages_to_read = input_page_id_end;
            while (pages_read < pages_to_read) {
                if (page_sync_count % pages_per_sync == 0) {
                    noc_semaphore_wait_min(out_ready_sem_ptr, sem_target_val + 1);
                    sem_target_val++;
                }
                page_sync_count++;
                uint32_t pages_remaining_to_read = pages_to_read - pages_read;
                uint32_t num_pages_to_read = std::min(pages_remaining_to_read, num_pages_to_write_per_packet);
                pages_read += num_pages_to_read;
            }
        }

        chunks_received += 1;
    }

    // Initialize as 0 for next use
    noc_semaphore_set(out_ready_sem_ptr, 0);
}
