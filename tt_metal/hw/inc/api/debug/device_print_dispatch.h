// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "internal/risc_attribs.h"
#include "risc_common.h"

namespace device_print_dispatch {

struct NocLocationInputInfo {
    uint16_t x;
    uint16_t y;
    uint32_t high_address_bits;
    uint32_t rw_ptr_addr;
    uint32_t buf_addr;
    uint32_t buf_size;
};

struct DramStreamMessageHeader {
    uint16_t x : 6;
    uint16_t y : 6;
    uint16_t align : 6;
    uint16_t buffer_wrapped : 1;
    uint16_t length : 13;
};

#if defined(ARCH_WORMHOLE)
constexpr uint32_t DEFAULT_MAX_NOC_LOCATIONS = 8 * 10    // Tensix cores
                                               + 8 * 2;  // ETH cores
// Noc reads/writes must be 16-byte aligned.
constexpr uint32_t NOC_L1_TO_L1_ALIGNMENT = 16;
constexpr uint32_t NOC_L1_TO_DRAM_ALIGNMENT = 32;
#elif defined(ARCH_BLACKHOLE)
constexpr uint32_t DEFAULT_MAX_NOC_LOCATIONS = 14 * 10    // Tensix cores
                                               + 14 * 1   // ETH cores
                                               + 2 * 12;  // DRAM cores
// Noc reads/writes must be 16-byte aligned.
constexpr uint32_t NOC_L1_TO_L1_ALIGNMENT = 16;
constexpr uint32_t NOC_L1_TO_DRAM_ALIGNMENT = 32;
#else
constexpr uint32_t DEFAULT_MAX_NOC_LOCATIONS = 0;
constexpr uint32_t NOC_L1_TO_L1_ALIGNMENT = 0;
constexpr uint32_t NOC_L1_TO_DRAM_ALIGNMENT = 0;
static_assert(false, "Unsupported architecture");
#endif

}  // namespace device_print_dispatch

template <
    bool EnableNocLocationCache = true,
    uint32_t MaxNocLocations = device_print_dispatch::DEFAULT_MAX_NOC_LOCATIONS,
    uint32_t NocL1ToL1Alignment = device_print_dispatch::NOC_L1_TO_L1_ALIGNMENT,
    uint32_t NocL1ToDramAlignment = device_print_dispatch::NOC_L1_TO_DRAM_ALIGNMENT>
class DevicePrintDispatch {
    static constexpr uint32_t rw_pointers_entry_size =
        std::max(NocL1ToL1Alignment, static_cast<uint32_t>(sizeof(uint32_t) * 2));
    static_assert((NocL1ToL1Alignment & (NocL1ToL1Alignment - 1)) == 0, "NocL1ToL1Alignment must be power of 2");
    static_assert((NocL1ToDramAlignment & (NocL1ToDramAlignment - 1)) == 0, "NocL1ToDramAlignment must be power of 2");

public:
    void init(
        uint32_t noc_locations,
        uint32_t noc_locations_count,
        uint32_t l1_cache_buffer_address,
        uint32_t l1_cache_buffer_size,
        uint64_t cycles_for_stall_detection,
        uint64_t cycles_for_full_dispatch) {
        this->noc_locations = (volatile tt_l1_ptr device_print_dispatch::NocLocationInputInfo*)noc_locations;
        this->noc_locations_count = noc_locations_count;
        this->l1_cache_buffer_address = l1_cache_buffer_address;
        this->l1_cache_buffer_end = l1_cache_buffer_address + l1_cache_buffer_size;
        this->cycles_for_stall_detection = cycles_for_stall_detection;
        this->cycles_for_full_dispatch = cycles_for_full_dispatch;
        num_noc_locations_to_process = 0;

        // Align the start of rw pointers buffer
        l1_rw_pointers_buffer_start = (l1_cache_buffer_address + NocL1ToL1Alignment - 1) & ~(NocL1ToL1Alignment - 1);
        l1_device_print_buffer_start = l1_rw_pointers_buffer_start + rw_pointers_entry_size * noc_locations_count;
        if constexpr (NocL1ToDramAlignment > NocL1ToL1Alignment) {
            l1_device_print_buffer_start =
                (l1_device_print_buffer_start + NocL1ToDramAlignment - 1) & ~(NocL1ToDramAlignment - 1);
        } else {
            l1_device_print_buffer_start =
                (l1_device_print_buffer_start + NocL1ToL1Alignment - 1) & ~(NocL1ToL1Alignment - 1);
        }

        // Check if buffer is large enough to hold necessary data and turn off feature in DRAM if needed.
        uint32_t min_buffer_end = l1_device_print_buffer_start;
        for (uint32_t i = 0; i < noc_locations_count; i++) {
            uint32_t buffer_size = noc_locations[i].buf_size + std::max(NocL1ToDramAlignment, NocL1ToL1Alignment);

            min_buffer_end = std::max(min_buffer_end, l1_device_print_buffer_start + buffer_size);
        }
        if (min_buffer_end > l1_cache_buffer_address + l1_cache_buffer_size) {
            // Buffer is not large enough to hold data for all NOC locations read/write pointers and the biggest
            // device_print buffer. Disable dispatching to DRAM and fallback to host only reading buffers.
            enabled = false;

            // TODO: Write data to DRAM that will tell host to do fallback.
        }

        // Initialize cache for noc addresses if enabled
        if constexpr (EnableNocLocationCache) {
            for (uint32_t i = 0; i < noc_locations_count; i++) {
                uint32_t remote_l1_address = noc_locations[i].rw_ptr_addr;
                uint64_t rw_noc_addr = get_noc_addr(noc_locations[i].x, noc_locations[i].y, remote_l1_address);
                rw_noc_addresses[i] = rw_noc_addr | ((uint64_t)noc_locations[i].high_address_bits << 32);
            }
        }
    }

    void notify_kernel_start() { next_stall_detection_timestamp = get_timestamp() + cycles_for_stall_detection; }

    void execute_stall_detection() {
        if (enabled && get_timestamp() >= next_stall_detection_timestamp) {
            read_rw_pointers();
            find_noc_locations_to_process<true>();
            process_noc_locations<true>();

            // Update timestamp for next stall detection
            next_stall_detection_timestamp = get_timestamp() + cycles_for_stall_detection;
        }
    }

    void execute_full_dispatch() {
        uint64_t current_timestamp = get_timestamp();
        if (enabled && current_timestamp >= next_full_dispatch_timestamp) {
            // Check if we should execute fetch read/write pointers or we can reuse what stall detection read recently.
            if (current_timestamp - last_rw_pointers_read_timestamp >= cycles_for_full_dispatch / 2) {
                read_rw_pointers();
            }

            find_noc_locations_to_process<false>();
            process_noc_locations<false>();

            // Update timestamp for next full dispatch
            next_full_dispatch_timestamp = get_timestamp() + cycles_for_full_dispatch;
        }
    }

private:
    void read_rw_pointers() {
        uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start;
        for (uint32_t i = 0; i < noc_locations_count; i++, rw_pointer_address_in_l1 += rw_pointers_entry_size) {
            // Get NOC address for read/write pointers.
            uint64_t rw_noc_address;

            if constexpr (EnableNocLocationCache) {
                rw_noc_address = rw_noc_addresses[i];
            } else {
                uint32_t remote_l1_address = noc_locations[i].rw_ptr_addr;
                rw_noc_address = get_noc_addr(noc_locations[i].x, noc_locations[i].y, remote_l1_address);
            }

            // Calculate alignment for the NOC read.
            uint32_t alignment = rw_noc_address & (NocL1ToL1Alignment - 1);

            // Issue NOC read to read the read/write pointers into L1 buffer.
            noc_async_read(rw_noc_address, rw_pointer_address_in_l1 + alignment, 8);
        }
        noc_async_read_barrier();
        last_rw_pointers_read_timestamp = get_timestamp();
    }

    template <bool stall_detection>
    void find_noc_locations_to_process() {
        uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start;
        uint32_t num_noc_locations_to_process = 0;

        for (uint32_t i = 0; i < noc_locations_count; i++, rw_pointer_address_in_l1 += rw_pointers_entry_size) {
            uint32_t remote_l1_address = noc_locations[i].rw_ptr_addr;
            uint32_t alignment = remote_l1_address & (NocL1ToL1Alignment - 1);
            volatile tt_l1_ptr uint32_t* rw_pointers =
                (volatile tt_l1_ptr uint32_t*)(rw_pointer_address_in_l1 + alignment);
            uint32_t write_position = rw_pointers[0];
            [[maybe_unused]] uint32_t read_position = rw_pointers[1];

            // Ignore cores that have their write pointer in starting magic or disabled magic, as those cores are not
            // active in printing.
            if (write_position == DEBUG_PRINT_SERVER_STARTING_MAGIC ||
                write_position == DEBUG_PRINT_SERVER_DISABLED_MAGIC) {
                continue;
            }

            if constexpr (stall_detection) {
                // Check if write pointer has stall flag active
                if ((write_position & DEVICE_PRINT_WRITE_STALL_FLAG) == 0) {
                    continue;
                }
            } else {
                // Ignore if write pointer is equal to read pointer, as that means there is no new messages in the
                // buffer.
                if (write_position == read_position) {
                    continue;
                }
            }
            noc_locations_to_process[num_noc_locations_to_process++] = i;
        }
        this->num_noc_locations_to_process = num_noc_locations_to_process;
    }

    void process_noc_locations() {
        uint32_t current_l1_buffer_address = l1_device_print_buffer_start;
        uint32_t next_index_to_dispatch = 0;

        for (uint32_t i = 0; i < num_noc_locations_to_process; i++) {
            uint32_t location_index = noc_locations_to_process[i];
            uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start + location_index * rw_pointers_entry_size;
            auto* noc_location = &noc_locations[location_index];
            uint32_t remote_rw_ptr_address = noc_location->rw_ptr_addr;
            uint32_t rw_ptr_alignment = remote_rw_ptr_address & (NocL1ToL1Alignment - 1);
            volatile tt_l1_ptr uint32_t* rw_pointers =
                (volatile tt_l1_ptr uint32_t*)(rw_pointer_address_in_l1 + rw_ptr_alignment);
            uint32_t write_position = rw_pointers[0];
            uint32_t read_position = rw_pointers[1];
            bool stall = (write_position & DEVICE_PRINT_WRITE_STALL_FLAG) == 0;
            write_position = write_position & ~DEVICE_PRINT_WRITE_STALL_FLAG;

            uint32_t remote_buffer_address = noc_location->buf_addr;
            uint32_t remote_buffer_size = noc_location->buf_size;

            if (write_position > read_position) {
                remote_buffer_size = write_position - read_position;
                remote_buffer_address = remote_buffer_address + read_position;
            }

            // Find L1 to L1 alignment for copying buffer here.
            uint32_t buffer_l1_alignment = remote_buffer_address & (NocL1ToL1Alignment - 1);

            // If there isn't enough space for DRAM stream message header, we need to slide the buffer in L1 to make
            // space and align it properly.
            if (buffer_l1_alignment < sizeof(device_print_dispatch::DramStreamMessageHeader)) {
                buffer_l1_alignment += NocL1ToL1Alignment;
            }

            // Check if there is enough space in the buffer
            uint32_t local_buffer_end = current_l1_buffer_address + buffer_l1_alignment + remote_buffer_size;

            if (local_buffer_end > l1_cache_buffer_end) {
                // Wait for all NOC read transfers to finish.
                noc_async_read_barrier();

                // Push data to DRAM before processing next buffers.
                push_data_to_dram(current_l1_buffer_address - l1_device_print_buffer_start);

                // Issue writes to NOC locations to update read pointers.
                update_read_pointers(next_index_to_dispatch, i);

                // Wait for NOC writes to finish.
                noc_async_write_barrier();

                // Update next index to dispatch.
                next_index_to_dispatch = i;

                // After processing reset the current buffer address to the start of device_print buffers in L1.
                current_l1_buffer_address = l1_device_print_buffer_start;
            }

            // Start NOC read to copy device_print buffer from remote L1 to local L1 for processing.
            uint64_t noc_remote_buffer_address = get_noc_addr(noc_location->x, noc_location->y, remote_buffer_address);
            noc_remote_buffer_address |= (uint64_t)noc_location->high_address_bits << 32;
            noc_async_read(
                noc_remote_buffer_address, current_l1_buffer_address + buffer_l1_alignment, remote_buffer_size);

            // Write DRAM stream message header to the buffer start.
            volatile tt_l1_ptr device_print_dispatch::DramStreamMessageHeader* header =
                (volatile tt_l1_ptr device_print_dispatch::DramStreamMessageHeader*)current_l1_buffer_address;
            bool buffer_wrapped = write_position < read_position;

            header->x = noc_location->x;
            header->y = noc_location->y;
            header->align = buffer_l1_alignment;
            header->buffer_wrapped = buffer_wrapped;
            header->length = remote_buffer_size;

            // Check if we need second DRAM stream message for read/write pointers.
            if (buffer_wrapped) {
                volatile tt_l1_ptr uint16_t* rw_pointers;
                constexpr uint32_t rw_pointers_size = sizeof(uint16_t) * 2;

                // Check if we should write second message right after header message, or after buffer.
                if (buffer_l1_alignment >= sizeof(device_print_dispatch::DramStreamMessageHeader) + rw_pointers_size) {
                    rw_pointers =
                        (volatile tt_l1_ptr uint16_t*)(current_l1_buffer_address +
                                                       sizeof(device_print_dispatch::DramStreamMessageHeader));
                } else {
                    rw_pointers = (volatile tt_l1_ptr uint16_t*)(current_l1_buffer_address + buffer_l1_alignment +
                                                                 remote_buffer_size);
                    remote_buffer_size += sizeof(device_print_dispatch::DramStreamMessageHeader);
                }
                rw_pointers[0] = write_position;
                rw_pointers[1] = read_position;
            }

            // Update read position that will be send to NOC location later.
            rw_pointers[1] = stall ? DEVICE_PRINT_RESET_BUFFER_MAGIC : write_position;

            // Update current buffer address for next iterations.
            current_l1_buffer_address += buffer_l1_alignment + remote_buffer_size;
            if constexpr (NocL1ToDramAlignment > NocL1ToL1Alignment) {
                current_l1_buffer_address =
                    (current_l1_buffer_address + NocL1ToDramAlignment - 1) & ~(NocL1ToDramAlignment - 1);
            } else {
                current_l1_buffer_address =
                    (current_l1_buffer_address + NocL1ToL1Alignment - 1) & ~(NocL1ToL1Alignment - 1);
            }
        }

        // Wait for all NOC read transfers to finish.
        noc_async_read_barrier();

        // Push data to DRAM before processing next buffers.
        push_data_to_dram(current_l1_buffer_address - l1_device_print_buffer_start);

        // Issue writes to NOC locations to update read pointers.
        update_read_pointers(next_index_to_dispatch, num_noc_locations_to_process);

        // Wait for NOC writes to finish.
        noc_async_write_barrier();
    }

    void push_data_to_dram(uint32_t buffer_size) {
        uint32_t buffer_address = l1_device_print_buffer_start;
        // TODO: Issue write to DRAM (from l1_device_print_buffer_start to current_l1_buffer_address).
        // TODO: Handle cases like there isn't enough space for whole buffer in DRAM and we need to split it into
        // multiple messages.
        // TODO: Wait for enough space in DRAM.
        // TODO: Update write pointer in DRAM.
    }

    void update_read_pointers(uint32_t start_index, uint32_t end_index) {
        for (uint32_t j = start_index; j < end_index; j++) {
            uint32_t index = noc_locations_to_process[j];
            uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start + index * rw_pointers_entry_size;
            uint32_t remote_l1_address = noc_locations[index].rw_ptr_addr;
            uint32_t alignment = remote_l1_address & (NocL1ToL1Alignment - 1);
            volatile tt_l1_ptr uint32_t* rw_pointers =
                (volatile tt_l1_ptr uint32_t*)(rw_pointer_address_in_l1 + alignment);
            uint64_t w_noc_addr;
            if constexpr (EnableNocLocationCache) {
                w_noc_addr = rw_noc_addresses[index];
            } else {
                w_noc_addr = get_noc_addr(noc_locations[index].x, noc_locations[index].y, remote_l1_address);
                w_noc_addr |= (uint64_t)noc_locations[index].high_address_bits << 32;
            }

            // +4 is to update read pointer which is after write pointer.
            noc_async_write(rw_pointer_address_in_l1 + alignment + 4, w_noc_addr + 4, 4);
        }
    }

    // Input data from host about NOC locations and their device_print buffer info
    volatile tt_l1_ptr device_print_dispatch::NocLocationInputInfo* noc_locations;
    uint32_t noc_locations_count;

    // Buffer in L1 for copying read/write pointers and device_print buffers of remote NOC locations for processing.
    uint32_t l1_cache_buffer_address;
    uint32_t l1_cache_buffer_end;
    uint32_t l1_rw_pointers_buffer_start;
    uint32_t l1_device_print_buffer_start;
    bool enabled = true;

    // NOC locations that are marked to be processed in current iteration.
    uint8_t noc_locations_to_process[MaxNocLocations];
    uint32_t num_noc_locations_to_process;

    // Number of cycles for events
    uint64_t cycles_for_stall_detection;
    uint64_t cycles_for_full_dispatch;

    // Timestamps for last events
    uint64_t last_rw_pointers_read_timestamp;
    uint64_t next_stall_detection_timestamp;
    uint64_t next_full_dispatch_timestamp;

    // Cache for NOC addresses of read/write pointers of remote NOC locations to avoid converting them in every
    // iteration.
    uint64_t rw_noc_addresses[EnableNocLocationCache ? MaxNocLocations : 0];
};

// TODO: Check if we should have separate implementation of process_noc_locations for D2H sockets...
