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
    uint16_t x : 6;
    uint16_t y : 6;
    uint64_t rw_ptr_addr : 52;
    uint16_t buf_offset;
    uint16_t buf_size;
} __attribute__((packed, aligned(4)));

static_assert(sizeof(NocLocationInputInfo) == 12, "NocLocationInputInfo must be 12 bytes");
static_assert(sizeof(NocLocationInputInfo) % 4 == 0, "NocLocationInputInfo must be 4-byte aligned");

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
        uint32_t noc_locations_ptr,
        uint32_t noc_locations_count,
        uint32_t l1_cache_buffer_address,
        uint32_t l1_cache_buffer_size,
        uint64_t cycles_for_stall_detection,
        uint64_t cycles_for_full_dispatch) {
        noc_locations = (volatile tt_l1_ptr device_print_dispatch::NocLocationInputInfo*)noc_locations_ptr;
        this->noc_locations_count = noc_locations_count;
        this->l1_cache_buffer_address = l1_cache_buffer_address;
        this->l1_cache_buffer_end = l1_cache_buffer_address + l1_cache_buffer_size;
        this->cycles_for_stall_detection = cycles_for_stall_detection;
        this->cycles_for_full_dispatch = cycles_for_full_dispatch;
        num_noc_locations_to_process = 0;

        // Align the start of rw pointers buffer
        l1_rw_pointers_buffer_start = l1_align(l1_cache_buffer_address);
        l1_dram_rw_pointers = dram_align(l1_rw_pointers_buffer_start + rw_pointers_entry_size * noc_locations_count);
        l1_device_print_buffer_start = dram_align(l1_dram_rw_pointers + sizeof(uint32_t));

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

            // Write data to DRAM that will tell host to do fallback.
            volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;
            dram_rw_pointers[0] = DEBUG_PRINT_SERVER_DISABLED_MAGIC;
            noc_async_write(l1_dram_rw_pointers, noc_dram_rw_pointers, sizeof(uint32_t));
            noc_async_write_barrier();
        }

        // Initialize cache for noc addresses if enabled
        if constexpr (EnableNocLocationCache) {
            for (uint32_t i = 0; i < noc_locations_count; i++) {
                rw_noc_addresses[i] =
                    get_noc_addr64(noc_locations[i].x, noc_locations[i].y, noc_locations[i].rw_ptr_addr);
                cache_buffer_offsets[i] = noc_locations[i].buf_offset;
                cache_buffer_sizes[i] = noc_locations[i].buf_size;
            }
        }
    }

    void notify_kernel_start() { next_stall_detection_timestamp = get_timestamp() + cycles_for_stall_detection; }

    void execute_stall_detection() {
        if (enabled && get_timestamp() >= next_stall_detection_timestamp) {
            read_rw_pointers();
            find_noc_locations_to_process<true>();
            process_noc_locations();

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
            process_noc_locations();

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
                rw_noc_address = get_noc_addr64(noc_locations[i].x, noc_locations[i].y, noc_locations[i].rw_ptr_addr);
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
            uint64_t remote_l1_address;
            if constexpr (EnableNocLocationCache) {
                remote_l1_address = rw_noc_addresses[i];
            } else {
                remote_l1_address = noc_locations[i].rw_ptr_addr;
            }
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
                // Ignore if write pointer is equal to read pointer, as there are no new messages in the buffer.
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
            uint64_t remote_rw_ptr_address;
            if constexpr (EnableNocLocationCache) {
                remote_rw_ptr_address = rw_noc_addresses[location_index];
            } else {
                remote_rw_ptr_address = noc_location->rw_ptr_addr;
            }
            uint32_t rw_ptr_alignment = remote_rw_ptr_address & (NocL1ToL1Alignment - 1);
            volatile tt_l1_ptr uint32_t* rw_pointers =
                (volatile tt_l1_ptr uint32_t*)(rw_pointer_address_in_l1 + rw_ptr_alignment);
            uint32_t write_position = rw_pointers[0];
            uint32_t read_position = rw_pointers[1];
            bool stall = (write_position & DEVICE_PRINT_WRITE_STALL_FLAG) == 0;
            write_position = write_position & ~DEVICE_PRINT_WRITE_STALL_FLAG;

            uint64_t remote_buffer_address;
            uint32_t remote_buffer_size;

            if constexpr (EnableNocLocationCache) {
                remote_buffer_address = cache_buffer_offsets[i] + remote_rw_ptr_address;
                remote_buffer_size = cache_buffer_sizes[i];
            } else {
                remote_buffer_address = noc_location->buf_offset + remote_rw_ptr_address;
                remote_buffer_size = noc_location->buf_size;
            }

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
            uint32_t local_buffer_end =
                dram_align(current_l1_buffer_address + buffer_l1_alignment + remote_buffer_size);

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
            uint64_t noc_remote_buffer_address;
            if constexpr (EnableNocLocationCache) {
                // Cached address already has x,y embedded
                noc_remote_buffer_address = remote_buffer_address;
            } else {
                // Non-cached address needs to be converted to NOC address with x,y coordinates
                noc_remote_buffer_address = get_noc_addr64(noc_location->x, noc_location->y, remote_buffer_address);
            }
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
            current_l1_buffer_address += dram_align(buffer_l1_alignment + remote_buffer_size);
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
        // All buffers and pointers are DRAM aligned. We don't need to think about that.
        // We just need to align local buffer size.
        uint32_t buffer_address = l1_device_print_buffer_start;

        buffer_size = dram_align(buffer_size);

        // Check if DRAM buffer is overflowing and write what we can until end of the buffer.
        if (dram_write_pointer >= dram_read_pointer) {
            // Check if we can write whole buffer without wrapping around.
            if (buffer_size <= dram_buffer_size - dram_write_pointer) {
                noc_async_write(buffer_address, dram_write_pointer + noc_dram_buffer_start, buffer_size);
                dram_write_pointer += buffer_size;
                buffer_size = 0;
            } else {
                // We need to split buffer into two parts and write them separately.
                uint32_t first_part_size = dram_buffer_size - dram_write_pointer;

                noc_async_write(buffer_address, dram_write_pointer + noc_dram_buffer_start, first_part_size);
                buffer_size -= first_part_size;
                buffer_address += first_part_size;
                dram_write_pointer = 0;
            }
        }

        // Pointer to DRAM read/write pointers in our local L1.
        volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;

        // Check if there is remaining buffer to write
        if (buffer_size > 0) {
            // We know that buffer is overflowing.
            // Wait until there is enough space in the buffer.
            while (buffer_size > dram_read_pointer - dram_write_pointer) {
                // Read updated read pointer from DRAM.
                noc_async_read(
                    l1_dram_rw_pointers + sizeof(uint32_t), noc_dram_rw_pointers + sizeof(uint32_t), sizeof(uint32_t));
                noc_async_read_barrier();
                dram_read_pointer = dram_rw_pointers[1];
            }

            noc_async_write(buffer_address, dram_write_pointer + noc_dram_buffer_start, buffer_size);
            dram_write_pointer += buffer_size;
        }

        // Wait until all data is written to DRAM before returning to make sure data is visible to host.
        noc_async_write_barrier();

        // Update write pointer in DRAM.
        dram_rw_pointers[0] = dram_write_pointer;
        noc_async_write(l1_dram_rw_pointers, noc_dram_rw_pointers, sizeof(uint32_t));
        noc_async_write_barrier();
    }

    void update_read_pointers(uint32_t start_index, uint32_t end_index) {
        for (uint32_t j = start_index; j < end_index; j++) {
            uint32_t i = noc_locations_to_process[j];
            uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start + i * rw_pointers_entry_size;
            uint64_t remote_l1_address;

            if constexpr (EnableNocLocationCache) {
                remote_l1_address = rw_noc_addresses[i];
            } else {
                remote_l1_address =
                    get_noc_addr64(noc_locations[i].x, noc_locations[i].y, noc_locations[i].rw_ptr_addr);
            }

            uint32_t alignment = remote_l1_address & (NocL1ToL1Alignment - 1);

            // +sizeof(uint32_t) is to update read pointer which is after write pointer.
            noc_async_write(
                rw_pointer_address_in_l1 + alignment + sizeof(uint32_t),
                remote_l1_address + sizeof(uint32_t),
                sizeof(uint32_t));
        }
    }

    static uint32_t l1_align(uint32_t address) {
        return (address + NocL1ToL1Alignment - 1) & ~(NocL1ToL1Alignment - 1);
    }

    static uint32_t dram_align(uint32_t address) {
        if constexpr (NocL1ToDramAlignment > NocL1ToL1Alignment) {
            return (address + NocL1ToDramAlignment - 1) & ~(NocL1ToDramAlignment - 1);
        } else {
            return (address + NocL1ToL1Alignment - 1) & ~(NocL1ToL1Alignment - 1);
        }
    }

    // Input data from host about NOC locations and their device_print buffer info
    volatile tt_l1_ptr device_print_dispatch::NocLocationInputInfo* noc_locations;
    uint32_t noc_locations_count;

    // Buffer in L1 for copying read/write pointers and device_print buffers of remote NOC locations for processing.
    uint32_t l1_cache_buffer_address;
    uint32_t l1_cache_buffer_end;
    uint32_t l1_rw_pointers_buffer_start;
    uint32_t l1_dram_rw_pointers;
    uint32_t l1_device_print_buffer_start;
    bool enabled = true;

    // NOC locations that are marked to be processed in current iteration.
    uint8_t noc_locations_to_process[MaxNocLocations];
    uint32_t num_noc_locations_to_process;

    // DRAM address where we will push data for host to read.
    uint32_t dram_read_pointer = 0;
    uint32_t dram_write_pointer = 0;
    uint64_t noc_dram_rw_pointers;
    uint64_t noc_dram_buffer_start;
    uint32_t dram_buffer_size;

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
    uint16_t cache_buffer_offsets[EnableNocLocationCache ? MaxNocLocations : 0];
    uint16_t cache_buffer_sizes[EnableNocLocationCache ? MaxNocLocations : 0];
};

// TODO: Check if we should have separate implementation of process_noc_locations for D2H sockets...
