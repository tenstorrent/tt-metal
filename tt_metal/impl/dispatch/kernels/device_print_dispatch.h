// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/device_print.h"
#include "internal/risc_attribs.h"
#include "noc_parameters.h"
#include "risc_common.h"

namespace device_print_dispatch {

#if defined(ARCH_WORMHOLE)
constexpr uint32_t DEFAULT_MAX_NOC_LOCATIONS = 8 * 10    // Tensix cores
                                               + 8 * 2;  // ETH cores
// Noc reads/writes must be 16-byte aligned.
constexpr uint32_t NOC_L1_TO_L1_ALIGNMENT = L1_ALIGNMENT;      // 16
constexpr uint32_t NOC_L1_TO_DRAM_ALIGNMENT = DRAM_ALIGNMENT;  // 32
#elif defined(ARCH_BLACKHOLE)
constexpr uint32_t DEFAULT_MAX_NOC_LOCATIONS = 14 * 10    // Tensix cores
                                               + 14 * 1   // ETH cores
                                               + 2 * 12;  // DRAM cores
// Noc reads/writes must be 16-byte aligned.
constexpr uint32_t NOC_L1_TO_L1_ALIGNMENT = L1_ALIGNMENT;      // 16
constexpr uint32_t NOC_L1_TO_DRAM_ALIGNMENT = DRAM_ALIGNMENT;  // 64
#else
constexpr uint32_t DEFAULT_MAX_NOC_LOCATIONS = 0;
constexpr uint32_t NOC_L1_TO_L1_ALIGNMENT = 0;
constexpr uint32_t NOC_L1_TO_DRAM_ALIGNMENT = 0;
static_assert(false, "Unsupported architecture");
#endif

// Default no-op guard. DevicePrintDispatch is generic and makes no assumptions about the
// caller's NOC cmd_buf state; callers that need to snapshot/restore cmd_buf state around
// the dispatcher's NOC traffic pass their own RAII type as the NocCmdBufGuard template
// argument.
struct EmptyNocCmdBufGuard {};

// Default guard with initialization of cmd_buf state.
template <typename DerivedGuard>
struct NocCmdBufGuardWithInit : DerivedGuard {
    NocCmdBufGuardWithInit() {
        noc_read_init_state<NCRISC_RD_CMD_BUF>(NOC_INDEX);
        noc_write_init_state<NCRISC_WR_CMD_BUF>(NOC_INDEX, NOC_UNICAST_WRITE_VC);
    }
};

}  // namespace device_print_dispatch

template <
    bool EnableNocLocationCache = true,
    uint32_t MaxNocLocations = device_print_dispatch::DEFAULT_MAX_NOC_LOCATIONS,
    uint32_t NocL1ToL1Alignment = device_print_dispatch::NOC_L1_TO_L1_ALIGNMENT,
    uint32_t NocL1ToDramAlignment = device_print_dispatch::NOC_L1_TO_DRAM_ALIGNMENT,
    typename NocCmdBufGuard = device_print_dispatch::EmptyNocCmdBufGuard>
class DevicePrintDispatch {
    static constexpr uint32_t rw_pointers_entry_size =
        std::max(NocL1ToL1Alignment, static_cast<uint32_t>(sizeof(uint32_t) * 2));
    static_assert((NocL1ToL1Alignment & (NocL1ToL1Alignment - 1)) == 0, "NocL1ToL1Alignment must be power of 2");
    static_assert((NocL1ToDramAlignment & (NocL1ToDramAlignment - 1)) == 0, "NocL1ToDramAlignment must be power of 2");
    using NocCmdBufGuardWithInit = device_print_dispatch::NocCmdBufGuardWithInit<NocCmdBufGuard>;

public:
    void init(
        uint32_t noc_locations_ptr,
        uint32_t noc_locations_count,
        uint32_t l1_cache_buffer_address,
        uint32_t l1_cache_buffer_size,
        uint16_t dram_x,
        uint16_t dram_y,
        uint64_t dram_rw_pointers,
        uint64_t dram_buffer_start,
        uint32_t dram_buffer_size,
        uint64_t cycles_for_stall_detection,
        uint64_t cycles_for_full_dispatch) {
        noc_locations = (volatile tt_l1_ptr device_print_dispatch::NocLocationInputInfo*)noc_locations_ptr;
        this->noc_locations_count = noc_locations_count;
        this->l1_cache_buffer_address = l1_cache_buffer_address;
        this->l1_cache_buffer_end = l1_cache_buffer_address + l1_cache_buffer_size;
        this->cycles_for_stall_detection = cycles_for_stall_detection;
        this->cycles_for_full_dispatch = cycles_for_full_dispatch;
        num_noc_locations_to_process = 0;
        dram_noc_xy = NOC_XY_ENCODING(DYNAMIC_NOC_X(NOC_INDEX, dram_x), DYNAMIC_NOC_Y(NOC_INDEX, dram_y));
        dram_rw_pointers_addr = dram_rw_pointers;
        dram_buffer_start_addr = dram_buffer_start;
        this->dram_buffer_size = dram_buffer_size;
        dram_read_pointer = 0;
        dram_write_pointer = 0;
        enabled = true;

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

        NocCmdBufGuardWithInit guard;

        if (min_buffer_end > l1_cache_buffer_address + l1_cache_buffer_size) {
            // Buffer is not large enough to hold data for all NOC locations read/write pointers and the biggest
            // device_print buffer. Disable dispatching to DRAM and fallback to host only reading buffers.
            enabled = false;

            // Write disabled magic plus diagnostics: word[0]=magic, word[1]=0 (rpos),
            // word[2]=provided L1 cache size, word[3]=minimum L1 cache size that would
            // have made the aggregator usable. Host reads these to produce an actionable
            // warning. The leading 32-byte (DRAM-aligned) slot in DRAM has plenty of
            // room for these four words.
            volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;
            dram_rw_pointers[0] = DEBUG_PRINT_SERVER_DISABLED_MAGIC;
            dram_rw_pointers[1] = 0;
            dram_rw_pointers[2] = l1_cache_buffer_size;
            dram_rw_pointers[3] = min_buffer_end - l1_cache_buffer_address;
            noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX, l1_dram_rw_pointers, dram_noc_xy, dram_rw_pointers_addr, 4 * sizeof(uint32_t));
            noc_async_write_barrier();
        } else {
            // Clear STARTING_MAGIC the host wrote at attach time so it can distinguish
            // "kernel hasn't booted yet" (STARTING_MAGIC) from "kernel is alive but has
            // produced no payload yet" (write_pointer == 0).
            volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;
            dram_rw_pointers[0] = 0;
            dram_rw_pointers[1] = 0;
            noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX, l1_dram_rw_pointers, dram_noc_xy, dram_rw_pointers_addr, 2 * sizeof(uint32_t));
            noc_async_write_barrier();
        }

        // Initialize cache for noc addresses if enabled
        if constexpr (EnableNocLocationCache) {
            for (uint32_t i = 0; i < noc_locations_count; i++) {
                cache_x[i] = noc_locations[i].x;
                cache_y[i] = noc_locations[i].y;
                cache_noc_xy_encodings[i] = NOC_XY_ENCODING(
                    DYNAMIC_NOC_X(NOC_INDEX, noc_locations[i].x), DYNAMIC_NOC_Y(NOC_INDEX, noc_locations[i].y));
                cache_rw_ptr_addrs[i] = noc_locations[i].rw_ptr_addr;
                cache_buffer_offsets[i] = noc_locations[i].buf_offset;
                cache_buffer_sizes[i] = noc_locations[i].buf_size;
            }
        }
    }

    void notify_kernel_start() { next_stall_detection_timestamp = get_timestamp() + cycles_for_stall_detection; }

    void shutdown() {
        if (!enabled) {
            return;
        }

        NocCmdBufGuardWithInit guard;

        // Execute last full dispatch to drain any remaining buffers in DRAM before shutdown.
        read_rw_pointers();
        find_noc_locations_to_process<false>();
        process_noc_locations();

        // Mark the DRAM rw-pointer cell as "finished" so the host knows
        // it can stop polling DRAM and fall back to per-L1 polling.
        volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;
        dram_rw_pointers[4] = 1;
        noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
            NOC_INDEX,
            l1_dram_rw_pointers + 4 * sizeof(uint32_t),
            dram_noc_xy,
            dram_rw_pointers_addr + 4 * sizeof(uint32_t),
            sizeof(uint32_t));
        noc_async_write_barrier();
    }

    void execute(bool force_stall = false) {
        if (!enabled) {
            return;
        }

        // Execute stall detection if needed
        if (force_stall || (get_timestamp() >= next_stall_detection_timestamp)) {
            NocCmdBufGuardWithInit guard;

            read_rw_pointers();
            find_noc_locations_to_process<true>();
            process_noc_locations();

            // Update timestamp for next stall detection
            next_stall_detection_timestamp = get_timestamp() + cycles_for_stall_detection;
        }

        // Execute full dispatch if needed
        uint64_t current_timestamp = get_timestamp();
        if (enabled && current_timestamp >= next_full_dispatch_timestamp) {
            NocCmdBufGuardWithInit guard;

            // Check if we should execute fetch read/write pointers or we can reuse what stall detection read recently.
            if (current_timestamp - last_rw_pointers_read_timestamp >= cycles_for_full_dispatch / 2) {
                read_rw_pointers();
            }

            find_noc_locations_to_process<false>();

            // Everything up until now was only in local L1. Before we do any processing,
            // that might involve DRAM, check for host reset. On host side, we will wait
            // for at least one full dispatch window after writing the reset magic before
            // we rely on it, so we are guaranteed to see it here if the host reset happened.
            check_for_host_reset();

            process_noc_locations();

            // Update timestamp for next full dispatch
            next_full_dispatch_timestamp = get_timestamp() + cycles_for_full_dispatch;
        }
    }

private:
    void read_rw_pointers() {
        uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start;
        for (uint32_t i = 0; i < noc_locations_count; i++, rw_pointer_address_in_l1 += rw_pointers_entry_size) {
            uint32_t noc_xy;
            uint64_t rw_ptr_addr;
            if constexpr (EnableNocLocationCache) {
                noc_xy = cache_noc_xy_encodings[i];
                rw_ptr_addr = cache_rw_ptr_addrs[i];
            } else {
                noc_xy = NOC_XY_ENCODING(
                    DYNAMIC_NOC_X(NOC_INDEX, noc_locations[i].x), DYNAMIC_NOC_Y(NOC_INDEX, noc_locations[i].y));
                rw_ptr_addr = noc_locations[i].rw_ptr_addr;
            }

            // Calculate alignment for the NOC read based on the local L1 address only.
            uint32_t alignment = (uint32_t)rw_ptr_addr & (NocL1ToL1Alignment - 1);

            // Issue NOC read to read the read/write pointers into L1 buffer.
            noc_read_with_state<DM_DEDICATED_NOC, NCRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX, noc_xy, rw_ptr_addr, rw_pointer_address_in_l1 + alignment, 8);
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
                remote_l1_address = cache_rw_ptr_addrs[i];
            } else {
                remote_l1_address = noc_locations[i].rw_ptr_addr;
            }
            uint32_t alignment = (uint32_t)remote_l1_address & (NocL1ToL1Alignment - 1);
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
                // Skip cores whose cached state reflects an in-flight stall handshake.
                if ((write_position & DEVICE_PRINT_WRITE_STALL_FLAG) != 0 ||
                    read_position == DEVICE_PRINT_RESET_BUFFER_MAGIC) {
                    continue;
                }
            }
            noc_locations_to_process[num_noc_locations_to_process++] = i;
        }
        this->num_noc_locations_to_process = num_noc_locations_to_process;
    }

    void check_for_host_reset() {
        volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;
        noc_read_with_state<DM_DEDICATED_NOC, NCRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
            NOC_INDEX, dram_noc_xy, dram_rw_pointers_addr, l1_dram_rw_pointers, sizeof(uint32_t));
        noc_async_read_barrier();
        if (dram_rw_pointers[0] == DEBUG_PRINT_SERVER_STARTING_MAGIC) {
            dram_read_pointer = 0;
            dram_write_pointer = 0;
            dram_rw_pointers[0] = 0;
            noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX, l1_dram_rw_pointers, dram_noc_xy, dram_rw_pointers_addr, sizeof(uint32_t));
            noc_async_write_barrier();
        }
    }

    void process_noc_locations() {
        uint32_t current_l1_buffer_address = l1_device_print_buffer_start;
        uint32_t next_index_to_dispatch = 0;

        for (uint32_t i = 0; i < num_noc_locations_to_process; i++) {
            uint32_t location_index = noc_locations_to_process[i];
            uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start + location_index * rw_pointers_entry_size;
            auto* noc_location = &noc_locations[location_index];
            uint32_t remote_noc_xy;
            uint64_t remote_rw_ptr_address;
            if constexpr (EnableNocLocationCache) {
                remote_noc_xy = cache_noc_xy_encodings[location_index];
                remote_rw_ptr_address = cache_rw_ptr_addrs[location_index];
            } else {
                remote_noc_xy = NOC_XY_ENCODING(
                    DYNAMIC_NOC_X(NOC_INDEX, noc_location->x), DYNAMIC_NOC_Y(NOC_INDEX, noc_location->y));
                remote_rw_ptr_address = noc_location->rw_ptr_addr;
            }
            uint32_t rw_ptr_alignment = (uint32_t)remote_rw_ptr_address & (NocL1ToL1Alignment - 1);
            volatile tt_l1_ptr uint32_t* rw_pointers =
                (volatile tt_l1_ptr uint32_t*)(rw_pointer_address_in_l1 + rw_ptr_alignment);
            uint32_t write_position = rw_pointers[0];
            uint32_t read_position = rw_pointers[1];
            // The kernel sets STALL_FLAG bit on wpos right before entering one of the
            // three wait_for_space busy loops. Each of those loops explicitly checks for
            // rpos == DEVICE_PRINT_RESET_BUFFER_MAGIC and atomically resets the buffer.
            // So when the flag is present we can safely send RESET_MAGIC. When the flag
            // is absent the kernel may be actively writing — racing it with RESET_MAGIC
            // would clobber in-flight writes — so we send rpos = wpos instead and let
            // the kernel catch up next iteration.
            bool kernel_in_wait_loop = (write_position & DEVICE_PRINT_WRITE_STALL_FLAG) != 0;
            write_position = write_position & ~DEVICE_PRINT_WRITE_STALL_FLAG;

            uint64_t remote_buffer_address;
            uint32_t remote_buffer_size;

            if constexpr (EnableNocLocationCache) {
                remote_buffer_address = cache_buffer_offsets[location_index] + remote_rw_ptr_address;
                remote_buffer_size = cache_buffer_sizes[location_index];
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

                // Update next index to dispatch.
                next_index_to_dispatch = i;

                // After processing reset the current buffer address to the start of device_print buffers in L1.
                current_l1_buffer_address = l1_device_print_buffer_start;
            }

            // Start NOC read to copy device_print buffer from remote L1 to local L1 for processing.
            buffer_read_chunked(
                remote_noc_xy,
                remote_buffer_address,
                current_l1_buffer_address + buffer_l1_alignment,
                remote_buffer_size);

            // Write DRAM stream message header to the buffer start.
            volatile tt_l1_ptr device_print_dispatch::DramStreamMessageHeader* header =
                (volatile tt_l1_ptr device_print_dispatch::DramStreamMessageHeader*)current_l1_buffer_address;
            bool buffer_wrapped = write_position < read_position;

            if constexpr (EnableNocLocationCache) {
                header->x = cache_x[location_index];
                header->y = cache_y[location_index];
            } else {
                header->x = noc_location->x;
                header->y = noc_location->y;
            }
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
                    remote_buffer_size += rw_pointers_size;
                }
                rw_pointers[0] = write_position;
                rw_pointers[1] = read_position;
            }

            // Update read position that will be sent to NOC location later. See comment
            // on `kernel_in_wait_loop` above for why MAGIC is safe iff that's true.
            rw_pointers[1] = kernel_in_wait_loop ? DEVICE_PRINT_RESET_BUFFER_MAGIC : write_position;

            // Update current buffer address for next iterations.
            current_l1_buffer_address += dram_align(buffer_l1_alignment + remote_buffer_size);
        }

        if (next_index_to_dispatch < num_noc_locations_to_process) {
            // Wait for all NOC read transfers to finish.
            noc_async_read_barrier();

            // Push data to DRAM before processing next buffers.
            push_data_to_dram(current_l1_buffer_address - l1_device_print_buffer_start);

            // Issue writes to NOC locations to update read pointers.
            update_read_pointers(next_index_to_dispatch, num_noc_locations_to_process);
        }
    }

    // Write `size` bytes from local L1 `src_addr` to DRAM at (`dram_noc_xy`, `dst_addr`), split into
    // <= NOC_MAX_BURST_SIZE packets. A single noc_wwrite_with_state with size > NOC_MAX_BURST_SIZE is
    // fragmented by the NIU into multiple packets (each bumping the hardware NIU_MST_NONPOSTED_WR_REQ_SENT
    // / NIU_MST_WR_ACK_RECEIVED counters) while the software counters are incremented only once. That
    // desync makes noc_async_write_barrier (exact HW==SW) unsatisfiable. Issuing each burst-sized packet
    // explicitly keeps the software counters in step with the NIU. (Device-print DRAM buffers can reach
    // tens of KB, far above the 8 KB burst, so this matters here; the 4-byte rw-pointer writes do not.)
    FORCE_INLINE void dram_write_chunked(uint32_t src_addr, uint64_t dst_addr, uint32_t size) {
        while (size > NOC_MAX_BURST_SIZE) {
            noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX, src_addr, dram_noc_xy, dst_addr, NOC_MAX_BURST_SIZE);
            src_addr += NOC_MAX_BURST_SIZE;
            dst_addr += NOC_MAX_BURST_SIZE;
            size -= NOC_MAX_BURST_SIZE;
        }
        noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
            NOC_INDEX, src_addr, dram_noc_xy, dst_addr, size);
    }

    // Read `size` bytes from remote L1 (`noc_xy` + `src_addr`) into local L1 `dst_addr`, split into
    // <= NOC_MAX_BURST_SIZE packets — same reasoning as dram_write_chunked, but for the read counters
    // (NIU_MST_RD_RESP_RECEIVED vs noc_reads_num_issued) that noc_async_read_barrier checks.
    FORCE_INLINE void buffer_read_chunked(uint32_t noc_xy, uint64_t src_addr, uint32_t dst_addr, uint32_t size) {
        while (size > NOC_MAX_BURST_SIZE) {
            noc_read_with_state<DM_DEDICATED_NOC, NCRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX, noc_xy, src_addr, dst_addr, NOC_MAX_BURST_SIZE);
            src_addr += NOC_MAX_BURST_SIZE;
            dst_addr += NOC_MAX_BURST_SIZE;
            size -= NOC_MAX_BURST_SIZE;
        }
        noc_read_with_state<DM_DEDICATED_NOC, NCRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
            NOC_INDEX, noc_xy, src_addr, dst_addr, size);
    }

    void push_data_to_dram(uint32_t buffer_size) {
        // All buffers and pointers are DRAM aligned. We don't need to think about that.
        // We just need to align local buffer size.
        uint32_t buffer_address = l1_device_print_buffer_start;

        buffer_size = dram_align(buffer_size);

        // Pointer to DRAM read/write pointers in our local L1.
        volatile tt_l1_ptr uint32_t* dram_rw_pointers = (volatile tt_l1_ptr uint32_t*)l1_dram_rw_pointers;

        if (buffer_size > 0) {
            // The write pointer must never be advanced to equal the read pointer. Both this producer and
            // the host reader treat dram_write_pointer == dram_read_pointer as EMPTY. If a write ever
            // made wp == rp while the ring actually held unread data (a FULL ring), the host would stop
            // draining and we would overwrite it — silent, timing-dependent data loss. We therefore
            // reserve a one-alignment-unit gap and wait until the ring has room for the WHOLE message
            // before writing any of it.
            while (true) {
                // Free space that keeps wp != rp. When wp == rp the ring is empty (whole ring free).
                uint32_t free;
                if (dram_read_pointer == dram_write_pointer) {
                    free = dram_buffer_size;
                } else if (dram_read_pointer > dram_write_pointer) {
                    free = dram_read_pointer - dram_write_pointer;
                } else {
                    free = dram_buffer_size - (dram_write_pointer - dram_read_pointer);
                }
                free -= dram_align(1);
                if (free >= buffer_size) {
                    break;
                }
                // Not enough room yet — re-read the host's read pointer from DRAM and wait.
                noc_read_with_state<DM_DEDICATED_NOC, NCRISC_RD_CMD_BUF, CQ_NOC_SNDL>(
                    NOC_INDEX,
                    dram_noc_xy,
                    dram_rw_pointers_addr + sizeof(uint32_t),
                    l1_dram_rw_pointers + sizeof(uint32_t),
                    sizeof(uint32_t));
                noc_async_read_barrier();
                dram_read_pointer = dram_rw_pointers[1];
            }

            // Write the message, splitting across the ring-end boundary if needed. The space check
            // above guarantees the resulting dram_write_pointer stays strictly short of the read
            // pointer (>= one gap unit away), so it can never alias the read pointer.
            uint32_t until_end = dram_buffer_size - dram_write_pointer;
            if (buffer_size <= until_end) {
                dram_write_chunked(buffer_address, dram_buffer_start_addr + dram_write_pointer, buffer_size);
                dram_write_pointer += buffer_size;
                if (dram_write_pointer == dram_buffer_size) {
                    dram_write_pointer = 0;
                }
            } else {
                dram_write_chunked(buffer_address, dram_buffer_start_addr + dram_write_pointer, until_end);
                dram_write_chunked(buffer_address + until_end, dram_buffer_start_addr, buffer_size - until_end);
                dram_write_pointer = buffer_size - until_end;
            }
        }

        // Wait until all data is written to DRAM before returning to make sure data is visible to host.
        noc_async_write_barrier();

        // Update write pointer in DRAM.
        dram_rw_pointers[0] = dram_write_pointer;
        noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
            NOC_INDEX, l1_dram_rw_pointers, dram_noc_xy, dram_rw_pointers_addr, sizeof(uint32_t));
        noc_async_write_barrier();
    }

    void update_read_pointers(uint32_t start_index, uint32_t end_index) {
        for (uint32_t j = start_index; j < end_index; j++) {
            uint32_t i = noc_locations_to_process[j];
            uint32_t rw_pointer_address_in_l1 = l1_rw_pointers_buffer_start + i * rw_pointers_entry_size;
            uint32_t remote_noc_xy;
            uint64_t remote_l1_address;

            if constexpr (EnableNocLocationCache) {
                remote_noc_xy = cache_noc_xy_encodings[i];
                remote_l1_address = cache_rw_ptr_addrs[i];
            } else {
                remote_noc_xy = NOC_XY_ENCODING(
                    DYNAMIC_NOC_X(NOC_INDEX, noc_locations[i].x), DYNAMIC_NOC_Y(NOC_INDEX, noc_locations[i].y));
                remote_l1_address = noc_locations[i].rw_ptr_addr;
            }

            uint32_t alignment = (uint32_t)remote_l1_address & (NocL1ToL1Alignment - 1);

            // +sizeof(uint32_t) is to update read pointer which is after write pointer.
            noc_wwrite_with_state<DM_DEDICATED_NOC, NCRISC_WR_CMD_BUF, CQ_NOC_SNDL>(
                NOC_INDEX,
                rw_pointer_address_in_l1 + alignment + sizeof(uint32_t),
                remote_noc_xy,
                remote_l1_address + sizeof(uint32_t),
                sizeof(uint32_t));
        }

        // Wait for NOC writes to finish.
        noc_async_write_barrier();
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

    // NOTE: None of the fields cannot be initialized here because this
    // class needs to be zero-initializable and any non-zero field would
    // generate .init_array entries for static instances of this class on
    // dispatch_s, which the dispatch_s ELF linker script rejects.

    // Input data from host about NOC locations and their device_print buffer info
    volatile tt_l1_ptr device_print_dispatch::NocLocationInputInfo* noc_locations;
    uint32_t noc_locations_count;

    // Buffer in L1 for copying read/write pointers and device_print buffers of remote NOC locations for processing.
    uint32_t l1_cache_buffer_address;
    uint32_t l1_cache_buffer_end;
    uint32_t l1_rw_pointers_buffer_start;
    uint32_t l1_dram_rw_pointers;
    uint32_t l1_device_print_buffer_start;
    bool enabled;

    // NOC locations that are marked to be processed in current iteration.
    uint8_t noc_locations_to_process[MaxNocLocations];
    uint32_t num_noc_locations_to_process;

    // DRAM address where we will push data for host to read.
    uint32_t dram_read_pointer;
    uint32_t dram_write_pointer;
    uint32_t dram_noc_xy;
    uint64_t dram_rw_pointers_addr;
    uint64_t dram_buffer_start_addr;
    uint32_t dram_buffer_size;

    // Number of cycles for events
    uint64_t cycles_for_stall_detection;
    uint64_t cycles_for_full_dispatch;

    // Timestamps for last events
    uint64_t last_rw_pointers_read_timestamp;
    uint64_t next_stall_detection_timestamp;
    uint64_t next_full_dispatch_timestamp;

    // Cache for the NOC xy encoding and local rw-pointer address per remote NOC location.
    uint32_t cache_noc_xy_encodings[EnableNocLocationCache ? MaxNocLocations : 0];
    uint64_t cache_rw_ptr_addrs[EnableNocLocationCache ? MaxNocLocations : 0];
    uint16_t cache_buffer_offsets[EnableNocLocationCache ? MaxNocLocations : 0];
    uint16_t cache_buffer_sizes[EnableNocLocationCache ? MaxNocLocations : 0];
    uint8_t cache_x[EnableNocLocationCache ? MaxNocLocations : 0];
    uint8_t cache_y[EnableNocLocationCache ? MaxNocLocations : 0];
};
