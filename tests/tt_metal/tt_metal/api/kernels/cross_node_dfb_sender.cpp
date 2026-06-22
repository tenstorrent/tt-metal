// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// CrossNodeDFB sender kernel
//
// Compile-time parameters (via kernel compile_args):
//   [0] remote_dfb_id      - runtime-assigned slot (return value of AttachCrossNodeDFB on host)
//   [1] entry_size         - bytes per entry (must be L1_ALIGNMENT multiple)
//   [2] num_entries        - number of entries to push per receiver
//   [3] write_primitive    - 0=write_multicast, 1=write_strided,
//                            2=write_to_receiver+push_back (receiver-contiguous),
//                            3=write_to_receiver+push_back_to_receiver (per-receiver credit)
//   [4] do_commit          - 1 to call commit() at end (for cross-program persistence)
//   [5] data_pattern       - 0=multicast counter layout, 1=strided per-receiver layout,
//                            2=per-receiver constant layout (see cross_node_dfb_test_utils.hpp)
//   [6] do_resize          - 1 to perform mid-flight resize after initial entries
//   [7] entry_size_resized - new entry_size after resize (valid if do_resize==1)
//   [8] num_entries_after  - entries to push after resize (valid if do_resize==1)
//   [9] do_barrier         - 1 to call barrier() after pushing all entries
//
// Runtime args:
//   [0] l1_staging_addr    - sender-local L1 scratch region pre-populated by the host

#include "api/dataflow/cross_node_dfb.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"

namespace {

constexpr uint32_t kPatternMulticastCounter = 0;
constexpr uint32_t kPatternStridedPerReceiver = 1;
constexpr uint32_t kPatternPerReceiverConstant = 2;

FORCE_INLINE uint32_t staging_addr(uint32_t staging_base, uint32_t byte_offset) { return staging_base + byte_offset; }

}  // namespace

void kernel_main() {
    constexpr uint8_t  remote_dfb_id      = get_compile_time_arg_val(0);
    constexpr uint32_t entry_size         = get_compile_time_arg_val(1);
    constexpr uint32_t num_entries        = get_compile_time_arg_val(2);
    constexpr uint32_t write_primitive    = get_compile_time_arg_val(3);
    constexpr uint32_t do_commit          = get_compile_time_arg_val(4);
    constexpr uint32_t data_pattern       = get_compile_time_arg_val(5);
    constexpr uint32_t do_resize          = get_compile_time_arg_val(6);
    constexpr uint32_t entry_size_resized = get_compile_time_arg_val(7);
    constexpr uint32_t num_entries_after  = get_compile_time_arg_val(8);
    constexpr uint32_t do_barrier         = get_compile_time_arg_val(9);

    const uint32_t staging_base = get_arg_val<uint32_t>(0);

    Noc noc;
    // Spot-check: log first byte of each entry (host pre-populated staging)
    DPRINT("l1_staging_addr: 0x{:x}\n", staging_base);
    for (uint32_t i = 0; i < num_entries; ++i) {
        const volatile uint8_t* p = reinterpret_cast<const volatile uint8_t*>(staging_base + i * entry_size);
        DPRINT("staging entry[{}] first byte: 0x{:02x}\n", i, (uint32_t)p[0]);
    }

    experimental::CrossNodeDFB gdfb(remote_dfb_id);
    DPRINT("gdfb initial write_ptr: 0x{:x}\n", gdfb.get_write_ptr());

    DPRINT("Running write_primitive: {}\n", write_primitive);

    static_assert(
        write_primitive != 0 || data_pattern == kPatternMulticastCounter,
        "write_multicast expects multicast counter staging");
    static_assert(
        write_primitive != 1 || data_pattern == kPatternStridedPerReceiver,
        "write_strided expects strided staging");
    static_assert(
        write_primitive != 2 || data_pattern == kPatternPerReceiverConstant,
        "write_to_receiver expects per-receiver staging");
    static_assert(
        write_primitive != 3 || data_pattern == kPatternPerReceiverConstant,
        "push_back_to_receiver expects per-receiver staging");

    if constexpr (write_primitive == 0) {
        for (uint32_t i = 0; i < num_entries; ++i) {
            DPRINT("Reserving back for multicast\n");
            gdfb.reserve_back(1);
            DPRINT("Done reserve back for multicast to {}\n", staging_addr(staging_base, i * entry_size));
            gdfb.write_multicast(staging_addr(staging_base, i * entry_size), 1, noc);
            DPRINT("Done write multicast\n");
            noc.async_write_barrier();
            DPRINT("Done async write barrier\n");
            gdfb.push_back(1, noc);
            DPRINT("Done push back\n");
        }
    } else if constexpr (write_primitive == 1) {
        const uint32_t num_recv = gdfb.num_receivers();
        const uint32_t row_bytes = num_recv * entry_size;
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            gdfb.write_strided(staging_addr(staging_base, i * row_bytes), 1, 1, entry_size, noc);
            noc.async_write_barrier();
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 2) {
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t i = 0; i < num_entries; ++i) {
            gdfb.reserve_back(1);
            for (uint32_t r = 0; r < num_recv; ++r) {
                gdfb.write_to_receiver(r, staging_addr(staging_base, r * entry_size), 1, noc);
            }
            noc.async_write_barrier();
            gdfb.push_back(1, noc);
        }
    } else if constexpr (write_primitive == 3) {
        const uint32_t num_recv = gdfb.num_receivers();
        for (uint32_t r = 0; r < num_recv; ++r) {
            for (uint32_t i = 0; i < num_entries; ++i) {
                gdfb.reserve_back_for_receiver(r, 1);
                gdfb.write_to_receiver(r, staging_addr(staging_base, r * entry_size), 1, noc);
                noc.async_write_barrier();
                gdfb.push_back_to_receiver(r, 1, noc);
            }
        }
    }

    if constexpr (do_resize) {
        gdfb.set_sender_entry_size(entry_size_resized, noc);

        if constexpr (write_primitive == 0) {
            const uint32_t resize_offset = num_entries * entry_size;
            for (uint32_t i = 0; i < num_entries_after; ++i) {
                gdfb.reserve_back(1);
                gdfb.write_multicast(staging_addr(staging_base, resize_offset + i * entry_size_resized), 1, noc);
                noc.async_write_barrier();
                gdfb.push_back(1, noc);
            }
        } else if constexpr (write_primitive == 1) {
            const uint32_t num_recv = gdfb.num_receivers();
            const uint32_t resize_offset = num_entries * num_recv * entry_size;
            const uint32_t row_bytes = num_recv * entry_size_resized;
            for (uint32_t i = 0; i < num_entries_after; ++i) {
                gdfb.reserve_back(1);
                gdfb.write_strided(
                    staging_addr(staging_base, resize_offset + i * row_bytes), 1, 1, entry_size_resized, noc);
                noc.async_write_barrier();
                gdfb.push_back(1, noc);
            }
        } else if constexpr (write_primitive == 2) {
            const uint32_t num_recv = gdfb.num_receivers();
            for (uint32_t i = 0; i < num_entries_after; ++i) {
                gdfb.reserve_back(1);
                for (uint32_t r = 0; r < num_recv; ++r) {
                    gdfb.write_to_receiver(r, staging_addr(staging_base, r * entry_size), 1, noc);
                }
                noc.async_write_barrier();
                gdfb.push_back(1, noc);
            }
        } else if constexpr (write_primitive == 3) {
            const uint32_t num_recv = gdfb.num_receivers();
            for (uint32_t r = 0; r < num_recv; ++r) {
                for (uint32_t i = 0; i < num_entries_after; ++i) {
                    gdfb.reserve_back_for_receiver(r, 1);
                    gdfb.write_to_receiver(r, staging_addr(staging_base, r * entry_size), 1, noc);
                    noc.async_write_barrier();
                    gdfb.push_back_to_receiver(r, 1, noc);
                }
            }
        }
    }

    if constexpr (do_barrier) {
        gdfb.barrier();
    }

    if constexpr (do_commit) {
        gdfb.commit();
    }
}
