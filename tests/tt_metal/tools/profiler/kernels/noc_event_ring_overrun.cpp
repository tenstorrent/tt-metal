// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernel of test_noc_event_ring_overrun. BRISC fills its L1 profiler ring to a known index with 2-word
// event records: the ring holds PROFILER_L1_VECTOR_SIZE = 512 words, user records start at CUSTOM_MARKERS = 12 and
// DeviceRecordEvent writes while `wIndex < 512 - stackSize` (no headroom for the quick push), so after BRISC_EVENTS
// records the index is 12 + 2 * BRISC_EVENTS (249 -> 510, 250 -> 512). Then one DRAM page read: with NoC-event
// tracing on, the event's own room check (which does reserve that headroom) fails and quick_push writes its START/END
// pair at the current index without any check, past the end of BRISC's ring and into NCRISC's. NCRISC only does the
// read, so its ring exists and its header gets pushed to DRAM.
//
// Compile-time defines: BRISC_EVENTS. Compile args: TensorAccessorArgs of the DRAM page buffer at 0. Runtime args:
// page size, DRAM buffer address.
#include <cstdint>

void kernel_main() {
    const uint32_t page_size = get_arg_val<uint32_t>(0);
    const uint32_t dram_base = get_arg_val<uint32_t>(1);
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto dram = TensorAccessor(accessor_args, dram_base, page_size);
    const uint32_t l1_dst = get_write_ptr(0);

#if defined(COMPILE_FOR_BRISC)
    for (uint32_t i = 0; i < BRISC_EVENTS; i++) {
        DeviceRecordEvent(i + 1);  // ids from 1: the host test treats a marker with timer id 0 as a corrupted header
    }
#endif
    noc_async_read_page(0, dram, l1_dst);
    noc_async_read_barrier();
}
