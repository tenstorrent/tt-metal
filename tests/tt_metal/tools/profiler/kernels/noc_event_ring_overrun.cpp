// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Data-movement kernel of test_noc_event_ring_overrun. BRISC fills its L1 profiler ring to a known index with 2-word
// event records: the ring holds PROFILER_L1_VECTOR_SIZE = 512 words, user records start at CUSTOM_MARKERS = 12 and
// DeviceRecordEvent writes while `wIndex < 512 - stackSize` (no headroom for the quick push), so after BRISC_EVENTS
// records the index is 12 + 2 * BRISC_EVENTS (249 -> 510, 250 -> 512). With BRISC_TAIL_ZONE it then opens and closes
// an empty zone (a zone opened at 510 only fits if its END is accounted for). Then one DRAM page read: with NoC-event
// tracing on, the event's own room check (which does reserve that headroom) fails and quick_push pushes the ring to
// DRAM; its own START/END pair must stay inside BRISC's ring, which the NCRISC ring follows in L1. NCRISC only does
// the read, so its ring exists and its header gets pushed to DRAM.
//
// With BRISC_MIDZONE_PUSH (no event records) BRISC instead opens an outer zone, fills the ring to 510 with empty
// nested zones, does the read inside the outer zone (a quick push with one zone open), then fills the fresh ring with
// nested zones again. The outer zone's END is written last: it fits only if the quick push left wIndex + stackSize on
// the 4-word grid that the zones' START-only room check relies on.
//
// Compile-time defines: BRISC_EVENTS, optionally BRISC_TAIL_ZONE or BRISC_MIDZONE_PUSH. Compile args:
// TensorAccessorArgs of the DRAM page buffer at 0. Runtime args: page size, DRAM buffer address.
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
#if defined(BRISC_TAIL_ZONE)
    {
        DeviceZoneScopedN("TAIL-ZONE");
    }
#endif
#if defined(BRISC_MIDZONE_PUSH)
    {
        DeviceZoneScopedN("OUTER-ZONE");  // START at 12, wIndex 14
        for (uint32_t i = 0; i < 124; i++) {
            DeviceZoneScopedN("FILL-ZONE");  // 124 x 4 words: wIndex 510
        }
        noc_async_read_page(0, dram, l1_dst);  // quick push with stackSize 2
        noc_async_read_barrier();
        for (uint32_t i = 0; i < 125; i++) {
            DeviceZoneScopedN("FILL-ZONE");  // fill the fresh ring as far as the room check allows
        }
    }
#endif
#endif
    noc_async_read_page(0, dram, l1_dst);
    noc_async_read_barrier();
}
