// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Calibration kernel of test_noc_event_mesh_fill (MESH_FILL_KERNEL=calib): BRISC fills its profiler ring with a
// known number of 2-word event records (DeviceRecordEvent, written while bufferHasRoom says so, never pushed), so
// the ring index is known exactly, then issues CALIB_READS DRAM page reads: the first NoC event's flush check runs
// with the ring at that index and, if it finds no room, quick_push writes its own START/END pair there. The ring is
// 512 words and starts at 12, so CALIB_EVENTS = 250 puts the index at exactly 512 (the first word of the NCRISC ring).
// CALIB_IN_ZONE = 1 fills inside a zone (its END closes the fill), 0 fills with no zone open. NCRISC only does the
// reads (its ring must exist and be pushed so a corrupted header reaches DRAM).
//
// Compile-time defines: CALIB_EVENTS, CALIB_IN_ZONE, CALIB_READS. Compile args: TensorAccessorArgs of the DRAM page
// buffer at 0. Runtime args 0-2: page size, page count, DRAM buffer address.
#include <cstdint>

void kernel_main() {
    const uint32_t page_size = get_arg_val<uint32_t>(0);
    const uint32_t num_pages = get_arg_val<uint32_t>(1);
    const uint32_t dram_base = get_arg_val<uint32_t>(2);
    constexpr auto accessor_args = TensorAccessorArgs<0>();
    const auto dram = TensorAccessor(accessor_args, dram_base, page_size);
    const uint32_t l1_dst = get_write_ptr(0);

#if defined(COMPILE_FOR_BRISC)
#if CALIB_IN_ZONE
    {
        DeviceZoneScopedN("CALIB-FILL");
        for (uint32_t i = 0; i < CALIB_EVENTS; i++) {
            DeviceRecordEvent(i);
        }
    }
#else
    for (uint32_t i = 0; i < CALIB_EVENTS; i++) {
        DeviceRecordEvent(i);
    }
#endif
#endif
    for (uint32_t r = 0; r < CALIB_READS; r++) {
        noc_async_read_page(r % num_pages, dram, l1_dst);
    }
    noc_async_read_barrier();
}
