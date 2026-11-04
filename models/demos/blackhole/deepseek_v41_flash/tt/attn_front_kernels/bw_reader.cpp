// attention-front bandwidth microbenchmark, NCRISC: stream the bank shard (14 blocks of 80 bfp8 tiles) into a 3-slot
// ring and forward every block to the partner core. modes: 0 none, 1 unicast by NCRISC (NOC0), 2 multicast (rectangle
// of the two cores) by NCRISC, 3 even blocks by NCRISC (NOC0) + odd blocks by BRISC (NOC1, semaphore) CT: 0 ring cb, 1
// bank, 2 vc, 3 w addr, 4 mode, 5 sem id; RT: partner x, y

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t bank = get_compile_time_arg_val(1);
    constexpr uint32_t vc = get_compile_time_arg_val(2);
    constexpr uint32_t waddr = get_compile_time_arg_val(3);
    constexpr uint32_t mode = get_compile_time_arg_val(4);
    constexpr uint32_t sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t TB = 1088, PAGE_B = 10 * TB, BLK_PAGES = 8, NBLK = 14, NSLOT = 3, NT = 3;
    const uint32_t px = get_arg_val<uint32_t>(0), py = get_arg_val<uint32_t>(1);
    const uint32_t ring = get_write_ptr(cb);
    reset_noc_trid_barrier_counter(NOC_CLEAR_OUTSTANDING_REQ_MASK, noc_index);
    uint64_t base = get_noc_addr_from_bank_id<true>(bank, waddr);
    noc_async_read_one_packet_set_state<true>(base, PAGE_B, vc);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(sem_id));
    if constexpr (mode == 5) {
        noc_async_write_one_packet_set_state<true>(get_noc_addr(px, py, 0), PAGE_B);
    }
    uint32_t off = 0;
    for (uint32_t i = 0; i < NBLK + NT; ++i) {
        if (i < NBLK) {
            const uint32_t trid = (i % NT) + 1;
            noc_async_read_set_trid(trid);
            uint32_t dst = ring + (i % NSLOT) * BLK_PAGES * PAGE_B;
            for (uint32_t p = 0; p < BLK_PAGES; ++p) {
                noc_async_read_one_packet_with_state_with_trid(base, off, dst, trid);
                off += PAGE_B;
                dst += PAGE_B;
            }
        }
        if (i >= NT - 1 && i - (NT - 1) < NBLK) {
            const uint32_t j = i - (NT - 1);
            noc_async_read_barrier_with_trid((j % NT) + 1);
            const uint32_t src = ring + (j % NSLOT) * BLK_PAGES * PAGE_B;
            if (mode == 1 || (mode == 3 && (j & 1) == 0)) {
                noc_async_write(src, get_noc_addr(px, py, src), BLK_PAGES * PAGE_B);
            } else if (mode == 4) {
                noc_async_write<NOC_MAX_BURST_SIZE + 1, true, true>(src, get_noc_addr(px, py, src), BLK_PAGES * PAGE_B);
            } else if (mode == 5) {
                for (uint32_t pg = 0; pg < BLK_PAGES; ++pg) {
                    noc_async_write_one_packet_with_state<true>(src + pg * PAGE_B, src + pg * PAGE_B);
                }
            } else if (mode == 2) {
                noc_async_write_multicast(src, get_noc_multicast_addr(px, py, px, py + 0, src), BLK_PAGES * PAGE_B, 1);
            } else if (mode == 3) {
                *sem = j;  // tell BRISC the odd block j is ready in slot j % NSLOT
            }
        }
    }
    noc_async_write_barrier();
}
