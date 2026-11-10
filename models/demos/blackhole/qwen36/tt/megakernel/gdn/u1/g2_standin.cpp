// G2 harness: the engine's side of the GDN role I/O on one core (the Q root's job in the engine). Per GDN layer g:
// spin `gap` cycles (the attention / in-proj time the engine spends before the layer's W-GDN start), in_ready += 1 by
// multicast on the GDN rects, then wait until out_ready (this core) reaches ncores * (g + 1). At the end: reset
// out_ready (its waiter, spec v1.33) and record the cycle stamps of every layer's in / out edges in `stamps`.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"

void kernel_main() {
    uint32_t a = 0;
    const uint32_t ng = get_arg_val<uint32_t>(a++), gap = get_arg_val<uint32_t>(a++);
    const uint32_t inr = get_arg_val<uint32_t>(a++), outr = get_arg_val<uint32_t>(a++), ncores = get_arg_val<uint32_t>(a++);
    const uint32_t stamps = get_arg_val<uint32_t>(a++);
    uint32_t r[2][5];
    for (uint32_t i = 0; i < 2; ++i) {
        for (uint32_t j = 0; j < 5; ++j) {
            r[i][j] = get_arg_val<uint32_t>(a++);
        }
    }
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamps);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(outr);
    for (uint32_t g = 0; g < ng; ++g) {
        const uint64_t t0 = get_timestamp();
        while (get_timestamp() - t0 < gap) {
        }
        st[2 * g] = static_cast<uint32_t>(get_timestamp());
        for (uint32_t i = 0; i < 2; ++i) {
            if (r[i][4]) {
                noc_semaphore_inc_multicast(get_noc_multicast_addr(r[i][0], r[i][1], r[i][2], r[i][3], inr), 1, r[i][4]);
            }
        }
        noc_async_atomic_barrier();
        {
            DeviceZoneScopedN("G2-OUTW");
            noc_semaphore_wait_min(o, ncores * (g + 1));
        }
        st[2 * g + 1] = static_cast<uint32_t>(get_timestamp());
    }
    *o = 0;
}
