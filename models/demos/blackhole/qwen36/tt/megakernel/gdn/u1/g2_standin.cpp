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
    // G-b harness (E1_GB): before each layer's in_ready, G-a's stand-in -- the layer's qkv row tiles (the same tensor the
    // stock-op reference reads) NoC-written into the GDN cores' staging slots (tile t -> core xy_t, gin + slot_t * 2 KB)
    const uint32_t ngin = get_arg_val<uint32_t>(a++), qkv_addr = get_arg_val<uint32_t>(a++), gin = get_arg_val<uint32_t>(a++);
    const uint32_t gtab = a;
    constexpr auto qkv_args = TensorAccessorArgs<0>();
    const auto qkv = TensorAccessor(qkv_args, qkv_addr, 2048);
    const uint32_t scr = get_write_ptr(0);
    volatile tt_l1_ptr uint32_t* st = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stamps);
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(outr);
    for (uint32_t g = 0; g < ng; ++g) {
        const uint64_t t0 = get_timestamp();
        while (get_timestamp() - t0 < gap) {
        }
        for (uint32_t t = 0; t < ngin; ++t) {
            const uint32_t xy = get_arg_val<uint32_t>(gtab + 2 * t), sl = get_arg_val<uint32_t>(gtab + 2 * t + 1);
            noc_async_read(qkv.get_noc_addr(t), scr, 2048);
            noc_async_read_barrier();
            noc_async_write(scr, get_noc_addr(xy & 0xFFFFu, xy >> 16, gin + sl * 2048), 2048);
            noc_async_write_barrier();
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
