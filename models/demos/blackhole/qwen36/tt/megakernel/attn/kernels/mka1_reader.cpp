// MKA1 reader (NCRISC, NOC1): stream this core's contiguous slice of its user's BFP4 KV cache, one 32-position block
// (8 K tiles + 8 V tiles) at a time, into CB_K / CB_V (depth-buffered in the ATTN working region).
// KV page index = ((u * NB + b) * 2 + kv) * 8 + t (kv 0 = K, 1 = V).
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include "mka1_common.hpp"

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

void kernel_main() {
    constexpr uint32_t KVTB = get_compile_time_arg_val(0);
    constexpr uint32_t OTB = get_compile_time_arg_val(1), OATB = get_compile_time_arg_val(2), NBB = get_compile_time_arg_val(3);
    constexpr uint32_t PLACE = get_compile_time_arg_val(4), NBANK = get_compile_time_arg_val(5), VNOC0 = get_compile_time_arg_val(6);
    constexpr auto kv_args = TensorAccessorArgs<7>();
    constexpr auto out_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t kv_addr = get_arg_val<uint32_t>(a++), u = get_arg_val<uint32_t>(a++);
    const uint32_t b0 = get_arg_val<uint32_t>(a++), b1 = get_arg_val<uint32_t>(a++), NB = get_arg_val<uint32_t>(a++);
    const uint32_t beta = get_arg_val<uint32_t>(a++), s0b = get_arg_val<uint32_t>(a++);
    const uint32_t has_parent = get_arg_val<uint32_t>(a++), par_x = get_arg_val<uint32_t>(a++), par_y = get_arg_val<uint32_t>(a++);
    const uint32_t par_lvl = get_arg_val<uint32_t>(a++);
    const uint32_t nc0 = get_arg_val<uint32_t>(a++);
    uint32_t c0x[FANOUT], c0y[FANOUT];
    for (uint32_t i = 0; i < FANOUT - 1; ++i) { c0x[i] = get_arg_val<uint32_t>(a++); c0y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t nc1 = get_arg_val<uint32_t>(a++);
    uint32_t c1x[FANOUT], c1y[FANOUT];
    for (uint32_t i = 0; i < FANOUT - 1; ++i) { c1x[i] = get_arg_val<uint32_t>(a++); c1y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t is_root = get_arg_val<uint32_t>(a++), out_addr = get_arg_val<uint32_t>(a++), user = get_arg_val<uint32_t>(a++);
    const uint32_t o_base = get_read_ptr(CB_O);   // identical on every core (same CB config)
    const auto kv = TensorAccessor(kv_args, kv_addr, KVTB);
    {
    DeviceZoneScopedN("R-KV");
    for (uint32_t b = b0; b < b1; b += NBB) {   // NBB blocks per reservation: ~NBB x 9 KB of reads in flight
        const uint32_t nb = (b1 - b) < NBB ? (b1 - b) : NBB;
        cb_reserve_back(CB_K, nb * DH);
        if constexpr (!VNOC0) cb_reserve_back(CB_V, nb * DH);
        const uint32_t wk = get_write_ptr(CB_K), wv = get_write_ptr(CB_V);
        for (uint32_t i = 0; i < nb; ++i) {
            if constexpr (PLACE) {
                // bank-local: this core's blocks b0..b1 all sit in its bank beta, 16 tiles each, from slot s0b
                const uint32_t s0 = (s0b + (b + i - b0)) * (2 * DH);
                for (uint32_t t = 0; t < DH; ++t) {
                    noc_async_read(kv.get_noc_addr((s0 + t) * NBANK + beta), wk + (i * DH + t) * KVTB, KVTB);
                    noc_async_read(kv.get_noc_addr((s0 + DH + t) * NBANK + beta), wv + (i * DH + t) * KVTB, KVTB);
                }
            } else {
                const uint32_t pk = ((u * NB + b + i) * 2) * DH;
                for (uint32_t t = 0; t < DH; ++t) {
                    noc_async_read(kv.get_noc_addr(pk + t), wk + (i * DH + t) * KVTB, KVTB);
                    if constexpr (!VNOC0) noc_async_read(kv.get_noc_addr(pk + DH + t), wv + (i * DH + t) * KVTB, KVTB);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_K, nb * DH);
        if constexpr (!VNOC0) cb_push_back(CB_V, nb * DH);
    }
    }
    {   // 4. tree reduce (pull)
        DeviceZoneScopedN("R-RED");
        auto pull = [&](uint32_t sem_id, uint32_t n, const uint32_t* cx, const uint32_t* cy) {
            if (n == 0) return;
            noc_semaphore_wait_min(sem(sem_id), n);
            for (uint32_t i = 0; i < n; ++i) {
                cb_reserve_back(CB_R, NO);
                noc_async_read(get_noc_addr(cx[i], cy[i], o_base), get_write_ptr(CB_R), NO * OTB);
                noc_async_read_barrier();
                cb_push_back(CB_R, NO);
                noc_semaphore_inc(get_noc_addr(cx[i], cy[i], get_semaphore(SEM_DONE)), 1);
            }
            *sem(sem_id) = 0;
        };
        pull(SEM_LVL0, nc0, c0x, c0y);
        pull(SEM_LVL1, nc1, c1x, c1y);
        if (has_parent) {
            cb_wait_front(CB_O, NO);
            noc_semaphore_inc(get_noc_addr(par_x, par_y, get_semaphore(SEM_LVL0 + par_lvl)), 1);
            noc_semaphore_wait_min(sem(SEM_DONE), 1);
            *sem(SEM_DONE) = 0;
            cb_pop_front(CB_O, NO);
        }
    }
    if (is_root) {
        DeviceZoneScopedN("R-OUT");
        const auto out = TensorAccessor(out_args, out_addr, OATB);
        cb_wait_front(CB_OA, DH);
        const uint32_t rp = get_read_ptr(CB_OA);
        for (uint32_t t = 0; t < DH; ++t) noc_async_write(rp + t * OATB, out.get_noc_addr(user * DH + t), OATB);
        noc_async_write_barrier();
        cb_pop_front(CB_OA, DH);
    }
    noc_async_atomic_barrier();
}
