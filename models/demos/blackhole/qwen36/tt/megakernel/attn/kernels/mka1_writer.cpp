// MKA1 writer (BRISC, NOC0).
//  1. ones tile for the denominator column (CB_ONE).
//  2. Q: the Q root reads all 4 own users' Q (32 tiles) and multicasts them (two rects: the DRAM column splits the
//     Tensix block) to every core, then multicasts the Q flag; every core exposes CB_Q to compute.
//  3. STREAM cores: the concurrent weight prefetch of the megakernel (this layer's wo slice + the next layer's MLP
//     slices), modelled as real DRAM reads of the same bytes into a scratch region (contends with the KV stream).
//  4. Fanout-6 tree reduce of the per-core O_aug partials (pull model: a parent reads each ready child's CB_O into
//     CB_R, then releases the child), then the group root writes the normalized output.
#include "api/debug/kernel_profiler.hpp"
#include "api/dataflow/dataflow_api.h"
#include "mka1_common.hpp"

FORCE_INLINE volatile tt_l1_ptr uint32_t* sem(uint32_t id) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(id));
}

void kernel_main() {
    constexpr uint32_t QTB = get_compile_time_arg_val(0), PFTB = get_compile_time_arg_val(1), OTB = get_compile_time_arg_val(2);
    constexpr uint32_t OATB = get_compile_time_arg_val(3);
    constexpr uint32_t VNOC0 = get_compile_time_arg_val(4), NBB = get_compile_time_arg_val(5);
    constexpr uint32_t KVTB = PFTB;   // KV and the prefetch stand-in are both BFP4
    constexpr auto q_args = TensorAccessorArgs<6>();
    constexpr auto pf_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<pf_args.next_compile_time_args_offset()>();
    constexpr auto kv_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    uint32_t a = 0;
    const uint32_t is_qroot = get_arg_val<uint32_t>(a++), q_addr = get_arg_val<uint32_t>(a++);
    uint32_t rc[2][5];
    for (uint32_t i = 0; i < 2; ++i) for (uint32_t k = 0; k < 5; ++k) rc[i][k] = get_arg_val<uint32_t>(a++);
    const uint32_t self_rect = get_arg_val<uint32_t>(a++);
    const uint32_t is_stream = get_arg_val<uint32_t>(a++), pf_addr = get_arg_val<uint32_t>(a++);
    const uint32_t pf_page0 = get_arg_val<uint32_t>(a++), pf_tiles = get_arg_val<uint32_t>(a++);
    const uint32_t has_parent = get_arg_val<uint32_t>(a++), par_x = get_arg_val<uint32_t>(a++), par_y = get_arg_val<uint32_t>(a++);
    const uint32_t par_lvl = get_arg_val<uint32_t>(a++);
    const uint32_t nc0 = get_arg_val<uint32_t>(a++);
    uint32_t c0x[FANOUT], c0y[FANOUT];
    for (uint32_t i = 0; i < FANOUT - 1; ++i) { c0x[i] = get_arg_val<uint32_t>(a++); c0y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t nc1 = get_arg_val<uint32_t>(a++);
    uint32_t c1x[FANOUT], c1y[FANOUT];
    for (uint32_t i = 0; i < FANOUT - 1; ++i) { c1x[i] = get_arg_val<uint32_t>(a++); c1y[i] = get_arg_val<uint32_t>(a++); }
    const uint32_t is_root = get_arg_val<uint32_t>(a++), out_addr = get_arg_val<uint32_t>(a++), user = get_arg_val<uint32_t>(a++);
    const uint32_t kv_addr = get_arg_val<uint32_t>(a++), vb0 = get_arg_val<uint32_t>(a++), vb1 = get_arg_val<uint32_t>(a++), NB = get_arg_val<uint32_t>(a++);

    const uint32_t o_base = get_read_ptr(CB_O);   // identical on every core (same CB config)
    {   // 1. ones tile (bf16 1.0 = 0x3F80): any layout is all-ones
        cb_reserve_back(CB_ONE, 1);
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CB_ONE));
        for (uint32_t i = 0; i < QTB / 4; ++i) p[i] = 0x3F803F80u;
        cb_push_back(CB_ONE, 1);
    }
    {   // 2. Q multicast
        DeviceZoneScopedN("W-Q");
        cb_reserve_back(CB_Q, QT);
        const uint32_t qdst = get_write_ptr(CB_Q);
        if (is_qroot) {
            const auto q = TensorAccessor(q_args, q_addr, QTB);
            for (uint32_t t = 0; t < QT; ++t) noc_async_read(q.get_noc_addr(t), qdst + t * QTB, QTB);
            noc_async_read_barrier();
            for (uint32_t i = 0; i < 2; ++i) {
                if (rc[i][4] == 0) continue;
                const uint64_t m = get_noc_multicast_addr(rc[i][0], rc[i][1], rc[i][2], rc[i][3], qdst);
                if (i == self_rect) noc_async_write_multicast_loopback_src(qdst, m, QT * QTB, rc[i][4], false);
                else noc_async_write_multicast(qdst, m, QT * QTB, rc[i][4], false);
            }
            noc_async_write_barrier();
            *sem(SEM_QFLAG) = 1;
            for (uint32_t i = 0; i < 2; ++i) {
                if (rc[i][4] == 0) continue;
                const uint64_t m = get_noc_multicast_addr(rc[i][0], rc[i][1], rc[i][2], rc[i][3], get_semaphore(SEM_QFLAG));
                if (i == self_rect) noc_semaphore_set_multicast_loopback_src(get_semaphore(SEM_QFLAG), m, rc[i][4], false);
                else noc_semaphore_set_multicast(get_semaphore(SEM_QFLAG), m, rc[i][4], false);
            }
            noc_async_write_barrier();
        }
        noc_semaphore_wait_min(sem(SEM_QFLAG), 1);
        cb_push_back(CB_Q, QT);
    }
    {   // 3. weight-prefetch contention (STREAM cores), and with VNOC0 the V half of the KV stream on NOC0,
        //    interleaved one prefetch chunk per V group
        DeviceZoneScopedN("W-PF");
        const auto pf = TensorAccessor(pf_args, pf_addr, PFTB);
        const auto kv = TensorAccessor(kv_args, kv_addr, KVTB);
        const uint32_t scratch = get_write_ptr(CB_PF);
        constexpr uint32_t CH = 32;
        uint32_t c = 0;
        auto pf_chunk = [&]() {
            if (!is_stream || c >= pf_tiles) return;
            const uint32_t n = (pf_tiles - c) < CH ? (pf_tiles - c) : CH;
            for (uint32_t t = 0; t < n; ++t) noc_async_read(pf.get_noc_addr(pf_page0 + c + t), scratch + t * PFTB, PFTB);
            c += n;
        };
        if constexpr (VNOC0) {
            for (uint32_t b = vb0; b < vb1; b += NBB) {
                const uint32_t nb = (vb1 - b) < NBB ? (vb1 - b) : NBB;
                cb_reserve_back(CB_V, nb * DH);
                const uint32_t wv = get_write_ptr(CB_V);
                for (uint32_t i = 0; i < nb; ++i) {
                    const uint32_t pk = ((user * NB + b + i) * 2) * DH;
                    for (uint32_t t = 0; t < DH; ++t) noc_async_read(kv.get_noc_addr(pk + DH + t), wv + (i * DH + t) * KVTB, KVTB);
                }
                pf_chunk();
                noc_async_read_barrier();
                cb_push_back(CB_V, nb * DH);
            }
        }
        while (is_stream && c < pf_tiles) { pf_chunk(); noc_async_read_barrier(); }
    }
    *sem(SEM_QFLAG) = 0;
    noc_async_atomic_barrier();
}
