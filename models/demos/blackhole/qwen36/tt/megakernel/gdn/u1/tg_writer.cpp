// lever 1 producer writer: the gate pairs from gt, the replay weights / decay from the previous stash's decay page,
// then gdn_spec_step's writer coefficient build (same float ops, same order) into 8 tiles + the gate page; all 9 tiles
// to `prefixed`.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tg_common.hpp"

static inline float f_from_bits(uint32_t b) {
    union { uint32_t u; float f; } x;
    x.u = b;
    return x.f;
}
static inline uint32_t bits_from_f(float f) {
    union { uint32_t u; float f; } x;
    x.f = f;
    return x.u;
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0), Nv = get_compile_time_arg_val(1);
    constexpr auto ring_args = TensorAccessorArgs<2>();
    constexpr auto pre_args = TensorAccessorArgs<ring_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 4096;
    uint32_t a = 0;
    const uint32_t ring_addr = get_arg_val<uint32_t>(a++), pre_addr = get_arg_val<uint32_t>(a++);
    const uint32_t h = get_arg_val<uint32_t>(a++), out0 = get_arg_val<uint32_t>(a++);
    const uint32_t NB = get_arg_val<uint32_t>(a++), batch = get_arg_val<uint32_t>(a++);
    const uint32_t a_u = a;  // per user: r_rep, previous decay page
    const auto ring = TensorAccessor(ring_args, ring_addr, TB);
    const auto pre = TensorAccessor(pre_args, pre_addr, TB);
    // tiles: NGB G E BETA NBE EYE W, then C_u (batch: 8 slots; else one), then the gate page GD
    const uint32_t nC = batch ? 8 : 1, nT = 7 + nC + 1, kGD = (7 + nC) * 1024;
    DeviceZoneScopedN("TG-W");
    cb_reserve_back(TG_COEF, nT);
    cb_reserve_back(TG_GP, 1);
    const uint32_t cbase = get_write_ptr(TG_COEF), gpa = get_write_ptr(TG_GP);
    {
        const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
        for (uint32_t o = 0; o < nT * TB; o += MEM_ZEROS_SIZE) noc_async_read(z, cbase + o, MEM_ZEROS_SIZE);
    }
    noc_async_read_barrier();
    cb_wait_front(TG_GT, 2);
    invalidate_l1_cache();
    volatile uint32_t* q32 = reinterpret_cast<volatile uint32_t*>(get_read_ptr(TG_GT));
    volatile uint32_t* c32 = reinterpret_cast<volatile uint32_t*>(cbase);
    constexpr uint32_t kNGB = 0, kG = 1024, kE = 2048, kBeta = 3072, kNBE = 4096, kEye = 5120, kW = 6144, kC = 7168;
    constexpr uint32_t kSign = 0x80000000u;
    for (uint32_t u = 0; u < NB; ++u) {
        const uint32_t r0 = u * T;
        const uint32_t r_rep = get_arg_val<uint32_t>(a_u + 2 * u), gp_page = get_arg_val<uint32_t>(a_u + 2 * u + 1);
        float wrep[T];
        float crep = 1.0f;
        if (r_rep > 0) {
            noc_async_read(ring.get_noc_addr(gp_page), gpa, TB);
            noc_async_read_barrier();
            volatile uint32_t* gp = reinterpret_cast<volatile uint32_t*>(gpa);
            float acc = 1.0f;
            for (uint32_t i = r_rep; i-- > 0;) {
                wrep[i] = acc;
                acc = acc * f_from_bits(gp[tg_elem(r0 + i, 0)]);
            }
            crep = acc;
        }
        uint32_t beta_all[T], dec_all[T];
        for (uint32_t t = 0; t < T; ++t) {
            dec_all[t] = q32[tg_elem(r0 + t, h & 31u)];
            beta_all[t] = q32[1024 + tg_elem(r0 + t, (Nv + h) & 31u)];
        }
        float beta[T], dec[T];
        for (uint32_t t = 0; t < T; ++t) {
            beta[t] = f_from_bits(beta_all[t]);
            dec[t] = f_from_bits(dec_all[t]);
        }
        float e_cum = 1.0f;
        for (uint32_t t = 0; t < T; ++t) {
            e_cum = e_cum * dec[t];
            const uint32_t row = r0 + t;
            float g = 1.0f;
            c32[kG + tg_elem(row, row)] = 0x3F800000u;
            for (uint32_t i = t; i-- > 0;) {
                g = g * dec[i + 1];
                c32[kG + tg_elem(row, r0 + i)] = bits_from_f(g);
                c32[kNGB + tg_elem(row, r0 + i)] = bits_from_f(beta[t] * g) ^ kSign;
            }
            c32[kEye + tg_elem(row, row)] = 0x3F800000u;
            const uint32_t eb = bits_from_f(e_cum);
            const uint32_t bb = beta_all[t];
            const uint32_t nbe = bits_from_f(beta[t] * e_cum) ^ kSign;
            const uint32_t wb = (t < r_rep) ? bits_from_f(wrep[t]) : 0u;
            for (uint32_t c = 0; c < 32; ++c) {
                const uint32_t idx = tg_elem(row, c);
                c32[kE + idx] = eb;
                c32[kBeta + idx] = bb;
                c32[kNBE + idx] = nbe;
                c32[kW + idx] = wb;
            }
            c32[kGD + tg_elem(row, 0)] = dec_all[t];
            c32[kGD + tg_elem(row, 1)] = beta_all[t];
        }
        const uint32_t cb = bits_from_f(crep);
        for (uint32_t i = 0; i < 1024; ++i) c32[kC + u * 1024 + i] = cb;
    }
    cb_pop_front(TG_GT, 2);
    for (uint32_t i = 0; i < nT; ++i) noc_async_write(cbase + i * TB, pre.get_noc_addr(out0 + i), TB);
    noc_async_write_barrier();
}
