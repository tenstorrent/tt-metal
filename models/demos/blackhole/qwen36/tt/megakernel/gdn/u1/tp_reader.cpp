// GDN_STEP_U1 op-layout prefix reader. kind 0: window tile (win[par], user 0), raw qkv tile and tap tile of tensor
// column `col`; builds SH_j (zero tiles + the T live rows copied from W = [window rows mi+1..mi+K-1 | qkv rows 0..T-1])
// and the placement one-hot. kind 1: the z tile at qkv page `col`.
#include "api/dataflow/dataflow_api.h"
#include "api/debug/kernel_profiler.hpp"
#include "tp_common.hpp"

static inline void zero_l1(uint32_t addr, uint32_t bytes) {
    const uint64_t z = get_noc_addr(MEM_ZEROS_BASE);
    for (uint32_t o = 0; o < bytes; o += MEM_ZEROS_SIZE) noc_async_read(z, addr + o, MEM_ZEROS_SIZE);
}

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0), K = get_compile_time_arg_val(1);
    constexpr auto win_args = TensorAccessorArgs<2>();
    constexpr auto qkv_args = TensorAccessorArgs<win_args.next_compile_time_args_offset()>();
    constexpr auto tap_args = TensorAccessorArgs<qkv_args.next_compile_time_args_offset()>();
    constexpr uint32_t TB = 2048;
    uint32_t a = 0;
    const uint32_t win_addr = get_arg_val<uint32_t>(a++), qkv_addr = get_arg_val<uint32_t>(a++);
    const uint32_t tap_addr = get_arg_val<uint32_t>(a++), kind = get_arg_val<uint32_t>(a++);
    const uint32_t col = get_arg_val<uint32_t>(a++), mi = get_arg_val<uint32_t>(a++);
    const uint32_t winn_addr = get_arg_val<uint32_t>(a++), wwrite = get_arg_val<uint32_t>(a++);
    const auto winn = TensorAccessor(win_args, winn_addr, TB);  // win[1 - par] (same spec as win[par])
    const auto win = TensorAccessor(win_args, win_addr, TB);
    const auto qkv = TensorAccessor(qkv_args, qkv_addr, TB);
    const auto tap = TensorAccessor(tap_args, tap_addr, TB);
    DeviceZoneScopedN("TP-R");
    if (kind == 1) {
        cb_reserve_back(TP_Z, 1);
        noc_async_read(qkv.get_noc_addr(col), get_write_ptr(TP_Z), TB);
        noc_async_read_barrier();
        cb_push_back(TP_Z, 1);
        return;
    }
    // users: NB (runtime, default 1) with per-user window mi (args 8 .. 8+NB); user u's rows are u*T .. u*T+T-1
    const uint32_t nb_arg = get_arg_val<uint32_t>(a);
    const uint32_t NB = nb_arg ? nb_arg : 1;
    const uint32_t a_mi = a + 1;
    cb_reserve_back(TP_ST, 2);
    cb_reserve_back(TP_TAP, 1);
    cb_reserve_back(TP_SH, K);
    cb_reserve_back(TP_P, 1);
    const uint32_t st = get_write_ptr(TP_ST), sh = get_write_ptr(TP_SH), pp = get_write_ptr(TP_P);
    noc_async_read(qkv.get_noc_addr(col), st + TB, TB);
    noc_async_read(tap.get_noc_addr(col), get_write_ptr(TP_TAP), TB);
    zero_l1(sh, K * TB);
    zero_l1(pp, TB);
    noc_async_read_barrier();
    cb_push_back(TP_TAP, 1);
    {
        volatile uint16_t* p = reinterpret_cast<volatile uint16_t*>(pp);
        for (uint32_t r = 0; r < NB * T; ++r) p[tp_elem(r, r)] = 0x3F80;
    }
    cb_push_back(TP_P, 1);
    constexpr uint32_t Lw = K - 1 + T, Lw_w = (Lw + 1) & ~1u;
    static_assert(Lw_w <= 16, "rows in the top face pair");
    if (wwrite) cb_reserve_back(TP_W, 1);
    const uint32_t wt = wwrite ? get_write_ptr(TP_W) : 0;
    for (uint32_t u = 0; u < NB; ++u) {
        const uint32_t mi_u = NB > 1 ? get_arg_val<uint32_t>(a_mi + u) : mi;
        noc_async_read(win.get_noc_addr(u * (NB > 1 ? get_arg_val<uint32_t>(a_mi + NB) : 0) + col), st, TB);
        noc_async_read_barrier();
        // W_u = [window rows mi_u+1 .. mi_u+K-1 | qkv rows u*T .. u*T+T-1]; SH_j row u*T+t = W_u row t+j
        for (uint32_t j = 0; j < K; ++j) {
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t i = t + j;
                const uint32_t sa = i < K - 1 ? st : st + TB;
                const uint32_t srow = i < K - 1 ? mi_u + 1 + i : u * T + i - (K - 1);
                volatile uint32_t* s32 = reinterpret_cast<volatile uint32_t*>(sa);
                volatile uint32_t* d32 = reinterpret_cast<volatile uint32_t*>(sh + j * TB);
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    const uint32_t so = tp_elem(srow, 16 * hh) / 2, dof = tp_elem(u * T + t, 16 * hh) / 2;
                    for (uint32_t w = 0; w < 8; ++w) d32[dof + w] = s32[so + w];
                }
            }
        }
        const uint32_t hold_mask = NB > 1 ? get_arg_val<uint32_t>(a_mi + NB + 1) : 0;
        if (wwrite && ((hold_mask >> u) & 1u)) {
            // HOLD: the op copies the held user's window tile through unchanged (win[par] -> win[1 - par])
            noc_async_write(st, winn.get_noc_addr(u * get_arg_val<uint32_t>(a_mi + NB) + col), TB);
            noc_async_write_barrier();
        } else if (wwrite) {
            // the op's window write (writer wr_win, non-HOLD): win[1 - par] rows 0..Lw_w-1 of user u's tile
            volatile uint32_t* d32 = reinterpret_cast<volatile uint32_t*>(wt);
            for (uint32_t i = 0; i < Lw_w; ++i) {
                for (uint32_t hh = 0; hh < 2; ++hh) {
                    const uint32_t dof = tp_elem(i, 16 * hh) / 2;
                    if (i >= Lw) {
                        for (uint32_t w = 0; w < 8; ++w) d32[dof + w] = 0;
                        continue;
                    }
                    const uint32_t sa = i < K - 1 ? st : st + TB;
                    const uint32_t srow = i < K - 1 ? mi_u + 1 + i : u * T + i - (K - 1);
                    volatile uint32_t* s32 = reinterpret_cast<volatile uint32_t*>(sa);
                    const uint32_t so = tp_elem(srow, 16 * hh) / 2;
                    for (uint32_t w = 0; w < 8; ++w) d32[dof + w] = s32[so + w];
                }
            }
            const uint64_t dst = winn.get_noc_addr(u * (NB > 1 ? get_arg_val<uint32_t>(a_mi + NB) : 0) + col);
            noc_async_write(wt, dst, Lw_w * 32);
            noc_async_write(wt + 512, dst + 512, Lw_w * 32);
            noc_async_write_barrier();
        }
    }
    cb_push_back(TP_SH, K);
    cb_push_back(TP_ST, 2);
}
