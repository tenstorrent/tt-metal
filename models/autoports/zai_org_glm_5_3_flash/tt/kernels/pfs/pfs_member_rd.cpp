// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream v3 (phase-ready), MEMBER NCRISC (tt/pf_stream.PfStreamMM). The stationary row tile in CB_X: the
// unit (PFS_UNIT) fills it from DRAM; in the PMK layer it is already in XS / IN0 (H2), so the body only hands the KT
// pages to the compute (reserve + push, no data movement). Then the PARK credit loop over the phase's blocks q in order
// (launch-global Q = QBASE + q): credits run AHEAD of the data — block c is credited (+1 on READY[i mod 2] of its
// streamer: stream c / QPS, lane c mod 2, i = c's index in that streamer's list) as soon as PARK slots v .. c are free,
// up to P past the next block to land (v); VALID(v) (the lane's semaphore >= Q / 2 + 1) is polled, not waited on.
// Member 0 first publishes the PARK base and the two VALID addresses to every streamer. CT / RT maps: pfs_common.hpp.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfs_common.hpp"

namespace pfs_m {
constexpr uint32_t NBK = get_compile_time_arg_val(PH_CT0 + pfs::CT_NBK);
constexpr uint32_t KB = get_compile_time_arg_val(PH_CT0 + pfs::CT_KB);
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + pfs::CT_NB);
constexpr uint32_t NW = get_compile_time_arg_val(PH_CT0 + pfs::CT_NW);
constexpr uint32_t P = get_compile_time_arg_val(PH_CT0 + pfs::CT_P);
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + pfs::CT_KT);
constexpr uint32_t CB_X = get_compile_time_arg_val(PH_CT0 + pfs::CT_X);
constexpr uint32_t CB_PARK = get_compile_time_arg_val(PH_CT0 + pfs::CT_PARK);
constexpr uint32_t SEM_V0 = get_compile_time_arg_val(PH_CT0 + pfs::CT_V0);
constexpr uint32_t SEM_V1 = get_compile_time_arg_val(PH_CT0 + pfs::CT_V1);
constexpr uint32_t NBLK = get_compile_time_arg_val(PH_CT0 + pfs::CT_NBLK);
constexpr uint32_t QBASE = get_compile_time_arg_val(PH_CT0 + pfs::CT_QBASE);
constexpr uint32_t SLOT = get_compile_time_arg_val(PH_CT0 + pfs::CT_SLOT);
constexpr uint32_t WAITSEM = get_compile_time_arg_val(PH_CT0 + pfs::CT_WAITSEM);
constexpr uint32_t WAITN = get_compile_time_arg_val(PH_CT0 + pfs::CT_WAITN);
constexpr uint32_t XSRC = get_compile_time_arg_val(PH_CT0 + pfs::CT_XSRC);
constexpr uint32_t SPC = get_compile_time_arg_val(PH_CT0 + pfs::CT_SPC);
constexpr uint32_t RT = get_compile_time_arg_val(PH_CT0 + pfs::CT_RT);
constexpr uint32_t BT = KB * NB;
constexpr uint32_t QPS = NW / NB * NBK;  // blocks per stream
constexpr uint32_t XT = RT * KT;         // the member's stationary tiles (RT row tiles)
static_assert(QBASE % 2 == 0, "lane parity continues across phases");
}  // namespace pfs_m

void kernel_main() {
    using namespace pfs_m;
    const uint32_t a = ph_rt0;
    const uint32_t sync = get_arg_val<uint32_t>(a + pfs::RT_SYNC) + SLOT * pfs::SYNC_WORDS * pfs::WORD;
    const uint32_t first = get_arg_val<uint32_t>(a + pfs::RT_FIRST);
    const uint32_t ns = get_arg_val<uint32_t>(a + pfs::RT_NS);
    const uint32_t probe = get_arg_val<uint32_t>(a + pfs::RT_PROBE);
    const uint32_t sxy = a + pfs::RT_SXY;
    uint64_t t_xf = get_timestamp();
    if constexpr (XSRC == 0) {  // XSRC 1: the previous phase's compute pushed this phase's in0 (SwiGLU act)
        cb_reserve_back(CB_X, XT);
#if defined(PFS_UNIT)
        {
            constexpr auto y_args = TensorAccessorArgs<PH_CT0 + pfs::CT_ACC>();
            constexpr auto x_args = TensorAccessorArgs<y_args.next_compile_time_args_offset()>();
            const uint32_t xaddr = get_arg_val<uint32_t>(a + pfs::RT_XADDR);
#if defined(PFS_WAIT_RT)
            // PER-MEMBER x readiness (Stage 14, o_proj row groups under the KDA scan): wait WAITSEM >= the count after
            // the streamer list (e.g. the scan leader's SEM_ON = the chunks this member's rectangle reads)
            {
                const uint32_t wn = get_arg_val<uint32_t>(a + pfs::RT_SXY + 2 * ns);
                if (wn) {
                    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(WAITSEM)), wn);
                }
            }
#endif
            if (xaddr) {  // 0: this phase reuses the stationary block an earlier phase of the launch left in CB_X
                const auto xx = TensorAccessor(x_args, xaddr);
                const uint32_t row = get_arg_val<uint32_t>(a + pfs::RT_ROW);
                const uint32_t xtb = get_tile_size(CB_X), xw = get_write_ptr(CB_X);
                for (uint32_t t = 0; t < XT; t++) {  // row tiles row .. row + RT - 1, K-major pages
                    noc_async_read_page(row * KT + t, xx, xw + t * xtb);
                }
                noc_async_read_barrier();
            }
        }
#else
        // the stationary block lands from elsewhere (H2: the AG's local slices + the lanes' remote slices raise this
        // core's FS_ARRIVE, cumulative over the launch's AG instances); in the PMK composite the generated main loop
        // waits instead
        if constexpr (WAITN > 0) {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(WAITSEM)), WAITN);
        }
#endif
        cb_push_back(CB_X, XT);
    }
    t_xf = get_timestamp() - t_xf;
    const uint32_t v0 = get_semaphore(SEM_V0), v1 = get_semaphore(SEM_V1);
    if (first) {
        const uint32_t park = get_write_ptr(CB_PARK);
        for (uint32_t s = 0; s < ns; s++) {
            const uint32_t x = get_arg_val<uint32_t>(sxy + 2 * s), y = get_arg_val<uint32_t>(sxy + 2 * s + 1);
            noc_inline_dw_write(get_noc_addr(x, y, sync + pfs::PARKADDR * pfs::WORD), park);
            noc_inline_dw_write(get_noc_addr(x, y, sync + pfs::VALIDADDR0 * pfs::WORD), v0);
            noc_inline_dw_write(get_noc_addr(x, y, sync + pfs::VALIDADDR1 * pfs::WORD), v1);
        }
        noc_async_write_barrier();  // the addresses landed before this member's first READY credit
    }
    volatile tt_l1_ptr uint32_t* vs[2] = {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v0), reinterpret_cast<volatile tt_l1_ptr uint32_t*>(v1)};
    const uint32_t nq = NBLK * NBK;
    uint64_t t_res = 0, t_val = 0, t0 = get_timestamp(), tq = t0;
    uint32_t v = 0, c = 0;
    while (v < nq) {
        invalidate_l1_cache();  // the compute's pops (tiles_acked) and the streamers' VALID sets
        if (c < nq && c - v < P && cb_pages_reservable_at_back(CB_PARK, (c - v + 1) * BT)) {
            // block c: stream c / QPS of the rectangle, on streamer core 2 (stream / SPC) + lane; i = c's index in that
            // core's list (its SPC streams in order, QPS / 2 blocks per lane each)
            const uint32_t sq = c / QPS, r = c % QPS;
            const uint32_t s = 2 * (sq / SPC) + (c & 1);
            const uint32_t i = (sq % SPC) * (QPS / 2) + (r / NBK) * (NBK / 2) + (r % NBK) / 2;
            noc_semaphore_inc(
                get_noc_addr(
                    get_arg_val<uint32_t>(sxy + 2 * s),
                    get_arg_val<uint32_t>(sxy + 2 * s + 1),
                    sync + ((i & 1) ? pfs::READY1 : pfs::READY) * pfs::WORD),
                1);
            c++;
            continue;
        }
        if (c > v && *vs[v & 1] >= (QBASE + v) / 2 + 1) {
            cb_push_back(CB_PARK, BT);
            v++;
            const uint64_t now = get_timestamp();
            t_val += now - tq;  // landing to landing (stream-bound if it dominates the span)
            tq = now;
            continue;
        }
        if (c == v) {  // nothing credited ahead: the compute holds every slot
            const uint64_t w = get_timestamp();
            cb_reserve_back(CB_PARK, BT);
            t_res += get_timestamp() - w;
        }
    }
    noc_async_atomic_barrier();
    if (probe) {
        *pfs::word(probe, pfs::M_RESERVE) = static_cast<uint32_t>(t_res);
        *pfs::word(probe, pfs::M_VALID) = static_cast<uint32_t>(t_val);
        *pfs::word(probe, pfs::M_SPAN) = static_cast<uint32_t>(get_timestamp() - t0);
        *pfs::word(probe, pfs::M_XFILL) = static_cast<uint32_t>(t_xf);
    }
}
