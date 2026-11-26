// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream v3 (phase-ready), MEMBER BRISC: the output store. Each N-block's NB tiles (CB_OUT) -> the output
// tensor (TensorAccessor at CT_ACC; DRAM tile-interleaved [rows, N]) at the member's row tile: page row NT + n NB + t.
// The unit's y; in the PMK layer the H3 store of a projection phase, or the RS partial P (DRAM) with the H5 raise:
// after the whole row is stored, +1 at (SIGX, SIGY, SIGA) = the RS lane's LREADY + k / the reducer's OWNRDY + i.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "pfs_common.hpp"

namespace pfs_w {
constexpr uint32_t NB = get_compile_time_arg_val(PH_CT0 + pfs::CT_NB);
constexpr uint32_t NBLK = get_compile_time_arg_val(PH_CT0 + pfs::CT_NBLK);
constexpr uint32_t CB_OUT = get_compile_time_arg_val(PH_CT0 + pfs::CT_OUT);
constexpr uint32_t EPI = get_compile_time_arg_val(PH_CT0 + pfs::CT_EPI);
constexpr uint32_t RT = get_compile_time_arg_val(PH_CT0 + pfs::CT_RT);
constexpr uint32_t NTT = get_compile_time_arg_val(PH_CT0 + pfs::CT_NTT);  // the store's N tiles (page row stride)
}  // namespace pfs_w

void kernel_main() {
    using namespace pfs_w;
    constexpr auto y_args = TensorAccessorArgs<PH_CT0 + pfs::CT_ACC>();
    const uint32_t a = ph_rt0;
    const auto yy = TensorAccessor(y_args, get_arg_val<uint32_t>(a + pfs::RT_OUTADDR));
    const uint32_t row = get_arg_val<uint32_t>(a + pfs::RT_ROW);
    const uint32_t col0 = get_arg_val<uint32_t>(a + pfs::RT_COL0);  // the rectangle's first column tile
    const uint32_t probe = get_arg_val<uint32_t>(a + pfs::RT_PROBE);
    const uint32_t tb = get_tile_size(CB_OUT);
    uint64_t t_w = 0, tq;
    constexpr uint32_t OT = RT * NB;  // CB_OUT pages per N-block (DST order: row tile r, column c at r NB + c)
    if constexpr (EPI == 0) {
        for (uint32_t n = 0; n < NBLK; n++) {
            cb_wait_front(CB_OUT, OT);
            const uint32_t rp = get_read_ptr(CB_OUT);
            for (uint32_t r = 0; r < RT; r++) {
                for (uint32_t t = 0; t < NB; t++) {
                    noc_async_write_page((row + r) * NTT + col0 + n * NB + t, yy, rp + (r * NB + t) * tb);
                }
            }
            tq = get_timestamp();
            noc_async_writes_flushed();
            t_w += get_timestamp() - tq;
            cb_pop_front(CB_OUT, OT);
        }
    } else if constexpr (EPI == 2) {  // ROW_MAJOR y (page = one row of N): the N-block's 32 row segments of NB tiles
        constexpr uint32_t SEG = NB * 32 * 2;
        for (uint32_t n = 0; n < NBLK; n++) {
            cb_wait_front(CB_OUT, OT);
            const uint32_t rp = get_read_ptr(CB_OUT);
            for (uint32_t r = 0; r < RT * 32; r++) {
                noc_async_write(rp + r * SEG, yy.get_noc_addr(row * 32 + r, (col0 / NB + n) * SEG), SEG);
            }
            tq = get_timestamp();
            noc_async_writes_flushed();
            t_w += get_timestamp() - tq;
            cb_pop_front(CB_OUT, OT);
        }
    }  // EPI 1: the act stays in L1 as the next phase's in0 (nothing to store)
    tq = get_timestamp();
    noc_async_write_barrier();  // the whole output row landed before the hand-off raise
    t_w += get_timestamp() - tq;
    if (probe) {
        *pfs::word(probe, pfs::M_WSTALL) = static_cast<uint32_t>(t_w);
    }
    const uint32_t siga = get_arg_val<uint32_t>(a + pfs::RT_SIGA);
    if (siga) {
        noc_semaphore_inc(
            get_noc_addr(get_arg_val<uint32_t>(a + pfs::RT_SIGX), get_arg_val<uint32_t>(a + pfs::RT_SIGY), siga), 1);
        noc_async_atomic_barrier();
    }
}
