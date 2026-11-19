// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// W7 step 5, D3 (P11x): the member data-movement parts of the x_ext store (xext_c.cpp untilizes).
//   XEXT_PUSH  (member DM1, after the phase's arrival wait): re-push the stationary XS block for the compute -- reserve
//              + push only: the AG landed it in place (pfs_member_rd.cpp's XSRC 0 protocol).
//   XEXT_WRITE (member DM0): per BW-tile chunk k, the 32 row segments (BW x 64 B each) to x_ext rows row0 .. row0 + 31
//              at column byte offset k x 64 (page = one row of H bf16), then a write barrier: x_ext is in DRAM when the
//              phase ends (the consumer's hand-off comes with the gang phase).
// CT at PH_CT0: [cb_xs, cb_rm, kt, bw] (+ TensorAccessorArgs(x_ext) for XEXT_WRITE). RT at ph_rt0: [run (0 = skip),
// x_ext address, row0] (XEXT_PUSH reads the first only).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace xed {
constexpr uint32_t CB_XS = get_compile_time_arg_val(PH_CT0 + 0);
constexpr uint32_t CB_RM = get_compile_time_arg_val(PH_CT0 + 1);
constexpr uint32_t KT = get_compile_time_arg_val(PH_CT0 + 2);
constexpr uint32_t BW = get_compile_time_arg_val(PH_CT0 + 3);
}  // namespace xed

void kernel_main() {
    using namespace xed;
    if (get_arg_val<uint32_t>(ph_rt0) == 0) {
        return;
    }
#if defined(XEXT_PUSH)
    cb_reserve_back(CB_XS, KT);
    cb_push_back(CB_XS, KT);
#elif defined(XEXT_WRITE)
    constexpr auto x_args = TensorAccessorArgs<PH_CT0 + 4>();
    const auto xe = TensorAccessor(x_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const uint32_t row0 = get_arg_val<uint32_t>(ph_rt0 + 2);
    constexpr uint32_t SEG = BW * 32 * 2;  // one row of a BW-tile chunk, bf16
    for (uint32_t k = 0; k < KT; k += BW) {
        cb_wait_front(CB_RM, BW);
        const uint32_t rp = get_read_ptr(CB_RM);
        for (uint32_t r = 0; r < 32; ++r) {
            noc_async_write(rp + r * SEG, xe.get_noc_addr(row0 + r, (k / BW) * SEG), SEG);
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_RM, BW);
    }
    noc_async_write_barrier();
#else
#error "xext_dm.cpp: define XEXT_PUSH or XEXT_WRITE"
#endif
}
