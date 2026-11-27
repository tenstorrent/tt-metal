// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Indexer phase member NCRISC (ixp_common.hpp, ixp_filter.hpp): the running top-512 of rows [0, NQA) over CB_SC.
// Compile-time: NQ, KC, CAP, NQA, then the selection's TensorAccessorArgs. Runtime: selection address, q0, nq, blocks,
// CS (the chunk's absolute start), pools [, IXP_CS_TENSOR: arg 4 = the start tensor's address, arg 6 = the row base's;
// their TensorAccessorArgs follow the selection's].
#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/ix/ixp_filter.hpp"

using namespace ixp;

void kernel_main() {
    constexpr uint32_t NQ = get_compile_time_arg_val(0);
    constexpr uint32_t KC = get_compile_time_arg_val(1);
    constexpr uint32_t CAP = get_compile_time_arg_val(2);
    constexpr uint32_t NQA = get_compile_time_arg_val(3);
    constexpr auto s_args = TensorAccessorArgs<4>();
    const uint32_t s_addr = get_arg_val<uint32_t>(0);
    const uint32_t q0 = get_arg_val<uint32_t>(1), nq = get_arg_val<uint32_t>(2);
    const uint32_t nblk = get_arg_val<uint32_t>(3);
    const uint32_t pools = get_arg_val<uint32_t>(5);
    const auto ss = TensorAccessor(s_args, s_addr);
    cb_reserve_back(CB_CAND, 1);  // scratch (never pushed)
#ifdef IXP_CS_TENSOR  // arg 4 = the start tensor's address, arg 6 = the row base tensor's (their accessors after
                      // s_args)
    constexpr auto c_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto b_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();
    const uint32_t cs = read_cs(
        TensorAccessor(c_args, get_arg_val<uint32_t>(4)),
        TensorAccessor(b_args, get_arg_val<uint32_t>(6)),
        (get_write_ptr(CB_CAND) + 63) & ~63u);  // a 64-B aligned slot inside the candidate scratch
#else
    const uint32_t cs = get_arg_val<uint32_t>(4);
#endif
    Filter<CAP> f;
    f.init(get_write_ptr(CB_CAND), 0, NQA, cs, q0, nq, pools);
    for (uint32_t b = 0; b < nblk; ++b) {
        cb_wait_front(CB_SC, 2 * KC);  // KC score tiles + KC row-max tiles (v8)
        f.template block<KC>(get_read_ptr(CB_SC), b);
        cb_pop_front(CB_SC, 2 * KC);
    }
    f.finish(ss, q0, nq);
}
