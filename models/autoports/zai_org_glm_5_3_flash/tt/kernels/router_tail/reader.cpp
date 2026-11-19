// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Fused router tail — reader (BRISC, NoC0): the token row's logits (10 x 2 face-row reads of 64 B from the router
// matmul's output, any interleaved / sharded layout the TensorAccessor addresses) packed into one tile + the packed
// bias. Compile-time args: TensorAccessorArgs of logits [1,1,B,N >= 320] fp32 and bias [32,32] fp32. Runtime:
// addresses, the row argument (router_tail_common.hpp: first row | (rows - 1) << 16).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/mk_common.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/probe.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/router_tail/router_tail_common.hpp"

using namespace rtail;

void kernel_main() {
    constexpr auto log_args = TensorAccessorArgs<PH_CT0 + 0>();
    constexpr auto bias_args = TensorAccessorArgs<log_args.next_compile_time_args_offset()>();
    const auto logits = TensorAccessor(log_args, get_arg_val<uint32_t>(ph_rt0 + 0));
    const auto bias = TensorAccessor(bias_args, get_arg_val<uint32_t>(ph_rt0 + 1));
    const uint32_t arg = get_arg_val<uint32_t>(ph_rt0 + 2);
    if (arg == ROW_SKIP) {
        return;  // megakernel: an owner without a token row
    }
    MK_MARK();  // tail probe 0: start
#if defined(MK_RTAIL_WAIT_SEM)
    mk::wait_handoff(MK_RTAIL_WAIT_SEM, get_arg_val<uint32_t>(ph_rt0 + 3));  // the router's logits landed
#endif
    MK_MARK();  // 1: logits landed
#if defined(RTAIL_PREFILL)
    noc_async_write_barrier();  // PMK P12: this RISC's PfStream writer stored the logits rows just before (program
                                // order)
#endif

    cb_reserve_back(CB_BIAS, 1);
    noc_async_read_page(0, bias, get_write_ptr(CB_BIAS));
    noc_async_read_barrier();
    cb_push_back(CB_BIAS, 1);
    for (uint32_t t = first_row(arg), end = first_row(arg) + row_count(arg); t < end; ++t) {
        // RTAIL_KSPLIT partials (one packed tile each; the compute sums them)
        cb_reserve_back(CB_LOG, RTAIL_KSPLIT);
        for (uint32_t c = 0; c < RTAIL_KSPLIT; ++c) {
            const uint32_t l1 = get_write_ptr(CB_LOG) + c * get_tile_size(CB_LOG);
            for (uint32_t r = 0; r < COL_TILES; ++r) {
                for (uint32_t h = 0; h < 2; ++h) {
                    noc_async_read(
                        logits.get_noc_addr(log_page(t, c * COL_TILES + r), tile_off(row_in_tile(t), 16 * h) * 4),
                        l1 + tile_off(r, 16 * h) * 4,
                        64);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(CB_LOG, RTAIL_KSPLIT);
    }
    MK_MARK();  // 2: the rows' logits in L1
#if defined(RTAIL_PROBE)
    MK_FLUSH(get_write_ptr(MK_PROBE_CB));
#endif
}
