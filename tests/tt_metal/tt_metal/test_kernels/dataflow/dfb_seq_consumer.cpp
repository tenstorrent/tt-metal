// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential cooperative DM consumer for multi-DFB TC-exhaustion tests (Gap 7).
//
// All N consumer threads cooperate on DFB_0, then DFB_1, and so on.  Each DFB
// has its own independent DRAM output buffer.  Both STRIDED and BLOCKED access
// patterns are supported on a per-DFB basis (controlled by the is_blocked
// CTA for each DFB).
//
// Per-DFB endpoints are bound by name (dfb::dfb_<i>, tensor::dst_<i>) and the kernel
// unrolls one block per declared DFB.  Per-DFB STRIDED/BLOCKED choice and per-DFB
// entries_per_consumer become CTAs (entries_per_consumer_<i>, is_blocked_<i>)
// since the values are known at host build time.
//
// Named args (CTAs):
//   args::implicit_sync
//   args::num_consumers
//   args::entries_per_consumer_0 .. _5
//   args::is_blocked_0 .. _5
// Compiler defines:
//   TEST_NUM_DFBS                  - matches the kernel's binding count (1..6).
//                                    Prefixed to avoid collision with
//                                    dfb::NUM_DFBS in dataflow_buffer_config.h.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"

// The implicit_sync=true branch uses NocOptions::TXN_ID, which is declared
// only under #ifdef ARCH_QUASAR in api/dataflow/noc.h. This kernel is only used
// by Quasar-only sequential-DFB harnesses, so the branch is unreachable on Gen1.
#ifdef ARCH_QUASAR
#define DFB_SEQ_CONSUME_IMPLICIT_SYNC(dfb_, tensor_accessor_, page_id_) \
    noc.async_write<NocOptions::TXN_ID>((dfb_), (tensor_accessor_), {}, {.page_id = (page_id_)})
#else
#define DFB_SEQ_CONSUME_IMPLICIT_SYNC(dfb_, tensor_accessor_, page_id_) ((void)0)
#endif

// Single per-DFB unrolled body. Variables visible from kernel_main:
//   implicit_sync, num_consumers, consumer_idx, noc.
#define DFB_SEQ_CONSUME(I)                                                                          \
    do {                                                                                            \
        constexpr uint32_t entries_per_consumer = get_arg(args::entries_per_consumer_##I);          \
        constexpr uint32_t is_blocked = get_arg(args::is_blocked_##I);                              \
        DataflowBuffer dfb(dfb::dfb_##I);                                                           \
        const uint32_t entry_size = dfb.get_entry_size();                                           \
        const auto tensor_accessor = TensorAccessor(tensor::dst_##I);                               \
        for (uint32_t tile_id = 0; tile_id < entries_per_consumer; tile_id++) {                     \
            const uint32_t page_id = is_blocked ? tile_id : tile_id * num_consumers + consumer_idx; \
            if constexpr (implicit_sync) {                                                          \
                DFB_SEQ_CONSUME_IMPLICIT_SYNC(dfb, tensor_accessor, page_id);                       \
            } else {                                                                                \
                dfb.wait_front(1);                                                                  \
                noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});        \
                noc.async_write_barrier();                                                          \
                dfb.pop_front(1);                                                                   \
            }                                                                                       \
        }                                                                                           \
        dfb.finish();                                                                               \
        dfb.write_barrier(noc);                                                                     \
    } while (0)

void kernel_main() {
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t num_consumers = get_arg(args::num_consumers);
    const uint32_t consumer_idx = get_my_thread_id();

    Noc noc;

#if TEST_NUM_DFBS >= 1
    DFB_SEQ_CONSUME(0);
#endif
#if TEST_NUM_DFBS >= 2
    DFB_SEQ_CONSUME(1);
#endif
#if TEST_NUM_DFBS >= 3
    DFB_SEQ_CONSUME(2);
#endif
#if TEST_NUM_DFBS >= 4
    DFB_SEQ_CONSUME(3);
#endif
#if TEST_NUM_DFBS >= 5
    DFB_SEQ_CONSUME(4);
#endif
#if TEST_NUM_DFBS >= 6
    DFB_SEQ_CONSUME(5);
#endif
}
