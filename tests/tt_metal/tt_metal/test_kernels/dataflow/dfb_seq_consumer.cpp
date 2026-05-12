// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential cooperative DM consumer for multi-DFB TC-exhaustion tests (Gap 7).
//
// All N consumer threads cooperate on DFB_0, then DFB_1, and so on.  Each DFB
// has its own independent DRAM output buffer.  Both STRIDED and BLOCKED access
// patterns are supported on a per-DFB basis (controlled by the is_blocked
// runtime arg / CTA for each DFB).
//
// On QUASAR, per-DFB endpoints are bound by name (dfb::dfb_<i>, ta::dst_<i>) and
// the kernel unrolls one block per declared DFB.  Per-DFB STRIDED/BLOCKED choice
// and per-DFB entries_per_consumer become CTAs (entries_per_consumer_<i>,
// is_blocked_<i>) since the values are known at host build time.  See the plan
// section "Known limitation: dfb_seq_consumer.cpp per-DFB destination tensors"
// for context.
//
// Compile-time args (legacy):
//   [0]: implicit_sync             - 0 or 1
//   [1..]: TensorAccessorArgs      - shared DRAM layout (same page_size for all bufs)
//
// Runtime args (legacy, per-core):
//   [0]: consumer_mask
//   [1]: num_dfbs
//   [2 .. 2+M-1]:     dst_addr[i]
//   [2+M .. 2+2M-1]:  entries_per_consumer[i]
//   [2+2M .. 2+3M-1]: is_blocked[i]
//
// QUASAR named args (CTAs):
//   args::implicit_sync
//   args::num_consumers
//   args::entries_per_consumer_0 .. _5
//   args::is_blocked_0 .. _5
// QUASAR compiler defines:
//   TEST_NUM_DFBS                  - matches the kernel's binding count (1..6).
//                                    Prefixed to avoid collision with
//                                    dfb::NUM_DFBS in dataflow_buffer_config.h.

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#ifdef ARCH_QUASAR
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"
#endif

#ifdef ARCH_QUASAR
// Single per-DFB unrolled body. Variables visible from kernel_main:
//   implicit_sync, num_consumers, consumer_idx, noc.
#define DFB_SEQ_CONSUME(I)                                                                          \
    do {                                                                                            \
        constexpr uint32_t entries_per_consumer = get_arg(args::entries_per_consumer_##I);          \
        constexpr uint32_t is_blocked = get_arg(args::is_blocked_##I);                              \
        DataflowBuffer dfb(dfb::dfb_##I);                                                           \
        const uint32_t entry_size = dfb.get_entry_size();                                           \
        const auto tensor_accessor = TensorAccessor(ta::dst_##I);                                   \
        for (uint32_t tile_id = 0; tile_id < entries_per_consumer; tile_id++) {                     \
            const uint32_t page_id = is_blocked ? tile_id : tile_id * num_consumers + consumer_idx; \
            if constexpr (implicit_sync) {                                                          \
                noc.async_write<Noc::TxnIdMode::ENABLED>(                                           \
                    dfb, tensor_accessor, {}, {.page_id = page_id});                                \
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
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
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
#else
    constexpr uint32_t implicit_sync = get_compile_time_arg_val(0);
    constexpr auto dst_args          = TensorAccessorArgs<1>();

    const uint32_t consumer_mask = get_arg_val<uint32_t>(0);
    const uint32_t num_dfbs      = get_arg_val<uint32_t>(1);

    const uint32_t consumer_idx = 0;
    const uint32_t num_consumers = static_cast<uint32_t>(__builtin_popcount(consumer_mask));

    Noc noc;

    for (uint32_t dfb_id = 0; dfb_id < num_dfbs; dfb_id++) {
        const uint32_t dst_addr = get_arg_val<uint32_t>(2 + dfb_id);
        const uint32_t entries_per_consumer = get_arg_val<uint32_t>(2 + num_dfbs + dfb_id);
        const uint32_t is_blocked = get_arg_val<uint32_t>(2 + 2 * num_dfbs + dfb_id);

        DataflowBuffer dfb(dfb_id);
        const uint32_t entry_size  = dfb.get_entry_size();
        const auto tensor_accessor = TensorAccessor(dst_args, dst_addr, entry_size);

        for (uint32_t tile_id = 0; tile_id < entries_per_consumer; tile_id++) {
            // BLOCKED: each consumer reads all entries sequentially (broadcast).
            // STRIDED: interleaved, consumer i reads pages i, i+C, i+2C, ...
            const uint32_t page_id = is_blocked
                ? tile_id
                : tile_id * num_consumers + consumer_idx;

            if constexpr (implicit_sync == 0) {
                dfb.wait_front(1);
                noc.async_write(dfb, tensor_accessor, entry_size, {}, {.page_id = page_id});
                noc.async_write_barrier();
                dfb.pop_front(1);
            }
        }
        // All consumers of this DFB must finish before moving to the next DFB.
        dfb.finish();
        dfb.write_barrier(noc);
    }
#endif
}
