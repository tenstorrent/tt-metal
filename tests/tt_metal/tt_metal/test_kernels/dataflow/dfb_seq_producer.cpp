// SPDX-FileCopyrightText: � 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Sequential cooperative DM producer for multi-DFB TC-exhaustion tests (Gap 7).
//
// All N producer threads cooperate on DFB_0 (each handling its own strided slice),
// then cooperate on DFB_1, and so on.  Every DFB has its own independent DRAM
// source buffer: each DFB binds via dfb::dfb_<i> + tensor::src_<i> (compile-time names);
// the kernel unrolls one block per declared DFB. TEST_NUM_DFBS compiler define
// gates how many tensor::src_<i> bindings the kernel references (must match the host's
// KernelSpec bindings count). The name is prefixed to avoid collision with
// dfb::NUM_DFBS from dataflow_buffer_config.h.
//
// After all producers call dfb.finish() for DFB_i, the barrier ensures they have
// all completed before any of them advances to DFB_i+1.  The consumer kernel uses
// the symmetric dfb_seq_consumer.cpp pattern so both sides stay in lock-step.
//
// Named args (CTAs):
//   args::num_entries_per_producer
//   args::implicit_sync
//   args::num_producers            - #DM producer threads in this kernel
// Compiler defines:
//   TEST_NUM_DFBS                  - matches the kernel's binding count

#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "api/kernel_thread_globals.h"

// Single per-DFB unrolled body.  All variables (num_entries_per_producer,
// implicit_sync, num_producers, producer_idx, noc) are visible from kernel_main.
//
// The implicit_sync=true branch uses NocOptions::TXN_ID, which is declared
// only under #ifdef ARCH_QUASAR in api/dataflow/noc.h. This kernel is only used
// by Quasar-only sequential-DFB harnesses, so the branch is unreachable on Gen1.
#ifdef ARCH_QUASAR
#define DFB_SEQ_PRODUCE_IMPLICIT_SYNC(tensor_accessor_, dfb_, page_id_) \
    noc.async_read<NocOptions::TXN_ID>((tensor_accessor_), (dfb_), {.page_id = (page_id_)}, {})
#else
#define DFB_SEQ_PRODUCE_IMPLICIT_SYNC(tensor_accessor_, dfb_, page_id_) ((void)0)
#endif

#define DFB_SEQ_PRODUCE(I)                                                                  \
    do {                                                                                    \
        DataflowBuffer dfb(dfb::dfb_##I);                                                   \
        const uint32_t entry_size = dfb.get_entry_size();                                   \
        const auto tensor_accessor = TensorAccessor(tensor::src_##I);                       \
        for (uint32_t tile_id = 0; tile_id < num_entries_per_producer; tile_id++) {         \
            const uint32_t page_id = tile_id * num_producers + producer_idx;                \
            if constexpr (implicit_sync) {                                                  \
                DFB_SEQ_PRODUCE_IMPLICIT_SYNC(tensor_accessor, dfb, page_id);               \
            } else {                                                                        \
                dfb.reserve_back(1);                                                        \
                noc.async_read(tensor_accessor, dfb, entry_size, {.page_id = page_id}, {}); \
                noc.async_read_barrier();                                                   \
                dfb.push_back(1);                                                           \
            }                                                                               \
        }                                                                                   \
        dfb.finish();                                                                       \
    } while (0)

void kernel_main() {
    constexpr uint32_t num_entries_per_producer = get_arg(args::num_entries_per_producer);
    constexpr uint32_t implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t num_producers = get_arg(args::num_producers);
    const uint32_t producer_idx = get_my_thread_id();

    Noc noc;

#if TEST_NUM_DFBS >= 1
    DFB_SEQ_PRODUCE(0);
#endif
#if TEST_NUM_DFBS >= 2
    DFB_SEQ_PRODUCE(1);
#endif
#if TEST_NUM_DFBS >= 3
    DFB_SEQ_PRODUCE(2);
#endif
#if TEST_NUM_DFBS >= 4
    DFB_SEQ_PRODUCE(3);
#endif
#if TEST_NUM_DFBS >= 5
    DFB_SEQ_PRODUCE(4);
#endif
#if TEST_NUM_DFBS >= 6
    DFB_SEQ_PRODUCE(5);
#endif
}
