// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Consumer half of dfb_multi_touch_producer.cpp. Every bound DFB is constructed from its
// generated dfb::dfb_<n> accessor, so the interface it touches is the device slot the host
// assigned rather than the binding's position. Each one writes a validation record to L1:
//   results[i*3 + 0] = entry_size
//   results[i*3 + 1] = get_id()   // device slot baked into the accessor
//   results[i*3 + 2] = touched_magic | i
//
// Explicit sync: wait/pop handshake. Implicit sync (Quasar default): config probe only.
//
// TEST_NUM_DFBS is a host-provided define rather than a compile-time arg: it guards references to
// dfb::dfb_<n> names, which only exist in the generated bindings header for accessors this
// kernel actually binds.

#include "api/dataflow/dataflow_buffer.h"
#include "api/debug/dprint.h"
#include "dev_mem_map.h"
#include "experimental/kernel_args.h"

#ifndef TEST_NUM_DFBS
#error "TEST_NUM_DFBS must be defined by the host (KernelSpec compiler_options.defines)"
#endif

namespace {

// Quasar host readback must go through the uncached L1 window.
inline uint32_t l1_ptr_addr(uint32_t byte_addr) {
#ifdef ARCH_QUASAR
    return byte_addr + MEM_L1_UNCACHED_BASE;
#else
    return byte_addr;
#endif
}

}  // namespace

template <bool ImplicitSync>
static inline void touch_one(
    DFBBindingToken token, uint32_t index, volatile tt_l1_ptr uint32_t* results, uint32_t touched_magic) {
    DPRINT("touch[{}] enter implicit={}\n", index, ImplicitSync ? 1u : 0u);
    DPRINT("touch[{}] before DataflowBuffer ctor\n", index);
    DataflowBuffer dfb(token);
    DPRINT("touch[{}] after ctor id={} entry_size={}\n", index, (uint32_t)dfb.get_id(), dfb.get_entry_size());
    if constexpr (ImplicitSync) {
        DPRINT("touch[{}] probe path (skip wait/pop/finish)\n", index);
    } else {
        DPRINT("touch[{}] before wait_front\n", index);
        dfb.wait_front(1);
        DPRINT("touch[{}] before pop_front\n", index);
        dfb.pop_front(1);
        DPRINT("touch[{}] before finish\n", index);
        dfb.finish();
        DPRINT("touch[{}] after finish\n", index);
    }
    const uint32_t entry_size = dfb.get_entry_size();
    const uint32_t id = dfb.get_id();
    const uint32_t magic = touched_magic | index;
    results[index * 3 + 0] = entry_size;
    results[index * 3 + 1] = id;
    results[index * 3 + 2] = magic;
    DPRINT("touch[{}] wrote es={} id={} magic={:#x} @results+{}\n", index, entry_size, id, magic, index * 3);
}

void kernel_main() {
    constexpr bool implicit_sync = get_arg(args::implicit_sync);
    constexpr uint32_t touched_magic = get_arg(args::touched_magic);
    const uint32_t result_l1_addr = get_arg(args::result_l1_addr);
    volatile tt_l1_ptr uint32_t* results = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1_ptr_addr(result_l1_addr));

    DPRINT(
        "cons_main enter TEST_NUM_DFBS={} implicit={} magic={:#x} results_cached={:#x} results_uncached={:#x}\n",
        (uint32_t)TEST_NUM_DFBS,
        implicit_sync ? 1u : 0u,
        touched_magic,
        result_l1_addr,
        (uint32_t)reinterpret_cast<uintptr_t>(results));

#if TEST_NUM_DFBS > 0
    touch_one<implicit_sync>(dfb::dfb_0, 0, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 1
    touch_one<implicit_sync>(dfb::dfb_1, 1, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 2
    touch_one<implicit_sync>(dfb::dfb_2, 2, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 3
    touch_one<implicit_sync>(dfb::dfb_3, 3, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 4
    touch_one<implicit_sync>(dfb::dfb_4, 4, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 5
    touch_one<implicit_sync>(dfb::dfb_5, 5, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 6
    touch_one<implicit_sync>(dfb::dfb_6, 6, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 7
    touch_one<implicit_sync>(dfb::dfb_7, 7, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 8
    touch_one<implicit_sync>(dfb::dfb_8, 8, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 9
    touch_one<implicit_sync>(dfb::dfb_9, 9, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 10
    touch_one<implicit_sync>(dfb::dfb_10, 10, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 11
    touch_one<implicit_sync>(dfb::dfb_11, 11, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 12
    touch_one<implicit_sync>(dfb::dfb_12, 12, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 13
    touch_one<implicit_sync>(dfb::dfb_13, 13, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 14
    touch_one<implicit_sync>(dfb::dfb_14, 14, results, touched_magic);
#endif
#if TEST_NUM_DFBS > 15
    touch_one<implicit_sync>(dfb::dfb_15, 15, results, touched_magic);
#endif

    DPRINT("cons_main done\n");
}
