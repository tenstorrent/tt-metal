// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Megakernel phase probe (data-movement RISCs): up to 16 wall-clock marks per (core, RISC), written at the end of the
// kernel as one 64-byte row of a DRAM uint32 tensor [rows, 16] (row = this RISC's slot). Compiled in only with
// MK_PROBE_ARG defined = the index of the first of the two probe runtime args (probe tensor address, slot); without it
// the macros vanish. Marks are the low 32 bits of the RISC wall clock (cycles of the AICLK domain); the host decodes
// the differences.
#pragma once

#include <cstdint>

#if defined(MK_PROBE_ARG)
#include "api/dataflow/dataflow_api.h"

#if !defined(MK_PROBE_LROWS)
#define MK_PROBE_LROWS 256  // probe rows per layer of the stack program (tt/mk_stack.py): layer l -> rows + l * this
#endif

namespace mkprobe {
inline uint32_t g_marks[16];
inline uint32_t g_n = 0;
inline uint32_t g_layer = 0;
// stack program: every layer's marks go to their own row block (the layer's flushes overwrite the same rows)
FORCE_INLINE void begin_layer(uint32_t layer) {
    g_layer = layer;
    g_n = 0;
}
FORCE_INLINE void mark() {
    if (g_n < 16) {
        g_marks[g_n++] = static_cast<uint32_t>(get_timestamp());
    }
}
// after the kernel's last NoC work: copy the marks into a 64-B aligned L1 scratch (the NoC cannot read RISC-local
// memory) and write it to row `slot` of the probe tensor
FORCE_INLINE void flush(uint32_t scratch_l1) {
    const uint32_t addr = get_arg_val<uint32_t>(MK_PROBE_ARG);
    const uint32_t slot = get_arg_val<uint32_t>(MK_PROBE_ARG + 1) + g_layer * MK_PROBE_LROWS;
    volatile tt_l1_ptr uint32_t* s = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_l1);
    for (uint32_t i = 0; i < 16; ++i) {
        s[i] = i < g_n ? g_marks[i] : 0u;
    }
    const InterleavedAddrGen<true> probe{.bank_base_address = addr, .page_size = 64};
    noc_async_write(scratch_l1, probe.get_noc_addr(slot), 64);
    noc_async_write_barrier();
}
}  // namespace mkprobe
#define MK_MARK() mkprobe::mark()
#define MK_FLUSH(scratch) mkprobe::flush(scratch)
#define MK_LAYER_BEGIN(layer) mkprobe::begin_layer(layer)
#else
#define MK_LAYER_BEGIN(layer) \
    do {                      \
        (void)(layer);        \
    } while (0)
#define MK_MARK() \
    do {          \
    } while (0)
#define MK_FLUSH(scratch) \
    do {                  \
    } while (0)
#endif
