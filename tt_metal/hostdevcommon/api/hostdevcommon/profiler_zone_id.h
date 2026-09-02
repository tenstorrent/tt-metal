// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Structural zone ids, and the ELF records that give them names, for the streaming profiler.
//
// An id here travels in low27 of a streaming marker's word0 (tools/profiler/spsc_packet.h). The
// push-to-DRAM backend keeps its own 16-bit timer_id; do not wire these constants into that path.
//
//   zone_id = tu_id << TT_ZONE_LOCAL_BITS | local
//
//     tu_id  13 bits  per-translation-unit id from a persistent, flock-guarded registry keyed on source
//                     identity (tt_metal/jit_build/build.cpp), injected as -DTT_PROFILER_TU_ID.
//     local  14 bits  the zone's raw __COUNTER__ value in this TU; other __COUNTER__ uses in the same TU
//                     eat the same budget.
//
// Both halves are unique by construction, so two zone sites can never share an id, and overflow of
// either half is a static_assert at the zone site. Nothing on the host may discriminate on an id's
// value: a structural id legitimately moves when a source line does, so a consumer that treats some
// marker specially must look it up by name.
//
// The record is emitted from assembler directives only (modelled on libmeta,
// github.com/strajabot/libmeta):
//   * zero instructions in .text, and zero device memory: neither .tt_zone_str nor .tt_zone_meta gets
//     SHF_ALLOC, so neither can land in a PT_LOAD. A linker script that does not mention them leaves
//     them as non-ALLOC orphans at sh_addr 0, which the host handles identically because it rebases
//     pointers by the section's own sh_addr.
//   * .tt_zone_str is "MS", so the __FILE__ string is stored once per file, not once per zone.
//   * .tt_zone_meta has a fixed 16-byte stride declared as a real sh_entsize, so the host walks a plain
//     array and has no variable record length to round. Rounding it wrong desynchronises the walk
//     rather than failing it, binding plausible ids to the wrong names.
//   * the id is stored explicitly rather than implied by array position: emission order is the
//     compiler's function-emission order, and a zone that is inlined away leaves a hole.
//
// Record layout (little-endian; must match ZoneMetaRecord in llrt/zone_meta.cpp):
//   [0] u32 zone_id    the 27-bit structural id, exactly as it appears in low27 on the wire
//   [4] u32 name_ptr   VMA of the NUL-terminated zone name in .tt_zone_str
//   [8] u32 file_ptr   VMA of the NUL-terminated __FILE__ in .tt_zone_str
//  [12] u32 line       __LINE__
#pragma once

#include <stdint.h>

// Total width of the id as it sits in low27 of a streaming marker word0.
#define TT_ZONE_ID_BITS 27

// The split: change this line and both halves resize together.
#define TT_ZONE_LOCAL_BITS 14
#define TT_ZONE_TU_BITS (TT_ZONE_ID_BITS - TT_ZONE_LOCAL_BITS)

#define TT_ZONE_LOCAL_COUNT (1u << TT_ZONE_LOCAL_BITS)
#define TT_ZONE_TU_COUNT (1u << TT_ZONE_TU_BITS)
#define TT_ZONE_ID_MASK ((1u << TT_ZONE_ID_BITS) - 1u)

#define TT_ZONE_MAKE_ID(tu, local) ((((unsigned)(tu)) << TT_ZONE_LOCAL_BITS) | ((unsigned)(local)))
#define TT_ZONE_TU_OF(id) (((unsigned)(id)) >> TT_ZONE_LOCAL_BITS)
#define TT_ZONE_LOCAL_OF(id) (((unsigned)(id)) & (TT_ZONE_LOCAL_COUNT - 1u))

// Bytes per .tt_zone_meta record. Also the section's sh_entsize -- see the host walk in llrt/zone_meta.cpp.
#define TT_ZONE_META_RECORD_BYTES 16

#define TT_ZONE_STR_(x) #x
#define TT_ZONE_STR(x) TT_ZONE_STR_(x)

// Per-TU half of the id, injected by the JIT build (tt_metal/jit_build/build.cpp). A TU the host did not
// compile (an offline compile, or a test driving the compiler directly) falls back to partition 0.
#ifndef TT_PROFILER_TU_ID
#define TT_PROFILER_TU_ID 0
#endif

// Local half is the raw __COUNTER__ at the site. Do not rebase it against a GAS `.set` symbol: under
// `-flto=auto` lto-wrapper partitions that symbol away from the .tt_zone_meta records and the link dies.
#define TT_ZONE_LOCAL_IDX(ctr) ((unsigned)(ctr))

// Declares `var` as this site's structural zone id and emits its .tt_zone_meta / .tt_zone_str record.
// Usable at namespace or block scope. `ctr` is a parameter rather than a direct __COUNTER__ use because
// __COUNTER__ increments on every appearance, and the C++ id and the record must see the same value.
#define TT_ZONE_DEFINE_ID_AT(var, name, ctr)                                                                  \
    static_assert(                                                                                            \
        TT_ZONE_LOCAL_IDX(ctr) < TT_ZONE_LOCAL_COUNT,                                                         \
        "too many KERNEL_PROFILER zone sites in one translation unit for TT_ZONE_LOCAL_BITS -- widen the "    \
        "split in hostdevcommon/profiler_zone_id.h");                                                         \
    static_assert(                                                                                            \
        (unsigned)(TT_PROFILER_TU_ID) < TT_ZONE_TU_COUNT,                                                     \
        "TT_PROFILER_TU_ID exceeds TT_ZONE_TU_BITS -- the tu-id registry handed out an id this split cannot " \
        "express; see get_or_assign_profiler_tu_id in jit_build/build.cpp");                                  \
    constexpr uint32_t var = (uint32_t)TT_ZONE_MAKE_ID(TT_PROFILER_TU_ID, TT_ZONE_LOCAL_IDX(ctr));            \
    asm(".pushsection .tt_zone_str,\"MS\",@progbits,1\n"                                                      \
        "8880:\t.asciz \"" name "\"\n"                                                                        \
        "8881:\t.asciz \"" __FILE__ "\"\n"                                                                    \
        ".popsection\n"                                                                                       \
        ".pushsection .tt_zone_meta,\"M\",@progbits," TT_ZONE_STR(TT_ZONE_META_RECORD_BYTES) "\n"             \
        ".balign 4\n"                                                                                         \
        ".long ((" TT_ZONE_STR(TT_PROFILER_TU_ID) ") << " TT_ZONE_STR(TT_ZONE_LOCAL_BITS) ") | (" TT_ZONE_STR( \
            ctr) ")\n"                                                                                        \
        ".long 8880b\n"                                                                                       \
        ".long 8881b\n"                                                                                       \
        ".long " TT_ZONE_STR(__LINE__) "\n"                                                                   \
        ".popsection\n")

#define TT_ZONE_DEFINE_ID(var, name) TT_ZONE_DEFINE_ID_AT(var, name, __COUNTER__)
