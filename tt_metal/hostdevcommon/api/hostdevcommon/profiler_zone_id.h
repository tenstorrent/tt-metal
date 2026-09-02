// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Structural zone ids and the ELF records that name them. An id travels in low27 of a streaming marker's
// word0; the DRAM backend keeps its own 16-bit timer_id.
//   zone_id = tu_id << TT_ZONE_LOCAL_BITS | local
//   tu_id  13 bits  per-TU id from the flock-guarded registry in jit_build/build.cpp, -DTT_PROFILER_TU_ID
//   local  14 bits  the zone's raw __COUNTER__ in this TU (other __COUNTER__ uses eat the same budget)
// Both halves are unique by construction and overflow is a static_assert at the site. Nothing on the host may
// discriminate on an id's value: it moves when a source line does, so special markers are matched by name.
// The record is emitted from assembler directives (after libmeta): zero .text and zero device memory, since
// neither .tt_zone_str nor .tt_zone_meta is SHF_ALLOC and the host rebases pointers by the section's own
// sh_addr; .tt_zone_str is "MS" so __FILE__ is stored once per file; .tt_zone_meta has a real sh_entsize of
// 16 so the host walks a plain array; the id is stored explicitly because an inlined-away zone leaves a hole
// in emission order.
// Record layout (little-endian; must match ZoneMetaRecord in llrt/zone_meta.cpp):
//   [0] u32 zone_id   [4] u32 name_ptr (VMA in .tt_zone_str)   [8] u32 file_ptr   [12] u32 line
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

// Injected by the JIT build; a TU the host did not compile falls back to partition 0.
#ifndef TT_PROFILER_TU_ID
#define TT_PROFILER_TU_ID 0
#endif

// Raw __COUNTER__, not rebased against a GAS `.set` symbol: under -flto=auto lto-wrapper partitions that
// symbol away from the records and the link dies.
#define TT_ZONE_LOCAL_IDX(ctr) ((unsigned)(ctr))

// Declares `var` as this site's zone id and emits its record; usable at namespace or block scope. `ctr` is a
// parameter because __COUNTER__ increments on every appearance and the id and the record must see one value.
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
