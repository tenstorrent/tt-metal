// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Structural observability event ids and the ELF records that name them. An id travels in low27 of a streaming marker's
// word0; the DRAM backend keeps its own 16-bit timer_id.
//   event_id = tu_id << TT_OBSERVABILITY_EVENT_LOCAL_BITS | local
//   tu_id  13 bits  per-TU id from the flock-guarded registry in jit_build/build.cpp, -DTT_PROFILER_TU_ID
//   local  14 bits  the event's raw __COUNTER__ in this TU (other __COUNTER__ uses eat the same budget)
// Both halves are unique by construction and overflow is a static_assert where the event is declared. Nothing on the
// host may discriminate on a structural id's value: it moves when a source line does. Ids that must be the same
// constant on both sides live in the reserved tu partition below, which the registry never hands out. The record is
// emitted from assembler directives (after libmeta): zero .text and zero device memory, since neither
// .tt_observability_event_str nor .tt_observability_event_meta is SHF_ALLOC and the host rebases pointers by the
// section's own sh_addr; .tt_observability_event_str is "MS" so __FILE__ is stored once per file;
// .tt_observability_event_meta has a real sh_entsize of 16 so the host walks a plain array; the id is stored explicitly
// because an inlined-away event leaves a hole in emission order. Record layout (little-endian; must match
// ObservabilityEventMetaRecord in llrt/observability_event_meta.cpp):
//   [0] u32 event_id   [4] u32 name_ptr (VMA in .tt_observability_event_str)   [8] u32 file_ptr   [12] u32 line
#pragma once

#include <stdint.h>

// Total width of the id as it sits in low27 of a streaming marker word0.
#define TT_OBSERVABILITY_EVENT_ID_BITS 27

// The split: change this line and both halves resize together.
#define TT_OBSERVABILITY_EVENT_LOCAL_BITS 14
#define TT_OBSERVABILITY_EVENT_TU_BITS (TT_OBSERVABILITY_EVENT_ID_BITS - TT_OBSERVABILITY_EVENT_LOCAL_BITS)

#define TT_OBSERVABILITY_EVENT_LOCAL_COUNT (1u << TT_OBSERVABILITY_EVENT_LOCAL_BITS)
#define TT_OBSERVABILITY_EVENT_TU_COUNT (1u << TT_OBSERVABILITY_EVENT_TU_BITS)
#define TT_OBSERVABILITY_EVENT_ID_MASK ((1u << TT_OBSERVABILITY_EVENT_ID_BITS) - 1u)

#define TT_OBSERVABILITY_EVENT_MAKE_ID(tu, local) \
    ((((unsigned)(tu)) << TT_OBSERVABILITY_EVENT_LOCAL_BITS) | ((unsigned)(local)))
#define TT_OBSERVABILITY_EVENT_TU_OF(id) (((unsigned)(id)) >> TT_OBSERVABILITY_EVENT_LOCAL_BITS)
#define TT_OBSERVABILITY_EVENT_LOCAL_OF(id) (((unsigned)(id)) & (TT_OBSERVABILITY_EVENT_LOCAL_COUNT - 1u))
#define TT_OBSERVABILITY_EVENT_RESERVED_TU (TT_OBSERVABILITY_EVENT_TU_COUNT - 1u)
// The profiler's own stall zone: recognized by value, so it has no ELF record and no source location.
#define TT_ZONE_STALL_ID TT_OBSERVABILITY_EVENT_MAKE_ID(TT_OBSERVABILITY_EVENT_RESERVED_TU, 0)

// Bytes per .tt_observability_event_meta record. Also the section's sh_entsize -- see the host walk in
// llrt/observability_event_meta.cpp.
#define TT_OBSERVABILITY_EVENT_META_RECORD_BYTES 16

#define TT_OBSERVABILITY_EVENT_STR_(x) #x
#define TT_OBSERVABILITY_EVENT_STR(x) TT_OBSERVABILITY_EVENT_STR_(x)

// Injected by the JIT build; a TU the host did not compile falls back to partition 0.
#ifndef TT_PROFILER_TU_ID
#define TT_PROFILER_TU_ID 0
#endif

// Raw __COUNTER__, not rebased against a GAS `.set` symbol: under -flto=auto lto-wrapper partitions that
// symbol away from the records and the link dies.
#define TT_OBSERVABILITY_EVENT_LOCAL_IDX(ctr) ((unsigned)(ctr))

// Declares `var` as this observability event's id and emits its record; usable at namespace or block scope. `ctr` is a
// parameter because __COUNTER__ increments on every appearance and the id and the record must see one value.
#define TT_OBSERVABILITY_EVENT_DEFINE_ID_AT(var, name, ctr)                                                            \
    static_assert(                                                                                                     \
        TT_OBSERVABILITY_EVENT_LOCAL_IDX(ctr) < TT_OBSERVABILITY_EVENT_LOCAL_COUNT,                                    \
        "too many KERNEL_PROFILER observability events in one translation unit for TT_OBSERVABILITY_EVENT_LOCAL_BITS " \
        "-- widen the "                                                                                                \
        "split in hostdev/observability_event_id.h");                                                                  \
    static_assert(                                                                                                     \
        (unsigned)(TT_PROFILER_TU_ID) < TT_OBSERVABILITY_EVENT_RESERVED_TU,                                            \
        "TT_PROFILER_TU_ID is not below TT_OBSERVABILITY_EVENT_RESERVED_TU -- the tu-id registry handed out an id "    \
        "this "                                                                                                        \
        "split cannot express; see get_or_assign_profiler_tu_id in jit_build/build.cpp");                              \
    constexpr uint32_t var =                                                                                           \
        (uint32_t)TT_OBSERVABILITY_EVENT_MAKE_ID(TT_PROFILER_TU_ID, TT_OBSERVABILITY_EVENT_LOCAL_IDX(ctr));            \
    asm(".pushsection .tt_observability_event_str,\"MS\",@progbits,1\n"                                                      \
        "8880:\t.asciz \"" name "\"\n"                                                                        \
        "8881:\t.asciz \"" __FILE__ "\"\n"                                                                    \
        ".popsection\n"                                                                                       \
        ".pushsection .tt_observability_event_meta,\"M\",@progbits," TT_OBSERVABILITY_EVENT_STR(TT_OBSERVABILITY_EVENT_META_RECORD_BYTES) "\n"             \
        ".balign 4\n"                                                                                         \
        ".long ((" TT_OBSERVABILITY_EVENT_STR(TT_PROFILER_TU_ID) ") << " TT_OBSERVABILITY_EVENT_STR(TT_OBSERVABILITY_EVENT_LOCAL_BITS) ") | (" TT_OBSERVABILITY_EVENT_STR( \
            ctr) ")\n"                                                                                        \
        ".long 8880b\n"                                                                                       \
        ".long 8881b\n"                                                                                       \
        ".long " TT_OBSERVABILITY_EVENT_STR(__LINE__) "\n"                                                                   \
        ".popsection\n")

#define TT_OBSERVABILITY_EVENT_DEFINE_ID(var, name) TT_OBSERVABILITY_EVENT_DEFINE_ID_AT(var, name, __COUNTER__)
