// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// ---- Structural zone ids + ELF source-location records, for the STREAMING profiler ------------------
//
// SCOPE: this file describes the id that travels in `low27` of every streaming marker's word0
// (tools/profiler/spsc_packet.h) and the ELF records that give that id a name. It says NOTHING about the
// push-to-DRAM backend (tools/profiler/kernel_profiler_push.hpp + impl/profiler/profiler.cpp), which keeps
// its own 16-bit `timer_id`. Do not wire these constants into that path.
//
// ---- WHY STRUCTURAL, NOT A HASH --------------------------------------------------------------------
//
// The streaming wire used to carry a 16-bit FNV fold of "name,file,line,KERNEL_PROFILER"
// (kernel_profiler::Hash16_CT). Two problems: 16 bits over a few thousand zone sites collides by birthday
// (and a collision silently RENAMES a zone), and the host could only learn names by grepping
// `#pragma message` lines out of the JIT build log. Structural ids fix both -- an id is *constructed* from
// two independent, collision-free coordinates, and the (id -> name) mapping travels in the kernel's own ELF.
//
//   zone_id = tu_id << TT_ZONE_LOCAL_BITS | local
//
//     tu_id  13 bits  a per-translation-unit id handed out by a persistent, flock-guarded registry
//                     (tt_metal/jit_build/build.cpp) keyed on SOURCE IDENTITY, injected into the kernel
//                     compile as -DTT_PROFILER_TU_ID.  8,192 translation units.
//     local  14 bits  the zone's __COUNTER__ value in this TU. Raw, not rebased: the assembler record
//                     must be integer literals only (see TT_ZONE_DEFINE_ID_AT), so we cannot subtract a
//                     `.set` symbol.  16,384; non-zone __COUNTER__ uses in the same TU eat the budget.
//
// Both halves are unique by construction, so two distinct zone sites can never share an id. THE SPLIT IS
// ONE CONSTANT: change TT_ZONE_LOCAL_BITS and both halves resize together. Overflow of either half is a
// static_assert at the zone site, never a silent truncation.
//
// 27 bits costs nothing on this wire: a streaming marker is word0 = type(5)|low27, word1 = timer_low, with
// the clock's high half carried by a separate 1-word PP_STICKY_TIMER, so the id shares its word with
// nothing. (Contrast the DRAM path, where the id and timestamp_hi share a word and every id bit is a
// timestamp bit; that is exactly why this is a separate file.)
//
// ---- NO RESERVED BAND, NO FIXED IDS ----------------------------------------------------------------
//
// Every marker that reaches this wire -- ordinary zones, the producer's back-pressure PRODUCER-STALL zone,
// the DRISC drainer's own self-profiling zones, the NoC trace/debug event tags -- gets an ORDINARY
// structural id from this scheme and an ORDINARY record in .tt_zone_meta. There is no magic-value band,
// no `kind` field, and nothing on the host may discriminate on an id's VALUE. Consumers that need to treat
// a particular marker specially (a colour, say) look it up BY NAME, because the name is the only thing
// about a zone that is stable across builds -- a structural id legitimately moves when a source line does.
//
// ---- THE RECORD --------------------------------------------------------------------------------------
//
// Design credit: modelled on libmeta (github.com/strajabot/libmeta) -- typed fixed-size records built from
// nothing but assembler directives, with the strings in a separate mergeable section and only pointers in
// the record. We do NOT vendor libmeta, and deliberately do not copy its TOKEN mechanism: its identity is
// the address of a symbol in a non-ALLOC section, and GNU ld gives every such section sh_addr 0, so every
// token links to literally 0. Our identity is the structural id computed above, which needs no linker
// cooperation at all.
//
// Properties that matter:
//   * ZERO instructions in .text. The asm body is only directives, so a zone site costs what it always did.
//   * ZERO device memory. Neither section carries the assembler's "a" flag, so neither gets SHF_ALLOC and
//     neither can land in a PT_LOAD. The linker scripts place them at a far-out VMA with (INFO) for the
//     same reason; a script that does not mention them leaves them as non-ALLOC orphans at sh_addr 0,
//     which the host handles identically because it rebases by the section's own sh_addr.
//   * STRINGS ARE DEDUPED BY THE LINKER. .tt_zone_str is "MS" (SHF_MERGE|SHF_STRINGS, entsize 1), so the
//     __FILE__ string -- which otherwise repeats once per zone in the file -- is stored once per file.
//   * FIXED 16-BYTE STRIDE, declared as a real sh_entsize via the "M" flag with entsize 16. The host walks
//     a plain array; it never has to round a variable record length up to an alignment. Get that rounding
//     wrong by one and the walk does not fail, it DESYNCHRONISES and starts reading the tail of one record
//     as the head of the next, minting plausible ids bound to the wrong names. A fixed stride makes that
//     class of bug unrepresentable.
//   * THE ID IS CARRIED EXPLICITLY, not implied by the record's position in the array. Emission order is
//     the compiler's function-emission order, not source order, and a zone in code that is inlined away or
//     never instantiated leaves a HOLE that would otherwise shift every later index onto another zone's
//     name. The 4 bytes are free in a section that never loads.
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

// THE SPLIT. One line to change; everything below follows.
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
// compile -- a standalone/offline compile, or a unit test driving the compiler directly -- falls back to
// partition 0 and shares it.
#ifndef TT_PROFILER_TU_ID
#define TT_PROFILER_TU_ID 0
#endif

// Local half is the raw __COUNTER__ at the site. Do not subtract a GAS `.set` symbol: under `-flto=auto`
// lto-wrapper partitions that `.set` away from the `.tt_zone_meta` records (ABS vs UND) and the fused
// kernel link dies. C++ id and assembler record both stringify the same `ctr` token, so they cannot drift.
#define TT_ZONE_LOCAL_IDX(ctr) ((unsigned)(ctr))

// ---- The primitive ---------------------------------------------------------------------------------
//
// Declares `var` as this site's structural zone id and emits its .tt_zone_meta / .tt_zone_str record.
// Usable at NAMESPACE scope and at BLOCK scope (both halves are plain declarations plus a basic asm
// statement, which is legal in either). `ctr` is a parameter rather than a direct __COUNTER__ use because
// __COUNTER__ increments on every appearance: the C++ id and the record must see the same value.
//
// The 8880/8881 local labels are redefined by every expansion; `8880b` binds to the nearest preceding
// definition, which is the one in this same asm statement.
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
