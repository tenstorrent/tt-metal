// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Zone ids and the ELF records that name them, for the streaming device profiler. An id travels in low27 of a
// streaming marker's word0; the DRAM backend keeps its own 16-bit timer_id.
//
// An id is the ADDRESS of a one-byte handle the zone site emits into the non-ALLOC section .tt_zone_ids. The
// linker lays the handles out back to back from TT_ZONE_IDS_LINK_VMA (hw/toolchain/main.ld), so one image's
// ids are dense in link order. When the host loads the image (llrt/zone_meta.cpp, from ll_api::memory) it
// gives the image the next free block of the process-wide 16-bit id space and rebases the section there by
// re-resolving the relocations against it: the lui/addi pair each site materializes its id with, and the id
// word of each .tt_zone_meta record. Ids are thus unique across every image a process loads, with no per-TU
// partition, no registry file, and nothing computed on the device: an id costs the two instructions any
// 32-bit constant does. Nothing on the host may discriminate on an id's value: it depends on load order. The
// one exception is TT_ZONE_STALL_ID, the top of the space, which the allocator never hands out.
//
// Emission is assembler directives inside the one asm statement of the site's id(), after libmeta: zero
// .text beyond the two instructions and zero device memory, since none of .tt_zone_ids, .tt_zone_str,
// .tt_zone_meta is SHF_ALLOC. Directives rather than C++ objects with section attributes because a zone site
// inside a vague-linkage function (inline, template, class-template member) makes such objects COMDAT, which
// GCC will not put in a named section (without LTO a "section type conflict", with LTO an lto1 ICE). The
// handle's label is local to the object and guarded by .ifndef, so however many times an inlined site is
// expanded there is one handle and one record for it; TT_ZONE_TU_TAG (this TU's index in its link, from
// jit_build/build.cpp) keeps labels apart when LTO merges a link's TUs into one assembly. .tt_zone_str is
// "MS" so __FILE__ is stored once per file; .tt_zone_meta has a real sh_entsize of 16 so the host walks a
// plain array.
// Record layout (little-endian; must match ZoneMetaRecord in llrt/zone_meta.cpp):
//   [0] u32 zone_id (the handle's VMA)   [4] u32 name_ptr (VMA in .tt_zone_str)   [8] u32 file_ptr   [12] u32 line
#pragma once

#include <stdint.h>

// The id space: a process assigns [0, TT_ZONE_STALL_ID) to images in load order.
#define TT_ZONE_ID_BITS 16
#define TT_ZONE_ID_COUNT (1u << TT_ZONE_ID_BITS)
#define TT_ZONE_ID_MASK (TT_ZONE_ID_COUNT - 1u)
// The profiler's own stall zone: recognized by value, so it has no ELF record and no source location.
#define TT_ZONE_STALL_ID (TT_ZONE_ID_COUNT - 1u)

// Bytes per .tt_zone_meta record. Also the section's sh_entsize -- see the host walk in llrt/zone_meta.cpp.
#define TT_ZONE_META_RECORD_BYTES 16

// Where the linker script places .tt_zone_ids before the host rebases it (hw/toolchain/main.ld carries the
// same value). Any address with nonzero upper 20 bits serves: it keeps linker relaxation from folding a site's
// lui/addi into one instruction, which the host could not rebase.
#define TT_ZONE_IDS_LINK_VMA 0x6800000

#define TT_ZONE_STR_(x) #x
#define TT_ZONE_STR(x) TT_ZONE_STR_(x)

// This TU's index among the objects of its link, injected by the JIT build; a TU built outside it is alone.
#ifndef TT_ZONE_TU_TAG
#define TT_ZONE_TU_TAG 0
#endif

#define TT_ZONE_LABEL(ctr) "__tt_zone_" TT_ZONE_STR(TT_ZONE_TU_TAG) "_" TT_ZONE_STR(ctr)

// Declares `site` as this zone site's type; site::id() returns the site's id in two instructions with no
// memory access. Usable at namespace or block scope. `ctr` is a parameter because __COUNTER__ increments on
// every appearance and the label needs one value. The asm is not volatile: beyond its result it has no effect
// the compiler must order, so repeated uses of one site in a function may share a materialization.
#define TT_ZONE_DEFINE_ID_AT(site, name, ctr)                                                                   \
    struct site {                                                                                               \
        static inline __attribute__((always_inline)) uint32_t id() {                                            \
            uint32_t v;                                                                                         \
            asm(".ifndef " TT_ZONE_LABEL(ctr) "\n"                                                            \
                ".pushsection .tt_zone_ids,\"\",@progbits\n" TT_ZONE_LABEL(ctr) ":\t.byte 0\n"               \
                ".popsection\n"                                                                               \
                ".pushsection .tt_zone_str,\"MS\",@progbits,1\n"                                              \
                "8880:\t.asciz \"" name "\"\n"                                                                \
                "8881:\t.asciz \"" __FILE__ "\"\n"                                                            \
                ".popsection\n"                                                                               \
                ".pushsection .tt_zone_meta,\"M\",@progbits," TT_ZONE_STR(TT_ZONE_META_RECORD_BYTES) "\n"     \
                ".balign 4\n"                                                                                 \
                ".long " TT_ZONE_LABEL(ctr) "\n"                                                              \
                ".long 8880b\n"                                                                               \
                ".long 8881b\n"                                                                               \
                ".long " TT_ZONE_STR(__LINE__) "\n"                                                           \
                ".popsection\n"                                                                               \
                ".endif\n\t"                                                                                  \
                "lui %0, %%hi(" TT_ZONE_LABEL(ctr) ")\n\t"                                                    \
                "addi %0, %0, %%lo(" TT_ZONE_LABEL(ctr) ")"                                                   \
                : "=r"(v)); \
            return v;                                                                                           \
        }                                                                                                       \
    }

#define TT_ZONE_DEFINE_ID(site, name) TT_ZONE_DEFINE_ID_AT(site, name, __COUNTER__)
