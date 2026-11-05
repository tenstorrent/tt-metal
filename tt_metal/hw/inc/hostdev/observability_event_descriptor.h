// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Observability event descriptors and their ids. An event's descriptor is where the event is declared in the source,
// together with the metadata attached to it; for now, the metadata is just the event's name. It lives only in the ELF,
// in .tt_event_descriptor_meta and .tt_event_descriptor_str, and never reaches the device. The descriptor id is the
// number the device sends in the descriptor's place; the host uses it to look the descriptor up.
//
// An id is the ADDRESS of a one-byte handle the event emits into the non-ALLOC section .tt_event_descriptor_ids.
// The linker lays the handles out back to back from TT_EVENT_DESCRIPTOR_IDS_LINK_VMA (hw/toolchain/main.ld), so one
// image's ids are dense in link order. When the host loads the image (llrt/observability_event_meta.cpp, from
// ll_api::memory) it gives the image the next free block of the process-wide 16-bit id space and rebases the section
// there by re-resolving the relocations against it: the lui/addi pair each event materializes its id with, and the id
// word of each .tt_event_descriptor_meta record. Ids are thus unique across every image a process loads, with no
// per-TU partition, no registry file, and nothing computed on the device: an id costs the two instructions any 32-bit
// constant does, and so does a word with constant bits packed above the id (descriptor_id_plus below), since those
// bits ride in the relocation's addend. Nothing on the host may discriminate on an id's value: it depends on load
// order. The one exception is TT_ZONE_STALL_ID, the top of the space, which the allocator never hands out.
#pragma once

#include <stdint.h>

// The id space: a process assigns [0, TT_ZONE_STALL_ID) to images in load order.
#define TT_EVENT_DESCRIPTOR_ID_BITS 16
#define TT_EVENT_DESCRIPTOR_ID_COUNT (1u << TT_EVENT_DESCRIPTOR_ID_BITS)
#define TT_EVENT_DESCRIPTOR_ID_MASK (TT_EVENT_DESCRIPTOR_ID_COUNT - 1u)
// The profiler's own stall zone: recognized by value, so it has no ELF record and no source location.
#define TT_ZONE_STALL_ID (TT_EVENT_DESCRIPTOR_ID_COUNT - 1u)

// Bytes per .tt_event_descriptor_meta record. Also the section's sh_entsize -- see the host walk in
// llrt/observability_event_meta.cpp.
#define TT_EVENT_DESCRIPTOR_META_RECORD_BYTES 16

// Where the linker script places .tt_event_descriptor_ids before the host rebases it (hw/toolchain/main.ld carries
// the same value). Any address with nonzero upper 20 bits serves: it keeps linker relaxation from folding an event's
// lui/addi into one instruction, which the host could not rebase.
#define TT_EVENT_DESCRIPTOR_IDS_LINK_VMA 0x6800000

#define TT_OBSERVABILITY_EVENT_STR_(x) #x
#define TT_OBSERVABILITY_EVENT_STR(x) TT_OBSERVABILITY_EVENT_STR_(x)

// This TU's index among the objects of its link, injected by the JIT build; a TU built outside it is alone.
#ifndef TT_OBSERVABILITY_EVENT_TU_ID
#define TT_OBSERVABILITY_EVENT_TU_ID 0
#endif

#define TT_OBSERVABILITY_EVENT_LABEL(ctr)                   \
    "__tt_observability_event_" TT_OBSERVABILITY_EVENT_STR( \
        TT_OBSERVABILITY_EVENT_TU_ID) "_" TT_OBSERVABILITY_EVENT_STR(ctr)

// The directives that define an event's handle and its record: guarded by .ifndef, so every materialization of the
// event's id can carry them and the first one to be assembled defines them.
#define TT_EVENT_DESCRIPTOR_DEFINE_ASM(name, ctr) \
    ".ifndef " TT_OBSERVABILITY_EVENT_LABEL(ctr) "\n"                                                             \
    ".pushsection .tt_event_descriptor_ids,\"\",@progbits\n" TT_OBSERVABILITY_EVENT_LABEL(ctr) ":\t.byte 0\n"     \
    ".popsection\n"                                                                                                \
    ".pushsection .tt_event_descriptor_str,\"MS\",@progbits,1\n"                                                   \
    "8880:\t.asciz \"" name "\"\n"                                                                                 \
    "8881:\t.asciz \"" __FILE__ "\"\n"                                                                             \
    ".popsection\n"                                                                                                \
    ".pushsection .tt_event_descriptor_meta,\"M\",@progbits," TT_OBSERVABILITY_EVENT_STR(                          \
        TT_EVENT_DESCRIPTOR_META_RECORD_BYTES) "\n"                                                                \
    ".balign 4\n"                                                                                                  \
    ".long " TT_OBSERVABILITY_EVENT_LABEL(ctr) "\n"                                                               \
    ".long 8880b\n"                                                                                                \
    ".long 8881b\n"                                                                                                \
    ".long " TT_OBSERVABILITY_EVENT_STR(__LINE__) "\n"                                                            \
    ".popsection\n"                                                                                                \
    ".endif\n\t"

// Declares `event` as this event's type; event::descriptor_id() returns the event's descriptor id in two instructions
// with no memory access. Usable at namespace or block scope. `ctr` is a parameter because __COUNTER__ increments on
// every appearance and the label needs one value. The asm is not volatile: beyond its result it has no effect the
// compiler must order, so repeated uses of one event in a function may share a materialization.
// event::descriptor_id_plus() returns descriptor_id() + `plus` in the same two instructions: `plus` rides in the
// relocation's addend, which the host keeps when it rebases the id, so a caller packing constant bits above the id
// (a packet type) gets the finished word for free. `plus` is a constant expression that leaves the id's bits clear.
#define TT_EVENT_DESCRIPTOR_DEFINE_ID_PLUS_AT(event, name, ctr, plus)                                                    \
    struct event {                                                                                                       \
        static inline __attribute__((always_inline)) uint32_t descriptor_id() {                                          \
            uint32_t v;                                                                                                  \
            asm(TT_EVENT_DESCRIPTOR_DEFINE_ASM(name, ctr) "lui %0, %%hi(" TT_OBSERVABILITY_EVENT_LABEL(                \
                    ctr) ")\n\t"                                                                                      \
                         "addi %0, %0, %%lo(" TT_OBSERVABILITY_EVENT_LABEL(ctr) ")"                                  \
                : "=r"(v)); \
            return v;                                                                                                    \
        }                                                                                                                \
        static inline __attribute__((always_inline)) uint32_t descriptor_id_plus() {                                     \
            static_assert(                                                                                               \
                ((plus) & TT_EVENT_DESCRIPTOR_ID_MASK) == 0, "plus must leave the descriptor id's bits clear");          \
            uint32_t v;                                                                                                  \
            asm(TT_EVENT_DESCRIPTOR_DEFINE_ASM(name, ctr) "lui %0, %%hi(" TT_OBSERVABILITY_EVENT_LABEL(                \
                    ctr) " + %1)\n\t"                                                                                 \
                         "addi %0, %0, %%lo(" TT_OBSERVABILITY_EVENT_LABEL(ctr) " + %1)"                             \
                : "=r"(v)                                                                                             \
                : "i"(plus)); \
            return v;                                                                                                    \
        }                                                                                                                \
    }
#define TT_EVENT_DESCRIPTOR_DEFINE_ID_AT(event, name, ctr) TT_EVENT_DESCRIPTOR_DEFINE_ID_PLUS_AT(event, name, ctr, 0u)
#define TT_EVENT_DESCRIPTOR_DEFINE_ID_PLUS(event, name, plus) \
    TT_EVENT_DESCRIPTOR_DEFINE_ID_PLUS_AT(event, name, __COUNTER__, plus)
#define TT_EVENT_DESCRIPTOR_DEFINE_ID(event, name) TT_EVENT_DESCRIPTOR_DEFINE_ID_AT(event, name, __COUNTER__)
