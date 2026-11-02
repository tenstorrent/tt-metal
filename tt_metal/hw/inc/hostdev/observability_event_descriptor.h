// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// DevicePublish descriptor ids and the ELF records that name them. An id travels in low27 of a streaming marker's
// word0; the DRAM backend keeps its own 16-bit timer_id.
//
// An id is the ADDRESS of a one-byte handle the call site emits into the non-ALLOC section
// .device_publish_descriptor_ids. The linker lays the handles out back to back from
// TT_DEVICE_PUBLISH_DESCRIPTOR_IDS_LINK_VMA (hw/toolchain/main.ld), so one image's ids are dense in link order. When
// the host loads the image (llrt/call_sites.cpp, from ll_api::memory) it gives the image the next free block of the
// process-wide 16-bit id space and rebases the section there by re-resolving the relocations against it: the lui/addi
// pair each call site materializes its id with, and the id of each call-site descriptor. Ids are thus unique across
// every image a process loads, with no per-TU partition, no registry file, and nothing computed on the device: an id
// costs the two instructions any 32-bit constant does. Nothing on the host may discriminate on an id's value: it
// depends on load order. The one exception is TT_DEVICE_PUBLISH_STALL_DESCRIPTOR_ID, the top of the space, which the
// allocator never hands out.
//
// Emission is assembler directives inside the one asm statement of the event's id(), after libmeta: zero
// .text beyond the two instructions and zero device memory, since none of .device_publish_descriptor_ids,
// .tt_observability_event_str, .tt_observability_event_meta is SHF_ALLOC. Directives rather than C++ objects with
// section attributes because an event inside a vague-linkage function (inline, template, class-template member)
// makes such objects COMDAT, which GCC will not put in a named section (without LTO a "section type conflict", with LTO
// an lto1 ICE). The handle's label is local to the object and guarded by .ifndef, so however many times an inlined
// event is expanded there is one handle and one record for it; TT_PROFILER_TU_ID (this TU's index in its link, from
// jit_build/build.cpp) keeps labels apart when LTO merges a link's TUs into one assembly. .tt_observability_event_str
// is "MS" so __FILE__ is stored once per file. Each .tt_observability_event_meta record carries the event's metadata
// struct; its layout is in hostdev/observability_event_meta.h. The whole path, with diagrams:
// tools/profiler/EVENT_DESCRIPTOR_IDS.md.
#pragma once

#include <stdint.h>

#include "hostdev/observability_event_meta.h"

// The id space: a process assigns [0, TT_DEVICE_PUBLISH_STALL_DESCRIPTOR_ID) to images in load order.
#define TT_DEVICE_PUBLISH_DESCRIPTOR_ID_BITS 16
#define TT_DEVICE_PUBLISH_DESCRIPTOR_ID_COUNT (1u << TT_DEVICE_PUBLISH_DESCRIPTOR_ID_BITS)
#define TT_DEVICE_PUBLISH_DESCRIPTOR_ID_MASK (TT_DEVICE_PUBLISH_DESCRIPTOR_ID_COUNT - 1u)
// The stall call site, the producer's own, which every image shares: recognized by value, so it has no ELF record and
// no source location (llrt/call_sites.cpp describes it).
#define TT_DEVICE_PUBLISH_STALL_DESCRIPTOR_ID (TT_DEVICE_PUBLISH_DESCRIPTOR_ID_COUNT - 1u)

// Where the linker script places .device_publish_descriptor_ids before the host rebases it (hw/toolchain/main.ld
// carries the same value). Any address with nonzero upper 20 bits serves: it keeps linker relaxation from folding a
// call site's lui/addi into one instruction, which the host could not rebase.
#define TT_DEVICE_PUBLISH_DESCRIPTOR_IDS_LINK_VMA 0x6800000

#define TT_OBSERVABILITY_EVENT_STR_(x) #x
#define TT_OBSERVABILITY_EVENT_STR(x) TT_OBSERVABILITY_EVENT_STR_(x)

// This TU's index among the objects of its link, injected by the JIT build; a TU built outside it is alone.
#ifndef TT_PROFILER_TU_ID
#define TT_PROFILER_TU_ID 0
#endif

#define TT_OBSERVABILITY_EVENT_LABEL(ctr) \
    "__tt_observability_event_" TT_OBSERVABILITY_EVENT_STR(TT_PROFILER_TU_ID) "_" TT_OBSERVABILITY_EVENT_STR(ctr)

// Declares `event` as this event's type; event::descriptor_id() returns the event's descriptor id in two instructions
// with no memory access. The variadic argument is the event's metadata, a constant value of any metadata struct (e.g.
// ::tt::observability_event::ZoneColorMeta{"name", 0xFF0000}); hostdev/observability_event_meta.h marshals it into the
// event's record; the handle's label is TT_OBSERVABILITY_EVENT_LABEL(ctr) plus a hash of that value, so template
// instantiations that give the event different values get different ids. Usable at namespace or block scope. `ctr` is a
// parameter because
// __COUNTER__ increments on every appearance and the label needs one value. The asm is not volatile: beyond its result
// it has no effect the compiler must order, so repeated uses of one event in a function may share a materialization.
#define TT_OBSERVABILITY_EVENT_AT(event, ctr, ...)                                     \
    struct event {                                                                     \
        static inline __attribute__((always_inline)) uint32_t descriptor_id() {        \
            uint32_t v;                                                                \
            static constexpr auto tt_event_meta = __VA_ARGS__;                         \
            asm((::tt::observability_event::detail::emit_event(                        \
                tt_event_meta, TT_OBSERVABILITY_EVENT_LABEL(ctr), __FILE__, __LINE__)) \
                : "=r"(v));                                                            \
            return v;                                                                  \
        }                                                                              \
    }

#define TT_OBSERVABILITY_EVENT(event, ...) TT_OBSERVABILITY_EVENT_AT(event, __COUNTER__, __VA_ARGS__)

// An event whose only metadata is its name.
#define TT_EVENT_DESCRIPTOR_DEFINE_ID(event, name) \
    TT_OBSERVABILITY_EVENT(event, ::tt::observability_event::EventMeta{name})
