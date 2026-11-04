// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Observability event descriptors and their ids. An event's descriptor is where the event is declared in the source,
// together with the metadata attached to it. It lives only in the ELF, in .tt_event_descriptor_meta and
// .tt_event_descriptor_str, and never reaches the device. The descriptor id is the number the device sends in the
// descriptor's place; the host uses it to look the descriptor up.
//
// An id is the ADDRESS of a one-byte handle the event emits into the non-ALLOC section .tt_event_descriptor_ids.
// The linker lays the handles out back to back from TT_EVENT_DESCRIPTOR_IDS_LINK_VMA (hw/toolchain/main.ld), so one
// image's ids are dense in link order. When the host loads the image (llrt/observability_event_meta.cpp, from
// ll_api::memory) it gives the image the next free block of the process-wide 16-bit id space and rebases the section
// there by re-resolving the relocations against it: the lui/addi pair each event materializes its id with, and the id
// word of each .tt_event_descriptor_meta record. Ids are thus unique across every image a process loads, with no
// per-TU partition, no registry file, and nothing computed on the device: an id costs the two instructions any 32-bit
// constant does. Nothing on the host may discriminate on an id's value: it depends on load order. The one exception is
// TT_ZONE_STALL_ID, the top of the space, which the allocator never hands out.
//
// Emission is assembler directives inside the one asm statement of the event's descriptor_id(), after libmeta: zero
// .text beyond the two instructions and zero device memory, since none of .tt_event_descriptor_ids,
// .tt_event_descriptor_str, .tt_event_descriptor_meta is SHF_ALLOC. Directives rather than C++ objects with section
// attributes because an event inside a vague-linkage function (inline, template, class-template member) makes such
// objects COMDAT, which GCC will not put in a named section (without LTO a "section type conflict", with LTO an lto1
// ICE). The handle's label is local to the object and guarded by .ifndef, so however many times an inlined event is
// expanded there is one handle and one record for it; TT_PROFILER_TU_ID (this TU's index in its link, from
// jit_build/build.cpp) keeps labels apart when LTO merges a link's TUs into one assembly. .tt_event_descriptor_str is
// "MS" so __FILE__ is stored once per file. Each .tt_event_descriptor_meta record carries the event's metadata struct,
// which emit_event() below marshals; the struct's rules are in observability/api/observability_event_meta.h. The whole
// path, with diagrams: tools/profiler/EVENT_DESCRIPTOR_IDS.md.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "observability/api/observability_event_meta.h"

// The id space: a process assigns [0, TT_ZONE_STALL_ID) to images in load order.
#define TT_EVENT_DESCRIPTOR_ID_BITS 16
#define TT_EVENT_DESCRIPTOR_ID_COUNT (1u << TT_EVENT_DESCRIPTOR_ID_BITS)
#define TT_EVENT_DESCRIPTOR_ID_MASK (TT_EVENT_DESCRIPTOR_ID_COUNT - 1u)
// The profiler's own stall zone: recognized by value, so it has no ELF record and no source location.
#define TT_ZONE_STALL_ID (TT_EVENT_DESCRIPTOR_ID_COUNT - 1u)

// Where the linker script places .tt_event_descriptor_ids before the host rebases it (hw/toolchain/main.ld carries
// the same value). Any address with nonzero upper 20 bits serves: it keeps linker relaxation from folding an event's
// lui/addi into one instruction, which the host could not rebase.
#define TT_EVENT_DESCRIPTOR_IDS_LINK_VMA 0x6800000

// ---- Marshalling -------------------------------------------------------------------------------------------------
// Record in .tt_event_descriptor_meta (little-endian; one per event; records are 4-byte aligned and contiguous):
//   [0]  u32 id          the handle's VMA in .tt_event_descriptor_ids (rebased by the host at load)
//   [4]  u32 signature   VMA in .tt_event_descriptor_str of the type's signature
//   [8]  u32 file        VMA in .tt_event_descriptor_str
//   [12] u32 line
//   [16] the leaf fields in declaration order, nested structs flattened in place, each aligned to min(its size, 4);
//        then padding to 4. A 's' field is a u32 VMA in .tt_event_descriptor_str.
// One event, one record per distinct metadata value: the event's label carries a hash of the value, so copies of an
// event with the same value (inlined, or template instantiations) share a handle and an id, and instantiations whose
// values differ (e.g. a colour taken from a template parameter) each get their own.
//
// emit_event() walks the struct's fields inside a constant expression and returns assembler text, which the event's
// asm((...)) statement emits. An event's strings, quoted, must fit its kMaxEventAsmBytes of assembler text.

namespace tt::observability_event {

// The assembler text one event may generate (its strings, quoted, plus a few hundred bytes of directives).
inline constexpr size_t kMaxEventAsmBytes = 2048;
inline constexpr size_t kRecordHeaderBytes = 16;

namespace detail {

// Never defined: an event whose assembler text would pass kMaxEventAsmBytes calls it during constant evaluation, and
// the compiler's error ("call to non-constexpr function") quotes this name.
void observability_event_metadata_too_long_shorten_its_strings();

// Assembler text, built in a constant expression and handed to asm((...)). Always NUL-terminated.
struct AsmText {
    char buf[kMaxEventAsmBytes] = {};
    size_t n = 0;
    constexpr void put(char c) {
        if (n + 1 >= kMaxEventAsmBytes) {
            observability_event_metadata_too_long_shorten_its_strings();
            return;
        }
        buf[n++] = c;
    }
    constexpr void put(const char* s) {
        while (*s != 0) {
            put(*s++);
        }
    }
    constexpr void put_u(uint64_t v) {
        char t[24] = {};
        int i = 0;
        do {
            t[i++] = static_cast<char>('0' + v % 10);
            v /= 10;
        } while (v != 0);
        while (i != 0) {
            put(t[--i]);
        }
    }
    // A quoted .asciz operand. Every byte outside printable ASCII -- a newline, a tab, UTF-8 -- becomes a three-digit
    // octal escape, so the assembler stores it unchanged and the text stays on one line.
    constexpr void put_quoted(const char* s) {
        put('"');
        for (; *s != 0; s++) {
            const auto u = static_cast<unsigned char>(*s);
            if (*s == '"' || *s == '\\') {
                put('\\');
                put(*s);
            } else if (u < 0x20 || u >= 0x7F) {
                put('\\');
                put(static_cast<char>('0' + (u >> 6)));
                put(static_cast<char>('0' + ((u >> 3) & 7)));
                put(static_cast<char>('0' + (u & 7)));
            } else {
                put(*s);
            }
        }
        put('"');
    }
    constexpr void put_hex(uint64_t v) {
        for (int i = 60; i >= 0; i -= 4) {
            put("0123456789abcdef"[(v >> i) & 0xF]);
        }
    }
    constexpr const char* data() const { return buf; }
    constexpr size_t size() const { return n; }
};

// FNV-1a over an event's metadata value: its signature, then every leaf field.
struct Fnv1a {
    uint64_t h = 0xcbf29ce484222325ull;
    constexpr void put(char c) {
        h ^= static_cast<uint8_t>(c);
        h *= 0x100000001b3ull;
    }
};

template <class T>
constexpr uint64_t meta_hash(const T& meta) {
    Fnv1a f;
    write_signature<T>(f);
    for_each_leaf(meta, [&](const auto& v) {
        using F = field_t<decltype(v)>;
        if constexpr (std::is_same_v<F, const char*>) {
            for (const char* s = v != nullptr ? v : ""; *s != 0; s++) {
                f.put(*s);
            }
            f.put(0);
        } else {
            const uint64_t bits = field_bits(v);
            for (int i = 0; i < 64; i += 8) {
                f.put(static_cast<char>(bits >> i));
            }
        }
    });
    return f.h;
}

// Never defined: an event whose name is null or empty calls it during constant evaluation, and the compiler's error
// ("call to non-constexpr function") quotes this name.
void observability_event_name_must_be_a_non_empty_string_literal();

template <class T>
constexpr const char* event_name(const T& meta) {
    const char* name = nullptr;
    bool first = true;
    for_each_field(meta, [&](const auto& v) {
        if constexpr (std::is_same_v<field_t<decltype(v)>, const char*>) {
            if (first) {
                name = v;
            }
        }
        first = false;
    });
    return name;
}

// The event's name -- its metadata's first field -- is set and non-empty.
template <class T>
constexpr bool event_name_ok(const T& meta) {
    const char* name = event_name(meta);
    return name != nullptr && *name != 0;
}

// A .long pointing at a fresh copy of S in .tt_event_descriptor_str (the section is "MS", so the linker keeps one
// copy). A null pointer is stored as "".
constexpr void put_string_ref(AsmText& a, const char* s) {
    a.put(".pushsection .tt_event_descriptor_str,\"MS\",@progbits,1\n8880:\t.asciz ");
    a.put_quoted(s != nullptr ? s : "");
    a.put("\n.popsection\n.long 8880b\n");
}

struct SignatureText {
    AsmText* a;
    constexpr void put(char c) { a->put(c == '"' || c == '\\' ? '_' : c); }
};

// The whole event: handle, record and the lui/addi that loads the id into %0. The handle's label is the event's
// label plus a hash of the metadata value, and guarded by .ifndef: copies of the event with the same value share
// one handle and one record, copies with different values (template instantiations) each get their own.
template <class T>
constexpr AsmText emit_event(const T& meta, const char* event_label, const char* file, uint32_t line) {
    if (!event_name_ok(meta)) {
        observability_event_name_must_be_a_non_empty_string_literal();
    }
    const uint64_t hash = meta_hash(meta);
    AsmText a;
    auto label = [&] {
        a.put(event_label);
        a.put('_');
        a.put_hex(hash);
    };
    a.put(".ifndef ");
    label();
    a.put("\n.pushsection .tt_event_descriptor_ids,\"\",@progbits\n");
    label();
    a.put(":\t.byte 0\n.popsection\n");

    a.put(".pushsection .tt_event_descriptor_meta,\"\",@progbits\n.balign 4\n.long ");
    label();
    a.put("\n.pushsection .tt_event_descriptor_str,\"MS\",@progbits,1\n8880:\t.asciz \"");
    SignatureText sig{&a};
    write_signature<T>(sig);
    a.put("\"\n.popsection\n.long 8880b\n");
    put_string_ref(a, file);
    a.put(".long ");
    a.put_u(line);
    a.put("\n");

    for_each_leaf(meta, [&](const auto& v) {
        using F = field_t<decltype(v)>;
        constexpr char code = field_code<F>();
        if constexpr (code == 's') {
            put_string_ref(a, v);
        } else {
            constexpr size_t bytes = code_bytes(code);
            a.put(".balign ");
            a.put_u(bytes < 4 ? bytes : 4);
            a.put(bytes == 1 ? "\n.byte " : bytes == 2 ? "\n.short " : bytes == 4 ? "\n.long " : "\n.quad ");
            a.put_u(field_bits(v));
            a.put("\n");
        }
    });
    a.put(".balign 4\n.popsection\n.endif\n\tlui %0, %%hi(");
    label();
    a.put(")\n\taddi %0, %0, %%lo(");
    label();
    a.put(")");
    return a;
}

}  // namespace detail

}  // namespace tt::observability_event

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
// ::tt::observability_event::ZoneColorMeta{"name", 0xFF0000}); emit_event() above marshals it into the event's
// record; the handle's label is TT_OBSERVABILITY_EVENT_LABEL(ctr) plus a hash of that value, so template
// instantiations that give the event different values get different ids. Usable at namespace or block scope. `ctr` is
// a parameter because __COUNTER__ increments on every appearance and the label needs one value. The asm is not
// volatile: beyond its result it has no effect the compiler must order, so repeated uses of one event in a function
// may share a materialization.
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
