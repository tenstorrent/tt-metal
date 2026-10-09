// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-site metadata for debug events. A metadata type is a plain C++ struct; a site carries one constant value
// of it, which the compiler marshals into the site's ELF record and the host unmarshals at load. The device
// never sees it: it only has the site's id (hostdev/profiler_zone_id.h).
//
// A metadata type must be a simple aggregate: a struct with public data members only -- no base classes,
// constructors, C arrays, const or reference members -- and 1..kMaxFields of them, each one of
//   const char*        set from a string literal
//   integer or enum    of 1, 2, 4 or 8 bytes (bool included)
//   a nested struct    itself a simple aggregate
// and every value must be a compile-time constant. check_type<T>() enforces this with readable errors.
//
// Marshalling: emit_site() walks the struct's fields (C++17 aggregate decomposition) inside a constant expression
// and returns assembler text, which the site's asm((...)) statement emits. Unmarshalling: the host walks the same
// fields of the same struct (SiteMeta::as<T>()).
//
// Record in .tt_zone_meta (little-endian; one per site; records are 4-byte aligned and contiguous):
//   [0]  u32 id          the handle's VMA in .tt_zone_ids (rebased by the host at load)
//   [4]  u32 signature   VMA in .tt_zone_str of "<type name>:<field codes>", e.g. "tt::debug_event::ZoneColorMeta:sw"
//   [8]  u32 file        VMA in .tt_zone_str
//   [12] u32 line
//   [16] the leaf fields in declaration order, nested structs flattened in place, each aligned to min(its size,
//        4); then padding to 4
// Field codes: 's' const char* (u32 VMA in .tt_zone_str), 'b' 1 byte, 'h' 2, 'w' 4, 'q' 8; a nested struct is its
// own codes in braces, so {const char* name; Style{u32, u8} style; u16 flags;} is "s{wb}h". Braces carry no bytes.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

namespace tt::debug_event {

// ---- Metadata types ----------------------------------------------------------------------------------------------
// By convention the first const char* field is the site's display name.

struct ZoneMeta {
    const char* name;
};

struct ZoneColorMeta {
    const char* name;
    uint32_t color;  // 0xRRGGBB
};

// ---- Field walk, shared by the device marshaller and the host unmarshaller -----------------------------------------

inline constexpr size_t kMaxFields = 8;
inline constexpr size_t kRecordHeaderBytes = 16;

namespace detail {

struct any_field {
    template <class T>
    constexpr operator T() const;  // never defined: only used to count fields
};

template <class T, class = void, class... A>
struct brace_ok : std::false_type {};
template <class T, class... A>
struct brace_ok<T, std::void_t<decltype(T{std::declval<A>()...})>, A...> : std::true_type {};

template <class T, class... A>
constexpr size_t count_fields() {
    if constexpr (sizeof...(A) <= kMaxFields && brace_ok<T, void, A..., any_field>::value) {
        return count_fields<T, A..., any_field>();
    } else {
        return sizeof...(A);
    }
}

// The same count with every initializer in its own braces, T{{x}, {x}}. Brace elision lets the bare form above fill
// a C array member one element per initializer, so the two counts differ exactly when T has an array member.
template <class T, class = void, class... A>
struct braced_ok : std::false_type {};
template <class T, class... A>
struct braced_ok<T, std::void_t<decltype(T{{std::declval<A>()}...})>, A...> : std::true_type {};

template <class T, class... A>
constexpr size_t count_braced_fields() {
    if constexpr (sizeof...(A) <= kMaxFields && braced_ok<T, void, A..., any_field>::value) {
        return count_braced_fields<T, A..., any_field>();
    } else {
        return sizeof...(A);
    }
}

// Calls f on each direct field of T. Does nothing for a T that check_type rejects, so a bad type reports its rule
// and not a cascade of structured-binding errors.
template <class T, class F>
constexpr void for_each_field(T& t, F&& f) {
    using U = std::remove_cv_t<T>;
    constexpr size_t n = count_fields<U>();
    if constexpr (n != count_braced_fields<U>() || n < 1 || n > kMaxFields) {
        return;
    } else if constexpr (n == 1) {
        auto& [a] = t;
        f(a);
    } else if constexpr (n == 2) {
        auto& [a, b] = t;
        f(a), f(b);
    } else if constexpr (n == 3) {
        auto& [a, b, c] = t;
        f(a), f(b), f(c);
    } else if constexpr (n == 4) {
        auto& [a, b, c, d] = t;
        f(a), f(b), f(c), f(d);
    } else if constexpr (n == 5) {
        auto& [a, b, c, d, e] = t;
        f(a), f(b), f(c), f(d), f(e);
    } else if constexpr (n == 6) {
        auto& [a, b, c, d, e, g] = t;
        f(a), f(b), f(c), f(d), f(e), f(g);
    } else if constexpr (n == 7) {
        auto& [a, b, c, d, e, g, h] = t;
        f(a), f(b), f(c), f(d), f(e), f(g), f(h);
    } else {
        auto& [a, b, c, d, e, g, h, i] = t;
        f(a), f(b), f(c), f(d), f(e), f(g), f(h), f(i);
    }
}

template <class>
inline constexpr bool always_false = false;

template <class F>
inline constexpr bool is_leaf_v = std::is_same_v<F, const char*> || std::is_integral_v<F> || std::is_enum_v<F>;

template <class V>
using field_t = std::remove_cv_t<std::remove_reference_t<V>>;

// Rejects anything the walker would mishandle, with a message that names the rule; true when T is usable.
template <class T>
constexpr bool check_type() {
    static_assert(
        std::is_class_v<T> && std::is_aggregate_v<T>,
        "a debug event metadata type must be a simple aggregate: a struct with public data members only");
    constexpr size_t n = count_fields<T>();
    static_assert(n == count_braced_fields<T>(), "a debug event metadata type cannot have C array members");
    static_assert(n >= 1 && n <= kMaxFields, "a debug event metadata type needs 1..kMaxFields fields");
    if constexpr (n == count_braced_fields<T>() && n >= 1 && n <= kMaxFields) {
        T probe{};
        for_each_field(probe, [](auto& v) {
            using V = std::remove_reference_t<decltype(v)>;
            using F = field_t<V>;
            static_assert(!std::is_const_v<V>, "a debug event metadata type cannot have const members");
            static_assert(
                is_leaf_v<F> || std::is_class_v<F>,
                "a metadata field must be a string literal (const char*), an integer, an enum, a bool or a nested "
                "struct");
            if constexpr (std::is_class_v<F>) {
                static_assert(check_type<F>());
            }
        });
    }
    return true;
}

// Calls leaf(v) on every non-struct field of t in declaration order, descending into nested structs.
template <class T, class Leaf>
constexpr void for_each_leaf(T& t, Leaf&& leaf) {
    for_each_field(t, [&](auto& v) {
        using F = field_t<decltype(v)>;
        if constexpr (std::is_class_v<F>) {
            for_each_leaf(v, leaf);
        } else if constexpr (is_leaf_v<F>) {
            leaf(v);
        }
    });
}

template <class F>
constexpr char field_code() {
    if constexpr (std::is_same_v<F, const char*>) {
        return 's';
    } else if constexpr (std::is_integral_v<F> || std::is_enum_v<F>) {
        static_assert(sizeof(F) == 1 || sizeof(F) == 2 || sizeof(F) == 4 || sizeof(F) == 8);
        return sizeof(F) == 1 ? 'b' : sizeof(F) == 2 ? 'h' : sizeof(F) == 4 ? 'w' : 'q';
    } else {
        static_assert(always_false<F>, "a metadata field must be const char* or an integer/enum");
        return 0;
    }
}

constexpr size_t code_bytes(char c) {
    return c == 's' || c == 'w' ? 4 : c == 'b' ? 1 : c == 'h' ? 2 : c == 'q' ? 8 : 0;
}

// The field's value as the unsigned integer of its own width, so a negative int32 is emitted as 4 bytes.
template <class F>
constexpr uint64_t field_bits(F v) {
    if constexpr (std::is_enum_v<F>) {
        return field_bits(static_cast<std::underlying_type_t<F>>(v));
    } else if constexpr (std::is_same_v<F, bool>) {
        return v ? 1 : 0;
    } else {
        return static_cast<std::make_unsigned_t<F>>(v);
    }
}

template <class T>
constexpr const char* pretty_function() {
    return __PRETTY_FUNCTION__;  // GCC "... [with T = X]", clang "... [T = X]"
}

template <class T, class Sink>
constexpr void write_codes(Sink& out) {
    T probe{};
    for_each_field(probe, [&](auto& v) {
        using F = field_t<decltype(v)>;
        if constexpr (std::is_class_v<F>) {
            out.put('{');
            write_codes<F>(out);
            out.put('}');
        } else if constexpr (is_leaf_v<F>) {
            out.put(field_code<F>());
        }
    });
}

// Writes "<type name>:<field codes>" into any sink with put(char).
template <class T, class Sink>
constexpr void write_signature(Sink& out) {
    static_assert(check_type<T>());
    const char* p = pretty_function<T>();
    while (*p != 0 && !(p[0] == 'T' && p[1] == ' ' && p[2] == '=' && p[3] == ' ')) {
        p++;
    }
    for (p += 4; *p != 0 && *p != ']' && *p != ';'; p++) {
        out.put(*p);
    }
    out.put(':');
    write_codes<T>(out);
}

// Assembler text, built in a constant expression and handed to asm((...)).
struct AsmText {
    char buf[2048] = {};
    size_t n = 0;
    constexpr void put(char c) { buf[n++] = c; }
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
    // A quoted .asciz operand.
    constexpr void put_quoted(const char* s) {
        put('"');
        for (; *s != 0; s++) {
            if (*s == '"' || *s == '\\') {
                put('\\');
            }
            put(*s);
        }
        put('"');
    }
    constexpr const char* data() const { return buf; }
    constexpr size_t size() const { return n; }
};

// A .long pointing at a fresh copy of S in .tt_zone_str (the section is "MS", so the linker keeps one copy). A null
// pointer is stored as "".
constexpr void put_string_ref(AsmText& a, const char* s) {
    a.put(".pushsection .tt_zone_str,\"MS\",@progbits,1\n8880:\t.asciz ");
    a.put_quoted(s != nullptr ? s : "");
    a.put("\n.popsection\n.long 8880b\n");
}

struct SignatureText {
    AsmText* a;
    constexpr void put(char c) { a->put(c == '"' || c == '\\' ? '_' : c); }
};

// The whole site: handle, record and the lui/addi that loads the id into %0. Guarded by .ifndef so an inlined
// site expanded many times still has one handle and one record.
template <class T>
constexpr AsmText emit_site(const T& meta, const char* label, const char* file, uint32_t line) {
    AsmText a;
    a.put(".ifndef ");
    a.put(label);
    a.put("\n.pushsection .tt_zone_ids,\"\",@progbits\n");
    a.put(label);
    a.put(":\t.byte 0\n.popsection\n");

    a.put(".pushsection .tt_zone_meta,\"\",@progbits\n.balign 4\n.long ");
    a.put(label);
    a.put("\n.pushsection .tt_zone_str,\"MS\",@progbits,1\n8880:\t.asciz \"");
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
    a.put(label);
    a.put(")\n\taddi %0, %0, %%lo(");
    a.put(label);
    a.put(")");
    return a;
}

}  // namespace detail

}  // namespace tt::debug_event

#if !defined(__riscv)
// ---- Host side --------------------------------------------------------------------------------------------------
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tt::debug_event {

// One unmarshalled field: s for 's', bits for the integer codes.
struct FieldValue {
    char code = 0;
    uint64_t bits = 0;
    const char* s = nullptr;  // valid for the life of the process
};

// A site's metadata as the host holds it before anyone asks for a type.
struct SiteMeta {
    std::string_view signature;  // "<type name>:<field codes>"
    std::vector<FieldValue> fields;  // one per leaf field, nested structs flattened

    std::string_view type_name() const { return signature.substr(0, signature.rfind(':')); }

    // The value as T, or nullopt when the site carries another type.
    template <class T>
    std::optional<T> as() const;
};

template <class T>
const std::string& signature_of() {
    static const std::string sig = [] {
        struct Sink {
            std::string s;
            void put(char c) { s.push_back(c == '"' || c == '\\' ? '_' : c); }
        } out;
        detail::write_signature<T>(out);
        return std::move(out.s);
    }();
    return sig;
}

template <class T>
std::optional<T> SiteMeta::as() const {
    if (signature != signature_of<T>()) {
        return std::nullopt;
    }
    T t{};
    size_t i = 0;
    detail::for_each_leaf(t, [&](auto& f) {
        using F = detail::field_t<decltype(f)>;
        const FieldValue& v = fields[i++];  // the signature matched, so there is one value per leaf
        if constexpr (std::is_same_v<F, const char*>) {
            f = v.s;
        } else if constexpr (std::is_same_v<F, bool>) {
            f = v.bits != 0;
        } else if constexpr (std::is_enum_v<F>) {
            f = static_cast<F>(static_cast<std::make_unsigned_t<std::underlying_type_t<F>>>(v.bits));
        } else {
            f = static_cast<F>(static_cast<std::make_unsigned_t<F>>(v.bits));
        }
    });
    return t;
}

}  // namespace tt::debug_event
#endif
