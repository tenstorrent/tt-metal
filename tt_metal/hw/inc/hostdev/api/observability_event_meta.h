// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Per-event metadata for observability events, as the device and the host both see it. A metadata type is a plain C++
// struct; an event carries one constant value of it, which the device compiler marshals into the event's ELF record
// and the host unmarshals at load. The device never sees the value at run time: it only has the event's descriptor id
// (hostdev/observability_event_descriptor.h).
//
// This header is public: the device compiler and host compilers, inside and outside tt-metal, compile it unchanged.
// It holds what both sides must agree on -- the metadata types, the rules a metadata type must follow and the
// signature that names a type in its record -- and nothing that only one side uses. The device's marshalling is in
// hostdev/observability_event_descriptor.h; the host reads a record's value back with Metadata::as<T>()
// (tt-metalium/experimental/observability_event_meta.hpp).
//
// A metadata type must be a simple aggregate: a struct with public data members only -- no base classes,
// constructors, C arrays, const or reference members -- and 1..kMaxFields of them, each one of
//   const char*        set from a string literal
//   integer or enum    of 1, 2, 4 or 8 bytes (bool included)
//   a nested struct    itself a simple aggregate
// and every value must be a compile-time constant. Its FIRST field is the event's name, a const char* that every
// event sets to a non-empty string. The type must be named at namespace scope (not in an anonymous namespace, not
// local to a function, not unnamed), because the host matches it by the name the compiler spells. check_event_type<T>()
// enforces all of this with readable errors.
//
// Both sides walk the struct's fields the same way (C++17 aggregate decomposition, for_each_leaf), and name the type
// with the same signature, "<type name>:<field codes>", e.g. "tt::observability_event::ZoneColorMeta:sw".
// Field codes: 's' const char*, 'b' 1 byte, 'h' 2, 'w' 4, 'q' 8; a nested struct is its own codes in braces, so
// {const char* name; Style{u32, u8} style; u16 flags;} is "s{wb}h". Braces carry no bytes.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>

namespace tt::observability_event {

// ---- Metadata types ----------------------------------------------------------------------------------------------
// The first field of every metadata type is the event's name.

struct EventMeta {
    const char* name;
};

struct ZoneColorMeta {
    const char* name;
    uint32_t color;  // 0xRRGGBB
};

// How a consumer should present a zone's duration. The device and the host still count in cycles; this is only a
// preference carried with the zone's metadata.
enum class TimeUnit : uint8_t { Cycles, Nanoseconds, Microseconds, Milliseconds };

struct ZoneUnitMeta {
    const char* name;
    uint32_t color;  // 0xRRGGBB
    TimeUnit unit;
};

// ---- Field walk, shared by the device marshaller and the host unmarshaller -----------------------------------------

inline constexpr size_t kMaxFields = 8;

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

// Converts only to a proper base class of T. A base is T's first initializer element, so T{any_base<T>{}} compiles
// exactly when T has one; a structured binding cannot see fields through it.
template <class T>
struct any_base {
    template <class B, class = std::enable_if_t<std::is_base_of_v<B, T> && !std::is_same_v<B, T>>>
    constexpr operator B() const;  // never defined: only used to detect a base
};

template <class T>
constexpr bool has_direct_base() {
    return brace_ok<T, void, any_base<T>>::value;
}

// True when the field walker can decompose T: no base class, no array members and 1..kMaxFields fields.
template <class T>
constexpr bool shape_ok() {
    constexpr size_t n = count_fields<T>();
    return !has_direct_base<T>() && n == count_braced_fields<T>() && n >= 1 && n <= kMaxFields;
}

// Calls f on each direct field of T. Does nothing for a T that check_type rejects, so a bad type reports its rule
// and not a cascade of structured-binding errors.
template <class T, class F>
constexpr void for_each_field(T& t, F&& f) {
    using U = std::remove_cv_t<T>;
    constexpr size_t n = count_fields<U>();
    if constexpr (!shape_ok<U>()) {
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

// ---- Rules for a metadata type, as plain predicates ---------------------------------------------------------------
// Each answers one rule and is true when it holds; check_type / check_event_type turn them into readable compile
// errors, and tests call them directly. Rules that need the fields (all but is_simple_aggregate) assume the type is a
// simple aggregate and descend into nested structs; when an earlier rule already fails they answer true, so a bad
// type reports only the rule it breaks.

template <class T>
inline constexpr bool is_simple_aggregate_v = std::is_class_v<T> && std::is_aggregate_v<T>;

// No base classes, at any depth.
template <class T>
constexpr bool has_no_bases() {
    if constexpr (!is_simple_aggregate_v<T>) {
        return true;
    } else {
        if (has_direct_base<T>()) {
            return false;
        }
        bool ok = true;
        T probe{};
        for_each_field(probe, [&](auto& v) {
            if constexpr (is_simple_aggregate_v<field_t<decltype(v)>>) {
                ok = ok && has_no_bases<field_t<decltype(v)>>();
            }
        });
        return ok;
    }
}

// No C array members, at any depth.
template <class T>
constexpr bool has_no_arrays() {
    if constexpr (!is_simple_aggregate_v<T> || has_direct_base<T>()) {
        return true;
    } else {
        if (count_fields<T>() != count_braced_fields<T>()) {
            return false;
        }
        bool ok = true;
        T probe{};
        for_each_field(probe, [&](auto& v) {
            if constexpr (is_simple_aggregate_v<field_t<decltype(v)>>) {
                ok = ok && has_no_arrays<field_t<decltype(v)>>();
            }
        });
        return ok;
    }
}

// 1..kMaxFields direct fields, in this struct and every nested one.
template <class T>
constexpr bool field_count_ok() {
    if constexpr (!is_simple_aggregate_v<T> || has_direct_base<T>()) {
        return true;
    } else {
        constexpr size_t n = count_braced_fields<T>();  // the true count even with arrays
        if (n < 1 || n > kMaxFields) {
            return false;
        }
        bool ok = true;
        T probe{};
        for_each_field(probe, [&](auto& v) {
            if constexpr (is_simple_aggregate_v<field_t<decltype(v)>>) {
                ok = ok && field_count_ok<field_t<decltype(v)>>();
            }
        });
        return ok;
    }
}

// No const members, at any depth.
template <class T>
constexpr bool has_no_const_members() {
    bool ok = true;
    if constexpr (is_simple_aggregate_v<T>) {
        T probe{};
        for_each_field(probe, [&](auto& v) {
            using V = std::remove_reference_t<decltype(v)>;
            ok = ok && !std::is_const_v<V>;
            if constexpr (is_simple_aggregate_v<field_t<V>>) {
                ok = ok && has_no_const_members<field_t<V>>();
            }
        });
    }
    return ok;
}

// Every field is a const char*, an integer, an enum, a bool or a nested simple aggregate, at any depth.
template <class T>
constexpr bool fields_supported() {
    bool ok = true;
    if constexpr (is_simple_aggregate_v<T>) {
        T probe{};
        for_each_field(probe, [&](auto& v) {
            using F = field_t<decltype(v)>;
            if constexpr (is_simple_aggregate_v<F>) {
                ok = ok && fields_supported<F>();
            } else {
                ok = ok && is_leaf_v<F>;
            }
        });
    }
    return ok;
}

template <class T>
constexpr bool is_valid_meta_type() {
    return is_simple_aggregate_v<T> && has_no_bases<T>() && has_no_arrays<T>() && field_count_ok<T>() &&
           has_no_const_members<T>() && fields_supported<T>();
}

// One readable compile error for each rule T breaks; true when T is usable.
template <class T>
constexpr bool check_type() {
    static_assert(
        is_simple_aggregate_v<T>,
        "an observability event metadata type must be a simple aggregate: a struct with public data members only");
    static_assert(
        has_no_bases<T>(), "an observability event metadata type and each struct in it cannot have a base class");
    static_assert(has_no_arrays<T>(), "an observability event metadata type cannot have C array members");
    static_assert(
        field_count_ok<T>(), "an observability event metadata type and each struct in it need 1..kMaxFields fields");
    static_assert(has_no_const_members<T>(), "an observability event metadata type cannot have const members");
    static_assert(
        fields_supported<T>(),
        "a metadata field must be a string literal (const char*), an integer, an enum, a bool or a nested struct "
        "with public data members only");
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

// Where the type's name starts inside __PRETTY_FUNCTION__; it runs to the first ']' or ';'.
template <class T>
constexpr const char* type_name_begin() {
    const char* p = pretty_function<T>();
    while (*p != 0 && !(p[0] == 'T' && p[1] == ' ' && p[2] == '=' && p[3] == ' ')) {
        p++;
    }
    return *p != 0 ? p + 4 : p;
}

constexpr bool type_name_end(char c) { return c == 0 || c == ']' || c == ';'; }

// False for names the device and host compilers spell differently: GCC "{anonymous}::X" vs clang
// "(anonymous namespace)::X", function-local "f()::X", GCC "<unnamed struct>" vs clang "(unnamed struct at ...)".
template <class T>
constexpr bool has_portable_name() {
    auto starts = [](const char* p, const char* word) {
        while (*word != 0 && *p == *word) {
            p++, word++;
        }
        return *word == 0;
    };
    for (const char* p = type_name_begin<T>(); !type_name_end(*p); p++) {
        if (*p == '(' || *p == '{' || starts(p, "<unnamed") || starts(p, "<anonymous")) {
            return false;
        }
    }
    return true;
}

template <class T>
constexpr bool first_field_is_name() {
    T probe{};
    bool first = true, is_name = false;
    for_each_field(probe, [&](auto& v) {
        if (first) {
            is_name = std::is_same_v<field_t<decltype(v)>, const char*>;
        }
        first = false;
    });
    return is_name;
}

// An event's metadata type: a valid metadata type, matched by name on the host, whose first field is the name.
template <class T>
constexpr bool is_valid_event_type() {
    return is_valid_meta_type<T>() && has_portable_name<T>() && first_field_is_name<T>();
}

// Everything an event's metadata type must satisfy, beyond check_type's per-struct rules.
template <class T>
constexpr bool check_event_type() {
    static_assert(check_type<T>());
    static_assert(
        has_portable_name<T>(),
        "an observability event metadata type must be named at namespace scope: not in an anonymous namespace, not "
        "local to a "
        "function, not an unnamed struct (the host matches it by name, and compilers spell those differently)");
    static_assert(
        !shape_ok<T>() || first_field_is_name<T>(),
        "the first field of an observability event metadata type must be the event's name, a const char*");
    return true;
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
    static_assert(check_event_type<T>());
    for (const char* p = type_name_begin<T>(); !type_name_end(*p); p++) {
        out.put(*p);
    }
    out.put(':');
    write_codes<T>(out);
}

}  // namespace detail

}  // namespace tt::observability_event
