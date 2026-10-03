// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <variant>

#include <enchantum/type_name.hpp>

namespace tt::tt_fabric {

// The types a field can have, each with only the data decode needs to read it.
namespace field {

// Unsigned integer.
struct Uint {
    constexpr bool operator==(const Uint&) const = default;
};
// Signed integer.
struct Int {
    constexpr bool operator==(const Int&) const = default;
};
// Enum.
struct Enum {
    // The enum type name
    std::string_view name;
    constexpr bool operator==(const Enum&) const = default;
};
// Struct.
struct Struct {
    // The struct type name
    std::string_view name;
    constexpr bool operator==(const Struct&) const = default;
};
// Raw bytes.
struct Bytes {
    constexpr bool operator==(const Bytes&) const = default;
};
// Padding.
struct Pad {
    constexpr bool operator==(const Pad&) const = default;
};
// A table of entries packed below byte granularity, least significant bit first.
struct Packed {
    // The key decode looks up to find the binding function that reads the entries, such as "direction_table"
    std::string_view table;
    uint32_t bits_per_element;
    constexpr bool operator==(const Packed&) const = default;
};

}  // namespace field

// The type of a field.
using FieldType =
    std::variant<field::Uint, field::Int, field::Enum, field::Struct, field::Bytes, field::Pad, field::Packed>;

// Representation of a field in a struct. Examples:
// - uint16_t counts[2] at offset 6:
//   {name: "counts", offset: 6, size: 4, intra_field_element_count: 2, type: field::Uint{}}
// - RouterState state at offset 0:
//   {name: "state", offset: 0, size: 4, intra_field_element_count: 0, type: field::Enum{"RouterState"}}
// - WorkerXY worker_xy at offset 32:
//   {name: "worker_xy", offset: 32, size: 4, intra_field_element_count: 0, type: field::Struct{"WorkerXY"}}
struct FieldLayout {
    std::string_view name;
    uint32_t offset;
    // Size of the whole field in bytes. For an array, one element is size / intra_field_element_count
    uint32_t size;
    // 0 for a scalar, N for T[N] or a packed table of N entries. An array's element type is the field's type
    uint32_t intra_field_element_count;
    FieldType type;
};

// Each described struct specializes StructLayout<T> with its field list. There is no primary definition, so if a
// member of struct A is itself a struct B with no StructLayout, A's field list fails to compile in field_type().
// B's StructLayout must come before A's.
// NOTE: Fields should be laid out in the order they are defined in the struct, using an array named fields.
template <typename T>
struct StructLayout;

// A struct is described if it has a StructLayout specialization.
template <typename T>
concept Described = requires { StructLayout<T>::fields; };

// Returns the FieldType for a field of type F.
template <typename F>
constexpr FieldType field_type() {
    if constexpr (std::is_enum_v<F>) { /* enum */
        return field::Enum{enchantum::type_name<F>};
    } else if constexpr (std::is_integral_v<F>) { /* integral (whole number) */
        if constexpr (std::is_signed_v<F>) {
            return field::Int{};
        } else {
            return field::Uint{};
        }
    } else if constexpr (Described<F>) { /* defined struct type, via StructLayout<F> */
        return field::Struct{enchantum::type_name<F>};
    } else { /* always fails if we get here */
        static_assert(!sizeof(F*), "member type has no StructLayout specialization; describe it or use LAYOUT_BYTES");
    }
}

// Ensures that every byte of struct T belongs to exactly one field, listed in T's declaration order,
// where each field starts where the previous one ended, and the last one ends at sizeof(T).
template <typename T, std::size_t N>
constexpr bool validate_struct_fields(const std::array<FieldLayout, N>& fields) {
    uint32_t expected_offset = 0;
    for (const auto& field : fields) {
        if (field.offset != expected_offset) {
            return false;
        }
        expected_offset += field.size;
    }
    return expected_offset == sizeof(T);
}

}  // namespace tt::tt_fabric

// Layout of a field whose type field_type() can describe: an integer, an enum, a described struct, or a
// one-dimensional array of one of those.
#define LAYOUT_FIELD(T, member)                                                                                  \
    [] {                                                                                                         \
        using M = decltype(T::member);                                                                           \
        static_assert(std::rank_v<M> <= 1, "multidimensional arrays are not supported");                         \
        constexpr ::tt::tt_fabric::FieldType type = ::tt::tt_fabric::field_type<std::remove_all_extents_t<M>>(); \
        return ::tt::tt_fabric::FieldLayout{                                                                     \
            #member, offsetof(T, member), sizeof(M), static_cast<uint32_t>(std::extent_v<M>), type};             \
    }()

// Layout of a padding member, which decode skips.
#define LAYOUT_PAD(T, member)                                                              \
    ::tt::tt_fabric::FieldLayout {                                                         \
        #member, offsetof(T, member), sizeof(T::member), 0, ::tt::tt_fabric::field::Pad {} \
    }

// Layout of a packed table member, whose type states its entry width as BITS_PER_COMPRESSED_ENTRY. `table` names the
// format, which is used as a key by decode to find the binding function that reads the entries.
#define LAYOUT_PACKED(T, member, table)                                                               \
    [] {                                                                                              \
        using M = decltype(T::member);                                                                \
        constexpr uint32_t bits = M::BITS_PER_COMPRESSED_ENTRY;                                       \
        static_assert(sizeof(M) * 8 % bits == 0, "packed table must hold a whole number of entries"); \
        return ::tt::tt_fabric::FieldLayout{                                                          \
            #member,                                                                                  \
            offsetof(T, member),                                                                      \
            sizeof(M),                                                                                \
            static_cast<uint32_t>(sizeof(M) * 8 / bits),                                              \
            ::tt::tt_fabric::field::Packed{table, bits}};                                             \
    }()

// Layout of a member decode shows as raw bytes without interpreting them, such as a table not yet described.
#define LAYOUT_BYTES(T, member)                                                              \
    ::tt::tt_fabric::FieldLayout {                                                           \
        #member, offsetof(T, member), sizeof(T::member), 0, ::tt::tt_fabric::field::Bytes {} \
    }
