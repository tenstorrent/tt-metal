// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <functional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tt::tt_metal::experimental {

// ============================================================================
//  SemiStrongType  —  a StrongType that permits implicit construction
// ============================================================================
//
// SemiStrongType is a near-verbatim copy of ttsl::StrongType
// (tt_stl/tt_stl/strong_type.hpp) that differs in exactly ONE respect: it is
// IMPLICITLY constructible from its wrapped value, whereas ttsl::StrongType is
// explicit-only.
//
// It is "semi"-strong because it is still fully strong on the axis that matters:
// distinct wrapper types do not interconvert. What it relaxes — deliberately —
// is construction from the underlying type, so that string-literal authoring of
// identifiers stays ergonomic:
//
//     using DFBSpecName       = SemiStrongType<std::string, struct DFBSpecNameTag>;
//     using SemaphoreSpecName = SemiStrongType<std::string, struct SemaphoreSpecNameTag>;
//
//     DFBSpecName a = "in0";       // OK (implicit) — ttsl::StrongType forbids this
//     DFBSpecName b = a;           // OK
//     SemaphoreSpecName s = a;     // DOES NOT COMPILE — different wrapper type
//
// WHY THIS EXISTS, AND WHY IT IS TEMPORARY:
//   ttsl::StrongType's explicit constructor is exactly right for its current
//   users, which are all numeric IDs (DeviceId, MeshId, QueueId, ...) where
//   bare-literal construction is undesirable and even bug-prone. The Metal 2.0
//   ProgramSpec cross-reference identifiers are the first STRING-valued strong
//   types, and they are authored inline as literals, where explicit construction
//   is pure friction. Rather than fork permanently, this is a temporary local
//   mirror: the intended endgame is an opt-in implicit mode on ttsl::StrongType
//   itself, after which this header collapses to a single alias and the call
//   sites are untouched (they only ever reference the `using` aliases).
//
//   Keep this type a faithful mirror of ttsl::StrongType so that the upstream
//   merge stays a minimal, mechanical diff. There are two deliberate deltas:
//     1. The constructor is implicit (the whole point of this type).
//     2. operator<< lives INSIDE this namespace, not at global scope as in
//        ttsl::StrongType. StrongType's global placement is found only by
//        ordinary unqualified lookup (global is never an ADL-associated
//        namespace for a type in `ttsl`), so it is fragile to a closer
//        operator<< hiding it at the call site. Placing it in the type's own
//        namespace makes ADL find it unconditionally. This is a robustness fix
//        that should be carried back into ttsl::StrongType at merge time.
// ============================================================================

template <typename T>
concept HasConstexprThreeWayCompare = requires(T t) {
    { std::bool_constant<(T{}.operator<=>(T{}), true)>() } -> std::same_as<std::true_type>;
};

// lambdas are guaranteed to be unique according to the standard,
// so the default Tag allows each SemiStrongType instantiation
// to be truly unique
template <typename T, typename Tag = decltype([]() {})>
class SemiStrongType {
public:
    using value_type = T;
    using tag_type = Tag;

    constexpr SemiStrongType() noexcept
        requires std::default_initializable<T>
        : value_(T{}) {}

    // IMPLICIT construction — this is the sole intentional difference from
    // ttsl::StrongType. A constrained forwarding constructor, rather than a plain
    // `SemiStrongType(T)`, so that anything T is constructible from converts in a
    // SINGLE user-defined conversion — notably `const char*` for T = std::string,
    // which would otherwise need const char* -> std::string -> wrapper (two hops,
    // which the language refuses to do implicitly). Self is excluded so the
    // copy/move constructors below are not shadowed, and types T cannot be built
    // from (and other wrapper tags) are SFINAE'd out — preserving strength.
    template <typename U>
        requires(!std::same_as<std::remove_cvref_t<U>, SemiStrongType> && std::constructible_from<T, U &&>)
    constexpr SemiStrongType(U&& v) noexcept(std::is_nothrow_constructible_v<T, U&&>) : value_(std::forward<U>(v)) {}

    constexpr SemiStrongType(const SemiStrongType&) = default;
    constexpr SemiStrongType(SemiStrongType&&) noexcept = default;
    constexpr SemiStrongType& operator=(const SemiStrongType&) = default;
    constexpr SemiStrongType& operator=(SemiStrongType&&) noexcept = default;

    constexpr const T& operator*() const { return value_; }
    constexpr const T& get() const { return value_; }

    // requires() = default on a constexpr function doesn't behave on gcc/clang
    // so it needed the explicit definition to compile.
    // We need the separate definition for non/constexpr to accommodate
    // types that don't have constexpr <=> (eg. std::unique_ptr)
    constexpr auto operator<=>(const SemiStrongType& rhs) const noexcept
        requires(HasConstexprThreeWayCompare<T>)
    {
        return value_ <=> rhs.value_;
    }

    auto operator<=>(const SemiStrongType&) const noexcept
        requires(!HasConstexprThreeWayCompare<T>)
    = default;

    static constexpr auto attribute_names = std::forward_as_tuple("value");
    constexpr auto attribute_values() const { return std::forward_as_tuple(value_); }

private:
    T value_;
};

// Type trait to detect SemiStrongType instantiations.
template <typename T>
struct is_semi_strong_type : std::false_type {};
template <typename T, typename Tag>
struct is_semi_strong_type<SemiStrongType<T, Tag>> : std::true_type {};
template <typename T>
inline constexpr bool is_semi_strong_type_v = is_semi_strong_type<T>::value;

// operator<< lives inside the namespace so that ADL finds it when the
// wrapper appears as an operand (callers in any namespace benefit equally).
template <typename T, typename Tag>
std::ostream& operator<<(std::ostream& os, const SemiStrongType<T, Tag>& h) {
    return os << *h;
}

}  // namespace tt::tt_metal::experimental

template <typename T, typename Tag>
struct std::hash<tt::tt_metal::experimental::SemiStrongType<T, Tag>> {
    std::size_t operator()(const tt::tt_metal::experimental::SemiStrongType<T, Tag>& h) const noexcept {
        return std::hash<T>{}(*h);
    }
};
