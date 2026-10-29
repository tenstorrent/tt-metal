// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// One callback that takes a record type it can't for each HOST_ERROR value. Each case must fail to compile with one
// error, that rule's message (CMakeLists.txt lists them).
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <variant>

#include <tt-metalium/experimental/streaming_profiler.hpp>

namespace host_errors {

using namespace tt::tt_metal::experimental::streaming_profiler;

struct Name {
    String name;
};
struct Stamp {
    Timestamp at;
};
struct WithString {
    uint32_t v;
    String s;
};
struct WithConstructor {
    WithConstructor(uint32_t v) : v(v) {}
    uint32_t v;
};
template <int N>
struct Templated {
    uint32_t v;
};
enum Unfixed { kUnfixed };
struct UnfixedField {
    Unfixed v;
};
union Word {
    uint32_t raw;
    float f;
};
struct ConstMember {
    const uint32_t v;
};
struct CopiedByHand {
    CopiedByHand() = default;
    CopiedByHand(const CopiedByHand&) {}
};
struct HoldsCopiedByHand {
    CopiedByHand c;
};
struct alignas(64) OverAligned {
    uint32_t v;
};
template <size_t... I>
Batch<Record<std::array<uint32_t, I + 1>, Name>...> batch_of_arrays(std::index_sequence<I...>);

void register_case() {
#if HOST_ERROR == 1
    using CaseBatch = Batch<Record<std::tuple<uint32_t>, Name>>;
#elif HOST_ERROR == 2
    using CaseBatch = Batch<Record<std::tuple<>, Name>>;
#elif HOST_ERROR == 3
    using CaseBatch = Batch<Record<std::tuple<Variadic<uint32_t>, float>, Name>>;
#elif HOST_ERROR == 4
    using CaseBatch = Batch<Record<std::variant<Variadic<uint32_t>, float>, Name>>;
#elif HOST_ERROR == 5
    using CaseBatch = Batch<Record<WithString, Name>>;
#elif HOST_ERROR == 6
    using CaseBatch = Batch<Record<std::variant<uint32_t, String>, Name>>;
#elif HOST_ERROR == 7
    using CaseBatch = Batch<Record<uint32_t, Stamp>>;
#elif HOST_ERROR == 8
    using CaseBatch = Batch<Record<uint32_t, uint32_t>>;
#elif HOST_ERROR == 9
    using CaseBatch = Batch<Record<std::variant<std::variant<uint32_t, float>, double>, Name>>;
#elif HOST_ERROR == 10
    using CaseBatch = Batch<Record<std::span<const std::variant<uint32_t, float>>, Name>>;
#elif HOST_ERROR == 11
    using CaseBatch = Batch<Record<std::tuple<std::tuple<uint32_t, float>, double>, Name>>;
#elif HOST_ERROR == 12
    using CaseBatch = Batch<Record<std::string, Name>>;
#elif HOST_ERROR == 13
    using CaseBatch = Batch<Record<WithConstructor, Name>>;
#elif HOST_ERROR == 14
    using CaseBatch = Batch<Record<uint32_t*, Name>>;
#elif HOST_ERROR == 15
    using CaseBatch = Batch<Record<long double, Name>>;
#elif HOST_ERROR == 16
    using CaseBatch = Batch<Record<std::span<const std::span<const uint32_t>>, Name>>;
#elif HOST_ERROR == 17
    using CaseBatch = Batch<Record<std::array<uint32_t, 0>, Name>>;
#elif HOST_ERROR == 18
    using CaseBatch = Batch<Record<Templated<1>, Name>>;
#elif HOST_ERROR == 19
    using CaseBatch = Batch<Record<Variadic<Variadic<uint32_t>>, Name>>;
#elif HOST_ERROR == 20
    using CaseBatch = Batch<Record<UnfixedField, Name>>;
#elif HOST_ERROR == 21
    using CaseBatch = Batch<Record<Word, Name>>;
#elif HOST_ERROR == 22
    using CaseBatch = Batch<Record<ConstMember, Name>>;
#elif HOST_ERROR == 23
    using CaseBatch = Batch<Record<HoldsCopiedByHand, Name>>;
#elif HOST_ERROR == 24
    using CaseBatch = Batch<Record<OverAligned, Name>>;
#elif HOST_ERROR == 25
    using CaseBatch = Batch<Record<uint32_t, Name>, Record<uint32_t, Name>>;
#elif HOST_ERROR == 26
    using CaseBatch = decltype(batch_of_arrays(std::make_index_sequence<65>()));
#endif
    auto callback = RegisterCallback([](const CaseBatch&) {});
}

}  // namespace host_errors
