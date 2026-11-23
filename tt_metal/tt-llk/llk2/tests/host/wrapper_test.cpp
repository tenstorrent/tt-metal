// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only self-test of the compile-time / run-time value wrappers (inc/common/wrapper/wrapper.h). The rules are
// static_asserts, so compiling this file is most of the test; main() checks what only a run can: a run-time value
// through a function that takes either kind.
//
//   g++ -std=c++17 -Wall -Wextra -Werror -I../../inc/common wrapper_test.cpp -o wrapper_test && ./wrapper_test
//
// It also compiles for the device with the kernels' flags (riscv-tt-elf-g++ -mcpu=tt-bh-tensix -std=c++17 -ftt-nttp
// -ftt-constinit -ftt-consteval, same include path), at any -O level. Only there (__cpp_nontype_template_args says
// so) a ct's value can be of class type: plain C++17 has no class-type template arguments, so the host build leaves
// that one block out.
// -DEXPECT_FAIL=<n> adds one statement that must NOT compile (the cases at the end); the driver
// tests/hardware_free/kernels/test_llk2_wrapper.py builds each and checks that the compiler refuses it.

#include "wrapper/wrapper.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace
{
using namespace llk2;
namespace w = llk2::wrapper;

enum class Format : std::uint8_t
{
    A = 1,
    B = 5
};

// ---- the wrappers themselves ----------------------------------------------------------------------------------------
static_assert(std::is_empty_v<ct<5>> && sizeof(rt<std::uint32_t>) == sizeof(std::uint32_t));
static_assert(w::is_ct_v<ct<5>> && w::is_ct_v<const ct<5>&> && !w::is_ct_v<rt<int>> && !w::is_ct_v<int>);
static_assert(w::is_rt_v<rt<int>> && w::is_rt_v<const rt<int>&> && !w::is_rt_v<ct<5>> && !w::is_rt_v<int>);
static_assert(w::value(ct<5> {}) == 5 && w::value(rt {6}) == 6 && w::value(7) == 7);
static_assert(std::is_same_v<decltype(rt {5u}), rt<unsigned>>);
static_assert(std::is_same_v<ct<Format::B>::type, Format> && ct<Format::B>::value == Format::B);

#if __cpp_nontype_template_args >= 201911L
// A value of class type (the kernels' flags have -ftt-nttp; plain C++17 does not): a modifier struct as a ct.
struct Mods
{
    bool relu;
    std::uint32_t tiles;
};

static_assert(ct<Mods {true, 8}>::value.tiles == 8 && w::value(ct<Mods {true, 8}> {}).relu && std::is_same_v<ct<Mods {true, 8}>::type, Mods>);
#endif

// ---- a ct never decays; an rt gives its value freely -----------------------------------------------------------------
static_assert(!std::is_convertible_v<ct<5>, int> && !std::is_convertible_v<ct<5>, std::uint32_t> && !std::is_convertible_v<ct<true>, bool>);
static_assert(!std::is_constructible_v<int, ct<5>> && !std::is_constructible_v<rt<int>, ct<5>> && !std::is_convertible_v<ct<Format::A>, Format>);
static_assert(std::is_convertible_v<rt<int>, int> && std::is_convertible_v<int, rt<int>>);

// ---- ctrt<X, T>: a parameter of either kind --------------------------------------------------------------------------
static_assert(w::is_ctrt_v<ct<8>, std::uint32_t> && w::is_ctrt_v<ct<8u>, std::uint32_t> && w::is_ctrt_v<rt<int>, std::uint32_t>);
static_assert(w::is_ctrt_v<ct<std::uint32_t {8}>, std::uint32_t> && w::is_ctrt_v<ct<std::size_t {8}>, std::uint32_t>);
static_assert(w::is_ctrt_v<int, std::uint32_t> && w::is_ctrt_v<ct<255>, std::uint8_t> && w::is_ctrt_v<ct<Format::A>, Format>);
static_assert(!w::is_ctrt_v<ct<-1>, std::uint32_t> && !w::is_ctrt_v<ct<256>, std::uint8_t> && !w::is_ctrt_v<ct<1>, Format>);
static_assert(!w::is_ctrt_v<const char*, std::uint32_t>);

// A function of either kind picks its form from the type, and its loop bound is a constant for a ct (llk-lib's
// execute functions are written this way).
template <class N, ctrt<N, std::uint32_t> = 0>
constexpr int form_of(const N&)
{
    if constexpr (w::is_ct_v<N>)
    {
        static_assert(N::value >= 1 && N::value <= 128, "the constant is usable as one");
        return 1; // the immediate form
    }
    else
    {
        return 2; // the run-time form
    }
}

template <class N, ctrt<N, std::uint32_t> = 0>
constexpr std::uint32_t sum_to(const N& n)
{
    std::uint32_t sum = 0;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(w::value(n)); i++)
    {
        sum += i;
    }
    return sum;
}

template <class X, class = void>
inline constexpr bool takes = false;
template <class X>
inline constexpr bool takes<X, std::void_t<decltype(sum_to(std::declval<X>()))>> = true;
static_assert(takes<ct<4>> && takes<rt<std::uint32_t>> && takes<int> && takes<const ct<4>&> && !takes<ct<-4>> && !takes<const char*>);
static_assert(form_of(ct<8> {}) == 1 && form_of(rt<std::uint32_t> {8}) == 2 && form_of(8) == 2);
static_assert(sum_to(ct<5> {}) == 10 && sum_to(rt<std::uint32_t> {5}) == 10 && sum_to(5) == 10);

#if defined(EXPECT_FAIL)
[[maybe_unused]] void must_not_compile()
{
#if EXPECT_FAIL == 1
    [[maybe_unused]] const std::uint32_t n = ct<5> {}; // a ct does not decay
#elif EXPECT_FAIL == 2
    [[maybe_unused]] const rt<std::uint32_t> n = ct<5> {}; // nor become run time
#elif EXPECT_FAIL == 3
    sum_to(ct<-4> {}); // not a std::uint32_t
#elif EXPECT_FAIL == 4
    if (ct<5> {}) // no contextual conversion
    {
    }
#else
#error "EXPECT_FAIL: no such case"
#endif
}
#endif

} // namespace

int main()
{
    volatile std::uint32_t tiles_now = 5; // genuinely run time
    const rt tiles {std::uint32_t {tiles_now}};
    int failed = 0;
    failed += sum_to(tiles) != 10 || form_of(tiles) != 2;
    failed += w::value(tiles) != 5 || std::uint32_t {tiles} != 5;
    return failed;
}
