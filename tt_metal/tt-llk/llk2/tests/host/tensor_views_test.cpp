// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Host-only test of the Tensor views (inc/common/tensor/tensor.h) and of the kinds of argument wait_front /
// reserve_back / pop_front take (inc/common/compute_api/dataflow.h). No board, no simulator.
//
//   g++ -std=c++20 -Wall -Wextra -Werror -DTENSIX_FIRMWARE -Istubs -I../helpers -I../../inc/common
//       -I<tt-metal>/tt_metal/hw/inc/internal/tt-1xx/blackhole tensor_views_test.cpp -o tensor_views_test && ./tensor_views_test
//
// What holds at compile time is in ../helpers/tensor_views_selftest.h (static_asserts, also built with the dataflow
// harness test); main() checks what only a run can: that a run-time start or length outside the tensor reaches
// LLK_ASSERT (counted by stubs/llk_assert.h). The host needs C++20 (class-type template arguments); the device
// compiler builds this file with the kernels' flags (-std=c++17 -ftt-nttp ...).
// -DEXPECT_FAIL=<n> adds one statement that must NOT compile (the cases at the end); the driver
// tests/hardware_free/kernels/test_llk2_views.py builds each and checks that the compiler refuses it with the
// message that says why.

#include <cstdint>

#include "compute_api/dataflow.h"
#include "tensor_views_selftest.h"

namespace
{
using namespace llk2;
using views_selftest::m_spec;
using views_selftest::s_spec;

inline constexpr const auto& row_spec     = l1_spec<32, 16 * 32, DataFormat::Float16_b>; // [1, 16] tiles
inline constexpr const auto& some_of_row  = runtime_cols<row_spec>;                      // [1, <= 16]
inline constexpr const auto& some_of_rows = runtime_rows<s_spec>;                        // [<= 3, 16]

/// A dataflow buffer as compute_api/dataflow.h needs one: it records the pages it was asked for.
struct Buffer
{
    std::uint32_t waited = 0, reserved = 0, popped = 0, pushed = 0;

    void wait_front(std::uint32_t pages)
    {
        waited += pages;
    }

    void reserve_back(std::uint32_t pages)
    {
        reserved += pages;
    }

    void pop_front(std::uint32_t pages)
    {
        popped += pages;
    }

    void push_back(std::uint32_t pages)
    {
        pushed += pages;
    }

    std::uint32_t get_read_ptr() const
    {
        return 0x100;
    }

    std::uint32_t get_write_ptr() const
    {
        return 0x200;
    }
};

// One definition of a spec with a run-time size, whichever name made it.
static_assert(&runtime_cols<row_spec> == &runtime_size_spec<row_spec, TileX> && &runtime_rows<s_spec> == &runtime_size_spec<s_spec, TileY>);

#if defined(EXPECT_FAIL)
[[maybe_unused]] void must_not_compile(Buffer& buffer, std::uint32_t n)
{
    const Tensor<s_spec> s(0x1000);
#if EXPECT_FAIL == 1
    narrow<Cols, 8, 9>(s); // [3, 16]: 9 + 8 > 16
#elif EXPECT_FAIL == 2
    narrow<Rows, 4>(s, 0); // longer than the tensor
#elif EXPECT_FAIL == 3
    narrow<FaceY, 1>(s, 0); // not a tile-grid dim
#elif EXPECT_FAIL == 4
    narrow_up_to<Cols, 17>(s, 0, n); // a bound above the tensor's size
#elif EXPECT_FAIL == 5
    cols<4>(cols_up_to<8>(s, 0, n), 0); // a view of a view with a run-time length
#elif EXPECT_FAIL == 6
    tile(s, 0); // s is 16 tiles wide
#elif EXPECT_FAIL == 7
    wait_front<row_spec>(buffer, n); // a count for a spec without a run-time size (runtime_cols<> forgotten)
#elif EXPECT_FAIL == 8
    wait_front<some_of_row>(buffer, true); // a when for a spec with a run-time size
#elif EXPECT_FAIL == 9
    reserve_back<row_spec>(buffer, n);
#elif EXPECT_FAIL == 10
    reserve_back<some_of_row>(buffer, n > 0);
#elif EXPECT_FAIL == 11
    pop_front<some_of_row>(buffer); // would pop the bound
#elif EXPECT_FAIL == 12
    wait_front<some_of_row>(buffer); // no count
#elif EXPECT_FAIL == 13
    wait_front<runtime_cols<s_spec> >(buffer, n); // run-time tile columns of several tile rows
#else
#error "EXPECT_FAIL: no such case"
#endif
}
#endif

} // namespace

int main()
{
    int failed                   = 0;
    volatile std::uint32_t three = 3; // genuinely run time
    const std::uint32_t n        = three;
    const Tensor<s_spec> s(0x1000);

    // Inside the tensor: no assert.
    failed += rows<2>(s, n - 2).address != 0x1000 + 16 * tile_bytes(s_spec) || llk_assert_failures != 0;
    failed += cols_up_to<8>(s, 8, n).count != 3 || narrow_up_to<Rows, 2>(s, 1, 2).count != 2 || llk_assert_failures != 0;
    // A run-time start past the end; a length of 0, above its bound, and past the end.
    rows<2>(s, n - 1);
    failed += llk_assert_failures != 1;
    cols_up_to<8>(s, 0, n - 3);
    failed += llk_assert_failures != 2;
    cols_up_to<2>(s, 0, n);
    failed += llk_assert_failures != 3;
    cols_up_to<8>(s, 14, n);
    failed += llk_assert_failures != 4;

    // The buffers: a count's pages follow it, and release pops or pushes the same pages; a when is a bool.
    Buffer in, out;
    const auto x = wait_front<some_of_row>(in, n);      // 3 tiles of a row
    const auto y = wait_front<some_of_rows>(in, n - 1); // 2 tile rows of 16
    const auto z = reserve_back<some_of_row>(out, 5);   // an int is a count too
    const auto w = wait_front<row_spec>(in, n == 0);    // not taken
    const auto v = reserve_back<row_spec>(out, n == 3); // taken
    failed += in.waited != 3 + 32 || out.reserved != 5 + 16 || x.count != 3 || y.count != 2 || z.count != 5;
    release(x);
    release(y);
    release(z);
    release(w);
    release(v);
    failed += in.popped != 3 + 32 || out.pushed != 5 + 16 || llk_assert_failures != 4;
    wait_front<some_of_row>(in, n + 14); // above the bound
    failed += llk_assert_failures != 5;
    return failed;
}
