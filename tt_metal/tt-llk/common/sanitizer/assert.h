#pragma once

#include <cstring>

#include "ckernel.h"
#include "sanitizer/output.h"
#include "sanitizer/types.h"

namespace llk::san
{

NOINLINE NOCLONE void _print_full_kernel()
{
    const ct_string kernel = CTSTR(FULL_KERNEL_NAME);
    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Current Kernel ]─\r"
        "│  └── {}\r",
        kernel);
}

template <typename T>
NOINLINE NOCLONE void _print_operand_expected(const State<T> expected)
{
    if (expected.is_known())
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Last time state was modified ]─\r"
            "│  ├── New state value ── {}\r",
            expected.get_underlying());
    }
    else if (expected.is_unknown())
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Last time state was modified ]─\r"
            "│  ├── New state value ── UNKNOWN (value never configured?)\r");
    }
    else
    {
        __builtin_unreachable();
    }
}

template <typename T>
NOINLINE NOCLONE void _print_operand_actual(const State<T> actual)
{
    if (actual.is_known())
    {
        DEVICE_PRINT(
            "│\r"
            "│  ┌[ Failed operand state check ]─\r"
            "│  ├── Provided state value ── {}\r",
            actual.get_underlying());
    }
    else
    {
        __builtin_unreachable();
    }
}

NOINLINE NOCLONE void _print_compute_info(const UnwindContext context)
{
    const ct_string unknown = CTSTR("<unknown>");
    const ct_string file    = CTSTR("<file>");
    const ct_string line    = CTSTR("<line>");

    DEVICE_PRINT(
        "│  ├── Compute API ─┬ {}\r"
        "│  │                └ {}:{}\r"
        "│  └── Callsite ────┬ {}\r"
        "│                   └ {}:{}\r",
        DevicePrintTopCallstack<2>(context),  // pc func first
        DevicePrintTopCallstack<0>(context),  // pc file
        DevicePrintTopCallstack<1>(context),  // pc line
        DevicePrintTopCallstack<5>(context),  // ra func first
        DevicePrintTopCallstack<3>(context),  // ra file
        DevicePrintTopCallstack<4>(context)); // ra line
}

template <typename T>
TT_ALWAYS_INLINE void operand_assert(const State<T> expected, const State<T> actual, ct_string message, const UnwindContext update, const UnwindContext current)
{
    if (expected.assert_cond(actual))
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ error ]──────\r"
        "│  {}\r",
        message);

    _print_full_kernel();
    _print_operand_expected(expected);
    _print_compute_info(update);
    _print_operand_actual(actual);
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");
}

NOINLINE NOCLONE void operation_assert(const Operation expected, const Operation actual, const UnwindContext update, const UnwindContext current)
{
    if (expected == actual)
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ error ]──────\r"
        "│  Called execute or uninit for operation that is not initialized\r");

    _print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Last init call ]─\r"
        "│  ├── Operation ── {}\r",
        expected);
    _print_compute_info(update);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ Violating execute / uninit call ]─\r"
        "│  ├── Operation ── {}\r",
        actual);
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");
}

NOINLINE NOCLONE void operation_argument_assert(
    const void* lhs, const void* rhs, size_t size, size_t idx, const UnwindContext update, const UnwindContext current)
{
    if (std::memcmp(lhs, rhs, size) == 0)
    {
        return;
    }

    DEVICE_PRINT(
        "┌─[ llk::san ]─[ error ]──────\r"
        "│  Argument {} of llk::san::operation_init and llk::san::operation_check is mismatched\r",
        idx);

    _print_full_kernel();

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ llk::san::operation_init called from ]─\r");
    _print_compute_info(update);

    DEVICE_PRINT(
        "│\r"
        "│  ┌[ llk::san::operation_check called from ]─\r");
    _print_compute_info(current);

    DEVICE_PRINT("└─────────────────────────────\n");
}

}; // namespace llk::san
