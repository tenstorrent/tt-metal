#pragma once

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
        "│  ├── Compute API ─┬ {:#x}\r"
        "│  │                └ {}:{}\r"
        "│  └── Callsite ────┬ {:#x}\r"
        "│                   └ {}:{}\r",
        context.pc,
        context.pc != UINTPTR_MAX ? file : unknown,
        context.pc != UINTPTR_MAX ? line : unknown,
        context.ra,
        context.ra != UINTPTR_MAX ? file : unknown,
        context.ra != UINTPTR_MAX ? line : unknown);
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

}; // namespace llk::san
