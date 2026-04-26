// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Stub for JIT-generated chlkc_math.cpp (clang-tidy analysis only)
// At JIT time this file wraps the compute kernel source, so kernel_main()
// is defined before chlkc_list.h::run_kernel() references it.
// Forward-declare it here to replicate that ordering.
void kernel_main();
namespace chlkc_math {
void math_main() {}
}  // namespace chlkc_math
