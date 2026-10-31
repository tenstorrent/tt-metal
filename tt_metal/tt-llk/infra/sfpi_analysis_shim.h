// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Compatibility shim for analysing JIT-compiled kernels with clang.
//
// SFPI's `tensix_builtins.h` deliberately provides declarations for compilers that do not
// implement the Tensix builtins ("so that code analysis tools will not just give up in the face of
// an unknown function call"), but the fallback omits `__xtt_vector8`, which
// `tensix_builtins.def` uses for `__builtin_rvtt_sfptransp` and `__builtin_rvtt_sfpselect8`.
// Without the alias clang reports `unknown type name '__xtt_vector8'` twice and then recovers,
// which is harmless for code generation but poisons the AST the checker reads.
//
// The alias is declared only on that fallback path (the same condition `tensix_builtins.h` uses),
// so a real SFPI compiler is untouched. Redefining an alias to the same type is legal C++, so an
// SFPI release that starts providing the alias does not conflict with this file.
//
// This is a workaround in the analysis harness, not a stub of the API: no compute or LLK symbol is
// declared here.
#pragma once

#if !defined(__riscv_xtttensixwh) && !defined(__riscv_xtttensixbh) && !defined(__riscv_xtttensixqsr)
#if !__has_builtin(__builtin_rvtt_synth_opcode)
using __xtt_vector8 = unsigned __attribute__((vector_size(sizeof(unsigned) * 32 * 8)));
#endif
#endif
