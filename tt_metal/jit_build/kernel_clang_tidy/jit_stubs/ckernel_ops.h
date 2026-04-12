// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// clang-tidy stub for ckernel_ops.h
//
// The real ckernel_ops.h defines:
//   #define INSTRUCTION_WORD(x)  __asm__ __volatile__(".ttinsn %0" : : "i"((x)))
//
// This causes two classes of errors when analysed with host clang:
//
//   1. "unknown directive"  — .ttinsn is a Tensix-specific assembler directive
//      that neither clang's integrated assembler nor the host system assembler
//      recognises.
//
//   2. "invalid operand for inline asm constraint 'i'" — the "i" constraint
//      requires a compile-time constant.  Some LLK call sites (e.g.
//      ckernel_addrmod.h: TTI_SETC16(addr_mod_src_reg_addr[mod_index], ...))
//      pass a runtime-indexed array element, which the clang frontend rejects
//      before the assembler is even invoked (so -fno-integrated-as doesn't help).
//
// Fix: pull in the real header first to get all the TTI_* macro definitions,
// then redefine INSTRUCTION_WORD to a void cast so every subsequent TTI_* call
// site expands to a harmless expression.
//
// The real header has #pragma once, so this stub is the unique include-guard
// for the "ckernel_ops.h" search path slot (jit_stubs/ comes first).

#pragma once

// Include the real ckernel_ops.h (next in include path, after jit_stubs/).
// This defines TT_OP, INSTRUCTION_WORD, and all the TTI_* macros.
#include_next "ckernel_ops.h"

// Override INSTRUCTION_WORD with a no-op.  Because macros are textually
// substituted at each call site, all TTI_* macros that expand to
// INSTRUCTION_WORD(...) will now emit nothing, silencing both error classes.
#undef INSTRUCTION_WORD
#define INSTRUCTION_WORD(x) ((void)(x))
