// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// word = constant + runtime vs constant | runtime: differ when the runtime value overlaps constant bits
void reference_a(const std::uint32_t x) { TT_INSN(0xb2050000u + x); }
void a(const std::uint32_t x) { TT_INSN(0xb2050000u | x); }
void reference_b(const std::uint32_t x) { TT_INSN(0xb2050000u + x); }
void b(const std::uint32_t x) { TT_INSN(0xb2050000u ^ x); }
void reference_c(const std::uint32_t x) { TT_INSN(0xb2050000u + x); }
void c(const std::uint32_t x) { TT_INSN(0xb2050000u - x); }
