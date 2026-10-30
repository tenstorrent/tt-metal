// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// A: constant address, different access width, same value text
void reference_a() { *reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE) = 1; }
void a()           { *reinterpret_cast<volatile std::uint8_t*>(TENSIX_MOP_CFG_BASE) = 1; }
// B: runtime base, sub-word offsets 2B vs 3B (offset // 4)
void reference_b(volatile std::uint8_t* p) { p[2] = 1; }
void b(volatile std::uint8_t* p)           { p[3] = 1; }
// C: direct deref of volatile pointer param is not recognised as MMIO at all
void reference_c(volatile std::uint32_t* p) { *p = 1; *p = 2; }
void c(volatile std::uint32_t* p)           { *p = 7; }
// D: constant base but 16-bit vs 32-bit at same address
void reference_d() { *reinterpret_cast<volatile std::uint16_t*>(TENSIX_MOP_CFG_BASE + 2) = 1; }
void d()           { *reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE + 0) = 1; }
