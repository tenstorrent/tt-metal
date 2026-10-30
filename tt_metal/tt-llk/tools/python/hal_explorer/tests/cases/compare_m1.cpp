// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// MMIO access width and sub-word offsets
void reference_w(const std::uint32_t v) { *reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE) = v; }
void w(const std::uint32_t v)           { *reinterpret_cast<volatile std::uint8_t*>(TENSIX_MOP_CFG_BASE) = 1; }
void reference_x(volatile std::uint32_t* p) { p[0] = 1; }
void x(volatile std::uint32_t* p)           { *reinterpret_cast<volatile std::uint8_t*>(p) = 1; }
void reference_y(volatile std::uint8_t* p) { p[0] = 1; }
void y(volatile std::uint8_t* p)           { p[1] = 1; }
void reference_z(volatile std::uint16_t* p) { *p = 1; }
void z(volatile std::uint32_t* p)           { *p = 1; }
