// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_l(const std::uint32_t n) { volatile std::uint32_t* r = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE); for (std::uint32_t i = 0; i < n; i++) r[i] = 1; }
void l(const std::uint32_t n)           { volatile std::uint32_t* r = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE); for (std::uint32_t i = 0; i < n; i++) r[2 * i] = 1; }
// value varies per iteration
void reference_m(const std::uint32_t n) { volatile std::uint32_t* r = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE); for (std::uint32_t i = 0; i < n; i++) r[0] = i; }
void m(const std::uint32_t n)           { volatile std::uint32_t* r = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE); for (std::uint32_t i = 0; i < n; i++) r[0] = 0; }
