// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_a() { *reinterpret_cast<volatile std::uint8_t*>(TENSIX_CFG_BASE + 4*184 + 1) = 1; }
void a() { *reinterpret_cast<volatile std::uint8_t*>(TENSIX_CFG_BASE + 4*184 + 2) = 1; }
void reference_b() { *reinterpret_cast<volatile std::uint32_t*>(TENSIX_CFG_BASE + 4*184) = 1; }
void b() { *reinterpret_cast<volatile std::uint8_t*>(TENSIX_CFG_BASE + 4*184) = 1; }
