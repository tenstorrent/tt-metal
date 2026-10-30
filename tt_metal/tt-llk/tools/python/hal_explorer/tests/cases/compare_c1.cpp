// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t x) { TT_SETC16(5, x); }
void f(const std::uint32_t x) { TT_SETC16(5, (std::uint8_t)x); }
