// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_t(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a > b ? a : b); }
void t(const std::uint32_t a, const std::uint32_t b) { TT_SETC16(1, a > b ? a : b); }
