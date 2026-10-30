// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Count right; mnemonic/word lost: tool prints "op 0x00  0x00000000 + rt" (real: SETRWC 0x37d00006 + runtime fields).
void next_face(std::uint32_t b, std::uint32_t d) { TT_SETRWC(0x3, 0x4, d, b, 0, 0x6); }
