// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_f(const std::uint32_t x) { TT_SETC16(1, (((((((((((((((((((((x ^ 1) + 11u) ^ x) * 3u) ^ 8) + 13u) ^ x) * 3u) ^ 10) + 15u) ^ x) * 3u) ^ 12) + 17u) ^ x) * 3u) ^ 14) + 19u) ^ x) * 3u) ^ 16)); }
void f(const std::uint32_t x) { TT_SETC16(1, (((((((((((((((((((((x ^ 2) + 11u) ^ x) * 3u) ^ 8) + 13u) ^ x) * 3u) ^ 10) + 15u) ^ x) * 3u) ^ 12) + 17u) ^ x) * 3u) ^ 14) + 19u) ^ x) * 3u) ^ 16)); }
