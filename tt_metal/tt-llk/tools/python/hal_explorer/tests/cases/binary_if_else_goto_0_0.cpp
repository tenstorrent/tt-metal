// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// GCC emits "[0:0] goto <bb N>" (no source location); LOCATION regex in gimple.py does not strip it,
// so the then-arm falls through into the else-arm block.  Real counts: a&&b=3, a&&!b=2, !a&&b=3, !a&&!b=2.
// Tool reports a&&b=4, a&&!b=3.
void dup_paths(const std::uint32_t a, const std::uint32_t b)
{
    if (a) { TTI_NOP; } else { TTI_SETC16(1, 2); }
    TT_SETC16(a, b);
    if (b) { TTI_NOP; }
}
