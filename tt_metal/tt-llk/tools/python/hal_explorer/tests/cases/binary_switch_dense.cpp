// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// OBJFLAGS: -fno-jump-tables
// Tool says: 1 path, count 2, "when always".  Binary: 1+0..4 NOPs depending on m (jump table).
void sw(const std::uint32_t m)
{
    switch (m)
    {
        case 0: TTI_NOP; break;
        case 1: TTI_NOP; TTI_NOP; break;
        case 2: TTI_NOP; TTI_NOP; TTI_NOP; break;
        case 3: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
        case 4: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
        case 5: TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; TTI_NOP; break;
    }
    TTI_SETC16(3, 3);
}
