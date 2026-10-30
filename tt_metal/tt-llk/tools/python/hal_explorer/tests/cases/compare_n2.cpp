// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_o(int a) { TTI_NOP; }
void o(int a) { TTI_STALLWAIT(1,2); }
void reference_o(float a) { TTI_NOP; }
void o(float a) { TTI_NOP; }
namespace ns {
void reference_x() { TTI_NOP; }
void x() { TTI_STALLWAIT(1,2); }
}
