// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_reference_r() { TTI_NOP; }
void reference_r() { TTI_NOP; }
void r() { TTI_STALLWAIT(1,2); }
