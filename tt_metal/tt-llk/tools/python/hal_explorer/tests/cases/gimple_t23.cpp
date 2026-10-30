// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void neg1(int d) { TT_RMWCIB0(0xff, d, 3); }
void neg2(unsigned d) { TT_RMWCIB0(0xff, d, 3); }
void neg3(int d) { TT_SETDMAREG(0, d, 0, 0); }
void neg4(int d, int e) { TT_RMWCIB0(e, d, 3); }
