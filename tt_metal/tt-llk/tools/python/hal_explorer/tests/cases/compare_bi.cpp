// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

void reference_a() { __builtin_rvtt_ttincrwc(0, 1, 2, 3); }
void a() { __builtin_rvtt_ttincrwc(1, 1, 2, 3); }
void reference_b() { __builtin_rvtt_ttincrwc(0, 1, 2, 3); }
void b() { }
