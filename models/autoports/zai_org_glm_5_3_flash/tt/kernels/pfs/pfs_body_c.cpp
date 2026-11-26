// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream phase body for the PMK family composite, TRISC: the member's matmul (half-sync DST, NB <= 4: the
// members' composite compute config); nothing on the other roles.
#if defined(PMK_ROLE_MEMBER)
#include "pfs_member_c.cpp"
#else
void kernel_main() {}
#endif
