// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream phase body for the PMK family composite, NCRISC: the member's PARK credit / VALID loop (H2 is the
// composite's wait before the phase); nothing on the other roles.
#if defined(PMK_ROLE_MEMBER)
#include "pfs_member_rd.cpp"
#else
#include "pfs_common.hpp"
void kernel_main() {}
#endif
