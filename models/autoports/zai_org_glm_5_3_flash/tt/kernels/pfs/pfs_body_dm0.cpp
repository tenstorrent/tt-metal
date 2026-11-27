// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a PfStream phase body for the PMK family composite (tt/pmk_layer.PhaseBody: ONE source per RISC for every role
// of the phase; the composite defines PMK_ROLE_<role>): BRISC = the member's output store (+ H5 raise) on members, the
// bank streamer on every other role that hosts streamers (a core without a stream in this phase has RT_ACTIVE 0 and
// returns at once). CT / RT: pfs_common.hpp (the phase's parts() lists).
#if defined(PMK_ROLE_MEMBER)
#include "pfs_member_wr.cpp"
#else
#include "pfs_stream.cpp"
#endif
