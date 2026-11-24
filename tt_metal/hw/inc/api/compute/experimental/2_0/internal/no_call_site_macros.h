// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Blackhole's call-site macros would take this API's experimental:: calls and operands: a kernel that includes it
// gets none, and its other calls run unmarked, like calls with explicit template arguments.
#define CKERNEL_NO_CALL_SITE_MACROS 1
#undef pack_tile
#undef pack_block
#undef pack_reconfig_data_format
