// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Low-priority copy. Registered FIRST, so add_include_dirs()' prepend
// behaviour must push include_high/ ahead of it. If the driver ever sees
// OOT_PROBE_ID == 1, search-dir ordering regressed.

#pragma once

#define OOT_PROBE_ID 1
