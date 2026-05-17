// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Hello-world Quasar DM kernel. Placeholder for perf scenarios — replace the
// body with the actual transfer logic and use DPRINT to record timing.

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

void kernel_main() { DPRINT << "Hello, World!" << ENDL(); }
