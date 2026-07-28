// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/debug/dprint.h"

void kernel_main() { DPRINT << "Old style print" << ENDL(); }
