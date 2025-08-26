// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/eth_link_status.h"

void kernel_main() { WATCHER_CHECK_ETH_LINK_STATUS(); }
