// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dev_mem_map.h"
// This is to support some deprecated mappings for versim to get build up
#ifndef TT_METAL_VERSIM_DISABLED
#define TEST_MAILBOX_ADDRESS MEM_MAILBOX_BASE
#endif
