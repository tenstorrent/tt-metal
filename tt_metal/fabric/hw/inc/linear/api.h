// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(ARCH_WORMHOLE) || defined(ARCH_BLACKHOLE)
#include "tt_metal/fabric/hw/inc/linear/tt-1xx/api_impl.h"
#else
#include "tt_metal/fabric/hw/inc/linear/tt-2xx/quasar_1/api_impl.h"
#endif
