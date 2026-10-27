// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The threshold kernel lives in tt-llk on this arch. This forwarding header gives the Compute API one
// include name for it on every arch (Quasar's kernel lives in this directory).
#include "sfpu/ckernel_sfpu_threshold.h"
