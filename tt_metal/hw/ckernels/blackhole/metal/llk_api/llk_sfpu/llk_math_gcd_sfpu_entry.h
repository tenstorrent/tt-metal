// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// gcd_tile calls this arch's gcd kernel directly. This header gives the Compute API one include name
// for its entry points on every arch (Quasar adapts its kernel to them in its copy of this header).
#include "ckernel_sfpu_gcd.h"
