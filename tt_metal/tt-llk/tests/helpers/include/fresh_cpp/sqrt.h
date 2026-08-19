// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Canonical semantic body for the sqrt op (storm contract:
// fresh_cpp/README.md).  The body is the SHARED sqrt/rsqrt template
// calculate_sqrt_rsqrt_fresh_cpp<RECIPROCAL> — the production kernels share
// theirs the same way — and its single definition lives in
// fresh_cpp/rsqrt.h (storm cross-lane convention, Lane S4's migration of
// the same Lane BR batch-2 body; RECIPROCAL=false selects the sqrt arm).
#include "fresh_cpp/rsqrt.h"
