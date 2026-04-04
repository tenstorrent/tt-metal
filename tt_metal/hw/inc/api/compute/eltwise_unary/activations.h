// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Aggregation header for activation function compute APIs.
// Individual activation headers are included here so that compute kernels
// can pull in all activations with a single #include.

#pragma once

#include "api/compute/eltwise_unary/hardsigmoid.h"
