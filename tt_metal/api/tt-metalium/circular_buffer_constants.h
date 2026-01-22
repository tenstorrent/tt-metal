// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Circular buffer constants are shared between host and device code.
// The canonical location is hostdevcommon/circular_buffer_constants.h
// This header re-exports them for backwards compatibility with host code.
#include <hostdevcommon/circular_buffer_constants.h>
