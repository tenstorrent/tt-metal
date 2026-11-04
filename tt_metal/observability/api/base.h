// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The standard headers of observability/api. Every header of the folder is compiled by host compilers, inside and
// outside tt-metal, and by the device compiler from the copy the JIT build stages, so what it includes every device
// kernel that uses it pays for. A header of the folder therefore includes only other headers of the folder, and gets
// the standard library from here alone. This list is closed: scripts/validate_api/validate_observability_api.py, run
// by pre-commit, rejects any other include.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>
