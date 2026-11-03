// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The standard headers of hostdev/api. Every header of the folder is compiled by the device compiler and by host
// compilers, inside and outside tt-metal, so what it includes every device kernel that uses it pays for. A header of
// the folder therefore includes only other headers of the folder, and gets the standard library from here alone.
// This list is closed: scripts/validate_api/validate_hostdev_api.py, run by pre-commit, rejects any other include.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <type_traits>
#include <utility>
