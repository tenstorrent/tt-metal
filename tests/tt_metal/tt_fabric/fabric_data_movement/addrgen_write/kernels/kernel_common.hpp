// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Enum definitions for compile-time parameters
enum class OperationType : std::uint8_t { BasicWrite = 0, Scatter = 1, FusedAtomicInc = 2 };

enum class ApiVariant : std::uint8_t { Basic = 0, WithState = 1, SetState = 2 };
