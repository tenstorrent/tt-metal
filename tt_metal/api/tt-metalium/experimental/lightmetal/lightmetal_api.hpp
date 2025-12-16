// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/experimental/lightmetal/lightmetal_binary.hpp>

namespace tt::tt_metal::experimental::lightmetal {

// clang-format off
/**
 * Begin Light Metal Binary capturing on host and all devices. This will trace host API calls and device (metal trace) workloads to a
 * binary blob returned to caller when tracing is finished, which can later be rerun directly from binary.
 * Note: This LightMetalBinary Trace/Replay feature is currently under active development and is not fully supported, use at own risk.
 *
 * Return value: void
 */
// clang-format on
void LightMetalBeginCapture();

// clang-format off
/**
 * Ends Light Metal Binary capturing on host and all devices returns the binary blob to the user.
 * Note: This LightMetalBinary Trace/Replay feature is currently under active development and is not fully supported, use at own risk.
 *
 * Return value: LightMetalBinary
 */
// clang-format on
LightMetalBinary LightMetalEndCapture();

};  // namespace tt::tt_metal::experimental::lightmetal
