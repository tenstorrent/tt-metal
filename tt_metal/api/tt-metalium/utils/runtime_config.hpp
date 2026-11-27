// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace tt::tt_metal {

// clang-format off
/**
 * Sets the root directory for TT Metal meta data files like kernel sources.
 *
 * Return value: void
 *
 * | Argument  | Description                                 | Type                | Valid range | Required |
 * |-----------|---------------------------------------------|---------------------|-------------|----------|
 * | root_dir  | Path to the root directory                  | const std::string & |             | No       |
 */
// clang-format on
void SetRootDir(const std::string& root_dir);

}  // namespace tt::tt_metal
