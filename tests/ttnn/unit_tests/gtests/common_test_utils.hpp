// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

namespace test_utils {

// Pearson Correlation Coefficient for two float vectors
float pcc(const std::vector<float>& x, const std::vector<float>& y);

}  // namespace test_utils
