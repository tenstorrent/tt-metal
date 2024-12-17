// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "linear_regression.hpp"

#include "modules/linear_module.hpp"

namespace ttml::models::linear_regression {
std::shared_ptr<ttml::modules::LinearLayer> create(uint32_t in_features, uint32_t out_features) {
    return std::make_shared<ttml::modules::LinearLayer>(in_features, out_features);
}
}  // namespace ttml::models::linear_regression
