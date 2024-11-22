#pragma once
#include <memory>

namespace ttml::modules {
class LinearLayer;
}

namespace ttml::models::linear_regression {
[[nodiscard]] std::shared_ptr<ttml::modules::LinearLayer> create(uint32_t in_features, uint32_t out_features);

}  // namespace ttml::models::linear_regression
