#pragma once

#include "tensor/tensor.hpp"

namespace tt {
namespace tt_metal {

Tensor fully_connected(const Tensor &act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias = std::nullopt);

}  // namespace tt_metal
}  // namespace tt
