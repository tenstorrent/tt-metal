#pragma once

#include "tensor/tensor.hpp"

namespace tt {
namespace tt_metal {

Tensor fully_connected(const Tensor &act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias = std::nullopt, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

}  // namespace tt_metal
}  // namespace tt
