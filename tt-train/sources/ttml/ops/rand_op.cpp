// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rand_op.hpp"

#include <limits>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "ttnn/operations/rand/rand.hpp"

namespace ttml::ops {

namespace {
// ttnn::rand treats seed==0 as "no seed" (uses random entropy).
// Shift by +1 to keep all seeds deterministic. UINT32_MAX maps to 1.
uint32_t avoid_zero_seed(uint32_t seed) {
    return seed == std::numeric_limits<uint32_t>::max() ? 1 : seed + 1;
}
}  // namespace

autograd::TensorPtr rand(
    const ttnn::Shape& shape,
    float low,
    float high,
    std::optional<uint32_t> seed,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout) {
    auto* device = &autograd::ctx().get_device();
    ttnn::MemoryConfig mem_config{};

    uint32_t effective_seed = avoid_zero_seed(seed.value_or(autograd::ctx().get_generator()()));

    auto t = ttnn::rand(shape, *device, dtype, layout, mem_config, low, high, effective_seed);
    return ttml::autograd::create_tensor(t);
}

}  // namespace ttml::ops
