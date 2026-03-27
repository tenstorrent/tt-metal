// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "randn_op.hpp"

#include <cmath>
#include <ttnn/distributed/types.hpp>

#include "autograd/auto_context.hpp"
#include "autograd/tensor.hpp"
#include "rand_op.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/uniform/uniform.hpp"

namespace ttml::ops {

autograd::TensorPtr randn(
    const ttnn::Shape& shape,
    float mean,
    float stddev,
    std::optional<uint32_t> seed,
    tt::tt_metal::DataType dtype,
    tt::tt_metal::Layout layout) {
    TT_FATAL(stddev >= 0.0f, "[ttml::ops::randn] stddev must be non-negative, got {}.", stddev);

    auto* device = &autograd::ctx().get_device();
    ttnn::MemoryConfig mem_config{};

    auto& gen = autograd::ctx().get_generator();
    uint32_t u1_seed = detail::avoid_zero_seed(seed.value_or(gen()));

    // ttnn::uniform assigns seed + core_index per core, so offset u2's seed
    // by at least num_cores to avoid overlapping LFSR sequences.
    auto grid = device->compute_with_storage_grid_size();
    const uint32_t seed_gap = grid.x * grid.y;
    uint32_t u2_seed = detail::avoid_zero_seed(u1_seed + seed_gap);

    // Box-Muller transform: z = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
    // u1 in (eps, 1] to avoid log(0); u2 in [0, 1)
    auto u1 = ttnn::empty(shape, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, device, mem_config);
    ttnn::uniform(u1, 1e-6f, 1.0f, u1_seed);
    auto u2 = ttnn::empty(shape, tt::tt_metal::DataType::FLOAT32, tt::tt_metal::Layout::TILE, device, mem_config);
    ttnn::uniform(u2, 0.0f, 1.0f, u2_seed);

    auto log_u1 = ttnn::log(u1, false, mem_config);
    auto neg2log = ttnn::multiply(log_u1, -2.0f, std::nullopt, mem_config);
    auto r = ttnn::sqrt(neg2log, false, mem_config);

    auto two_pi_u2 = ttnn::multiply(u2, static_cast<float>(2.0 * M_PI), std::nullopt, mem_config);
    auto cos_u2 = ttnn::cos(two_pi_u2, mem_config);

    auto z = ttnn::multiply(r, cos_u2, std::nullopt, mem_config);

    if (stddev != 1.0f) {
        z = ttnn::multiply(z, stddev, std::nullopt, mem_config);
    }
    if (mean != 0.0f) {
        z = ttnn::add(z, mean, std::nullopt, mem_config);
    }

    if (dtype != tt::tt_metal::DataType::FLOAT32) {
        z = ttnn::typecast(z, dtype);
    }
    if (layout != tt::tt_metal::Layout::TILE) {
        z = ttnn::to_layout(z, layout);
    }
    return ttml::autograd::create_tensor(z);
}

}  // namespace ttml::ops
