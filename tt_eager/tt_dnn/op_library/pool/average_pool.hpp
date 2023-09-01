/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"


namespace tt {
namespace tt_metal {

enum class PoolType {
    AVG = 0
};

Tensor average_pool_2d(const Tensor& input, std::optional<MemoryConfig> output_mem_config = std::nullopt);

}  // namespace tt_metal
}  // namespace tt
