// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/buffer_distribution_spec.hpp>

namespace tt::tt_metal::detail {

UncompressedBufferPageMapping compute_page_mapping(
    const Shape& tensor_shape, const Shape& shard_shape, const std::vector<CoreCoord>& cores);

};
