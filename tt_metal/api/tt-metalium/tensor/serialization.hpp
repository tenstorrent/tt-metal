// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tensor/tensor.hpp>

#include <string>

namespace tt::tt_metal {

Tensor load_tensor_flatbuffer(const std::string& file_name, distributed::MeshDevice* device = nullptr);

}  // namespace tt::tt_metal
