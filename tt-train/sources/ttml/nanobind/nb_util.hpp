// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/ndarray.h>

#include "nb_bfloat16.hpp"
#include "nb_fwd.hpp"
#include "ttnn/tensor/tensor.hpp"

nb::ndarray<nb::numpy> make_numpy_tensor(
    const tt::tt_metal::Tensor& tensor, std::optional<tt::tt_metal::DataType> new_type = std::nullopt);
tt::tt_metal::Tensor make_metal_tensor(
    nb::ndarray<> data, std::optional<tt::tt_metal::DataType> new_type = std::nullopt);
