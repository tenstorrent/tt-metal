// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/xtensor/conversion_utils.hpp"
#include "ttnn/tensor/xtensor/partition.hpp"
#include "ttnn/tensor/xtensor/xtensor_all_includes.hpp"

namespace ttnn {
namespace {

using ::tt::tt_metal::Tensor;
using ::ttnn::experimental::xtensor::chunk;
using ::ttnn::experimental::xtensor::concatenate;
using ::ttnn::experimental::xtensor::from_vector;

}  // namespace
}  // namespace ttnn
