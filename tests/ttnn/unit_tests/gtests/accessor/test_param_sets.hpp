// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared param sets for `test_tensor_accessor_on_device.cpp` (SW TensorAccessor on-device tests).

#include <optional>
#include <vector>

#include <tt-metalium/shape.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>

#include "hostdevcommon/tensor_accessor/arg_config.hpp"

namespace tensor_accessor_test_params {

// Reshard params: input -> output with potentially different shard specs.
// Used by ShardedAccessorTestsReshardOnDevice and copy fixtures in on_device.cpp.
// The `crta_config` field controls static-vs-runtime CTA placement for the SW path.
struct InputOutputBufferParams {
    tt::tt_metal::Shape tensor_shape;
    tt::tt_metal::Layout layout;
    tt::tt_metal::DataType dtype;
    tt::tt_metal::BufferType input_buffer_type;
    tt::tt_metal::BufferType output_buffer_type;

    std::optional<tt::tt_metal::NdShardSpec> input_shard_spec;
    std::optional<tt::tt_metal::NdShardSpec> output_shard_spec;
    tensor_accessor::ArgsConfig crta_config;
};

// Copy params: single-tensor copy. `input_shard_spec` is nullopt for interleaved cases.
struct CopyParams {
    tt::tt_metal::Shape tensor_shape;
    tt::tt_metal::Layout layout;
    tt::tt_metal::DataType dtype;
    tt::tt_metal::BufferType buffer_type;

    std::optional<tt::tt_metal::NdShardSpec> input_shard_spec;
};

// 9 base reshard configurations. on_device.cpp expands these over the CRTA matrix
// and interleaved-swap toggles.
std::vector<InputOutputBufferParams> get_sharded_reshard_base_params();

// 16-entry interleaved copy params (2D / 3D / 4D / higher-rank, L1 + DRAM).
std::vector<CopyParams> get_interleaved_copy_params();

// ~17-entry sharded copy params (2D / 3D / N-D, mostly L1, mostly multi-core).
std::vector<CopyParams> get_sharded_copy_params();

}  // namespace tensor_accessor_test_params
