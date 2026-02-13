// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Benchmark & correctness test: Original Conv2d (ProgramFactoryConcept)
// vs Conv2dNew (MeshWorkloadFactoryConcept with ProgramDescriptor).
//
// Replicates the conv2d_L1 preparation pipeline, then calls both
// prim::conv2d (old) and prim::conv2d_new (new) to get a direct
// apples-to-apples comparison at the device-operation level.

#include <chrono>
#include <iostream>

#include "gtest/gtest.h"
#include "ttnn_test_fixtures.hpp"

#include <tt-metalium/distributed.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/operations/conv/conv2d/conv2d.hpp"
#include "ttnn/operations/conv/conv2d/conv2d_utils.hpp"
#include "ttnn/operations/conv/conv2d/prepare_conv2d_weights.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation.hpp"
#include "ttnn/operations/conv/conv2d/device/conv2d_device_operation_types.hpp"
#include "ttnn/operations/conv2d_new/device/conv2d_new_device_operation.hpp"
#include "ttnn/operations/sliding_window/halo/halo.hpp"
#include "ttnn/operations/sliding_window/sliding_window.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/move/move.hpp"

namespace {

using Clock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::milli>;
using namespace ttnn::operations::conv;
using namespace ttnn::operations::conv::conv2d;
using ttnn::DeviceComputeKernelConfig;
using ttnn::MeshDevice;
namespace sliding_window = ttnn::operations::sliding_window;

class Conv2dDescriptorBenchmark : public ttnn::TTNNFixtureWithDevice {
public:
    // Conv2d needs L1_SMALL for config tensors (halo indices etc.)
    Conv2dDescriptorBenchmark() : TTNNFixtureWithDevice(DEFAULT_TRACE_REGION_SIZE, /*l1_small_size=*/16384) {}
};

// -----------------------------------------------------------------------
// Helper: prepared inputs for calling prim::conv2d / prim::conv2d_new
// -----------------------------------------------------------------------
struct PreparedConv2dPrimInputs {
    ttnn::Tensor input_tensor_post_tm;
    ttnn::Tensor weight_tensor_on_device;
    std::optional<ttnn::Tensor> bias_tensor_on_device;
    sliding_window::SlidingWindowConfig sliding_window_config;
    Conv2dParallelizationConfig parallel_cfg;
    Conv2dBlockConfig block_cfg;
    MemoryConfig conv_out_memory_config;
    DataType output_dtype;
    DeviceComputeKernelConfig compute_config;
    Conv2dConfig conv_config;
    std::array<uint32_t, 4> input_tensor_shape;
    uint32_t out_channels;
    uint32_t groups;
};

// Replicates the conv2d_L1 preparation pipeline from conv2d.cpp.
// Input tensor should be NHWC on-device or host.
PreparedConv2dPrimInputs prepare_conv2d_prim_inputs(
    const ttnn::Tensor& input_tensor_,
    const ttnn::Tensor& weight_tensor_,
    MeshDevice* device,
    uint32_t in_channels,
    uint32_t out_channels,
    uint32_t batch_size,
    uint32_t input_height,
    uint32_t input_width,
    std::array<uint32_t, 2> kernel_size,
    std::array<uint32_t, 2> stride,
    std::array<uint32_t, 2> padding,
    std::array<uint32_t, 2> dilation,
    uint32_t groups) {
    Conv2dConfig conv_config{};
    const DataType output_dtype = input_tensor_.dtype();
    std::array<uint32_t, 4> padding_n4 = sliding_window::get_pair_n4_padding(padding);
    const auto& weight_tensor = weight_tensor_;
    bool mm_conv = use_matmul_for_1x1_conv(kernel_size, stride, padding_n4, dilation, groups, conv_config);
    auto orig_stride = stride;

    auto input_tensor = fold_input_tensor_if_required(
        input_tensor_,
        device,
        batch_size,
        input_height,
        input_width,
        in_channels,
        kernel_size,
        stride,
        dilation,
        padding_n4,
        mm_conv,
        conv_config);

    auto [output_height, output_width] =
        calculate_output_image_size({input_height, input_width}, kernel_size, stride, padding_n4, dilation);

    DataType weight_dtype = conv_config.weights_dtype.value_or(weight_tensor_.dtype());
    DeviceComputeKernelConfig compute_config =
        get_conv_default_compute_kernel_config(device, input_tensor_.dtype(), weight_dtype);

    const auto compute_grid_size = device->compute_with_storage_grid_size();

    bool auto_shard = false;
    if (!input_tensor.is_sharded() && !conv_config.shard_layout.has_value()) {
        if (!conv_config.weights_dtype.has_value()) {
            conv_config.weights_dtype = weight_tensor.dtype();
        }
        conv_config = determine_conv_config_for_auto_shard(
            conv_config,
            mm_conv,
            batch_size,
            in_channels,
            out_channels,
            output_height,
            output_width,
            kernel_size[1],
            input_height,
            input_width,
            compute_grid_size,
            input_tensor.layout(),
            input_tensor.dtype(),
            output_dtype,
            tt::tt_metal::is_device_tensor(input_tensor) ? std::make_optional(input_tensor.memory_config())
                                                         : std::nullopt,
            kernel_size,
            stride,
            dilation,
            padding_n4,
            groups,
            false /*no bias*/,
            compute_config);
        auto_shard = true;
    }

    auto [input_tensor_post_tm, parallel_config, output_parallel_config] = shard_or_reshard_tensor_if_required(
        device,
        input_tensor,
        conv_config,
        batch_size,
        output_height,
        output_width,
        in_channels,
        out_channels,
        mm_conv,
        auto_shard);

    const uint32_t input_channels_alignment = get_input_channels_alignment(
        input_tensor_post_tm.memory_config().memory_layout(),
        input_tensor_post_tm.layout(),
        false,
        mm_conv,
        input_tensor_post_tm.memory_config());
    const uint32_t in_channels_padded = tt::round_up(
        in_channels, get_num_cores_channels_from_parallel_config(parallel_config) * input_channels_alignment);

    const bool conv_is_1d_depthwise = is_1d_depthwise_conv(
        groups, in_channels, out_channels, kernel_size[0], kernel_size[1], input_height, false /*no bias*/);

    auto [opt_conv_op_parallel_config, opt_conv_op_block_config, conv_out_memory_config] = get_conv_configs(
        conv_config,
        compute_config,
        parallel_config,
        output_parallel_config,
        in_channels_padded,
        out_channels,
        batch_size,
        output_height,
        output_width,
        kernel_size,
        compute_grid_size,
        conv_is_1d_depthwise);

    // Prepare weights
    ttnn::Tensor weight_tensor_on_device = weight_tensor;
    Conv2dWeightsBiasPrepConfig prep_params(
        input_channels_alignment,
        conv_config.weights_dtype,
        opt_conv_op_block_config.act_block_w_ntiles,
        opt_conv_op_block_config.out_subblock_w_ntiles,
        parallel_config,
        output_parallel_config,
        groups,
        opt_conv_op_block_config.act_block_h_ntiles,
        input_height,
        input_width,
        mm_conv && auto_shard,
        out_channels,
        false /*no bias*/,
        conv_config.enable_kernel_stride_folding.value(),
        conv_config.full_inner_dim,
        conv_config.enable_activation_reuse,
        orig_stride);

    if (!tt::tt_metal::is_device_tensor(weight_tensor)) {
        std::tie(weight_tensor_on_device, std::ignore) =
            prepare_conv_weights_biases_and_move_to_device(weight_tensor, std::nullopt, prep_params, device);
    }

    // Halo
    TT_ASSERT(!mm_conv, "This benchmark only supports non-1x1 convolutions");

    sliding_window::SlidingWindowConfig sliding_window_config{
        .batch_size = batch_size,
        .input_hw = {input_height, input_width},
        .window_hw = {kernel_size[0], kernel_size[1]},
        .stride_hw = {stride[0], stride[1]},
        .padding = {{padding_n4[0], padding_n4[1], padding_n4[2], padding_n4[3]}},
        .dilation_hw = {dilation[0], dilation[1]},
        .num_cores_nhw = opt_conv_op_parallel_config.num_cores_nhw,
        .core_range_set = input_tensor_post_tm.memory_config().shard_spec().value().grid,
        .snap_to_tile = true,
    };

    if (parallel_config.shard_scheme != TensorMemoryLayout::WIDTH_SHARDED ||
        input_tensor_post_tm.layout() != Layout::ROW_MAJOR || sliding_window_config.get_pad_h() != 0 ||
        sliding_window_config.get_pad_w() != 0) {
        ttnn::Tensor halo_output = ttnn::halo(
            input_tensor_post_tm,
            sliding_window_config,
            0,
            false,
            parallel_config.shard_orientation == ShardOrientation::COL_MAJOR,
            input_tensor_post_tm.memory_config(),
            true,
            conv_config.config_tensors_in_dram);

        if (conv_config.deallocate_activation && !input_tensor_post_tm.memory_config().is_dram()) {
            input_tensor_post_tm.deallocate(/*force*/ true);
        }
        input_tensor_post_tm = std::move(halo_output);
        if (conv_config.reallocate_halo_output) {
            input_tensor_post_tm = ttnn::move(input_tensor_post_tm);
        }
    }

    const std::array<uint32_t, 4> input_tensor_shape = {batch_size, input_height, input_width, in_channels};

    return PreparedConv2dPrimInputs{
        .input_tensor_post_tm = std::move(input_tensor_post_tm),
        .weight_tensor_on_device = std::move(weight_tensor_on_device),
        .bias_tensor_on_device = std::nullopt,
        .sliding_window_config = sliding_window_config,
        .parallel_cfg = opt_conv_op_parallel_config,
        .block_cfg = opt_conv_op_block_config,
        .conv_out_memory_config = conv_out_memory_config,
        .output_dtype = output_dtype,
        .compute_config = compute_config,
        .conv_config = conv_config,
        .input_tensor_shape = input_tensor_shape,
        .out_channels = out_channels,
        .groups = groups,
    };
}

// Calls prim::conv2d (old factory)
Tensor call_old_prim(const PreparedConv2dPrimInputs& p) {
    return ttnn::prim::conv2d(
        p.input_tensor_post_tm,
        p.weight_tensor_on_device,
        p.bias_tensor_on_device,
        p.sliding_window_config,
        p.out_channels,
        p.groups,
        p.conv_config.output_layout == Layout::ROW_MAJOR,
        p.conv_config.activation,
        p.parallel_cfg,
        p.block_cfg,
        p.conv_out_memory_config,
        p.output_dtype,
        p.input_tensor_shape,
        p.compute_config,
        p.conv_config.enable_act_double_buffer,
        p.conv_config.enable_weights_double_buffer,
        p.conv_config.full_inner_dim,
        p.conv_config.enable_activation_reuse,
        p.conv_config.config_tensors_in_dram,
        p.conv_config.force_split_reader);
}

// Calls prim::conv2d_new (new descriptor factory)
Tensor call_new_prim(const PreparedConv2dPrimInputs& p) {
    return ttnn::prim::conv2d_new(
        p.input_tensor_post_tm,
        p.weight_tensor_on_device,
        p.bias_tensor_on_device,
        p.sliding_window_config,
        p.out_channels,
        p.groups,
        p.conv_config.output_layout == Layout::ROW_MAJOR,
        p.conv_config.activation,
        p.parallel_cfg,
        p.block_cfg,
        p.conv_out_memory_config,
        p.output_dtype,
        p.input_tensor_shape,
        p.compute_config,
        p.conv_config.enable_act_double_buffer,
        p.conv_config.enable_weights_double_buffer,
        p.conv_config.full_inner_dim,
        p.conv_config.enable_activation_reuse,
        p.conv_config.config_tensors_in_dram,
        p.conv_config.force_split_reader);
}

// -----------------------------------------------------------------------
// Correctness: Verify old and new produce the same output
// -----------------------------------------------------------------------
TEST_F(Conv2dDescriptorBenchmark, CorrectnessOldVsNew) {
    uint32_t batch_size = 5;
    uint32_t in_channels = 3;
    uint32_t out_channels = 17;
    uint32_t input_height = 111;
    uint32_t input_width = 25;
    std::array<uint32_t, 2> kernel_size = {3, 3};
    std::array<uint32_t, 2> stride = {1, 1};
    std::array<uint32_t, 2> padding = {1, 1};
    std::array<uint32_t, 2> dilation = {1, 1};

    ttnn::random::seed(42);

    MemoryConfig dram_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape in_shape{batch_size, in_channels, input_height, input_width};
    ttnn::Shape w_shape{out_channels, in_channels, kernel_size[0], kernel_size[1]};

    Tensor input_nchw = ttnn::random::random(in_shape, DataType::BFLOAT16).to_device(this->device_, dram_config);
    Tensor input_nhwc = ttnn::permute(input_nchw, ttnn::SmallVector<int64_t>{0, 2, 3, 1});
    Tensor weight = ttnn::random::random(w_shape, DataType::BFLOAT16);

    auto prep = prepare_conv2d_prim_inputs(
        input_nhwc,
        weight,
        this->device_,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        1);

    // Run old prim (cache miss)
    Tensor out_old = call_old_prim(prep);
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});

    // Run new prim (cache miss)
    Tensor out_new = call_new_prim(prep);
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});

    // Compare outputs
    Tensor old_cpu = ttnn::to_memory_config(out_old, dram_config).cpu();
    Tensor new_cpu = ttnn::to_memory_config(out_new, dram_config).cpu();

    ASSERT_EQ(old_cpu.padded_shape(), new_cpu.padded_shape()) << "Output shapes differ";
    ASSERT_EQ(old_cpu.dtype(), new_cpu.dtype()) << "Output dtypes differ";

    // Custom allclose that treats inf == inf and -inf == -inf as equal
    auto old_buf = tt::tt_metal::host_buffer::get_as<bfloat16>(old_cpu);
    auto new_buf = tt::tt_metal::host_buffer::get_as<bfloat16>(new_cpu);
    ASSERT_EQ(old_buf.size(), new_buf.size());
    uint32_t mismatches = 0;
    constexpr float rtol = 0.1f;
    constexpr float atol = 0.1f;
    for (size_t i = 0; i < old_buf.size(); ++i) {
        float a = static_cast<float>(old_buf[i]);
        float b = static_cast<float>(new_buf[i]);
        if (a == b) {
            continue;  // handles inf == inf, -inf == -inf, exact matches
        }
        if (std::isnan(a) && std::isnan(b)) {
            continue;
        }
        float diff = std::abs(a - b);
        float norm = std::min(std::abs(a) + std::abs(b), std::numeric_limits<float>::max());
        if (diff >= std::max(atol, rtol * norm)) {
            if (mismatches < 5) {
                std::cerr << "  Mismatch at [" << i << "]: old=" << a << " new=" << b << "\n";
            }
            ++mismatches;
        }
    }
    ASSERT_EQ(mismatches, 0u) << mismatches << " elements differ between old and new Conv2d prim";
}

// -----------------------------------------------------------------------
// Performance: old vs new dispatch overhead comparison.
// Both prims use the exact same prepared inputs.
// First call is cache-miss (compile), subsequent calls hit cache.
// -----------------------------------------------------------------------
TEST_F(Conv2dDescriptorBenchmark, DispatchPerformanceOldVsNew) {
    constexpr uint32_t N = 100000;

    uint32_t batch_size = 5;
    uint32_t in_channels = 3;
    uint32_t out_channels = 17;
    uint32_t input_height = 111;
    uint32_t input_width = 25;
    std::array<uint32_t, 2> kernel_size = {3, 3};
    std::array<uint32_t, 2> stride = {1, 1};
    std::array<uint32_t, 2> padding = {1, 1};
    std::array<uint32_t, 2> dilation = {1, 1};

    ttnn::random::seed(42);

    MemoryConfig dram_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    ttnn::Shape in_shape{batch_size, in_channels, input_height, input_width};
    ttnn::Shape w_shape{out_channels, in_channels, kernel_size[0], kernel_size[1]};

    Tensor input_nchw = ttnn::random::random(in_shape, DataType::BFLOAT16).to_device(this->device_, dram_config);
    Tensor input_nhwc = ttnn::permute(input_nchw, ttnn::SmallVector<int64_t>{0, 2, 3, 1});
    Tensor weight = ttnn::random::random(w_shape, DataType::BFLOAT16);

    auto prep = prepare_conv2d_prim_inputs(
        input_nhwc,
        weight,
        this->device_,
        in_channels,
        out_channels,
        batch_size,
        input_height,
        input_width,
        kernel_size,
        stride,
        padding,
        dilation,
        1);

    // --- Warm up both paths (cache miss) ---
    {
        auto out = call_old_prim(prep);
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    {
        auto out = call_new_prim(prep);
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});

    // --- Benchmark OLD prim ---
    auto t0_old = Clock::now();
    for (uint32_t i = 0; i < N; ++i) {
        auto out = call_old_prim(prep);
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    auto t1_old = Clock::now();
    Duration old_total = t1_old - t0_old;

    // --- Benchmark NEW prim ---
    auto t0_new = Clock::now();
    for (uint32_t i = 0; i < N; ++i) {
        auto out = call_new_prim(prep);
    }
    tt::tt_metal::distributed::Synchronize(this->device_, std::nullopt, {});
    auto t1_new = Clock::now();
    Duration new_total = t1_new - t0_new;

    double old_per_op = old_total.count() / N;
    double new_per_op = new_total.count() / N;
    double overhead_pct = ((new_per_op - old_per_op) / old_per_op) * 100.0;

    std::cout << "\n=== Conv2d Old vs New Dispatch Benchmark ===\n";
    std::cout << "  Iterations:      " << N << "\n";
    std::cout << "  OLD total:       " << old_total.count() << " ms\n";
    std::cout << "  OLD per-op:      " << old_per_op << " ms/op\n";
    std::cout << "  NEW total:       " << new_total.count() << " ms\n";
    std::cout << "  NEW per-op:      " << new_per_op << " ms/op\n";
    std::cout << "  Overhead:        " << overhead_pct << " %\n";
    std::cout << "=============================================\n\n";

    // Allow up to 5% overhead for the new path
    EXPECT_LE(overhead_pct, 5.0) << "New descriptor-based conv2d exceeds 5% overhead vs old: " << new_per_op
                                 << " ms vs " << old_per_op << " ms";
}

}  // namespace
