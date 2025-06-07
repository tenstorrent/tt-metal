// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/shape2d.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt-metalium/tile.hpp>
#include "impl/context/metal_context.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/cpp/ttnn/operations/functions.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/trace.hpp"
#include "ttnn/tensor/enum_types.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn_test_fixtures.hpp"

std::string dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::BFLOAT16: return "BFLOAT16";
        case DataType::BFLOAT8_B: return "BFLOAT8_B";
        case DataType::BFLOAT4_B: return "BFLOAT4_B";
        default: return "UNKNOWN";
    }
}

std::string fidelity_to_string(MathFidelity fidelity) {
    std::ostringstream oss;
    oss << fidelity;
    return oss.str();
}
int get_cycles_per_tile_for_fidelity(MathFidelity fidelity) {
    const int LoFi_cycle = 16;

    switch (fidelity) {
        case MathFidelity::LoFi: return LoFi_cycle;
        case MathFidelity::HiFi2: return LoFi_cycle * 2;
        case MathFidelity::HiFi3: return LoFi_cycle * 3;
        case MathFidelity::HiFi4: return LoFi_cycle * 4;
        default: return LoFi_cycle;
    }
}
std::array<Shape2D, 20> kSubblockHwChoices = {{{4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2},
                                               {2, 3}, {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1},
                                               {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1}}};

Shape2D get_subblock_sizes(
    int m_tiles_per_core, int n_tiles_per_core, bool out_sharded = false, bool fp32_dest_acc_en = false) {
    for (const Shape2D& subblock_hw : kSubblockHwChoices) {
        const int out_subblock_h = subblock_hw.height();
        const int out_subblock_w = subblock_hw.width();

        if (fp32_dest_acc_en) {
            if ((out_subblock_h * out_subblock_w) > 4) {
                continue;
            }
        }

        if (out_sharded) {
            if (n_tiles_per_core % out_subblock_w != 0 || out_subblock_h != 1) {
                continue;
            }
        }

        if (m_tiles_per_core % out_subblock_h == 0 && n_tiles_per_core % out_subblock_w == 0) {
            return {out_subblock_h, out_subblock_w};
        }
    }

    return {1, 1};
}

struct MatmulTestConfig {
    DataType dtype = DataType::BFLOAT16;
    MathFidelity fidelity = MathFidelity::HiFi2;
    bool enable_tracing = false;
    int num_warmup_iterations = 1;
    int num_measurement_iterations = 1;
    bool enable_program_cache = true;
};

struct MatmulShape {
    int m = 512;
    int k = 512;
    int n = 512;

    bool in0_sharded = true;
    bool out_sharded = true;

    int in0_block_w_div = 1;
    int num_out_blocks_h = 1;
    int num_out_blocks_w = 1;

    Shape2D grid_size = {8, 8};
    Shape2D tile_shape = {32, 32};
};

class Matmul2DHostPerfTestFixture : public ttnn::TTNNFixtureWithDevice,
                                    public testing::WithParamInterface<std::tuple<MatmulTestConfig, MatmulShape>> {
public:
    Matmul2DHostPerfTestFixture() :
        ttnn::TTNNFixtureWithDevice(/*trace_region_size=*/65536, /*l1_small_size=*/200000) {}
};

TEST_P(Matmul2DHostPerfTestFixture, Matmul2DHostPerfTest) {
    GTEST_SKIP() << "Benchmark is not intended to be run as part of CI and can be manually run locally";

    // Parse test config
    const MatmulTestConfig& test_config = std::get<0>(GetParam());
    const int num_warmup_iterations = test_config.num_warmup_iterations;
    const int num_measurement_iterations = test_config.num_measurement_iterations;

    DataType dtype = test_config.dtype;
    MathFidelity math_fidelity = test_config.fidelity;
    const bool use_trace = test_config.enable_tracing;

    TT_FATAL(num_measurement_iterations > 0, "Won't have data without at least one measurement iteration");

    const bool enable_program_cache = test_config.enable_program_cache;
    if (use_trace) {
        TT_FATAL(enable_program_cache, "Tracing requires program cache to be enabled");
    }
    if (enable_program_cache) {
        device_->enable_program_cache();
    }

    // Parse shape params
    const MatmulShape& MatmulShape = std::get<1>(GetParam());

    const Shape2D& grid_size = MatmulShape.grid_size;
    const int tile_h = MatmulShape.tile_shape.height();
    const int tile_w = MatmulShape.tile_shape.width();

    const int m = MatmulShape.m;
    const int k = MatmulShape.k;
    const int n = MatmulShape.n;
    const bool in0_sharded = MatmulShape.in0_sharded;
    const bool out_sharded = MatmulShape.out_sharded;
    const int in0_block_w_div = MatmulShape.in0_block_w_div;
    const int num_out_blocks_h = MatmulShape.num_out_blocks_h;
    const int num_out_blocks_w = MatmulShape.num_out_blocks_w;

    // Validate user compute grid is feasible
    TT_FATAL(grid_size.height() > 0 && grid_size.width() > 0, "Invalid grid size");

    auto compute_grid_size = device_->compute_with_storage_grid_size();
    if (compute_grid_size.y < grid_size.height() || compute_grid_size.x < grid_size.width()) {
        GTEST_SKIP() << "Skipping test as requested compute grid size " << grid_size
                     << " exceeds available compute grid " << compute_grid_size.str();
    }

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    std::string artifacts_dir = std::string(tt_metal_home) + "/generated";
    std::string file_name =
        artifacts_dir + "/matmul_2d_host_perf_report_" + dtype_to_string(dtype) + fidelity_to_string(math_fidelity);

    if (use_trace) {
        file_name += "_traced.csv";
    } else {
        file_name += ".csv";
    }

    std::ofstream file(file_name);
    file << "m,k,n,use_trace,grid_size,in0_sharded,out_sharded,in0_storage_type,in1_storage_type,out_storage_type,"
            "dtype,math_fidelity,inference_time_avg (ns),TFLOPs (avg),Utilization (vs user grid),Utilization (vs 8x8 "
            "full grid)\n";

    log_info(
        tt::LogTest, "Running test with dtype: {}, math_fidelity: {}, use_trace: {}", dtype, math_fidelity, use_trace);

    const int in0_block_w = k / grid_size.height() / 32 / in0_block_w_div;
    const int per_core_M = m / grid_size.width() / tile_h;
    const int per_core_N = n / grid_size.height() / tile_w;
    const int out_block_h = per_core_M / num_out_blocks_h;
    const int out_block_w = per_core_N / num_out_blocks_w;
    const Shape2D out_subblock = get_subblock_sizes(out_block_h, out_block_w, out_sharded);

    log_info(
        tt::LogTest,
        "M*K*N = {}*{}*{} out_subblock_h: {}, out_subblock_w: {}",
        m,
        k,
        n,
        out_subblock.height(),
        out_subblock.width());

    std::string in0_storage_type = in0_sharded ? "L1" : "DRAM";
    std::string in1_storage_type = "DRAM";
    std::string out_storage_type = out_sharded ? "L1" : "DRAM";

    const ttnn::MemoryConfig in0_memory_config =
        in0_sharded ? ttnn::operations::data_movement::create_sharded_memory_config(
                          ttnn::Shape{1, 1, m, k},
                          ttnn::CoreRangeSet(ttnn::CoreRange(
                              CoreCoord(0, 0), ttnn::CoreCoord(grid_size.height() - 1, grid_size.width() - 1))),
                          ttnn::operations::data_movement::ShardStrategy::BLOCK,
                          tt::tt_metal::ShardOrientation::ROW_MAJOR)
                    : ttnn::DRAM_MEMORY_CONFIG;

    // In0 is all ones
    const std::vector<float> in0_data(m * k, 1.0f);
    ttnn::Tensor input_tensor_0 = Tensor::from_vector(
        in0_data,
        ttnn::TensorSpec(
            ttnn::Shape({m, k}), tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, in0_memory_config)),
        device_);
    // In1 is random data
    std::vector<float> in1_data(k * n);
    std::generate(in1_data.begin(), in1_data.end(), []() {
        float value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        return value;
    });

    ttnn::Tensor input_tensor_1 = Tensor::from_vector(
        in1_data,
        ttnn::TensorSpec(
            ttnn::Shape({k, n}),
            tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device_);

    /*
    There are three main program configs available for matmul:
        1. MatmulMultiCoreReuseMultiCast1DProgramConfig ("1D")
            -Used for width or height sharded tensors, or very narrow interleaved tensors
        2. MatmulMultiCoreReuseMultiCastProgramConfig ("2D")
            -Used for block sharded tensors, and general interleaved tensors
        3. MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig ("DRAM")
            -A very specialised config for very narrow tensors in DRAM
    */
    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config{
        /* compute_with_storage_grid_size */ {grid_size.height(), grid_size.width()},
        in0_block_w,
        out_subblock.height(),
        out_subblock.width(),
        out_block_h,
        out_block_w,
        per_core_M,
        per_core_N,
        /*transpose_mcast=*/false,
        /*fused_activation=*/std::nullopt};

    const ttnn::DeviceComputeKernelConfig compute_kernel_config = ttnn::init_device_compute_kernel_config(
        device_->arch(),
        /*device_kernel_config=*/std::nullopt,
        math_fidelity,
        /*default_approx_mode=*/true,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/true,
        /*default_dst_full_sync_en=*/false);

    const ttnn::MemoryConfig out_mem_config =
        out_sharded ? ttnn::MemoryConfig{ttnn::TensorMemoryLayout::BLOCK_SHARDED, ttnn::BufferType::L1}
                    : ttnn::DRAM_MEMORY_CONFIG;

    const Tile output_tile =
        (out_sharded && tile_h <= 16) ? tt::tt_metal::Tile({tile_h, 32}) : tt::tt_metal::Tile({tile_h, tile_w});

    const ttnn::operations::matmul::Matmul matmul_params(
        program_config,
        /*bcast_batch=*/std::nullopt,
        out_mem_config,
        dtype,
        compute_kernel_config,
        /*untilize_out=*/false,
        /*user_core_coord=*/std::nullopt,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        output_tile);

    ttnn::Tensor output_tensor;
    // Warmup iterations
    for (int iter = 0; iter < num_warmup_iterations; ++iter) {
        output_tensor = ttnn::operations::matmul::matmul(
            input_tensor_0,
            input_tensor_1,
            /*bias=*/std::nullopt,
            /*parameters=*/matmul_params);
        Synchronize(device_, std::nullopt, std::vector<SubDeviceId>());
        output_tensor.deallocate();
    }

    // Note that time is measured with chrono around the ops, so there is some overhead from measuring on the
    // host side instead of on device.
    std::chrono::duration<double> total_time = std::chrono::duration<double>::zero();

    // Performance measurement iterations
    if (use_trace) {
        auto tid = ttnn::operations::trace::begin_trace_capture(device_, ttnn::DefaultQueueId);
        for (int iter = 0; iter < num_measurement_iterations; ++iter) {
            output_tensor = ttnn::operations::matmul::matmul(
                input_tensor_0,
                input_tensor_1,
                /*bias=*/std::nullopt,
                /*parameters=*/matmul_params);
            output_tensor.deallocate();
        }
        ttnn::operations::trace::end_trace_capture(device_, tid, ttnn::DefaultQueueId);

        auto start_time = std::chrono::high_resolution_clock::now();
        {
            ZoneScopedN("Matmul trace iterations");
            ttnn::operations::trace::execute_trace(device_, tid, ttnn::DefaultQueueId, false);
            Synchronize(device_, std::nullopt, std::vector<SubDeviceId>());
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        total_time += end_time - start_time;
        ttnn::operations::trace::release_trace(device_, tid);
    } else {
        {
            ZoneScopedN("Matmul iterations");
            for (int iter = 0; iter < num_measurement_iterations; ++iter) {
                auto start_time = std::chrono::high_resolution_clock::now();
                output_tensor = ttnn::operations::matmul::matmul(
                    input_tensor_0,
                    input_tensor_1,
                    /*bias=*/std::nullopt,
                    /*parameters=*/matmul_params);
                Synchronize(device_, std::nullopt, std::vector<SubDeviceId>());
                auto end_time = std::chrono::high_resolution_clock::now();
                total_time += end_time - start_time;
                output_tensor.deallocate();
            }
        }
    }

    const double inference_time_avg_s = total_time.count() / num_measurement_iterations;
    double tflops = 2.0 * m * k * n / 1e12 / inference_time_avg_s;
    int cycle_per_tile = get_cycles_per_tile_for_fidelity(math_fidelity);
    int num_cores_user_grid = grid_size.height() * grid_size.width();

    int num_cores_full_grid = compute_grid_size.x * compute_grid_size.y;
    const double dim_per_tile = (double)m * (double)k * (double)n / tile_h / tile_w;
    double ideal_cycle_full_grid = dim_per_tile / 32 * cycle_per_tile / num_cores_full_grid;
    double ideal_cycle_user_grid = dim_per_tile / 32 * cycle_per_tile / num_cores_user_grid;

    const int freq_mhz = tt::tt_metal::MetalContext::instance().get_cluster().get_device_aiclk(device_->id());
    double inference_cycle = inference_time_avg_s * freq_mhz * 1e6;

    double utilization_full_grid = ideal_cycle_full_grid / inference_cycle;
    double utilization_user_grid = ideal_cycle_user_grid / inference_cycle;
    std::string utilization_full_grid_percentage = std::to_string(utilization_full_grid * 100);
    std::string utilization_user_grid_percentage = std::to_string(utilization_user_grid * 100);
    log_info(
        tt::LogTest,
        "M*K*N = {}*{}*{} == inference time (avg): {}, tflops (avg): {}, utilization (vs user grid): {}%, "
        "utilization (vs 8x8 grid): {}%",
        m,
        k,
        n,
        inference_time_avg_s,
        tflops,
        utilization_user_grid_percentage,
        utilization_full_grid_percentage);

    file << m << "," << k << "," << n << "," << (use_trace ? "true" : "false") << "," << grid_size.height() << "x"
         << grid_size.width() << "," << in0_sharded << "," << out_sharded << "," << in0_storage_type << ","
         << in1_storage_type << "," << out_storage_type << "," << dtype_to_string(dtype) << "," << math_fidelity << ","
         << inference_time_avg_s * 1e9 << "," << tflops << "," << utilization_user_grid_percentage << ","
         << utilization_full_grid_percentage << "\n";

    // Deallocate input tensors
    input_tensor_0.deallocate();
    input_tensor_1.deallocate();
}

INSTANTIATE_TEST_SUITE_P(
    MatmulTests_BFLOAT16,
    Matmul2DHostPerfTestFixture,
    ::testing::Combine(
        // MatmulTestConfigs to use
        ::testing::ValuesIn(std::vector<MatmulTestConfig>{
            {DataType::BFLOAT16, MathFidelity::HiFi2, /*enable_tracing=*/false},
            {DataType::BFLOAT16, MathFidelity::HiFi2, /*enable_tracing=*/true},
            {DataType::BFLOAT16, MathFidelity::HiFi4, /*enable_tracing=*/false},
            {DataType::BFLOAT16, MathFidelity::HiFi4, /*enable_tracing=*/true}}),
        // MatmulShapes to use
        ::testing::ValuesIn(std::vector<MatmulShape>{
            {/*m=*/512,
             /*k=*/512,
             /*n=*/512,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/512,
             /*k=*/1024,
             /*n=*/1024,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/512,
             /*k=*/1024,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/1024,
             /*n=*/1024,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/1024,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/2048,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/2048,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/2048,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/3072,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/3072,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/4,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/3072,
             /*n=*/4096,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/4096,
             /*n=*/4096,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/4096,
             /*k=*/4096,
             /*n=*/4096,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/2,
             /*num_out_blocks_w=*/2},
            {/*m=*/8192,
             /*k=*/8192,
             /*n=*/8192,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/4,
             /*num_out_blocks_w=*/4},
            {/*m=*/16384,
             /*k=*/16384,
             /*n=*/16384,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/4,
             /*num_out_blocks_h=*/8,
             /*num_out_blocks_w=*/8}})));

INSTANTIATE_TEST_SUITE_P(
    MatmulTests_BFLOAT8B,
    Matmul2DHostPerfTestFixture,
    ::testing::Combine(
        // MatmulTestConfigs to use
        ::testing::ValuesIn(std::vector<MatmulTestConfig>{
            {DataType::BFLOAT8_B, MathFidelity::HiFi2, /*enable_tracing=*/false},
            {DataType::BFLOAT8_B, MathFidelity::HiFi2, /*enable_tracing=*/true},
            {DataType::BFLOAT8_B, MathFidelity::LoFi, /*enable_tracing=*/false},
            {DataType::BFLOAT8_B, MathFidelity::LoFi, /*enable_tracing=*/true}}),
        // MatmulShapes to use
        ::testing::ValuesIn(std::vector<MatmulShape>{
            {/*m=*/512,
             /*k=*/512,
             /*n=*/512,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/512,
             /*k=*/1024,
             /*n=*/1024,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/512,
             /*k=*/1024,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/1024,
             /*n=*/1024,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/1024,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/2048,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/2048,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/2048,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/3072,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/3072,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/3072,
             /*n=*/4096,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/4096,
             /*k=*/4096,
             /*n=*/4096,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/2,
             /*num_out_blocks_w=*/2},
            {/*m=*/8192,
             /*k=*/8192,
             /*n=*/8192,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/4,
             /*num_out_blocks_w=*/4},
            {/*m=*/16384,
             /*k=*/16384,
             /*n=*/16384,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/4,
             /*num_out_blocks_h=*/8,
             /*num_out_blocks_w=*/8}})));

INSTANTIATE_TEST_SUITE_P(
    MatmulTests_BFLOAT4B,
    Matmul2DHostPerfTestFixture,
    ::testing::Combine(
        // MatmulTestConfigs to use
        ::testing::ValuesIn(std::vector<MatmulTestConfig>{
            {DataType::BFLOAT4_B, MathFidelity::LoFi, /*enable_tracing=*/false},
            {DataType::BFLOAT4_B, MathFidelity::LoFi, /*enable_tracing=*/true}}),
        // MatmulShapes to use
        ::testing::ValuesIn(std::vector<MatmulShape>{
            {/*m=*/512,
             /*k=*/512,
             /*n=*/512,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/512,
             /*k=*/1024,
             /*n=*/1024,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/512,
             /*k=*/1024,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/1024,
             /*n=*/1024,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/1024,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/1024,
             /*k=*/2048,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/2048,
             /*n=*/2048,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/2048,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/2048,
             /*k=*/3072,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/3072,
             /*n=*/3072,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/3072,
             /*n=*/4096,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/1,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/3072,
             /*k=*/4096,
             /*n=*/4096,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/4096,
             /*k=*/4096,
             /*n=*/4096,
             /*in0_sharded=*/true,
             /*out_sharded=*/true,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/1,
             /*num_out_blocks_w=*/1},
            {/*m=*/8192,
             /*k=*/8192,
             /*n=*/8192,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/2,
             /*num_out_blocks_h=*/2,
             /*num_out_blocks_w=*/2},
            {/*m=*/16384,
             /*k=*/16384,
             /*n=*/16384,
             /*in0_sharded=*/false,
             /*out_sharded=*/false,
             /*in0_block_w_div=*/4,
             /*num_out_blocks_h=*/4,
             /*num_out_blocks_w=*/4}})));
