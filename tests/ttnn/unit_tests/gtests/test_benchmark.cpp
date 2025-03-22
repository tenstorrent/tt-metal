#include "gtest/gtest.h"
#include "ttnn/device.hpp"
#include <vector>
#include <utility>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include <stdlib.h>

#include <tt-metalium/logger.hpp>
#include "ttnn_test_fixtures.hpp"

#include "tools/profiler/op_profiler.hpp"
#include "ttnn/cpp/ttnn/operations/creation.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/layout/tensor_layout.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/cpp/ttnn/operations/functions.hpp"
#include "ttnn/cpp/ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/common.hpp"

#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/trace.hpp"

int get_device_freq() { return 1e9; }

std::vector<std::pair<int, int>> SUBBLOCK_HW_CHOICES = {{4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2},
                                                        {2, 3}, {6, 1}, {1, 6}, {5, 1}, {1, 5}, {2, 2}, {4, 1},
                                                        {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1}};

std::pair<int, int> get_subblock_sizes(
    int m_tiles_per_core, int n_tiles_per_core, bool out_sharded = false, bool fp32_dest_acc_en = false) {
    for (const auto& subblock_hw : SUBBLOCK_HW_CHOICES) {
        int out_subblock_h = subblock_hw.first;
        int out_subblock_w = subblock_hw.second;

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

std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes_bfloat16 = {
    //    {16384, 16384, 16384, false, false, 4, 8, 8},
    {16384, 65536, 16384, false, false, 16, 8, 8},
};

std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes_bfloat8_b = {
    // {16384, 16384, 16384, false, false, 4, 8, 8},
    {512, 4096, 1024, true, true, 4, 1, 1},
};

std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes_bfloat4_b = {
    {512, 512, 512, true, true, 1, 1, 1},
    {512, 1024, 1024, true, true, 1, 1, 1},
    {512, 1024, 2048, true, true, 1, 1, 1},
    {1024, 1024, 1024, true, true, 1, 1, 1},
    {1024, 1024, 2048, true, true, 1, 1, 1},
    {1024, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 2048, true, true, 1, 1, 1},
    {2048, 2048, 3072, true, true, 1, 1, 1},
    {2048, 3072, 3072, true, true, 1, 1, 1},
    {3072, 3072, 3072, true, true, 1, 1, 1},
};

std::vector<std::tuple<DataType, MathFidelity, bool>> matmul_configs = {
    {DataType::BFLOAT16, MathFidelity::HiFi2, false},
    // {DataType::BFLOAT16, MathFidelity::HiFi4, false},
    {DataType::BFLOAT8_B, MathFidelity::HiFi2, false},
    //  {DataType::BFLOAT8_B, MathFidelity::LoFi, false},
    //  {DataType::BFLOAT4_B, MathFidelity::LoFi, false},
    // TODO: Enable tracing
    //  {DataType::BFLOAT16, MathFidelity::HiFi2, true},
    //  {DataType::BFLOAT16, MathFidelity::HiFi4, true},
    //  {DataType::BFLOAT8_B, MathFidelity::HiFi2, true},
    //  {DataType::BFLOAT8_B, MathFidelity::LoFi, true},
    //  {DataType::BFLOAT4_B, MathFidelity::LoFi, true},
};

class Matmul2DHostPerfTestFixture : public ttnn::distributed::test::TTNNFixtureWithTraceEnabledDevice,
                                    public testing::WithParamInterface<std::tuple<
                                        /* grid_size */ std::tuple<int, int>,
                                        /* tile_h */ int,
                                        /* tile_w */ int,
                                        /* num_warmup_iterations */ int,
                                        /* num_measurement_iterations */ int,
                                        /* use_program_cache */ bool>> {
public:
    Matmul2DHostPerfTestFixture() : ttnn::distributed::test::TTNNFixtureWithTraceEnabledDevice(24576, 200000) {}
};

void export_nops(int unpack_nops, int math_nops, int pack_nops) {
    setenv("UNPACK_NOPS", std::to_string(unpack_nops).c_str(), 1);
    setenv("MATH_NOPS", std::to_string(math_nops).c_str(), 1);
    setenv("PACK_NOPS", std::to_string(pack_nops).c_str(), 1);
}

struct matmul_struct_t {
    ttnn::Tensor in0_t;
    ttnn::Tensor in1_t;
    ttnn::operations::matmul::Matmul matmul_params;
};

ttnn::operations::matmul::Matmul get_matmul_info(
    int config_index,
    const std::tuple<int, int>& grid_size,
    const int tile_h,
    const int tile_w,
    tt::tt_metal::IDevice* device,
    ttnn::Tensor& in0_t,
    ttnn::Tensor& in1_t) {
    auto config = matmul_configs[config_index];
    DataType dtype = std::get<0>(config);
    MathFidelity math_fidelity = std::get<1>(config);
    const bool use_trace = std::get<2>(config);
    tt::log_info("Running test with dtype: {}, math_fidelity: {}", dtype, math_fidelity);

    std::vector<std::tuple<int, int, int, bool, bool, int, int, int>> matmul_shapes;
    if (dtype == DataType::BFLOAT16) {
        matmul_shapes = matmul_shapes_bfloat16;
    } else if (dtype == DataType::BFLOAT8_B) {
        matmul_shapes = matmul_shapes_bfloat8_b;
    } else if (dtype == DataType::BFLOAT4_B) {
        matmul_shapes = matmul_shapes_bfloat4_b;
    }

    auto shape = matmul_shapes[0];
    int m = std::get<0>(shape);
    int k = std::get<1>(shape);
    int n = std::get<2>(shape);
    m = m / 8 * std::get<1>(grid_size);
    n = n / 8 * std::get<0>(grid_size);
    k = k / 8 * std::get<0>(grid_size);
    const bool in0_sharded = std::get<3>(shape);
    const bool out_sharded = std::get<4>(shape);
    const int in0_block_w_div = std::get<5>(shape);
    const int num_out_blocks_h = std::get<6>(shape);
    const int num_out_blocks_w = std::get<7>(shape);
    const std::vector<int64_t> in0_shape = {1, 1, m, k};
    const std::vector<int64_t> in1_shape = {1, 1, k, n};

    const int in0_block_w = k / std::get<0>(grid_size) / 32 / in0_block_w_div;
    const int per_core_M = m / std::get<1>(grid_size) / tile_h;
    const int per_core_N = n / std::get<0>(grid_size) / tile_w;
    const int out_block_h = per_core_M / num_out_blocks_h;
    const int out_block_w = per_core_N / num_out_blocks_w;
    const auto [out_subblock_h, out_subblock_w] = get_subblock_sizes(out_block_h, out_block_w, out_sharded);

    tt::log_info("M*K*N = {}*{}*{} out_subblock_h: {}, out_subblock_w: {}", m, k, n, out_subblock_h, out_subblock_w);

    std::string in0_storage_type = in0_sharded ? "L1" : "DRAM";
    std::string in1_storage_type = "DRAM";
    std::string out_storage_type = out_sharded ? "L1" : "DRAM";

    const ttnn::MemoryConfig in0_memory_config =
        in0_sharded
            ? ttnn::operations::data_movement::create_sharded_memory_config(
                  ttnn::Shape{1, 1, m, k},
                  ttnn::CoreRangeSet(ttnn::CoreRange(
                      CoreCoord(0, 0), ttnn::CoreCoord(std::get<0>(grid_size) - 1, std::get<1>(grid_size) - 1))),
                  ttnn::operations::data_movement::ShardStrategy::BLOCK,
                  tt::tt_metal::ShardOrientation::ROW_MAJOR)
            : ttnn::DRAM_MEMORY_CONFIG;

    // In0 is all ones
    const std::vector<float> in0_data(m * k, 1.0f);
    in0_t = Tensor::from_vector(
        in0_data,
        ttnn::TensorSpec(
            ttnn::Shape({m, k}), tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, in0_memory_config)),
        device);
    // In1 is random data
    std::vector<float> in1_data(k * n);
    std::generate(in1_data.begin(), in1_data.end(), []() {
        float value = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        return std::round(value * 15.0f) / 15.0f;
    });

    in1_t = Tensor::from_vector(
        in1_data,
        ttnn::TensorSpec(
            ttnn::Shape({k, n}),
            tt::tt_metal::TensorLayout(dtype, tt::tt_metal::Layout::TILE, ttnn::DRAM_MEMORY_CONFIG)),
        device);

    ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig program_config{
        /* compute_with_storage_grid_size */ {std::get<0>(grid_size), std::get<1>(grid_size)},
        /* in0_block_w */ in0_block_w,
        /* out_subblock_h */ out_subblock_h,
        /* out_subblock_w */ out_subblock_w,
        /* out_block_h */ out_block_h,
        /* out_block_w */ out_block_w,
        /* per_core_M */ per_core_M,
        /* per_core_N */ per_core_N,
        /* transpose_mcast */ false,
        /* fused_activation */ std::nullopt};

    const ttnn::WormholeComputeKernelConfig compute_kernel_config =
        ttnn::WormholeComputeKernelConfig{math_fidelity, true, false, true};

    const ttnn::MemoryConfig out_mem_config =
        out_sharded ? ttnn::MemoryConfig{ttnn::TensorMemoryLayout::BLOCK_SHARDED, ttnn::BufferType::L1}
                    : ttnn::DRAM_MEMORY_CONFIG;

    const Tile output_tile =
        out_sharded ? (tile_h <= 16 ? tt::tt_metal::Tile({tile_h, 32}) : tt::tt_metal::Tile({tile_h, tile_w}))
                    : tt::tt_metal::Tile({tile_h, tile_w});

    return ttnn::operations::matmul::Matmul(
        program_config,
        /*bcast_batch*/ std::nullopt,
        /* output_mem_config */ out_mem_config,
        /* output_dtype */ dtype,
        /* compute_kernel_config */ compute_kernel_config,
        /* untilize_out */ false,
        /* user_core_coord */ std::nullopt,
        /* user_fused_activation */ std::nullopt,
        /* user_run_batched */ false,
        /* transpose_a */ false,
        /* transpose_b */ false,
        /* output_tile */ output_tile);
}

TEST_P(Matmul2DHostPerfTestFixture, Matmul2DHostPerfTest) {
    const std::tuple<int, int>& grid_size = std::get<0>(GetParam());
    const int& tile_h = std::get<1>(GetParam());
    const int& tile_w = std::get<2>(GetParam());
    const int& num_warmup_iterations = std::get<3>(GetParam());
    const int& num_measurement_iterations = std::get<4>(GetParam());
    const bool& use_program_cache = std::get<5>(GetParam());

    TT_FATAL(std::get<0>(grid_size) > 0 && std::get<1>(grid_size) > 0, "Invalid grid size");
    TT_ASSERT(num_measurement_iterations > 0, "Won't have data without at least one measurement iteration");
    tt::tt_metal::IDevice* device = &getDevice();

    for (int cindex = 0; cindex < 1; cindex++) {
        ttnn::Tensor output_tensor;
        ttnn::Tensor in0_t, in1_t;
        auto res = get_matmul_info(0, grid_size, tile_h, tile_w, device, in0_t, in1_t);
        for (int iter = 0; iter < num_measurement_iterations; ++iter) {
            std::cout << "Came here " << std::endl;
            output_tensor = ttnn::operations::matmul::matmul(
                in0_t,
                in1_t,
                /* bias */ std::nullopt,
                /* parameters */
                res);
            output_tensor.deallocate();
            std::cout << "Came here too" << std::endl;
        }
        // Deallocate input tensors
        // res.in0_t.deallocate();
        // res.in1_t.deallocate();
    }
}

INSTANTIATE_TEST_SUITE_P(
    /*Prefix for the instantiated tests*/ MatmulTests,
    /*Test suite*/ Matmul2DHostPerfTestFixture,
    ::testing::Values(std::make_tuple(
        /* grid_size */ std::make_tuple(1, 1),
        /* tile_h */ 32,
        /* tile_w */ 32,
        /* num_warmup_iterations */ 5,
        /* num_measurement_iterations */ 1,
        /* use_program_cache */ false)));
