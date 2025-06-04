// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/base_types.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "device_fixture.hpp"
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_golden_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/env_vars.hpp"
#include "umd/device/types/arch.h"
#include <tt-metalium/utils.hpp>

namespace tt {
namespace tt_metal {
class IDevice;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal {

using std::vector;
using namespace tt;
using namespace tt::test_utils;
using namespace constants;

namespace unit_tests::compute::reduce {

enum ReduceDim : uint8_t { H = 0, W = 1, HW = 2 };

enum ReduceType : uint8_t { SUM = 0, AVG = 1, MAX = 2 };
struct ReduceConfig {
    bool short_init = false;
    tt_metal::Tile tile_shape = tt_metal::Tile({TILE_HEIGHT, TILE_WIDTH});
    std::vector<uint32_t> shape;
    ReduceDim reduce_dim;
    ReduceType reduce_type = ReduceType::SUM;
    float data_gen_rand_max;
    int data_gen_seed;
    float data_gen_offset;
    float atol;
    float rtol;
    std::function<std::vector<uint16_t>(
        const std::vector<uint16_t>&, const std::vector<uint32_t>&, float, uint8_t, bool)>
        golden_function;
    std::vector<uint32_t> result_shape;
    bool math_only_reduce = false;
    // Whether or not we want the result to be stored in DST in FP32:
    bool fp32_dest_acc_en = false;
    // Whether or not to sync full/half DST between MATH and PACK:
    bool dst_full_sync_en = false;
    bool at_start = false;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
};

float get_scaler(const ReduceConfig& test_config) {
    uint32_t H = test_config.shape[2];
    uint32_t W = test_config.shape[3];
    // If PoolType is MAX or SUM, then the operation is determined by PoolType,
    // but the scaler is 1
    if (test_config.reduce_type != ReduceType::AVG) {
        return 1.0f;
    } else {
        // If PoolType is AVG, the scaler depends on PoolDim, but the op is SUM
        switch (test_config.reduce_dim) {
            case ReduceDim::H: return (1.0f / H);
            case ReduceDim::W: return (1.0f / W);
            case ReduceDim::HW: return (1.0f / (H * W));
            default: {
                TT_THROW("Unsupported ReduceDim={}", test_config.reduce_dim);
                break;
            }
        }
    }
}

void set_math_fid_masks_binary(
    uint16_t& srca_fid_mask, uint16_t& srcb_fid_mask, MathFidelity math_fidelity = MathFidelity::HiFi4) {
    auto arch = get_arch_from_string(get_umd_arch_name());
    switch (math_fidelity) {
        case MathFidelity::HiFi4:
        case MathFidelity::HiFi3: {
            break;
        }
        case MathFidelity::HiFi2: {
            srcb_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            ;
            break;
        }
        case MathFidelity::LoFi: {
            srca_fid_mask = 0xFFF8;
            srcb_fid_mask = (arch == tt::ARCH::GRAYSKULL) ? 0xFFF8 : 0xFFFE;
            break;
        }
        default: {
            TT_THROW("Unsupported MathFidelity={}", math_fidelity);
            break;
        }
    }
}

void add_reader_writer_kernels(
    tt_metal::Program& program,
    const CoreCoord& logical_core,
    const ReduceConfig& test_config,
    const std::shared_ptr<tt_metal::Buffer>& src_dram_buffer,
    const std::shared_ptr<tt_metal::Buffer>& dst_dram_buffer) {
    uint32_t tile_H = test_config.tile_shape.get_tile_shape()[0], tile_W = test_config.tile_shape.get_tile_shape()[1];
    uint32_t W = test_config.shape[3], H = test_config.shape[2], NC = test_config.shape[1] * test_config.shape[0];
    uint32_t HW = H * W;
    uint32_t N = test_config.shape[0] * test_config.shape[1];
    uint32_t Wt = W / tile_W;
    uint32_t Ht = H / tile_H;
    uint32_t num_tensor_tiles = NC * H * W / (tile_W * tile_H);
    float scaler = get_scaler(test_config);
    switch (test_config.reduce_dim) {
        case ReduceDim::H: {
            bfloat16 bfloat_scaler_value = bfloat16(scaler);
            uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
            std::vector<uint32_t> reader_compile_args = {(std::uint32_t)true, packed_scaler_value};
            std::map<string, string> reader_defines = {{"REDUCE_SCALER", "1"}};

            auto unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp",
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = reader_compile_args,
                    .defines = reader_defines});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",  // no need to transpose the
                                                                                         // output since output Ht=1
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            tt_metal::SetRuntimeArgs(
                program, unary_reader_kernel, logical_core, {src_dram_buffer->address(), N, Ht, Wt, Ht * Wt});

            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                logical_core,
                {
                    dst_dram_buffer->address(),
                    (uint32_t)0,           // dram bank id
                    num_tensor_tiles / Ht  // num tiles
                });

            break;
        }
        case ReduceDim::HW: {
            scaler = std::sqrt(scaler);
        }  // Needed because AVG pool multiplies twice by the scaler
        case ReduceDim::W: {
            auto unary_reader_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_unary_8bank_reduce.cpp",
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

            auto unary_writer_kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
                logical_core,
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel,
                logical_core,
                {
                    src_dram_buffer->address(),
                    (uint32_t)0,  // dram bank id
                    (uint32_t)0,  // unused
                    num_tensor_tiles,
                    NC,
                    Ht,
                    Wt,
                    Ht * Wt,
                    *reinterpret_cast<uint32_t*>(&scaler),
                });

            uint32_t num_tiles =
                test_config.reduce_dim == ReduceDim::W ? (num_tensor_tiles / Wt) : (num_tensor_tiles / (Wt * Ht));
            tt_metal::SetRuntimeArgs(
                program,
                unary_writer_kernel,
                logical_core,
                {dst_dram_buffer->address(),
                 (uint32_t)0,  // dram bank id
                 num_tiles});

            break;
        }
        default: TT_THROW("Unsupported reduce dim!");
    }
}

std::string get_reduce_dim_define_string(const ReduceDim& reduce_dim) {
    std::string reduce_dim_define_str;
    switch (reduce_dim) {
        case ReduceDim::H: reduce_dim_define_str = "ReduceDim::REDUCE_COL"; break;
        case ReduceDim::W: reduce_dim_define_str = "ReduceDim::REDUCE_ROW"; break;
        case ReduceDim::HW: reduce_dim_define_str = "ReduceDim::REDUCE_SCALAR"; break;
        default: TT_THROW("Unsupported reduce dim!");
    }
    return reduce_dim_define_str;
}

std::string get_compute_kernel_name(const ReduceDim& reduce_dim) {
    std::string compute_kernel_name;
    switch (reduce_dim) {
        case ReduceDim::H: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_h.cpp"; break;
        case ReduceDim::W: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_w.cpp"; break;
        case ReduceDim::HW: compute_kernel_name = "tests/tt_metal/tt_metal/test_kernels/compute/reduce_hw.cpp"; break;
        default: TT_THROW("Unsupported reduce dim!");
    }
    return compute_kernel_name;
}

void run_single_core_reduce_program(tt_metal::IDevice* device, const ReduceConfig& test_config) {
    Program program = tt_metal::CreateProgram();

    CoreCoord core = {0, 0};

    uint32_t tile_H = test_config.tile_shape.get_tile_shape()[0], tile_W = test_config.tile_shape.get_tile_shape()[1];
    uint32_t W = test_config.shape[3], H = test_config.shape[2], NC = test_config.shape[1] * test_config.shape[0];
    uint32_t HW = H * W;
    uint32_t N = test_config.shape[0] * test_config.shape[1];
    TT_FATAL((tile_H == 16 && tile_W == 32) || (tile_H == 32 && tile_W == 32), "Error: Invalid tile shape");
    TT_FATAL(W % tile_W == 0 && H % tile_H == 0, "Error: Tensor height/width must be multiple of tile height/width");
    TT_FATAL(H > 0 && W > 0 && NC > 0, "Error: All tensor dims must be greater than 0");
    uint32_t Wt = W / tile_W;
    uint32_t Ht = H / tile_H;

    uint32_t num_golden_elements;
    switch (test_config.reduce_dim) {
        case ReduceDim::H:
            num_golden_elements = NC * W * 32 / 2;
            break;  // expecting one tile in H, and half the elements since the vector packs 2 uint16_ts
        case ReduceDim::W:
            num_golden_elements = NC * H * tile_W / 2;
            break;  // expecting one tile in H, and half the elements since the vector packs 2 uint16_ts
        case ReduceDim::HW:
            num_golden_elements = NC * 32 * 32 / 2;
            break;  // expecting one tile in H, and half the elements since the vector packs 2 uint16_ts
        default: TT_THROW("Unsupported reduce dim!");
    }

    float scaler = get_scaler(test_config);

    uint32_t num_tensor_tiles = NC * H * W / (tile_W * tile_H);
    uint32_t divisor = test_config.reduce_dim == ReduceDim::W ? Wt : Ht;
    TT_FATAL(num_tensor_tiles % divisor == 0, "Error");

    uint32_t single_tile_bytes = 2 * (tile_W * tile_H);
    uint32_t dram_buffer_size = single_tile_bytes * num_tensor_tiles;

    uint32_t src_page_size = single_tile_bytes;
    uint32_t dst_page_size = single_tile_bytes;

    tt_metal::InterleavedBufferConfig src_config{
        .device = device,
        .size = dram_buffer_size,
        .page_size = src_page_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    uint32_t output_size_bytes;
    switch (test_config.reduce_dim) {
        case ReduceDim::H: output_size_bytes = dram_buffer_size / Ht; break;
        case ReduceDim::W: output_size_bytes = dram_buffer_size / Wt; break;
        case ReduceDim::HW: output_size_bytes = dram_buffer_size / (Ht * Wt); break;
        default: TT_THROW("Unsupported reduce dim!");
    }

    tt_metal::InterleavedBufferConfig dst_config{
        .device = device,
        .size = output_size_bytes,
        .page_size = dst_page_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt_metal::Buffer> src_dram_buffer = CreateBuffer(src_config);
    std::shared_ptr<tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dst_config);

    uint32_t src0_cb_index = 0;
    uint32_t num_buffer_tiles = 32;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(
            num_buffer_tiles * single_tile_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_bytes)
            .set_tile_dims(src0_cb_index, test_config.tile_shape);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t ouput_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_buffer_tiles = 32;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_buffer_tiles * single_tile_bytes, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_bytes)
            .set_tile_dims(ouput_cb_index, test_config.tile_shape);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    tt_metal::CircularBufferConfig cb_temp_reduce_tile_config =
        tt_metal::CircularBufferConfig(2 * (2 * TILE_WIDTH * TILE_HEIGHT), {{CBIndex::c_2, tt::DataFormat::Float16_b}})
            .set_page_size(CBIndex::c_2, single_tile_bytes)
            .set_tile_dims(CBIndex::c_2, tt_metal::Tile({32, 32}));
    auto cb_temp_reduce_tile = tt_metal::CreateCircularBuffer(program, core, cb_temp_reduce_tile_config);

    add_reader_writer_kernels(program, core, test_config, src_dram_buffer, dst_dram_buffer);

    vector<uint32_t> compute_kernel_args = {
        uint(Ht),
        uint(Wt),
        uint(NC),
        test_config.at_start,
    };

    std::map<string, string> reduce_defines = {{"REDUCE_DIM", get_reduce_dim_define_string(test_config.reduce_dim)}};
    switch (test_config.reduce_type) {
        case ReduceType::SUM: {
            reduce_defines["REDUCE_OP"] = "PoolType::SUM";
            break;
        }
        case ReduceType::AVG: {
            reduce_defines["REDUCE_OP"] = "PoolType::AVG";
            break;
        }
        case ReduceType::MAX: {
            reduce_defines["REDUCE_OP"] = "PoolType::MAX";
            break;
        }
    }
    if (test_config.short_init) {
        reduce_defines["SHORT_INIT"] = "1";
    }
    reduce_defines["MATH_ONLY"] = test_config.math_only_reduce ? "1" : "0";
    reduce_defines["DST_ACCUM_MODE"] = test_config.fp32_dest_acc_en ? "1" : "0";

    std::string compute_kernel_name = get_compute_kernel_name(test_config.reduce_dim);

    auto reduce_compute_kernel = tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        core,
        tt_metal::ComputeConfig{
            .math_fidelity = test_config.math_fidelity,
            .fp32_dest_acc_en = test_config.fp32_dest_acc_en,
            .dst_full_sync_en = test_config.dst_full_sync_en,
            .compile_args = compute_kernel_args,
            .defines = reduce_defines});

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    vector<uint32_t> src_vec = create_random_vector_of_bfloat16(
        dram_buffer_size, test_config.data_gen_rand_max, test_config.data_gen_seed, test_config.data_gen_offset);

    tt_metal::detail::WriteToBuffer(src_dram_buffer, src_vec);

    tt_metal::detail::LaunchProgram(device, program);

    // The kernel will view the input as TILED_NFACES
    std::vector<uint32_t> result_vec;
    tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

    EXPECT_EQ(result_vec.size(), num_golden_elements);

    int argfail = -1;
    auto comparison_function = [&](float a, float b) {
        float maxabs = fmaxf(fabsf(a), fabsf(b));
        float absdiff = fabsf(a - b);
        auto result = (absdiff <= test_config.atol) || (absdiff <= test_config.rtol * maxabs);
        return result;
    };

    auto u16_src0_vec = u16_from_u32_vector(src_vec);
    if (test_config.reduce_type == ReduceType::AVG) {
        uint16_t srca_fid_mask = 0xFFFF;
        uint16_t srcb_fid_mask = 0xFFFF;
        set_math_fid_masks_binary(srca_fid_mask, srcb_fid_mask, test_config.math_fidelity);
        uint32_t uint32_scaler = *reinterpret_cast<uint32_t*>(&scaler);
        uint32_scaler &= (0xFFFFFFFF & (srcb_fid_mask << 16));
        scaler = *reinterpret_cast<float*>(&uint32_scaler);
        for (auto i = 0; i < u16_src0_vec.size(); i++) {
            u16_src0_vec[i] = u16_src0_vec[i] & srca_fid_mask;
        }
    }
    // recover a linear view of input vector for consumption by gold_ function
    std::vector<uint16_t> src_linear = convert_layout<uint16_t>(
        u16_src0_vec,
        test_config.shape,
        TensorLayoutType::TILED_NFACES,
        TensorLayoutType::LIN_ROW_MAJOR,
        PhysicalSize{tile_H, tile_W});
    std::vector<uint16_t> gold_reduced = test_config.golden_function(
        src_linear, test_config.shape, scaler, uint8_t(test_config.reduce_type), true);  // result is uint16_t untilized

    // Tilize from row major and convert to pairs (uint32_t)
    auto gold_4f_u32 = u32_from_u16_vector(convert_layout<uint16_t>(
        gold_reduced,
        test_config.result_shape,
        TensorLayoutType::LIN_ROW_MAJOR,
        TensorLayoutType::TILED_NFACES,
        PhysicalSize{tile_H, tile_W}));

    bool pass = packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
    if (!pass) {
        log_error(LogTest, "Failure position={}", argfail);
    }

    EXPECT_TRUE(pass);
    log_info(
        LogTest,
        "TileDimH = {}, TileDimW = {}, MathFid = {}, ReduceType = {}, FP32DestAcc = {}, DstSyncFull = {}, at_start = "
        "{}",
        test_config.tile_shape.get_tile_shape()[0],
        test_config.tile_shape.get_tile_shape()[1],
        test_config.math_fidelity,
        test_config.reduce_type,
        test_config.fp32_dest_acc_en,
        test_config.dst_full_sync_en,
        test_config.at_start);
}

}  // namespace unit_tests::compute::reduce

using namespace unit_tests::compute::reduce;

TEST_F(DeviceFixture, TensixComputeReduceH) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19 * TILE_HEIGHT, 17 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .shape = shape,
                            .reduce_dim = ReduceDim::H,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_h,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid),
                        };
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceW) {
    std::vector<uint32_t> shape = {1, 3, 17 * TILE_HEIGHT, 19 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                if ((fp32_dest_acc_en) && (this->arch_ == tt::ARCH::GRAYSKULL)) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .shape = shape,
                            .reduce_dim = ReduceDim::W,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_w,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid),
                        };
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceHW) {
    std::vector<uint32_t> shape = {1, 2, 7 * TILE_HEIGHT, 5 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc_en) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .shape = shape,
                            .reduce_dim = ReduceDim::HW,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_hw,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceHMathOnly) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19 * TILE_HEIGHT, 17 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .shape = shape,
                            .reduce_dim = ReduceDim::H,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_h,
                            .result_shape = result_shape,
                            .math_only_reduce = true,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceWMathOnly) {
    std::vector<uint32_t> shape = {1, 3, 17 * TILE_HEIGHT, 19 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                if ((fp32_dest_acc_en) && (this->arch_ == tt::ARCH::GRAYSKULL)) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .shape = shape,
                            .reduce_dim = ReduceDim::W,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_w,
                            .result_shape = result_shape,
                            .math_only_reduce = true,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}
// Disabled due to GH issue #14510
TEST_F(DeviceFixture, DISABLED_TensixComputeReduceHWMathOnly) {
    std::vector<uint32_t> shape = {1, 2, 7 * TILE_HEIGHT, 5 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc_en) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .shape = shape,
                            .reduce_dim = ReduceDim::HW,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_hw,
                            .result_shape = result_shape,
                            .math_only_reduce = true,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceHShortInit) {
    if (this->arch_ != tt::ARCH::BLACKHOLE) {
        // (issue #10181: disabling due to sporadic failures in slow dispatch mode)
        GTEST_SKIP();
    }
    std::vector<uint32_t> shape = {1, 3, 19 * TILE_HEIGHT, 17 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], TILE_HEIGHT, shape[3]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .short_init = true,
                            .shape = shape,
                            .reduce_dim = ReduceDim::H,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_h,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceWShortInit) {
    std::vector<uint32_t> shape = {1, 3, 17 * TILE_HEIGHT, 19 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                if ((fp32_dest_acc_en) && (this->arch_ == tt::ARCH::GRAYSKULL)) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .short_init = true,
                            .shape = shape,
                            .reduce_dim = ReduceDim::W,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_w,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}
// Disabled due to GH issue #14510
TEST_F(DeviceFixture, DISABLED_TensixComputeReduceHWShortInit) {
    std::vector<uint32_t> shape = {1, 2, 7 * TILE_HEIGHT, 5 * TILE_WIDTH};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], 32, 32};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                // Currently fp32 dest unsupported with reduce scalar
                if (fp32_dest_acc_en) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .short_init = true,
                            .shape = shape,
                            .reduce_dim = ReduceDim::HW,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 10.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = -10.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_hw,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid)};
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceWTinyTiles) {
    tt_metal::Tile tile_shape = tt_metal::Tile({TILE_HEIGHT / 2, TILE_WIDTH});
    std::vector<uint32_t> shape = {1, 1, 1 * tile_shape.get_tile_shape()[0], 13 * tile_shape.get_tile_shape()[1]};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], tile_shape.get_tile_shape()[1]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                if ((fp32_dest_acc_en) && (this->arch_ == tt::ARCH::GRAYSKULL)) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .tile_shape = tile_shape,
                            .shape = shape,
                            .reduce_dim = ReduceDim::W,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 0.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = 1.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_w,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid),
                        };
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

TEST_F(DeviceFixture, TensixComputeReduceWTinyTilesShortInit) {
    tt_metal::Tile tile_shape = tt_metal::Tile({TILE_HEIGHT / 2, TILE_WIDTH});
    std::vector<uint32_t> shape = {1, 1, 1 * tile_shape.get_tile_shape()[0], 13 * tile_shape.get_tile_shape()[1]};
    std::vector<uint32_t> result_shape = {shape[0], shape[1], shape[2], tile_shape.get_tile_shape()[1]};
    for (uint8_t math_fid = uint8_t(MathFidelity::LoFi); math_fid <= uint8_t(MathFidelity::HiFi4); math_fid++) {
        // MathFidelity : {0, 2, 3, 4}; so skip value 1
        if (math_fid == 1) {
            continue;
        }
        for (uint8_t reduce_type = uint8_t(ReduceType::SUM); reduce_type <= uint8_t(ReduceType::MAX); reduce_type++) {
            for (bool fp32_dest_acc_en : {true, false}) {
                if ((fp32_dest_acc_en) && (this->arch_ == tt::ARCH::GRAYSKULL)) {
                    continue;
                }
                for (bool dst_full_sync_en : {true, false}) {
                    for (bool at_start : {true, false}) {
                        ReduceConfig test_config = {
                            .short_init = true,
                            .tile_shape = tile_shape,
                            .shape = shape,
                            .reduce_dim = ReduceDim::W,
                            .reduce_type = ReduceType(reduce_type),
                            .data_gen_rand_max = 0.0f,
                            .data_gen_seed = std::chrono::system_clock::now().time_since_epoch().count(),
                            .data_gen_offset = 1.0f,
                            .atol = 1e-2f,
                            .rtol = 0.08f,
                            .golden_function = ::unit_tests::compute::gold_reduce_w,
                            .result_shape = result_shape,
                            .fp32_dest_acc_en = fp32_dest_acc_en,
                            .dst_full_sync_en = dst_full_sync_en,
                            .at_start = at_start,
                            .math_fidelity = MathFidelity(math_fid),
                        };
                        run_single_core_reduce_program(this->devices_.at(0), test_config);
                    }
                }
            }
        }
    }
}

}  // namespace tt::tt_metal
