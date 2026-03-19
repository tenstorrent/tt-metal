// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include "tt-metalium/base_types.hpp"

using namespace tt::tt_metal;
#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int /*argc*/, char** /*argv*/) {
    bool pass = true;

    try {
        constexpr int device_id = 0;
        std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
        distributed::MeshWorkload workload;
        auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
        Program program = CreateProgram();

        constexpr CoreCoord core = {0, 0};

        constexpr uint32_t M = 200;
        constexpr uint32_t K = 300;
        constexpr uint32_t N = 400;
        constexpr uint32_t tile_size = 32;

        constexpr uint32_t padded_M = ((M + tile_size - 1) / tile_size) * tile_size;
        constexpr uint32_t padded_K = ((K + tile_size - 1) / tile_size) * tile_size;
        constexpr uint32_t padded_N = ((N + tile_size - 1) / tile_size) * tile_size;

        constexpr uint32_t Mt = padded_M / tile_size;
        constexpr uint32_t Kt = padded_K / tile_size;
        constexpr uint32_t Nt = padded_N / tile_size;
        constexpr uint32_t A_total_tiles = Mt * Kt;
        constexpr uint32_t B_total_tiles = Kt * Nt;
        constexpr uint32_t C_total_tiles = Mt * Nt;

        constexpr uint32_t elements_per_tile = tile_size * tile_size;
        constexpr uint32_t tile_size_bytes = sizeof(bfloat16) * elements_per_tile;

        distributed::DeviceLocalBufferConfig dram_config{.page_size = tile_size_bytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig A_buffer_config{.size = A_total_tiles * tile_size_bytes};
        distributed::ReplicatedBufferConfig B_buffer_config{.size = B_total_tiles * tile_size_bytes};
        distributed::ReplicatedBufferConfig C_buffer_config{.size = C_total_tiles * tile_size_bytes};

        auto srcA_dram_buffer = distributed::MeshBuffer::create(A_buffer_config, dram_config, mesh_device.get());
        auto srcB_dram_buffer = distributed::MeshBuffer::create(B_buffer_config, dram_config, mesh_device.get());
        auto srcC_dram_buffer = distributed::MeshBuffer::create(C_buffer_config, dram_config, mesh_device.get());
        auto dst_dram_buffer = distributed::MeshBuffer::create(C_buffer_config, dram_config, mesh_device.get());

        std::vector<bfloat16> a_tensor_data(elements_per_tile * A_total_tiles);
        std::vector<bfloat16> b_tensor_data(elements_per_tile * B_total_tiles);
        std::vector<bfloat16> c_tensor_data(elements_per_tile * C_total_tiles);

        std::fill(a_tensor_data.begin(), a_tensor_data.end(), bfloat16(0.0f));
        std::fill(b_tensor_data.begin(), b_tensor_data.end(), bfloat16(0.0f));
        std::fill(c_tensor_data.begin(), c_tensor_data.end(), bfloat16(0.0f));

        for (uint32_t tile_row = 0; tile_row < Mt; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < Kt; tile_col++) {
                uint32_t tile_id = tile_row * Kt + tile_col;
                uint32_t tile_offset = tile_id * elements_per_tile;

                for (uint32_t r = 0; r < tile_size; r++) {
                    for (uint32_t c = 0; c < tile_size; c++) {
                        uint32_t global_row = tile_row * tile_size + r;
                        uint32_t global_col = tile_col * tile_size + c;
                        uint32_t tile_element_idx = r * tile_size + c;

                        if (global_row < M && global_col < K) {
                            a_tensor_data[tile_offset + tile_element_idx] = bfloat16(2.0f);
                        }
                    }
                }
            }
        }

        for (uint32_t tile_row = 0; tile_row < Kt; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < Nt; tile_col++) {
                uint32_t tile_id = tile_row * Nt + tile_col;
                uint32_t tile_offset = tile_id * elements_per_tile;

                for (uint32_t r = 0; r < tile_size; r++) {
                    for (uint32_t c = 0; c < tile_size; c++) {
                        uint32_t global_row = tile_row * tile_size + r;
                        uint32_t global_col = tile_col * tile_size + c;
                        uint32_t tile_element_idx = r * tile_size + c;

                        if (global_row < K && global_col < N) {
                            b_tensor_data[tile_offset + tile_element_idx] = bfloat16(3.0f);
                        }
                    }
                }
            }
        }

        for (uint32_t tile_row = 0; tile_row < Mt; tile_row++) {
            for (uint32_t tile_col = 0; tile_col < Nt; tile_col++) {
                uint32_t tile_id = tile_row * Nt + tile_col;
                uint32_t tile_offset = tile_id * elements_per_tile;

                for (uint32_t r = 0; r < tile_size; r++) {
                    for (uint32_t c = 0; c < tile_size; c++) {
                        uint32_t global_row = tile_row * tile_size + r;
                        uint32_t global_col = tile_col * tile_size + c;
                        uint32_t tile_element_idx = r * tile_size + c;

                        if (global_row < M && global_col < N) {
                            c_tensor_data[tile_offset + tile_element_idx] = bfloat16(4.0f);
                        }
                    }
                }
            }
        }

        distributed::EnqueueWriteMeshBuffer(cq, srcA_dram_buffer, a_tensor_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, srcB_dram_buffer, b_tensor_data, false);
        distributed::EnqueueWriteMeshBuffer(cq, srcC_dram_buffer, c_tensor_data, false);

        constexpr uint32_t tiles_per_cb = 2;
        tt::CBIndex src0_cb_index = tt::CBIndex::c_0;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, tile_size_bytes));

        tt::CBIndex src1_cb_index = tt::CBIndex::c_1;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, tile_size_bytes));

        tt::CBIndex src2_cb_index = tt::CBIndex::c_2;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{src2_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src2_cb_index, tile_size_bytes));

        tt::CBIndex dst_cb_index = tt::CBIndex::c_16;
        CreateCircularBuffer(
            program,
            core,
            CircularBufferConfig(tiles_per_cb * tile_size_bytes, {{dst_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(dst_cb_index, tile_size_bytes));

        std::vector<uint32_t> reader_compile_time_args;
        TensorAccessorArgs(*srcA_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*srcB_dram_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*srcC_dram_buffer).append_to(reader_compile_time_args);
        auto reader = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa/kernels/dataflow/read.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = reader_compile_time_args});

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
        auto writer = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa/kernels/dataflow/write.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = writer_compile_time_args});

        std::vector<uint32_t> compute_compile_time_args;
        compute_compile_time_args.push_back(Mt);
        compute_compile_time_args.push_back(Kt);
        compute_compile_time_args.push_back(Nt);
        auto compute = CreateKernel(
            program,
            OVERRIDE_KERNEL_PREFIX "jaksa/kernels/compute/compute.cpp",
            core,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = compute_compile_time_args});

        SetRuntimeArgs(
            program,
            reader,
            core,
            {srcA_dram_buffer->address(), srcB_dram_buffer->address(), srcC_dram_buffer->address(), Mt, Kt, Nt});
        SetRuntimeArgs(program, writer, core, {dst_dram_buffer->address(), Mt, Nt});
        SetRuntimeArgs(program, compute, core, {2});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(cq, workload, false);
        distributed::Finish(cq);

        std::vector<bfloat16> result_vec;
        distributed::EnqueueReadMeshBuffer(cq, result_vec, dst_dram_buffer, true);

        TT_FATAL(result_vec.size() == c_tensor_data.size(), "Result vector size mismatch");
        fmt::print("result size: {}\n", result_vec.size());
        fmt::print("result tile 1: {}\n", static_cast<float>(result_vec[0]));

        float total_expected = 0.0f;
        float total_actual = 0.0f;
        const float expected_val = 2.0f * 3.0f * K;
        for (size_t i = 0; i < result_vec.size(); ++i) {
            float actual = static_cast<float>(result_vec[i]);
            total_expected += expected_val;
            total_actual += actual;
        }
        fmt::print("expected / actual: {}\n", total_expected / total_actual);

        pass &= mesh_device->close();
    } catch (const std::exception& e) {
        fmt::print(stderr, "Test failed with exception!\n");
        fmt::print(stderr, "{}\n", e.what());
        throw;
    }

    if (pass) {
        fmt::print("Test Passed\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}
