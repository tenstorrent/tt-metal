// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/distributed/utils.hpp"

namespace tt::tt_metal::distributed::test::utils {

std::vector<std::shared_ptr<Program>> create_eltwise_bin_programs(
    std::shared_ptr<MeshDevice>& mesh_device,
    std::vector<std::shared_ptr<MeshBuffer>>& src0_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& src1_bufs,
    std::vector<std::shared_ptr<MeshBuffer>>& output_bufs) {
    const std::vector<std::string> op_id_to_op_define = {"add_tiles", "mul_tiles"};
    const std::vector<std::string> op_id_to_op_type_define = {"EltwiseBinaryType::ELWADD", "EltwiseBinaryType::ELWMUL"};

    CoreCoord worker_grid_size = mesh_device->compute_with_storage_grid_size();

    std::vector<std::shared_ptr<Program>> programs = {std::make_shared<Program>(), std::make_shared<Program>()};
    auto full_grid = CoreRange({0, 0}, {worker_grid_size.x - 1, worker_grid_size.y - 1});

    for (std::size_t eltwise_op = 0; eltwise_op < op_id_to_op_define.size(); eltwise_op++) {
        auto& program = *programs[eltwise_op];
        uint32_t single_tile_size = 2 * 1024;
        uint32_t num_tiles = 2048;
        uint32_t dram_buffer_size =
            single_tile_size * num_tiles;  // num_tiles of FP16_B, hard-coded in the reader/writer kernels
        uint32_t page_size = single_tile_size;

        ReplicatedBufferConfig global_buffer_config{.size = dram_buffer_size};
        DeviceLocalBufferConfig per_device_buffer_config{
            .page_size = page_size,
            .buffer_type = tt_metal::BufferType::DRAM,
            .buffer_layout = TensorMemoryLayout::INTERLEAVED,
            .bottom_up = true};

        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                auto src0_dram_buffer =
                    MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
                src0_bufs.push_back(src0_dram_buffer);

                auto src1_dram_buffer =
                    MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
                src1_bufs.push_back(src1_dram_buffer);
                auto dst_dram_buffer =
                    MeshBuffer::create(global_buffer_config, per_device_buffer_config, mesh_device.get());
                output_bufs.push_back(dst_dram_buffer);
            }
        }

        uint32_t src0_cb_index = tt::CBIndex::c_0;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, full_grid, cb_src0_config);

        uint32_t src1_cb_index = tt::CBIndex::c_1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, full_grid, cb_src1_config);

        uint32_t ouput_cb_index = tt::CBIndex::c_16;
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, full_grid, cb_output_config);

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_dual_8bank.cpp",
            full_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_unary_8bank.cpp",
            full_grid,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {};

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        std::map<string, string> binary_defines = {
            {"ELTWISE_OP", op_id_to_op_define[eltwise_op]}, {"ELTWISE_OP_TYPE", op_id_to_op_type_define[eltwise_op]}};
        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            full_grid,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});

        SetRuntimeArgs(program, eltwise_binary_kernel, full_grid, {2048, 1});

        for (std::size_t col_idx = 0; col_idx < worker_grid_size.x; col_idx++) {
            for (std::size_t row_idx = 0; row_idx < worker_grid_size.y; row_idx++) {
                CoreCoord curr_core = {col_idx, row_idx};
                const std::array<uint32_t, 7> reader_args = {
                    src0_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(),
                    0,
                    num_tiles,
                    src1_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(),
                    0,
                    num_tiles,
                    0};

                const std::array<uint32_t, 3> writer_args = {
                    output_bufs.at(col_idx * worker_grid_size.y + row_idx)->address(), 0, num_tiles};

                SetRuntimeArgs(program, unary_writer_kernel, curr_core, writer_args);
                SetRuntimeArgs(program, binary_reader_kernel, curr_core, reader_args);
            }
        }
    }
    return programs;
}

}  // namespace tt::tt_metal::distributed::test::utils
