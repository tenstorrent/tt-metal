#include "tt_dnn/op_library/split/split_last_dim_qk_tiled.hpp"

#include "tt_dnn/op_library/auto_format.hpp"

#include "common/constants.hpp"
#include "tt_metal/host_api.hpp"

#include <iostream>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks split_last_dim_qk_tiled(
    const Tensor &input_tensor, std::vector<Tensor> &output_tensors, const MemoryConfig &mem_config) {
    SplitLastDimQKTiled op(mem_config);
    uint32_t dim = op.dim;
    uint32_t num_chunks = op.num_chunks;

    auto input_shape = input_tensor.shape();

    Program program{};
    tt_metal::Device *device = input_tensor.device();
    op.boiler_plate_asserts(input_tensor);
    op.shape_asserts(input_tensor);

    // TODO: CHANGE TO FUNCTION CONVERSION
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = input_tensor.buffer();

    // Output buffers
    TT_ASSERT(output_tensors.size() == num_chunks);
    tt_metal::Tensor &q = output_tensors[0];
    tt_metal::Tensor &k = output_tensors[1];

    tt_metal::Buffer *q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer *k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t z = input_shape[1];
    uint32_t num_tiles_dim_2 = input_shape[2] / TILE_HEIGHT;
    uint32_t num_tiles_dim_3 = input_shape[3] / TILE_WIDTH;
    uint32_t num_cores_x_limit = device->compute_and_storage_grid_size().x;
    uint32_t num_cores_y_limit = device->compute_and_storage_grid_size().y;
    uint32_t dim_1_num_batches = input_shape[1];
    uint32_t num_cores_x = 1;
    uint32_t num_cores_y = 1;

    // TODO: parallelize for num_cores_x and num_cores_y != 1
    // Currently only working single core, will work on multicore implementation
    // uint32_t num_cores_x = get_max_cores_divisible_by_tiles(num_tiles_dim_2, num_cores_x_limit/dim_1_num_batches);
    // uint32_t num_cores_y = get_max_cores_divisible_by_tiles(num_tiles_dim_3, num_cores_y_limit);
    uint32_t dim_2_num_batches = num_tiles_dim_2 / num_cores_x;
    uint32_t dim_3_num_batches = num_tiles_dim_3 / num_cores_y;

    uint32_t per_core_tiles_x = dim_2_num_batches;
    uint32_t per_core_tiles_y = dim_3_num_batches;
    uint32_t per_core_tiles = per_core_tiles_x * per_core_tiles_y * z;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_chunks;
    uint32_t num_tiles_per_tensor_x = per_core_tiles_x;
    uint32_t num_tiles_per_tensor_y = per_core_tiles_y / num_chunks;

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;

    uint32_t num_cores_r = num_cores_y;
    uint32_t num_cores_c = num_cores_x;

    CoreRange all_cores{
        .start = {(std::size_t)start_core_x, (std::size_t)start_core_y},
        .end = {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1},
    };

    bool tile_dtype_is_bfloat16 = input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    TT_ASSERT(q_buffer->buffer_type() == k_buffer->buffer_type(), "Output buffers should be the same type");

    uint32_t num_tiles_per_z = num_tiles_per_tensor_x * num_tiles_per_tensor_y;
    uint32_t z_stride = num_tiles_per_z * num_chunks;
    uint32_t y_stride = num_tiles_per_tensor_y * num_chunks;

    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)tile_dtype_is_bfloat16,
        // by default in dram
        (std::uint32_t)in0_is_dram,

        // READER COMPILE TIME ARGS
        (std::uint32_t)z,
        (std::uint32_t)num_tiles_per_tensor,
        (std::uint32_t)num_tiles_per_tensor_x,  // out_num_tiles_per_tensor
        (std::uint32_t)num_tiles_per_tensor_y,  // out_num_tiles_per_tensor
        (std::uint32_t)z_stride,
        (std::uint32_t)y_stride};

    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t)tile_dtype_is_bfloat16,
        (std::uint32_t)out_is_dram,

        // WRITER COMPILE TIME ARGS
        (std::uint32_t)num_tiles_per_tensor  // out_num_tiles_per_tensor
    };

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_tm_tile_layout_split_qk.cpp",
        all_cores,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_tm_tile_layout_split_qk.cpp",
        all_cores,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // Dummy compute kernel
    std::vector<uint32_t> compute_args = {0};  // dummy
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto dummy_compute_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        all_cores,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program, src0_cb_index, all_cores, num_input_tiles, num_input_tiles * single_tile_size, cb_data_format);

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            uint32_t core_id = core_idx_x + core_idx_y * num_cores_c;

            std::vector<uint32_t> reader_runtime_args = {
                core_id * per_core_tiles,
                (std::uint32_t)in0_buffer->address(),  // in0_tensor_addr
            };

            std::vector<uint32_t> writer_runtime_args = {
                core_id * num_tiles_per_tensor,
                (std::uint32_t)q_buffer->address(),  // first base addr
                (std::uint32_t)k_buffer->address(),  // second base addr
            };

            tt_metal::SetRuntimeArgs(reader_kernel, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(writer_kernel, core, writer_runtime_args);
        }
    }

    auto override_runtime_args_callback = [
            reader_kernel,
            writer_kernel,
            num_cores_r,
            num_cores_c,
            start_core_x,
            start_core_y
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_0_dram_buffer = output_buffers.at(0);
        auto dst_1_dram_buffer = output_buffers.at(0);

        for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

                {
                    auto runtime_args = GetRuntimeArgs(reader_kernel, core);
                    runtime_args[1] = src_dram_buffer->address();
                    SetRuntimeArgs(reader_kernel, core, runtime_args);
                }

                {
                    auto runtime_args = GetRuntimeArgs(writer_kernel, core);
                    runtime_args[1] = dst_0_dram_buffer->address();
                    runtime_args[2] = dst_1_dram_buffer->address();
                    SetRuntimeArgs(writer_kernel, core, runtime_args);
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks SplitLastDimQKTiled::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    return split_last_dim_qk_tiled(input_tensor, output_tensors, this->output_mem_config);
}

std::vector<Tensor> split_last_dim_qk_tiled(const Tensor &input_tensor, const MemoryConfig &mem_config) {
    SplitLastDimQKTiled op(mem_config);

    tt_metal::Device *device;
    // Get the device
    if (input_tensor.storage_type() == StorageType::HOST) {
        device = AutoFormat::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = input_tensor.device();
    }

    auto input_shape = input_tensor.shape();
    auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_shape);
    if (AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape)) {
        return operation::run(op, {input_tensor});
    } else {
        TT_ASSERT(input_tensor.memory_config().buffer_type == tt_metal::BufferType::DRAM, "Untiled splits should be in DRAM");
        TT_ASSERT(mem_config.buffer_type == tt_metal::BufferType::DRAM, "Untiled splits should be in DRAM");
        auto device = input_tensor.device();
        auto output_shape = op.compute_output_shapes({input_tensor}).at(0);
        const auto padded_tensor = AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape);
        auto output_tensors = operation::run(op, {padded_tensor});
        for (auto &output_tensor : output_tensors) {
            output_tensor = AutoFormat::format_output_tensor(output_tensor, output_shape, device);
        }
        return output_tensors;
    }
}

}  // namespace tt_metal

}  // namespace tt
