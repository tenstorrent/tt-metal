#include <algorithm>
#include "tt_dnn/op_library/reduce/reduce_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;
using u32 = std::uint32_t;

namespace tt {

namespace tt_metal {

Program reduce_multi_core_w(const Tensor &a, Tensor& output, ReduceOpMath::Enum reduce_op, ReduceOpDim::Enum reduce_dim, float scaler) {

    TT_ASSERT(reduce_dim == ReduceOpDim::W);
    const auto shape = a.shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    uint32_t HW = H*W;
    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    tt_metal::Program program = tt_metal::Program();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to reduce op needs to be on device!");
    TT_ASSERT(a.device() != nullptr, "Operand to reduce op needs to be on device!");

    uint32_t single_tile_size = 2 * 1024;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume()/TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;
    auto num_rows = NC * Ht;
    auto num_cores = std::min(num_rows, num_cores_x * num_cores_y);
    std::vector<uint32_t> num_rows_per_core(num_cores, num_rows / num_cores);
    for(uint32_t i = 0; i < num_rows % num_cores; i++){
        num_rows_per_core[i]++;
    }

    string compute_kernel_name = reduce_op_utils::dim_to_kernel_name(reduce_dim, reduce_op);

    std::vector<tt_metal::DataMovementKernel *> unary_reader_kernels;
    std::vector<tt_metal::DataMovementKernel *> unary_writer_kernels;
    for (uint32_t i = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            DataFormat::Float16_b
        );

        auto cb_scaler = tt_metal::CreateCircularBuffer(
            program,
            device,
            CB::c_in2,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 2;
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            DataFormat::Float16_b
        );
        bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
        std::vector<uint32_t> reader_writer_compile_time_args;
        if (tile_size_is_power_of_two) {
            // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
            reader_writer_compile_time_args = {1, (std::uint32_t)log2(single_tile_size)};
        } else {
            reader_writer_compile_time_args = {0, 0};
        }
        tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_unary_8bank_start_id.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);
        unary_reader_kernels.push_back(reader_kernel);

        tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_8bank_start_id.cpp",
            core,
            reader_writer_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);
        unary_writer_kernels.push_back(writer_kernel);

        vector<uint32_t> compute_kernel_args = {
            uint32_t(*reinterpret_cast<uint32_t*>(&scaler)), // scaler
            num_rows_per_core[i], // Ht
            Wt, // Wt
            1, // NC
        };
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;

        auto reduce_compute_kernel = tt_metal::CreateComputeKernel(
            program,
            compute_kernel_name,
            core,
            compute_kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        reduce_op_utils::add_defines(reduce_compute_kernel, reduce_op, reduce_dim);
    }

    uint32_t out_dim_divider = Wt;
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tensor_tiles_per_core = num_rows_per_core[i]*Wt;
        tt_metal::WriteRuntimeArgsToDevice(
            device, unary_reader_kernels[i], core,
            {
                a.buffer()->address(),
                0, // unused by multibank reader
                0, // unused by multibank reader
                num_tensor_tiles_per_core,
                num_tiles_read, // tile index of row to start reading from
                uint32_t(*reinterpret_cast<uint32_t*>(&scaler)), // scaler
            }
        );

        tt_metal::WriteRuntimeArgsToDevice(
            device, unary_writer_kernels[i], core,
            {
                output.buffer()->address(),
                0, // unused by multibank writer
                0, // unused by multibank writer
                num_tensor_tiles_per_core / out_dim_divider, // number of tiles to write
                num_tiles_read / out_dim_divider // output tile start index
            }
        );
        num_tiles_read+=num_tensor_tiles_per_core;
    }

    // output does not hold any data, contains pointer to buffer on device with the data
    return program;
}

}  // namespace tt_metal

}  // namespace tt
