#include <algorithm>
#include "tt_dnn/op_library/reduce/reduce_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

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

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume()/TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;
    auto num_rows = NC * Ht;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] = split_work_to_cores(compute_and_storage_grid_size, num_rows);

    string compute_kernel_name = reduce_op_utils::dim_to_kernel_name(reduce_dim, reduce_op);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    auto cb_scaler = tt_metal::CreateCircularBuffers(
        program,
        device,
        CB::c_in2,
        all_cores,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        all_cores,
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
        all_cores,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank_start_id.cpp",
        all_cores,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args_group_1 = {
        uint32_t(*reinterpret_cast<uint32_t*>(&scaler)), // scaler
        num_rows_per_core_group_1, // Ht
        Wt, // Wt
        1, // NC
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto reduce_compute_kernel_group_1 = tt_metal::CreateComputeKernel(
        program,
        compute_kernel_name,
        core_group_1,
        compute_kernel_args_group_1,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    reduce_op_utils::add_defines(reduce_compute_kernel_group_1, reduce_op, reduce_dim);

    if(!core_group_2.ranges().empty()){
        vector<uint32_t> compute_kernel_args_group_2 = {
            uint32_t(*reinterpret_cast<uint32_t*>(&scaler)), // scaler
            num_rows_per_core_group_2, // Ht
            Wt, // Wt
            1, // NC
        };

        auto reduce_compute_kernel_group_2 = tt_metal::CreateComputeKernel(
            program,
            compute_kernel_name,
            core_group_2,
            compute_kernel_args_group_2,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        reduce_op_utils::add_defines(reduce_compute_kernel_group_2, reduce_op, reduce_dim);
    }

    uint32_t out_dim_divider = Wt;
    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_rows_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        uint32_t num_tensor_tiles_per_core = num_rows_per_core*Wt;
        tt_metal::WriteRuntimeArgsToDevice(
            device, reader_kernel, core,
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
            device, writer_kernel, core,
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
