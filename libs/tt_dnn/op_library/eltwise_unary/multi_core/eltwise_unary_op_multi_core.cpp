#include <algorithm>

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_dnn/op_library/program_cache.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

tt_metal::Program eltwise_unary_multi_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type) {

    tt_metal::Program program{};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_and_storage_grid_size, num_tiles);

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

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src1_cb_index,
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
    bool tile_size_is_power_of_two = (ceil(std::log2(single_tile_size)) == floor(std::log2(single_tile_size)));
    std::vector<uint32_t> reader_writer_compile_time_args;
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_writer_compile_time_args = {1, (std::uint32_t)std::log2(single_tile_size)};
    } else {
        reader_writer_compile_time_args = {0, 0};
    }
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank_start_id.cpp",
        all_cores,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank_start_id.cpp",
        all_cores,
        reader_writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1, // per_core_block_cnt
        1 // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel_group_1 = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        compute_kernel_args_group_1,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    eltwise_unary_op_utils::add_defines(eltwise_unary_kernel_group_1, op_type);
    if(!core_group_2.ranges().empty()){
        vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2, // per_core_block_cnt
            1 // per_core_block_size
        };

        auto eltwise_unary_kernel_group_2 = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            compute_kernel_args_group_2,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        eltwise_unary_op_utils::add_defines(eltwise_unary_kernel_group_2, op_type);
    }

    if (not program_cache::is_enabled()) {

        tt_metal::Buffer *src0_dram_buffer = a.buffer();
        auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

        tt_metal::Buffer *dst_dram_buffer = output.buffer();
        TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
        auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};
            uint32_t num_tiles_per_core;
            if (core_group_1.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_1;
            } else if (core_group_2.core_coord_in_core_ranges(core)) {
                num_tiles_per_core = num_tiles_per_core_group_2;
            } else {
                TT_ASSERT(false, "Core not in specified core ranges");
            }
            tt_metal::WriteRuntimeArgsToDevice(
                device,
                unary_reader_kernel,
                core,
                {src0_dram_buffer->address(),
                uint32_t(dram_src0_noc_xy.x),
                uint32_t(dram_src0_noc_xy.y),
                num_tiles_per_core,
                num_tiles_written, 0 /*disable scaler*/ }
            );

            tt_metal::WriteRuntimeArgsToDevice(
                device,
                unary_writer_kernel,
                core,
                {dst_dram_buffer->address(),
                uint32_t(dram_dst_noc_xy.x),
                uint32_t(dram_dst_noc_xy.y),
                num_tiles_per_core,
                num_tiles_written }
            );
            num_tiles_written+=num_tiles_per_core;
        }
    }

    return program;
}

}  // namespace tt_metal

}  // namespace tt
