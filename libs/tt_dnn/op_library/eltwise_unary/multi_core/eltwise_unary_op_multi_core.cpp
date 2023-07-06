#include <algorithm>

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_unary_multi_core(const Tensor &a, Tensor &output, UnaryOpType::Enum op_type,std::optional<float> param /* = {} */) {
    tt_metal::Program program{};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to eltwise unary needs to be on device!");
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
        src0_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
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
        ouput_cb_index,
        all_cores,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    // Op not uplifted for L1 yet, but need to provide arg to kernel
    bool dst_is_dram = true;
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(DataFormat::Float16_b), (uint32_t)dst_is_dram};

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
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1, // per_core_block_cnt
        1 // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = eltwise_unary_op_utils::get_op_approx_mode(op_type);
    auto eltwise_unary_kernel_group_1 = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        compute_kernel_args_group_1,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    eltwise_unary_op_utils::add_defines(eltwise_unary_kernel_group_1, op_type, param);
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

        eltwise_unary_op_utils::add_defines(eltwise_unary_kernel_group_2, op_type, param);
    }

    auto src_dram_buffer = a.buffer();
    auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

    auto dst_dram_buffer = output.buffer();
    auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

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

        tt_metal::SetRuntimeArgs(
            unary_reader_kernel,
            core,
            {
                src_dram_buffer->address(),
                uint32_t(src_dram_noc_xy.x),
                uint32_t(src_dram_noc_xy.y),
                num_tiles_per_core,
                num_tiles_written,
                0 /*disable scaler*/
            }
        );

        tt_metal::SetRuntimeArgs(
            unary_writer_kernel,
            core,
            {
                dst_dram_buffer->address(),
                uint32_t(dst_dram_noc_xy.x),
                uint32_t(dst_dram_noc_xy.y),
                num_tiles_per_core,
                num_tiles_written
            }
        );
        num_tiles_written+=num_tiles_per_core;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel,
            unary_writer_kernel,
            num_cores,
            num_cores_y
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);
        auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

        auto dst_dram_buffer = output_buffers.at(0);
        auto dst_dram_noc_xy = dst_dram_buffer->noc_coordinates();

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(unary_reader_kernel, core);
                runtime_args[0] = src_dram_buffer->address();
                runtime_args[1] = uint32_t(src_dram_noc_xy.x);
                runtime_args[2] = uint32_t(src_dram_noc_xy.y);
                SetRuntimeArgs(unary_reader_kernel, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(unary_writer_kernel, core);
                runtime_args[0] = dst_dram_buffer->address();
                runtime_args[1] = uint32_t(dst_dram_noc_xy.x);
                runtime_args[2] = uint32_t(dst_dram_noc_xy.y);
                SetRuntimeArgs(unary_writer_kernel, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
