// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks eltwise_unary_multi_core_height_sharded(const Tensor &input, Tensor &output, const std::vector<UnaryWithParam> op_chain){
    Program program = CreateProgram();
    Device *device = input.device();
    uint32_t in_w = input.get_legacy_shape()[2];

    auto shard_spec = input.shard_spec().value();
    auto all_cores = shard_spec.grid;

    uint32_t ncores = shard_spec.num_cores();
    uint32_t ncores_x = device->compute_with_storage_grid_size().x;
    uint32_t ncores_nhw = ncores;

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(out_shard_spec.num_cores() == ncores, "Output tensor should have same number of cores {} as input tensor {}", out_shard_spec.num_cores(), ncores);

    uint32_t in_nsticks_per_core = shard_spec.shape[0];
    TT_FATAL(in_nsticks_per_core % in_w == 0, "Restriction: Input sticks per core {} should be divisible by input width {}. TODO to remove this restriction", in_nsticks_per_core, in_w);

    DataFormat act_df = tt_metal::datatype_to_dataformat_converter(input.get_dtype()); //fix this later. the vaklue is already there.
    DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);

    uint32_t num_tile_per_core = shard_spec.numel() * datum_size(act_df) /input_tile_size;
    std::cout << shard_spec.numel() << "  " << input_tile_size << "  " << num_tile_per_core << std::endl;

    uint32_t in_cb_id = CB::c_in0; //match this with kernel
    uint32_t buffering_factor = 1;  // data is already fully buffered in the CBs since its sharded
    uint32_t aligned_input_tile_nbytes = round_up_to_mul32(input_tile_size); //will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
    uint32_t in_cb_npages = num_tile_per_core * buffering_factor;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{in_cb_id, act_df}})
                                          .set_page_size(in_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*input.buffer());
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    // output sharded CB with upsampled data
    uint32_t out_cb_id = CB::c_out0;   //need to change this guy
    /*uint32_t aligned_output_stick_nbytes = round_up_to_mul32(input_tile_size);
    uint32_t out_cb_pagesize = aligned_output_stick_nbytes;
    uint32_t out_cb_npages = in2 * buffering_factor;*/
    CircularBufferConfig out_cb_config = CircularBufferConfig(
                                            in_cb_pagesize * in_cb_npages,
                                            {{out_cb_id, out_df}})
                                          .set_page_size(out_cb_id, in_cb_pagesize)
                                          .set_globally_allocated_address(*output.buffer());
    auto out_cb = tt_metal::CreateCircularBuffer(program, all_cores, out_cb_config);

    log_debug(LogOp, "input_cb: {}, npages: {}, pagesize: {}", in_cb_id, in_cb_npages, in_cb_pagesize);
    //log_debug(LogOp, "output_cb: {}, npages: {}, pagesize: {}", out_cb_id, out_cb_npages, out_cb_pagesize);
    log_debug(LogOp, "input_stick_nbytes: {}", input_stick_nbytes);
    log_debug(LogOp, "ncores: {}, ncores_x: {}", ncores, ncores_x);
    log_debug(LogOp, "input_tile_size: {}", input_tile_size);

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        in_cb_id,
        out_cb_id,
        //(uint32_t)src_is_dram
        };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_cb_id,
        (std::uint32_t) dst_is_dram
    };
    CoreRange temp_core({0, 0}, {0, 0});
    std::map<string, string> kernel_defines;

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_sharded.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    vector<uint32_t> compute_kernel_args_group_1 = {
         num_tile_per_core, // per_core_block_cnt
         1 // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = std::all_of(op_chain.begin(), op_chain.end(), [](const auto& u) {return eltwise_unary_op_utils::get_op_approx_mode(u.op_type);});
    std::map<string, string> unary_defines = eltwise_unary_op_utils::get_block_defines(op_chain);
    auto eltwise_unary_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::LoFi,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines
        }
    );

    uint32_t start_input_stick_id = 0;
    std::cout << "element 8 " << std::endl;
    for (int32_t core = 0; core < ncores_nhw; ++core) {
            CoreCoord core_coord(core % ncores_x, core / ncores_x); // logical
            tt_metal::SetRuntimeArgs(
                program,
                unary_reader_kernel_id,
                core_coord,
                {
                    (uint32_t)(num_tile_per_core),
                    (uint32_t)(input_tile_size)
                }
            );
            tt_metal::SetRuntimeArgs(
                program,
                eltwise_unary_kernel_group_1_id,
                core_coord,
                {
                }
            );

    }


    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            ncores_nhw,
            ncores_x
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);
        for (uint32_t core = 0; core < ncores_nhw; core++){
            std::cout << "tesing overriding the kernels call" << std::endl;
            CoreCoord core_coord(core % ncores_x, core / ncores_x); //
            {

            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks eltwise_unary_multi_core(const Tensor &a, Tensor &output, const std::vector<UnaryWithParam> op_chain) {
    if(a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED){
        return eltwise_unary_multi_core_height_sharded(a, output, op_chain);
    }

    tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t num_tiles = a.volume() / TILE_HW;

    tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    std::cout << "num_cores: " << num_cores << " num_tiles_per_core_group_1: " << num_tiles_per_core_group_1 << " num_tiles_per_core_group_2: " << num_tiles_per_core_group_2 << std::endl;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto src_buffer = a.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1, // per_core_block_cnt
        1 // per_core_block_size
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = std::all_of(op_chain.begin(), op_chain.end(), [](const auto& u) {return eltwise_unary_op_utils::get_op_approx_mode(u.op_type);});
    std::map<string, string> unary_defines = eltwise_unary_op_utils::get_block_defines(op_chain);
    auto eltwise_unary_kernel_group_1_id = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines
        }
    );

    if(!core_group_2.ranges().empty()){
        vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2, // per_core_block_cnt
            1 // per_core_block_size
        };

        auto eltwise_unary_kernel_group_2_id = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines
            }
        );
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program,
            unary_reader_kernel_id,
            core,
            {
                src_buffer->address(),
                num_tiles_per_core,
                num_tiles_written
            }
        );

        tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {
                dst_buffer->address(),
                num_tiles_per_core,
                num_tiles_written
            }
        );
        num_tiles_written+=num_tiles_per_core;
    }

    auto override_runtime_args_callback = [
            unary_reader_kernel_id,
            unary_writer_kernel_id,
            num_cores,
            num_cores_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
