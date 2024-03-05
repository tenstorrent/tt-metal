// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_arange/moreh_arange_op.hpp"

#include <cmath>

#include "common/test_tiles.hpp"
#include "tt_eager/tt_dnn/op_library/moreh_helper_functions.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

operation::ProgramWithCallbacks moreh_arange_(
    const Tensor &input, float start, float end, float step, const CoreRange core_range) {
    // split work
    // N and C are always 1
    // H is always TILE_HEIGHT
    auto shape = input.get_legacy_shape();
    auto W = shape[3];
    auto Wt = W / TILE_WIDTH;

    uint32_t units_to_divide = Wt;
    uint32_t core_w = core_range.end.x - core_range.start.x + 1;
    uint32_t core_h = core_range.end.y - core_range.start.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(core_range, units_to_divide);

    Program program = Program();

    // create circular buffers
    tt::DataFormat data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());

    CreateCircularBuffer(
        program,
        all_cores,
        data_format,
        {
            {CB::c_out0, 1},  // output
        });

    // create read/wrtie kernel
    bool dst_is_dram = input.buffer()->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    std::map<string, string> writer_defines;

    auto kernel_id = CreateWriteKernel(
        program,
        "tt_eager/tt_dnn/op_library/moreh_arange/kernels/writer_moreh_arange.cpp",
        all_cores,
        {dst_is_dram},
        writer_defines);

    // Set Runtime Args
    auto core_x_offset = core_range.start.x;
    auto core_y_offset = core_range.start.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {i / core_h + core_x_offset, i % core_h + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_FATAL(false, "Core not in specified core ranges");
        }

        vector<uint32_t> writer_args = {
            input.buffer()->address(),
            tile_offset,
            num_tiles_per_core,
            *reinterpret_cast<uint32_t *>(&start),
            *reinterpret_cast<uint32_t *>(&step)};

        SetRuntimeArgs(program, kernel_id, core, writer_args);

        tile_offset += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [kernel_id = kernel_id, num_cores, core_h](
                                              const Program &program,
                                              const std::vector<Buffer *> &input_buffers,
                                              const std::vector<Buffer *> &output_buffers) {
        TT_ASSERT(input_buffers.size() == 1);

        auto src_dram_buffer = input_buffers.at(0);

        for (uint32_t icore = 0; icore < num_cores; icore++) {
            CoreCoord core = {icore / core_h, icore % core_h};

            {
                auto runtime_args = GetRuntimeArgs(program, kernel_id, core);
                runtime_args[0] = src_dram_buffer->address();
                SetRuntimeArgs(program, kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void MorehArange::validate(const std::vector<Tensor> &input_tensors) const {
    TT_ASSERT(this->step > 0 || this->step < 0, "step must be nonzero");
    TT_ASSERT(
        ((this->step > 0) && (this->end >= this->start)) || ((this->step < 0) && (this->end <= this->start)),
        "upper bound and larger bound inconsistent with step sign");
}

std::vector<Shape> MorehArange::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    if (this->inplace) {
        return {};
    } else {
        // return size is ceil((end - start) / step)
        uint32_t num_elems = static_cast<uint32_t>(ceil(((this->end - this->start) / this->step)));
        Shape output_shape = {1, 1, TILE_HEIGHT, ((num_elems + TILE_WIDTH - 1) / TILE_WIDTH) * TILE_WIDTH};
        return {output_shape};
    }
}

std::vector<Tensor> MorehArange::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    if (this->inplace) {
        return {};
    } else {
        return operation::generic_create_output_tensors(
            *this, input_tensors, DataType::BFLOAT16, Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks MorehArange::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    return moreh_arange_(
        this->inplace ? input_tensors.at(0) : output_tensors.at(0),
        this->start,
        this->end,
        this->step,
        this->core_range);
}

Tensor moreh_arange(float start, float end, float step, const Tensor &any, const MemoryConfig &output_mem_config) {
    auto device = any.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    return operation::run(
               MorehArange{
                   .start = start,
                   .end = end,
                   .step = step,
                   .core_range = all_cores,
                   .output_mem_config = output_mem_config,
                   .inplace = false},
               {any})
        .at(0);
}

Tensor moreh_arange_inplace(Tensor &input_tensor, float start, float end, float step) {
    auto device = input_tensor.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange all_cores({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});

    operation::run(
        MorehArange{
            .start = start,
            .end = end,
            .step = step,
            .core_range = all_cores,
            .output_mem_config = input_tensor.memory_config(),
            .inplace = true},
        {input_tensor});
    return input_tensor;
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
