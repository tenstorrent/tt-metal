// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// Each element of outer vector corresponds to a core
// Each core has a pair of std::vector<uint32_t>
// First of pair is reader args
// Second of pair is writer args
std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > get_runtime_args_mc_cn(const Tensor &input_tensor,
                                                                                        Tensor &output_tensor,
                                                                                        uint32_t num_cores_total,
                                                                                        uint32_t num_cores,
                                                                                        uint32_t num_cores_y,
                                                                                        CoreRangeSet core_group_1,
                                                                                        uint32_t num_tiles_per_core_group_1,
                                                                                        CoreRangeSet core_group_2,
                                                                                        uint32_t num_tiles_per_core_group_2
                                                                                        ){

    auto input_buffer = input_tensor.buffer();
    auto output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_tiles;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    std::vector<std::pair<std::vector<uint32_t>, std::vector<uint32_t> > > ret_val;
    ret_val.reserve(num_cores_total);

    for(uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            //no-op
            num_tiles_per_core = 0;
        }
        uint32_t hw = num_tiles_read % HtWt;
        uint32_t n = num_tiles_read / HtWt % N;

        std::vector<uint32_t> reader_runtime_args = {
            input_buffer->address(),
            N,
            C,
            HtWt,
            batch_step,
            channel_step,
            num_tiles_per_core,
            num_tiles_read,
            hw,
            n
        };


        std::vector<uint32_t> writer_runtime_args = {
                output_buffer->address(),
                num_tiles_per_core,
                num_tiles_read
            };
        ret_val[i] = std::make_pair(std::move(reader_runtime_args), std::move(writer_runtime_args));
        num_tiles_read += num_tiles_per_core;
    }

    return ret_val;
}

operation::ProgramWithCallbacks transpose_cn_single_core(const Tensor &a, Tensor &output) {

    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    tt_metal::Program program = tt_metal::Program();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    uint32_t num_tensor_tiles = a.volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);


    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src0_is_dram
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transpose/kernels/dataflow/reader_unary_transpose_cn_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args});

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    auto all_runtime_args =  std::move(get_runtime_args_mc_cn(a, output, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2));

    for(uint32_t i = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            all_runtime_args[i].first
        );

        tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            all_runtime_args[i].second

        );
    }


    auto override_runtime_args_callback = [
            reader_kernel_id,
            writer_kernel_id,
            compute_with_storage_grid_size

        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        uint32_t num_cores_x = compute_with_storage_grid_size.x;
        uint32_t num_cores_y = compute_with_storage_grid_size.y;

        uint32_t num_cores_total = num_cores_x * num_cores_y;

        uint32_t num_tensor_tiles = src_tensor.volume() / TILE_HW;

        auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);
        auto all_runtime_args =  get_runtime_args_mc_cn(src_tensor, dst_tensor, num_cores_total, num_cores, num_cores_y, core_group_1, num_tiles_per_core_group_1, core_group_2, num_tiles_per_core_group_2);

        for(uint32_t i = 0; i < num_cores_total; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[i].first);
            }

            {
                SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[i].second);
            }
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}


}  // namespace tt_metal

}  // namespace tt
