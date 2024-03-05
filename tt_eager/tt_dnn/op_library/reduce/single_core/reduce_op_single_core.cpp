// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/reduce/reduce_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks reduce_single_core(const Tensor &a, Tensor& output, ReduceOpMath reduce_op, ReduceOpDim reduce_dim, float scaler) {

    const auto shape = a.get_legacy_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1]*shape[0];
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;
    if (reduce_dim == ReduceOpDim::HW)
        scaler = sqrt(scaler);

    uint32_t num_tensor_tiles = NC*H*W / TILE_HW;

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});


    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    // Scaler datatype is hardcoded bfloat16 due to tile creation in reader
    tt::DataFormat scaler_cb_data_format = tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt_metal::detail::TileSize(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    uint32_t num_tiles = a.volume()/TILE_HW;

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    tt_metal::CircularBufferConfig cb_scaler_config = tt_metal::CircularBufferConfig(num_input_tiles * scaler_single_tile_size, {{CB::c_in2, scaler_cb_data_format}})
		.set_page_size(CB::c_in2, scaler_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_scaler_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    bfloat16 bfloat_scaler_value = bfloat16(scaler);
    uint32_t packed_scaler_value = pack_two_bfloat16_into_uint32({bfloat_scaler_value, bfloat_scaler_value});
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        packed_scaler_value
    };

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,
        (std::uint32_t) dst_is_dram
    };
    std::map<string, string> reader_defines;
    if (reduce_dim == ReduceOpDim::H) {
        reader_defines["REDUCE_SCALER"] = "1";
    }
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        reduce_dim == ReduceOpDim::H ?
            "tt_eager/tt_dnn/op_library/transpose/kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp" :
            "tt_eager/tt_dnn/op_library/reduce/kernels/dataflow/reader_unary_reduce_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_kernel_args = {
        Ht, // Ht
        Wt, // Wt
        NC, // NC
    };
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    TT_ASSERT(int(reduce_dim) >= 0 && int(reduce_dim) <= magic_enum::enum_count<ReduceOpDim>());

    string compute_kernel_name = reduce_op_utils::dim_to_kernel_name(reduce_dim, reduce_op);

    auto reduce_compute_kernel_id = tt_metal::CreateKernel(
        program,
        compute_kernel_name,
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = reduce_op_utils::get_defines(reduce_op, reduce_dim)}
    );

    if (reduce_dim == ReduceOpDim::H) {
        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core,
            {
                a.buffer()->address(),
                NC, Ht, Wt, HtWt
            }
        );
    } else {
        tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core,
            {
                a.buffer()->address(),
                num_tensor_tiles, 0
            }
        );
    }

    uint32_t out_dim_divider = 1;
    switch (reduce_dim) {
        case ReduceOpDim::H: out_dim_divider = Ht; break;
        case ReduceOpDim::W: out_dim_divider = Wt; break;
        case ReduceOpDim::HW: out_dim_divider = Ht*Wt; break;
        default: TT_ASSERT(false && "Unsupported reduce_dim!");
    }

    tt_metal::SetRuntimeArgs(
        program, writer_kernel_id, core,
        {
            output.buffer()->address(),
            num_tensor_tiles/out_dim_divider,
            0
        }
    );

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
