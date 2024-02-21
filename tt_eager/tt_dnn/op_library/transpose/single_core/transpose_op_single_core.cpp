// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/transpose/transpose_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

std::vector< std::vector<uint32_t> >  get_runtime_args_wh(const Tensor &input_tensor,
                                                       Tensor &output_tensor
                                                        )
{

    auto input_shape = input_tensor.get_legacy_shape();
    auto output_shape = output_tensor.get_legacy_shape();
    uint32_t W = input_shape[3], H = input_shape[2], NC = input_shape[1]*input_shape[0];
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = input_tensor.volume() / TILE_HW;
    std::vector<uint32_t> reader_runtime_args = {input_tensor.buffer()->address(),
                                                NC, Ht, Wt, Ht*Wt,
                                                };
    std::vector<uint32_t> compute_runtime_args = {num_tensor_tiles};
    std::vector<uint32_t> writer_runtime_args =   {output_tensor.buffer()->address(),num_tensor_tiles, 0};
    return {reader_runtime_args, compute_runtime_args, writer_runtime_args};

}


operation::ProgramWithCallbacks transpose_wh_single_core(const Tensor &a, Tensor& output) {

    const auto shape = a.get_legacy_shape();

    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume()/TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    Shape output_shape = output.get_legacy_shape();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
		.set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t) src0_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    //TODO: move this kernel, currently being used in reduce, can't move to op library
    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transpose/kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{writer_compile_time_args});



    auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transpose/kernels/compute/transpose_wh.cpp",
        core,
        tt_metal::ComputeConfig{}
    );

    auto all_runtime_args = get_runtime_args_wh(a, output);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        all_runtime_args[0]
    );

    tt_metal::SetRuntimeArgs(
        program,
        eltwise_binary_kernel_id,
        core,
        all_runtime_args[1]
    );

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        all_runtime_args[2]
    );

    auto override_runtime_args_callback = [reader_kernel_id, eltwise_binary_kernel_id, writer_kernel_id](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>> & ,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        auto all_runtime_args = get_runtime_args_wh(src_tensor, dst_tensor);

        CoreCoord core = {0, 0};

        {
            SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args[0]);
        }

        {
            SetRuntimeArgs(program, eltwise_binary_kernel_id, core, all_runtime_args[1]);
        }

        {
            SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args[2]);
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};

}

std::pair< std::vector<uint32_t>, std::vector<uint32_t> > get_runtime_args_hc(const Tensor &input_tensor, Tensor &output_tensor){

    const auto input_shape = input_tensor.get_legacy_shape();
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t HW = H*W;
    uint32_t HW_bytes = HW * input_tensor.element_size();
    uint32_t CHW = C*H*W;
    uint32_t CHW_bytes = CHW * input_tensor.element_size();

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;
    uint32_t Ct = C/TILE_HEIGHT;
    uint32_t CtHWt = Ct*H*Wt;
    uint32_t CtWt = Ct * Wt;

    uint32_t num_tensor_tiles = input_tensor.volume() / TILE_HW;

    std::vector<uint32_t> reader_runtime_args = {
            input_tensor.buffer()->address(),
            (std::uint32_t) Wt,
            (std::uint32_t) H,
            (std::uint32_t) Ct,
            (std::uint32_t) HW_bytes,
            (std::uint32_t) CHW_bytes,
            0, num_tensor_tiles,
            0, 0, 0, 0, 0, 0
            };

    std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(), num_tensor_tiles};

    return {reader_runtime_args, writer_runtime_args};

}

operation::ProgramWithCallbacks transpose_hc_single_core(const Tensor &a, Tensor &output) {
    // 16 is size of face row
    uint32_t sub_tile_line_bytes = 16 * a.element_size();


    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    log_debug("transpose_hc_single_core");
    log_debug("sub_tile_line_bytes: {}", sub_tile_line_bytes);
    log_debug("src0_cb_data_format: {}", src0_cb_data_format);
    log_debug("src0_single_tile_size: {}", src0_single_tile_size);


    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    Shape output_shape = output.get_legacy_shape();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
		.set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) sub_tile_line_bytes,
        (std::uint32_t) (src0_cb_data_format == tt::DataFormat::Float32)
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transpose/kernels/dataflow/reader_unary_transpose_hc_interleaved_partitioned.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{writer_compile_time_args});


    auto all_runtime_args = get_runtime_args_hc(a, output);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        all_runtime_args.first
    );

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        all_runtime_args.second
    );

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>> & ,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        auto all_runtime_args = get_runtime_args_hc(src_tensor, dst_tensor);

        CoreCoord core = {0, 0};

        {
            SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args.first);
        }

        {
            SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args.second);
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}

std::pair< std::vector<uint32_t>, std::vector<uint32_t> > get_runtime_args_cn(const Tensor &input_tensor, Tensor &output_tensor){

    const auto input_shape = input_tensor.get_legacy_shape();
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_tiles;

    std::vector<uint32_t> reader_runtime_args =  {input_tensor.buffer()->address(),
                                                N, C, HtWt, CHtWt - HtWt, NCHtWt - HtWt};

    std::vector<uint32_t> writer_runtime_args = {output_tensor.buffer()->address(),
            num_tensor_tiles, 0};
    return {reader_runtime_args, writer_runtime_args};
}

operation::ProgramWithCallbacks transpose_cn_single_core(const Tensor &a, Tensor &output) {

    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");



    tt_metal::Program program = tt_metal::CreateProgram();

    CoreRange core({0, 0}, {0, 0});

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    log_debug("transpose_cn_single_core");
    log_debug("cb_data_format: {}", cb_data_format);
    log_debug("single_tile_size: {}", single_tile_size);

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();


    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t) src0_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) dst_is_dram
    };

    tt_metal::KernelHandle reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/transpose/kernels/dataflow/reader_unary_transpose_cn_interleaved.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    tt_metal::KernelHandle writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{writer_compile_time_args});

    auto all_runtime_args = get_runtime_args_cn(a, output);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel_id,
        core,
        all_runtime_args.first
    );

    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel_id,
        core,
        all_runtime_args.second
    );

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id](
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>> & ,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_tensor = input_tensors.at(0);

        auto dst_tensor = output_tensors.at(0);

        auto all_runtime_args = get_runtime_args_cn(src_tensor, dst_tensor);
        CoreCoord core = {0, 0};

        {
            SetRuntimeArgs(program, reader_kernel_id, core, all_runtime_args.first);
        }

        {
            SetRuntimeArgs(program, writer_kernel_id, core, all_runtime_args.second);
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}


operation::ProgramWithCallbacks transpose_single_core(const Tensor &a, Tensor &output, TransposeOpDim transpose_dim) {
    if (transpose_dim == TransposeOpDim::WH){
        return transpose_wh_single_core(a, output);
    } else if (transpose_dim == TransposeOpDim::HC) {
        return transpose_hc_single_core(a, output);
    } else if (transpose_dim == TransposeOpDim::CN) {
        return transpose_cn_single_core(a, output);
    } else {
        TT_THROW("Unsupported Transpose Op Dim");
    }
}

}  // namespace tt_metal

}  // namespace tt
