// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/pad/pad_op.hpp"
#include "tt_dnn/op_library/copy/copy_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tensor/owned_buffer_functions.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks pad_rm_reader_writer(const Tensor &a,
                                                     Tensor &output,
                                                     const Shape &output_tensor_shape,
                                                     const Shape &input_tensor_start,
                                                     const float pad_value) {
    Program program{};

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    // construct const buffer with the pad_value
    Device *device = a.device();
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    const Tensor pad_value_const_tensor = Tensor(OwnedStorage{pad_value_const_buffer},
                                                 Shape({1, 1, 1, pad_value_const_buffer_size}),
                                                 DataType::BFLOAT16, Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    CoreRange cores({0, 0}, {0, 0});
    uint32_t cb_id = CB::c_in0;
    uint32_t cb_npages = 16; // multibuffering
    uint32_t cb_pagesize = round_up(padded_row_size_nbytes, constants::TILE_WIDTH);
    DataFormat in_df = datatype_to_dataformat_converter(a.get_dtype());
    tt_metal::CircularBufferConfig cb_config = tt_metal::CircularBufferConfig(cb_npages * cb_pagesize, {{cb_id, in_df}})
		.set_page_size(cb_id, cb_pagesize);
    auto cb = tt_metal::CreateCircularBuffer(program, cores, cb_config);


    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unpadded_row_size_nbytes);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(unpadded_row_size_nbytes) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(padded_row_size_nbytes);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(padded_row_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_log2_stick_size,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_log2_stick_size};
    std::vector<uint32_t> writer_ct_args = reader_ct_args;

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    bfloat16 bfloat_zero = bfloat16(0.0f);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_zero, bfloat_pad_value});

    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/reader_pad_dims_rm_interleaved.cpp",
                                                        cores,
                                                        ReaderDataMovementConfig(reader_ct_args));
    KernelHandle writer_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/writer_pad_dims_rm_interleaved.cpp",
                                                        cores,
                                                        WriterDataMovementConfig(writer_ct_args));
    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

    #if 0
    {
        log_debug("src0_buffer_addr: {}", src0_buffer->address());
        log_debug("dst_buffer_addr: {}", dst_buffer->address());
        log_debug("a.shape[0]: {}", a.get_legacy_shape()[0]);
        log_debug("out.shape[0]: {}", output_shape[0]);
        log_debug("a.shape[1]: {}", a.get_legacy_shape()[1]);
        log_debug("out.shape[1]: {}", output_shape[1]);
        log_debug("a.shape[2]: {}", a.get_legacy_shape()[2]);
        log_debug("out.shape[2]: {}", output_shape[2]);
        log_debug("s.shape[3]: {}", a.get_legacy_shape()[3]);
        log_debug("out.shape[3]: {}", output_shape[3]);
        log_debug("unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug("padded_row_size_nbytes: {}", padded_row_size_nbytes);
        log_debug("padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug("pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        log_debug("pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug("packed_pad_value: {}", packed_pad_value);
    }
    #endif

    uint32_t start_src_stick_id = 0;
    uint32_t start_dst_stick_id = 0;
    vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                       dst_buffer->address(),
                                       a.get_legacy_shape()[0],
                                       output_shape[0],
                                       a.get_legacy_shape()[1],
                                       output_shape[1],
                                       a.get_legacy_shape()[2],
                                       output_shape[2],
                                       a.get_legacy_shape()[3],
                                       output_shape[3],
                                       unpadded_row_size_nbytes,
                                       padded_row_size_nbytes,
                                       padded_row_diff_size_nbytes,
                                       pad_value_const_tensor_addr,
                                       pad_value_const_buffer_nbytes,
                                       packed_pad_value,
                                       start_src_stick_id,
                                       start_dst_stick_id,
                                       0,
                                       0,
                                       0,
                                       output_shape[2],
                                       a.get_legacy_shape()[2],
                                       unpadded_row_size_nbytes,
                                       padded_row_size_nbytes,
                                       0,
                                       output.get_legacy_shape()[0]
                                       };
    vector<uint32_t> writer_rt_args = reader_rt_args;
    SetRuntimeArgs(program,
                   reader_kernel_id,
                   cores,
                   reader_rt_args);
    SetRuntimeArgs(program,
                   writer_kernel_id,
                   cores,
                   writer_rt_args);

    auto override_runtime_args_callback =
        [reader_kernel_id=reader_kernel_id, writer_kernel_id=writer_kernel_id](
            const Program &program,
            const std::vector<Buffer*>& input_buffers,
            const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto &runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
        {
            auto &runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_rm_opt(const Tensor &a,
                                           Tensor &output,
                                           const Shape &output_tensor_shape,
                                           const Shape &input_tensor_start,
                                           const float pad_value) {
    Program program{};

    auto output_shape = output_tensor_shape;

    uint32_t unpadded_row_size_nbytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_nbytes = output_shape[3] * a.element_size();   // Assuming output is same datatype as input
    TT_ASSERT(unpadded_row_size_nbytes <= padded_row_size_nbytes, "Padded output tensor size should be >= input tensor size");

    Device *device = a.device();
    auto dst_buffer_l1 = Buffer(device, padded_row_size_nbytes, padded_row_size_nbytes, BufferType::L1);

    // construct const buffer with the pad_value
    uint32_t pad_value_const_buffer_size = 32;  // noc transfers in chunks of 32
    uint32_t pad_value_const_buffer_nbytes = pad_value_const_buffer_size * a.element_size();
    auto pad_value_const_buffer = owned_buffer::create(std::vector<bfloat16>(pad_value_const_buffer_size, bfloat16(pad_value)));
    const Tensor pad_value_const_tensor = Tensor(OwnedStorage{pad_value_const_buffer},
                                                 Shape({1, 1, 1, pad_value_const_buffer_size}),
                                                 DataType::BFLOAT16, Layout::ROW_MAJOR)
                                            .to(device, MemoryConfig{.memory_layout = TensorMemoryLayout::INTERLEAVED, .buffer_type = BufferType::L1});
    auto pad_value_const_tensor_addr = pad_value_const_tensor.buffer()->address();

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(unpadded_row_size_nbytes);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(unpadded_row_size_nbytes) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(padded_row_size_nbytes);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(padded_row_size_nbytes) : 0;
    std::vector<uint32_t> reader_ct_args = {(std::uint32_t) src0_is_dram,
                                            (std::uint32_t) dst_is_dram,
                                            (std::uint32_t) src_stick_size_is_power_of_two,
                                            (std::uint32_t) src_log2_stick_size,
                                            (std::uint32_t) dst_stick_size_is_power_of_two,
                                            (std::uint32_t) dst_log2_stick_size};

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    bfloat16 bfloat_zero = bfloat16(0.0f);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_zero, bfloat_pad_value});

    CoreRange core({0, 0}, {0, 0});
    KernelHandle reader_kernel_id = CreateKernel(program,
                                                        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/pad_dims_rm_interleaved_opt.cpp",
                                                        core,
                                                        ReaderDataMovementConfig(reader_ct_args));
    uint32_t padded_row_diff_size_nbytes = padded_row_size_nbytes - unpadded_row_size_nbytes;

    #if 0
    {
        log_debug("src0_buffer_addr: {}", src0_buffer->address());
        log_debug("dst_buffer_addr: {}", dst_buffer->address());
        log_debug("a.shape[0]: {}", a.get_legacy_shape()[0]);
        log_debug("out.shape[0]: {}", output_shape[0]);
        log_debug("a.shape[1]: {}", a.get_legacy_shape()[1]);
        log_debug("out.shape[1]: {}", output_shape[1]);
        log_debug("a.shape[2]: {}", a.get_legacy_shape()[2]);
        log_debug("out.shape[2]: {}", output_shape[2]);
        log_debug("s.shape[3]: {}", a.get_legacy_shape()[3]);
        log_debug("out.shape[3]: {}", output_shape[3]);
        log_debug("unpadded_row_size_nbytes: {}", unpadded_row_size_nbytes);
        log_debug("padded_row_size_nbytes: {}", padded_row_size_nbytes);
        log_debug("padded_row_diff_size_nbytes: {}", padded_row_diff_size_nbytes);
        log_debug("pad_value_const_tensor_addr: {}", pad_value_const_tensor_addr);
        log_debug("pad_value_const_buffer_nbytes: {}", pad_value_const_buffer_nbytes);
        log_debug("packed_pad_value: {}", packed_pad_value);
        log_debug("dst_buffer_l1_addr: {}", dst_buffer_l1.address());
    }
    #endif

    vector<uint32_t> reader_rt_args = {src0_buffer->address(),
                                       dst_buffer->address(),
                                       a.get_legacy_shape()[0],
                                       output_shape[0],
                                       a.get_legacy_shape()[1],
                                       output_shape[1],
                                       a.get_legacy_shape()[2],
                                       output_shape[2],
                                       a.get_legacy_shape()[3],
                                       output_shape[3],
                                       unpadded_row_size_nbytes,
                                       padded_row_size_nbytes,
                                       padded_row_diff_size_nbytes,
                                       pad_value_const_tensor_addr,
                                       pad_value_const_buffer_nbytes,
                                       packed_pad_value,
                                       dst_buffer_l1.address()};
    SetRuntimeArgs(program,
                   reader_kernel_id,
                   core,
                   reader_rt_args);

    auto override_runtime_args_callback = [kernel_id=reader_kernel_id](const Program &program,
                                                                             const std::vector<Buffer*>& input_buffers,
                                                                             const std::vector<Buffer*>& output_buffers) {
        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto &runtime_args = GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_rm(const Tensor &a, Tensor &output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value) {

    tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;

    tt_metal::Buffer *src0_buffer = a.buffer();

    uint32_t unpadded_row_size_bytes = a.get_legacy_shape()[3] * a.element_size();
    uint32_t padded_row_size_bytes = output_shape[3] * a.element_size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src_stick_size = unpadded_row_size_bytes;
    uint32_t dst_stick_size = padded_row_size_bytes;

    uint32_t dst_buffer_size = dst_stick_size;

    tt_metal::InterleavedBufferConfig buff_config{
                    .device= device,
                    .size = dst_buffer_size,
                    .page_size = dst_buffer_size,
                    .buffer_type = tt_metal::BufferType::L1
        };

    auto dst_buffer_l1 = CreateBuffer(buff_config);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        dst_buffer->address(),
        a.get_legacy_shape()[0],
        output_shape[0],
        a.get_legacy_shape()[1],
        output_shape[1],
        a.get_legacy_shape()[2],
        output_shape[2],
        a.get_legacy_shape()[3],
        output_shape[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        padded_row_size_bytes - unpadded_row_size_bytes,
        packed_pad_value,
        dst_buffer_l1->address()
    };
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(src_stick_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t) std::log2(src_stick_size) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(dst_stick_size);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t) std::log2(dst_stick_size) : 0;
    std::vector<uint32_t> compile_time_args_vec = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) src_stick_size_is_power_of_two,
        (std::uint32_t) src_log2_stick_size,
        (std::uint32_t) dst_stick_size_is_power_of_two,
        (std::uint32_t) dst_log2_stick_size,

    };

    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/pad_dims_rm_interleaved.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(compile_time_args_vec));

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    auto override_runtime_args_callback = [kernel_id=unary_reader_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks pad_tile(const Tensor &a, Tensor& output, const Shape &output_tensor_shape, const Shape &input_tensor_start, const float pad_value) {

    tt_metal::Program program{};

    CoreRange core({0, 0}, {0, 0});

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1; // For pad buffer
    uint32_t num_pad_tiles = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_pad_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    uint32_t num_unpadded_Xt = a.get_legacy_shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.get_legacy_shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.get_legacy_shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.get_legacy_shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    uint32_t num_unpadded_tiles = a.volume() / TILE_HW;

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_unpadded_tiles, 0
    };
    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_unpadded_W,
        num_padded_Wt,
        num_unpadded_Z,
        num_padded_Zt,
        num_unpadded_Yt,
        num_padded_Yt,
        num_unpadded_Xt,
        num_padded_Xt,
        packed_pad_value,
    };

    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_is_dram
    };
    std::vector<uint32_t> writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src1_cb_index,
        (std::uint32_t) dst_is_dram
    };
    // Tilized reader
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/pad/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp",
        core,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


void Pad::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operand to pad needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE || input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_FATAL(
        (this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0),
        "On device padding only supports padding at end of dims"
    );
    TT_FATAL(input_tensor.get_legacy_shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3], "Output size cannot fit input with offset");

    if (input_tensor.get_layout() == Layout::TILE) {
        TT_FATAL((this->output_tensor_shape[2] % TILE_HEIGHT == 0), "Can only pad tilized tensor with full tiles");
        TT_FATAL((this->output_tensor_shape[3] % TILE_WIDTH == 0), "Can only pad tilized tensor with full tiles");
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "Cannot pad tilized tensor with specified format");
    } else if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        TT_FATAL(this->output_tensor_shape[3] % 2 == 0, "RM padding requires output X dim to be a multiple of 2");
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16, "Cannot pad RM tensor with specified format");
    }
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Pad does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Pad does not currently support sharding");
}

std::vector<Shape> Pad::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {Shape(this->output_tensor_shape)};
}

std::vector<Tensor> Pad::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
operation::ProgramWithCallbacks Pad::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
        if (use_multicore) {
            return pad_rm_reader_writer_multi_core(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
        } else {
            return pad_rm_reader_writer(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
        }
    } else if (input_tensor.get_layout() == Layout::TILE) {
        if (this->use_multicore) {
            log_warning(LogType::LogOp, "TILE layout does not have multicore implementation yet. Falling back to 1 core.");
        }
        return pad_tile(input_tensor, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
    } else {
        TT_ASSERT(false, "Unsupported layout for pad");
        return {};
    }
}

tt::stl::reflection::Attributes Pad::attributes() const {
    return {
        {"output_tensor_shape", this->output_tensor_shape},
        {"input_tensor_start", this->input_tensor_start},
        {"pad_value", this->pad_value},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor pad(const Tensor &input_tensor, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value, const MemoryConfig& output_mem_config, bool use_multicore) {
    if (input_tensor.get_legacy_shape() == output_tensor_shape) {
        return AutoFormat::move_tensor_to_mem_config(input_tensor, output_mem_config);
    }
    return operation::run_without_autoformat(Pad{output_tensor_shape, input_tensor_start, pad_value, output_mem_config, use_multicore}, {input_tensor}).at(0);

}

void PadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() != StorageType::DEVICE);
    TT_FATAL(input_tensor.get_layout() == Layout::ROW_MAJOR);
    TT_FATAL(input_tensor.get_legacy_shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_FATAL(input_tensor.get_legacy_shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3], "Output size cannot fit input with offset");
}

std::vector<Shape> PadOnHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    auto input_shape = input_tensors.at(0).get_legacy_shape();
    auto dimensions_pads = std::vector<Padding::PadDimension>();
    for (auto index = 0; index < input_shape.rank(); index++) {
        auto front = this->input_tensor_start[index];
        auto back = this->output_tensor_shape[index] - (this->input_tensor_start[index] + input_shape[index]);
        dimensions_pads.push_back(Padding::PadDimension{.front=front, .back=back});
    }
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);
    return {Shape(this->output_tensor_shape, padding)};
}

std::vector<Tensor> PadOnHost::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    auto output_shape = this->compute_output_shapes(input_tensors).at(0);
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.get_legacy_shape() == this->output_tensor_shape) {
        return {input_tensor};
    } else {
        return {input_tensor.pad(output_shape, this->input_tensor_start, this->pad_value)};
    }
}

tt::stl::reflection::Attributes PadOnHost::attributes() const {
    return {
        {"output_tensor_shape", this->output_tensor_shape},
        {"input_tensor_start", this->input_tensor_start},
        {"pad_value", this->pad_value},
    };
}

Tensor pad_on_host(const Tensor &input_tensor, const Shape &output_tensor_shape, const Shape &input_tensor_start, float pad_value) {
    return operation::run(PadOnHost{output_tensor_shape, input_tensor_start, pad_value}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
