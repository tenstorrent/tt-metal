#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks unpad_rm_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    const Shape output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    uint32_t padded_row_size_bytes = a.shape()[3] * a.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[3] * a.element_size();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src_stick_size = padded_row_size_bytes;
    uint32_t dst_stick_size = unpadded_row_size_bytes;

    uint32_t dst_buffer_size = dst_stick_size;
    auto dst_buffer_l1 = tt_metal::Buffer(device, dst_buffer_size, dst_buffer_size, tt_metal::BufferType::L1);

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        dst_buffer->address(),
        output_shape[0],
        a.shape()[0],
        output_shape[1],
        a.shape()[1],
        output_shape[2],
        a.shape()[2],
        output_shape[3],
        a.shape()[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        padded_row_size_bytes - unpadded_row_size_bytes,
        dst_buffer_l1.address()
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(src_stick_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(src_stick_size) : 0;
    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(dst_stick_size);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(dst_stick_size) : 0;
    std::vector<uint32_t> compile_time_args_vec = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) src_stick_size_is_power_of_two,
        (std::uint32_t) src_log2_stick_size,
        (std::uint32_t) dst_stick_size_is_power_of_two,
        (std::uint32_t) dst_log2_stick_size,
    };

    // Tilized reader
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unpad_dims_rm_interleaved.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = compile_time_args_vec});


    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        reader_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_writer_kernel_id=unary_reader_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_writer_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
            SetRuntimeArgs(program, reader_writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks unpad_tile_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {


    const Shape output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t num_dims = a.shape().rank();

    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_padded_tiles_per_dim(num_dims);
    std::vector<uint32_t> id_per_dim(num_dims, 0);

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = a.shape()[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = a.shape()[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    for(int32_t i = 2; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = a.shape()[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t num_unpadded_tiles = output.volume() / TILE_HW;

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_dims,
        0,
        num_unpadded_tiles
    };
    reader_kernel_args.insert(reader_kernel_args.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    reader_kernel_args.insert(reader_kernel_args.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());
    reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_unpadded_tiles, 0
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
        (std::uint32_t) dst_is_dram
    };

    // Tilized reader
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

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
            auto runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer->address();
            SetRuntimeArgs(program, unary_reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
            SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
