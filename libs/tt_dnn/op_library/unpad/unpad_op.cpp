#include "tt_dnn/op_library/unpad/unpad_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks unpad_rm(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {

    const std::array<uint32_t, 4> output_shape = output.shape();

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
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unpad_dims_rm_interleaved.cpp",
        core,
        compile_time_args_vec,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);


    tt_metal::SetRuntimeArgs(
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_writer_kernel=unary_reader_kernel
    ](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_writer_kernel, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
            SetRuntimeArgs(reader_writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

operation::ProgramWithCallbacks unpad_tile(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {


    const std::array<uint32_t, 4> output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;

    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t num_unpadded_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_total_Xt = a.shape()[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = a.shape()[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = output_shape[1];
    uint32_t num_total_Z = a.shape()[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = output_shape[0];
    uint32_t num_total_W = a.shape()[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;


    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        dst_buffer->address(),
        num_unpadded_W,
        num_padded_Wt,
        num_unpadded_Z,
        num_padded_Zt,
        num_unpadded_Yt,
        num_padded_Yt,
        num_unpadded_Xt,
        num_padded_Xt,
    };

    // Reader compile-time args
    // Data is 32 byte aligned
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> compile_time_args_vec = {
        // interleaved accessor args
        (std::uint32_t) static_cast<uint32_t>(cb_data_format),
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) dst_is_dram
    };

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unpad_dims_interleaved.cpp",
        core,
        compile_time_args_vec,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::SetRuntimeArgs(
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_writer_kernel=unary_reader_kernel
    ](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_writer_kernel, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = dst_buffer->address();
            SetRuntimeArgs(reader_writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


void Unpad::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to unpad need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR);

    TT_ASSERT(
        (this->output_tensor_start[0] == 0 && this->output_tensor_start[1] == 0 && output_tensor_start[2] == 0 && output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    TT_ASSERT(this->output_tensor_start[0] < input_tensor_a.shape()[0]);
    TT_ASSERT(this->output_tensor_end[0] < input_tensor_a.shape()[0]);
    TT_ASSERT(this->output_tensor_start[1] < input_tensor_a.shape()[1]);
    TT_ASSERT(this->output_tensor_end[1] < input_tensor_a.shape()[1]);
    TT_ASSERT(this->output_tensor_start[2] < input_tensor_a.shape()[2]);
    TT_ASSERT(this->output_tensor_end[2] < input_tensor_a.shape()[2]);
    TT_ASSERT(this->output_tensor_start[3] < input_tensor_a.shape()[3]);
    TT_ASSERT(this->output_tensor_end[3] < input_tensor_a.shape()[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(this->output_tensor_start[0] <= this->output_tensor_end[0]);
    TT_ASSERT(this->output_tensor_start[1] <= this->output_tensor_end[1]);
    TT_ASSERT(this->output_tensor_start[2] <= this->output_tensor_end[2]);
    TT_ASSERT(this->output_tensor_start[3] <= this->output_tensor_end[3]);

    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
        TT_ASSERT((output_tensor_shape[2] % TILE_HEIGHT == 0), "Can only unpad tilized tensor with full tiles");
        TT_ASSERT((output_tensor_shape[3] % TILE_WIDTH == 0), "Can only unpad tilized tensor with full tiles");
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        TT_ASSERT(output_tensor_shape[3] % 2 == 0, "RM unpadding requires output X dim to be a multiple of 2");
    }
}
std::vector<Shape> Unpad::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> Unpad::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.layout(), this->output_mem_config);
}

// TODO: If unpad is called on a tile and output is not tile, we could untilize then unpad, and output is RM
// Currently calling unpad on a tile requires the output unpad shape to be tile
operation::ProgramWithCallbacks Unpad::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        return unpad_rm(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
    } else if (input_tensor_a.layout() == Layout::TILE) {
        return unpad_tile(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
    } else {
        TT_ASSERT(false, "Unsupported layout for unpad");
        return {};
    }
}

operation::Hash Unpad::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

tt::stl::reflection::Attributes Unpad::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor unpad(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    const std::array<uint32_t, 4> output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.shape() == output_tensor_shape) {
        return input_tensor_a;
    }
    return operation::run_without_autoformat(Unpad{output_tensor_start, output_tensor_end, mem_config}, {input_tensor_a}).at(0);

}

void UnpadOnHost::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_ASSERT(input_tensor.storage_type() == StorageType::OWNED);
    TT_ASSERT(input_tensor.layout() == Layout::ROW_MAJOR);

    TT_ASSERT(this->output_tensor_start[0] < input_tensor.shape()[0]);
    TT_ASSERT(this->output_tensor_end[0] < input_tensor.shape()[0]);
    TT_ASSERT(this->output_tensor_start[1] < input_tensor.shape()[1]);
    TT_ASSERT(this->output_tensor_end[1] < input_tensor.shape()[1]);
    TT_ASSERT(this->output_tensor_start[2] < input_tensor.shape()[2]);
    TT_ASSERT(this->output_tensor_end[2] < input_tensor.shape()[2]);
    TT_ASSERT(this->output_tensor_start[3] < input_tensor.shape()[3]);
    TT_ASSERT(this->output_tensor_end[3] < input_tensor.shape()[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(this->output_tensor_start[0] <= this->output_tensor_end[0]);
    TT_ASSERT(this->output_tensor_start[1] <= this->output_tensor_end[1]);
    TT_ASSERT(this->output_tensor_start[2] <= this->output_tensor_end[2]);
    TT_ASSERT(this->output_tensor_start[3] <= this->output_tensor_end[3]);
}
std::vector<Shape> UnpadOnHost::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> UnpadOnHost::compute_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (input_tensor.shape() == UnpadOnHost::compute_output_shapes(input_tensors).at(0)) {
        return {input_tensor};
    } else {
        return {input_tensor.unpad(this->output_tensor_start, this->output_tensor_end)};
    }
}

tt::stl::reflection::Attributes UnpadOnHost::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
    };
}

Tensor unpad_on_host(const Tensor &input_tensor, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end, const MemoryConfig& mem_config) {
    return operation::run(UnpadOnHost{output_tensor_start, output_tensor_end}, {input_tensor}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
