#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


operation::ProgramWithCallbacks untilize_single_core(const Tensor &a, Tensor& output) {

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t stick_size = a.shape()[3] * a.element_size();

    uint32_t stick_s = a.shape()[3];
    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (2 * single_tile_size); // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    // Writer compile-time args
    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(cb_data_format),
        (std::uint32_t) src0_is_dram
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Untilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block), // per_core_block_cnt
        uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto untilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        unary_reader_kernel,
        core,
        {src0_buffer->address(),
        uint32_t(num_tiles), 0 }
    );

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_kernel=unary_reader_kernel,
        writer_kernel=unary_writer_kernel
    ](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_kernel, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(writer_kernel, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


void Untilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
}

std::vector<Shape> Untilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.shape()};
}

std::vector<Tensor> Untilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks Untilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {untilize_single_core(input_tensor_a, output_tensor)};
}

operation::Hash Untilize::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

tt::stl::reflection::Attributes Untilize::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        log_warning("Perf warning: Trying to untilize non-tilized data.");
        return input_tensor_a;
    }
    return operation::run_without_autoformat(Untilize{mem_config}, {input_tensor_a}).at(0);
}


operation::ProgramWithCallbacks untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {

    const std::array<uint32_t, 4> output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_padded_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t num_unpadded_sticks = a.shape()[0] * a.shape()[1] * output_shape[2];
    uint32_t padded_stick_size = a.shape()[3] * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_shape[3] * a.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = a.shape()[3] / TILE_WIDTH;
    uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = unpadded_stick_size / block_row_size;
    uint32_t num_blocks_w_input = padded_stick_size / block_row_size;
    uint32_t block_row_leftover_size = unpadded_stick_size - num_blocks_w_output * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_input - num_blocks_w_output - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (a.shape()[2] - output_shape[2]) / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_Z_diff_blocks = (a.shape()[1] - output_shape[1]) * a.shape()[2] / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_W_diff_blocks = (a.shape()[0] - output_shape[0]) * a.shape()[1] * a.shape()[2] / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t num_leftover_Y = output_shape[2] - output_shape[2] / TILE_HEIGHT * TILE_HEIGHT;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        output_shape[0],
        padded_W_diff_blocks,
        output_shape[1],
        padded_Z_diff_blocks,
        output_shape[2],
        padded_Y_diff_blocks,
        num_leftover_Y,
        output_shape[3],
        unpadded_stick_size,
        padded_stick_size,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(cb_data_format),
        (std::uint32_t) src0_is_dram
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_stick_size;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Untilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp",
        core,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block),
        uint32_t(num_tiles_per_block)
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto untilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        unary_reader_kernel,
        core,
        {src0_buffer->address(),
        uint32_t(num_tiles), 0}
    );

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_kernel=unary_reader_kernel,
        writer_kernel=unary_writer_kernel
    ](
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(reader_kernel, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(reader_kernel, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(writer_kernel, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(writer_kernel, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void UntilizeWithUnpadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operandsneed to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(
        (this->output_tensor_start[0] == 0 && this->output_tensor_start[1] == 0 && this->output_tensor_start[2] == 0 && this->output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
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

    TT_ASSERT(((this->output_tensor_end[3] - this->output_tensor_start[3] + 1) % 2 == 0), "Can only unpad to row major tensor of even width");

}
std::vector<Shape> UntilizeWithUnpadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> UntilizeWithUnpadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks UntilizeWithUnpadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {untilize_with_unpadding_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end)};
}

operation::Hash UntilizeWithUnpadding::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

tt::stl::reflection::Attributes UntilizeWithUnpadding::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize_with_unpadding(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    const std::array<uint32_t, 4> output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.layout() != Layout::TILE) {
        if (input_tensor_a.shape() == output_tensor_shape) {
            log_warning("Perf warning: Untilize with unpadding called on already untilized tensor of target shape");
            return input_tensor_a;
        } else {
            TT_ASSERT(false, "Cannot untilize and unpad input which is not tilized");
        }
    }
    return operation::run_without_autoformat(UntilizeWithUnpadding{output_tensor_start, output_tensor_end, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
