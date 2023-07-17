#include <math.h>

#include "tt_dnn/op_library/tilize/tilize_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "dtx/dtx.hpp"
#include "dtx/dtx_passes.hpp"

#include "tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


operation::ProgramWithCallbacks tilize_single_core(const Tensor &a, Tensor& output) {

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt_metal::Buffer *src0_buffer = a.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    auto output_shape = output.shape();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);

    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t stick_s =  a.layout() == Layout::ROW_MAJOR ? a.shape()[3] : a.shape()[1];
    uint32_t num_sticks = a.layout() == Layout::ROW_MAJOR ? a.shape()[0] * a.shape()[1] * a.shape()[2] : a.shape()[0] * a.shape()[2] * a.shape()[3];
    uint32_t stick_size = stick_s * a.element_size(); // Assuming bfloat16 dataformat

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

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
    };

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        static_cast<uint32_t>(DataFormat::Float16_b),
        (std::uint32_t) out_is_dram
    };

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
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
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel,
        core,
        {dst_buffer->address(),
        (uint32_t) num_tiles,
        (uint32_t) 0}
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

void Tilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to tilize need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to tilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.layout() == Layout::ROW_MAJOR or input_tensor_a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);

    uint32_t stick_s =  input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.shape()[3] : input_tensor_a.shape()[1];
    uint32_t num_sticks = input_tensor_a.layout() == Layout::ROW_MAJOR ? input_tensor_a.volume() / input_tensor_a.shape()[3] : input_tensor_a.volume() / input_tensor_a.shape()[1];
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);

    uint32_t stick_size = stick_s * input_tensor_a.element_size(); // Assuming bfloat16 dataformat

    TT_ASSERT((stick_size % 2) == 0, "Stick size must be divisible by 2");
}

std::vector<Shape> Tilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto output_shape = input_tensor_a.shape();
    if(input_tensor_a.layout() == Layout::CHANNELS_LAST) {
        // Set channels last in the innermost dim in the shape
        output_shape = {input_tensor_a.shape()[0], input_tensor_a.shape()[2], input_tensor_a.shape()[3], input_tensor_a.shape()[1]};
    }
    return {output_shape};
}

std::vector<Tensor> Tilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks Tilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return tilize_single_core(input_tensor_a, output_tensor);
}

operation::Hash Tilize::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

tt::stl::reflection::Attributes Tilize::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor tilize(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return input_tensor_a;
    }
    return operation::run_without_autoformat(Tilize{mem_config}, {input_tensor_a}).at(0);
}

operation::ProgramWithCallbacks tilize_with_val_padding_single_core(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, const float pad_value) {


    auto output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);

    int32_t num_tiles = output.volume() / TILE_HW;

    auto true_input_shape = a.shape();
    auto true_output_shape = output.shape();
    if (a.layout() == Layout::CHANNELS_LAST) {
        true_input_shape = {a.shape()[0], a.shape()[2], a.shape()[3], a.shape()[1]};
    }

    uint32_t unpadded_row_size_datum = true_input_shape[3];
    uint32_t padded_row_size_datum = true_output_shape[3];

    uint32_t num_rows_padded = true_output_shape[2];
    uint32_t num_cols_padded = true_output_shape[3] - unpadded_row_size_datum;


    uint32_t num_2d_faces = true_output_shape[0] * true_output_shape[1];

    uint32_t unpadded_row_size_bytes = unpadded_row_size_datum * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = padded_row_size_datum * a.element_size(); // Assuming bfloat16 dataformat

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = true_output_shape[3] / TILE_WIDTH;
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
    uint32_t num_blocks_w_output = padded_row_size_bytes / block_row_size;
    uint32_t num_blocks_w_input = unpadded_row_size_bytes / block_row_size;

    // Leftover size if input is not divisible by block size
    uint32_t block_row_leftover_size = unpadded_row_size_bytes - num_blocks_w_input * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_output - num_blocks_w_input - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (true_output_shape[2] - true_input_shape[2]) / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_Z_diff_blocks = (true_output_shape[1] - true_input_shape[1]) * true_output_shape[2] / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t padded_W_diff_blocks = (true_output_shape[0] - true_input_shape[0]) * true_output_shape[1] * true_output_shape[2] / TILE_HEIGHT * num_blocks_w_output;
    const uint32_t num_leftover_Y = true_input_shape[2] - true_input_shape[2] / TILE_HEIGHT * TILE_HEIGHT;


    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    assert(num_input_tiles > 0);
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


    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_buffer->address(),
        true_input_shape[0],
        padded_W_diff_blocks,
        true_input_shape[1],
        padded_Z_diff_blocks,
        true_input_shape[2],
        padded_Y_diff_blocks,
        num_leftover_Y,
        true_input_shape[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        packed_pad_value,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
    };

    // Reader compile-time args
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_row_size_bytes;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        static_cast<uint32_t>(DataFormat::Float16_b),
        (std::uint32_t) out_is_dram
    };

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_pad_dims_split_rows.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Tilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        uint32_t(num_tiles / num_tiles_per_block),
        uint32_t(num_tiles_per_block)
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto tilize_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/tilize.cpp",
        core,
        compute_kernel_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    tt_metal::SetRuntimeArgs(
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::SetRuntimeArgs(
        unary_writer_kernel,
        core,
        {dst_buffer->address(),
        (uint32_t) num_tiles, 0}
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

void TilizeWithValPadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.layout() == Layout::ROW_MAJOR or input_tensor_a.layout() == Layout::CHANNELS_LAST, "Can only tilize row major or channels last data");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16);

    TT_ASSERT(input_tensor_a.shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0]);
    TT_ASSERT(input_tensor_a.shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1]);
    TT_ASSERT(input_tensor_a.shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2]);
    TT_ASSERT(input_tensor_a.shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3]);
    TT_ASSERT((this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0), "On device padding only supports padding at end of dims");

    uint32_t num_rows = this->output_tensor_shape[2];
    uint32_t inner_dim = this->output_tensor_shape[3];
    if (input_tensor_a.layout() == Layout::CHANNELS_LAST) {
        num_rows = this->output_tensor_shape[3];
        inner_dim = this->output_tensor_shape[1];
    }
    TT_ASSERT(num_rows % TILE_HEIGHT == 0, "Output shape must be tilizable");
    TT_ASSERT(inner_dim % TILE_WIDTH == 0, "Output shape must be tilizable");
}
std::vector<Shape> TilizeWithValPadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto output_shape = this->output_tensor_shape;
    if (input_tensor_a.layout() == Layout::CHANNELS_LAST) {
        output_shape = {output_shape[0], output_shape[2], output_shape[3], output_shape[1]};
    }
    return {output_shape};
}
std::vector<Tensor> TilizeWithValPadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::TILE, this->output_mem_config);
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
operation::ProgramWithCallbacks TilizeWithValPadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return tilize_with_val_padding_single_core(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
}

operation::Hash TilizeWithValPadding::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

tt::stl::reflection::Attributes TilizeWithValPadding::attributes() const {
    return {
        {"output_tensor_shape", this->output_tensor_shape},
        {"input_tensor_start", this->input_tensor_start},
        {"pad_value", this->pad_value},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor tilize_with_val_padding(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, const float pad_value, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    if (input_tensor_a.layout() == Layout::TILE) {
        if (output_tensor_shape == input_tensor_a.shape()) {
            log_warning("Perf warning: tilize with padding called on already tilized tensor of target shape.");
            return input_tensor_a;
        } else {
            TT_ASSERT(false, "Cannot tilize and pad tensor that is already tilized");
        }
    }
    return operation::run_without_autoformat(TilizeWithValPadding{output_tensor_shape, input_tensor_start, pad_value, mem_config}, {input_tensor_a}).at(0);

}

Tensor tilize_with_zero_padding(const Tensor &input_tensor_a, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::TILE) {
        log_warning("Perf warning: tilize called on already tilized tensor.");
        return input_tensor_a;
    }
    auto shape = input_tensor_a.shape();

    if (input_tensor_a.layout() == Layout::CHANNELS_LAST) {
        shape[3] = roundup(shape[3], TILE_HEIGHT);
        shape[1] = roundup(shape[1], TILE_WIDTH);
    } else {
        shape[2] = roundup(shape[2], TILE_HEIGHT);
        shape[3] = roundup(shape[3], TILE_WIDTH);
    }
    return tilize_with_val_padding(input_tensor_a, shape, {0, 0, 0, 0}, 0, mem_config);
}

}  // namespace tt_metal

}  // namespace tt
