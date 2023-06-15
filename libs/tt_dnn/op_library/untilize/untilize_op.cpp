#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


Program untilize_single_core(const Tensor &a, Tensor& output) {

    TT_ASSERT(not a.on_host(), "Operand to untilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to untilize needs to be allocated in a buffer on device!");

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    uint32_t single_tile_size = 2 * TILE_HW; // Assuming bfloat16 dataformat

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t stick_size = a.shape()[3] * a.element_size(); // Assuming bfloat16 dataformat

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

    // std::cout << "NUM STICKS: " << num_sticks << ", STICK SIZE: " << stick_size << std::endl;
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    // Writer compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(stick_size)) == floor(log2(stick_size)));
    vector<uint32_t> writer_kernel_args = {
        dst_dram_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
    };
    std::vector<uint32_t> compile_time_args;
    if (stick_size_is_power_of_two) {
        writer_kernel_args.push_back(log2(stick_size));
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Untilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_split_rows_8_bank.cpp",
        core,
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

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        uint32_t(dram_src0_noc_xy.x),
        uint32_t(dram_src0_noc_xy.y),
        uint32_t(num_tiles), 0,0,0,0,0 } // TODO(AP): [8] is scaler
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        writer_kernel_args
    );

    return program;
}


void Untilize::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    TT_ASSERT(input_tensor_a.dtype() != DataType::BFLOAT8_B, "Bfloat8_b can only exist as tilized data");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
}

std::vector<Shape> Untilize::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return {input_tensor_a.shape()};
}

std::vector<Tensor> Untilize::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::ROW_MAJOR);
}

Program Untilize::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    return untilize_single_core(input_tensor_a, output_tensor);
}

Tensor untilize(const Tensor &input_tensor_a) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        log_warning("Perf warning: Trying to untilize non-tilized data.");
        return input_tensor_a;
    }
    return operation::run_without_autopad(Untilize(), input_tensor_a);
}


Program untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {

    TT_ASSERT(not a.on_host(), "Operand to untilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to untilize needs to be allocated in a buffer on device!");

    const std::array<uint32_t, 4> output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    uint32_t single_tile_size = a.element_size() * TILE_HW; // Assuming bfloat16 dataformat

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    // std::cout << "NUM STICKS: " << num_sticks << ", STICK SIZE: " << stick_size << std::endl;
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_padded_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t num_unpadded_sticks = a.shape()[0] * a.shape()[1] * output_shape[2];
    uint32_t padded_stick_size = a.shape()[3] * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_shape[3] * a.element_size();

    const uint32_t alignment = 32;

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
    const uint32_t num_leftover_Y = output_shape[2] - output_shape[2] / 32 * 32;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t temp_buffer_size = alignment + block_row_size;

    auto l1_bank_ids = device->bank_ids_from_logical_core(core.start);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);

    // Cache buffer needs to hold 32B max per bank
    auto temp_buffer_l1 = tt_metal::Buffer(device, temp_buffer_size, l1_bank_id, temp_buffer_size, tt_metal::BufferType::L1);

    vector<uint32_t> writer_kernel_args = {
        dst_dram_buffer->address(),
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
        temp_buffer_l1.address(),
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
    };
    // Writer compile-time args
    bool stick_size_is_power_of_two = (ceil(log2(unpadded_stick_size)) == floor(log2(unpadded_stick_size)));
    std::vector<uint32_t> compile_time_args;
    if (stick_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args = {1};
        writer_kernel_args.push_back((std::uint32_t)log2(unpadded_stick_size));
    } else {
        compile_time_args = {0};
    }


    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    // Untilized writer
    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp",
        core,
        compile_time_args,
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

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        uint32_t(dram_src0_noc_xy.x),
        uint32_t(dram_src0_noc_xy.y),
        uint32_t(num_tiles), 0,0,0,0,0 } // TODO(AP): [8] is scaler
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        writer_kernel_args
    );

    return program;
}

void UntilizeWithUnpadding::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    TT_ASSERT(input_tensor_a.dtype() != DataType::BFLOAT8_B, "Bfloat8_b can only exist as tilized data");
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
std::vector<Shape> UntilizeWithUnpadding::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return {this->output_tensor_shape};
}
std::vector<Tensor> UntilizeWithUnpadding::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return operation::generic_create_output_tensors(*this, input_tensors, Layout::ROW_MAJOR);
}

Program UntilizeWithUnpadding::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    return untilize_with_unpadding_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end);
}

Tensor untilize_with_unpadding(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {
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
    return operation::run_without_autopad(UntilizeWithUnpadding(output_tensor_start, output_tensor_end), input_tensor_a);
}

}  // namespace tt_metal

}  // namespace tt
