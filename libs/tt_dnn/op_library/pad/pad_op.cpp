#include "tt_dnn/op_library/pad/pad_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

Program pad_rm(const Tensor &a, Tensor &output, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {

    TT_ASSERT(not a.on_host(), "Operand to pad needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    uint32_t unpadded_row_size_bytes = a.shape()[3] * a.element_size();
    uint32_t padded_row_size_bytes = output_shape[3] * a.element_size();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t alignment = 32;

    uint32_t src_stick_size = unpadded_row_size_bytes;
    uint32_t dst_stick_size = padded_row_size_bytes;

    uint32_t cache_buffer_size = alignment * a.device()->num_dram_channels();
    uint32_t src_buffer_size = alignment + src_stick_size;
    uint32_t dst_buffer_size = alignment + dst_stick_size;

    auto l1_bank_ids = device->bank_ids_from_logical_core(core.start);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto cache_buffer_l1 = tt_metal::Buffer(device, cache_buffer_size, l1_bank_id, cache_buffer_size, tt_metal::BufferType::L1);
    auto dst_buffer_l1 = tt_metal::Buffer(device, dst_buffer_size, l1_bank_id, dst_buffer_size, tt_metal::BufferType::L1);
    auto src_buffer_l1 = tt_metal::Buffer(device, src_buffer_size, l1_bank_id, src_buffer_size, tt_metal::BufferType::L1);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_dram_buffer->address(),
        dst_dram_buffer->address(),
        a.shape()[0],
        output_shape[0],
        a.shape()[1],
        output_shape[1],
        a.shape()[2],
        output_shape[2],
        a.shape()[3],
        output_shape[3],
        unpadded_row_size_bytes,
        padded_row_size_bytes,
        padded_row_size_bytes - unpadded_row_size_bytes,
        packed_pad_value,
        cache_buffer_l1.address(),
        src_buffer_l1.address(),
        dst_buffer_l1.address()
    };

    std::vector<uint32_t> compile_time_args_vec;
    // Reader compile-time args
    // Data is 32 byte aligned
    bool src_stick_size_is_power_of_two = (ceil(log2(src_stick_size)) == floor(log2(src_stick_size)));
    if (src_stick_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args_vec.push_back(1);
        reader_kernel_args.push_back((std::uint32_t)log2(src_stick_size));
    } else {
        compile_time_args_vec.push_back(0);
        reader_kernel_args.push_back(0);
    }

    bool dst_stick_size_is_power_of_two = (ceil(log2(dst_stick_size)) == floor(log2(dst_stick_size)));
    if (dst_stick_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args_vec.push_back(1);
        reader_kernel_args.push_back((std::uint32_t)log2(dst_stick_size));
    } else {
        compile_time_args_vec.push_back(0);
        reader_kernel_args.push_back(0);
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/pad_dims_rm_8bank.cpp",
        core,
        compile_time_args_vec,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);


    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        0 // dummy
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program, "tt_metal/kernels/compute/blank.cpp",
        core, compute_args, MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    // output does not hold any data, contains pointer to buffer on device with the data
    return program;
}

Program pad_tile(const Tensor &a, Tensor& output, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {

    TT_ASSERT(not a.on_host(), "Operand to pad needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 1;

    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1; // For pad buffer

    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    uint32_t num_unpadded_Xt = a.shape()[3] / TILE_WIDTH;
    uint32_t num_total_Xt = output_shape[3] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = a.shape()[2] / TILE_HEIGHT;
    uint32_t num_total_Yt = output_shape[2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;
    uint32_t num_unpadded_Z = a.shape()[1];
    uint32_t num_total_Z = output_shape[1];
    uint32_t num_padded_Zt = (num_total_Z - num_unpadded_Z) * num_total_Yt * num_total_Xt;
    uint32_t num_unpadded_W = a.shape()[0];
    uint32_t num_total_W = output_shape[0];
    uint32_t num_padded_Wt = (num_total_W - num_unpadded_W) * num_total_Z * num_total_Yt * num_total_Xt;

    vector<uint32_t> reader_kernel_args = {
        src0_dram_buffer->address(),
        dst_dram_buffer->address(),
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

    std::vector<uint32_t> compile_time_args_vec;
    // Reader compile-time args
    // Data is 32 byte aligned
    bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
    if (tile_size_is_power_of_two) {
        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        compile_time_args_vec.push_back(1);
        reader_kernel_args.push_back((std::uint32_t)log2(single_tile_size));
    } else {
        compile_time_args_vec.push_back(0);
        reader_kernel_args.push_back(0);
    }

    // Tilized reader
    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/pad_dims_8bank.cpp",
        core,
        compile_time_args_vec,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);


    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        0 // dummy
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program, "tt_metal/kernels/compute/blank.cpp",
        core, compute_args, MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);


    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    return program;
}


void Pad::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(
        (this->input_tensor_start[0] == 0 && this->input_tensor_start[1] == 0 && this->input_tensor_start[2] == 0 && this->input_tensor_start[3] == 0),
        "On device padding only supports padding at end of dims"
    );
    TT_ASSERT(input_tensor_a.shape()[0] + this->input_tensor_start[0] <= this->output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_ASSERT(input_tensor_a.shape()[1] + this->input_tensor_start[1] <= this->output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_ASSERT(input_tensor_a.shape()[2] + this->input_tensor_start[2] <= this->output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_ASSERT(input_tensor_a.shape()[3] + this->input_tensor_start[3] <= this->output_tensor_shape[3], "Output size cannot fit input with offset");

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_ASSERT((this->output_tensor_shape[2] % TILE_HEIGHT == 0), "Can only pad tilized tensor with full tiles");
        TT_ASSERT((this->output_tensor_shape[3] % TILE_WIDTH == 0), "Can only pad tilized tensor with full tiles");
    } else if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        TT_ASSERT(this->output_tensor_shape[3] % 2 == 0, "RM padding requires output X dim to be a multiple of 2");
    }
}
std::vector<Shape> Pad::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return {this->output_tensor_shape};
}
std::vector<Tensor> Pad::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.layout());
}

// TODO: If pad is called on a tile and output is not tile, we could untilize then pad, and output is RM
// Currently calling pad on a tile requires the output pad shape to be tile
Program Pad::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        return pad_rm(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
    } else if (input_tensor_a.layout() == Layout::TILE) {
        return pad_tile(input_tensor_a, output_tensor, this->output_tensor_shape, this->input_tensor_start, this->pad_value);
    } else {
        TT_ASSERT(false, "Unsupported layout for pad");
        return tt_metal::Program();
    }
}

Tensor pad(const Tensor &input_tensor_a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    if (input_tensor_a.shape() == output_tensor_shape) {
        log_warning("Perf warning: padding called on tensor with same shape as target shape.");
        return input_tensor_a;
    }
    return operation::run_without_autopad(Pad{output_tensor_shape, input_tensor_start, pad_value}, input_tensor_a);

}

}  // namespace tt_metal

}  // namespace tt
