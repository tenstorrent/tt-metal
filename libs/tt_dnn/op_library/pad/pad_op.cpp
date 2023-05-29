#include "tt_dnn/op_library/pad/pad_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

// This function needs to be up to date with pad_rm to ensure accurate l1 usage calcaulation
bool check_pad_rm_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start) {
    if (a.layout() == Layout::ROW_MAJOR) {
        uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
        uint32_t alignment = 32;

        uint32_t src_stick_size = a.shape()[3] * a.element_size();
        uint32_t dst_stick_size = output_tensor_shape[3] * a.element_size();

        uint32_t src_buffer_size = alignment + src_stick_size;
        uint32_t dst_buffer_size = alignment + dst_stick_size;
        uint32_t cache_buffer_size = alignment * a.device()->num_dram_channels();
        return max_l1_size >= src_buffer_size + dst_buffer_size + cache_buffer_size;
    } else {
        return false;
    }
}

// This function needs to be up to date with pad_tile to ensure accurate l1 usage calcaulation
bool check_pad_tile_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start) {
    if (a.layout() == Layout::TILE) {
        uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
        uint32_t single_tile_size = a.element_size() * TILE_HW;
        return max_l1_size >= 2 * single_tile_size;
    } else {
        return false;
    }
}

bool check_pad_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start) {
    if (a.layout() == Layout::ROW_MAJOR) {
        return check_pad_rm_l1_size(a, output_tensor_shape, input_tensor_start);
    } else if (a.layout() == Layout::TILE) {
        return check_pad_tile_l1_size(a, output_tensor_shape, input_tensor_start);
    } else {
        return false;
    }
}

Tensor pad_rm(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {

    TT_ASSERT(a.layout() == Layout::ROW_MAJOR);
    TT_ASSERT(not a.on_host(), "Operand to pad needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_ASSERT(
        (input_tensor_start[0] == 0 && input_tensor_start[1] == 0 && input_tensor_start[2] == 0 && input_tensor_start[3] == 0),
        "On device padding only supports padding at end of dims"
    );
    TT_ASSERT(a.shape()[0] + input_tensor_start[0] <= output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_ASSERT(a.shape()[1] + input_tensor_start[1] <= output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_ASSERT(a.shape()[2] + input_tensor_start[2] <= output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_ASSERT(a.shape()[3] + input_tensor_start[3] <= output_tensor_shape[3], "Output size cannot fit input with offset");

    TT_ASSERT(output_tensor_shape[3] % 2 == 0, "RM tile requires X to be a multiple of 2");

    if (a.shape() == output_tensor_shape) {
        log_warning("Perf warning: padding called on tensor with same shape as target shape.");
        return a;
    }

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;


    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), a.layout(), device);

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

    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
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

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CompileProgram(device, program, false);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor pad_tile(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {

    TT_ASSERT(a.layout() == Layout::TILE);
    TT_ASSERT(not a.on_host(), "Operand to pad needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_ASSERT(
        (input_tensor_start[0] == 0 && input_tensor_start[1] == 0 && input_tensor_start[2] == 0 && input_tensor_start[3] == 0),
        "On device padding only supports padding at end of dims"
    );
    TT_ASSERT(a.shape()[0] + input_tensor_start[0] <= output_tensor_shape[0], "Output size cannot fit input with offset");
    TT_ASSERT(a.shape()[1] + input_tensor_start[1] <= output_tensor_shape[1], "Output size cannot fit input with offset");
    TT_ASSERT(a.shape()[2] + input_tensor_start[2] <= output_tensor_shape[2], "Output size cannot fit input with offset");
    TT_ASSERT(a.shape()[3] + input_tensor_start[3] <= output_tensor_shape[3], "Output size cannot fit input with offset");

    TT_ASSERT((output_tensor_shape[2] % TILE_HEIGHT == 0), "Can only pad tilized tensor with full tiles");
    TT_ASSERT((output_tensor_shape[3] % TILE_WIDTH == 0), "Can only pad tilized tensor with full tiles");

    if (a.shape() == output_tensor_shape) {
        log_warning("Perf warning: padding called on tensor with same shape as target shape.");
        return a;
    }

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto output_shape = output_tensor_shape;


    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), a.layout(), device);

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);
    auto pad_value_buffer_l1 = tt_metal::Buffer(device, single_tile_size, l1_bank_id, single_tile_size, tt_metal::BufferType::L1);
    auto src_buffer_l1 = tt_metal::Buffer(device, single_tile_size, l1_bank_id, single_tile_size, tt_metal::BufferType::L1);

    bfloat16 bfloat_pad_value = bfloat16(pad_value);
    uint32_t packed_pad_value = pack_two_bfloat16_into_uint32({bfloat_pad_value, bfloat_pad_value});

    vector<uint32_t> reader_kernel_args = {
        src0_dram_buffer->address(),
        dst_dram_buffer->address(),
        a.shape()[0],
        output_shape[0],
        a.shape()[1],
        output_shape[1],
        a.shape()[2] / TILE_HEIGHT,
        output_shape[2] / TILE_HEIGHT,
        a.shape()[3] / TILE_WIDTH,
        output_shape[3]/ TILE_WIDTH,
        single_tile_size,
        packed_pad_value,
        pad_value_buffer_l1.address(),
        src_buffer_l1.address()
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

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::CompileProgram(device, program, false);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor pad (const Tensor &a, const std::array<uint32_t, 4> &output_tensor_shape, const std::array<uint32_t, 4> &input_tensor_start, float pad_value) {
    if (a.layout() == Layout::ROW_MAJOR) {
        return pad_rm(a, output_tensor_shape, input_tensor_start, pad_value);
    } else if (a.layout() == Layout::TILE) {
        return pad_tile(a, output_tensor_shape, input_tensor_start, pad_value);
    } else {
        TT_ASSERT(false, "Unsupported layout for pad");
        return a;
    }
}

}  // namespace tt_metal

}  // namespace tt
