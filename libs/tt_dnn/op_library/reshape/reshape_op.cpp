#include "tt_dnn/op_library/reshape/reshape_op.hpp"
#include "common/test_tiles.hpp"
#include <algorithm>
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

Tensor reshape_tilized(const Tensor &a, int N, int C, int H, int W) {

    TT_ASSERT(a.layout() == Layout::TILE, "Only tile and row major reshape supported!");

    tt_metal::Program program = tt_metal::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to eltwise unary needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to eltwise unary needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    uint32_t num_tiles = a.volume() / TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    std::array<uint32_t, 4> output_shape = infer_dims_for_reshape(N, C, H, W, a.volume());

    TT_ASSERT(output_shape[2] % TILE_HEIGHT == 0 && output_shape[3] % TILE_WIDTH == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) for reshape!");
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );


    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );
    // no need to create c_in2 buffer since we pass scaler=0 to reader

    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reshape_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_kernel_args = {
        num_tiles, // per_core_block_cnt
        1 // per_core_block_size
    };
    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_kernel_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        {src0_dram_buffer->address(),
        a.shape()[3] / TILE_WIDTH,
        (uint32_t) output_shape[0],
        (uint32_t) output_shape[1],
        (uint32_t) output_shape[2] / TILE_HEIGHT,
        (uint32_t) output_shape[3] / TILE_WIDTH }
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        uint32_t(dram_dst_noc_xy.x),
        uint32_t(dram_dst_noc_xy.y),
        num_tiles }
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor reshape_rm(const Tensor &a, int N, int C, int H, int W) {


    TT_ASSERT(a.layout() == Layout::ROW_MAJOR, "Only tile and row major reshape supported!");

    // Reshape for row major data requires rebanking if and only if the last
    // dimension is changed. W dictates the stick size in DRAM banks

    TT_ASSERT(N != -1 and C != -1 and H != -1 and W != -1, "-1 reshape not yet supported for rebanking row major reshape");

    tt_metal::Program program = tt_metal::Program();
    tt_xy_pair core = {0, 0};

    TT_ASSERT(not a.on_host(), "Operand to reshape needs to be on device");
    TT_ASSERT(a.buffer() != nullptr, "Operand to reshape needs to be allocated in a buffer on device!");
    TT_ASSERT(a.volume() == N*C*H*W, "New shape volume must match old shape volume");
    TT_ASSERT(a.shape()[3] % TILE_WIDTH == 0 && W % TILE_WIDTH == 0, "Operand/target width must be a multiple of 32");

    uint32_t num_old_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t num_new_sticks = N*C*H;
    TT_ASSERT(num_old_sticks % TILE_HEIGHT == 0 && num_new_sticks % TILE_HEIGHT == 0, "Operand/target number of rows must be a multiple of 32");

     // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    std::array<uint32_t, 4> output_shape = {uint32_t(N), uint32_t(C), uint32_t(H), uint32_t(W)};
    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);
    tt_metal::Buffer *src0_dram_buffer = a.buffer();
    tt_metal::Buffer *dst_dram_buffer = output.buffer();

    uint32_t old_stick_size = a.shape()[3] * 2; // Assuming bfloat16 data format
    uint32_t new_stick_size = W * 2; // Assuming bfloat16 data format

    uint32_t single_tile_size = 2 * TILE_HW; // Assuming bfloat16 dataformat
    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = (a.shape()[1] * a.shape()[2] * a.shape()[3] / TILE_HW);
    uint32_t num_output_tiles = (C * H * W / TILE_HW);

    // Currently added to support Bert large, TODO: Make op more generic, parallelize
    uint32_t available_l1 = 1024*1024 - UNRESERVED_BASE;
    if (num_input_tiles * single_tile_size + num_output_tiles * single_tile_size > available_l1) {
        if (old_stick_size >= new_stick_size) {
            if (old_stick_size % new_stick_size == 0) {
                // Maximize L1 usage. Is this needed or do we just need to double buffer 32 sticks (64)
                // Evenly divide L1 between input/output
                uint32_t w_tiles = a.shape()[3] / 32;
                num_input_tiles = ((available_l1 / 2) / single_tile_size) / w_tiles * w_tiles;
                num_output_tiles = num_input_tiles;
            } else {
                // Not needed for Bert large at the moment so will trigger L1 OOM assert
            }
        } else {
            if (new_stick_size % old_stick_size == 0) {
                // Maximize L1 usage. Is this needed or do we just need to double buffer 32 sticks (64)
                // Evenly divide L1 between input/output
                uint32_t w_tiles = (W / 32);
                num_output_tiles = ((available_l1 / 2) / single_tile_size) / w_tiles * w_tiles;
                num_input_tiles = num_output_tiles;
            } else {
                // Not needed for Bert large at the moment so will trigger L1 OOM assert
            }
        }
        TT_ASSERT(num_input_tiles > 0 && num_output_tiles > 0, "Cannot fit input/output rows into L1");
    }

    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    // Reader compile-time args
    bool old_stick_size_is_power_of_two = (ceil(log2(old_stick_size)) == floor(log2(old_stick_size)));
    vector<uint32_t> reader_kernel_args = {src0_dram_buffer->address(), num_old_sticks, old_stick_size};
    KernelArgs reader_compile_time_args;
    if (old_stick_size_is_power_of_two) {
        reader_kernel_args.push_back(log2(old_stick_size));

        // Use the fast stick size power of 2 path (get noc addr uses just shift operations, no slow multiply algorithm)
        reader_compile_time_args = tt_metal::KernelArgs(core, {1});
    } else {
        reader_compile_time_args = tt_metal::KernelArgs(core, {0});
    }

    // Writer compile-time args
    bool new_stick_size_is_power_of_two = (ceil(log2(new_stick_size)) == floor(log2(new_stick_size)));
    vector<uint32_t> writer_kernel_args = {dst_dram_buffer->address(), num_new_sticks, new_stick_size};
    KernelArgs writer_compile_time_args;
    if (new_stick_size_is_power_of_two) {
        writer_kernel_args.push_back(log2(new_stick_size));
    }

    tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_stick_layout_8bank.cpp",
        core,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    // No compute required, so using blank kernel
    vector<uint32_t> compute_args = {
        uint(a.volume() / TILE_HW), // per_core_block_cnt
        1 // per_core_block_size
    };
    tt_metal::KernelArgs eltwise_unary_args = tt_metal::KernelArgs(core, compute_args);

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        core,
        eltwise_unary_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    // Compile kernels

    tt_metal::CompileProgram(device, program);
    tt_metal::ConfigureDeviceWithProgram(device, program);

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_reader_kernel,
        core,
        reader_kernel_args
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        writer_kernel_args
    );

    tt_metal::LaunchKernels(device, program);

    return output;
}

Tensor reshape_(const Tensor &a, int N, int C, int H, int W) {
    if (a.layout() == Layout::TILE) {
        return reshape_tilized(a, N, C, H, W);
    } else if (a.layout() == Layout::ROW_MAJOR) {
        return reshape_rm(a, N, C, H, W);
    } else {
        TT_ASSERT(false, "Unsupported layout for reshape");
    }
    return a;
}

Tensor reshape(Tensor &a, int N, int C, int H, int W) {

    if (
        ((a.layout() == Layout::TILE or a.layout() == Layout::ROW_MAJOR) && W == a.shape()[3]) ||
        ((a.layout() == Layout::CHANNELS_LAST) && C == a.shape()[1])
    ) {
        // Don't need to do a check here to see the H and W both divisible by 32
        // since handled within the tensor reshape method
        a.reshape(N, C, H, W);
        return a;
    }

    Device * device;

    // Get the device
    if (a.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    if (a.on_host()) {
        auto output = reshape_(a.to(device), N, C, H, W);
        // Convert tensor back to original
        AutoPad::format_output_tensor(a, output, infer_dims_for_reshape(N, C, H, W, a.volume()), device);
        return output;
    } else {
        return reshape_(a, N, C, H, W);
    }

}

} // namespace tt_metal
} // namespace tt
