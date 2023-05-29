#include <math.h>


#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

namespace untilize {
// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
struct hlk_args_t {
    int32_t per_core_block_cnt; // Number of blocks of size 1xN tiles (1 rows and N cols)
    int32_t per_core_block_tile_cnt; // Block tile count = (1xN)
};
}

using namespace tt::constants;

namespace tt {

namespace tt_metal {

// This function needs to be up to date with untilize to ensure accurate l1 usage calcaulation
bool check_untilize_l1_size(const Tensor &a) {
    if(a.layout() == Layout::TILE) {
        uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
        uint32_t single_tile_size = a.element_size() * TILE_HW;
        uint32_t stick_s = a.shape()[3];
        uint32_t cb_buffers_size = 2 * (stick_s / TILE_WIDTH * single_tile_size);
        return max_l1_size >=cb_buffers_size;
    } else {
        return false;
    }
}

// This function needs to be up to date with untilize_with_unpadding to ensure accurate l1 usage calcaulation
bool check_untilize_with_unpadding_l1_size(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {
    if(a.layout() == Layout::TILE) {
        uint32_t max_l1_size = a.device()->l1_size() - UNRESERVED_BASE;
        uint32_t single_tile_size = a.element_size() * TILE_HW;
        uint32_t stick_s = a.shape()[3];
        const uint32_t alignment = 32;
        uint32_t output_shape_3 = output_tensor_end[3] - output_tensor_start[3] + 1;
        uint32_t unpadded_stick_size = output_shape_3 * a.element_size();
        uint32_t cb_buffers_size = 2 * (stick_s / TILE_WIDTH * single_tile_size);
        uint32_t cache_buffer_size = alignment * a.device()->num_dram_channels();
        uint32_t temp_buffer_size = alignment + unpadded_stick_size;
        return max_l1_size >= cb_buffers_size + temp_buffer_size + cache_buffer_size;
    } else {
        return false;
    }
}

Tensor untilize(const Tensor &a) {

    if (a.layout() == Layout::ROW_MAJOR) {
        log_warning("Perf warning: Trying to untilize non-tilized data.");
        return a;
    }

    TT_ASSERT(a.layout() == Layout::TILE, "Can only untilize tile major data");

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to untilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to untilize needs to be allocated in a buffer on device!");

    uint32_t single_tile_size = 2 * TILE_HW; // Assuming bfloat16 dataformat

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    TT_ASSERT(a.volume() % TILE_HW == 0);
    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t stick_size = a.shape()[3] * 2; // Assuming bfloat16 dataformat


    // std::cout << "NUM STICKS: " << num_sticks << ", STICK SIZE: " << stick_size << std::endl;
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    tt_metal::Tensor output = tt_metal::Tensor(a.shape(), a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = a.shape()[3] / 32;
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
    uint32_t num_output_tiles = a.shape()[3] / 32;
    auto cb_output = tt_metal::CreateCircularBuffer(
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
    vector<uint32_t> writer_kernel_args = {src0_dram_buffer->address(), num_sticks, stick_size};
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
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_8bank.cpp",
        core,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_sticks / 32), // per_core_block_cnt
        uint32_t(a.shape()[3] / 32) // per_core_block_tile_cnt
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
        {src0_dram_buffer->address(),
        uint32_t(dram_src0_noc_xy.x),
        uint32_t(dram_src0_noc_xy.y),
        uint32_t(num_tiles), 0,0,0,0,0 } // TODO(AP): [8] is scaler
    );

    tt_metal::WriteRuntimeArgsToDevice(
        device,
        unary_writer_kernel,
        core,
        {dst_dram_buffer->address(),
        num_sticks,
        stick_size}
    );

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

Tensor untilize_with_unpadding(const Tensor &a, const std::array<uint32_t, 4> &output_tensor_start, const std::array<uint32_t, 4> &output_tensor_end) {

    TT_ASSERT(a.dtype() != DataType::BFLOAT8_B, "Bfloat8_b can only exist as tilized data");
    TT_ASSERT(a.layout() == Layout::TILE, "Can only untilize tile major data");

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host(), "Operand to untilize needs to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operand to untilize needs to be allocated in a buffer on device!");

    TT_ASSERT(
        (output_tensor_start[0] == 0 && output_tensor_start[1] == 0 && output_tensor_start[2] == 0 && output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    TT_ASSERT(a.volume() % TILE_HW == 0);
    TT_ASSERT(output_tensor_start[0] < a.shape()[0]);
    TT_ASSERT(output_tensor_end[0] < a.shape()[0]);
    TT_ASSERT(output_tensor_start[1] < a.shape()[1]);
    TT_ASSERT(output_tensor_end[1] < a.shape()[1]);
    TT_ASSERT(output_tensor_start[2] < a.shape()[2]);
    TT_ASSERT(output_tensor_end[2] < a.shape()[2]);
    TT_ASSERT(output_tensor_start[3] < a.shape()[3]);
    TT_ASSERT(output_tensor_end[3] < a.shape()[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(output_tensor_start[0] <= output_tensor_end[0]);
    TT_ASSERT(output_tensor_start[1] <= output_tensor_end[1]);
    TT_ASSERT(output_tensor_start[2] <= output_tensor_end[2]);
    TT_ASSERT(output_tensor_start[3] <= output_tensor_end[3]);

    TT_ASSERT(((output_tensor_end[3] - output_tensor_start[3] + 1) % 2 == 0), "Can only unpad to row major tensor of even width");

    const std::array<uint32_t, 4> output_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };

    if (a.layout() != Layout::TILE) {
        if (a.shape() == output_shape) {
            log_warning("Perf warning: Untilize with unpadding called on already untilized tensor of target shape");
            return a;
        } else {
            TT_ASSERT(false, "Cannot untilize and unpad input which is not tilized");
        }
    }

    tt_metal::Program program = tt_metal::Program();

    CoreCoord core = {0, 0};

    uint32_t single_tile_size = a.element_size() * TILE_HW; // Assuming bfloat16 dataformat

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    // std::cout << "NUM STICKS: " << num_sticks << ", STICK SIZE: " << stick_size << std::endl;
    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::ROW_MAJOR, device);

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_padded_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t num_unpadded_sticks = a.shape()[0] * a.shape()[1] * output_shape[2];
    uint32_t padded_stick_size = a.shape()[3] * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_shape[3] * a.element_size();

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = a.shape()[3] / TILE_WIDTH;
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
    uint32_t num_output_tiles = a.shape()[3] / TILE_WIDTH;
    auto cb_output = tt_metal::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        DataFormat::Float16_b
    );

    uint32_t alignment = 32;
    uint32_t cache_buffer_size = alignment * a.device()->num_dram_channels();
    uint32_t temp_buffer_size = alignment + unpadded_stick_size;

    auto l1_bank_ids = device->bank_ids_from_logical_core(core);
    TT_ASSERT(not l1_bank_ids.empty());
    auto l1_bank_id = l1_bank_ids.at(0);

    // Cache buffer needs to hold 32B max per bank
    auto cache_buffer_l1 = tt_metal::Buffer(device, cache_buffer_size, l1_bank_id, cache_buffer_size, tt_metal::BufferType::L1);
    auto temp_buffer_l1 = tt_metal::Buffer(device, temp_buffer_size, l1_bank_id, temp_buffer_size, tt_metal::BufferType::L1);

    vector<uint32_t> writer_kernel_args = {
        dst_dram_buffer->address(),
        output_shape[0],
        a.shape()[0],
        output_shape[1],
        a.shape()[1],
        output_shape[2],
        a.shape()[2],
        output_shape[3],
        a.shape()[3],
        unpadded_stick_size,
        padded_stick_size,
        cache_buffer_l1.address(),
        temp_buffer_l1.address()
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
        "tt_metal/kernels/dataflow/writer_unary_unpad_dims.cpp",
        core,
        compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        uint32_t(num_padded_sticks / TILE_HEIGHT), // per_core_block_cnt
        uint32_t(a.shape()[3] / TILE_WIDTH) // per_core_block_tile_cnt
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

    tt_metal::LaunchKernels(device, program);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace tt_metal

}  // namespace tt
