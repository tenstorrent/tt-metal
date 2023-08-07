#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

/*
* 1. Host creates two vectors of data.
* 2. Device eltwise adds them together.
* 3. Intermediate result read back to host.
* 4. Create another vector and send vectors to input DRAMs again.
* 5. Device eltwise muls them together.
* 6. Read result back and compare to golden.
* */

/*
 * We need to copy the types of the compute kernel arguments to use them host-
 * side.
 */

struct BinaryOpType {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

void add_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type){
    // TODO(AP): remove duplication
    string op_name, op_code;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; op_code = "0"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; op_code = "1"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; op_code = "2"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_code.c_str());
}

int main(int argc, char **argv) {
    bool pass = true;

    try {
        /*
        * Silicon accelerator setup
        */
        constexpr int pci_express_slot = 0;
        Device *device =
            CreateDevice(tt::ARCH::GRAYSKULL, pci_express_slot);

        pass &= InitializeDevice(device);;

        /*
        * Setup program to execute along with its buffers and kernels to use
        */
        Program program = Program();

        constexpr CoreCoord compute_core = {9, 9};
        constexpr CoreCoord jump_core = {0, 0};

        uint32_t single_tile_size = stoi(argv[2]);
        uint32_t num_tiles = stoi(argv[1]);
        uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        constexpr uint32_t dram_buffer_src0_addr = 0;
        constexpr int dram_src0_channel_id = 0;
        constexpr uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
        constexpr int dram_src1_channel_id = 1;
        constexpr uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        constexpr int dram_dst_channel_id = 0;

        Buffer src0_dram_buffer = Buffer(device, dram_buffer_size, dram_buffer_src0_addr, dram_src0_channel_id, dram_buffer_size, BufferType::DRAM);
        Buffer src1_dram_buffer = Buffer(device, dram_buffer_size, dram_buffer_src1_addr, dram_src1_channel_id, dram_buffer_size, BufferType::DRAM);
        Buffer dst_dram_buffer = Buffer(device, dram_buffer_size, dram_buffer_dst_addr, dram_dst_channel_id, dram_buffer_size, BufferType::DRAM);

        /*
         * Use circular buffers to set input and output buffers that the
         * compute engine will use.
         */
        constexpr uint32_t src0_cb_index = CB::c_in0;
        constexpr uint32_t num_input_tiles = 2;
        CircularBuffer *compute_cb_src0 = CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            compute_core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
        CircularBuffer *jump_cb_src0 = CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            jump_core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        constexpr uint32_t src1_cb_index = CB::c_in1;
        CircularBuffer *compute_cb_src1 = CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            compute_core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
        CircularBuffer *jump_cb_src1 = CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            jump_core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        constexpr uint32_t output_cb_index = CB::c_out0;
        constexpr uint32_t num_output_tiles = 2;
        CircularBuffer *compute_cb_output = CreateCircularBuffer(
            program,
            device,
            output_cb_index,
            compute_core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );
        CircularBuffer *jump_cb_output = CreateCircularBuffer(
            program,
            device,
            output_cb_index,
            jump_core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            tt::DataFormat::Float16_b
        );

        /*
         * Specify data movement kernels for reading/writing data to/from
         * DRAM.
         */
        DataMovementKernel *compute_binary_reader_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            compute_core,
            DataMovementProcessor::RISCV_1,
            NOC::RISCV_1_default);
        DataMovementKernel *jump_binary_reader_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            jump_core,
            DataMovementProcessor::RISCV_1,
            NOC::RISCV_1_default);

        DataMovementKernel *compute_unary_writer_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            compute_core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default);
        DataMovementKernel *jump_unary_writer_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            jump_core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default);

        /*
         * Set the parameters that the compute kernel will use.
         */
        vector<uint32_t> compute_kernel_args = {
            num_tiles, // per_core_block_cnt
            1 // per_core_block_size
        };

        constexpr bool fp32_dest_acc_en = false;
        constexpr bool math_approx_mode = false;

        /*
         * Use the add_tiles operation available in the eltwise_binary
         * compute kernel.
         */
        ComputeKernel *compute_eltwise_binary_kernel = CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            compute_core,
            compute_kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        add_defines(compute_eltwise_binary_kernel, BinaryOpType::ADD);
        ComputeKernel *jump_eltwise_binary_kernel = CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            jump_core,
            compute_kernel_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        add_defines(jump_eltwise_binary_kernel, BinaryOpType::MUL);

        /*
        * Compile kernels used during execution
        */
        constexpr bool profiler_kernel = true;
        pass &= CompileProgram(device, program, profiler_kernel);

        /*
         * Create source data and write to DRAM.
         */
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());

        WriteToBuffer(src0_dram_buffer, src0_vec);

        constexpr float val_to_add = -1.0f;
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_add);

        WriteToBuffer(src1_dram_buffer, src1_vec);

        /*
         * Configure program and runtime kernel arguments, then execute.
         */
        pass &= ConfigureDeviceWithProgram(device, program);

        WriteRuntimeArgsToDevice(
            device,
            compute_binary_reader_kernel,
            compute_core,
            {
                jump_cb_output->address(),
                static_cast<uint32_t>(jump_core.x),
                static_cast<uint32_t>(jump_core.y),
                num_tiles,
                jump_cb_output->address(),
                static_cast<uint32_t>(jump_core.x),
                static_cast<uint32_t>(jump_core.y),
                num_tiles,
                0
            }
        );
        WriteRuntimeArgsToDevice(
            device,
            jump_binary_reader_kernel,
            jump_core,
            {
                src0_dram_buffer.address(),
                static_cast<uint32_t>(src0_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src0_dram_buffer.noc_coordinates().y),
                num_tiles,
                src1_dram_buffer.address(),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().y),
                num_tiles,
                0
            }
        );

        WriteRuntimeArgsToDevice(
            device,
            compute_unary_writer_kernel,
            compute_core,
            {
                jump_cb_src0->address(),
                static_cast<uint32_t>(jump_core.x),
                static_cast<uint32_t>(jump_core.y),
                num_tiles
            }
        );
        WriteRuntimeArgsToDevice(
            device,
            jump_unary_writer_kernel,
            jump_core,
            {
                dst_dram_buffer.address(),
                static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(dst_dram_buffer.noc_coordinates().y),
                num_tiles
            }
        );

        pass &= LaunchKernels(device, program);
        if (profiler_kernel) tt_metal::DumpDeviceProfileResults(device, program);

        /*
         * Read in result into a host vector.
         */
        std::vector<uint32_t> result_vec;
        ReadFromBuffer(dst_dram_buffer, result_vec);
        // tt_metal::DumpHostProfileResults("first");

        std::function<bfloat16(const bfloat16 &)> transform_to_golden = [](const bfloat16 &a) {
            // return bfloat16((a.to_float() + val_to_add) * val_to_mul);
            return bfloat16((a.to_float() - val_to_add) * val_to_add);
        };
        std::vector<uint32_t> golden_vec = pack_bfloat16_vec_into_uint32_vec(unpack_uint32_vec_into_bfloat16_vec(src0_vec, transform_to_golden));

        constexpr float abs_tolerance = 0.01f;
        constexpr float rel_tolerance = 0.001f;
        std::function<bool(const float, const float)> comparison_function = [](const float a, const float b) {
            return is_close(a, b, rel_tolerance, abs_tolerance);
        };

        pass &= packed_uint32_t_vector_comparison(golden_vec, result_vec, comparison_function);

        pass &= CloseDevice(device);

    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    // if (pass) {
    //     tt::log_info(tt::LogTest, "Test Passed");
    // } else {
    //     tt::log_fatal(tt::LogTest, "Test Failed");
    // }

    // TT_ASSERT(pass);

    return 0;
}
