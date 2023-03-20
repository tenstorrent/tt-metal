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
namespace eltwise_binary {
struct compute_kernel_args_t {
    std::int32_t per_core_block_cnt;
    std::int32_t per_core_block_size;
};
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
        Program *program = new Program();

        constexpr tt_xy_pair core = {0, 0};

        constexpr uint32_t single_tile_size = 2 * 1024;
        constexpr uint32_t num_tiles = 2048;
        constexpr uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

        constexpr uint32_t dram_buffer_src0_addr = 0;
        constexpr int dram_src0_channel_id = 0;
        constexpr uint32_t dram_buffer_src1_addr = 256 * 1024 * 1024;
        constexpr int dram_src1_channel_id = 1;
        constexpr uint32_t dram_buffer_dst_addr = 512 * 1024 * 1024; // 512 MB (upper half)
        constexpr int dram_dst_channel_id = 0;

        DramBuffer *src0_dram_buffer = CreateDramBuffer(device, dram_src0_channel_id, dram_buffer_size, dram_buffer_src0_addr);
        DramBuffer *src1_dram_buffer = CreateDramBuffer(device, dram_src1_channel_id, dram_buffer_size, dram_buffer_src1_addr);
        DramBuffer *dst_dram_buffer = CreateDramBuffer(device, dram_dst_channel_id, dram_buffer_size, dram_buffer_dst_addr);

        /*
         * Use circular buffers to set input and output buffers that the
         * compute engine will use.
         */
        constexpr uint32_t src0_cb_index = CB::c_in0;
        constexpr uint32_t src0_cb_addr = 200 * 1024;
        constexpr uint32_t num_input_tiles = 2;
        CircularBuffer *cb_src0 = CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        constexpr uint32_t src1_cb_index = CB::c_in1;
        constexpr uint32_t src1_cb_addr = 300 * 1024;
        CircularBuffer *cb_src1 = CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src1_cb_addr,
            tt::DataFormat::Float16_b
        );

        constexpr uint32_t output_cb_index = CB::c_out0;
        constexpr uint32_t output_cb_addr = 400 * 1024;
        constexpr uint32_t num_output_tiles = 2;
        CircularBuffer *cb_output = CreateCircularBuffer(
            program,
            device,
            output_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        /*
         * Specify data movement kernels for reading/writing data to/from
         * DRAM.
         */
        DataMovementKernel *binary_reader_kernel = CreateDataMovementKernel(
            program,
            "kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            DataMovementProcessor::RISCV_1,
            NOC::RISCV_1_default);

        DataMovementKernel *unary_writer_kernel = CreateDataMovementKernel(
            program,
            "kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default);

        /*
         * Set the parameters that the compute kernel will use.
         */
        vector<uint32_t> compute_kernel_args = {
            2048, // per_core_block_cnt
            1 // per_core_block_size
        };
        ComputeKernelArgs *eltwise_binary_args = InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

        constexpr bool fp32_dest_acc_en = false;
        constexpr bool math_approx_mode = false;

        /*
         * Use the add_tiles operation available in the eltwise_binary
         * compute kernel.
         */
        ComputeKernel *eltwise_binary_kernel = CreateComputeKernel(
            program,
            "kernels/compute/eltwise_binary.cpp",
            core,
            eltwise_binary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
        eltwise_binary_kernel->add_define("ELTWISE_OP", "add_tiles");

        /*
        * Compile kernels used during execution
        */
        constexpr bool skip_hlkc = false;
        pass &= CompileProgram(device, program, skip_hlkc);

        /*
         * Create source data and write to DRAM.
         */
        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 1, std::chrono::system_clock::now().time_since_epoch().count());

        pass &= WriteToDeviceDRAM(src0_dram_buffer, src0_vec);

        constexpr float val_to_add = -1.0f;
        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_add);

        pass &= WriteToDeviceDRAM(src1_dram_buffer, src1_vec);

        /*
         * Configure program and runtime kernel arguments, then execute.
         */
        pass &= ConfigureDeviceWithProgram(device, program);

        WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernel,
            core,
            {
                src0_dram_buffer->address(),
                static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().y),
                num_tiles,
                src1_dram_buffer->address(),
                static_cast<uint32_t>(src1_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(src1_dram_buffer->noc_coordinates().y),
                num_tiles,
                0
            }
        );

        WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {
                dst_dram_buffer->address(),
                static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().y),
                num_tiles
            }
        );

        pass &= LaunchKernels(device, program);

        /*
         * Read in result into a host vector.
         */
        std::vector<uint32_t> result_vec;
        ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

        /*
         * Move src data back into DRAM src buffer 0 to do another eltwise calculation
         */
        Program *program_mul = new Program();

        /*
         * Because we're using a new program, we must redeclare all the
         * circular buffers and kernels.
         */
        cb_src0 = CreateCircularBuffer(
            program_mul,
            device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src0_cb_addr,
            tt::DataFormat::Float16_b
        );

        cb_src1 = CreateCircularBuffer(
            program_mul,
            device,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            src1_cb_addr,
            tt::DataFormat::Float16_b
        );

        cb_output = CreateCircularBuffer(
            program_mul,
            device,
            output_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            output_cb_addr,
            tt::DataFormat::Float16_b
        );

        binary_reader_kernel = CreateDataMovementKernel(
            program_mul,
            "kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            DataMovementProcessor::RISCV_1,
            NOC::RISCV_1_default);

        unary_writer_kernel = CreateDataMovementKernel(
            program_mul,
            "kernels/dataflow/writer_unary.cpp",
            core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default);

        eltwise_binary_args = InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

        eltwise_binary_kernel = CreateComputeKernel(
            program_mul,
            "kernels/compute/eltwise_binary.cpp",
            core,
            eltwise_binary_args,

            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        /*
         * But now let's do an eltwise mul!
         */
        eltwise_binary_kernel->add_define("ELTWISE_OP", "mul_tiles");

        /*
         * Compile kernels.
         */
        pass &= CompileProgram(device, program_mul, skip_hlkc);

        /*
         * Send new input data.
         */
        pass &= WriteToDeviceDRAM(src0_dram_buffer, result_vec);

        constexpr float val_to_mul = 2.0f;
        src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, val_to_mul);

        pass &= WriteToDeviceDRAM(src1_dram_buffer, src1_vec);

        pass &= ConfigureDeviceWithProgram(device, program_mul);

        /*
         * Configure program and runtime kernel arguments.
         */
        WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernel,
            core,
            {
                src0_dram_buffer->address(),
                static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().y),
                num_tiles,
                src1_dram_buffer->address(),
                static_cast<uint32_t>(src1_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(src1_dram_buffer->noc_coordinates().y),
                num_tiles,
                0
            }
        );

        WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            {
                dst_dram_buffer->address(),
                static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().x),
                static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().y),
                num_tiles
            }
        );


        /*
         * Execute.
         */
        pass &= LaunchKernels(device, program_mul);

        /*
         * Read the result and compare to a golden result. Record pass/fail
         * and teardown.
         */
        ReadFromDeviceDRAM(dst_dram_buffer, result_vec);

        std::function<bfloat16(const bfloat16 &)> transform_to_golden = [](const bfloat16 &a) {
            return bfloat16((a.to_float() + val_to_add) * val_to_mul);
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

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        tt::log_fatal(tt::LogTest, "Test Failed");
    }

    TT_ASSERT(pass);

    return 0;
}
