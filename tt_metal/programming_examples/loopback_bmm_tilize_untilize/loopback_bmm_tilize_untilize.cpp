#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "common/bfloat16.hpp"

#include "llrt/tt_debug_print_server.hpp"
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

        pass &= InitializeDevice(device);

        /*
        * Setup program to execute along with its buffers and kernels to use
        */
        Program program = Program();

        constexpr CoreCoord compute_core = {0, 0};
        constexpr CoreCoord jump_core = {0, 1};

        constexpr bool profiler_debugBar = true;

        if (!profiler_debugBar) {
            CoreCoord debug_core = {1, 1};
            tt_start_debug_print_server(device->cluster(), {0}, {debug_core});
        }

        uint32_t num_blocks = stoi(argv[1]);
        uint32_t num_tiles = stoi(argv[2]);
        uint32_t single_tile_size = stoi(argv[3]);
        uint32_t dram_buffer_size = single_tile_size * num_tiles * num_blocks; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

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


        uint32_t in0_block_h = 1;
        uint32_t in0_block_w = num_tiles;
        uint32_t in1_block_w = 1;
        uint32_t in0_num_blocks_h = 1;
        uint32_t in0_num_blocks_w = num_blocks;
        uint32_t in1_num_blocks_w = 1;
        uint32_t in1_block_h = in0_block_w;
        uint32_t in0_block_num_tiles = in0_block_h * in0_block_w;
        uint32_t out_subblock_height_ntiles = 1;
        uint32_t out_subblock_width_ntiles = 1;
        uint32_t out_subblock_ntiles = out_subblock_height_ntiles * out_subblock_width_ntiles;
        uint32_t in0_subblock_h = out_subblock_height_ntiles;
        uint32_t in0_num_subblocks = in0_block_h / in0_subblock_h;
        uint32_t in1_num_subblocks = in1_block_w / out_subblock_width_ntiles;
        uint32_t in0_subblock_num_tiles = in0_subblock_h * in0_block_w;
        uint32_t in1_block_num_tiles = in1_block_w * in1_block_h;
        const uint32_t tile_size_bytes = 2 * constants::TILE_HW;

        uint32_t in0_cb                                 = CB::c_in0;
        uint32_t in1_cb                                 = CB::c_in1;
        uint32_t tilize_mode_tilized_in0_cb             = CB::c_intermed0;
        uint32_t matmul_partials_cb                     = CB::c_intermed1;
        uint32_t untilize_mode_final_matmul_partials_cb = CB::c_intermed2;
        uint32_t untilize_mode_reblock_cb               = CB::c_intermed3;
        uint32_t out0_cb                                = CB::c_out0;

        const uint32_t cb0_ntiles = in0_block_h * in0_block_w * 2;  // double buffer
        CircularBuffer *compute_cb_in0 = CreateCircularBuffer(
            program,
            device,
            in0_cb,
            compute_core,
            cb0_ntiles,
            cb0_ntiles * tile_size_bytes,
            tt::DataFormat::Float16_b
        );
        const uint32_t cb1_ntiles = in1_block_h * in1_block_w * 2;   // double buffer
        CircularBuffer *compute_cb_in1 = CreateCircularBuffer(
            program,
            device,
            in1_cb,
            compute_core,
            cb1_ntiles,
            cb1_ntiles * tile_size_bytes,
            tt::DataFormat::Float16_b
        );
        const uint32_t out_ntiles = in0_block_h * in1_block_w;
        CircularBuffer *compute_cb_output = CreateCircularBuffer(
            program,
            device,
            out0_cb,
            compute_core,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            tt::DataFormat::Float16_b
        );
        CircularBuffer *compute_cb_src0_tilized = CreateCircularBuffer(
            program,
            device,
            tilize_mode_tilized_in0_cb,
            compute_core,
            cb0_ntiles,
            cb0_ntiles * tile_size_bytes,
            tt::DataFormat::Float16_b
        );
        CircularBuffer *compute_cb_matmul_partials = CreateCircularBuffer(
            program,
            device,
            matmul_partials_cb,
            compute_core,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            tt::DataFormat::Float16_b
        );
        // Shares same address space as matmul partials
        CircularBuffer *compute_cb_final_matmul_partials = CreateCircularBuffer(
            program,
            device,
            untilize_mode_final_matmul_partials_cb,
            compute_core,
            out_ntiles,
            out_ntiles * tile_size_bytes,
            tt::DataFormat::Float16_b
        );
        // CB responsible for reorganizing output blocks to fill the whole "per compute_core output block width"
        CircularBuffer *compute_cb_reblock = CreateCircularBuffer(
            program,
            device,
            untilize_mode_reblock_cb,
            compute_core,
            in1_block_w,                    // a single row of tiles
            in1_block_w * tile_size_bytes,
            tt::DataFormat::Float16_b
        );

        constexpr uint32_t src0_cb_index = CB::c_in0;
        const uint32_t num_input_tiles = num_tiles * 2;
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
        const uint32_t num_output_tiles = num_tiles * 2;
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
            "tt_metal/kernels/dataflow/reader_binary_bmm_tilize_untilize.cpp",
            compute_core,
            DataMovementProcessor::RISCV_1,
            NOC::RISCV_1_default);
        DataMovementKernel *jump_binary_reader_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_bmm_tilize_untilize.cpp",
            jump_core,
            DataMovementProcessor::RISCV_1,
            NOC::RISCV_1_default);

        DataMovementKernel *compute_unary_writer_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary_bmm_tilize_untilize.cpp",
            compute_core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default);
        DataMovementKernel *jump_unary_writer_kernel = CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_bmm_tilize_untilize.cpp",
            jump_core,
            DataMovementProcessor::RISCV_0,
            NOC::RISCV_0_default);

        /*
         * Set the parameters that the compute kernel will use.
         */
        vector<uint32_t> compute_kernel_args = {
            num_blocks, // per_core_block_cnt
            1 // per_core_block_size
        };

        constexpr bool fp32_dest_acc_en = false;
        constexpr bool math_approx_mode = false;

        /*
         * Use the add_tiles operation available in the eltwise_binary
         * compute kernel.
         */
        // ComputeKernel *compute_eltwise_binary_kernel = CreateComputeKernel(
        //     program,
        //     "tt_metal/kernels/compute/eltwise_binary.cpp",
        //     compute_core,
        //     compute_kernel_args,
        //     MathFidelity::HiFi4,
        //     fp32_dest_acc_en,
        //     math_approx_mode
        // );
        // add_defines(compute_eltwise_binary_kernel, BinaryOpType::ADD);
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

        std::string compute_kernel = "tt_metal/kernels/compute/bmm_tilize_untilize.cpp";
        std::vector<uint32_t> compute_comptime_args = {
            in0_block_w,
            in0_num_subblocks,
            in0_block_num_tiles,
            in0_subblock_num_tiles,
            in0_subblock_h,
            in1_num_subblocks,
            in1_block_num_tiles,
            in1_block_w,
            in0_num_blocks_h,
            in0_num_blocks_w,
            in1_num_blocks_w,
            out_subblock_height_ntiles,
            out_subblock_width_ntiles,
            out_subblock_ntiles
        };
        auto compute_bmm_compute = CreateComputeKernel(
            program,
            compute_kernel,
            compute_core,
            compute_comptime_args,
            MathFidelity::HiFi4,
            false,  // fp32_dest_acc_en
            false   // math_approx_mode
        );

        /*
        * Compile kernels used during execution
        */
        pass &= CompileProgram(device, program, profiler_debugBar);

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
                // jump_cb_output->address(),
                src0_dram_buffer.address(),
                static_cast<uint32_t>(jump_core.x),
                static_cast<uint32_t>(jump_core.y),
                num_blocks,
                // jump_cb_output->address(),
                src1_dram_buffer.address(),
                static_cast<uint32_t>(jump_core.x),
                static_cast<uint32_t>(jump_core.y),
                num_blocks
                // num_tiles
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
                num_blocks,
                src1_dram_buffer.address(),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().x),
                static_cast<uint32_t>(src1_dram_buffer.noc_coordinates().y),
                num_blocks
                // num_tiles
            }
        );

        WriteRuntimeArgsToDevice(
            device,
            compute_unary_writer_kernel,
            compute_core,
            {
                // jump_cb_src0->address(),
                dst_dram_buffer.address(),
                static_cast<uint32_t>(jump_core.x),
                static_cast<uint32_t>(jump_core.y),
                1
                // num_tiles
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
                1
                // num_tiles
            }
        );

        pass &= LaunchKernels(device, program);
        if (profiler_debugBar) tt_metal::DumpDeviceProfileResults(device, program);

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
