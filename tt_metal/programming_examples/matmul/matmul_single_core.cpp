// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

//#include "tt_metal/include/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "common/bfloat16.hpp"

using namespace tt::constants;
using namespace std;
using namespace tt;
using namespace tt::tt_metal;


void golden_matmul(vector<uint32_t>& a, vector<uint32_t>& b, vector<uint32_t>& output,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B) {
    std::uint32_t idx_c = 0;
    std::uint32_t idx_a = 0;
    std::uint32_t idx_b = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            idx_c = j+ (i * N);
            output[idx_c] = 0;
            idx_a = i * K;
            idx_b = j * M;
            for (int k_m = 0; k_m < K; k_m++) {
                idx_a += 1;
                idx_b += K;
                output[idx_c] += a[idx_a] * b[idx_b];
            }
        }
    }

}

void matmul_single_core(vector<uint32_t>& a, vector<uint32_t>& b, vector<uint32_t>& output, bool bcast_batch,
                        uint32_t M, uint32_t N, uint32_t K, uint32_t B, Device* device) {

    /*
    * Setup program to execute along with its buffers and kernels to use
    * Core range is just single core
    */
    Program program{};
    CoreRange core = {.start={0, 0}, .end={0, 0}};

    /*
    * EXtracting MAtrix dimensions from input/output vectors
    */
    // C = A*B
    // MN = MK*KN
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    /*
    * Create DRAM Buffers for input and output vectors
    * Writing data from input vectors to source buffers
    */
    DataFormat cb_data_format = DataFormat::Float16_b;
    uint32_t single_tile_size = detail::TileSize(cb_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t single_datum_size = sizeof(std::uint32_t);

    uint32_t dram_buffer_A_size = single_datum_size * M * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_B_size = single_datum_size * N * K; // num_tiles of FP16_B, hard-coded in the reader/writer kernels
    uint32_t dram_buffer_C_size = single_datum_size * M * N; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    Buffer src0_dram_buffer = CreateBuffer(device, dram_buffer_A_size, dram_buffer_A_size, BufferType::DRAM);
    Buffer src1_dram_buffer = CreateBuffer(device, dram_buffer_B_size, dram_buffer_B_size, BufferType::DRAM);
    Buffer dst_dram_buffer = CreateBuffer(device, dram_buffer_C_size, dram_buffer_C_size, BufferType::DRAM);
    uint32_t src0_addr = src0_dram_buffer.address();
    uint32_t src1_addr = src1_dram_buffer.address();
    uint32_t dst_addr = dst_dram_buffer.address();

    WriteToBuffer(src0_dram_buffer, a);
    WriteToBuffer(src1_dram_buffer, b);

    /*
    * Config of Circular Buffer in the device L1
    * input tiles count is = 2 because it's single tile process, and double-buffer
    */
    uint32_t src0_cb_index = CB::c_in0; //0
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = CB::c_in1; // 1
    CircularBufferConfig cb_src1_config = CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config = CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /*
    * Compile time arguments
    */
    bool src0_is_dram = src0_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_dram_buffer.buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

    /*
    * Create Kernels (Reader, Writer, Compute)
    */
    auto reader_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_bmm_8bank.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        B, // B
        Mt, // Mt
        Kt, // Kt
        Nt // Nt
    };
    auto matmul_single_core_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args}
    );

    /*
    * Kernels - Runtime arguments
    */
    tt_metal::SetRuntimeArgs(
        program, reader_id, core,
        {src0_addr, src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, uint32_t(bcast_batch ? 1 : 0)}
    );

    tt_metal::SetRuntimeArgs(
        program, writer_id, core,
        {dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
    );

    /*
    auto override_runtime_args_callback = [
        reader_kernel_id=reader_id,
        writer_kernel_id=writer_id
    ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_dram_buffer_a->address();
            runtime_args[1] = src_dram_buffer_b->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_dram_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };
    */

    /* Launch program & read in output buffer result into the host vector */
    LaunchProgram(device, program);
    ReadFromBuffer(dst_dram_buffer, output);
}


///////////////////////////////////////



int main(int argc, char **argv) {
    bool pass = true;
    //auto slow_dispatch_mode = 1;

    try {
        /* Silicon accelerator setup */
        constexpr int device_id = 0;
        Device *device = CreateDevice(device_id);

        /* Create source data */
        constexpr uint32_t M = 640;
        constexpr uint32_t N = 640;
        constexpr uint32_t K = 640;
        constexpr uint32_t B = 1;

        /* input vectors with row-major config */
        std::vector<uint32_t> src0_vec = create_random_vector2d_of_bfloat16(
            M, K, 1, std::chrono::system_clock::now().time_since_epoch().count());
        std::vector<uint32_t> src1_vec = create_random_vector2d_of_bfloat16(
            K, N, 1, std::chrono::system_clock::now().time_since_epoch().count());

        /* Calling the MatMul host program. Read in result into a host vector */
        vector<uint32_t> result_vec;
        matmul_single_core(src0_vec, src1_vec, result_vec, false, M, N, K, B, device);
        cout << "----m--" << endl;
        cout << result_vec.size() << endl;
        for (int i = 0; i < 32; i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(result_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            cout << a1<< "  " << a2 << endl;
        }

        vector<uint32_t> golden_vec(M * N);
        golden_matmul(src0_vec, src1_vec, golden_vec, M, N, K, B);
        cout << "----g--" << endl;
        cout << golden_vec.size() << endl;
        for (int i = 0; i < 32; i++) {
            std::pair<bfloat16, bfloat16> as = unpack_two_bfloat16_from_uint32(golden_vec.at(i));
            float a1 = as.first.to_float();
            float a2 = as.second.to_float();
            cout << a1 << "  " << a2 << endl;
        }

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
