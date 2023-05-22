#include "tt_dnn/op_library/bmm/bmm_op.hpp"

#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

using namespace tt::constants;
using namespace tt;


namespace reuse_optimized_bert_large_helpers {
using namespace tt::constants;
using namespace tt;

tt_metal::Program create_program(
    tt_metal::Device *device,
    tt::DataFormat cb_data_format,
    MathFidelity math_fidelity,
    uint32_t single_tile_size,
    tt_xy_pair core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    tt_metal::Buffer* in0_buffer, tt_metal::Buffer* in1_buffer, tt_metal::Buffer* out_buffer
) {

    tt_metal::Program program = tt_metal::Program();

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_size = in0_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_size = in1_block_tiles * 2 * single_tile_size; // double buffer
    uint32_t out_CB_tiles = per_core_M * per_core_N;
    uint32_t out_CB_size = out_CB_tiles * single_tile_size;

    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    uint32_t num_cores = core_range.x * core_range.y;
    uint32_t num_block_rows_per_batch = (M / per_core_M);
    uint32_t num_block_cols_per_batch = (N / per_core_N);
    uint32_t num_output_blocks_per_batch = num_block_rows_per_batch * num_block_cols_per_batch;
    uint32_t num_output_blocks_total = B * (M / per_core_M) * (N / per_core_N);
    uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;
    std::vector<uint32_t> num_output_blocks_per_core(num_cores, num_evenly_divided_output_blocks);
    for(uint32_t i = 0; i < num_output_blocks_total % num_cores; i++){
        num_output_blocks_per_core[i]++;
    }

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        uint32_t core_idx_x = i / core_range.y;
        uint32_t core_idx_y = i % core_range.y;
        uint32_t output_idx_batch = num_blocks_written  / num_output_blocks_per_batch;
        uint32_t output_idx_x = num_blocks_written % num_block_cols_per_batch;
        uint32_t output_idx_y = num_blocks_written % num_block_rows_per_batch;
        tt_xy_pair core = {(std::size_t) core_idx_x, (std::size_t) core_idx_y};

        uint32_t src0_cb_index = 0;
        uint32_t cb0_tiles = in0_block_tiles * 2; // double buffer
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src0_cb_index,
            core,
            cb0_tiles,
            cb0_tiles * single_tile_size,
            cb_data_format
        );

        uint32_t src1_cb_index = 1;
        uint32_t cb1_tiles = in1_block_tiles * 2; // double buffer
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            program,
            device,
            src1_cb_index,
            core,
            cb1_tiles,
            cb1_tiles * single_tile_size,
            cb_data_format
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        auto cb_output = tt_metal::CreateCircularBuffer(
            program,
            device,
            ouput_cb_index,
            core,
            out_CB_tiles,
            out_CB_size,
            cb_data_format
        );

        uint32_t interm0_cb_index = 24;
        auto cb_interm0 = tt_metal::CreateCircularBuffer(
            program,
            device,
            interm0_cb_index,
            core,
            out_CB_tiles,
            out_CB_size,
            cb_output->address(),
            cb_data_format
        );

        bool tile_size_is_power_of_two = (ceil(log2(single_tile_size)) == floor(log2(single_tile_size)));
        std::uint32_t tile_size_pow2_exponent = tile_size_is_power_of_two ? (std::uint32_t)log2(single_tile_size) : 0;
        bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        bool in1_is_dram = in1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        tt_metal::KernelArgs reader_writer_compile_time_args = tt_metal::KernelArgs(
            core, {
                // interleaved accessor args
                (std::uint32_t) tile_size_is_power_of_two,
                (std::uint32_t) tile_size_pow2_exponent,
                (std::uint32_t) in0_is_dram,
                (std::uint32_t) in1_is_dram,
                (std::uint32_t) out_is_dram
            }
        );

        vector<uint32_t> compute_kernel_args = {
            in0_block_w, // in0_block_w
            in0_num_subblocks, // in0_num_subblocks
            in0_block_num_tiles, // in0_block_num_tiles
            in0_subblock_num_tiles, // in0_subblock_num_tiles

            in1_num_subblocks, // in1_num_subblocks
            in1_block_num_tiles, // in1_block_num_tiles
            in1_per_core_w, // in1_per_core_w

            num_blocks, // num_blocks

            out_subblock_h, // out_subblock_h
            out_subblock_w, // out_subblock_w
            out_subblock_num_tiles, // out_subblock_num_tiles
            num_output_blocks_per_core[i] // batch
        };
        // Create compute kernel
        tt_metal::KernelArgs mm_args = tt_metal::KernelArgs(core, compute_kernel_args);
        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto mm_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/bmm_large_block_zm.cpp",
            core,
            mm_args,
            math_fidelity,
            fp32_dest_acc_en,
            math_approx_mode
        );

        // Write runtime args to device
        std::vector<uint32_t> mm_reader_args = {
            (std::uint32_t)  in0_buffer->address(), // in0_tensor_addr
            (std::uint32_t)  K * per_core_M * (output_idx_y + output_idx_batch * num_block_rows_per_batch), // in0_tensor_start_tile_id
            (std::uint32_t)  1, // in0_tensor_stride_w
            (std::uint32_t)  K, // in0_tensor_stride_h
            (std::uint32_t)  in0_block_w, // in0_tensor_next_block_stride

            (std::uint32_t)  in0_block_w, // in0_block_w
            (std::uint32_t)  per_core_M, // in0_block_h
            (std::uint32_t)  in0_block_w * per_core_M, //in0_block_num_tiles

            (std::uint32_t)  in1_buffer->address(), // in1_tensor_addr
            (std::uint32_t)  per_core_N * (output_idx_x + K * output_idx_batch * num_block_cols_per_batch), //in1_tensor_start_tile_id
            (std::uint32_t)  1, // in1_tensor_stride_w
            (std::uint32_t)  N, // in1_tensor_stride_h
            (std::uint32_t)  in0_block_w * N, //in1_tensor_next_block_stride

            (std::uint32_t)  per_core_N, // in1_block_w
            (std::uint32_t)  in0_block_w, //in1_block_h
            (std::uint32_t)  per_core_N * in0_block_w, // in1_block_num_tiles

            (std::uint32_t)  K / in0_block_w, // num_blocks

            (std::uint32_t)  M * K, // MtKt
            (std::uint32_t)  K * N, // KtNt
            (std::uint32_t)  num_output_blocks_per_core[i], // batch
            (std::uint32_t)  bcast_batch, // bcast_B
        };

        std::vector<uint32_t> writer_args = {
            (std::uint32_t) out_buffer->address(), // out_tensor_addr
            (std::uint32_t) output_idx_x * per_core_N + (output_idx_y + output_idx_batch * num_block_rows_per_batch) * per_core_M * N, // out_tensor_start_tile_id
            (std::uint32_t) 1, // out_tensor_stride_w
            (std::uint32_t) N,  // out_tensor_stride_h
            (std::uint32_t) out_subblock_w, // out_tensor_next_subblock_stride_w
            (std::uint32_t) out_subblock_h * N, // out_tensor_next_subblock_stride_h

            (std::uint32_t) out_subblock_w, // out_subblock_w
            (std::uint32_t) out_subblock_h, // out_subblock_h
            (std::uint32_t) (out_subblock_w * out_subblock_h), // out_subblocks_w * out_subblocks_h
            (std::uint32_t) (per_core_N / out_subblock_w), // out_num_subblocks_w
            (std::uint32_t) (per_core_M / out_subblock_h), // out_num_subblocks_h

            (std::uint32_t) M * N, // MtNt
            (std::uint32_t) num_output_blocks_per_core[i] // batch
        };

        // Create reader and writer kernels per core
        // Checkerboard
        // white
        if ((core_idx_x + core_idx_y) % 2 == 0) {
            auto mm_kernel_in0_reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
                core,
                reader_writer_compile_time_args,
                tt_metal::DataMovementProcessor::RISCV_1,
                tt_metal::NOC::RISCV_1_default
            );

            auto mm_kernel_in1_reader_writer = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
                core,
                reader_writer_compile_time_args,
                tt_metal::DataMovementProcessor::RISCV_0,
                tt_metal::NOC::RISCV_0_default
            );

            tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_reader, core, mm_reader_args);
            mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end()-1);
            tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_reader_writer, core, mm_reader_args);
        }
        // black
        else {
            auto mm_kernel_in0_reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
                core,
                reader_writer_compile_time_args,
                tt_metal::DataMovementProcessor::RISCV_1,
                tt_metal::NOC::RISCV_0_default
            );

            auto mm_kernel_in1_reader_writer = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
                core,
                reader_writer_compile_time_args,
                tt_metal::DataMovementProcessor::RISCV_0,
                tt_metal::NOC::RISCV_1_default
            );

            tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_reader, core, mm_reader_args);
            mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end()-1);
            tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_reader_writer, core, mm_reader_args);
        }

        /* Uncomment if we don't checkerboard
        auto mm_kernel_in0_reader = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
            core,
            reader_writer_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_1,
            num_output_blocks_per_core[i] > num_evenly_divided_output_blocks ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default);

        auto mm_kernel_in1_reader_writer = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
            core,
            reader_writer_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_0,
            num_output_blocks_per_core[i] > num_evenly_divided_output_blocks ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default);

        tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in0_reader, core, mm_reader_args);
        mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end()-1);
        tt_metal::WriteRuntimeArgsToDevice(device, mm_kernel_in1_reader_writer, core, mm_reader_args);
        */

        num_blocks_written += num_output_blocks_per_core[i];
    }

    return std::move(program);
}

}


namespace tt {

namespace tt_metal {


Tensor matmul_multi_core_reuse_optimized_bert_large_(const Tensor &a, const Tensor &b, const MemoryConfig& mem_config, bool bcast_batch, tt_xy_pair compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {

    const auto& ashape = a.shape(), bshape = b.shape();

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(not a.on_host() and not b.on_host(), "Operands to matmul need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to matmul need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to matmul need to be allocated in buffers on device!");

    TT_ASSERT(bcast_batch == false, "Bcast batch not supported for this parallelization");

    TT_ASSERT(a.dtype() == b.dtype());
    TT_ASSERT(a.dtype() == tt::tt_metal::DataType::BFLOAT16 || a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    tt_metal::Device *device = a.device();

    // TODO: CHANGE TO FUNCTION CONVERSION
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }
    if (cb_data_format != output_cb_data_format) {
        log_warning("Input tensor datatype does not match target output dataformat. Defaulting to input dataformat.");
    }
    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    tt_metal::Buffer *in1_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);
    TT_ASSERT(in1_buffer->size() % single_tile_size == 0);

    TT_ASSERT(ashape[3] == bshape[2], "Dimension K (A.shape[2] and B.shape[3]) must match for A and B in bmm_op"); // A.K == B.K
    TT_ASSERT(ashape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(ashape[3] % TILE_WIDTH == 0);
    TT_ASSERT(bshape[2] % TILE_HEIGHT == 0);
    TT_ASSERT(bshape[3] % TILE_WIDTH == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      Matmul Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // NOTE: Only supports matmuls where output is blocks of 16 x 16 tiles (ie. multiples of 16*32 x 16*32)
    // NOTE: Maximum number of tiles in output is 120 * 16^2 = 30,720 (eg. [1, 1, 5120, 6144])
    uint32_t B = ashape[0]*ashape[1];
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;

    TT_ASSERT(fuse_batch, "Only fuse_batch=true is supported for bert large optimized bmm!");
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;

    // Get large matmul params

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t num_blocks_total = B * (Mt / per_core_M) * (Nt / per_core_N);
    tt_xy_pair core_range = compute_and_storage_grid_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    std::array<uint32_t, 4> cshape{ashape[0], ashape[1], ashape[2], bshape[3]}; // C=A*B, N1MK*11KN->N1MN
    tt_metal::Tensor output = tt_metal::Tensor(cshape, a.dtype(), tt::tt_metal::Layout::TILE, device, mem_config);
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = reuse_optimized_bert_large_helpers::create_program(
        device,
        cb_data_format,
        math_fidelity,
        single_tile_size,
        core_range,
        B, Mt, Nt, Kt,
        bcast_batch,
        in0_block_w,
        out_subblock_h, out_subblock_w,
        per_core_M, per_core_N,
        in0_buffer, in1_buffer, out_buffer
    );

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool pass = true;
    pass &= tt_metal::CompileProgram(device, program);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    TT_ASSERT(pass);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

// matmul_multi_core_reuse_optimized_bert_large not used
Tensor bmm_multi_core_reuse_optimized_bert_large(const Tensor& a, const Tensor& b, const MemoryConfig& mem_config, tt_xy_pair compute_and_storage_grid_size, tt::DataFormat output_cb_data_format, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {
    return matmul_multi_core_reuse_optimized_bert_large_(a, b, mem_config, false, compute_and_storage_grid_size, output_cb_data_format, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

}  // namespace tt_metal

}  // namespace tt
