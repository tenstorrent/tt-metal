// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;


namespace reuse_optimized_helpers {
using namespace tt::constants;
using namespace tt;
using namespace tt_metal;
operation::ProgramWithCallbacks create_program(
    tt_metal::Device *device,
    MathFidelity math_fidelity,
    CoreCoord core_range,
    uint32_t B, uint32_t M, uint32_t N, uint32_t K,
    bool bcast_batch,
    uint32_t in0_block_w,
    uint32_t out_subblock_h, uint32_t out_subblock_w,
    uint32_t per_core_M, uint32_t per_core_N,
    tt_metal::Buffer* in0_buffer, tt_metal::Buffer* in1_buffer, tt_metal::Buffer* out_buffer,
    tt::DataFormat in0_data_format, tt::DataFormat in1_data_format, tt::DataFormat output_data_format
) {

    tt_metal::Program program{};

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles * 2; // double buffer
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles * 2; // double buffer
    uint32_t in1_CB_size = in1_CB_tiles * in1_single_tile_size;
    uint32_t out_block_tiles = per_core_M * per_core_N;
    uint32_t out_CB_tiles = out_block_tiles; // No double buffer
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;


    // Compute kernel compile time args
    uint32_t num_blocks = (K/in0_block_w);

    uint32_t in0_num_subblocks = (per_core_M/out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;

    uint32_t in1_num_subblocks = (per_core_N/out_subblock_w);
    uint32_t in1_block_num_tiles = out_subblock_w*in0_block_w*in1_num_subblocks;
    uint32_t in1_per_core_w = out_subblock_w * in1_num_subblocks;

    uint32_t out_subblock_num_tiles = out_subblock_h*out_subblock_w;

    uint32_t num_block_rows_per_batch = (M / per_core_M);
    uint32_t num_block_cols_per_batch = (N / per_core_N);
    uint32_t num_output_blocks_per_batch = num_block_rows_per_batch * num_block_cols_per_batch;
    uint32_t num_output_blocks_total = B * (M / per_core_M) * (N / per_core_N);

    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] = tt_metal::split_work_to_cores(core_range, num_output_blocks_total);
    // TODO: This contains same information as above; refactor this?
    uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;
    std::vector<uint32_t> num_output_blocks_per_core(num_cores, num_evenly_divided_output_blocks);
    for(uint32_t i = 0; i < num_output_blocks_total % num_cores; i++){
        num_output_blocks_per_core[i]++;
    }

    // Assume all of core_range is used (ie. num_evenly_divided_output_blocks > 0)
    TT_ASSERT(num_evenly_divided_output_blocks > 0, "Not all cores from core_range was used!");
    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange left_half{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + 5, (std::size_t) start_core_y + num_cores_r - 1}};

    CoreRange right_half{
        .start={(std::size_t) start_core_x + 6, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1}};

    // Compile time args
    bool in0_is_dram = in0_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    bool in1_is_dram = in1_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    bool out_is_dram = out_buffer->buffer_storage() == tt_metal::BufferStorage::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) in0_is_dram,
    };
    std::vector<uint32_t> reader_writer_compile_time_args = {
        // interleaved accessor args
        (std::uint32_t) in1_is_dram,
        (std::uint32_t) out_is_dram
    };

    // left half
    auto mm_kernel_in0_reader_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
        left_half,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_compile_time_args}
    );

    auto mm_kernel_in1_reader_writer_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
        left_half,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_writer_compile_time_args}
    );

    // right half
    auto mm_kernel_in0_reader_other_noc_setup_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
        right_half,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args}
    );

    auto mm_kernel_in1_reader_writer_other_noc_setup_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
        right_half,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = reader_writer_compile_time_args}
    );

    vector<uint32_t> compute_kernel_args_group_1 = {
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
        num_blocks_per_core_group_1 // batch
    };

    vector<uint32_t> compute_kernel_args_group_2 = {
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
        num_blocks_per_core_group_2 // batch
    };

    // Create compute kernel
    auto mm_kernel_group_1_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_large_block_zm_mixed_precision.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args_group_1}
    );
    auto mm_kernel_group_2_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/bmm_large_block_zm_mixed_precision.cpp",
        core_group_2,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_kernel_args_group_2}
    );

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
		.set_page_size(src0_cb_index, in0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(in1_CB_size, {{src1_cb_index, in1_data_format}})
		.set_page_size(src1_cb_index, in1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t interm0_cb_index = 24;
    std::map<uint8_t, tt::DataFormat> output_cb_data_format_spec = {
        {output_cb_index, output_data_format},
        {interm0_cb_index, output_data_format}
    };
    tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(out_CB_size, output_cb_data_format_spec)
		.set_page_size(output_cb_index, output_single_tile_size)
        .set_page_size(interm0_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, CoreRangeSet({all_cores}), cb_output_config);

    std::vector<KernelID> reader_kernel_ids;
    std::vector<KernelID> writer_kernel_ids;
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        uint32_t core_idx_x = i / core_range.y;
        uint32_t core_idx_y = i % core_range.y;
        uint32_t output_idx_batch = num_blocks_written  / num_output_blocks_per_batch;
        uint32_t output_idx_x = num_blocks_written % num_block_cols_per_batch;
        uint32_t output_idx_y = num_blocks_written % num_block_rows_per_batch;
        CoreCoord core = {(std::size_t) core_idx_x, (std::size_t) core_idx_y};

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
        };

        // left half
        if (core_idx_x <= 5) {
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_reader_id, core, mm_reader_args);
            mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end());
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_reader_writer_id, core, mm_reader_args);
            reader_kernel_ids.push_back(mm_kernel_in0_reader_id);
            writer_kernel_ids.push_back(mm_kernel_in1_reader_writer_id);
        }
        // right half
        else {
            tt_metal::SetRuntimeArgs(program, mm_kernel_in0_reader_other_noc_setup_id, core, mm_reader_args);
            mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end());
            tt_metal::SetRuntimeArgs(program, mm_kernel_in1_reader_writer_other_noc_setup_id, core, mm_reader_args);
            reader_kernel_ids.push_back(mm_kernel_in0_reader_other_noc_setup_id);
            writer_kernel_ids.push_back(mm_kernel_in1_reader_writer_other_noc_setup_id);
        }
        /* Checkerboard logic
        // white
        if ((core_idx_x + core_idx_y) % 2 == 0) {
            auto mm_kernel_in0_reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
                core,
                reader_compile_time_args,
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

            tt_metal::SetRuntimeArgs(mm_kernel_in0_reader, core, mm_reader_args);
            mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end()-1);
            tt_metal::SetRuntimeArgs(mm_kernel_in1_reader_writer, core, mm_reader_args);
        }
        // black
        else {
            auto mm_kernel_in0_reader = tt_metal::CreateDataMovementKernel(
                program,
                "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
                core,
                reader_compile_time_args,
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

            tt_metal::SetRuntimeArgs(mm_kernel_in0_reader, core, mm_reader_args);
            mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end()-1);
            tt_metal::SetRuntimeArgs(mm_kernel_in1_reader_writer, core, mm_reader_args);
        }
        */

        /* Uncomment if we don't checkerboard
        auto mm_kernel_in0_reader = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_bmm_tile_layout_in0.cpp",
            core,
            reader_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_1,
            num_output_blocks_per_core[i] > num_evenly_divided_output_blocks ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default);

        auto mm_kernel_in1_reader_writer = tt_metal::CreateDataMovementKernel(
            program,
            "tt_metal/kernels/dataflow/reader_writer_bmm_tile_layout_in1.cpp",
            core,
            reader_writer_compile_time_args,
            tt_metal::DataMovementProcessor::RISCV_0,
            num_output_blocks_per_core[i] > num_evenly_divided_output_blocks ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default);

        tt_metal::SetRuntimeArgs(mm_kernel_in0_reader, core, mm_reader_args);
        mm_reader_args.insert(mm_reader_args.end(), writer_args.begin(), writer_args.end()-1);
        tt_metal::SetRuntimeArgs(mm_kernel_in1_reader_writer, core, mm_reader_args);
        */

        num_blocks_written += num_output_blocks_per_core[i];
    }

    auto override_runtime_args_callback = [
            reader_kernel_ids,
            writer_kernel_ids,
            num_cores,
            core_range
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0; i < num_cores; i++){
            uint32_t core_idx_x = i / core_range.y;
            uint32_t core_idx_y = i % core_range.y;
            CoreCoord core = {(std::size_t) core_idx_x, (std::size_t) core_idx_y};

            {
                auto reader_kernel_id = reader_kernel_ids.at(i);
                auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer_a->address();
                runtime_args[8] = src_dram_buffer_b->address();
                SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
            }

            {
                auto writer_kernel_id = writer_kernel_ids.at(i);
                auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
                runtime_args[0] = src_dram_buffer_a->address();
                runtime_args[8] = src_dram_buffer_b->address();
                runtime_args[21] = dst_dram_buffer->address();
                SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}


namespace tt {

namespace tt_metal {


operation::ProgramWithCallbacks matmul_multi_core_reuse_optimized_(const Tensor &a, const Tensor &b, const Shape &ashape, const Shape &bshape, Tensor& output, bool bcast_batch, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {

    // Pass in a and b shapes instead

    TT_ASSERT(bcast_batch == false, "Bcast batch not supported for this parallelization");

    // CB dataformats
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype()); // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype()); // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output_dtype); // output

    tt_metal::Device *device = a.device();

    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    tt_metal::Buffer *in1_buffer = b.buffer();
    if (bcast_batch)
        TT_ASSERT(bshape[0]*bshape[1] == 1 && "matmul (batch bcast variant) expects input tensors of shapes BCMK*11KN=BCMN");
    else {
        // same condition as above, different message
        TT_ASSERT(ashape[1] == bshape[1] && ashape[0] == bshape[0]
            && "bmm (non-bcast matmul) expects input tensors of shapes BCMK*BCKN=BCMN");
    }
    TT_ASSERT(in0_buffer->size() % in0_single_tile_size == 0);
    TT_ASSERT(in1_buffer->size() % in1_single_tile_size == 0);

    TT_ASSERT(ashape[3] == bshape[2], "Dimension K (A.shape[3] and B.shape[2]) must match for A and B in bmm_op"); // A.K == B.K
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

    // TODO: Generalize
    TT_ASSERT(fuse_batch, "Only fuse_batch=true is supported for bert large optimized bmm!");
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // Get large matmul params

    TT_ASSERT(Mt % per_core_M == 0);
    TT_ASSERT(Nt % per_core_N == 0);
    TT_ASSERT(Kt % in0_block_w == 0);

    uint32_t num_blocks_total = B * (Mt / per_core_M) * (Nt / per_core_N);
    CoreCoord core_range = compute_with_storage_grid_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // Pass in cshape instead
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    return reuse_optimized_helpers::create_program(
        device,
        math_fidelity,
        core_range,
        B, Mt, Nt, Kt,
        bcast_batch,
        in0_block_w,
        out_subblock_h, out_subblock_w,
        per_core_M, per_core_N,
        in0_buffer, in1_buffer, out_buffer,
        in0_data_format, in1_data_format, output_data_format
    );
}

// TODO: Get rid of no-op reshapes when we generalize
// matmul_multi_core_reuse_optimized_bert_large not used
operation::ProgramWithCallbacks bmm_multi_core_reuse_optimized(const Tensor& a, const Tensor& b, const Shape& ashape, const Shape& bshape, Tensor& output, CoreCoord compute_with_storage_grid_size, tt::tt_metal::DataType output_dtype, MathFidelity math_fidelity, uint32_t in0_block_w, uint32_t out_subblock_h, uint32_t out_subblock_w, uint32_t per_core_M, uint32_t per_core_N, bool fuse_batch) {
    /*
     * For pre-softmax and post-softmax bmm, do an additional no-op reshape by changing cshape and ashape
     * - pre-softmax: [9, 16, 384, 64] x [9, 16, 64, 384] = ([9, 16, 384, 384] -> [9, 1, 6144, 384])
     * - post-softmax: ([9, 1, 6144, 384] -> [9, 16, 384, 384]) x [9, 16, 384, 64] = [9, 16, 384, 64]
     * NOTE: Only need to pass in the right cshape and ashape for these no-op reshapes.
     * The actual bmm op works on [9, 16, 384, 64] x [9, 16, 64, 384] and [9, 16, 384, 384] x [9, 16, 384, 64].
    */
    return matmul_multi_core_reuse_optimized_(a, b, ashape, bshape, output, false, compute_with_storage_grid_size, output_dtype, math_fidelity, in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N, fuse_batch);
}

}  // namespace tt_metal

}  // namespace tt
