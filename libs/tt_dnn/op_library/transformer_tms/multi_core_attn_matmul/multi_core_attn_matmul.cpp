#include "tt_dnn/op_library/transformer_tms/transformer_tms.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;
using namespace tt;

namespace tt {
namespace operations {
namespace primary {
namespace transformers {


operation::ProgramWithCallbacks multi_core_attn_matmul(const Tensor &a, const Tensor &b, Tensor& output, CoreCoord compute_with_storage_grid_size, DataType output_dtype) {

    tt_metal::Program program{};

    const auto& ashape = a.shape(), bshape = b.shape();

    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t in0_single_tile_size = tt_metal::detail::TileSize(in0_data_format);
    uint32_t in1_single_tile_size = tt_metal::detail::TileSize(in1_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);
    MathFidelity math_fidelity = MathFidelity::HiFi4;

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();
    Shape cshape = output.shape();

    // A block of work is one MtNt
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto num_output_blocks_total = ashape[1]; // ashape[1] is Q num_heads; only parallelize on this
    auto [num_cores, all_cores, core_group_1, core_group_2, num_output_blocks_per_core_group_1, num_output_blocks_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, num_output_blocks_total);

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // C = torch.matmul(A.transpose(0, 2) * B).transpose(0, 2)
    // MN = MK*KN
    uint32_t B = ashape[1];  // ashape[0] is q_len
    uint32_t Mt = ashape[2]/TILE_HEIGHT;
    uint32_t Kt = ashape[3]/TILE_WIDTH;
    uint32_t Nt = bshape[3]/TILE_WIDTH;
    uint32_t KtNt = Kt * Nt;
    uint32_t MtKt = Mt * Kt;
    uint32_t MtNt = Mt * Nt;

    uint32_t src0_addr = src0_buffer->address();
    uint32_t src1_addr = src1_buffer->address();
    uint32_t dst_addr = dst_buffer->address();

    uint32_t src0_cb_index = 0;
    uint32_t cb0_num_input_tiles = Kt * 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        all_cores,
        cb0_num_input_tiles,
        cb0_num_input_tiles * in0_single_tile_size,
        in0_data_format
    );

    uint32_t src1_cb_index = 1;
    uint32_t cb1_num_input_tiles = 2;
    auto cb_src1 = tt_metal::CreateCircularBuffers(
        program,
        src1_cb_index,
        all_cores,
        2,
        cb1_num_input_tiles * in1_single_tile_size,
        output_data_format
    );

    uint32_t cb_intermed0_index = 24;
    auto cb_interm0 = tt_metal::CreateCircularBuffers(
        program,
        cb_intermed0_index,
        all_cores,
        1,
        1 * output_single_tile_size,
        output_data_format
    );

    uint32_t cb_intermed1_index = 25;
    auto cb_interm1 = tt_metal::CreateCircularBuffers(
        program,
        cb_intermed1_index,
        all_cores,
        1,
        1 * output_single_tile_size,
        output_data_format
    );

    uint32_t cb_intermed2_index = 26;
    auto cb_interm2 = tt_metal::CreateCircularBuffers(
        program,
        cb_intermed2_index,
        all_cores,
        1,
        1 * output_single_tile_size, // 64 is one row of bfloat16
        output_data_format
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = 2;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        all_cores,
        num_output_tiles,
        num_output_tiles * output_single_tile_size,
        output_data_format
    );
    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram
    };

    auto reader_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_transformer_attn_matmul.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    auto writer_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args_group_1 = {
        1, // B
        1, // Mt
        Kt, // Kt
        num_output_blocks_per_core_group_1 * MtNt // Nt
    }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

    auto eltwise_binary_kernel_group_1_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/transformer_attn_matmul.cpp",
        core_group_1,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_1}
    );

    if (!core_group_2.ranges().empty()) {
        vector<uint32_t> compute_args_group_2 = {
            1, // B
            1, // Mt
            Kt, // Kt
            num_output_blocks_per_core_group_2 * MtNt // Nt
        }; // bmm compute kernel the B, Mt, Nt are just 3 for loops that technically act as 1 large loop, so only set Nt for simplicity

        auto eltwise_binary_kernel_group_2_id = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/transformer_attn_matmul.cpp",
            core_group_2,
            tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_args_group_2}
        );
    }

    uint32_t num_output_blocks_per_core;
    constexpr uint32_t num_rows_in_one_tile = 32;
    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_output_blocks_per_core = num_output_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt_metal::SetRuntimeArgs(
            program, reader_id, core,
            {
                src0_addr,
                src1_addr,
                Mt,
                Kt,
                Nt,
                MtKt,
                KtNt * num_rows_in_one_tile, // itileB stride; skips 32 * KtNt in bshape[0] for one block of MtNt
                num_output_blocks_per_core,
                num_blocks_written * MtKt, // itileA_start
                0, // itileB_start; always read in same in1 per core TODO: multi-cast
            }
        );
        tt_metal::SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                dst_addr,
                num_output_blocks_per_core * MtNt,
                num_blocks_written * MtNt,
            }
        );
        num_blocks_written += num_output_blocks_per_core;
    }

    auto override_runtime_args_callback = [
            reader_kernel_id=reader_id,
            writer_kernel_id=writer_id,
            num_cores,
            num_cores_y
        ]
    (
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer_a = input_buffers.at(0);
        auto src_dram_buffer_b = input_buffers.at(1);

        auto dst_dram_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

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
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
