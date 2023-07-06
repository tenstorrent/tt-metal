#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;
using namespace tt;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks multi_core_split_fused_qkv(const Tensor &a, std::vector<Tensor>& output, CoreCoord compute_and_storage_grid_size) {

    const auto& ashape = a.shape();

    TT_ASSERT(ashape[0] == 9 and ashape[1] == 1 and ashape[2] == 384 and ashape[3] == 3072, "Input shape to this TM must be [9, 1, 384, 3072]!");
    TT_ASSERT(a.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_ASSERT(a.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_ASSERT(a.dtype() == tt::tt_metal::DataType::BFLOAT16 || a.dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");

    tt_metal::Device *device = a.device();

    // TODO: CHANGE TO FUNCTION CONVERSION
    tt::DataFormat cb_data_format = tt::DataFormat::Bfp8_b;
    if (a.dtype() == tt::tt_metal::DataType::BFLOAT16) {
        cb_data_format = tt::DataFormat::Float16_b;
    }

    uint32_t single_tile_size = tt_metal::TileSize(cb_data_format);
    tt_metal::Buffer *in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);


    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_core_tiles = ashape[3] / TILE_WIDTH;
    uint32_t num_tensors = 3;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;

    // Parallelize ashape[2] (384 / 32 = 12 tiles) across columns
    // Parallelize ashape[0] (9) across rows
    uint32_t num_cores_x = ashape[2] / TILE_HEIGHT;
    uint32_t num_cores_y = ashape[0];
    TT_ASSERT(num_cores_x <= compute_and_storage_grid_size.x);
    TT_ASSERT(num_cores_y <= compute_and_storage_grid_size.y);
    CoreCoord core_range = {num_cores_x, num_cores_y};


    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    tt_metal::Tensor& q = output[0];
    tt_metal::Tensor& k = output[1];
    tt_metal::Tensor& v = output[2];

    tt_metal::Buffer *q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer *k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer *v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores{
        .start={(std::size_t) start_core_x, (std::size_t) start_core_y},
        .end={(std::size_t) start_core_x + num_cores_c - 1, (std::size_t) start_core_y + num_cores_r - 1},
    };

    bool tile_dtype_is_bfloat16 = a.dtype() == tt::tt_metal::DataType::BFLOAT16;
    bool in0_is_dram = in0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool out_is_dram = q_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) tile_dtype_is_bfloat16,
            (std::uint32_t) in0_is_dram,

            // READER COMPILE TIME ARGS
            (std::uint32_t) num_tensors, // out_num_tensors
            (std::uint32_t) num_tiles_per_tensor, // out_num_tiles_per_tensor
    };
    std::vector<uint32_t> writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) tile_dtype_is_bfloat16,
            (std::uint32_t) out_is_dram,

            // WRITER COMPILE TIME ARGS
            (std::uint32_t) num_tensors, // out_num_tensors
            (std::uint32_t) num_tiles_per_tensor, // out_num_tiles_per_tensor
    };

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_tm_tile_layout_split_qkv.cpp",
        all_cores,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);
    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_tm_tile_layout_split_qkv.cpp",
        all_cores,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_1_default);

    // Dummy compute kernel
    std::vector<uint32_t> compute_args = {0}; // dummy
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto dummy_compute_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/blank.cpp",
        all_cores,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );

    // Create circular buffers
    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = per_core_tiles * 2; // double buffer
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        all_cores,
        cb0_tiles,
        cb0_tiles * single_tile_size,
        cb_data_format
    );

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};
            uint32_t core_id = core_idx_x + core_idx_y * num_cores_c;

            std::vector<uint32_t> reader_runtime_args = {
                (std::uint32_t) in0_buffer->address(),
                core_id * per_core_tiles,
            };
            std::vector<uint32_t> writer_runtime_args = {
                (std::uint32_t) q_buffer->address(), // q_tensor_addr
                (std::uint32_t) k_buffer->address(), // k_tensor_addr
                (std::uint32_t) v_buffer->address(), // v_tensor_addr
                core_id * num_tiles_per_tensor,
            };

            tt_metal::SetRuntimeArgs(reader_kernel, core, reader_runtime_args);
            tt_metal::SetRuntimeArgs(writer_kernel, core, writer_runtime_args);
        }
    }

    auto override_runtime_args_callback = [
            reader_kernel,
            writer_kernel,
            num_cores_r,
            num_cores_c,
            start_core_x,
            start_core_y
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);

        auto dst_dram_buffer_query = output_buffers.at(0);
        auto dst_dram_buffer_key = output_buffers.at(1);
        auto dst_dram_buffer_value = output_buffers.at(2);

        for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
            for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
                CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

                {
                    auto runtime_args = GetRuntimeArgs(reader_kernel, core);
                    runtime_args[0] = src_dram_buffer->address();
                    SetRuntimeArgs(reader_kernel, core, runtime_args);
                }

                {
                    auto runtime_args = GetRuntimeArgs(writer_kernel, core);
                    runtime_args[0] = dst_dram_buffer_query->address();
                    runtime_args[1] = dst_dram_buffer_key->address();
                    runtime_args[2] = dst_dram_buffer_value->address();
                    SetRuntimeArgs(writer_kernel, core, runtime_args);
                }
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

} // namespace tt_metal

} // namespace tt
