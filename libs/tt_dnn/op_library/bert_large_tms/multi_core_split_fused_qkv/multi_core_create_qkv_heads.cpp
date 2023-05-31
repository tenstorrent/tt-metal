#include "tt_dnn/op_library/bert_large_tms/bert_large_tms.hpp"
#include "tt_metal/host_api.hpp"
#include "common/constants.hpp"

using namespace tt::constants;
using namespace tt;

namespace tt {

namespace tt_metal {

Tensor multi_core_create_qkv_heads(const Tensor &a, const MemoryConfig& mem_config, CoreCoord compute_and_storage_grid_size, bool transpose_hw) {

    const auto& ashape = a.shape();

    TT_ASSERT(ashape[0] == 9 and ashape[1] == 1 and ashape[2] == 384 and ashape[3] == 1024, "Input shape to this TM must be [9, 1, 384, 1024]!");
    TT_ASSERT(not a.on_host(), "Operands to TM need to be on device!");
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
    // Output shape is: [9, 16, 384, 64] (transpose_hw=false) or [9, 16, 64, 384] (transpose_hw=true)
    // For transpose_hw=true, we write "w_dim" to h_dim instead, but keep nomenclature the same
    uint32_t per_core_tiles = ashape[3] / TILE_WIDTH;
    uint32_t out_h_tiles = ashape[2] / TILE_HEIGHT;

    uint32_t out_w = 64;
    uint32_t out_w_tiles = out_w / TILE_WIDTH;
    uint32_t out_c = per_core_tiles / out_w_tiles;
    uint32_t out_HtWt = out_h_tiles * out_w_tiles;
    uint32_t out_CHtWt = out_c * out_HtWt;

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
    std::array<uint32_t, 4> output_shape = {ashape[0], out_c, ashape[2], out_w};
    if (transpose_hw) {
        output_shape = {ashape[0], out_c, out_w, ashape[2]};
    }

    tt_metal::Tensor output = tt_metal::Tensor(output_shape, a.dtype(), tt::tt_metal::Layout::TILE, device, mem_config);
    tt_metal::Buffer *out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");


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
    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) tile_dtype_is_bfloat16,
            (std::uint32_t) in0_is_dram,

            // READER COMPILE TIME ARGS
            (std::uint32_t) in0_buffer->address(), // in0_tensor_addr
            (std::uint32_t) out_w_tiles, // out_w_tiles
            (std::uint32_t) out_c, // out_c
    };
    std::vector<uint32_t> writer_compile_time_args = {
            // interleaved accessor args
            (std::uint32_t) tile_dtype_is_bfloat16,
            (std::uint32_t) out_is_dram,

            // WRITER COMPILE TIME ARGS
            (std::uint32_t) out_buffer->address(), // out_tensor_addr
            (std::uint32_t) out_w_tiles, // out_w_tiles
            (std::uint32_t) out_h_tiles, // out_h_tiles
            (std::uint32_t) out_c, // out_c
            (std::uint32_t) out_HtWt, // out_HtWt
            (std::uint32_t) transpose_hw, // transpose_hw
    };

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_tm_tile_layout_create_head.cpp",
        all_cores,
        reader_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_0_default);
    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_tm_tile_layout_create_head.cpp",
        all_cores,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_1_default);

    // Compute kernel
    // For transpose_hw=false, write and read from cb 0 without compute
    // For transpose_hw=true, write to cb 0, compute, then read from cb 16
    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    if (transpose_hw) {
        std::vector<uint32_t> compute_args = {per_core_tiles};
        auto compute_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/transpose_wh.cpp",
            all_cores,
            compute_args,
            MathFidelity::HiFi4, // TODO: LoFi faster?
            fp32_dest_acc_en,
            math_approx_mode
        );
    }
    else {
        std::vector<uint32_t> compute_args = {0}; // dummy
        auto compute_kernel = tt_metal::CreateComputeKernel(
            program,
            "tt_metal/kernels/compute/blank.cpp",
            all_cores,
            compute_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );
    }

    uint32_t src0_cb_index = 0;
    uint32_t cb0_tiles = per_core_tiles * 2; // double buffer
    uint32_t out_cb_index = 16;
    uint32_t out_cb_tiles = per_core_tiles;
    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t) start_core_x + core_idx_x, (std::size_t) start_core_y + core_idx_y};

            auto cb_src0 = tt_metal::CreateCircularBuffer(
                program,
                device,
                src0_cb_index,
                core,
                cb0_tiles,
                cb0_tiles * single_tile_size,
                cb_data_format
            );
            uint32_t out_tensor_tile_id = core_idx_x * out_w_tiles + core_idx_y * out_CHtWt;

            if (transpose_hw) {
                auto cb_out = tt_metal::CreateCircularBuffer(
                    program,
                    device,
                    out_cb_index,
                    core,
                    out_cb_tiles,
                    out_cb_tiles * single_tile_size,
                    cb_data_format
                );
                out_tensor_tile_id = core_idx_x + core_idx_y * out_CHtWt;
            }

            std::vector<uint32_t> reader_runtime_args = {
                (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles, // in0_tensor_tile_id
            };
            std::vector<uint32_t> writer_runtime_args = {
                out_tensor_tile_id, // out_tensor_tile_id
            };

            tt_metal::WriteRuntimeArgsToDevice(device, reader_kernel, core, reader_runtime_args);
            tt_metal::WriteRuntimeArgsToDevice(device, writer_kernel, core, writer_runtime_args);
        }
    }


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

} // namespace tt_metal

} // namespace tt
