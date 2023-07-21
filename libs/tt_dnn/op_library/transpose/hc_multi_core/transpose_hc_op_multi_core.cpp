#include "tt_dnn/op_library/transpose/transpose_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks transpose_hc_multi_core(const Tensor &a, Tensor &output) {

    const auto shape = a.shape();
    u32 W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    u32 HW = H*W;
    u32 CHW = C*H*W;

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;
    u32 Ct = C/TILE_HEIGHT;

    uint32_t num_tensor_tiles = N*C*H*W / TILE_HW;

    tt_metal::Program program = tt_metal::Program();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_dram_buffer = a.buffer();

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] = split_work_to_cores(compute_and_storage_grid_size, num_tensor_tiles);

    Shape output_shape = output.shape();

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    // Op not uplifted for L1 yet, but need to provide arg to kernel
    bool dst_is_dram = dst_dram_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        static_cast<uint32_t>(DataFormat::Float16_b),
        (std::uint32_t) dst_is_dram
    };

    tt_metal::DataMovementKernel *reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/transpose_hc_8bank_partitioned.cpp",
        all_cores,
        tt_metal::DataMovementProcessor::RISCV_1,
        tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        writer_compile_time_args,
        tt_metal::DataMovementProcessor::RISCV_0,
        tt_metal::NOC::RISCV_0_default);


    for(uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }
        tt_metal::SetRuntimeArgs(
            reader_kernel,
            core,
            {
                src0_dram_buffer->address(),
                (std::uint32_t)dram_src0_noc_xy.x,
                (std::uint32_t)dram_src0_noc_xy.y,
                W, H, C, HW, N, CHW, num_tiles_read, num_tiles_per_core
            }
        );

        tt_metal::SetRuntimeArgs(
            writer_kernel,
            core,
            {
                dst_dram_buffer->address(),
                num_tiles_per_core,
                num_tiles_read
            }
        );
        num_tiles_read += num_tiles_per_core;
    }


    auto override_runtime_args_callback = [
            reader_kernel,
            writer_kernel,
            num_cores,
            num_cores_y
        ]
    (
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_dram_buffer = input_buffers.at(0);
        auto src_dram_noc_xy = src_dram_buffer->noc_coordinates();

        auto dst_dram_buffer = output_buffers.at(0);

        for(uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(reader_kernel, core);
                runtime_args[0] = src_dram_buffer->address();
                runtime_args[1] = uint32_t(src_dram_noc_xy.x);
                runtime_args[2] = uint32_t(src_dram_noc_xy.y);
                SetRuntimeArgs(reader_kernel, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(writer_kernel, core);
                runtime_args[0] = dst_dram_buffer->address();
                SetRuntimeArgs(writer_kernel, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}


}  // namespace tt_metal

}  // namespace tt
