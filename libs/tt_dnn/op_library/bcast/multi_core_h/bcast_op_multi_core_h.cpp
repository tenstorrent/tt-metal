#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks bcast_multi_core_h(const Tensor &a, const Tensor &b, Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim) {
    TT_ASSERT(bcast_dim == BcastOpDim::H);

    const auto ashape = a.shape();
    const auto bshape = b.shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

	uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

    tt_metal::Program program = tt_metal::Program();

    tt_metal::Device *device = a.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Ht);

	auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
	TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_H);
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

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

	uint32_t src1_cb_index = 1;
	auto cb_src1 = tt_metal::CreateCircularBuffers(
		program,
		src1_cb_index,
		all_cores,
		num_input_tiles,
		num_input_tiles * single_tile_size,
		cb_data_format
	);

	uint32_t output_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = 2;
	auto cb_output = tt_metal::CreateCircularBuffers(
		program,
		output_cb_index,
		all_cores,
		num_output_tiles,
		num_output_tiles * single_tile_size,
		cb_data_format
	);

	bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

	KernelID binary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
		program,
		reader_name,
		all_cores,
		tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

	KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
		program,
		"tt_metal/kernels/dataflow/writer_unary_8bank_input_cols_batched.cpp",
		all_cores,
		tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

	// TODO(AP): add dimensions and op params
	vector<uint32_t> compute_kernel_args_group_1 = {
		NC, // B
		Ht_per_core_group_1, // Ht
		Wt  // Wt
	};

	std::map<std::string, std::string> bcast_defines = bcast_op_utils::get_defines(bcast_dim, bcast_math);
	auto bcast_kernel_group_1_id = tt_metal::CreateComputeKernel(
		program,
		compute_name,
		core_group_1,
		tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_1, .defines = bcast_defines}
	);
	if (!core_group_2.ranges().empty()) {
		// TODO(AP): add dimensions and op params
		vector<uint32_t> compute_kernel_args_group_2 = {
			NC, // B
			Ht_per_core_group_2, // Ht
			Wt  // Wt
		};
		auto bcast_kernel_group_2_id = tt_metal::CreateComputeKernel(
			program,
			compute_name,
			core_group_2,
			tt_metal::ComputeConfig{.compile_args = compute_kernel_args_group_2, .defines = bcast_defines}
		);
	}

	for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores; i++){
		CoreCoord core = {i / num_cores_y, i % num_cores_y};
		uint32_t Ht_per_core;
		if (core_group_1.core_coord_in_core_ranges(core)) {
			Ht_per_core = Ht_per_core_group_1;
		} else if (core_group_2.core_coord_in_core_ranges(core)) {
			Ht_per_core = Ht_per_core_group_2;
		} else {
			TT_ASSERT(false, "Core not in specified core ranges");
		}
		uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

		tt_metal::SetRuntimeArgs(
			program,
			binary_reader_kernel_id,
			core,
			{
				a.buffer()->address(), // 0
				0, // 1
				0, // 2
				num_tensor_tiles_per_core, // 3
				b.buffer()->address(), // 4
				0, // 5
				0, // 6
				num_btensor_tiles, // 7
				num_tensor_tiles_per_core, // 8
				NC, // 9
				Ht_per_core, // 10
				Wt, // 11
				bnc1, // 12
				num_Wtiles_read, // 13
				Ht*Wt, // 14
			}
		);

		tt_metal::SetRuntimeArgs(
			program, unary_writer_kernel_id, core,
			{
				output.buffer()->address(),
				0,
				0,
				Ht_per_core,
				Wt,
				num_Wtiles_read,
				0,
				NC,
				Ht*Wt,
			}
		);

		num_Wtiles_read += Ht_per_core * Wt;
	}

    auto override_runtime_args_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
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

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto runtime_args = GetRuntimeArgs(program, binary_reader_kernel_id, core);
                runtime_args[0] = src_dram_buffer_a->address();
                runtime_args[4] = src_dram_buffer_b->address();
                SetRuntimeArgs(program, binary_reader_kernel_id, core, runtime_args);
            }

            {
                auto runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_dram_buffer->address();
                SetRuntimeArgs(program, unary_writer_kernel_id, core, runtime_args);
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal

}  // namespace tt
