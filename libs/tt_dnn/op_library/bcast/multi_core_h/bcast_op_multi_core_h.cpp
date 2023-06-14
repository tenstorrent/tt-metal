#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

Program bcast_multi_core_h(const Tensor &a, const Tensor &b, Tensor& output, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {
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

    auto compute_and_storage_grid_size = device->compute_and_storage_grid_size();
    uint32_t num_cores_x = compute_and_storage_grid_size.x;
    uint32_t num_cores_y = compute_and_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] = split_work_to_cores(compute_and_storage_grid_size, Ht);

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr and b.device() != nullptr, "Operands to bcast need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to bcast need to be allocated in buffers on device!");

    uint32_t single_tile_size = a.element_size() * TILE_HW;

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_H);
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

	uint32_t src0_cb_index = 0;
	uint32_t num_input_tiles = 2;
	auto cb_src0 = tt_metal::CreateCircularBuffers(
		program,
		device,
		src0_cb_index,
		all_cores,
		num_input_tiles,
		num_input_tiles * single_tile_size,
		DataFormat::Float16_b
	);

	uint32_t src1_cb_index = 1;
	auto cb_src1 = tt_metal::CreateCircularBuffers(
		program,
		device,
		src1_cb_index,
		all_cores,
		num_input_tiles,
		num_input_tiles * single_tile_size,
		DataFormat::Float16_b
	);

	uint32_t ouput_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = 2;
	auto cb_output = tt_metal::CreateCircularBuffers(
		program,
		device,
		ouput_cb_index,
		all_cores,
		num_output_tiles,
		num_output_tiles * single_tile_size,
		DataFormat::Float16_b
	);

	tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
		program,
		reader_name,
		all_cores,
		tt_metal::DataMovementProcessor::RISCV_1,
		tt_metal::NOC::RISCV_1_default);

	tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
		program,
		"tt_metal/kernels/dataflow/writer_unary_8bank_input_cols_batched.cpp",
		all_cores,
		tt_metal::DataMovementProcessor::RISCV_0,
		tt_metal::NOC::RISCV_0_default);


	// TODO(AP): add dimensions and op params
	vector<uint32_t> compute_kernel_args_group_1 = {
		NC, // B
		Ht_per_core_group_1, // Ht
		Wt  // Wt
	};

	bool fp32_dest_acc_en = false;
	bool math_approx_mode = false;
	auto bcast_kernel_group_1 = tt_metal::CreateComputeKernel(
		program,
		compute_name,
		core_group_1,
		compute_kernel_args_group_1,
		MathFidelity::HiFi4,
		fp32_dest_acc_en,
		math_approx_mode
	);
	bcast_op_utils::add_defines(bcast_kernel_group_1, bcast_dim, bcast_math);
	if (!core_group_2.ranges().empty()) {
		// TODO(AP): add dimensions and op params
		vector<uint32_t> compute_kernel_args_group_2 = {
			NC, // B
			Ht_per_core_group_2, // Ht
			Wt  // Wt
		};
		auto bcast_kernel_group_2 = tt_metal::CreateComputeKernel(
			program,
			compute_name,
			core_group_2,
			compute_kernel_args_group_2,
			MathFidelity::HiFi4,
			fp32_dest_acc_en,
			math_approx_mode
		);
		bcast_op_utils::add_defines(bcast_kernel_group_2, bcast_dim, bcast_math);
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

		tt_metal::WriteRuntimeArgsToDevice(
			device,
			binary_reader_kernel,
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

		tt_metal::WriteRuntimeArgsToDevice(
			device, unary_writer_kernel, core,
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

    return program;
}

}  // namespace tt_metal

}  // namespace tt
