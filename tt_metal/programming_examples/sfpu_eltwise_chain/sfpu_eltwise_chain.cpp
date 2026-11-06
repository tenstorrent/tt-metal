// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include <cmath>
#include <random>
#include <cstdint>
#include <vector>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

/**
 * @brief Computes the softplus activation function element-wise on input vector
 *
 * The softplus function is defined as: softplus(x) = log(1 + exp(x))
 * This is a smooth approximation to the ReLU function that outputs positive values.
 *
 * @param src_vec Input vector containing bfloat16 values to apply softplus to
 * @param result_vec Output vector where computed softplus values will be stored
 *                   Must be the same size as src_vec
 *
 * @throws TT_FATAL if input and output vectors have different sizes
 *
 * @note This is a reference implementation used for validation/testing purposes
 * @note The function uses std::log1p for numerical stability
 */
void golden_softplus(const std::vector<bfloat16>& src_vec, std::vector<bfloat16>& result_vec) {
    TT_FATAL(src_vec.size() == result_vec.size(), "Input and output vectors must be the same size");
    for (size_t i = 0; i < src_vec.size(); ++i) {
        result_vec[i] = bfloat16(std::log1p(std::exp(static_cast<float>(src_vec[i]))));  // Softplus function
    }
}

/**
 * @brief Calculates the Pearson Correlation Coefficient (PCC) between two bfloat16 vectors.
 *
 * This function computes the linear correlation coefficient between two vectors of bfloat16 values,
 * which measures the strength and direction of the linear relationship between the two datasets.
 * The PCC value ranges from -1 to 1, where:
 * - 1 indicates a perfect positive linear relationship
 * - 0 indicates no linear relationship
 * - -1 indicates a perfect negative linear relationship
 *
 * @param vec_a First input vector of bfloat16 values
 * @param vec_b Second input vector of bfloat16 values (must be same size as vec_a)
 *
 * @return float The Pearson correlation coefficient between the two input vectors
 *
 * @note The function assumes both input vectors have the same size
 * @note bfloat16 values are converted to float for calculations to maintain precision
 */
inline float check_bfloat16_vector_pcc(const std::vector<bfloat16>& vec_a, const std::vector<bfloat16>& vec_b) {
    // Calculate the mean of x and y values
    float x_mean = 0.0f;
    float y_mean = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        x_mean += static_cast<float>(vec_a[i]);
        y_mean += static_cast<float>(vec_b[i]);
    }

    x_mean /= vec_a.size();
    y_mean /= vec_b.size();

    // Calculate the covariance and standard deviation of x and y values
    float covariance = 0.0f;
    float x_stddev = 0.0f;
    float y_stddev = 0.0f;

    for (size_t i = 0; i < vec_a.size(); i++) {
        float x_diff = static_cast<float>(vec_a[i]) - x_mean;
        float y_diff = static_cast<float>(vec_b[i]) - y_mean;

        covariance += x_diff * y_diff;
        x_stddev += x_diff * x_diff;
        y_stddev += y_diff * y_diff;
    }

    covariance /= vec_a.size();
    x_stddev /= vec_a.size();
    y_stddev /= vec_b.size();

    // Calculate the correlation coefficient
    float correlation_coefficient_ = covariance / (std::sqrt(x_stddev) * std::sqrt(y_stddev));
    return correlation_coefficient_;
}

int main() {
    // Device setup
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(0);

    // Device command queue and program setup
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // Core range setup
    constexpr experimental::NodeCoord core = {0, 0};

    // Input data preparation
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.0f);

    // Fill the source vector with random values
    std::vector<bfloat16> src_vec(constants::TILE_HW);
    for (bfloat16& v : src_vec) {
        v = bfloat16(dist(rng));
    }

    // Calculate golden function results on CPU
    std::vector<bfloat16> golden_vec(constants::TILE_HW, 0);
    golden_softplus(src_vec, golden_vec);

    // Tilize the input vectors to match the expected tiled layout for the device
    // The Tenstorrent hardware operates on data in 32x32 tiles rather than standard row-major format.
    // tilize_nfaces() converts the input matrices from row-major layout to the tiled layout expected by the device.
    // This transformation groups elements into 32x32 blocks and reorders them in memory so that each tile
    // (32x32 elements) is stored contiguously. This matches the native data access patterns of the matrix engine
    // and enables efficient operations on the accelerator.
    src_vec = tilize_nfaces(src_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Dram tensor config: one row-major row of one tile's worth of elements, so the page size is one tile
    constexpr uint32_t single_tile_size = sizeof(bfloat16) * constants::TILE_HEIGHT * constants::TILE_WIDTH;
    TensorSpec dram_tensor_spec(
        Shape({1, constants::TILE_HW}),
        TensorLayout(
            DataType::BFLOAT16,
            PageConfig(Layout::ROW_MAJOR),
            MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    MeshTensor src_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);  // Input tensor
    MeshTensor dst_dram_tensor = MeshTensor::allocate_on_device(*mesh_device, dram_tensor_spec);  // Output tensor

    // DRAM transfer
    HostTensor src_host_tensor = HostTensor::from_vector(src_vec, dram_tensor_spec);
    cq.enqueue_write_tensor(src_host_tensor, src_dram_tensor);

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader_kernel{"reader"};
    const experimental::KernelSpecName writer_kernel{"writer"};
    const experimental::KernelSpecName compute_kernel{"compute"};
    const experimental::DFBSpecName src_dfb{"src"};
    const experimental::DFBSpecName ones_dfb{"ones"};
    const experimental::DFBSpecName result_dfb{"result"};
    const experimental::TensorParamName src_tensor_param{"src"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    // L1 dataflow buffer setup
    experimental::DataflowBufferSpec src_dfb_spec{
        .unique_id = src_dfb,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::DataflowBufferSpec ones_dfb_spec{
        .unique_id = ones_dfb,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    experimental::DataflowBufferSpec result_dfb_spec{
        .unique_id = result_dfb,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = tt::DataFormat::Float16_b,
    };

    // Kernels setup
    // Data movement kernels
    experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/reader.cpp",
        .dfb_bindings = {experimental::ProducerOf(src_dfb, "src"), experimental::ProducerOf(ones_dfb, "ones")},
        .tensor_bindings = {{.tensor_parameter_name = src_tensor_param, .accessor_name = "input"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_0},
            },
    };
    experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/dataflow/writer.cpp",
        .dfb_bindings = {experimental::ConsumerOf(result_dfb, "result")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "output"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_1},
            },
    };

    // Compute kernel
    experimental::KernelSpec compute_kernel_spec{
        .unique_id = compute_kernel,
        .source = OVERRIDE_KERNEL_PREFIX "sfpu_eltwise_chain/kernels/compute/compute.cpp",
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {experimental::ConsumerOf(src_dfb, "src"),
             experimental::ConsumerOf(ones_dfb, "ones"),
             experimental::ProducerOf(result_dfb, "result")},
        .hw_config =
            experimental::ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = Precision::Precise,
            },
    };

    // Program setup
    experimental::ProgramSpec program_spec{
        .name = "sfpu_eltwise_chain",
        .kernels = {reader_kernel_spec, writer_kernel_spec, compute_kernel_spec},
        .dataflow_buffers = {src_dfb_spec, ones_dfb_spec, result_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src_tensor_param, .spec = dram_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dram_tensor_spec}},
        .work_units =
            {{.name = "sfpu_eltwise_chain",
              .kernels = {reader_kernel, writer_kernel, compute_kernel},
              .target_nodes = core}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // Runtime args setup: the kernels only take the tensors bound to the tensor parameters
    experimental::ProgramRunArgs run_args{
        .tensor_args =
            {{src_tensor_param, experimental::TensorArgument{src_dram_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_dram_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    // Program enqueue
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // Data transfer back to host machine
    std::vector<bfloat16> result_vec = cq.enqueue_read_tensor(dst_dram_tensor, /*blocking=*/true).to_vector<bfloat16>();

    // Reverse the tilization to get the result in the row-major format that the CPU expects
    result_vec = untilize_nfaces(result_vec, constants::TILE_WIDTH, constants::TILE_HEIGHT);

    // Calculate the Pearson correlation coefficient (PCC) between the golden vector and the result vector
    // This is a measure of how similar the two vectors are.
    // A PCC close to 1 indicates that the two vectors are very similar.
    const float pearson = check_bfloat16_vector_pcc(golden_vec, result_vec);
    fmt::print("Metalium vs Golden -- PCC = {}\n", pearson);
    TT_FATAL(pearson > 0.999, "PCC not high enough. Result PCC: {}, Expected PCC: 0.999", pearson);

    mesh_device->close();
}
