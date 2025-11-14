// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mean_all_cores_program_factory.hpp"

#include <cstdint>
#include <enchantum/enchantum.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "fmt/base.h"
#include "mean_all_cores_device_operation_types.hpp"
#include "metal/ops/common/program_utils.hpp"

namespace {

constexpr auto kWorkerReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/examples/mean_all_cores/device/kernels/dataflow/"
    "worker_reader.cpp";

constexpr auto kWorkerWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/examples/mean_all_cores/device/kernels/dataflow/"
    "worker_writer.cpp";

constexpr auto kWorkerComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/examples/mean_all_cores/device/kernels/compute/worker_compute_kernel.cpp";

constexpr auto kReduceReaderKernelPath =
    "tt-train/sources/ttml/metal/ops/examples/mean_all_cores/device/kernels/dataflow/"
    "reduce_reader.cpp";

constexpr auto kReduceWriterKernelPath =
    "tt-train/sources/ttml/metal/ops/examples/mean_all_cores/device/kernels/dataflow/"
    "reduce_writer.cpp";

constexpr auto kReduceComputeKernelPath =
    "tt-train/sources/ttml/metal/ops/examples/mean_all_cores/device/kernels/compute/reduce_compute_kernel.cpp";

// Circular buffer indices
constexpr auto kInputCbIndex = tt::CBIndex::c_0;
constexpr auto kReductionScalerCbIndex = tt::CBIndex::c_1;
constexpr auto kIntermOutputCbIndex = tt::CBIndex::c_2;
constexpr auto kTransferBuffer_01 = tt::CBIndex::c_3;
constexpr auto kTransferBuffer_02 = tt::CBIndex::c_4;
constexpr auto kOutputCbIndex = tt::CBIndex::c_5;

constexpr uint32_t kSingleTileBuffer = 1U;

}  // namespace

namespace ttml::metal::ops::examples::mean_all_cores::device {

/**
 *   Helper struct to hold references to all kernels we create,
 *        used during runtime argument setup.
 */
struct MeanAllCoresKernels {
    tt::tt_metal::KernelHandle worker_reader;
    tt::tt_metal::KernelHandle worker_writer;
    tt::tt_metal::KernelHandle worker_compute;
    tt::tt_metal::KernelHandle reduce_reader;
    tt::tt_metal::KernelHandle reduce_writer;
    tt::tt_metal::KernelHandle reduce_compute;
};

void assign_worker_core_runtime_args(
    tt::tt_metal::Program& program,
    const MeanAllCoresKernels& kernels,
    const tt::tt_metal::Buffer* input_buffer,
    uint32_t num_worker_cores,
    uint32_t num_rows_per_worker_core,
    const tt::tt_metal::CoreRangeSet& worker_cores,
    const tt::tt_metal::CoreCoord& reduction_core_physical_coord,
    const uint32_t semaphore) {
    for (uint32_t i = 0, num_rows_written = 0; i < num_worker_cores; i++) {
        CoreCoord core = {0, i};
        fmt::print("Assigning runtime args for worker core at coord: ({}, {})\n", core.x, core.y);
        fmt::print(
            "  num_rows_per_worker_core: {}, num_rows_written: {}\n",
            num_rows_per_worker_core,
            num_rows_written);
        // Reader kernel: (input_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.worker_reader,
            core,
            {input_buffer->address(), num_rows_per_worker_core, num_rows_written});

        // Writer kernel: (dst_addr, number_of_rows, offset_in_rows)
        SetRuntimeArgs(
            program,
            kernels.worker_writer,
            core,
            {num_rows_per_worker_core,
             num_rows_written,
             static_cast<uint32_t>(reduction_core_physical_coord.x),
             static_cast<uint32_t>(reduction_core_physical_coord.y),
             semaphore});

        num_rows_written += num_rows_per_worker_core;
    }
}

void assign_reduction_core_runtime_args(
    tt::tt_metal::Program& program,
    const MeanAllCoresKernels& kernels,
    const tt::tt_metal::Buffer* output_buffer,
    uint32_t num_worker_cores,
    const std::vector<CoreCoord>& worker_cores_physical_coords,
    const tt::tt_metal::CoreRangeSet& reduction_cores,
    const uint32_t semaphore) {
    TT_FATAL(reduction_cores.size() == 1, "Only one reduction core is supported");
    CoreCoord core = {0, 2U};  // only one reduction core at (0,2)
    TT_FATAL(reduction_cores.contains(core), "Reduction core coordinate mismatch");
    fmt::print("Assigning runtime args for reduction core at coord: ({}, {})\n", core.x, core.y);
    // Reader kernel: (number_of_rows, worker0_core_x, worker0_core_y, worker1_core_x, worker1_core_y,
    // semaphore_addr)
    SetRuntimeArgs(
        program,
        kernels.reduce_reader,
        core,
        {num_worker_cores,
         static_cast<uint32_t>(worker_cores_physical_coords[0].x),
         static_cast<uint32_t>(worker_cores_physical_coords[0].y),
         static_cast<uint32_t>(worker_cores_physical_coords[1].x),
         static_cast<uint32_t>(worker_cores_physical_coords[1].y),
         semaphore});

    // Writer kernel: (output_addr)
    SetRuntimeArgs(program, kernels.reduce_writer, core, {output_buffer->address()});
}

MeanAllCoresProgramFactory::cached_program_t MeanAllCoresProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;
    auto* device = input.device();

    tt::tt_metal::Program program{};

    tt::DataFormat input_data_format = datatype_to_dataformat_converter(input.dtype());
    TT_FATAL(input_data_format == tt::DataFormat::Float16_b, "Input data format must be Float16_b");

    uint32_t bfloat16_single_tile_size_bytes = tt::tile_size(tt::DataFormat::Float16_b);

    auto input_shape = input.logical_shape();
    auto padded_tensor_shape = input.padded_shape();
    uint32_t Wt = padded_tensor_shape[-1] / tt::constants::TILE_WIDTH;  // <- number of tiles in inner dimension
    uint32_t Ht = padded_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    uint32_t NC = padded_tensor_shape[0] * padded_tensor_shape[1];
    uint32_t total_rows_to_process = NC * Ht;

    // -------------------------------------------------------------------------
    // 2) Determine core grid
    // -------------------------------------------------------------------------
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // uint32_t num_cores_x = compute_with_storage_grid_size.x;
    // uint32_t num_cores_y = compute_with_storage_grid_size.y;

    // uint32_t num_worker_cores = (num_cores_x - 1) * num_cores_y;  // Reserve one column for reduction cores
    // uint32_t num_reduction_cores = num_cores_y;

    /*
     * Right now we will set up example to use semaphores for first time.
     * we will use 3 cores, two worker cores and one reduction core.
     */
    uint32_t num_cores = 3U;
    uint32_t num_worker_cores = 2U;
    uint32_t num_reduction_cores = 1U;
    TT_FATAL(
        total_rows_to_process % num_worker_cores == 0,
        "Total rows to process must be divisible by number of worker cores");
    uint32_t num_rows_per_worker_core = total_rows_to_process / num_worker_cores;
    fmt::print("num_rows_per_worker_core: {}\n", num_rows_per_worker_core);

    // CoreRangeSet all_cores = num_cores_to_corerangeset({num_cores_x, num_cores_y}, {num_cores_x, num_cores_y});
    auto all_cores = CoreRangeSet(CoreRange({0, 0}, {0, 2}));  // use first 3 cores only
    fmt::print("all_cores: {}\n", all_cores.num_cores());

    auto worker_cores = CoreRangeSet(CoreRange({0, 0}, {0, 1}));  // first two columns
    fmt::print("worker_cores: {}\n", worker_cores.num_cores());

    // calculate physical coords of worker cores
    auto worker_cores_coord = corerange_to_cores(worker_cores);
    std::vector<CoreCoord> worker_cores_physical_coords;
    worker_cores_physical_coords.reserve(worker_cores_coord.size());
    fmt::print("worker_cores_coord size: {}\n", worker_cores_coord.size());
    for (const auto& logical_coord : worker_cores_coord) {
        worker_cores_physical_coords.emplace_back(device->worker_core_from_logical_core(logical_coord));
    }

    auto reduction_core_coords = CoreCoord({0, 2});
    auto reduction_cores = CoreRangeSet(CoreRange(reduction_core_coords));  // last column
    fmt::print("reduction_cores: {}\n", reduction_cores.num_cores());

    // calculate physical coords of reduction cores
    const auto reduction_core_physical_coord = device->worker_core_from_logical_core(reduction_core_coords);

    const uint32_t sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    // -------------------------------------------------------------------------
    // 3) Create circular buffers
    // -------------------------------------------------------------------------

    [[maybe_unused]] auto cb_input = create_circular_buffer(
        program, worker_cores, kInputCbIndex, input_data_format, bfloat16_single_tile_size_bytes, Wt);

    [[maybe_unused]] auto cb_reduction_scaler = create_circular_buffer(
        program,
        worker_cores,
        kReductionScalerCbIndex,
        input_data_format,
        bfloat16_single_tile_size_bytes,
        kSingleTileBuffer);

    [[maybe_unused]] auto cb_interm_output = create_circular_buffer(
        program,
        worker_cores,
        kIntermOutputCbIndex,
        input_data_format,
        bfloat16_single_tile_size_bytes,
        kSingleTileBuffer);

    [[maybe_unused]] auto cb_transfer_buffer = create_circular_buffer(
        program, all_cores, kTransferBuffer_01, input_data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_transfer_buffer_2 = create_circular_buffer(
        program, all_cores, kTransferBuffer_02, input_data_format, bfloat16_single_tile_size_bytes, kSingleTileBuffer);

    [[maybe_unused]] auto cb_output = create_circular_buffer(
        program,
        reduction_cores,
        kOutputCbIndex,
        input_data_format,
        bfloat16_single_tile_size_bytes,
        kSingleTileBuffer);

    // -------------------------------------------------------------------------
    // 4) Create kernels
    // -------------------------------------------------------------------------
    auto* input_buffer = input.buffer();
    auto* output_buffer = output.buffer();

    std::map<std::string, std::string> defines;
    defines["REDUCE_OP"] = "PoolType::SUM";
    defines["REDUCE_DIM"] = "ReduceDim::REDUCE_ROW";

    MeanAllCoresKernels kernels;
    {
        std::vector<uint32_t> reader_compile_time_args = {Wt};
        tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(reader_compile_time_args);
        kernels.worker_reader =
            create_reader_kernel(program, worker_cores, reader_compile_time_args, defines, kWorkerReaderKernelPath);
    }

    {
        std::vector<uint32_t> writer_compile_time_args = {};
        kernels.worker_writer =
            create_writer_kernel(program, worker_cores, writer_compile_time_args, defines, kWorkerWriterKernelPath);
    }

    {
        std::vector<uint32_t> compute_compile_time_args = {num_rows_per_worker_core, Wt};
        kernels.worker_compute = create_compute_kernel(
            program,
            worker_cores,
            compute_compile_time_args,
            defines,
            kWorkerComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    }

    {
        std::vector<uint32_t> reader_compile_time_args = {};
        kernels.reduce_reader =
            create_reader_kernel(program, reduction_cores, reader_compile_time_args, defines, kReduceReaderKernelPath);
    }

    {
        std::vector<uint32_t> writer_compile_time_args = {};
        tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
        kernels.reduce_writer =
            create_writer_kernel(program, reduction_cores, writer_compile_time_args, defines, kReduceWriterKernelPath);
    }

    {
        std::vector<uint32_t> compute_compile_time_args = {num_worker_cores};
        kernels.reduce_compute = create_compute_kernel(
            program,
            reduction_cores,
            compute_compile_time_args,
            defines,
            kReduceComputeKernelPath,
            /*fp32_dest_acc_en=*/true);
    }
    // -------------------------------------------------------------------------
    // 5) Set runtime arguments
    // -------------------------------------------------------------------------
    assign_worker_core_runtime_args(
        program,
        kernels,
        input_buffer,
        num_worker_cores,
        num_rows_per_worker_core,
        worker_cores,
        reduction_core_physical_coord,
        sem_id);

    assign_reduction_core_runtime_args(
        program, kernels, output_buffer, num_worker_cores, worker_cores_physical_coords, reduction_cores, sem_id);

    // -------------------------------------------------------------------------
    // 6) Return cached program
    // -------------------------------------------------------------------------
    return {std::move(program), {.all_cores = all_cores, .num_cores = num_cores}};
}

void MeanAllCoresProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& input = tensor_args.input;
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;

    auto input_buffer = input.buffer();
    auto output_buffer = output.buffer();

    uint32_t num_cores = shared_vars.num_cores;
    auto* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
}

}  // namespace ttml::metal::ops::examples::mean_all_cores::device
