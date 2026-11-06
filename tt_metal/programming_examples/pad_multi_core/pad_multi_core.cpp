// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tensor/host_tensor.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    // Initialize Mesh API constructs: mesh device, command queue, workload, device range
    int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // initialize source data
    constexpr uint32_t src_M = 64;
    constexpr uint32_t src_N = 32;
    constexpr uint32_t packed_data_size = sizeof(uint32_t);
    constexpr uint32_t unpacked_data_size = sizeof(bfloat16);
    constexpr uint32_t packing_ratio = packed_data_size / unpacked_data_size;
    uint32_t src_num_values_unpacked = src_M * src_N;
    uint32_t src_num_values_packed = src_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> src_vec(src_num_values_packed, 0);
    // source vector = {1, 2, 3, ... , 30, 31, 32,   2048}
    for (uint32_t i = 0; i < src_vec.size(); i++) {
        bfloat16 bfloat_val1 = bfloat16((2 * i) + 1);
        bfloat16 bfloat_val2 = bfloat16((2 * i) + 2);
        src_vec[i] = pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat_val1, bfloat_val2));
    }

    // create pad vector
    bfloat16 pad_value = bfloat16(2);
    std::vector<uint32_t> pad_vec(
        1, pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(pad_value, pad_value)));

    // create destination vector
    constexpr uint32_t dst_M = 64;
    constexpr uint32_t dst_N = 64;
    uint32_t dst_num_values_unpacked = dst_M * dst_N;
    uint32_t dst_num_values_packed = dst_num_values_unpacked / packing_ratio;
    std::vector<uint32_t> dst_vec(dst_num_values_packed, 0);

    // designate cores and core specs
    experimental::NodeCoord start_core = {0, 0};
    experimental::NodeCoord end_core = {0, 3};
    experimental::NodeRange cores(start_core, end_core);
    uint32_t num_cores = cores.size();

    // configure and create DRAM tensors for input, pad, output
    // Each row of a row-major tensor is one page, so a row of one packed uint32 gives a page size of packed_data_size.
    auto dram_tensor_spec = [](uint32_t num_pages) {
        return TensorSpec(
            Shape({num_pages, 1}),
            TensorLayout(
                DataType::UINT32,
                PageConfig(Layout::ROW_MAJOR),
                MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM}));
    };
    TensorSpec src_tensor_spec = dram_tensor_spec(src_num_values_packed);
    MeshTensor src_tensor = MeshTensor::allocate_on_device(*mesh_device, src_tensor_spec);

    TensorSpec pad_tensor_spec = dram_tensor_spec(pad_vec.size());
    MeshTensor pad_tensor = MeshTensor::allocate_on_device(*mesh_device, pad_tensor_spec);

    TensorSpec dst_tensor_spec = dram_tensor_spec(dst_num_values_packed);
    MeshTensor dst_tensor = MeshTensor::allocate_on_device(*mesh_device, dst_tensor_spec);

    // Names identify the kernels, the dataflow buffers and the tensor parameters within the program spec.
    const experimental::KernelSpecName reader_kernel{"pad_reader"};
    const experimental::KernelSpecName writer_kernel{"pad_writer"};
    const experimental::DFBSpecName in0_dfb{"in0"};
    const experimental::DFBSpecName pad_dfb{"pad"};
    const experimental::DFBSpecName pad_align_dfb{"pad_align"};
    const experimental::TensorParamName src_tensor_param{"src"};
    const experimental::TensorParamName dst_tensor_param{"dst"};

    // configure dataflow buffers expected by TTNN reader/writer: in0 (main), pad (pad), pad_align (align)
    tt::DataFormat dfb_df = tt::DataFormat::UInt32;
    const uint32_t stick_size_bytes = packed_data_size;              // 4 bytes per stick (one packed uint32)
    // Use TTNN row-major minimum of 64B per stick in L1 for padding pattern
    const uint32_t stick_size_padded = 64;
    const uint32_t stick_size_padded_aligned = 64;
    const uint32_t num_packed_row_src = src_N / packing_ratio;
    const uint32_t num_packed_row_dst = dst_N / packing_ratio;
    const uint32_t num_sticks_per_barrier = num_packed_row_dst;      // process one row per barrier
    // in0 needs capacity for one row of padded sticks
    // The three dataflow buffers are aliases: they share the same L1 memory of dfb_size bytes.
    // pad and pad_align are only used by the reader, so the reader is both their producer and consumer.
    const uint32_t dfb_size = num_sticks_per_barrier * stick_size_padded_aligned;
    experimental::DataflowBufferSpec in0_dfb_spec{
        .unique_id = in0_dfb,
        .entry_size = stick_size_padded_aligned,
        .num_entries = dfb_size / stick_size_padded_aligned,
        .data_format_metadata = dfb_df,
        .advanced_options = {.alias_with = {pad_dfb, pad_align_dfb}},
    };
    experimental::DataflowBufferSpec pad_dfb_spec{
        .unique_id = pad_dfb,
        .entry_size = stick_size_padded,
        .num_entries = dfb_size / stick_size_padded,
        .data_format_metadata = dfb_df,
        .advanced_options = {.alias_with = {in0_dfb, pad_align_dfb}},
    };
    experimental::DataflowBufferSpec pad_align_dfb_spec{
        .unique_id = pad_align_dfb,
        .entry_size = stick_size_bytes,
        .num_entries = dfb_size / stick_size_bytes,
        .data_format_metadata = dfb_df,
        .advanced_options = {.alias_with = {in0_dfb, pad_dfb}},
    };

    // specify compile time args for TTNN reader/writer
    experimental::KernelSpec::CompileTimeArgs reader_compile_time_args = {
        // N, H, C (treat rows as N, single H=1, columns (packed) as C)
        {"N", src_M},
        {"H", 1},
        {"C", num_packed_row_src},
        {"stick_size_bytes", stick_size_bytes},
        {"N_padded", src_M},  // (same)
        {"H_padded", 1},
        {"C_padded", num_packed_row_dst},
        {"stick_size_padded", stick_size_padded},  // (32B pad pattern)
        {"stick_size_padded_front", 0},            // (left-align)
        {"stick_size_padded_end", 0},
        {"num_zero_pad_sticks_read", 1},
        {"last_zero_stick_size", stick_size_padded},  // (32)
        {"not_pad_by_zero", 1},
        // pass packed pad value directly as compile-time arg as TTNN does
        {"packed_pad_value", pad_vec[0]},
        {"row_major_min_bytes", stick_size_padded},  // (32)
        {"num_front_pad_sticks_read", 0},
        {"num_end_pad_sticks_read", 0},
        {"num_sticks_padded_read", 1},                             // per stick
        {"stick_size_padded_aligned", stick_size_padded_aligned},  // (32)
        {"unaligned", 0},                                          // false
    };

    experimental::KernelSpec::CompileTimeArgs writer_compile_time_args = {
        {"stick_size_bytes", stick_size_bytes}, {"stick_size_padded_aligned", stick_size_padded_aligned}};

    // create kernels (borrowed from TTNN production code)
    experimental::KernelSpec reader_kernel_spec{
        .unique_id = reader_kernel,
        .source = "tt_metal/programming_examples/pad_multi_core/kernels/pad_reader_dims_rm_interleaved.cpp",
        .dfb_bindings =
            {experimental::ProducerOf(in0_dfb, "in0"),
             experimental::ProducerOf(pad_dfb, "pad"),
             experimental::ConsumerOf(pad_dfb, "pad"),
             experimental::ProducerOf(pad_align_dfb, "pad_align"),
             experimental::ConsumerOf(pad_align_dfb, "pad_align")},
        .tensor_bindings = {{.tensor_parameter_name = src_tensor_param, .accessor_name = "src"}},
        .compile_time_args = reader_compile_time_args,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core",
                  "num_sticks_per_barrier",
                  "start_id",
                  "front_pad_n",
                  "front_pad_c",
                  "front_pad_h"},
             .runtime_array_args = {{.name = "start_dim_offset", .size = 4}}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default},
            },
    };
    experimental::KernelSpec writer_kernel_spec{
        .unique_id = writer_kernel,
        .source = "tt_metal/programming_examples/pad_multi_core/kernels/pad_writer_dims_rm_interleaved.cpp",
        .dfb_bindings = {experimental::ConsumerOf(in0_dfb, "out0")},
        .tensor_bindings = {{.tensor_parameter_name = dst_tensor_param, .accessor_name = "dst"}},
        .compile_time_args = writer_compile_time_args,
        .runtime_arg_schema = {.runtime_arg_names = {"num_sticks_per_core", "num_sticks_per_barrier", "start_id"}},
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default},
            },
    };

    experimental::ProgramSpec program_spec{
        .name = "pad_multi_core",
        .kernels = {reader_kernel_spec, writer_kernel_spec},
        .dataflow_buffers = {in0_dfb_spec, pad_dfb_spec, pad_align_dfb_spec},
        .tensor_parameters =
            {{.unique_id = src_tensor_param, .spec = src_tensor_spec},
             {.unique_id = dst_tensor_param, .spec = dst_tensor_spec}},
        .work_units = {{.name = "pad_multi_core", .kernels = {reader_kernel, writer_kernel}, .target_nodes = cores}},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, program_spec);

    // set kernel runtime arguments
    uint32_t start_src_idx = 0;
    uint32_t start_dst_idx = 0;
    uint32_t num_rows_per_core = src_M / num_cores;
    uint32_t num_src_sticks_per_core = num_packed_row_src * num_rows_per_core;
    experimental::ProgramRunArgs::KernelRunArgs reader_run_args{.kernel = reader_kernel};
    experimental::ProgramRunArgs::KernelRunArgs writer_run_args{.kernel = writer_kernel};
    for (uint32_t core_idx = 0; core_idx < num_cores; core_idx++) {
        experimental::NodeCoord core = {0, core_idx};
        uint32_t num_sticks_per_core = num_rows_per_core * num_packed_row_dst;
        uint32_t num_sticks_per_barrier_rt = num_packed_row_dst;  // one row per barrier
        // Reader runtime: num_sticks_per_core, num_sticks_per_barrier, start_id, front_pad_n,c,h,
        // start_dim_offset[0..3] (the src address comes from the src tensor binding)
        experimental::AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core", num_sticks_per_core},
             {"num_sticks_per_barrier", num_sticks_per_barrier_rt},
             {"start_id", start_src_idx},
             {"front_pad_n", 0},
             {"front_pad_c", 0},
             {"front_pad_h", 0}});
        reader_run_args.runtime_array_arg_values["start_dim_offset"][core] = {0, 0, 0, 0};
        // Writer runtime: num_sticks_per_core, num_sticks_per_barrier, start_id
        // (the dst address comes from the dst tensor binding)
        experimental::AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_sticks_per_core", num_sticks_per_core},
             {"num_sticks_per_barrier", num_sticks_per_barrier_rt},
             {"start_id", start_dst_idx}});
        start_src_idx += num_src_sticks_per_core;
        start_dst_idx += num_packed_row_dst * num_rows_per_core;
    }
    experimental::ProgramRunArgs run_args{
        .kernel_run_args = {reader_run_args, writer_run_args},
        .tensor_args =
            {{src_tensor_param, experimental::TensorArgument{src_tensor}},
             {dst_tensor_param, experimental::TensorArgument{dst_tensor}}},
    };
    experimental::SetProgramRunArgs(program, run_args);

    printf(
        "Padding tensor of shape (%d, %d) to shape (%d, %d) with pad value: %d\n",
        src_M,
        src_N,
        dst_M,
        dst_N,
        std::bit_cast<uint16_t>(pad_value));
    printf("Original tensor with shape (%d, %d):\n", src_M, src_N);
    for (uint32_t m = 0; m < src_M; m++) {
        for (uint32_t n = 0; n < num_packed_row_src; n++) {
            printf("%d ", (uint16_t)src_vec[(m * num_packed_row_src) + n]);
            printf("%d ", (uint16_t)(src_vec[(m * num_packed_row_src) + n] >> 16));
        }
        printf("\n");
    }
    printf("\n");

    // Upload inputs (non-blocking), enqueue mesh workload (non-blocking), read back result, then wait for completion
    HostTensor src_host_tensor = HostTensor::from_vector(src_vec, src_tensor_spec);
    cq.enqueue_write_tensor(src_host_tensor, src_tensor);
    HostTensor pad_host_tensor = HostTensor::from_vector(pad_vec, pad_tensor_spec);
    cq.enqueue_write_tensor(pad_host_tensor, pad_tensor);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    dst_vec = cq.enqueue_read_tensor(dst_tensor, /*blocking=*/true).to_vector<uint32_t>();
    distributed::Finish(cq);

    printf("Padded tensor with shape (%d, %d):\n", dst_M, dst_N);
    for (uint32_t m = 0; m < dst_M; m++) {
        for (uint32_t n = 0; n < num_packed_row_dst; n++) {
            printf("%d ", (uint16_t)dst_vec[(m * num_packed_row_dst) + n]);
            printf("%d ", (uint16_t)(dst_vec[(m * num_packed_row_dst) + n] >> 16));
        }
        printf("\n");
    }

    mesh_device->close();
}
