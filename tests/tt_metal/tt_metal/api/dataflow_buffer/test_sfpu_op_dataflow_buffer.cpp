// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// SfpuOp (api/compute/experimental/sfpu.h) end to end through DataflowBuffers, Quasar only:
//   DM -> dfb_in -> compute kernel -> dfb_out -> DM, one bfloat16 tile per entry.
// One test per route: on Dest, on Dest after an FPU op, on SrcS, an FPU op on Dest feeding SrcS, and a
// two-input SrcS op with one input from Dest and one from L1.

#include <cmath>
#include <string>
#include <vector>

#include "dfb_test_common.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"

namespace tt::tt_metal {

namespace {

enum class SfpuOpRoute {
    Dest,                // unpack, math and pack own the DFB ends; the SFPU op runs on Dest
    Srcs,                // trisc3 owns both DFB ends; the SFPU op runs on SrcS
    FpuDestThenSrcs,     // unpack / pack own in and the producer end of mid; trisc3 owns the rest
    FpuDestAndL1ToSrcs,  // FpuDestThenSrcs plus a second input, in1, that trisc3 reads straight from L1
};

struct SfpuOpDfbCase {
    std::string kernel;
    SfpuOpRoute route;
    float (*expected)(float in, float in1);  // in1 is 0 on the routes without a second input
    uint32_t ring_entries;                   // capacity of every DFB
    uint32_t total_entries;                  // tiles sent through
};

float exp_of_in(float in, float /*in1*/) { return std::exp(in); }
float exp_of_2in(float in, float /*in1*/) { return std::exp(2.0f * in); }
float two_in_plus_in1(float in, float in1) { return 2.0f * in + in1; }

void run_sfpu_op_dfb_case(distributed::MeshDevice& mesh_device, const SfpuOpDfbCase& test_case) {
    constexpr uint32_t entry_size = 2 * 32 * 32;  // bf16 tile = 2048 B
    const uint32_t total_entries = test_case.total_entries;
    const bool has_in1 = test_case.route == SfpuOpRoute::FpuDestAndL1ToSrcs;
    const NodeCoord node{0, 0};

    const auto tensor_spec = make_flat_dram_tensor_spec(entry_size, total_entries, DataType::BFLOAT16);
    auto in_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto in1_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);
    auto out_tensor = MeshTensor::allocate_on_device(mesh_device, tensor_spec);

    const DFBSpecName DFB_IN{"dfb_in"};
    const DFBSpecName DFB_IN1{"dfb_in1"};
    const DFBSpecName DFB_MID{"dfb_mid"};
    const DFBSpecName DFB_OUT{"dfb_out"};
    const KernelSpecName PRODUCER{"producer"};
    const KernelSpecName PRODUCER1{"producer1"};
    const KernelSpecName CONSUMER{"consumer"};
    const KernelSpecName COMPUTE{"compute"};
    const TensorParamName IN_TENSOR{"in_tensor"};
    const TensorParamName IN1_TENSOR{"in1_tensor"};
    const TensorParamName OUT_TENSOR{"out_tensor"};

    auto make_spec = [&](const DFBSpecName& name) {
        return DataflowBufferSpec{
            .unique_id = name,
            .entry_size = entry_size,
            .num_entries = test_case.ring_entries,
            .data_format_metadata = tt::DataFormat::Float16_b,
        };
    };
    auto binding = [](const DFBSpecName& dfb, const char* accessor, DFBEndpointType endpoint, bool use_trisc3) {
        return KernelSpec::DFBBinding{
            .dfb_spec_name = dfb,
            .accessor_name = accessor,
            .endpoint_type = endpoint,
            .access_pattern = DFBAccessPattern::STRIDED,
            .use_trisc3 = use_trisc3,
        };
    };
    constexpr auto CONSUME = DFBEndpointType::CONSUMER;
    constexpr auto PRODUCE = DFBEndpointType::PRODUCER;

    auto producer = make_dm_dfb_producer(PRODUCER, DFB_IN, IN_TENSOR, total_entries, /*implicit_sync=*/false);
    auto producer1 = make_dm_dfb_producer(PRODUCER1, DFB_IN1, IN1_TENSOR, total_entries, /*implicit_sync=*/false);
    auto consumer = make_dm_dfb_consumer(
        CONSUMER, DFB_OUT, OUT_TENSOR, total_entries, /*blocked_consumer=*/false, /*implicit_sync=*/false);
    disable_implicit_sync_for(producer, DFB_IN);
    disable_implicit_sync_for(producer1, DFB_IN1);
    disable_implicit_sync_for(consumer, DFB_OUT);

    auto compute = make_compute_kernel(COMPUTE, test_case.kernel);
    std::vector<DataflowBufferSpec> dataflow_buffers = {make_spec(DFB_IN), make_spec(DFB_OUT)};
    switch (test_case.route) {
        case SfpuOpRoute::Dest:
            compute.dfb_bindings = {
                binding(DFB_IN, "in", CONSUME, /*use_trisc3=*/false),
                binding(DFB_OUT, "out", PRODUCE, /*use_trisc3=*/false),
            };
            break;
        case SfpuOpRoute::Srcs:
            compute.dfb_bindings = {
                binding(DFB_IN, "in", CONSUME, /*use_trisc3=*/true),
                binding(DFB_OUT, "out", PRODUCE, /*use_trisc3=*/true),
            };
            break;
        case SfpuOpRoute::FpuDestThenSrcs:
        case SfpuOpRoute::FpuDestAndL1ToSrcs:
            compute.dfb_bindings = {
                binding(DFB_IN, "in", CONSUME, /*use_trisc3=*/false),
                binding(DFB_MID, "mid", PRODUCE, /*use_trisc3=*/false),
                binding(DFB_MID, "mid", CONSUME, /*use_trisc3=*/true),
                binding(DFB_OUT, "out", PRODUCE, /*use_trisc3=*/true),
            };
            dataflow_buffers.push_back(make_spec(DFB_MID));
            if (has_in1) {
                compute.dfb_bindings.push_back(binding(DFB_IN1, "in1", CONSUME, /*use_trisc3=*/true));
                dataflow_buffers.push_back(make_spec(DFB_IN1));
            }
            break;
    }
    // Every kernel here reads a subset of these.
    compute.compile_time_args = {
        {"num_entries", total_entries},
        {"data_format", static_cast<uint32_t>(tt::DataFormat::Float16_b)},
        {"per_core_block_cnt", total_entries},
        {"per_core_block_size", 1u},
    };

    WorkUnitSpec wu{.name = "wu", .kernels = {PRODUCER, CONSUMER, COMPUTE}, .target_nodes = node};
    ProgramSpec spec{
        .name = "sfpu_op_dfb",
        .kernels = {producer, consumer, compute},
        .dataflow_buffers = dataflow_buffers,
        .tensor_parameters =
            {{.unique_id = IN_TENSOR, .spec = in_tensor.tensor_spec()},
             {.unique_id = OUT_TENSOR, .spec = out_tensor.tensor_spec()}},
    };
    if (has_in1) {
        wu.kernels.push_back(PRODUCER1);
        spec.kernels.push_back(producer1);
        spec.tensor_parameters.push_back({.unique_id = IN1_TENSOR, .spec = in1_tensor.tensor_spec()});
    }
    spec.work_units = {wu};
    Program program(mesh_device, spec);

    const auto dm_run_args = [&](const KernelSpecName& kernel) {
        return ProgramRunArgs::KernelRunArgs{
            .kernel = kernel,
            .runtime_arg_values =
                MakeRuntimeArgsForSingleNode(node, {{"chunk_offset", 0u}, {"entries_per_core", total_entries}})};
    };
    ProgramRunArgs params;
    params.kernel_run_args = {
        dm_run_args(PRODUCER), dm_run_args(CONSUMER), ProgramRunArgs::KernelRunArgs{.kernel = COMPUTE}};
    params.tensor_args = {{IN_TENSOR, std::cref(in_tensor)}, {OUT_TENSOR, std::cref(out_tensor)}};
    if (has_in1) {
        params.kernel_run_args.push_back(dm_run_args(PRODUCER1));
        params.tensor_args.emplace(IN1_TENSOR, std::cref(in1_tensor));
    }
    program.set_program_run_args(params);

    // Inputs in [0, 1), so every expected value stays well inside bfloat16 range.
    auto input = create_random_vector_of_bfloat16(entry_size * total_entries, 1.0f, 0x5F9E);
    auto input1 = create_random_vector_of_bfloat16(entry_size * total_entries, 1.0f, 0x1D2B);
    slow_dispatch::WriteToBuffer(in_tensor.mesh_buffer(), input);
    m2_writeshard_barrier_uint32(mesh_device, in_tensor, input);
    if (has_in1) {
        slow_dispatch::WriteToBuffer(in1_tensor.mesh_buffer(), input1);
        m2_writeshard_barrier_uint32(mesh_device, in1_tensor, input1);
    }

    LaunchProgram(mesh_device, std::move(program));

    std::vector<uint32_t> output;
    slow_dispatch::ReadFromBuffer(out_tensor.mesh_buffer(), output);

    // Same tolerance as the bf16 exponential check in test_sfpu_compute.cpp.
    const auto in_values = unpack_uint32_vec_into_bfloat16_vec(input);
    const auto in1_values = unpack_uint32_vec_into_bfloat16_vec(input1);
    const auto out_values = unpack_uint32_vec_into_bfloat16_vec(output);
    ASSERT_EQ(out_values.size(), in_values.size());
    constexpr size_t elements_per_tile = 32 * 32;
    size_t mismatches = 0;
    for (size_t i = 0; i < in_values.size(); i++) {
        const float in_value = static_cast<float>(in_values[i]);
        const float in1_value = has_in1 ? static_cast<float>(in1_values[i]) : 0.0f;
        const float out_value = static_cast<float>(out_values[i]);
        const float expected = test_case.expected(in_value, in1_value);
        if (!is_close(out_value, expected, 0.05f, 0.05f)) {
            if (mismatches++ < 8) {
                ADD_FAILURE() << "tile " << i / elements_per_tile << " element " << i % elements_per_tile << ": in "
                              << in_value << " in1 " << in1_value << " out " << out_value << " expected " << expected;
            }
        }
    }
    EXPECT_EQ(mismatches, 0u) << test_case.kernel;
}

}  // namespace

#define SFPU_OP_DFB_TEST(name, kernel_file, op_route, expected_fn, ring, total)        \
    TEST_F(UnitMeshFixture, name) {                                                    \
        if (this->device().arch() != ARCH::QUASAR) {                                   \
            GTEST_SKIP() << "SfpuOp is Quasar-only";                                   \
        }                                                                              \
        run_sfpu_op_dfb_case(                                                          \
            this->device(),                                                            \
            SfpuOpDfbCase{                                                             \
                .kernel = "tests/tt_metal/tt_metal/test_kernels/compute/" kernel_file, \
                .route = SfpuOpRoute::op_route,                                        \
                .expected = (expected_fn),                                             \
                .ring_entries = (ring),                                                \
                .total_entries = (total)});                                            \
    }

// SFPU op alone on Dest: copy a tile to Dest, exp in place, pack.
SFPU_OP_DFB_TEST(SfpuOpDestExp, "sfpu_exp_dest_quasar.cpp", Dest, exp_of_in, 8, 8)
// FPU op, then SFPU op on Dest: the FPU adds the tile to itself, exp in place, pack.
SFPU_OP_DFB_TEST(SfpuOpDestExpAfterFpuAdd, "fpu_add_sfpu_exp_dest_quasar.cpp", Dest, exp_of_2in, 8, 8)
// SFPU op on SrcS: trisc3 pops dfb_in and pushes dfb_out around unpacker 2 -> exp -> packer 1.
SFPU_OP_DFB_TEST(SfpuOpSrcsExp, "sfpu_exp_srcs_dfb_quasar.cpp", Srcs, exp_of_in, 8, 8)
// The same with rings smaller than the run: a credit posted before unpacker 2 or packer 1 is done shows up as
// wrong data.
SFPU_OP_DFB_TEST(SfpuOpSrcsExpRingPressure, "sfpu_exp_srcs_dfb_quasar.cpp", Srcs, exp_of_in, 2, 16)
// Mixed: FPU add on Dest -> pack into a DFB inside the core -> trisc3 runs exp on SrcS.
SFPU_OP_DFB_TEST(SfpuOpMixedFpuAddThenSrcsExp, "fpu_add_sfpu_exp_srcs_quasar.cpp", FpuDestThenSrcs, exp_of_2in, 8, 8)
// Two-input SrcS op: one input is the FPU result from Dest, the other a tile trisc3 reads straight from L1.
SFPU_OP_DFB_TEST(
    SfpuOpSrcsAddDestAndL1Inputs, "fpu_add_sfpu_add_srcs_quasar.cpp", FpuDestAndL1ToSrcs, two_in_plus_in1, 8, 8)

#undef SFPU_OP_DFB_TEST

}  // namespace tt::tt_metal
