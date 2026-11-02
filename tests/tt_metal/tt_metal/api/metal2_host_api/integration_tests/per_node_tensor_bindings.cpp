// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>

#include "impl/program/program_impl.hpp"
#include "tt_metal/impl/dispatch/slow_dispatch.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/program_spec_hw_fixture.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeMinimalGen1ComputeKernel;
using test_helpers::MakeMinimalGen1DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::MakePerNodeTensorParameter;
using test_helpers::ProgramSpecHWTest;

// The HYBRID allocator is the only mode that supports per-node allocation; it is selected at device creation.
class ProgramSpecHybridAllocatorHWTest : public ProgramSpecHWTest {
protected:
    void SetUp() override {
        setenv("TT_METAL_ALLOCATOR_MODE_HYBRID", "1", /*overwrite=*/1);
        ProgramSpecHWTest::SetUp();
    }
    void TearDown() override {
        ProgramSpecHWTest::TearDown();
        unsetenv("TT_METAL_ALLOCATOR_MODE_HYBRID");
    }
};

uint32_t ShardAddress(const MeshTensor& tensor, NodeCoord node) {
    const Buffer& buffer = *tensor.mesh_buffer().get_reference_buffer();
    return static_cast<uint32_t>(per_core_allocation::get_per_core_address(buffer, node));
}

// A 2-node tensor from MakePerNodeTensorParameter whose shards sit at different L1 addresses: a one-node blocker
// allocated first on node (0, 0) pushes that node's shard above the node (1, 0) one.
struct DistinctAddressesTensor {
    MeshTensor blocker;  // Must be declared, and so allocated, before `tensor`.
    MeshTensor tensor;

    DistinctAddressesTensor(distributed::MeshDevice& mesh_device, const TensorParameter& param) :
        blocker{
            MeshTensor::allocate_on_device(mesh_device, MakePerNodeTensorParameter("blocker", /*num_cores=*/1).spec)},
        tensor{MeshTensor::allocate_on_device(mesh_device, param.spec)} {
        TT_FATAL(address({0, 0}) != address({1, 0}), "The blocker failed to give the shards distinct addresses");
    }

    uint32_t address(NodeCoord node) const { return ShardAddress(tensor, node); }
};

TEST_F(ProgramSpecHybridAllocatorHWTest, PerNodeTensorBindingLocalTensorAccessor) {
    auto mesh_device = devices_.at(0);
    const TensorParameter tensor_param = MakePerNodeTensorParameter("local_t", /*num_cores=*/2);
    const DistinctAddressesTensor local(*mesh_device, tensor_param);
    const NodeCoord node0{0, 0};
    const NodeCoord node1{1, 0};
    auto compute = MakeMinimalGen1ComputeKernel("compute");
    compute.source = "tests/tt_metal/tt_metal/test_kernels/compute/local_tensor_accessor_compute.cpp";
    compute.runtime_arg_schema = {.runtime_arg_names = {"report_addr"}};
    BindTensorParameterToKernel(compute, "local_t", "local_t");
    Program program = MakeProgramFromSpec(
        *mesh_device,
        ProgramSpec{
            .name = "per_node_local_tensor_accessor",
            .kernels = {compute},
            .tensor_parameters = {tensor_param},
            .work_units = {MakeMinimalWorkUnit("work_unit_0", NodeRange{node0, node1}, {"compute"})},
        });
    // The kernel reports the address it resolves once per API, in this order (must match the writes in
    // local_tensor_accessor_compute.cpp).
    constexpr std::array<const char*, 4> kReportedApis = {
        "AddressAccess::address(acc.local_mem())", "&*acc.local_mem()", "&acc[0]", "&acc_legacy[0]"};
    constexpr uint32_t kReportAddr = 100 * 1024;
    std::vector<uint32_t> zero_report(kReportedApis.size(), 0u);
    ProgramRunArgs::KernelRunArgs::RuntimeArgValues runtime_arg_values;
    for (const NodeCoord& node : {node0, node1}) {
        slow_dispatch::WriteToL1(*mesh_device, node, kReportAddr, zero_report);
        AddRuntimeArgsForNode(runtime_arg_values, node, {{"report_addr", kReportAddr}});
    }
    SetProgramRunArgs(
        program,
        ProgramRunArgs{
            .kernel_run_args = {{
                .kernel = KernelSpecName{"compute"},
                .runtime_arg_values = std::move(runtime_arg_values),
            }},
            .tensor_args = {{TensorParamName{"local_t"}, TensorArgument{local.tensor}}},
        });
    LaunchProgram(*mesh_device, std::move(program));
    for (const NodeCoord& node : {node0, node1}) {
        std::vector<uint32_t> reported;
        slow_dispatch::ReadFromL1(*mesh_device, node, kReportAddr, kReportedApis.size() * sizeof(uint32_t), reported);
        ASSERT_EQ(reported.size(), kReportedApis.size());
        for (size_t i = 0; i < kReportedApis.size(); ++i) {
            EXPECT_EQ(reported[i], local.address(node)) << kReportedApis[i] << " mismatch on node " << node.str();
        }
    }
}

TEST_F(ProgramSpecHybridAllocatorHWTest, PerNodeTensorBindingSurvivesArgUpdates) {
    auto mesh_device = devices_.at(0);
    const TensorParameter per_node_param = MakePerNodeTensorParameter("per_node", /*num_cores=*/2);
    const TensorParameter plain_param = MakePerNodeTensorParameter("plain", /*num_cores=*/2, /*per_node=*/false);
    const DistinctAddressesTensor per_node_a(*mesh_device, per_node_param);
    MeshTensor per_node_b = MeshTensor::allocate_on_device(*mesh_device, per_node_param.spec);
    MeshTensor plain = MeshTensor::allocate_on_device(*mesh_device, plain_param.spec);
    // The per-node binding's RTA word sits between the named RTAs and the vararg, so a misplaced write to any of
    // these slots shows up in the report.
    const char* source = R"(
#include <cstdint>
#include "api/tensor/local_tensor_accessor.h"
#include "experimental/kernel_args.h"
#include "internal/local_mem/address_access.h"
#include "internal/risc_attribs.h"
void kernel_main() {
    auto* report = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg(args::report_addr));
    LocalTensorAccessor<uint32_t> per_node(tensor::per_node);
    LocalTensorAccessor<uint32_t> plain(tensor::plain);
    report[0] = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(per_node.local_mem()));
    report[1] = static_cast<uint32_t>(local_mem::detail::AddressAccess::address(plain.local_mem()));
    report[2] = get_arg(args::tag);
    report[3] = get_vararg(0);
}
)";
    enum ReportWord : uint32_t { kPerNodeAddress, kPlainAddress, kTag, kVararg, kNumReportWords };
    auto dm = MakeMinimalGen1DMKernel("dm");
    dm.source = KernelSpec::SourceCode{source};
    dm.runtime_arg_schema = {.runtime_arg_names = {"report_addr", "tag"}};
    dm.advanced_options = KernelAdvancedOptions{.num_runtime_varargs = 1};
    BindTensorParameterToKernel(dm, "per_node", "per_node");
    BindTensorParameterToKernel(dm, "plain", "plain");
    const NodeCoord node0{0, 0};
    const NodeCoord node1{1, 0};
    const distributed::MeshCoordinateRange device_range(mesh_device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(
        device_range,
        MakeProgramFromSpec(
            *mesh_device,
            ProgramSpec{
                .name = "per_node_arg_updates",
                .kernels = {dm},
                .tensor_parameters = {per_node_param, plain_param},
                .work_units = {MakeMinimalWorkUnit("work_unit_0", NodeRange{node0, node1}, {"dm"})},
            }));
    Program& program = workload.get_programs().at(device_range);
    constexpr uint32_t kReportAddr = 100 * 1024;
    auto kernel_run_args = [&](uint32_t tag, bool with_varargs) {
        ProgramRunArgs::KernelRunArgs::RuntimeArgValues runtime_arg_values;
        AdvancedKernelRunArgs advanced_options;
        for (const NodeCoord& node : {node0, node1}) {
            AddRuntimeArgsForNode(runtime_arg_values, node, {{"report_addr", kReportAddr}, {"tag", tag + node.x}});
            if (with_varargs) {
                advanced_options.runtime_varargs[node] = {0x7000u + node.x};
            }
        }
        return ProgramRunArgs::KernelRunArgs{
            .kernel = KernelSpecName{"dm"},
            .runtime_arg_values = std::move(runtime_arg_values),
            .advanced_options = std::move(advanced_options),
        };
    };
    auto run_and_check = [&](const char* step, const MeshTensor& expected_per_node, uint32_t expected_tag) {
        SCOPED_TRACE(step);
        std::vector<uint32_t> zero_report(kNumReportWords, 0u);
        for (const NodeCoord& node : {node0, node1}) {
            slow_dispatch::WriteToL1(*mesh_device, node, kReportAddr, zero_report);
        }
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/true);
        for (const NodeCoord& node : {node0, node1}) {
            std::vector<uint32_t> report;
            slow_dispatch::ReadFromL1(*mesh_device, node, kReportAddr, kNumReportWords * sizeof(uint32_t), report);
            ASSERT_EQ(report.size(), kNumReportWords);
            EXPECT_EQ(report[kPerNodeAddress], ShardAddress(expected_per_node, node)) << "on node " << node.str();
            EXPECT_EQ(report[kPlainAddress], plain.address()) << "on node " << node.str();
            EXPECT_EQ(report[kTag], expected_tag + node.x) << "on node " << node.str();
            EXPECT_EQ(report[kVararg], 0x7000u + node.x) << "on node " << node.str();
        }
    };
    for (const NodeCoord& node : {node0, node1}) {
        ASSERT_NE(per_node_a.address(node), ShardAddress(per_node_b, node)) << "Test precondition";
    }
    SetProgramRunArgs(
        program,
        ProgramRunArgs{
            .kernel_run_args = {kernel_run_args(0x1000, /*with_varargs=*/true)},
            .tensor_args =
                {{TensorParamName{"per_node"}, TensorArgument{per_node_a.tensor}},
                 {TensorParamName{"plain"}, TensorArgument{plain}}},
        });
    run_and_check("SetProgramRunArgs", per_node_a.tensor, 0x1000);
    UpdateTensorArgs(
        program,
        {{TensorParamName{"per_node"}, TensorArgument{per_node_b}}, {TensorParamName{"plain"}, TensorArgument{plain}}});
    run_and_check("UpdateTensorArgs", per_node_b, 0x1000);
    // Omits the ordinary tensor and the varargs, which must keep their values.
    UpdateProgramRunArgs(
        program,
        ProgramRunArgs{
            .kernel_run_args = {kernel_run_args(0x2000, /*with_varargs=*/false)},
            .tensor_args = {{TensorParamName{"per_node"}, TensorArgument{per_node_a.tensor}}},
        });
    run_and_check("partial UpdateProgramRunArgs", per_node_a.tensor, 0x2000);
}

}  // namespace
}  // namespace tt::tt_metal::experimental
