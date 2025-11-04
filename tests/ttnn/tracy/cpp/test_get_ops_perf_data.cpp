// write test that by time it's done it's generated json artifacts, dump json artifacts to generated/profiler/...
// compare these artifacts to ensure they're correct
// try to check durations,start & end timestamps as well

// add pytest to test_device_profiler that calls this binary

#include <gtest/gtest.h>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <umd/device/types/cluster_descriptor_types.hpp>
#include "context/metal_context.hpp"

using namespace tt::tt_metal;

class GetOpsPerfDataFixture : public testing::Test {
protected:
    std::shared_ptr<distributed::MeshDevice> mesh_device_;

    void SetUp() override {
        if (!MetalContext::instance().rtoptions().get_profiler_enabled()) {
            GTEST_SKIP() << "Skipping test, since it can only be run with profiler enabled.";
        }

        if (!MetalContext::instance().rtoptions().get_profiler_mid_run_dump()) {
            GTEST_SKIP() << "Skipping test, since it can only be run with profiler mid-run dump enabled.";
        }

        if (!MetalContext::instance().rtoptions().get_profiler_cpp_post_process()) {
            GTEST_SKIP() << "Skipping test, since it can only be run with profiler C++ post-processing enabled.";
        }

        constexpr tt::ChipId device_id = 0;
        mesh_device_ = distributed::MeshDevice::create_unit_mesh(device_id);
    }

    void TearDown() override {
        if (mesh_device_) {
            mesh_device_->close();
        }
    }

    void RunWorkload() {
        CoreCoord compute_with_storage_size = mesh_device_->compute_with_storage_grid_size();
        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {compute_with_storage_size.x - 1, compute_with_storage_size.y - 1};
        CoreRange all_cores(start_core, end_core);

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device_->shape());
        Program program = CreateProgram();
        program.set_runtime_id(program_runtime_id_++);

        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op.cpp",
            all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

        CreateKernel(
            program,
            "tt_metal/programming_examples/profiler/test_multi_op/kernels/multi_op_compute.cpp",
            all_cores,
            ComputeConfig{});

        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), workload, false);
    }

private:
    uint32_t program_runtime_id_ = 1;
};

// Test that calls GetLatestOpsPerfData() and GetAllOpsPerfData() before any ReadMeshDeviceProfilerResults() calls
TEST_F(GetOpsPerfDataFixture, TestGetOpsPerfDataBeforeReadMeshDeviceProfilerResultsCall) {
    // RunWorkload();
    // RunWorkload();
    // RunWorkload();

    const std::map<tt::ChipId, std::set<OpAnalysisData>> latest_ops_perf_data = GetLatestOpsPerfData();
    const std::map<tt::ChipId, std::set<OpAnalysisData>> all_ops_perf_data = GetAllOpsPerfData();

    ReadMeshDeviceProfilerResults(*mesh_device_);

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_TRUE(latest_ops_perf_data.at(0).empty());

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_TRUE(all_ops_perf_data.at(0).empty());
}

// Test that calls GetLatestOpsPerfData() and GetAllOpsPerfData() after a single ReadMeshDeviceProfilerResults() call
TEST_F(GetOpsPerfDataFixture, TestGetOpsPerfDataAfterSingleReadMeshDeviceProfilerResultsCall) {
    RunWorkload();
    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);

    const std::map<tt::ChipId, std::set<OpAnalysisData>> latest_ops_perf_data = GetLatestOpsPerfData();
    const std::map<tt::ChipId, std::set<OpAnalysisData>> all_ops_perf_data = GetAllOpsPerfData();

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_EQ(latest_ops_perf_data.at(0).size(), 3);

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 3);
}

// Test that calls ReadMeshDeviceProfilerResults() multiple times and calls GetLatestOpsPerfData() and
// GetAllOpsPerfData() after each call
TEST_F(GetOpsPerfDataFixture, TestGetOpsPerfDataAfterMultipleReadMeshDeviceProfilerResultsCalls) {
    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    std::map<tt::ChipId, std::set<OpAnalysisData>> latest_ops_perf_data = GetLatestOpsPerfData();
    std::map<tt::ChipId, std::set<OpAnalysisData>> all_ops_perf_data = GetAllOpsPerfData();

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_EQ(latest_ops_perf_data.at(0).size(), 2);

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 2);

    RunWorkload();
    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    latest_ops_perf_data = GetLatestOpsPerfData();
    all_ops_perf_data = GetAllOpsPerfData();

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_EQ(latest_ops_perf_data.at(0).size(), 3);

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 5);
}
