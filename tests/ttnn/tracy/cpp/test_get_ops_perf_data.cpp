// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_paths.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal {
void to_json(nlohmann::json& j, const OpId& op_id) {
    j = nlohmann::json{
        {"runtime_id", op_id.runtime_id}, {"trace_id", op_id.trace_id}, {"trace_id_counter", op_id.trace_id_counter}};
}

void to_json(nlohmann::json& j, const OpSingleAnalysisResult& op_single_analysis_result) {
    j = nlohmann::json{
        {"start_timestamp", op_single_analysis_result.start_timestamp},
        {"end_timestamp", op_single_analysis_result.end_timestamp},
        {"duration", op_single_analysis_result.duration}};
}

void to_json(nlohmann::json& j, const OpAnalysisData& op_analysis_data) {
    j = nlohmann::json{
        {"op_id", op_analysis_data.op_id}, {"op_analyses_results", op_analysis_data.op_analyses_results}};
}

void to_json(nlohmann::json& j, const std::map<tt::ChipId, std::set<OpAnalysisData>>& ops_perf_data) {
    for (const auto& [device_id, op_analysis_set] : ops_perf_data) {
        nlohmann::json device_ops_analysis_data_json;
        device_ops_analysis_data_json["device"] = device_id;
        device_ops_analysis_data_json["ops_analysis_data"] = op_analysis_set;
        j.push_back(std::move(device_ops_analysis_data_json));
    }
}

}  // namespace tt::tt_metal

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

    void WriteOpsPerfDataToJson(
        const std::vector<std::map<tt::ChipId, std::set<OpAnalysisData>>>& ops_perf_data_list,
        const std::string& file_name) {
        nlohmann::json json_ops_perf_data_list;
        for (const auto& ops_perf_data : ops_perf_data_list) {
            json_ops_perf_data_list.push_back(ops_perf_data);
        }
        std::ofstream file(get_profiler_logs_dir() + file_name);
        file << std::setw(4) << json_ops_perf_data_list << std::endl;
    }

private:
    uint32_t program_runtime_id_ = 1;
};

// Test that calls GetLatestOpsPerfData() and GetAllOpsPerfData() before any ReadMeshDeviceProfilerResults() calls
TEST_F(GetOpsPerfDataFixture, TestGetOpsPerfDataBeforeReadMeshDeviceProfilerResultsCall) {
    RunWorkload();
    RunWorkload();
    RunWorkload();

    const std::map<tt::ChipId, std::set<OpAnalysisData>> latest_ops_perf_data = GetLatestOpsPerfData();
    const std::map<tt::ChipId, std::set<OpAnalysisData>> all_ops_perf_data = GetAllOpsPerfData();

    ReadMeshDeviceProfilerResults(*mesh_device_);

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_TRUE(latest_ops_perf_data.at(0).empty());

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_TRUE(all_ops_perf_data.at(0).empty());

    WriteOpsPerfDataToJson({latest_ops_perf_data}, "test_get_ops_perf_data_latest.json");
    WriteOpsPerfDataToJson({all_ops_perf_data}, "test_get_ops_perf_data_all.json");
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

    uint32_t expected_runtime_id = 1;
    for (const OpAnalysisData& op_analysis_data : latest_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 3);

    expected_runtime_id = 1;
    for (const OpAnalysisData& op_analysis_data : all_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    EXPECT_EQ(latest_ops_perf_data, all_ops_perf_data);

    WriteOpsPerfDataToJson({latest_ops_perf_data}, "test_get_ops_perf_data_latest.json");
    WriteOpsPerfDataToJson({all_ops_perf_data}, "test_get_ops_perf_data_all.json");
}

// Test that calls ReadMeshDeviceProfilerResults() multiple times and calls GetLatestOpsPerfData() and
// GetAllOpsPerfData() after each call
TEST_F(GetOpsPerfDataFixture, TestGetOpsPerfDataAfterMultipleReadMeshDeviceProfilerResultsCalls) {
    std::vector<std::map<tt::ChipId, std::set<OpAnalysisData>>> latest_ops_perf_data_list;
    std::vector<std::map<tt::ChipId, std::set<OpAnalysisData>>> all_ops_perf_data_list;

    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    std::map<tt::ChipId, std::set<OpAnalysisData>> latest_ops_perf_data = GetLatestOpsPerfData();
    std::map<tt::ChipId, std::set<OpAnalysisData>> all_ops_perf_data = GetAllOpsPerfData();

    latest_ops_perf_data_list.push_back(latest_ops_perf_data);
    all_ops_perf_data_list.push_back(all_ops_perf_data);

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_EQ(latest_ops_perf_data.at(0).size(), 2);

    uint32_t expected_runtime_id = 1;
    for (const OpAnalysisData& op_analysis_data : latest_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 2);

    expected_runtime_id = 1;
    for (const OpAnalysisData& op_analysis_data : all_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    EXPECT_EQ(latest_ops_perf_data, all_ops_perf_data);

    RunWorkload();
    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    latest_ops_perf_data = GetLatestOpsPerfData();
    all_ops_perf_data = GetAllOpsPerfData();

    latest_ops_perf_data_list.push_back(latest_ops_perf_data);
    all_ops_perf_data_list.push_back(all_ops_perf_data);

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_EQ(latest_ops_perf_data.at(0).size(), 3);

    expected_runtime_id = 3;
    for (const OpAnalysisData& op_analysis_data : latest_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 5);

    expected_runtime_id = 1;
    for (const OpAnalysisData& op_analysis_data : all_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    ReadMeshDeviceProfilerResults(*mesh_device_);

    latest_ops_perf_data = GetLatestOpsPerfData();
    all_ops_perf_data = GetAllOpsPerfData();

    latest_ops_perf_data_list.push_back(latest_ops_perf_data);
    all_ops_perf_data_list.push_back(all_ops_perf_data);

    EXPECT_EQ(latest_ops_perf_data.size(), 1);
    EXPECT_TRUE(latest_ops_perf_data.contains(0));
    EXPECT_TRUE(latest_ops_perf_data.at(0).empty());

    EXPECT_EQ(all_ops_perf_data.size(), 1);
    EXPECT_TRUE(all_ops_perf_data.contains(0));
    EXPECT_EQ(all_ops_perf_data.at(0).size(), 6);

    expected_runtime_id = 1;
    for (const OpAnalysisData& op_analysis_data : all_ops_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(op_analysis_data.op_id.runtime_id).base_program_id, expected_runtime_id++);
        EXPECT_EQ(op_analysis_data.op_id.trace_id, INVALID_NUM_OP_ID);
        EXPECT_EQ(op_analysis_data.op_id.trace_id_counter, INVALID_NUM_OP_ID);
    }

    WriteOpsPerfDataToJson(latest_ops_perf_data_list, "test_get_ops_perf_data_latest.json");
    WriteOpsPerfDataToJson(all_ops_perf_data_list, "test_get_ops_perf_data_all.json");
}
