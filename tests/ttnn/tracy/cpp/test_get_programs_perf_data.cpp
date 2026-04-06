// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/profiler.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/profiler/profiler_paths.hpp"
#include <umd/device/types/cluster_descriptor_types.hpp>

namespace tt::tt_metal::experimental {
void to_json(nlohmann::json& j, const ProgramExecutionUID& program_execution_uid) {
    j = nlohmann::json{
        {"runtime_id", program_execution_uid.runtime_id},
        {"trace_id", program_execution_uid.trace_id},
        {"trace_id_counter", program_execution_uid.trace_id_counter}};
}

void to_json(nlohmann::json& j, const ProgramSingleAnalysisResult& program_single_analysis_result) {
    j = nlohmann::json{
        {"start_timestamp", program_single_analysis_result.start_timestamp},
        {"end_timestamp", program_single_analysis_result.end_timestamp},
        {"duration", program_single_analysis_result.duration}};
}

void to_json(nlohmann::json& j, const ProgramAnalysisData& program_analysis_data) {
    j = nlohmann::json{
        {"program_execution_uid", program_analysis_data.program_execution_uid},
        {"program_analyses_results", program_analysis_data.program_analyses_results},
        {"core_count", program_analysis_data.core_count},
        {"num_available_cores", program_analysis_data.num_available_cores}};
}

void to_json(nlohmann::json& j, const std::map<tt::ChipId, std::set<ProgramAnalysisData>>& programs_perf_data) {
    for (const auto& [device_id, program_analysis_set] : programs_perf_data) {
        nlohmann::json device_programs_analysis_data_json;
        device_programs_analysis_data_json["device"] = device_id;
        device_programs_analysis_data_json["programs_analysis_data"] = program_analysis_set;
        j.push_back(std::move(device_programs_analysis_data_json));
    }
}

}  // namespace tt::tt_metal::experimental

using namespace tt::tt_metal;

class GetProgramsPerfDataFixture : public testing::Test {
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

    void WriteProgramsPerfDataToJson(
        const std::vector<std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>>>& programs_perf_data_list,
        const std::string& file_name) {
        nlohmann::json json_programs_perf_data_list;
        for (const auto& programs_perf_data : programs_perf_data_list) {
            json_programs_perf_data_list.push_back(programs_perf_data);
        }
        std::ofstream file(get_profiler_logs_dir() + file_name);
        file << std::setw(4) << json_programs_perf_data_list << std::endl;
    }

private:
    uint32_t program_runtime_id_ = 1;
};

// Test that calls GetLatestProgramsPerfData() and GetAllProgramsPerfData() before any ReadMeshDeviceProfilerResults()
// calls
TEST_F(GetProgramsPerfDataFixture, TestGetProgramsPerfDataBeforeReadMeshDeviceProfilerResultsCall) {
    RunWorkload();
    RunWorkload();
    RunWorkload();

    const std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>> latest_programs_perf_data =
        experimental::GetLatestProgramsPerfData();
    const std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>> all_programs_perf_data =
        experimental::GetAllProgramsPerfData();
    const std::map<tt::ChipId, experimental::KernelDurationSummary> latest_kernel_duration_summary =
        experimental::GetLatestKernelDurationSummary();
    const std::map<tt::ChipId, experimental::KernelDurationSummary> all_kernel_duration_summary =
        experimental::GetAllKernelDurationSummary();

    ReadMeshDeviceProfilerResults(*mesh_device_);

    EXPECT_EQ(latest_programs_perf_data.size(), 1);
    EXPECT_TRUE(latest_programs_perf_data.contains(0));
    EXPECT_TRUE(latest_programs_perf_data.at(0).empty());

    EXPECT_EQ(all_programs_perf_data.size(), 1);
    EXPECT_TRUE(all_programs_perf_data.contains(0));
    EXPECT_TRUE(all_programs_perf_data.at(0).empty());

    EXPECT_EQ(latest_kernel_duration_summary.size(), 1);
    EXPECT_TRUE(latest_kernel_duration_summary.contains(0));
    EXPECT_EQ(latest_kernel_duration_summary.at(0).count, 0);
    EXPECT_EQ(latest_kernel_duration_summary.at(0).histogram.num_buckets, 10);
    EXPECT_EQ(latest_kernel_duration_summary.at(0).histogram.bucket_edges_ns.size(), 11);
    EXPECT_EQ(latest_kernel_duration_summary.at(0).histogram.bucket_counts.size(), 10);

    EXPECT_EQ(all_kernel_duration_summary.size(), 1);
    EXPECT_TRUE(all_kernel_duration_summary.contains(0));
    EXPECT_EQ(all_kernel_duration_summary.at(0).count, 0);
    EXPECT_EQ(all_kernel_duration_summary.at(0).histogram.num_buckets, 10);
    EXPECT_EQ(all_kernel_duration_summary.at(0).histogram.bucket_edges_ns.size(), 11);
    EXPECT_EQ(all_kernel_duration_summary.at(0).histogram.bucket_counts.size(), 10);

    WriteProgramsPerfDataToJson({latest_programs_perf_data}, "test_get_programs_perf_data_latest.json");
    WriteProgramsPerfDataToJson({all_programs_perf_data}, "test_get_programs_perf_data_all.json");
}

// Test that calls GetLatestProgramsPerfData() and GetAllProgramsPerfData() after a single
// ReadMeshDeviceProfilerResults() call
TEST_F(GetProgramsPerfDataFixture, TestGetProgramsPerfDataAfterSingleReadMeshDeviceProfilerResultsCall) {
    RunWorkload();
    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);

    const std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>> latest_programs_perf_data =
        experimental::GetLatestProgramsPerfData();
    const std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>> all_programs_perf_data =
        experimental::GetAllProgramsPerfData();
    const std::map<tt::ChipId, experimental::KernelDurationSummary> latest_kernel_duration_summary =
        experimental::GetLatestKernelDurationSummary();
    const std::map<tt::ChipId, experimental::KernelDurationSummary> all_kernel_duration_summary =
        experimental::GetAllKernelDurationSummary();

    EXPECT_EQ(latest_programs_perf_data.size(), 1);
    EXPECT_TRUE(latest_programs_perf_data.contains(0));
    EXPECT_EQ(latest_programs_perf_data.at(0).size(), 3);

    uint32_t expected_runtime_id = 1;
    for (const experimental::ProgramAnalysisData& program_analysis_data : latest_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    EXPECT_EQ(all_programs_perf_data.size(), 1);
    EXPECT_TRUE(all_programs_perf_data.contains(0));
    EXPECT_EQ(all_programs_perf_data.at(0).size(), 3);

    expected_runtime_id = 1;
    for (const experimental::ProgramAnalysisData& program_analysis_data : all_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    EXPECT_EQ(latest_programs_perf_data, all_programs_perf_data);

    // Summary API should be available and return a populated histogram schema.
    EXPECT_EQ(latest_kernel_duration_summary.size(), 1);
    EXPECT_TRUE(latest_kernel_duration_summary.contains(0));
    EXPECT_EQ(latest_kernel_duration_summary.at(0).histogram.num_buckets, 10);
    EXPECT_EQ(latest_kernel_duration_summary.at(0).histogram.bucket_edges_ns.size(), 11);
    EXPECT_EQ(latest_kernel_duration_summary.at(0).histogram.bucket_counts.size(), 10);

    EXPECT_EQ(all_kernel_duration_summary.size(), 1);
    EXPECT_TRUE(all_kernel_duration_summary.contains(0));
    EXPECT_EQ(all_kernel_duration_summary.at(0).histogram.num_buckets, 10);
    EXPECT_EQ(all_kernel_duration_summary.at(0).histogram.bucket_edges_ns.size(), 11);
    EXPECT_EQ(all_kernel_duration_summary.at(0).histogram.bucket_counts.size(), 10);

    WriteProgramsPerfDataToJson({latest_programs_perf_data}, "test_get_programs_perf_data_latest.json");
    WriteProgramsPerfDataToJson({all_programs_perf_data}, "test_get_programs_perf_data_all.json");
}

// Test that calls ReadMeshDeviceProfilerResults() multiple times and calls GetLatestProgramsPerfData() and
// GetAllProgramsPerfData() after each call
TEST_F(GetProgramsPerfDataFixture, TestGetProgramsPerfDataAfterMultipleReadMeshDeviceProfilerResultsCalls) {
    std::vector<std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>>> latest_programs_perf_data_list;
    std::vector<std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>>> all_programs_perf_data_list;

    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>> latest_programs_perf_data =
        experimental::GetLatestProgramsPerfData();
    std::map<tt::ChipId, std::set<experimental::ProgramAnalysisData>> all_programs_perf_data =
        experimental::GetAllProgramsPerfData();

    latest_programs_perf_data_list.push_back(latest_programs_perf_data);
    all_programs_perf_data_list.push_back(all_programs_perf_data);

    EXPECT_EQ(latest_programs_perf_data.size(), 1);
    EXPECT_TRUE(latest_programs_perf_data.contains(0));
    EXPECT_EQ(latest_programs_perf_data.at(0).size(), 2);

    uint32_t expected_runtime_id = 1;
    for (const experimental::ProgramAnalysisData& program_analysis_data : latest_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    EXPECT_EQ(all_programs_perf_data.size(), 1);
    EXPECT_TRUE(all_programs_perf_data.contains(0));
    EXPECT_EQ(all_programs_perf_data.at(0).size(), 2);

    expected_runtime_id = 1;
    for (const experimental::ProgramAnalysisData& program_analysis_data : all_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    EXPECT_EQ(latest_programs_perf_data, all_programs_perf_data);

    RunWorkload();
    RunWorkload();
    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    latest_programs_perf_data = experimental::GetLatestProgramsPerfData();
    all_programs_perf_data = experimental::GetAllProgramsPerfData();

    latest_programs_perf_data_list.push_back(latest_programs_perf_data);
    all_programs_perf_data_list.push_back(all_programs_perf_data);

    EXPECT_EQ(latest_programs_perf_data.size(), 1);
    EXPECT_TRUE(latest_programs_perf_data.contains(0));
    EXPECT_EQ(latest_programs_perf_data.at(0).size(), 3);

    expected_runtime_id = 3;
    for (const experimental::ProgramAnalysisData& program_analysis_data : latest_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    EXPECT_EQ(all_programs_perf_data.size(), 1);
    EXPECT_TRUE(all_programs_perf_data.contains(0));
    EXPECT_EQ(all_programs_perf_data.at(0).size(), 5);

    expected_runtime_id = 1;
    for (const experimental::ProgramAnalysisData& program_analysis_data : all_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    RunWorkload();

    ReadMeshDeviceProfilerResults(*mesh_device_);
    ReadMeshDeviceProfilerResults(*mesh_device_);

    latest_programs_perf_data = experimental::GetLatestProgramsPerfData();
    all_programs_perf_data = experimental::GetAllProgramsPerfData();

    latest_programs_perf_data_list.push_back(latest_programs_perf_data);
    all_programs_perf_data_list.push_back(all_programs_perf_data);

    EXPECT_EQ(latest_programs_perf_data.size(), 1);
    EXPECT_TRUE(latest_programs_perf_data.contains(0));
    EXPECT_TRUE(latest_programs_perf_data.at(0).empty());

    EXPECT_EQ(all_programs_perf_data.size(), 1);
    EXPECT_TRUE(all_programs_perf_data.contains(0));
    EXPECT_EQ(all_programs_perf_data.at(0).size(), 6);

    expected_runtime_id = 1;
    for (const experimental::ProgramAnalysisData& program_analysis_data : all_programs_perf_data.at(0)) {
        EXPECT_EQ(
            detail::DecodePerDeviceProgramID(program_analysis_data.program_execution_uid.runtime_id).base_program_id,
            expected_runtime_id++);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id, experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
        EXPECT_EQ(
            program_analysis_data.program_execution_uid.trace_id_counter,
            experimental::INVALID_NUM_PROGRAM_EXECUTION_UID);
    }

    WriteProgramsPerfDataToJson(latest_programs_perf_data_list, "test_get_programs_perf_data_latest.json");
    WriteProgramsPerfDataToJson(all_programs_perf_data_list, "test_get_programs_perf_data_all.json");
}
