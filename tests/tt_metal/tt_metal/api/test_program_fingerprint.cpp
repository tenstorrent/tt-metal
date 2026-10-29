// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/program_fingerprint.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tile.hpp>

#include "multi_device_fixture.hpp"

namespace tt::tt_metal {
namespace {

using ProgramFingerprintTest = MeshDevice1x1Fixture;
namespace fingerprint = experimental::program_fingerprint;

struct ProgramParams {
    std::vector<uint32_t> compile_args = {1, 2, 3};
    std::map<std::string, std::string> defines = {{"FLAVOUR", "a"}};
    std::vector<uint32_t> runtime_args = {7};
    uint32_t cb_page_size = 2048;
    uint32_t semaphore_value = 0;
    Tile cb_tile;
};

Program make_program(const ProgramParams& params) {
    const CoreRange cores({0, 0}, {1, 1});
    Program program;
    const auto kernel = CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/blank.cpp",
        cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = params.compile_args,
            .defines = params.defines});
    SetRuntimeArgs(program, kernel, cores, params.runtime_args);
    CreateCircularBuffer(
        program,
        cores,
        CircularBufferConfig(2 * params.cb_page_size, {{0, tt::DataFormat::Float16_b}})
            .set_page_size(0, params.cb_page_size)
            .set_tile_dims(0, params.cb_tile));
    CreateSemaphore(program, cores, params.semaphore_value);
    return program;
}

std::vector<std::string> describe_with(const ProgramParams& params) {
    auto program = make_program(params);
    return fingerprint::describe(program);
}

TEST_F(ProgramFingerprintTest, RebuildIsIdentical) {
    const auto first = describe_with({});
    EXPECT_FALSE(first.empty());
    EXPECT_EQ(first, describe_with({}));
}

TEST_F(ProgramFingerprintTest, RuntimeArgsAreLeftOut) {
    EXPECT_EQ(describe_with({}), describe_with({.runtime_args = {8}}));
}

TEST_F(ProgramFingerprintTest, RuntimeArgCountChangesIt) {
    EXPECT_NE(describe_with({}), describe_with({.runtime_args = {7, 8}}));
}

TEST_F(ProgramFingerprintTest, CompileArgsChangeIt) {
    EXPECT_NE(describe_with({}), describe_with({.compile_args = {1, 2, 4}}));
}

TEST_F(ProgramFingerprintTest, DefinesChangeIt) {
    EXPECT_NE(describe_with({}), describe_with({.defines = {{"FLAVOUR", "b"}}}));
}

TEST_F(ProgramFingerprintTest, CircularBufferPageSizeChangesIt) {
    EXPECT_NE(describe_with({}), describe_with({.cb_page_size = 1024}));
}

TEST_F(ProgramFingerprintTest, CircularBufferTileGeometryChangesIt) {
    EXPECT_NE(describe_with({}), describe_with({.cb_tile = Tile({32, 32}, /*transpose_tile=*/true)}));
    EXPECT_NE(
        describe_with({.cb_tile = Tile({16, 32}, {16, 16})}), describe_with({.cb_tile = Tile({16, 32}, {8, 16})}));
}

TEST_F(ProgramFingerprintTest, SemaphoreInitialValueChangesIt) {
    EXPECT_NE(describe_with({}), describe_with({.semaphore_value = 1}));
}

TEST_F(ProgramFingerprintTest, CompiledProgramMatchesFreshOne) {
    const auto range = distributed::MeshCoordinateRange(get_mesh_device()->shape());
    distributed::MeshWorkload dispatched;
    dispatched.add_program(range, make_program({}));
    auto& cq = get_mesh_device()->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, dispatched, false);
    distributed::Finish(cq);

    distributed::MeshWorkload fresh;
    fresh.add_program(range, make_program({}));
    EXPECT_EQ(fingerprint::describe(dispatched), fingerprint::describe(fresh));
}

TEST_F(ProgramFingerprintTest, WorkloadPrefixesEachProgramWithItsRange) {
    const auto range = distributed::MeshCoordinateRange(get_mesh_device()->shape());
    distributed::MeshWorkload workload;
    workload.add_program(range, make_program({}));
    const auto lines = fingerprint::describe(workload);
    const auto program_lines = describe_with({});
    ASSERT_EQ(lines.size(), program_lines.size());
    for (size_t i = 0; i < lines.size(); i++) {
        EXPECT_TRUE(lines[i].ends_with(program_lines[i])) << lines[i];
    }
}

}  // namespace
}  // namespace tt::tt_metal
