// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <unistd.h>

#include <chrono>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "multi_device_fixture.hpp"

namespace tt::tt_metal {

using EmuleNamedCompileTimeArgsTest = GenericMeshDeviceFixture;

TEST_F(EmuleNamedCompileTimeArgsTest, FreshJitUsesTheEmulatorNamedApi) {
    const auto mesh_device = get_mesh_device();
    distributed::MeshWorkload workload;
    const auto range = distributed::MeshCoordinateRange(mesh_device->shape());
    workload.add_program(range, CreateProgram());
    auto& program = workload.get_programs().at(range);

    // Change the source identity each run so a disk-cache hit cannot hide a broken
    // wrapper include sequence. The static_assert checks the API at JIT compile time.
    const std::string source = "// fresh JIT " + std::to_string(getpid()) + " " +
                               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                               "\n"
                               "#include \"api/compile_time_args.h\"\n"
                               "static_assert(get_named_compile_time_arg_val(\"n\") == 7);\n"
                               "void kernel_main() {}\n";
    CreateKernelFromString(
        program,
        source,
        CoreCoord{0, 0},
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .named_compile_args = {{"n", 7}}});
    auto& cq = mesh_device->mesh_command_queue();
    ASSERT_NO_THROW(distributed::EnqueueMeshWorkload(cq, workload, false));
    ASSERT_NO_THROW(distributed::Finish(cq));
}

}  // namespace tt::tt_metal
