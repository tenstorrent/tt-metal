// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Driver to execute sfpi execution tests.

#include "command_queue_fixture.hpp"

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include <tt-metalium/kernel_types.hpp>

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

#include <gtest/gtest.h>
#include <tt-metalium/distributed.hpp>

namespace {

// We recursively scan this directory for kernels named '*.cpp'.
constexpr std::string_view KernelDir = "tests/tt_metal/tt_metal/test_kernels/sfpi";

using namespace tt::tt_metal;

bool runTest(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    const CoreCoord& coord,
    const std::string& path,
    unsigned baseLen) {
    uint32_t args_addr = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    std::vector<uint32_t> compile_args{args_addr};

    auto program(tt::tt_metal::CreateProgram());
    distributed::MeshWorkload workload = distributed::CreateMeshWorkload();
    auto kernel = CreateKernel(
        program,
        path,
        coord,
        tt::tt_metal::ComputeConfig{
            .compile_args = compile_args,
        });
    distributed::AddProgramToMeshWorkload(
        workload, std::move(program), distributed::MeshCoordinateRange(mesh_device->shape()));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);

    distributed::Finish(mesh_device->mesh_command_queue());

    tt::tt_metal::MetalContext::instance().get_cluster().l1_barrier(mesh_device->get_devices()[0]->id());
    auto noc_xy = mesh_device->worker_core_from_logical_core(coord);
    unsigned expected = 0;
    // If path ends in -[digits], extract the expected value
    // If we need more sofphisticate tuning, we should add tags to the
    // body of the file itself.
    auto pos = path.find_last_of('.');
    // NOLINTNEXTLINE(bugprone-inc-dec-in-conditions)
    while (--pos && path[pos] >= '0' && path[pos] <= '9') {
        continue;
    }
    if (path[pos] == '-') {
        while (path[++pos] != '.') {
            expected = expected * 10 + (path[pos] - '0');
        }
        expected |= 0x4000;
    }
    std::vector<uint32_t> args = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        mesh_device->get_devices()[0]->id(), noc_xy, args_addr, sizeof(uint32_t));
    unsigned result = args[0];
    bool pass = result == expected;
    if (pass) {
        std::printf("%s: PASSED\n", path.c_str() + baseLen);
    } else if (expected || (result & 0xc000) != 0x4000) {
        std::printf("%s: FAILED result %#x\n", path.c_str() + baseLen, result);
    } else {
        unsigned line = result & 0x3fff;
        std::printf("%s: FAILED line %u\n", path.c_str() + baseLen, line);
    }
    return pass;
}

bool runTests(
    std::shared_ptr<distributed::MeshDevice> mesh_device,
    const tt::tt_metal::CoreCoord coord,
    std::string& path,
    unsigned baseLen) {
    bool pass = true;
    std::vector<std::string> files;
    std::vector<std::string> dirs;

    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (entry.is_directory()) {
            dirs.push_back(entry.path().filename());
        } else if (entry.path().filename().extension() == ".cpp") {
            files.push_back(entry.path().filename());
        }
    }
    std::sort(files.begin(), files.end());
    std::sort(dirs.begin(), dirs.end());

    path.push_back('/');
    for (const auto& file : files) {
        path.append(file);
        pass &= runTest(mesh_device, coord, path, baseLen);
        path.erase(path.size() - file.size());
    }

    for (const auto& dir : dirs) {
        path.append(dir);
        pass &= runTests(mesh_device, coord, path, baseLen);
        path.erase(path.size() - dir.size());
    }
    path.pop_back();

    return pass;
}

bool runTestsuite(std::shared_ptr<distributed::MeshDevice> mesh_device, const tt::tt_metal::CoreCoord coord) {
    std::string path;
    if (auto* var = std::getenv("TT_METAL_HOME")) {
        path.append(var);
        if (!path.empty()) {
            path.push_back('/');
        }
    }
    path.append(KernelDir);
    return runTests(mesh_device, coord, path, path.find_last_of('/') + 1);
}

using tt::tt_metal::CommandQueueSingleCardProgramFixture;

TEST_F(UnitMeshCQFixture, TensixSFPI) {
    CoreCoord core{0, 0};
    for (auto mesh_device : devices_) {
        EXPECT_TRUE(runTestsuite(mesh_device, core));
    }
}
}
