// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <capnp/rpc-twoparty.h>
#include <gtest/gtest.h>
#include <kj/async-io.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdlib>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/context/metal_env.hpp>
#include <tt-metalium/experimental/inspector.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "impl/debug/inspector/rpc_server_controller.hpp"
#include "impl/program/program_impl.hpp"

namespace tt::tt_metal {
namespace {

int find_free_port() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        return 0;
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = 0;
    socklen_t len = sizeof(addr);
    int port = 0;
    if (bind(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0 &&
        getsockname(sock, reinterpret_cast<sockaddr*>(&addr), &len) == 0) {
        port = ntohs(addr.sin_port);
    }
    close(sock);
    return port;
}

// Sets environment variables for the object's lifetime. Each MetalEnv reads them when it is constructed.
class ScopedEnvVars {
public:
    explicit ScopedEnvVars(std::vector<std::pair<std::string, std::string>> vars) {
        for (auto& [name, value] : vars) {
            const char* previous = getenv(name.c_str());
            saved_.emplace_back(name, previous ? std::optional<std::string>(previous) : std::nullopt);
            setenv(name.c_str(), value.c_str(), /*overwrite=*/1);
        }
    }
    ~ScopedEnvVars() {
        for (const auto& [name, previous] : saved_) {
            if (previous.has_value()) {
                setenv(name.c_str(), previous->c_str(), /*overwrite=*/1);
            } else {
                unsetenv(name.c_str());
            }
        }
    }
    ScopedEnvVars(const ScopedEnvVars&) = delete;
    ScopedEnvVars& operator=(const ScopedEnvVars&) = delete;

private:
    std::vector<std::pair<std::string, std::optional<std::string>>> saved_;
};

// What an inspector session reports over RPC.
struct SessionContents {
    std::unordered_set<int64_t> mesh_ids;
    std::unordered_set<uint64_t> program_ids;
};

SessionContents query_session(const std::string& address) {
    auto io = kj::setupAsyncIo();
    auto& wait_scope = io.waitScope;
    auto conn = io.provider->getNetwork().parseAddress(address).wait(wait_scope)->connect().wait(wait_scope);
    capnp::TwoPartyClient client(*conn);
    auto inspector = client.bootstrap().castAs<inspector::rpc::Inspector>();

    SessionContents contents;
    auto mesh_devices = inspector.getMeshDevicesRequest().send().wait(wait_scope);
    for (auto mesh_device : mesh_devices.getMeshDevices()) {
        contents.mesh_ids.insert(mesh_device.getMeshId());
    }
    auto programs = inspector.getProgramsRequest().send().wait(wait_scope);
    for (auto program : programs.getPrograms()) {
        contents.program_ids.insert(program.getProgramId());
    }
    return contents;
}

experimental::ProgramSpec make_no_op_program_spec() {
    const experimental::KernelSpecName kernel_id{"dm_kernel"};
    experimental::KernelSpec dm_kernel{
        .unique_id = kernel_id,
        .source = experimental::KernelSpec::SourceCode{"void kernel_main() {}"},
        .num_threads = 1,
        .hw_config =
            experimental::DataMovementHardwareConfig{
                .config_1xx =
                    experimental::DataMovementHardwareConfig::DataMovement1XXConfig{
                        .processor = DataMovementProcessor::RISCV_0,
                        .noc = NOC::NOC_0,
                    },
            },
    };
    return experimental::ProgramSpec{
        .name = "inspector_session_noop",
        .kernels = {std::move(dm_kernel)},
        .work_units = {experimental::WorkUnitSpec{
            .name = "work_unit_0",
            .kernels = {kernel_id},
            .target_nodes = experimental::NodeCoord{0, 0},
        }},
    };
}

// Builds a no-op workload bound to `mesh_device`'s context and runs it.
distributed::MeshWorkload run_no_op_workload(distributed::MeshDevice& mesh_device) {
    distributed::MeshWorkload workload = experimental::MakeMeshWorkloadFromSpec(mesh_device, make_no_op_program_spec());
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
    return workload;
}

std::vector<uint64_t> program_ids_of(distributed::MeshWorkload& workload) {
    std::vector<uint64_t> ids;
    for (auto& [_, program] : workload.get_programs()) {
        ids.push_back(program.impl().get_id());
    }
    return ids;
}

}  // namespace

// A mock context has no inspector session, and objects it creates must not show up in the hardware session.
TEST(InspectorSessionTest, HardwareSessionDoesNotRecordMockObjects) {
    const int port = find_free_port();
    ASSERT_NE(port, 0);
    const std::string address = "localhost:" + std::to_string(port);
    ScopedEnvVars env_vars({
        {"TT_METAL_INSPECTOR", "1"},
        {"TT_METAL_INSPECTOR_RPC", "1"},
        {"TT_METAL_INSPECTOR_RPC_SERVER_ADDRESS", address},
        {"TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT", "1"},
    });

    MetalEnv silicon_env;
    auto silicon = distributed::MeshDevice::open_mesh_device(
        silicon_env, distributed::MeshDeviceConfig(silicon_env.get_system_mesh().shape()));
    ASSERT_TRUE(experimental::inspector::IsEnabled(*silicon));

    // Same arch as the silicon: the process-wide build env tables are sized by the first context's HAL.
    MetalEnv mock_env(
        {.mock_cluster_desc_path =
             experimental::get_mock_cluster_desc_name(silicon_env.get_device_properties().arch(), /*num_chips=*/1)});
    auto mock = distributed::MeshDevice::open_mesh_device(
        mock_env, distributed::MeshDeviceConfig(mock_env.get_system_mesh().shape()));
    EXPECT_FALSE(experimental::inspector::IsEnabled(*mock));
    EXPECT_FALSE(experimental::inspector::ShouldCaptureTensorSpecs(*mock));

    // Programs are dropped from the session when destroyed, so keep the workloads alive until the query.
    auto silicon_workload = run_no_op_workload(*silicon);
    auto mock_workload = run_no_op_workload(*mock);

    const SessionContents contents = query_session(address);

    EXPECT_TRUE(contents.mesh_ids.contains(silicon->id()));
    EXPECT_FALSE(contents.mesh_ids.contains(mock->id()));
    for (uint64_t program_id : program_ids_of(silicon_workload)) {
        EXPECT_TRUE(contents.program_ids.contains(program_id)) << "silicon program " << program_id;
    }
    for (uint64_t program_id : program_ids_of(mock_workload)) {
        EXPECT_FALSE(contents.program_ids.contains(program_id)) << "mock program " << program_id;
    }
}

}  // namespace tt::tt_metal
