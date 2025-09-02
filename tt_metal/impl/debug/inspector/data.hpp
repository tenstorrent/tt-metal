// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "impl/debug/inspector/logger.hpp"
#include "impl/debug/inspector/rpc_server_controller.hpp"

namespace tt::tt_metal::inspector {

class Data {
public:
    ~Data();

private:
    Data(); // NOLINT - False alarm, tt::tt_metal::Inspector is calling this constructor.

    void serialize();
    RpcServer& get_rpc_server();
    void rpc_get_programs(rpc::Inspector::GetProgramsResults::Builder& results);
    void rpc_get_mesh_devices(rpc::Inspector::GetMeshDevicesResults::Builder& results);
    void rpc_get_mesh_workloads(rpc::Inspector::GetMeshWorkloadsResults::Builder& results);
    void rpc_get_devices_in_use(rpc::Inspector::GetDevicesInUseResults::Builder& results);
    void rpc_get_kernel(rpc::Inspector::GetKernelParams::Reader params, rpc::Inspector::GetKernelResults::Builder results);

    static rpc::BinaryStatus convert_binary_status(ProgramBinaryStatus status);

    inspector::Logger logger;
    RpcServerController rpc_server_controller;
    std::mutex programs_mutex;
    std::mutex mesh_devices_mutex;
    std::mutex mesh_workloads_mutex;
    std::unordered_map<uint64_t, inspector::ProgramData> programs_data{};
    std::unordered_map<int, uint64_t> kernel_id_to_program_id{};
    std::unordered_map<int, inspector::MeshDeviceData> mesh_devices_data{};
    std::unordered_map<uint64_t, inspector::MeshWorkloadData> mesh_workloads_data{};

    friend class tt::tt_metal::Inspector;
};

}  // namespace tt::tt_metal::inspector
