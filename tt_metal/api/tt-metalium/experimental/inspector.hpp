// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string_view>

#include <tt-metalium/mesh_workload.hpp>

// Before including this file, make sure to compile capnp generated files from rpc.capnp
// and add directory of generated files to include path.
#include <tt-metalium/experimental/inspector_rpc.capnp.h>
#include <string>

namespace tt::tt_metal::experimental::inspector {

// Inspector feature flag
bool IsEnabled();

// Inspector-only annotation for correlating higher-level execution with MeshWorkload runs.
void EmitMeshWorkloadAnnotation(
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation_name,
    std::string_view operation_parameters);

// Inspector-only runtime id for correlating workload enqueues/runs across tools.
void EmitMeshWorkloadRuntimeId(tt::tt_metal::distributed::MeshWorkload& workload, uint64_t runtime_id);

/**
 * @brief Registers a new Inspector RPC channel with the given name.
 *
 * This function registers an Inspector RPC channel, making it accessible through the
 * InspectorChannelRegistry interface. Use this function to expose a new InspectorChannel
 * to the system, allowing clients to communicate with it via RPC.
 *
 * @param name    The unique name to associate with the Inspector RPC channel.
 * @param channel The InspectorChannel::Client instance representing the RPC channel to register.
 *
 * Call this function when you want to make a new Inspector RPC channel available for use.
 */
void RegisterInspectorRpcChannel(
    const std::string& name, tt::tt_metal::inspector::rpc::InspectorChannel::Client channel);

}  // namespace tt::tt_metal::experimental::inspector
