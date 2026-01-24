// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string_view>

// Before including this file, make sure to compile capnp generated files from rpc.capnp
// and add directory of generated files to include path.
#include <tt-metalium/experimental/inspector_rpc.capnp.h>

namespace tt::tt_metal::experimental::inspector {

// Inspector feature flag
bool IsEnabled();

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
void RegisterInspectorRpcChannel(std::string_view name, tt::tt_metal::inspector::rpc::InspectorChannel::Client channel);

}  // namespace tt::tt_metal::experimental::inspector
