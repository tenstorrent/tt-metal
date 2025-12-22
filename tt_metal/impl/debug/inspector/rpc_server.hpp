// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mutex>
#include <map>
#include <filesystem>
#include <capnp/ez-rpc.h>
#include "tt-metalium/experimental/inspector_rpc.capnp.h"

namespace tt::tt_metal::inspector {

class RpcServer final : public rpc::InspectorChannelRegistry::Server {
public:
    RpcServer() = default;
    ~RpcServer() = default;

    void serialize(const std::filesystem::path& directory);

    ::kj::Promise<void> getChannel(rpc::InspectorChannelRegistry::Server::GetChannelContext context) override;
    ::kj::Promise<void> getChannelNames(rpc::InspectorChannelRegistry::Server::GetChannelNamesContext context) override;

    void registerChannel(std::string name, rpc::InspectorChannel::Client channelCap);

private:
    std::map<std::string, rpc::InspectorChannel::Client> channels;
    std::mutex channels_mutex;
};

}  // namespace tt::tt_metal::inspector
