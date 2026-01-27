// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rpc_server.hpp"
#include <kj/async-io.h>

namespace tt::tt_metal::inspector {

void RpcServer::registerChannel(std::string name, rpc::InspectorChannel::Client channelCap) {
    std::lock_guard<std::mutex> lock(channels_mutex);
    channels.emplace(std::move(name), kj::mv(channelCap));
}

kj::Promise<void> RpcServer::getChannel(rpc::InspectorChannelRegistry::Server::GetChannelContext context) {
    auto name = context.getParams().getName();
    std::lock_guard<std::mutex> lock(channels_mutex);
    auto it = channels.find(name);
    if (it == channels.end()) {
        // Return null or throw; here, return null.
        context.getResults().setChannel(nullptr);
    } else {
        context.getResults().setChannel(it->second);
    }
    return kj::READY_NOW;
}

kj::Promise<void> RpcServer::getChannelNames(rpc::InspectorChannelRegistry::Server::GetChannelNamesContext context) {
    std::lock_guard<std::mutex> lock(channels_mutex);
    auto namesBuilder = context.getResults().initNames(channels.size());
    size_t index = 0;
    for (const auto& pair : channels) {
        namesBuilder.set(index++, pair.first);
    }
    return kj::READY_NOW;
}

void RpcServer::serialize(const std::filesystem::path& directory) {
    std::lock_guard<std::mutex> lock(channels_mutex);
    for (auto& pair : channels) {
        auto channel_name = pair.first;
        auto& channel = pair.second;
        auto channel_directory = directory / channel_name;
        channel_directory.make_preferred();
        auto request = channel.serializeRpcRequest();
        request.setPath(channel_directory.string());
        auto io = ::kj::setupAsyncIo();
        auto& waitScope = io.waitScope;
        auto response = request.send().wait(waitScope);
    }
}

}  // namespace tt::tt_metal::inspector
