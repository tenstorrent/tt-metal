// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_rpc_client.hpp"

#include <capnp/ez-rpc.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "impl/jit_server/rpc.capnp.h"

namespace tt::tt_metal::jit_server {

namespace {

constexpr const char* kJitServerEndpointEnv = "TT_METAL_JIT_SERVER_ENDPOINT";
constexpr const char* kJitServerEnableEnv = "TT_METAL_JIT_SERVER_ENABLE";

void fill_target_recipe(rpc::TargetRecipe::Builder& builder, const TargetRecipe& target) {
    builder.setTargetName(target.target_name);
    builder.setCflags(target.cflags);
    builder.setDefines(target.defines);
    builder.setIncludes(target.includes);
    builder.setCompilerOptLevel(target.compiler_opt_level);
    auto srcs = builder.initSrcs(target.srcs.size());
    for (std::size_t i = 0; i < target.srcs.size(); ++i) {
        srcs.set(i, target.srcs[i]);
    }
    auto objs = builder.initObjs(target.objs.size());
    for (std::size_t i = 0; i < target.objs.size(); ++i) {
        objs.set(i, target.objs[i]);
    }
    builder.setLflags(target.lflags);
    builder.setExtraLinkObjs(target.extra_link_objs);
    builder.setLinkerScript(target.linker_script);
    builder.setWeakenedFirmwareName(target.weakened_firmware_name);
    builder.setFirmwareIsKernelObject(target.firmware_is_kernel_object);
    builder.setLinkerOptLevel(target.linker_opt_level);
}

void fill_compile_request(rpc::CompileRequest::Builder& builder, const CompileRequest& request) {
    builder.setBuildKey(request.build_key);
    builder.setKernelName(request.kernel_name);
    builder.setGpp(request.gpp);

    auto targets = builder.initTargets(request.targets.size());
    for (std::size_t i = 0; i < request.targets.size(); ++i) {
        auto t = targets[i];
        fill_target_recipe(t, request.targets[i]);
    }

    if (!request.generated_files.empty()) {
        auto files = builder.initGeneratedFiles(request.generated_files.size());
        for (std::size_t i = 0; i < request.generated_files.size(); ++i) {
            files[i].setName(request.generated_files[i].name);
            files[i].setContent(
                kj::arrayPtr(request.generated_files[i].content.data(), request.generated_files[i].content.size()));
        }
    }
}

CompileResponse read_compile_response(rpc::CompileResponse::Reader reader) {
    CompileResponse response;
    response.success = reader.getSuccess();
    response.error_message = reader.getErrorMessage().cStr();
    for (auto blob : reader.getElfBlobs()) {
        ElfBlob elf;
        elf.name = blob.getName().cStr();
        auto data = blob.getData();
        elf.data.assign(data.begin(), data.end());
        response.elf_blobs.push_back(std::move(elf));
    }
    return response;
}

}  // namespace

JitCompileRpcClient::JitCompileRpcClient(std::string endpoint) : endpoint_(std::move(endpoint)) {}

bool JitCompileRpcClient::enabled() {
    const char* enabled_value = std::getenv(kJitServerEnableEnv);
    if (enabled_value == nullptr) {
        return false;
    }
    return std::string_view(enabled_value) == "1";
}

std::string JitCompileRpcClient::endpoint_from_env() {
    const char* endpoint_value = std::getenv(kJitServerEndpointEnv);
    if (endpoint_value == nullptr) {
        return {};
    }
    return endpoint_value;
}

CompileResponse JitCompileRpcClient::compile(const CompileRequest& request) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT compile");
    }

    try {
        capnp::EzRpcClient client(endpoint_);
        auto cap = client.getMain<rpc::JitCompile>();
        auto rpc_request = cap.compileRequest();
        auto builder = rpc_request.initRequest();
        fill_compile_request(builder, request);
        auto result = rpc_request.send().wait(client.getWaitScope());
        return read_compile_response(result.getResponse());
    } catch (const kj::Exception& e) {
        throw std::runtime_error(
            "Failed to connect to remote JIT compile server at " + endpoint_ + ": " + e.getDescription().cStr());
    }
}

std::vector<CompileResponse> JitCompileRpcClient::compile_batch(const std::vector<CompileRequest>& requests) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT compile");
    }
    if (requests.empty()) {
        return {};
    }

    try {
        capnp::EzRpcClient client(endpoint_);
        auto cap = client.getMain<rpc::JitCompile>();

        using CompilePromise = capnp::RemotePromise<rpc::JitCompile::CompileResults>;
        std::vector<CompilePromise> promises;
        promises.reserve(requests.size());
        for (const auto& request : requests) {
            auto rpc_request = cap.compileRequest();
            auto builder = rpc_request.initRequest();
            fill_compile_request(builder, request);
            promises.push_back(rpc_request.send());
        }

        std::vector<CompileResponse> responses;
        responses.reserve(promises.size());
        for (auto& promise : promises) {
            auto result = promise.wait(client.getWaitScope());
            responses.push_back(read_compile_response(result.getResponse()));
        }
        return responses;
    } catch (const kj::Exception& e) {
        throw std::runtime_error(
            "Failed to connect to remote JIT compile server at " + endpoint_ + ": " + e.getDescription().cStr());
    }
}

// -- JitCompileRpcSession --

struct JitCompileRpcSession::Impl {
    capnp::EzRpcClient client;
    rpc::JitCompile::Client cap;
    using CompilePromise = capnp::RemotePromise<rpc::JitCompile::CompileResults>;
    std::vector<CompilePromise> promises;

    explicit Impl(const std::string& endpoint) : client(endpoint), cap(client.getMain<rpc::JitCompile>()) {}
};

JitCompileRpcSession::JitCompileRpcSession(const std::string& endpoint) : impl_(std::make_unique<Impl>(endpoint)) {}

JitCompileRpcSession::~JitCompileRpcSession() = default;

void JitCompileRpcSession::send(const CompileRequest& request) {
    auto rpc_request = impl_->cap.compileRequest();
    auto builder = rpc_request.initRequest();
    fill_compile_request(builder, request);
    impl_->promises.push_back(rpc_request.send());
}

std::vector<CompileResponse> JitCompileRpcSession::wait_all() {
    std::vector<CompileResponse> responses;
    responses.reserve(impl_->promises.size());
    for (auto& promise : impl_->promises) {
        auto result = promise.wait(impl_->client.getWaitScope());
        responses.push_back(read_compile_response(result.getResponse()));
    }
    impl_->promises.clear();
    return responses;
}

}  // namespace tt::tt_metal::jit_server
