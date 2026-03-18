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

PrepareGenfilesResponse JitCompileRpcClient::prepareGenfiles(const PrepareGenfilesRequest& request) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT compile");
    }

    try {
        capnp::EzRpcClient client(endpoint_);
        auto jit_compile_client = client.getMain<rpc::JitCompile>();
        auto rpc_request = jit_compile_client.prepareGenfilesRequest();
        auto builder = rpc_request.initRequest();

        builder.setBuildKey(request.build_key);
        builder.setKernelName(request.kernel_name);
        auto files = builder.initFiles(request.files.size());
        for (std::size_t i = 0; i < request.files.size(); ++i) {
            files[i].setName(request.files[i].name);
            files[i].setContent(kj::arrayPtr(request.files[i].content.data(), request.files[i].content.size()));
        }

        auto result = rpc_request.send().wait(client.getWaitScope());
        auto response_reader = result.getResponse();

        PrepareGenfilesResponse response;
        response.success = response_reader.getSuccess();
        response.error_message = response_reader.getErrorMessage().cStr();
        return response;
    } catch (const kj::Exception& e) {
        throw std::runtime_error(
            "Failed to connect to remote JIT compile server at " + endpoint_ + ": " + e.getDescription().cStr());
    }
}

CompileResponse JitCompileRpcClient::compile(const CompileRequest& request) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT compile");
    }

    try {
        capnp::EzRpcClient client(endpoint_);
        auto jit_compile_client = client.getMain<rpc::JitCompile>();
        auto compile_request = jit_compile_client.compileRequest();
        auto builder = compile_request.initRequest();

        builder.setBuildKey(request.build_key);
        builder.setKernelName(request.kernel_name);
        builder.setTargetName(request.target_name);
        builder.setGpp(request.gpp);
        builder.setCflags(request.cflags);
        builder.setDefines(request.defines);
        builder.setIncludes(request.includes);
        builder.setCompilerOptLevel(request.compiler_opt_level);

        auto srcs = builder.initSrcs(request.srcs.size());
        for (std::size_t i = 0; i < request.srcs.size(); ++i) {
            srcs.set(i, request.srcs[i]);
        }
        auto objs = builder.initObjs(request.objs.size());
        for (std::size_t i = 0; i < request.objs.size(); ++i) {
            objs.set(i, request.objs[i]);
        }

        builder.setLflags(request.lflags);
        builder.setExtraLinkObjs(request.extra_link_objs);
        builder.setLinkerScript(request.linker_script);
        builder.setWeakenedFirmwareName(request.weakened_firmware_name);
        builder.setFirmwareIsKernelObject(request.firmware_is_kernel_object);
        builder.setLinkerOptLevel(request.linker_opt_level);

        auto compile_result = compile_request.send().wait(client.getWaitScope());
        auto response_reader = compile_result.getResponse();

        CompileResponse response;
        response.success = response_reader.getSuccess();
        response.error_message = response_reader.getErrorMessage().cStr();
        for (auto blob : response_reader.getElfBlobs()) {
            ElfBlob elf;
            elf.name = blob.getName().cStr();
            auto data = blob.getData();
            elf.data.assign(data.begin(), data.end());
            response.elf_blobs.push_back(std::move(elf));
        }
        return response;
    } catch (const kj::Exception& e) {
        throw std::runtime_error(
            "Failed to connect to remote JIT compile server at " + endpoint_ + ": " + e.getDescription().cStr());
    }
}

}  // namespace tt::tt_metal::jit_server
