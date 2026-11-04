// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_rpc_client.hpp"

#include <capnp/rpc-twoparty.h>
#include <kj/async-io.h>
#include <kj/timer.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "impl/jit_server/rpc.capnp.h"

namespace tt::tt_metal::jit_server {

namespace {

constexpr const char* kJitServerEndpointEnv = "TT_METAL_JIT_SERVER_ENDPOINT";
constexpr const char* kJitServerEndpointsEnv = "TT_METAL_JIT_SERVER_ENDPOINTS";
constexpr const char* kJitServerEnableEnv = "TT_METAL_JIT_SERVER_ENABLE";
// Seconds to wait for a connection or for each reply; 0 waits forever. A reply comes back once the server has
// compiled the kernel, so this must cover a slow compile on a busy server.
constexpr const char* kJitServerTimeoutEnv = "TT_METAL_JIT_SERVER_TIMEOUT_S";
constexpr unsigned kDefaultTimeoutSeconds = 240;

unsigned timeout_seconds() {
    static const unsigned seconds = [] {
        const char* value = std::getenv(kJitServerTimeoutEnv);
        if (value == nullptr || *value == '\0') {
            return kDefaultTimeoutSeconds;
        }
        try {
            return static_cast<unsigned>(std::max(0L, std::stol(value)));
        } catch (const std::exception&) {
            return kDefaultTimeoutSeconds;
        }
    }();
    return seconds;
}

template <typename T>
T wait_bounded(kj::Promise<T> promise, kj::AsyncIoContext& io) {
    const unsigned seconds = timeout_seconds();
    if (seconds == 0) {
        return promise.wait(io.waitScope);
    }
    return io.provider->getTimer().timeoutAfter(seconds * kj::SECONDS, kj::mv(promise)).wait(io.waitScope);
}

// kj allows one event loop per thread, so every session on a thread shares one, as EzRpcClient does.
std::shared_ptr<kj::AsyncIoContext> thread_io_context() {
    thread_local std::weak_ptr<kj::AsyncIoContext> current;
    auto io = current.lock();
    if (!io) {
        io = std::make_shared<kj::AsyncIoContext>(kj::setupAsyncIo());
        current = io;
    }
    return io;
}

// Keepalive probes find a connection whose peer vanished in about 11 s while it waits for a reply, and the user
// timeout drops one whose sent data goes unacknowledged for 15 s. A slow reply on a live connection trips neither.
void enable_keepalive(kj::AsyncIoStream& stream) {
    const int on = 1;
    const int idle_s = 5;
    const int interval_s = 2;
    const int probes = 3;
    const unsigned user_timeout_ms = 15000;
    try {
        stream.setsockopt(SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
        stream.setsockopt(IPPROTO_TCP, TCP_KEEPIDLE, &idle_s, sizeof(idle_s));
        stream.setsockopt(IPPROTO_TCP, TCP_KEEPINTVL, &interval_s, sizeof(interval_s));
        stream.setsockopt(IPPROTO_TCP, TCP_KEEPCNT, &probes, sizeof(probes));
        stream.setsockopt(IPPROTO_TCP, TCP_USER_TIMEOUT, &user_timeout_ms, sizeof(user_timeout_ms));
    } catch (const kj::Exception&) {
        // Not a TCP socket; the reply timeout still applies.
    }
}

std::string trim_ascii_whitespace(std::string_view input) {
    size_t begin = 0u;
    while (begin < input.size() && std::isspace(static_cast<unsigned char>(input[begin]))) {
        ++begin;
    }
    size_t end = input.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(input[end - 1]))) {
        --end;
    }
    return std::string(input.substr(begin, end - begin));
}

std::vector<std::string> parse_endpoint_list(std::string_view endpoints) {
    std::vector<std::string> parsed;
    size_t start = 0u;
    while (start <= endpoints.size()) {
        size_t comma = endpoints.find(',', start);
        size_t end = (comma == std::string_view::npos) ? endpoints.size() : comma;
        std::string token = trim_ascii_whitespace(endpoints.substr(start, end - start));
        if (!token.empty()) {
            parsed.push_back(std::move(token));
        }
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1u;
    }
    return parsed;
}

void fill_target_recipe(rpc::TargetRecipe::Builder& builder, const TargetRecipe& target) {
    builder.setTargetName(target.target_name);
    builder.setCflags(target.cflags);
    auto defines = builder.initDefines(target.defines.size());
    for (std::size_t i = 0; i < target.defines.size(); ++i) {
        defines.set(i, target.defines[i]);
    }
    builder.setIncludes(target.includes);
    builder.setPchUmbrella(target.pch_umbrella);
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
    builder.setClientRoot(request.client_root);

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

void fill_upload_firmware_request(rpc::UploadFirmwareRequest::Builder& builder, const UploadFirmwareRequest& request) {
    builder.setBuildKey(request.build_key);
    auto artifacts = builder.initArtifacts(request.artifacts.size());
    for (std::size_t i = 0; i < request.artifacts.size(); ++i) {
        artifacts[i].setTargetName(request.artifacts[i].target_name);
        artifacts[i].setFileName(request.artifacts[i].file_name);
        artifacts[i].setIsKernelObject(request.artifacts[i].is_kernel_object);
        artifacts[i].setData(kj::arrayPtr(request.artifacts[i].data.data(), request.artifacts[i].data.size()));
    }
}

UploadFirmwareResponse read_upload_firmware_response(rpc::UploadFirmwareResponse::Reader reader) {
    UploadFirmwareResponse response;
    response.success = reader.getSuccess();
    response.error_message = reader.getErrorMessage().cStr();
    return response;
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

std::vector<std::string> JitCompileRpcClient::endpoints_from_env() {
    const char* endpoints_value = std::getenv(kJitServerEndpointsEnv);
    if (endpoints_value != nullptr) {
        auto parsed = parse_endpoint_list(endpoints_value);
        if (!parsed.empty()) {
            return parsed;
        }
    }

    const std::string endpoint = endpoint_from_env();
    if (!endpoint.empty()) {
        return {endpoint};
    }
    return {};
}

CompileResponse JitCompileRpcClient::compile(const CompileRequest& request) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT compile");
    }
    JitCompileRpcSession session(endpoint_);
    return session.send_and_wait(request);
}

UploadFirmwareResponse JitCompileRpcClient::upload_firmware(const UploadFirmwareRequest& request) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT firmware upload");
    }
    JitCompileRpcSession session(endpoint_);
    return session.upload_firmware(request);
}

std::vector<CompileResponse> JitCompileRpcClient::compile_batch(const std::vector<CompileRequest>& requests) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT compile");
    }
    if (requests.empty()) {
        return {};
    }
    JitCompileRpcSession session(endpoint_);
    for (const auto& request : requests) {
        session.send(request);
    }
    return session.wait_all();
}

// -- JitCompileRpcSession --

struct JitCompileRpcSession::Impl {
    // Declared first so it outlives the connection built on it.
    std::shared_ptr<kj::AsyncIoContext> io;
    std::string endpoint;
    kj::Own<kj::AsyncIoStream> stream;
    kj::Own<capnp::TwoPartyClient> rpc_client;
    kj::Maybe<rpc::JitCompile::Client> cap;
    using CompilePromise = capnp::RemotePromise<rpc::JitCompile::CompileResults>;
    std::vector<CompilePromise> promises;

    explicit Impl(std::string endpoint_) : io(thread_io_context()), endpoint(std::move(endpoint_)) {
        transport([&] {
            auto address = wait_bounded(io->provider->getNetwork().parseAddress(endpoint), *io);
            stream = wait_bounded(address->connect(), *io);
            enable_keepalive(*stream);
            rpc_client = kj::heap<capnp::TwoPartyClient>(*stream);
            cap = rpc_client->bootstrap().castAs<rpc::JitCompile>();
        });
    }

    rpc::JitCompile::Client& server() { return KJ_ASSERT_NONNULL(cap); }

    // Runs fn, reporting a kj failure (refused or lost connection, timeout) as RemoteCompileTransportError.
    template <typename Fn>
    auto transport(Fn&& fn) -> decltype(fn()) {
        try {
            return fn();
        } catch (const kj::Exception& e) {
            promises.clear();
            throw RemoteCompileTransportError(
                "Remote JIT compile server at " + endpoint + " unavailable: " + e.getDescription().cStr());
        }
    }
};

JitCompileRpcSession::JitCompileRpcSession(const std::string& endpoint) : impl_(std::make_unique<Impl>(endpoint)) {}

JitCompileRpcSession::~JitCompileRpcSession() = default;

CompileResponse JitCompileRpcSession::send_and_wait(const CompileRequest& request) {
    return impl_->transport([&] {
        auto rpc_request = impl_->server().compileRequest();
        auto builder = rpc_request.initRequest();
        fill_compile_request(builder, request);
        auto result = wait_bounded<capnp::Response<rpc::JitCompile::CompileResults>>(rpc_request.send(), *impl_->io);
        return read_compile_response(result.getResponse());
    });
}

void JitCompileRpcSession::send(const CompileRequest& request) {
    auto rpc_request = impl_->server().compileRequest();
    auto builder = rpc_request.initRequest();
    fill_compile_request(builder, request);
    impl_->promises.push_back(rpc_request.send());
}

std::vector<CompileResponse> JitCompileRpcSession::wait_all() {
    // The pipelined requests are compiled concurrently, so each reply gets the full timeout from when the
    // previous one arrived.
    auto responses = impl_->transport([&] {
        std::vector<CompileResponse> collected;
        collected.reserve(impl_->promises.size());
        for (auto& promise : impl_->promises) {
            auto result = wait_bounded<capnp::Response<rpc::JitCompile::CompileResults>>(kj::mv(promise), *impl_->io);
            collected.push_back(read_compile_response(result.getResponse()));
        }
        return collected;
    });
    impl_->promises.clear();
    return responses;
}

UploadFirmwareResponse JitCompileRpcSession::upload_firmware(const UploadFirmwareRequest& request) {
    return impl_->transport([&] {
        auto rpc_request = impl_->server().uploadFirmwareRequest();
        auto builder = rpc_request.initRequest();
        fill_upload_firmware_request(builder, request);
        auto result =
            wait_bounded<capnp::Response<rpc::JitCompile::UploadFirmwareResults>>(rpc_request.send(), *impl_->io);
        return read_upload_firmware_response(result.getResponse());
    });
}

}  // namespace tt::tt_metal::jit_server
