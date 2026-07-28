// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_rpc_client.hpp"

#include <capnp/ez-rpc.h>
#include <capnp/rpc-twoparty.h>
#include <kj/async.h>
#include <kj/async-io.h>
#include <kj/time.h>
#include <kj/timer.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <utility>

#include "impl/jit_server/rpc.capnp.h"

namespace tt::tt_metal::jit_server {

namespace {

constexpr const char* kJitServerEndpointEnv = "TT_METAL_JIT_SERVER_ENDPOINT";
constexpr const char* kJitServerEndpointsEnv = "TT_METAL_JIT_SERVER_ENDPOINTS";
constexpr const char* kJitServerEnableEnv = "TT_METAL_JIT_SERVER_ENABLE";

// Per-response wait deadline. A remote compile must arrive within this window or the wait is
// abandoned as a transport failure (the connection wedged / went half-open under load). Sized
// generously so that legitimate server-side queueing under heavy concurrency never trips it —
// it only needs to be finite so a permanently lost response can't wedge the client forever.
// Override with TT_METAL_JIT_SERVER_TIMEOUT_S; set to 0 to disable (legacy unbounded wait).
constexpr const char* kJitServerTimeoutEnv = "TT_METAL_JIT_SERVER_TIMEOUT_S";
constexpr unsigned kDefaultCompileTimeoutSeconds = 240;

unsigned compile_timeout_seconds() {
    const char* value = std::getenv(kJitServerTimeoutEnv);
    if (value == nullptr || *value == '\0') {
        return kDefaultCompileTimeoutSeconds;
    }
    try {
        const long parsed = std::stol(value);
        if (parsed <= 0) {
            return 0u;  // explicit opt-out: wait forever (old behavior)
        }
        return static_cast<unsigned>(parsed);
    } catch (const std::exception&) {
        return kDefaultCompileTimeoutSeconds;
    }
}

// TCP keepalive on the client connection: a genuinely-dead (half-open) connection is detected by
// the kernel in ~idle + cnt*intvl seconds and surfaces as a disconnect — regardless of how long a
// legitimate slow response takes. This decouples dead-detection from the latency-tuned app timeout
// (which had to exceed the whole-batch latency to avoid false positives). A slow-but-alive
// connection keeps ACKing probes and is never killed.
constexpr const char* kKeepaliveEnableEnv = "TT_METAL_JIT_SERVER_KEEPALIVE";         // default on (1)
constexpr const char* kKeepaliveIdleEnv = "TT_METAL_JIT_SERVER_KEEPALIVE_IDLE_S";    // default 5
constexpr const char* kKeepaliveIntvlEnv = "TT_METAL_JIT_SERVER_KEEPALIVE_INTVL_S";  // default 2
constexpr const char* kKeepaliveCntEnv = "TT_METAL_JIT_SERVER_KEEPALIVE_CNT";        // default 3
constexpr const char* kUserTimeoutMsEnv = "TT_METAL_JIT_SERVER_USER_TIMEOUT_MS";     // default 15000

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    try {
        return static_cast<int>(std::stol(v));
    } catch (const std::exception&) {
        return fallback;
    }
}

// Connect to `endpoint` and arm TCP keepalive + user-timeout so a dead connection breaks fast.
kj::Own<kj::AsyncIoStream> connect_with_keepalive(kj::AsyncIoContext& io, const std::string& endpoint) {
    auto addr = io.provider->getNetwork().parseAddress(endpoint).wait(io.waitScope);
    auto stream = addr->connect().wait(io.waitScope);
    if (env_int(kKeepaliveEnableEnv, 1) != 0) {
        const int idle = env_int(kKeepaliveIdleEnv, 5);
        const int intvl = env_int(kKeepaliveIntvlEnv, 2);
        const int cnt = env_int(kKeepaliveCntEnv, 3);
        const int user_to = env_int(kUserTimeoutMsEnv, 15000);
        const int on = 1;
        try {
            stream->setsockopt(SOL_SOCKET, SO_KEEPALIVE, &on, sizeof(on));
            stream->setsockopt(IPPROTO_TCP, TCP_KEEPIDLE, &idle, sizeof(idle));
            stream->setsockopt(IPPROTO_TCP, TCP_KEEPINTVL, &intvl, sizeof(intvl));
            stream->setsockopt(IPPROTO_TCP, TCP_KEEPCNT, &cnt, sizeof(cnt));
#ifdef TCP_USER_TIMEOUT
            if (user_to > 0) {
                stream->setsockopt(IPPROTO_TCP, TCP_USER_TIMEOUT, &user_to, sizeof(user_to));
            }
#endif
        } catch (const kj::Exception&) {
            // Best-effort: if a sockopt isn't supported, fall back to the app-level timeout in
            // wait_all(). Never fail the connection over keepalive tuning.
        }
    }
    return stream;
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

UploadFirmwareResponse JitCompileRpcClient::upload_firmware(const UploadFirmwareRequest& request) const {
    if (endpoint_.empty()) {
        throw std::runtime_error("Missing TT_METAL_JIT_SERVER_ENDPOINT for remote JIT firmware upload");
    }

    try {
        capnp::EzRpcClient client(endpoint_);
        auto cap = client.getMain<rpc::JitCompile>();
        auto rpc_request = cap.uploadFirmwareRequest();
        auto builder = rpc_request.initRequest();
        fill_upload_firmware_request(builder, request);
        auto result = rpc_request.send().wait(client.getWaitScope());
        return read_upload_firmware_response(result.getResponse());
    } catch (const kj::Exception& e) {
        throw std::runtime_error(
            "Failed firmware upload to remote JIT server at " + endpoint_ + ": " + e.getDescription().cStr());
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
    // Manual two-party setup (not EzRpcClient) so we can arm TCP keepalive on the socket before
    // wrapping it in the RPC client — EzRpcClient hides the connection.
    kj::AsyncIoContext io;
    kj::Own<kj::AsyncIoStream> stream;
    capnp::TwoPartyClient rpc_client;
    rpc::JitCompile::Client cap;
    using CompilePromise = capnp::RemotePromise<rpc::JitCompile::CompileResults>;
    std::vector<CompilePromise> promises;

    explicit Impl(const std::string& endpoint) :
        io(kj::setupAsyncIo()),
        stream(connect_with_keepalive(io, endpoint)),
        rpc_client(*stream),
        cap(rpc_client.bootstrap().castAs<rpc::JitCompile>()) {}
};

JitCompileRpcSession::JitCompileRpcSession(const std::string& endpoint) : impl_(std::make_unique<Impl>(endpoint)) {}

JitCompileRpcSession::~JitCompileRpcSession() = default;

CompileResponse JitCompileRpcSession::send_and_wait(const CompileRequest& request) {
    auto rpc_request = impl_->cap.compileRequest();
    auto builder = rpc_request.initRequest();
    fill_compile_request(builder, request);
    auto result = rpc_request.send().wait(impl_->io.waitScope);
    return read_compile_response(result.getResponse());
}

void JitCompileRpcSession::send(const CompileRequest& request) {
    auto rpc_request = impl_->cap.compileRequest();
    auto builder = rpc_request.initRequest();
    fill_compile_request(builder, request);
    impl_->promises.push_back(rpc_request.send());
}

std::vector<CompileResponse> JitCompileRpcSession::wait_all() {
    std::vector<CompileResponse> responses;
    responses.reserve(impl_->promises.size());

    const unsigned timeout_s = compile_timeout_seconds();
    auto& wait_scope = impl_->io.waitScope;

    try {
        for (auto& promise : impl_->promises) {
            if (timeout_s == 0u) {
                // Opt-out: legacy unbounded wait.
                auto result = promise.wait(wait_scope);
                responses.push_back(read_compile_response(result.getResponse()));
                continue;
            }

            // Race the RPC response against a timer. If the timer wins (the response never
            // arrived — connection wedged / half-open), it throws and we surface a transport
            // error so the caller can fall back to a local compile instead of hanging forever.
            kj::Timer& timer = impl_->io.provider->getTimer();
            kj::Promise<capnp::Response<rpc::JitCompile::CompileResults>> rpc = kj::mv(promise);
            auto timed_out =
                timer.afterDelay(timeout_s * kj::SECONDS)
                    .then([]() -> capnp::Response<rpc::JitCompile::CompileResults> {
                        // String literal only: KJ_EXCEPTION renders non-literal args as
                        // "expr = value", so keep the seconds out of here (it's in the wrapper).
                        kj::throwFatalException(KJ_EXCEPTION(FAILED, "remote JIT compile response timed out"));
                    });
            auto result = rpc.exclusiveJoin(kj::mv(timed_out)).wait(wait_scope);
            responses.push_back(read_compile_response(result.getResponse()));
        }
    } catch (const kj::Exception& e) {
        // Timeout, disconnect, or any other transport-layer failure. Drop the rest of the
        // pipelined promises (their connection is suspect) and signal the caller to retry
        // locally. NOT a genuine compile failure — that is reported via CompileResponse.success.
        impl_->promises.clear();
        throw RemoteCompileTransportError(
            std::string("Remote JIT compile transport failure: ") + e.getDescription().cStr());
    }

    impl_->promises.clear();
    return responses;
}

UploadFirmwareResponse JitCompileRpcSession::upload_firmware(const UploadFirmwareRequest& request) {
    auto rpc_request = impl_->cap.uploadFirmwareRequest();
    auto builder = rpc_request.initRequest();
    fill_upload_firmware_request(builder, request);
    auto result = rpc_request.send().wait(impl_->io.waitScope);
    return read_upload_firmware_response(result.getResponse());
}

}  // namespace tt::tt_metal::jit_server
