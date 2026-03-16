// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>
#include <functional>
#include <thread>

#include <taskflow/taskflow.hpp>

#include "impl/jit_server/in_flight_compile_deduper.hpp"
#include "impl/jit_server/rpc.capnp.h"
#include "impl/jit_server/types.hpp"

namespace tt::tt_metal::jit_server {

class JitCompileService final : public rpc::JitCompile::Server {
public:
    using CompileCallback = std::function<CompileResponse(const CompileRequest&)>;
    using UploadFirmwareCallback = std::function<UploadFirmwareResponse(const UploadFirmwareRequest&)>;

    explicit JitCompileService(CompileCallback compile_callback, UploadFirmwareCallback upload_fw_callback = {});

    kj::Promise<void> compile(CompileContext context) override;
    kj::Promise<void> uploadFirmware(UploadFirmwareContext context) override;

private:
    std::string make_dedup_key(const CompileRequest& request) const;

    CompileCallback compile_callback_;
    UploadFirmwareCallback upload_fw_callback_;
    InFlightCompileDeduper<CompileResponse> compile_deduper_;
    tf::Executor thread_pool_{std::thread::hardware_concurrency()};
};

}  // namespace tt::tt_metal::jit_server
