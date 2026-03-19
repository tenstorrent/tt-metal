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
    using PrepareGenfilesCallback = std::function<PrepareGenfilesResponse(const PrepareGenfilesRequest&)>;

    JitCompileService(CompileCallback compile_callback, PrepareGenfilesCallback prepare_genfiles_callback);

    kj::Promise<void> prepareGenfiles(PrepareGenfilesContext context) override;
    kj::Promise<void> compile(CompileContext context) override;

private:
    std::string make_compile_dedup_key(const CompileRequest& request) const;

    CompileCallback compile_callback_;
    PrepareGenfilesCallback prepare_genfiles_callback_;
    InFlightCompileDeduper<CompileResponse> compile_deduper_;
    InFlightCompileDeduper<PrepareGenfilesResponse> genfiles_deduper_;
    tf::Executor thread_pool_{std::thread::hardware_concurrency()};
};

}  // namespace tt::tt_metal::jit_server
