// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_service.hpp"

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include <kj/async.h>
#include <tt-logger/tt-logger.hpp>

namespace tt::tt_metal::jit_server {

JitCompileService::JitCompileService(
    CompileCallback compile_callback, PrepareGenfilesCallback prepare_genfiles_callback) :
    compile_callback_(std::move(compile_callback)), prepare_genfiles_callback_(std::move(prepare_genfiles_callback)) {}

std::string JitCompileService::make_compile_dedup_key(const CompileRequest& request) const {
    return std::to_string(request.build_key) + ":" + request.kernel_name + ":" + request.target_name;
}

kj::Promise<void> JitCompileService::prepareGenfiles(PrepareGenfilesContext context) {
    PrepareGenfilesRequest request;
    auto reader = context.getParams().getRequest();
    request.build_key = reader.getBuildKey();
    request.kernel_name = reader.getKernelName().cStr();
    for (auto file : reader.getFiles()) {
        GeneratedFile gf;
        gf.name = file.getName().cStr();
        auto content = file.getContent();
        gf.content.assign(content.begin(), content.end());
        request.files.push_back(std::move(gf));
    }

    const std::string dedup_key = std::to_string(request.build_key) + ":" + request.kernel_name;

    auto paf = kj::newPromiseAndCrossThreadFulfiller<PrepareGenfilesResponse>();
    auto fulfiller =
        std::make_shared<kj::Own<kj::CrossThreadPromiseFulfiller<PrepareGenfilesResponse>>>(kj::mv(paf.fulfiller));

    thread_pool_.silent_async([this, dedup_key, request = std::move(request), fulfiller]() mutable {
        PrepareGenfilesResponse response;
        try {
            response = genfiles_deduper_.run(dedup_key, [&]() {
                if (!prepare_genfiles_callback_) {
                    throw std::runtime_error("No prepareGenfiles callback configured on server");
                }
                return prepare_genfiles_callback_(request);
            });
        } catch (const std::exception& e) {
            response.success = false;
            response.error_message = e.what();
        }
        (*fulfiller)->fulfill(kj::mv(response));
    });

    return paf.promise.then([context](PrepareGenfilesResponse response) mutable {
        auto response_builder = context.getResults().initResponse();
        response_builder.setSuccess(response.success);
        response_builder.setErrorMessage(response.error_message);
    });
}

kj::Promise<void> JitCompileService::compile(CompileContext context) {
    CompileRequest request;
    auto reader = context.getParams().getRequest();
    request.build_key = reader.getBuildKey();
    request.kernel_name = reader.getKernelName().cStr();
    request.target_name = reader.getTargetName().cStr();
    request.gpp = reader.getGpp().cStr();
    request.cflags = reader.getCflags().cStr();
    request.defines = reader.getDefines().cStr();
    request.includes = reader.getIncludes().cStr();
    request.compiler_opt_level = reader.getCompilerOptLevel().cStr();
    for (auto src : reader.getSrcs()) {
        request.srcs.push_back(src.cStr());
    }
    for (auto obj : reader.getObjs()) {
        request.objs.push_back(obj.cStr());
    }
    request.lflags = reader.getLflags().cStr();
    request.extra_link_objs = reader.getExtraLinkObjs().cStr();
    request.linker_script = reader.getLinkerScript().cStr();
    request.weakened_firmware_name = reader.getWeakenedFirmwareName().cStr();
    request.firmware_is_kernel_object = reader.getFirmwareIsKernelObject();
    request.linker_opt_level = reader.getLinkerOptLevel().cStr();

    const std::string dedup_key = make_compile_dedup_key(request);

    auto paf = kj::newPromiseAndCrossThreadFulfiller<CompileResponse>();
    auto fulfiller = std::make_shared<kj::Own<kj::CrossThreadPromiseFulfiller<CompileResponse>>>(kj::mv(paf.fulfiller));

    thread_pool_.silent_async([this, dedup_key, request = std::move(request), fulfiller]() mutable {
        CompileResponse response;
        try {
            response = compile_deduper_.run(dedup_key, [&]() {
                if (!compile_callback_) {
                    throw std::runtime_error("No JIT compile callback configured on server");
                }
                return compile_callback_(request);
            });
        } catch (const std::exception& e) {
            response.success = false;
            response.error_message = e.what();
        }
        (*fulfiller)->fulfill(kj::mv(response));
    });

    return paf.promise.then([context](CompileResponse response) mutable {
        if (!response.success) {
            log_warning(tt::LogMetal, "compile FAIL: {}", response.error_message);
        }

        auto response_builder = context.getResults().initResponse();
        response_builder.setSuccess(response.success);
        response_builder.setErrorMessage(response.error_message);
        auto blobs_builder = response_builder.initElfBlobs(response.elf_blobs.size());
        for (std::size_t i = 0; i < response.elf_blobs.size(); ++i) {
            blobs_builder[i].setName(response.elf_blobs[i].name);
            blobs_builder[i].setData(
                kj::arrayPtr(response.elf_blobs[i].data.data(), response.elf_blobs[i].data.size()));
        }
    });
}

}  // namespace tt::tt_metal::jit_server
