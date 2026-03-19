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

JitCompileService::JitCompileService(CompileCallback compile_callback) :
    compile_callback_(std::move(compile_callback)) {}

std::string JitCompileService::make_dedup_key(const CompileRequest& request) const {
    return std::to_string(request.build_key) + ":" + request.kernel_name;
}

kj::Promise<void> JitCompileService::compile(CompileContext context) {
    CompileRequest request;
    auto reader = context.getParams().getRequest();
    request.build_key = reader.getBuildKey();
    request.kernel_name = reader.getKernelName().cStr();
    request.gpp = reader.getGpp().cStr();
    for (auto target : reader.getTargets()) {
        TargetRecipe t;
        t.target_name = target.getTargetName().cStr();
        t.cflags = target.getCflags().cStr();
        t.defines = target.getDefines().cStr();
        t.includes = target.getIncludes().cStr();
        t.compiler_opt_level = target.getCompilerOptLevel().cStr();
        for (auto src : target.getSrcs()) {
            t.srcs.push_back(src.cStr());
        }
        for (auto obj : target.getObjs()) {
            t.objs.push_back(obj.cStr());
        }
        t.lflags = target.getLflags().cStr();
        t.extra_link_objs = target.getExtraLinkObjs().cStr();
        t.linker_script = target.getLinkerScript().cStr();
        t.weakened_firmware_name = target.getWeakenedFirmwareName().cStr();
        t.firmware_is_kernel_object = target.getFirmwareIsKernelObject();
        t.linker_opt_level = target.getLinkerOptLevel().cStr();
        request.targets.push_back(std::move(t));
    }
    for (auto file : reader.getGeneratedFiles()) {
        GeneratedFile gf;
        gf.name = file.getName().cStr();
        auto content = file.getContent();
        gf.content.assign(content.begin(), content.end());
        request.generated_files.push_back(std::move(gf));
    }

    const std::string dedup_key = make_dedup_key(request);

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
