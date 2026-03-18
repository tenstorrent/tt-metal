// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/jit_server/jit_compile_service.hpp"

#include <cstddef>
#include <stdexcept>
#include <utility>

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

    log_info(
        tt::LogMetal,
        "JIT server prepareGenfiles received: build_key={}, kernel={}, file_count={}",
        request.build_key,
        request.kernel_name,
        request.files.size());

    const std::string dedup_key = std::to_string(request.build_key) + ":" + request.kernel_name;
    const PrepareGenfilesResponse response = genfiles_deduper_.run(dedup_key, [&]() {
        try {
            if (!prepare_genfiles_callback_) {
                throw std::runtime_error("No prepareGenfiles callback configured on server");
            }
            return prepare_genfiles_callback_(request);
        } catch (const std::exception& e) {
            PrepareGenfilesResponse failed;
            failed.success = false;
            failed.error_message = e.what();
            log_warning(
                tt::LogMetal,
                "JIT server prepareGenfiles failed: build_key={}, kernel={}, error={}",
                request.build_key,
                request.kernel_name,
                failed.error_message);
            return failed;
        }
    });

    auto response_builder = context.getResults().initResponse();
    response_builder.setSuccess(response.success);
    response_builder.setErrorMessage(response.error_message);
    return kj::READY_NOW;
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
    log_info(
        tt::LogMetal,
        "JIT server compile received: dedup_key={}, build_key={}, kernel={}, target={}",
        dedup_key,
        request.build_key,
        request.kernel_name,
        request.target_name);

    const CompileResponse response = compile_deduper_.run(dedup_key, [&]() {
        try {
            if (!compile_callback_) {
                throw std::runtime_error("No JIT compile callback configured on server");
            }
            return compile_callback_(request);
        } catch (const std::exception& e) {
            CompileResponse failed;
            failed.success = false;
            failed.error_message = e.what();
            log_warning(
                tt::LogMetal, "JIT server compile failed: dedup_key={}, error={}", dedup_key, failed.error_message);
            return failed;
        }
    });

    if (response.success) {
        log_info(
            tt::LogMetal,
            "JIT server compile succeeded: dedup_key={}, elf_count={}",
            dedup_key,
            response.elf_blobs.size());
    } else {
        log_warning(
            tt::LogMetal,
            "JIT server compile completed with failure: dedup_key={}, error={}",
            dedup_key,
            response.error_message);
    }

    auto response_builder = context.getResults().initResponse();
    response_builder.setSuccess(response.success);
    response_builder.setErrorMessage(response.error_message);
    auto blobs_builder = response_builder.initElfBlobs(response.elf_blobs.size());
    for (std::size_t i = 0; i < response.elf_blobs.size(); ++i) {
        blobs_builder[i].setName(response.elf_blobs[i].name);
        blobs_builder[i].setData(kj::arrayPtr(response.elf_blobs[i].data.data(), response.elf_blobs[i].data.size()));
    }

    return kj::READY_NOW;
}

}  // namespace tt::tt_metal::jit_server
