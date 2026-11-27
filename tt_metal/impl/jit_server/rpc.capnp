# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

@0xdbf1fcd2b54d6e11;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("tt::tt_metal::jit_server::rpc");

struct GeneratedFile {
    name @0 :Text;
    content @1 :Data;
}

struct TargetRecipe {
    targetName @0 :Text;

    # Compile recipe.
    cflags @1 :Text;
    defines @2 :List(Text);
    includes @3 :Text;
    compilerOptLevel @4 :Text;
    srcs @5 :List(Text);
    objs @6 :List(Text);

    # Link recipe.
    lflags @7 :Text;
    extraLinkObjs @8 :Text;
    linkerScript @9 :Text;
    weakenedFirmwareName @10 :Text;
    firmwareIsKernelObject @11 :Bool;
    linkerOptLevel @12 :Text;

    # Source umbrella path; empty means no PCH. Ignored for preprocessed inputs.
    pchUmbrella @13 :Text;

    # Hash of the effective build parameters (JitBuildState::build_state_hash_). Lets an
    # out-of-process prewarm write the .build_state gate file so a cold (e.g. profiler) build_key
    # sees the prewarmed artifacts as HITs instead of forcing a recompile. 0 when not captured.
    buildStateHash @14 :UInt64;
}

struct CompileRequest {
    buildKey @0 :UInt64;
    kernelName @1 :Text;

    # Toolchain (shared by all targets).
    gpp @2 :Text;

    # Per-target compile/link recipes.
    targets @3 :List(TargetRecipe);

    # Generated files to write before compiling (shared by all targets).
    generatedFiles @4 :List(GeneratedFile);

    # Full commit of the source tree that captured this request. Filled only for kernel_prewarm
    # manifest entries (the JIT RPC leaves it empty): a tree path reused across commits keeps the
    # same build_key, so the prewarm needs the commit to tell its own entries from an older one's.
    treeCommit @5 :Text;
}

struct ElfBlob {
    name @0 :Text;
    data @1 :Data;
}

struct CompileResponse {
    success @0 :Bool;
    errorMessage @1 :Text;
    elfBlobs @2 :List(ElfBlob);
}

struct FirmwareArtifact {
    targetName @0 :Text;
    fileName @1 :Text;
    isKernelObject @2 :Bool;
    data @3 :Data;
}

struct UploadFirmwareRequest {
    buildKey @0 :UInt64;
    artifacts @1 :List(FirmwareArtifact);
}

struct UploadFirmwareResponse {
    success @0 :Bool;
    errorMessage @1 :Text;
}

interface JitCompile {
    compile @0 (request :CompileRequest) -> (response :CompileResponse);
    uploadFirmware @1 (request :UploadFirmwareRequest) -> (response :UploadFirmwareResponse);
}
