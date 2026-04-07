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
    defines @2 :Text;
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
