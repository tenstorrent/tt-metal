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

struct PrepareGenfilesRequest {
    buildKey @0 :UInt64;
    kernelName @1 :Text;
    files @2 :List(GeneratedFile);
}

struct PrepareGenfilesResponse {
    success @0 :Bool;
    errorMessage @1 :Text;
}

struct CompileRequest {
    buildKey @0 :UInt64;
    kernelName @1 :Text;
    targetName @2 :Text;

    # Toolchain.
    gpp @3 :Text;

    # Compile recipe.
    cflags @4 :Text;
    defines @5 :Text;
    includes @6 :Text;
    compilerOptLevel @7 :Text;
    srcs @8 :List(Text);
    objs @9 :List(Text);

    # Link recipe.
    lflags @10 :Text;
    extraLinkObjs @11 :Text;
    linkerScript @12 :Text;
    weakenedFirmwareName @13 :Text;
    firmwareIsKernelObject @14 :Bool;
    linkerOptLevel @15 :Text;

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

interface JitCompile {
    prepareGenfiles @0 (request :PrepareGenfilesRequest) -> (response :PrepareGenfilesResponse);
    compile @1 (request :CompileRequest) -> (response :CompileResponse);
}
