# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

@0xf0fd8bd7829da9a2;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("tt::tt_metal::jit_server::broker_rpc");

struct KernelKey {
    buildKey @0 :UInt64;
    kernelName @1 :Text;
}

enum FirmwareState {
    present @0;
    absent @1;
}

struct Assignment {
    serverEndpoint @0 :Text;
    handle @1 :UInt64;
    firmwareState @2 :FirmwareState;
}

enum FirmwareUploadAction {
    skipAlreadyPresent @0;
    youUpload @1;
    waitForOther @2;
}

interface JitDispatchBroker {
    assign @0 (buildKey :UInt64, kernelKeys :List(Text)) -> (assignments :List(Assignment));
    claimFirmwareUpload @1 (buildKey :UInt64, serverEndpoint :Text) -> (action :FirmwareUploadAction);
    releaseFirmwareUpload @2 (buildKey :UInt64, serverEndpoint :Text, success :Bool);
    waitFirmwareReady @3 (buildKey :UInt64, serverEndpoint :Text);
    registerServer @4 (serverEndpoint :Text);
    reportCacheState @5 (
        serverEndpoint :Text,
        kernelKeys :List(KernelKey),
        firmwareBuildKeys :List(UInt64)
    );
    release @6 (handle :UInt64, kernelKey :KernelKey, wasRealCompile :Bool);
}
