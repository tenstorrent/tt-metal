# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Cap'N Proto file schema: https://capnproto.org/language.html

@0xf8b5d6e3c2a19074;

using Cxx = import "/capnp/c++.capnp";
$Cxx.namespace("tt::tt_metal::inspector::rpc");

# Inspector RPC interface for querying TT-Metal runtime state

struct KernelData {
    watcherKernelId @0 :Int32;
    name @1 :Text;
    path @2 :Text;
    source @3 :Text;
    programId @4 :UInt64;
}

enum BinaryStatus {
    notSent @0;
    inFlight @1;
    committed @2;
}

struct DeviceBinaryStatus {
    metalDeviceId @0 :UInt64;
    status @1 :BinaryStatus;
}

struct ProgramData {
    programId @0 :UInt64;
    compiled @1 :Bool;
    binaryStatusPerDevice @2 :List(DeviceBinaryStatus);
    kernels @3 :List(KernelData);
}

struct MeshDeviceData {
    meshId @0 :Int64;
    devices @1 :List(UInt32);
    shape @2 :List(UInt32);
    parentMeshId @3 :Int64;  # -1 if no parent
    initialized @4 :Bool;
}

struct MeshCoordinate {
    coordinates @0 :List(Int32);
}

struct MeshWorkloadProgramData {
    programId @0 :UInt64;
    coordinates @1 :List(MeshCoordinate);
}

struct MeshDeviceBinaryStatus {
    meshId @0 :UInt64;
    status @1 :BinaryStatus;
}

struct MeshWorkloadData {
    meshWorkloadId @0 :UInt64;
    programs @1 :List(MeshWorkloadProgramData);
    binaryStatusPerMeshDevice @2 :List(MeshDeviceBinaryStatus);
    # High-level operation metadata
    # Empty if workload was not created by a tracked operation
    name @3 :Text;        # Operation name
    parameters @4 :Text;  # Operation parameters
}

struct MeshWorkloadRuntimeIdEntry {
    workloadId @0 :UInt64;
    runtimeId @1 :UInt64;
}

# Build environment info for a specific device
# Used to get correct firmware path for each device and build config,
# enabling correct firmware path resolution without relying on relative
# paths
struct BuildEnvData {
    buildKey @0 :UInt64; # Unique identifier for the build configuration
    firmwarePath @1 :Text; # Absolute path to the firmware directory for this device
    fwCompileHash @2 :UInt64; # Hash of the firmware compilation settings
}

struct BuildEnvPerDevice {
    metalDeviceId @0 :UInt64;
    buildInfo @1 :BuildEnvData;
}

# Per Dispatch Core Information
struct CoreInfo {
    workType @0: Text;
    metalDeviceId @1: Int32;
    servicingMetalDeviceId @2: Int32;
    eventID @3: UInt32;
    cqId @4: UInt8;
}

# Virtual core coordinates are used as a unique key to fetch dispatch/prefetch core information
# Same as tt_cxy_pair
struct VirtualCore {
    chip @0: UInt64;
    x @1: UInt64;
    y @2: UInt64;
}

# Per entry information of a dispatch/dispatch_s/prefetch core
# Contains virtual coordinates and associated information
struct CoreEntry {
  key  @0 :VirtualCore;       # chip,x,y (virtual)
  info @1 :CoreInfo;  # metalDeviceId, servicingMetalDeviceId, workType, cqId
}

# Simplified core type enum for grouping cores by their storage category
# Maps to the three internal data structures: dispatch_core_info,
# dispatch_s_core_info, prefetch_core_info
enum CoreCategory {
    prefetch @0;
    dispatch @1;
    dispatchS @2;
    count @3; # Sentinel value representing the total count of CoreCategory enum values
}

# Struct to group core entries by category
struct CoreEntriesByCategory {
    category @0 :CoreCategory;
    entries @1 :List(CoreEntry);
}

# Mapping from logical metal ID to unique ID
struct MetalDeviceIdToUniqueId {
    metalDeviceId @0 :UInt64;
    uniqueId @1 :UInt64;
}

interface Inspector {
    # Get programs currently alive
    getPrograms @0 () -> (programs :List(ProgramData));

    # Get mesh devices currently alive
    getMeshDevices @1 () -> (meshDevices :List(MeshDeviceData));

    # Get mesh workloads currently alive
    getMeshWorkloads @2 () -> (meshWorkloads :List(MeshWorkloadData));

    # Get list of local devices that are being used by this Metal runtime
    getDevicesInUse @3 () -> (metalDeviceIds :List(UInt64));

    # Search for a kernel
    getKernel @4 (watcherKernelId :Int32) -> (kernel :KernelData);

    # Get build environment information for all devices
    # Returns device-specific firmware paths and build configuration.
    # This replaces the old approach of constructing relative paths,
    # providing correct firmware locations for each device
    getAllBuildEnvs @5 () -> (buildEnvs :List(BuildEnvPerDevice));

    # Get all core Info
    getAllDispatchCoreInfos @6 () -> (coresByCategory :List(CoreEntriesByCategory));

    # Get mapping from metal device ID to unique ID for all devices
    getMetalDeviceIdMappings @7 () -> (mappings :List(MetalDeviceIdToUniqueId));

    # Get runtime IDs for mesh workloads
    getMeshWorkloadsRuntimeIds @8 () -> (runtimeIds :List(MeshWorkloadRuntimeIdEntry));
}
