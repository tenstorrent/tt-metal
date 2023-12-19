// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "impl/debug/dprint_server.hpp"

#include "tools/profiler/profiler.hpp"

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

namespace tt_metal {

void DumpDeviceProfileResults(Device *device, const Program &program) {
    const auto &all_logical_cores = program.logical_cores();
    detail::DumpDeviceProfileResults(device, program.logical_cores());
}


namespace detail {

static Profiler tt_metal_profiler = Profiler();

void DumpDeviceProfileResults(
    Device *device, const std::unordered_map<CoreType, std::vector<CoreCoord>> &logical_cores) {
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        ProfileTTMetalScope profile_this = ProfileTTMetalScope("DumpDeviceProfileResults");
        //TODO: (MO) This global is temporary need to update once the new interface is in
        if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
            Finish(GetCommandQueue(device));
        }

        TT_FATAL(DprintServerIsRunning() == false, "Debug print server is running, cannot dump device profiler data");
        auto device_id = device->id();
        tt_metal_profiler.setDeviceArchitecture(device->arch());
        if (logical_cores.find(CoreType::WORKER) != logical_cores.end()) {
            auto worker_cores_used_in_program =
                device->worker_cores_from_logical_cores(logical_cores.at(CoreType::WORKER));
            tt_metal_profiler.dumpTensixDeviceResults(device_id, worker_cores_used_in_program);
        }
        if (logical_cores.find(CoreType::ETH) != logical_cores.end()) {
            auto ethernet_cores_used_in_program =
                device->ethernet_cores_from_logical_cores(logical_cores.at(CoreType::ETH));
            tt_metal_profiler.dumpEthernetDeviceResults(device_id, ethernet_cores_used_in_program);
        }
    }
#endif
}

void SetProfilerDir(std::string output_dir){
#if defined(PROFILER)
     tt_metal_profiler.setOutputDir(output_dir);
#endif
}

void FreshProfilerHostLog(){
#if defined(PROFILER)
     tt_metal_profiler.setHostNewLogFlag(true);
#endif
}

void FreshProfilerDeviceLog(){
#if defined(PROFILER)
     tt_metal_profiler.setDeviceNewLogFlag(true);
#endif
}

ProfileTTMetalScope::ProfileTTMetalScope (const string& scopeNameArg) : scopeName(scopeNameArg){
#if defined(PROFILER)
    tt_metal_profiler.markStart(scopeName);
#endif
}

ProfileTTMetalScope::~ProfileTTMetalScope ()
{
#if defined(PROFILER)
    tt_metal_profiler.markStop(scopeName);
#endif
}

}  // namespace detail

}  // namespace tt_metal

}  // namespace tt
