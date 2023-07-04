#include "tt_metal/host_api.hpp"
#include "llrt/tt_debug_print_server.hpp"

#include "tools/profiler/profiler.hpp"

#include "tt_metal/detail/tt_metal.hpp"

namespace tt {

namespace tt_metal {

namespace detail {

static Profiler tt_metal_profiler = Profiler();

void DumpDeviceProfileResults(Device *device, const Program &program) {
#if defined(PROFILER)
    ZoneScoped;
    if (getDeviceProfilerState())
    {
        ProfileTTMetalScope profile_this = ProfileTTMetalScope("DumpDeviceProfileResults");
        //TODO: (MO) This global is temporary need to update once the new interface is in
        if (GLOBAL_CQ) {
            Finish(*GLOBAL_CQ);
        }
        TT_ASSERT(tt_is_print_server_running() == false, "Debug print server is running, cannot dump device profiler data");
        auto worker_cores_used_in_program =\
            device->worker_cores_from_logical_cores(program.logical_cores());
        auto cluster = device->cluster();
        auto pcie_slot = device->pcie_slot();
        tt_metal_profiler.dumpDeviceResults(cluster, pcie_slot, worker_cores_used_in_program);
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
