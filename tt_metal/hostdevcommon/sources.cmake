set(HOSTDEVCOMMON_JIT_API_HEADERS
    api/hostdevcommon/common_values.hpp
    api/hostdevcommon/dispatch_telemetry_types.hpp
    api/hostdevcommon/dprint_common.h
    api/hostdevcommon/fabric_mux_v2_common.h
    api/hostdevcommon/kernel_structs.h
    api/hostdevcommon/flags.hpp
    api/hostdevcommon/fabric_common.h
    api/hostdevcommon/tensor_accessor/arg_config.hpp
)

# streaming_profiler_common.h is deliberately absent: it includes hostdev/profiler_common.h, which is only on
# the include path of the `hw` target. Listing it here puts it in this INTERFACE library's verified header set,
# where each header is compiled standalone against `api` alone, and the include fails. `hw` carries it in
# HW_JIT_API_HEADERS -- installed to the same path, and exempt from verification because these are device
# headers ("will require cross compiling to verify", tt_metal/hw/CMakeLists.txt).
set(HOSTDEVCOMMON_HOSTDEV_HEADERS
    ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h
    ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/realtime_profiler_msgs.h
)
