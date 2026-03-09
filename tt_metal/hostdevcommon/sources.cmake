set(HOSTDEVCOMMON_JIT_API_HEADERS
    api/hostdevcommon/common_values.hpp
    api/hostdevcommon/dprint_common.h
    api/hostdevcommon/kernel_structs.h
    api/hostdevcommon/profiler_common.h
    api/hostdevcommon/flags.hpp
    api/hostdevcommon/fabric_common.h
    api/hostdevcommon/tensor_accessor/arg_config.hpp
)

set(HOSTDEVCOMMON_HOSTDEV_HEADERS ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/hostdev/fabric_telemetry_msgs.h)
