add_library(ttmetalium_hostdevcommon INTERFACE)
add_library(TT::Metalium::HostDevCommon ALIAS ttmetalium_hostdevcommon)

target_sources(
    ttmetalium_hostdevcommon
    INTERFACE
        api/hostdevcommon/common_runtime_address_map.h
        api/hostdevcommon/common_values.hpp
        api/hostdevcommon/dprint_common.h
        api/hostdevcommon/kernel_structs.h
        api/hostdevcommon/profiler_common.h
)

target_include_directories(ttmetalium_hostdevcommon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/api)
