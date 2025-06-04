add_library(ttmetalium_hostdevcommon INTERFACE)
add_library(TT::Metalium::HostDevCommon ALIAS ttmetalium_hostdevcommon)

target_sources(
    ttmetalium_hostdevcommon
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS api
        FILES
            api/hostdevcommon/common_values.hpp
            api/hostdevcommon/dprint_common.h
            api/hostdevcommon/kernel_structs.h
            api/hostdevcommon/profiler_common.h
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS api
        FILES
            api/hostdevcommon/common_values.hpp
            api/hostdevcommon/dprint_common.h
            api/hostdevcommon/kernel_structs.h
            api/hostdevcommon/profiler_common.h
)

target_include_directories(ttmetalium_hostdevcommon INTERFACE "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>")

target_link_libraries(
    ttmetalium_hostdevcommon
    INTERFACE
        fmt::fmt-header-only
        umd::device
)

install(
    TARGETS
        ttmetalium_hostdevcommon
    EXPORT Metalium
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hostdevcommon/api # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
    FILE_SET
    api
        COMPONENT metalium-dev
)
