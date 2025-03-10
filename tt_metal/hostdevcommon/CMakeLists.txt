add_library(ttmetalium_hostdevcommon INTERFACE)
add_library(TT::Metalium::HostDevCommon ALIAS ttmetalium_hostdevcommon)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
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
    )
endif()

target_include_directories(ttmetalium_hostdevcommon INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/api)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    install(
        TARGETS
            ttmetalium_hostdevcommon
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/hostdevcommon/api # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )
endif()
