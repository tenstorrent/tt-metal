add_library(tt_metal)
add_library(Metalium::Metal ALIAS tt_metal)

target_sources(
    tt_metal
    PRIVATE
        tt_metal.cpp
        graph/graph_tracking.cpp
        experimental/hal.cpp
)

target_link_libraries(
    tt_metal
    PUBLIC
        umd::device
        magic_enum
        fmt::fmt-header-only
        span
        small_vector
        TracyClient
        nlohmann_json::nlohmann_json
        TT::Metalium::HostDevCommon
        Reflect::Reflect
    PRIVATE
        Metalium::Metal::Impl
        Metalium::Metal::STL
        metal_common_libs
        profiler
        common
        jit_build
        llrt
        detail
        distributed
        HAL::grayskull
        HAL::wormhole
        HAL::blackhole
        FlatBuffers::FlatBuffers
)

# TODO(afuller): this should be self-describing modules.
#   For now just cherry-pick all the files I discovered empirally by trying to run a test.
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    target_sources(
        tt_metal
        PUBLIC
            FILE_SET jit_api
            TYPE HEADERS
            BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
            FILES
                api/tt-metalium/dev_msgs.h
                api/tt-metalium/tt_log.h
                api/tt-metalium/circular_buffer_constants.h
                api/tt-metalium/cq_commands.hpp
                soc_descriptors/grayskull_120_arch.yaml
                soc_descriptors/wormhole_b0_80_arch.yaml
                soc_descriptors/blackhole_140_arch.yaml
                core_descriptors/grayskull_120_arch.yaml
                core_descriptors/wormhole_b0_80_arch.yaml
                core_descriptors/blackhole_140_arch.yaml
                third_party/tt_llk_blackhole/common/inc/ckernel.h
                third_party/tt_llk_blackhole/common/inc/ckernel_include.h
                third_party/tt_llk_blackhole/common/inc/ckernel_defs.h
                third_party/tt_llk_blackhole/common/inc/ckernel_instr_params.h
                third_party/tt_llk_blackhole/common/inc/ckernel_addrmod.h
                third_party/tt_llk_blackhole/common/inc/ckernel_gpr_map.h
                third_party/tt_llk_blackhole/common/inc/ckernel_structs.h
                third_party/tt_llk_blackhole/common/inc/ckernel_ops.h
                third_party/tt_llk_blackhole/common/inc/ckernel_globals.h
                third_party/tt_llk_blackhole/llk_lib/llk_defs.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_include.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_defs.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_instr_params.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_addrmod.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_gpr_map.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_structs.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_ops.h
                third_party/tt_llk_wormhole_b0/common/inc/ckernel_globals.h
                third_party/tt_llk_wormhole_b0/llk_lib/llk_defs.h
                third_party/tt_llk_grayskull/common/inc/ckernel.h
                third_party/tt_llk_grayskull/common/inc/ckernel_include.h
                third_party/tt_llk_grayskull/common/inc/ckernel_defs.h
                third_party/tt_llk_grayskull/common/inc/ckernel_instr_params.h
                third_party/tt_llk_grayskull/common/inc/ckernel_addrmod.h
                third_party/tt_llk_grayskull/common/inc/ckernel_gpr_map.h
                third_party/tt_llk_grayskull/common/inc/ckernel_structs.h
                third_party/tt_llk_grayskull/common/inc/ckernel_ops.h
                third_party/tt_llk_grayskull/common/inc/ckernel_globals.h
                third_party/tt_llk_grayskull/llk_lib/llk_defs.h
                tools/profiler/kernel_profiler.hpp
                impl/dispatch/kernels/cq_common.hpp
                impl/dispatch/kernels/cq_helpers.hpp
                impl/dispatch/kernels/packet_queue_ctrl.hpp
                impl/dispatch/kernels/packet_queue.hpp
                # Kernel sources
                impl/dispatch/kernels/cq_dispatch_slave.cpp
                impl/dispatch/kernels/cq_dispatch.cpp
                impl/dispatch/kernels/cq_prefetch.cpp
    )
endif()

target_precompile_headers(
    tt_metal
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tracy/public/tracy/Tracy.hpp
        <functional>
        <map>
        <memory>
        <unordered_map>
        <variant>
        <vector>
)

target_include_directories(
    tt_metal
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/api>"
    PRIVATE
        api/tt-metalium # FIXME: Re-home the tests and remove this
)
target_compile_options(tt_metal PUBLIC -Wno-int-to-pointer-cast)
add_dependencies(tt_metal hw_toolchain)

set_target_properties(
    tt_metal
    PROPERTIES
        INSTALL_RPATH
            "${PROJECT_BINARY_DIR}/lib"
        ADDITIONAL_CLEAN_FILES
            "${PROJECT_BINARY_DIR}/lib;${PROJECT_BINARY_DIR}/obj"
)

if(BUILD_PROGRAMMING_EXAMPLES)
    add_subdirectory(programming_examples)
endif()

# Allow internal files to access the public API "by default" and without the
# scoping that external consumers must use.  Scaoping may still be used if desired.
include_directories(
    api
    api/tt-metalium
)

add_subdirectory(hw)
add_subdirectory(hostdevcommon)
add_subdirectory(common)
add_subdirectory(jit_build)
add_subdirectory(llrt)
add_subdirectory(tools)
add_subdirectory(impl)
add_subdirectory(detail)
add_subdirectory(distributed)
add_subdirectory(tt_stl)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.23)
    install(
        TARGETS
            tt_metal
        LIBRARY
            COMPONENT metalium-runtime
        FILE_SET
        jit_api
            DESTINATION
                ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal # FIXME: fix the include paths for jit_build
            COMPONENT metalium-runtime
    )
endif()
