add_library(lite_fabric OBJECT)
add_library(TT::Metalium::FabricLite ALIAS lite_fabric)

# These headers are for the device, not host; will require cross compiling to lint them (future work).
set_target_properties(
    lite_fabric
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
target_sources(
    lite_fabric
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            hw/inc/channel_util.hpp
            hw/inc/channels.hpp
            hw/inc/constants.hpp
            hw/inc/lf_dev_mem_map.hpp
            hw/inc/header.hpp
            hw/inc/host_interface.hpp
            hw/inc/init-fsm-basic.hpp
            hw/inc/risc_interface.hpp
            hw/inc/types.hpp
            hw/inc/kernel_api.hpp
    PRIVATE
        build.cpp
        host_util.cpp
        host_interface.cpp
)

target_include_directories(lite_fabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    lite_fabric
    PRIVATE
        TT::Metalium::Fabric
        Metalium::Metal::LLRT
        umd::device
        metal_common_libs
        enchantum::enchantum
        fmt::fmt-header-only
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
        FlatBuffers::FlatBuffers
)

target_precompile_headers(lite_fabric REUSE_FROM TT::CommonPCH)

target_compile_options(lite_fabric PRIVATE -Wno-int-to-pointer-cast)

install(
    TARGETS
        lite_fabric
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/lite_fabric # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
)
