add_library(fabric_lite OBJECT)
add_library(TT::Metalium::FabricLite ALIAS fabric_lite)

# These headers are for the device, not host; will require cross compiling to lint them (future work).
set_target_properties(
    fabric_lite
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
target_sources(
    fabric_lite
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            hw/inc/channel_util.hpp
            hw/inc/channels.hpp
            hw/inc/constants.hpp
            hw/inc/fl_dev_mem_map.hpp
            hw/inc/header.hpp
            hw/inc/host_interface.hpp
            hw/inc/init-fsm-basic.hpp
            hw/inc/risc_interface.hpp
            hw/inc/types.hpp
    PRIVATE
        build.cpp
        host_util.cpp
        host_interface.cpp
)

target_include_directories(fabric_lite PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    fabric_lite
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

target_precompile_headers(fabric_lite REUSE_FROM TT::CommonPCH)

target_compile_options(fabric_lite PRIVATE -Wno-int-to-pointer-cast)

install(
    TARGETS
        fabric_lite
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/fabric_lite # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
)
