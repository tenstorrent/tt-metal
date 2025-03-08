add_library(fabric OBJECT)
add_library(TT::Metalium::Fabric ALIAS fabric)

target_sources(
    fabric
    PRIVATE
        control_plane.cpp
        routing_table_generator.cpp
        mesh_graph.cpp
)

target_sources(
    fabric
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES hw/inc/tt_fabric_api.h hw/inc/tt_fabric.h hw/inc/tt_fabric_interface.h hw/inc/eth_chan_noc_mapping.h
)

target_include_directories(fabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(
    fabric
    PRIVATE
        Metalium::Metal::LLRT
        umd::device
        metal_common_libs
        magic_enum::magic_enum
        fmt::fmt-header-only
        yaml-cpp::yaml-cpp
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
)

target_precompile_headers(
    fabric
    PRIVATE
        <functional>
        <map>
        <memory>
        <unordered_map>
        <variant>
        <vector>
)

target_compile_options(fabric PRIVATE -Wno-int-to-pointer-cast)

install(
    TARGETS
        fabric
    FILE_SET
    jit_api
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/tt_metal/fabric # FIXME: fix the include paths for jit_build
        COMPONENT metalium-runtime
)
