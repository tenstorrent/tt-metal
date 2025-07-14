add_library(fabric OBJECT)
add_library(TT::Metalium::Fabric ALIAS fabric)

# These headers are for the device, not host; will require cross compiling to lint them (future work).
set_target_properties(
    fabric
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
target_sources(
    fabric
    PUBLIC
        FILE_SET jit_api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            hw/inc/tt_fabric_api.h
            hw/inc/tt_fabric.h
            hw/inc/tt_fabric_interface.h
            hw/inc/tt_fabric_mux.hpp
            hw/inc/tt_fabric_mux_interface.hpp
            hw/inc/fabric_routing_mode.h
            hw/inc/edm_fabric/edm_fabric_flow_control_helpers.hpp
            hw/inc/edm_fabric/edm_fabric_utils.hpp
            hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp
            hw/inc/edm_fabric/edm_handshake.hpp
            hw/inc/edm_fabric/fabric_connection_manager.hpp
            hw/inc/edm_fabric/fabric_edm_packet_header_validate.hpp
            hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp
            hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
    PRIVATE
        control_plane.cpp
        routing_table_generator.cpp
        mesh_graph.cpp
        erisc_datamover_builder.cpp
        fabric.cpp
        fabric_host_utils.cpp
        fabric_context.cpp
        serialization/intermesh_link_table.cpp
)

# Include helper functions and generate headers from flatbuffer schemas
include(flatbuffers)
set(FLATBUFFER_SCHEMAS ${CMAKE_CURRENT_SOURCE_DIR}/serialization/intermesh_link_table.fbs)

# Collect all generated headers
set(GENERATED_FBS_HEADERS)
foreach(FBS_FILE ${FLATBUFFER_SCHEMAS})
    GENERATE_FBS_HEADER(${FBS_FILE})
    list(APPEND GENERATED_FBS_HEADERS ${FBS_GENERATED_HEADER_FILE})
endforeach()

# Add the generated headers to your target
target_sources(fabric PRIVATE ${GENERATED_FBS_HEADERS})

target_include_directories(
    fabric
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/flatbuffers
)

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
        FlatBuffers::FlatBuffers
)

target_precompile_headers(fabric REUSE_FROM TT::CommonPCH)

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
