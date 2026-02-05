add_library(fabric OBJECT)
add_library(TT::Metalium::Fabric ALIAS fabric)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(fabric)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

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
            hw/inc/edm_fabric/fabric_txq_setup.h
            hw/inc/edm_fabric/fabric_erisc_datamover_channels.hpp
            hw/inc/fabric_direction_table_interface.h
            fabric_edm_packet_header.hpp
    PRIVATE
        ${FABRIC_SOURCES}
)

# Include helper functions and generate headers from protobuf schemas
include(protobuf)
set(PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/mesh_graph_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/physical_system_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/router_port_directions.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/port_descriptor_table.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/intermesh_connection_table.proto
)

# Collect all generated protobuf files
set(GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# Exclude protobuf generated files from unity builds (they have conflicting static symbols)
set_source_files_properties(
    ${GENERATED_PROTO_SRCS}
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

# Add the generated protobuf files to your target
target_sources(fabric PRIVATE ${GENERATED_PROTO_SRCS})

target_include_directories(fabric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(fabric SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    fabric
    PRIVATE
        TT::ScaleoutTools
        Metalium::Metal::LLRT
        umd::device
        metal_common_libs
        enchantum::enchantum
        fmt::fmt-header-only
        yaml-cpp::yaml-cpp
        Metalium::Metal::Impl
        TT::Metalium::HostDevCommon
        FlatBuffers::FlatBuffers
        protobuf::libprotobuf
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

# Install proto schema files for external use
install(
    DIRECTORY
        protobuf
    DESTINATION ${CMAKE_INSTALL_DATADIR}/tt-metalium/proto
    COMPONENT metalium-dev
    FILES_MATCHING
    PATTERN
    "*.proto"
)
