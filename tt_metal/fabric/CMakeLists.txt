include(sources.cmake)

add_library(fabric OBJECT)
add_library(TT::Metalium::Fabric ALIAS fabric)

# Enable unity builds for faster compilation
TT_ENABLE_UNITY_BUILD(fabric)

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
        FILES ${FABRIC_JIT_API_HEADERS}
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
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/physical_grouping_descriptor.proto
)

# Collect all generated protobuf files
set(GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# Expose a custom target for the generated protobuf files so that external consumers
# can depend on them directly without going through the fabric OBJECT library.
add_custom_target(fabric_generated_protos DEPENDS ${GENERATED_PROTO_SRCS})
add_dependencies(fabric fabric_generated_protos)
if(TARGET all_generated_files)
    add_dependencies(all_generated_files fabric_generated_protos)
endif()

# Exclude protobuf generated files from unity builds (they have conflicting static symbols)
set_source_files_properties(
    ${GENERATED_PROTO_SRCS}
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

# Add the generated protobuf files to your target
target_sources(fabric PRIVATE ${GENERATED_PROTO_SRCS})

target_include_directories(fabric PRIVATE .)
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

tt_reuse_precompile_headers(fabric TT::CommonPCH)

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
