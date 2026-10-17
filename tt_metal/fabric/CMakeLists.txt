include(sources.cmake)

add_library(fabric OBJECT)
add_library(TT::Metalium::Fabric ALIAS fabric)

# emule: append_fabric_connection_rt_args records each worker's fwd/bwd connection->neighbor for the
# software-emulation teleport's 1D dst resolution (the firmware connection table emule skips). The define
# is PRIVATE on the tt_metal target, which doesn't reach this OBJECT lib, so set it here too.
if(TT_METAL_USE_EMULE)
    target_compile_definitions(fabric PRIVATE TT_METAL_USE_EMULE=1)
endif()

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
# Shared value types (FabricNodeId & friends, PhysicalNodeId) are compiled into TT::Metalium::Types, declared in
# tt_metal/CMakeLists.txt.
target_sources(tt_metalium_types PRIVATE ${FABRIC_TYPES_SOURCES})

# Include helper functions and generate headers from protobuf schemas
include(protobuf)

# Descriptor protos consumed by the runtime-free topology lib vs. fabric-runtime protos. All generate into the
# same protobuf/ dir so consumers resolve headers via ${CMAKE_CURRENT_BINARY_DIR}.
set(FABRIC_TOPOLOGY_UTILS_PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/mesh_graph_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/physical_system_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/physical_grouping_descriptor.proto
)
set(FABRIC_PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/router_port_directions.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/port_descriptor_table.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/intermesh_connection_table.proto
)

set(FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${FABRIC_TOPOLOGY_UTILS_PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

set(FABRIC_GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${FABRIC_PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND FABRIC_GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# ---------------------------------------------------------------------------
# TT::FabricTopologyUtils: runtime-free topology library (MGD/PSD/PGD + protos, topology solver, mapper utils,
# FSD->PSD builder). A shared library: libtt_metal.so links it, and an external consumer (Fabric Manager) links it
# from the installed package without libtt_metal. protobuf, yaml-cpp, cadical and scaleout_tools are linked in
# privately and stay out of the exported interface, as they do for libtt_metal.so.
# ---------------------------------------------------------------------------
add_library(
    tt_fabric_topology_utils
    SHARED
    ${FABRIC_TOPOLOGY_UTILS_SOURCES}
    ${FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS}
)
add_library(TT::FabricTopologyUtils ALIAS tt_fabric_topology_utils)
set_target_properties(
    tt_fabric_topology_utils
    PROPERTIES
        EXPORT_NAME
            FabricTopologyUtils
        OUTPUT_NAME
            tt-fabric-topology-utils
        INSTALL_RPATH
            "$ORIGIN"
)
# Generated protos have conflicting static symbols, so they stay out of the unity batch.
set_source_files_properties(
    ${FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS}
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

add_custom_target(fabric_topology_utils_generated_protos DEPENDS ${FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS})
add_dependencies(
    tt_fabric_topology_utils
    fabric_topology_utils_generated_protos
    scaleout_generated_protos
)
if(TARGET all_generated_files)
    add_dependencies(all_generated_files fabric_topology_utils_generated_protos)
endif()

target_include_directories(
    tt_fabric_topology_utils
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tt_metal/api>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tools/scaleout # board/board.hpp
    SYSTEM
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} # generated proto headers
        ${Metalium_BINARY_DIR}/tools/scaleout # FSD proto header
)
target_link_libraries(
    tt_fabric_topology_utils
    PUBLIC
        TT::Metalium::Types
        TT::STL
        fmt::fmt-header-only
        umd::tt-umd
    PRIVATE
        TT::ScaleoutTools
        enchantum::enchantum
        tt-logger::tt-logger
        yaml-cpp::yaml-cpp
        protobuf::libprotobuf
        cadical::cadical
)
target_compile_features(tt_fabric_topology_utils PUBLIC cxx_std_20)
# yaml-cpp and cadical are implementation details: do not export their symbols. The process already has
# yaml-cpp via libtt-umd.so, and a second exported copy trips ASan's ODR check.
target_link_options(tt_fabric_topology_utils PRIVATE "LINKER:--exclude-libs=libyaml-cpp.a:libcadical.a")
TT_ENABLE_UNITY_BUILD(tt_fabric_topology_utils)
tt_reuse_precompile_headers(tt_fabric_topology_utils TT::CommonPCH)
install(TARGETS tt_fabric_topology_utils EXPORT Metalium LIBRARY COMPONENT metalium-runtime)

# --- fabric: remaining runtime protos + link the topology lib -------
add_custom_target(fabric_generated_protos DEPENDS ${FABRIC_GENERATED_PROTO_SRCS})
add_dependencies(fabric fabric_generated_protos)
if(TARGET all_generated_files)
    add_dependencies(all_generated_files fabric_generated_protos)
endif()

set_source_files_properties(
    ${FABRIC_GENERATED_PROTO_SRCS}
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

target_sources(fabric PRIVATE ${FABRIC_GENERATED_PROTO_SRCS})

target_include_directories(fabric PRIVATE .)
target_include_directories(
    fabric
    SYSTEM
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}
        ${Metalium_BINARY_DIR}/tools/scaleout # FSD proto header for physical_descriptor_builder.cpp
)

target_link_libraries(
    fabric
    PRIVATE
        TT::FabricTopologyUtils
        TT::ScaleoutTools
        Metalium::Metal::LLRT
        umd::tt-umd
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

add_subdirectory(experimental/x280)
