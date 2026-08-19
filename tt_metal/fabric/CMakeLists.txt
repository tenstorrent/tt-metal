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

# Include helper functions and generate headers from protobuf schemas
include(protobuf)

# Descriptor protos consumed by the runtime-free topology lib vs. fabric-runtime protos. All generate into the
# same protobuf/ dir so consumers resolve headers via ${CMAKE_CURRENT_BINARY_DIR}.
set(TOPOLOGY_PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/mesh_graph_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/physical_system_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/physical_grouping_descriptor.proto
)
set(FABRIC_PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/router_port_directions.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/port_descriptor_table.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/protobuf/intermesh_connection_table.proto
)

set(TOPOLOGY_GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${TOPOLOGY_PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND TOPOLOGY_GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

set(FABRIC_GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${FABRIC_PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND FABRIC_GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# ---------------------------------------------------------------------------
# TT::ScaleoutTopology — lean, runtime-free static library: MGD + PSD + PGD (proto + class), the topology solver,
# the topology mapper UTILS, and the offline FSD->PSD builder. mesh_graph + the runtime TopologyMapper class stay
# in fabric. Pulls tt-metalium headers via include-dirs only (no libtt_metal link), so FM/offline tools can link
# just this lib without the metal runtime.
# ---------------------------------------------------------------------------
add_library(
    scaleout_topology
    STATIC
    ${TOPOLOGY_SOURCES}
    ${TOPOLOGY_GENERATED_PROTO_SRCS}
)
add_library(TT::ScaleoutTopology ALIAS scaleout_topology)
set_source_files_properties(
    ${TOPOLOGY_GENERATED_PROTO_SRCS}
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

add_custom_target(topology_generated_protos DEPENDS ${TOPOLOGY_GENERATED_PROTO_SRCS})
add_dependencies(
    scaleout_topology
    topology_generated_protos
    scaleout_generated_protos
)
if(TARGET all_generated_files)
    add_dependencies(all_generated_files topology_generated_protos)
endif()

target_include_directories(
    scaleout_topology
    PRIVATE
        "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>" # tt-metalium public headers (incl physical_descriptor_builder.hpp)
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scaleout # board/board.hpp
    SYSTEM
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} # this lib's generated proto headers
        ${Metalium_BINARY_DIR}/tools/scaleout # scaleout-generated FSD proto header (for physical_descriptor_builder)
)
target_link_libraries(
    scaleout_topology
    PUBLIC
        TT::STL
        umd::tt-umd
    PRIVATE
        TT::ScaleoutTools
        TT::Metalium::HostDevCommon
        enchantum::enchantum
        fmt::fmt-header-only
        tt-logger::tt-logger
        yaml-cpp::yaml-cpp
        protobuf::libprotobuf
        cadical::cadical
)
target_compile_features(scaleout_topology PUBLIC cxx_std_20)
# Repo build accelerators (match fabric / scaleout_tools): unity build + shared PCH. Generated proto sources are
# already excluded from unity above via SKIP_UNITY_BUILD_INCLUSION.
TT_ENABLE_UNITY_BUILD(scaleout_topology)
tt_reuse_precompile_headers(scaleout_topology TT::CommonPCH)
# Hidden visibility (same as TT::ScaleoutTools): this lib carries protobuf message classes, so its descriptor
# symbols must be hidden to avoid cross-module "File already exists in database" aborts. Consumers reaching these
# symbols only transitively through libtt_metal.so must therefore link TT::ScaleoutTopology explicitly.
set_target_properties(
    scaleout_topology
    PROPERTIES
        CXX_VISIBILITY_PRESET
            "hidden"
        VISIBILITY_INLINES_HIDDEN
            TRUE
)

# Install the archive so external consumers (e.g. Fabric Manager) can link it. Not added to the "Metalium"
# EXPORT set: as a STATIC lib its private deps (protobuf/cadical/yaml-cpp) become link-interface requirements,
# and install(EXPORT) would need each of those in an export set too. Wiring find_package(Metalium) to
# re-find_dependency() them is a follow-up; for now consumers link the installed archive directly.
install(
    TARGETS
        scaleout_topology
    ARCHIVE
        COMPONENT metalium-dev
)

# --- fabric: remaining runtime protos + link the topology lib -------
add_custom_target(fabric_generated_protos DEPENDS ${FABRIC_GENERATED_PROTO_SRCS})
add_dependencies(
    fabric
    fabric_generated_protos
    scaleout_topology
)
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
target_include_directories(fabric SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    fabric
    PUBLIC
        TT::ScaleoutTopology
    PRIVATE
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
        cadical::cadical
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
