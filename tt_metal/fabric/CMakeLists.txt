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
# TT::ScaleoutTopology — the topology sources grouped into their own library: MGD + PSD + PGD (proto + class),
# the topology solver, the topology mapper UTILS, and the offline FSD->PSD builder. mesh_graph + the runtime
# TopologyMapper class stay in fabric. fabric links this lib so its objects land in libtt_metal.so and its
# symbols are exported from there (default visibility) exactly as before the split — in-tree consumers keep
# reaching these symbols via tt_metal, unchanged. Linking this lib standalone (Fabric Manager without the metal
# runtime) is a follow-up once it is fully runtime-free; see TOPOLOGY_LIB_PLAN.md.
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
# Default visibility (NOT hidden): fabric links this lib into libtt_metal.so, so its symbols are exported from
# the .so exactly as they were before the topology sources were split out of fabric. In-tree consumers keep
# resolving MGD/PSD/PGD/solver symbols through libtt_metal.so (linking tt_metal) and must NOT also link this
# archive directly, or the descriptor protos would register twice ("File already exists in database"). Linking
# the archive standalone (e.g. Fabric Manager without libtt_metal) is a follow-up once the lib is fully
# runtime-free (see TOPOLOGY_LIB_PLAN.md); until then it is consumed only via libtt_metal.so.

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
