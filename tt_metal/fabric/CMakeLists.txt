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
# TT::FabricTopologyUtils — runtime-free topology library: MGD + PSD + PGD (proto + class), the topology solver, the
# topology mapper utils and the offline FSD->PSD builder. It links no tt_metal runtime target. fabric links it so its objects also land in libtt_metal.so for in-tree consumers; external consumers
# (e.g. Fabric Manager) can link TT::FabricTopologyUtils from the installed tt-metalium package without libtt_metal.
# ---------------------------------------------------------------------------
add_library(
    tt_fabric_topology_utils
    STATIC
    ${FABRIC_TOPOLOGY_UTILS_SOURCES}
    ${FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS}
)
add_library(TT::FabricTopologyUtils ALIAS tt_fabric_topology_utils)
target_sources(tt_metalium_types PRIVATE ${FABRIC_TYPES_SOURCES})
set_target_properties(
    tt_fabric_topology_utils
    PROPERTIES
        EXPORT_NAME
            FabricTopologyUtils # installed name == in-tree alias
        OUTPUT_NAME
            tt-fabric-topology-utils # libtt-fabric-topology-utils.a
)
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
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/tt_metal/api>" # public headers (installed by tt_metal's api file set)
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../tools/scaleout # board/board.hpp
    SYSTEM
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR} # this lib's generated proto headers
        ${Metalium_BINARY_DIR}/tools/scaleout # scaleout-generated FSD proto header (for physical_descriptor_builder)
)
# PUBLIC deps are used by the public headers and are all in the Metalium export. Private deps: cadical is bundled
# into the archive below (no consumer has it). protobuf and yaml-cpp are declared on the installed target by NAME
# (protobuf::libprotobuf / yaml-cpp::yaml-cpp): the consumer provides those two targets (its own build or packages)
# and CMake then orders them after the archive on its link line. The names are split with $<1:> so the export does
# not resolve them to this build's in-tree targets, which cannot be exported. (yaml-cpp is not bundled: a consumer
# that links its own copy would get every symbol twice.)
target_link_libraries(
    tt_fabric_topology_utils
    PUBLIC
        TT::Metalium::Types
        TT::STL
        TT::Metalium::HostDevCommon
        enchantum::enchantum
        tt-logger::tt-logger
        umd::tt-umd
    PRIVATE
        TT::ScaleoutTools
        fmt::fmt-header-only
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>$<INSTALL_INTERFACE:$<1:yaml-cpp>::$<1:yaml-cpp>>
        $<BUILD_INTERFACE:protobuf::libprotobuf>$<INSTALL_INTERFACE:$<1:protobuf>::$<1:libprotobuf>>
        $<BUILD_INTERFACE:cadical::cadical>
)
target_compile_features(tt_fabric_topology_utils PUBLIC cxx_std_20)
# Repo build accelerators (match fabric / scaleout_tools): unity build + shared PCH. Generated proto sources are
# already excluded from unity above via SKIP_UNITY_BUILD_INCLUSION.
TT_ENABLE_UNITY_BUILD(tt_fabric_topology_utils)
tt_reuse_precompile_headers(tt_fabric_topology_utils TT::CommonPCH)
# Default visibility (NOT hidden): fabric links this lib into libtt_metal.so, so its symbols are exported from the .so
# exactly as before the split. A consumer must link EITHER tt_metal OR TT::FabricTopologyUtils, never both: the
# descriptor protos would register twice ("File already exists in database").
# Bundle cadical (SAT solver) object files into the archive so an external consumer does not need to build it.
# In-tree, libtt_metal.so links libcadical.a itself; the archive members are then never pulled, so nothing is
# defined twice.
target_sources(tt_fabric_topology_utils PRIVATE $<TARGET_OBJECTS:cadical>)
install(TARGETS tt_fabric_topology_utils EXPORT Metalium ARCHIVE COMPONENT metalium-dev)
# The archive carries the compiled MGD/PSD/PGD protos, so a consumer must use these generated headers rather than
# compile the .proto schemas again (two copies register twice: "File already exists in database").
set(FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_HDRS ${FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_SRCS})
list(FILTER FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_HDRS INCLUDE REGEX "\\.pb\\.h$")
install(
    FILES
        ${FABRIC_TOPOLOGY_UTILS_GENERATED_PROTO_HDRS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt-metalium/protobuf
    COMPONENT metalium-dev
)
# physical_descriptor_builder.cpp is the only TU that reaches into TT::ScaleoutTools (FSD proto + board lookup).
# Keep it out of the unity batch so a consumer that never calls the FSD->PSD builder does not pull the FSD proto
# objects (and a second registration of that schema) into its link.
set_source_files_properties(
    physical_descriptor_builder.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            TRUE
)

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

# libtt_metal.so must carry every topology object (tools, tests and external consumers reach these symbols through
# tt_metal), so link the library's objects directly rather than relying on which archive members fabric's own
# objects happen to reference. (Whole-archive is not an option: scaleout_tools is an OBJECT library whose objects
# are embedded in the archive and also linked into tt_metal directly.)
target_link_libraries(
    fabric
    PRIVATE
        $<TARGET_OBJECTS:tt_fabric_topology_utils>
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
