project(scaleout_tools)

add_executable(
    run_cluster_validation
    validation/run_cluster_validation.cpp
    validation/utils/ethernet_link_metrics_serialization.cpp
    validation/utils/cluster_validation_utils.cpp
)

add_executable(
    run_fabric_manager
    fabric_manager/run_fabric_manager.cpp
    fabric_manager/utils/fabric_manager_utils.cpp
)

# Library target
add_library(scaleout_tools OBJECT)
add_library(TT::ScaleoutTools ALIAS scaleout_tools)

include(protobuf)
set(PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/factory_system_descriptor/schemas/factory_system_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cabling_descriptor/schemas/node_config.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cabling_descriptor/schemas/cluster_config.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/deployment_descriptor/schemas/deployment.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/validation/schemas/ethernet_link_metrics.proto
)

set(GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# Ensure protobufs are generated before compiling the executable
add_custom_target(scaleout_generated_protos DEPENDS ${GENERATED_PROTO_SRCS})
add_dependencies(run_cluster_validation scaleout_generated_protos)

# TODO: Should refactor into an API directory
target_sources(
    scaleout_tools
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            board/board.hpp
            cabling_generator/cabling_generator.hpp
            cabling_generator/descriptor_merger.hpp
            cabling_generator/protobuf_utils.hpp
            connector/connector.hpp
            factory_system_descriptor/utils.hpp
            node/node_types.hpp
    PRIVATE
        node/node.cpp
        node/node_types.cpp
        board/board.cpp
        cabling_generator/cabling_generator.cpp
        cabling_generator/descriptor_merger.cpp
        connector/connector.cpp
        factory_system_descriptor/utils.cpp
        ${GENERATED_PROTO_SRCS}
)

target_include_directories(scaleout_tools PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(scaleout_tools SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(run_cluster_validation PRIVATE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_include_directories(run_cluster_validation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(run_cluster_validation SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(run_fabric_manager PRIVATE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_include_directories(run_fabric_manager PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(run_fabric_manager SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    scaleout_tools
    PUBLIC
        TT::STL
        umd::device
    PRIVATE
        enchantum::enchantum
        tt-logger::tt-logger
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
        $<BUILD_INTERFACE:protobuf::libprotobuf>
)

# We are adding yaml-cpp include like this to avoid including
# scaleout_tools headers because that would cause protobuf issues.
target_include_directories(
    run_cluster_validation
    SYSTEM
    PRIVATE
        $<TARGET_PROPERTY:yaml-cpp::yaml-cpp,INTERFACE_INCLUDE_DIRECTORIES>
)

target_link_libraries(
    run_cluster_validation
    PRIVATE
        enchantum::enchantum
        protobuf::libprotobuf
        tt_metal
        cxxopts::cxxopts
)

# Separate binary to get functional coverage for CI. End user fabric manager tool is productized at https://github.com/tenstorrent/tt-fabric-manager
target_link_libraries(
    run_fabric_manager
    PRIVATE
        TT::ScaleoutTools
        enchantum::enchantum
        protobuf::libprotobuf
        tt_metal
)
target_compile_features(scaleout_tools PUBLIC cxx_std_20)
target_precompile_headers(scaleout_tools REUSE_FROM TT::CommonPCH)

install(TARGETS scaleout_tools EXPORT Metalium LIBRARY COMPONENT metalium-runtime FILE_SET api COMPONENT metalium-dev)

add_executable(2d_big_mesh_cabling_gen src/2d_big_mesh_cabling_gen.cpp)

set_target_properties(
    2d_big_mesh_cabling_gen
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/tools/scaleout
)

add_dependencies(2d_big_mesh_cabling_gen scaleout_generated_protos)

target_include_directories(2d_big_mesh_cabling_gen SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    2d_big_mesh_cabling_gen
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

add_executable(run_cabling_generator src/run_cabling_generator.cpp)

set_target_properties(
    run_cabling_generator
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/tools/scaleout
)

add_dependencies(run_cabling_generator scaleout_generated_protos)

target_include_directories(run_cabling_generator SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    run_cabling_generator
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

add_executable(generate_cluster_descriptor src/generate_cluster_descriptor.cpp)

set_target_properties(
    generate_cluster_descriptor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/tools/scaleout
)

add_dependencies(generate_cluster_descriptor scaleout_generated_protos)

target_include_directories(generate_cluster_descriptor SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(
    generate_cluster_descriptor
    PRIVATE
        scaleout_tools
        protobuf::libprotobuf
        cxxopts::cxxopts
)

# Generate MGD library
add_library(generate_mgd_lib OBJECT generate_mgd/generate_mgd.cpp)

add_dependencies(
    generate_mgd_lib
    scaleout_generated_protos
    fabric
)

target_include_directories(
    generate_mgd_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/generate_mgd
)
target_include_directories(
    generate_mgd_lib
    SYSTEM
    PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_BINARY_DIR}/tt_metal/fabric
)

target_link_libraries(
    generate_mgd_lib
    PUBLIC
        scaleout_tools
        protobuf::libprotobuf
)

# Generate MGD CLI tool (standalone, links against fabric for protobuf implementation)
add_executable(generate_mgd generate_mgd/generate_mgd_main.cpp)

add_dependencies(
    generate_mgd
    scaleout_generated_protos
    fabric
)

set_target_properties(
    generate_mgd
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_BINARY_DIR}/tools/scaleout
)

target_include_directories(
    generate_mgd
    SYSTEM
    PRIVATE
        ${CMAKE_BINARY_DIR}/tools/scaleout
        ${CMAKE_BINARY_DIR}/tt_metal/fabric
)

target_sources(generate_mgd PRIVATE $<TARGET_OBJECTS:fabric>)

target_link_libraries(
    generate_mgd
    PRIVATE
        generate_mgd_lib
        scaleout_tools
        tt_metal
        protobuf::libprotobuf
        cxxopts::cxxopts
)
