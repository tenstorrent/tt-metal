project(scaleout_tools)

add_executable(
    run_cluster_validation
    validation/run_cluster_validation.cpp
    validation/utils/ethernet_link_metrics_serialization.cpp
    validation/utils/cluster_validation_utils.cpp
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
            connector/connector.hpp
            factory_system_descriptor/utils.hpp
            node/node_types.hpp
    PRIVATE
        node/node.cpp
        node/node_types.cpp
        board/board.cpp
        cabling_generator/cabling_generator.cpp
        connector/connector.cpp
        factory_system_descriptor/utils.cpp
        ${GENERATED_PROTO_SRCS}
)

target_include_directories(scaleout_tools PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(scaleout_tools SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_include_directories(run_cluster_validation PRIVATE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_include_directories(run_cluster_validation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(run_cluster_validation SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(run_cluster_validation SYSTEM PRIVATE ${CMAKE_BINARY_DIR}/tt_metal/fabric)

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

target_link_libraries(
    run_cluster_validation
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
