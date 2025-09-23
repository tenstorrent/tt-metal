project(scaleout_tools)

# Library target
add_library(scaleout_tools OBJECT)
add_library(TT::ScaleoutTools ALIAS scaleout_tools)

include(protobuf)
set(PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/factory_system_descriptor/schemas/factory_system_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cabling_descriptor/schemas/node_config.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/cabling_descriptor/schemas/cluster_config.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/deployment_descriptor/schemas/deployment.proto
)

set(GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

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

target_compile_features(scaleout_tools PUBLIC cxx_std_20)
target_precompile_headers(scaleout_tools REUSE_FROM TT::CommonPCH)

install(TARGETS scaleout_tools EXPORT Metalium LIBRARY COMPONENT metalium-runtime FILE_SET api COMPONENT metalium-dev)
