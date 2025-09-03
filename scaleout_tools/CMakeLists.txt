project(scaleout_tools)

include(protobuf)
set(PROTO_SCHEMAS
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleout_tools/factory_system_descriptor/schemas/factory_system_descriptor.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleout_tools/cabling_descriptor/schemas/node_config.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleout_tools/cabling_descriptor/schemas/cluster_config.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/scaleout_tools/deployment_descriptor/schemas/deployment.proto
)

set(GENERATED_PROTO_SRCS)
foreach(PROTO_FILE ${PROTO_SCHEMAS})
    GENERATE_PROTO_FILES(${PROTO_FILE} OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
    list(APPEND GENERATED_PROTO_SRCS ${PROTO_SRCS})
endforeach()

# Library target
add_library(scaleout_tools OBJECT)
add_library(TT::ScaleoutTools ALIAS scaleout_tools)

target_sources(
    scaleout_tools
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            scaleout_tools/board/board.hpp
            scaleout_tools/cabling_generator/cabling_generator.hpp
            scaleout_tools/connector/connector.hpp
            scaleout_tools/factory_system_descriptor/utils.hpp
    PRIVATE
        scaleout_tools/board/board.cpp
        scaleout_tools/cabling_generator/cabling_generator.cpp
        scaleout_tools/connector/connector.cpp
        scaleout_tools/factory_system_descriptor/utils.cpp
        ${GENERATED_PROTO_SRCS}
)

target_include_directories(
    scaleout_tools
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR} # Protobuf headers
)

target_link_libraries(
    scaleout_tools
    PUBLIC
        TT::STL
        umd::device
    PRIVATE
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
        enchantum::enchantum
        protobuf::libprotobuf
)

target_compile_features(scaleout_tools PUBLIC cxx_std_20)
target_precompile_headers(scaleout_tools REUSE_FROM TT::CommonPCH)

install(TARGETS scaleout_tools EXPORT Metalium LIBRARY COMPONENT metalium-runtime FILE_SET api COMPONENT metalium-dev)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
