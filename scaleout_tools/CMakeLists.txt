project(scaleout_tools)

find_package(Protobuf REQUIRED)

# Protobuf schemas
set(PROTO_SCHEMAS
    scaleout_tools/factory_system_descriptor/schemas/factory_system_descriptor.proto
    scaleout_tools/cabling_descriptor/schemas/pod_config.proto
    scaleout_tools/cabling_descriptor/schemas/cluster_config.proto
    scaleout_tools/deployment_descriptor/schemas/deployment.proto
)

# Generate protobuf sources
protobuf_generate_cpp(
    PROTO_SRCS
    PROTO_HDRS
    ${PROTO_SCHEMAS}
)

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
        ${PROTO_SRCS}
        ${PROTO_HDRS}
)

target_include_directories(
    scaleout_tools
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> # Protobuf headers
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(
    scaleout_tools
    PUBLIC
        TT::STL
        umd::device
        enchantum::enchantum
        ${Protobuf_LIBRARIES}
    PRIVATE
        $<BUILD_INTERFACE:yaml-cpp::yaml-cpp>
)

target_compile_features(scaleout_tools PUBLIC cxx_std_20)
target_precompile_headers(scaleout_tools REUSE_FROM TT::CommonPCH)

install(TARGETS scaleout_tools EXPORT Metalium LIBRARY COMPONENT metalium-runtime FILE_SET api COMPONENT metalium-dev)

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()
