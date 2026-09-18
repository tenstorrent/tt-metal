enable_testing()
include(GoogleTest)

# Suppress narrowing warnings for tests (being cleaned up in tt_metal first)
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

add_library(test_common_libs INTERFACE)
target_include_directories(test_common_libs INTERFACE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_link_libraries(
    test_common_libs
    INTERFACE
        pthread
        gmock_main
        nlohmann_json::nlohmann_json
        enchantum::enchantum
        fmt::fmt-header-only
        Boost::algorithm
        Taskflow::Taskflow
        TT::Metalium
)

# Test-wide precompiled header: extends TT::CommonPCH's content with gtest/gmock and the
# heavy tt-metalium headers that nearly every test TU parses (host_api/distributed/
# metal_context/tt_cluster). Compiled once and reused by the test targets below in place
# of metal_common_pch, which lacks all of these.
add_library(metal_test_pch STATIC)
add_library(TT::TestPCH ALIAS metal_test_pch)
file(GENERATE OUTPUT metal_test_pch.cpp CONTENT "/*dummy test pch file*/\n")
target_sources(metal_test_pch PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/metal_test_pch.cpp)
target_precompile_headers(
    metal_test_pch
    PUBLIC
        <fmt/core.h>
        <fmt/format.h>
        <nlohmann/json.hpp>
        <algorithm>
        <cstddef>
        <cstdint>
        <functional>
        <map>
        <memory>
        <optional>
        <tuple>
        <unordered_map>
        <variant>
        <vector>
        <reflect>
        <tt_stl/reflection.hpp>
        <gtest/gtest.h>
        <gmock/gmock.h>
        <tt-metalium/host_api.hpp>
        <tt-metalium/tt_metal.hpp>
        <tt-metalium/distributed.hpp>
        <tt-metalium/mesh_device.hpp>
        ${PROJECT_SOURCE_DIR}/tt_metal/impl/context/metal_context.hpp
        ${PROJECT_SOURCE_DIR}/tt_metal/llrt/tt_cluster.hpp
)
target_link_libraries(metal_test_pch PRIVATE test_common_libs)
if(COMMAND tt_configure_ccache_for_pch)
    tt_configure_ccache_for_pch(metal_test_pch)
endif()

if(TT_METAL_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_metal)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/scale_out)
endif(TT_METAL_BUILD_TESTS)

if(TTNN_BUILD_TESTS)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tt_eager) # this should go away and be replaced with link to ttnn
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ttnn)
endif(TTNN_BUILD_TESTS)

if(TT_METAL_USE_EMULE)
    add_subdirectory(emule)
endif()
