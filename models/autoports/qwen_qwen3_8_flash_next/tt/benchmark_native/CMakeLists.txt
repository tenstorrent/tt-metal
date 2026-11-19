cmake_minimum_required(VERSION 3.18)
project(qwen_benchmark_event_query LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
find_package(
    Python
    3.12
    COMPONENTS
        Interpreter
        Development.Module
    REQUIRED
)
find_package(nanobind CONFIG REQUIRED)
if(NOT TT_METAL_ROOT)
    message(FATAL_ERROR "Set TT_METAL_ROOT to the built checkout")
endif()
# Match the loaded TTNN module's non-limited nanobind ABI. Stable-ABI modules
# have a separate nanobind registry and cannot accept TTNN MeshEvent objects.
nanobind_add_module(_benchmark_event_query event_query.cpp)
target_include_directories(_benchmark_event_query PRIVATE "${TT_METAL_ROOT}/build/include")
target_compile_definitions(
    _benchmark_event_query
    PRIVATE
        SPDLOG_FMT_EXTERNAL
        FMT_HEADER_ONLY=1
)
target_link_libraries(_benchmark_event_query PRIVATE "${TT_METAL_ROOT}/build/lib/libtt_metal.so")
set_target_properties(
    _benchmark_event_query
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY
            "${CMAKE_CURRENT_SOURCE_DIR}/.."
        BUILD_RPATH
            "${TT_METAL_ROOT}/build/lib"
)
