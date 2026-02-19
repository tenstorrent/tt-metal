# tt-train must be built as part of tt-metal, not standalone
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(
        FATAL_ERROR
        "tt-train cannot be built standalone. "
        "Build from the tt-metal root directory using: ./build_metal.sh --build-tt-train"
    )
endif()

cmake_minimum_required(VERSION 3.18...4.2)

# Add cmake directory to module path for flatbuffers helper
list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(cmake/dependencies.cmake)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ttnn header files are not clean against narrowing conversion checks
# therefore tt-train likely is not clean against them
# must disable checks to build with tt-metal as super project
add_compile_options("$<$<CXX_COMPILER_ID:Clang>:-Wno-c++11-narrowing>")
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-narrowing>")

# ttml projects

add_subdirectory(sources)
include(CTest)
enable_testing()
add_subdirectory(tests)
