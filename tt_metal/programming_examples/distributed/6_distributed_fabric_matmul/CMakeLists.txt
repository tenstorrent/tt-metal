cmake_minimum_required(VERSION 3.22...3.30)
project(fabric_matmul LANGUAGES CXX)

add_executable(fabric_matmul)
file(GLOB_RECURSE kernels kernels/*)
target_sources(
    fabric_matmul
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        FILES ${kernels}
    PRIVATE
        fabric_matmul.cpp
)
target_link_libraries(
    fabric_matmul
    PRIVATE
        TT::Metalium
        TTNN::CPP
)
set_target_properties(
    fabric_matmul
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributed
)
