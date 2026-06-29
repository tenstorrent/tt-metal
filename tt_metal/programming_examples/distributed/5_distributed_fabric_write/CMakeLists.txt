cmake_minimum_required(VERSION 3.22...3.30)
project(fabric_write LANGUAGES CXX)

add_executable(fabric_write)
file(GLOB_RECURSE kernels kernels/*)
target_sources(
    fabric_write
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        FILES ${kernels}
    PRIVATE
        fabric_write.cpp
)
target_link_libraries(
    fabric_write
    PRIVATE
        TT::Metalium
        TTNN::CPP
)
set_target_properties(
    fabric_write
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/distributed
)
