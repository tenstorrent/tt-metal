# Definition of the d2d_stream_benchmarks target, factored out so it can be pulled in two ways:
#   1. via add_subdirectory(microbench) (the root-CMakeLists edit path), through this dir's
#      CMakeLists.txt which just include()s this file, or
#   2. via a DEFERred include() from microbench/inject.cmake (the no-root-edit path).
#
# Because path (2) runs this file in the ROOT directory scope at the end of configure,
# all source paths must be absolute. CMAKE_CURRENT_LIST_DIR always points at THIS file's
# directory regardless of which path included it, so it is the right anchor.

add_executable(d2d_stream_benchmarks)
target_sources(d2d_stream_benchmarks PRIVATE ${CMAKE_CURRENT_LIST_DIR}/benchmark_d2d_stream_service.cpp)

# TTNN::CPP brings in the D2D / H2D stream services (and tt_metal transitively);
# benchmark::benchmark_main supplies main(); test_common_libs for the shared test deps.
target_link_libraries(
    d2d_stream_benchmarks
    PRIVATE
        TTNN::CPP
        test_common_libs
        benchmark::benchmark
        benchmark::benchmark_main
)

target_include_directories(
    d2d_stream_benchmarks
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests # reach tests/ttnn/.../stream_service_test_utils.hpp
        ${PROJECT_SOURCE_DIR}
)

set_target_properties(
    d2d_stream_benchmarks
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/microbench/d2d_stream_service
)
