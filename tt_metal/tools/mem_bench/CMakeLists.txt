include(sources.cmake)

add_executable(
    mem_bench
    ${MEM_BENCH_SOURCES}
    ${MEM_BENCH_HEADERS}
)
target_link_libraries(
    mem_bench
    PRIVATE
        Metalium::Metal
        Metalium::Metal::Hardware
        numa
        benchmark::benchmark
)
target_include_directories(
    mem_bench
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tests
        .
)
set_target_properties(
    mem_bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

tt_reuse_precompile_headers(mem_bench TT::CommonPCH)
