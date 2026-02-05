# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(
    mem_bench
    ${IMPL_SRC}
    ${HEADERS_SRC}
)
target_link_libraries(
    mem_bench
    PRIVATE
        Metalium::Metal
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
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    mem_bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/tools
)

target_precompile_headers(mem_bench REUSE_FROM TT::CommonPCH)
