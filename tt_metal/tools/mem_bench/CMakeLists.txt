set(IMPL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/mem_bench.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_utils.cpp
)

set(HEADERS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/host_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/device_utils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/work_thread.hpp
)

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

check_function_exists(
    backtrace
    HAVE_BACKTRACE
)

if(NOT HAVE_BACKTRACE)
    find_library(EXECINFO_LIB execinfo)
    if(EXECINFO_LIB)
        target_link_libraries(mem_bench PRIVATE ${EXECINFO_LIB})
    endif()
endif()

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
