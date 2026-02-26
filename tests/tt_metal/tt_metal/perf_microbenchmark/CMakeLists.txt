# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Add x86_64 tests only on appropriate platforms.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    list(APPEND PERF_MICROBENCH_TESTS_SRCS ${X86_64_ONLY_TESTS})
endif()

foreach(TEST_SRC ${PERF_MICROBENCH_TESTS_SRCS})
    get_filename_component(TEST_TARGET ${TEST_SRC} NAME_WE)
    get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)
    # test_rw_buffer have two versions >:/ can we remove one?
    if(${TEST_SRC} STREQUAL "old/pcie/test_rw_buffer.cpp")
        set(TEST_TARGET "test_rw_buffer_old")
    endif()
    string(REPLACE "wormhole" "wormhole_b0" TEST_TARGET ${TEST_TARGET})

    add_executable(${TEST_TARGET} ${TEST_SRC})
    target_precompile_headers(${TEST_TARGET} REUSE_FROM metal_common_pch)
    target_link_libraries(${TEST_TARGET} PRIVATE test_metal_common_libs)

    if(${TEST_SRC} STREQUAL "dispatch/test_pgm_dispatch.cpp")
        target_link_libraries(${TEST_TARGET} PRIVATE benchmark::benchmark)
    endif()
    if(${TEST_SRC} STREQUAL "dispatch/benchmark_rw_buffer.cpp")
        target_link_libraries(${TEST_TARGET} PRIVATE benchmark::benchmark)
    endif()

    if(${TEST_SRC} STREQUAL "routing/test_tt_fabric.cpp")
        target_sources(${TEST_TARGET} PRIVATE ${TEST_TT_FABRIC_ADDITIONAL_SOURCES})
        target_link_libraries(${TEST_TARGET} PRIVATE yaml-cpp::yaml-cpp)
    endif()
    target_include_directories(
        ${TEST_TARGET}
        BEFORE
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/perf_microbenchmark/${TEST_DIR}
    )
    target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND PERF_MICROBENCH_TEST_TARGETS ${TEST_TARGET})
endforeach()

add_custom_target(metal_perf_microbenchmark_tests DEPENDS ${PERF_MICROBENCH_TEST_TARGETS})
