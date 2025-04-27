set(BENCHMARK_SRCS "host_tilizer_untilizer/tilizer_untilizer.cpp")

foreach(TEST_SRC ${BENCHMARK_SRCS})
    get_filename_component(TEST_TARGET ${TEST_SRC} NAME_WE)
    get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)
    # test_rw_buffer have two versions >:/ can we remove one?

    add_executable(${TEST_TARGET} ${TEST_SRC})

    target_link_libraries(
        ${TEST_TARGET}
        PUBLIC
            test_metal_common_libs
            benchmark::benchmark
    )
    target_include_directories(
        ${TEST_TARGET}
        BEFORE
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/ttnn/cpp/ttnn/deprecated # this all should go away and be replaced with link to ttnn
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/tt_metal/benchmark/${TEST_DIR}
    )
    target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
    list(APPEND PERF_MICROBENCH_TEST_TARGETS ${TEST_TARGET})
endforeach()

add_custom_target(metal_benchmark DEPENDS ${PERF_MICROBENCH_TEST_TARGETS})
