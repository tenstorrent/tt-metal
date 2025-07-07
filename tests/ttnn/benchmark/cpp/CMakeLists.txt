set(BENCHMARK_SRCS
    "host_tilizer_untilizer/tilizer_untilizer.cpp"
    "padding/pad_rm.cpp"
    "operations/ternary/benchmark_where.cpp"
    "host_alloc_on_tensor_readback.cpp"
)

foreach(TEST_SRC ${BENCHMARK_SRCS})
    get_filename_component(TEST_TARGET ${TEST_SRC} NAME_WE)
    get_filename_component(TEST_DIR ${TEST_SRC} DIRECTORY)

    add_executable(${TEST_TARGET} ${TEST_SRC})

    target_link_libraries(
        ${TEST_TARGET}
        PUBLIC
            ttnn
            test_common_libs
            benchmark::benchmark
            Python3::Python
    )
    target_include_directories(
        ${TEST_TARGET}
        BEFORE
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/test_utils
            ${PROJECT_SOURCE_DIR}/ttnn/cpp
            ${PROJECT_SOURCE_DIR}/ttnn/
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/tests/ttnn//benchmark/${TEST_DIR}
    )
    message(STATUS "TEST_TARGET: ${TEST_TARGET}")
    target_compile_options(${TEST_TARGET} PUBLIC ${COMPILE_OPTIONS})
endforeach()
