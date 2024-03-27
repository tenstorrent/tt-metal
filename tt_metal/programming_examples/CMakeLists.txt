
set(PROGRAMMING_EXAMPLES_SRCS
    eltwise_binary/eltwise_binary
    eltwise_sfpu/eltwise_sfpu
    loopback/loopback
    matmul_multi_core/matmul_multi_core
    matmul_multicore_reuse/matmul_multicore_reuse
    matmul_multicore_reuse_mcast/matmul_multicore_reuse_mcast
    matmul_single_core/matmul_single_core
    # none of the profiler tests are run?
)

foreach (TEST ${PROGRAMMING_EXAMPLES_SRCS})
    get_filename_component(TEST_TARGET ${TEST} NAME)

    add_executable(${TEST_TARGET} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST})
    target_link_libraries(${TEST_TARGET} PUBLIC tt_metal_libs stdc++fs yaml-cpp m)   # these should prob be converted to gtests
    target_include_directories(${TEST_TARGET} PRIVATE
        ${UMD_HOME}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/tt_metal
        ${CMAKE_SOURCE_DIR}/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples)
    list(APPEND PROGRAMMING_EXAMPLES_TEST_TARGETS ${TEST_TARGET})
endforeach()

add_custom_target(programming_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
