
set(UNIT_TESTS_FD_SINGLEC_MULTIQ_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueTrace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueWaitForEvent.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp
)

add_executable(unit_tests_fast_dispatch_single_chip_multi_queue ${UNIT_TESTS_FD_SINGLEC_MULTIQ_SRCS})

target_link_libraries(unit_tests_fast_dispatch_single_chip_multi_queue PUBLIC tt_metal_libs stdc++fs yaml-cpp gtest gtest_main m)
target_include_directories(unit_tests_fast_dispatch_single_chip_multi_queue PRIVATE
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(unit_tests_fast_dispatch_single_chip_multi_queue PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_metal)
