
set(UNIT_TESTS_FD_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_CommandQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueTrace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_HostAsyncCQ.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/sfpu/sfpu_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/test_eth_EnqueueProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/test_eth_ring_gather_EnqueueProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipelining/basic_pipeline.cpp
)

add_executable(unit_tests_fast_dispatch ${UNIT_TESTS_FD_SRC} $<TARGET_OBJECTS:unit_tests_common_o>)

target_link_libraries(unit_tests_fast_dispatch PUBLIC tt_metal_libs stdc++fs yaml-cpp gtest gtest_main m)
target_include_directories(unit_tests_fast_dispatch PRIVATE
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(unit_tests_fast_dispatch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_metal)
