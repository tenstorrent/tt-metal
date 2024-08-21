
set(UNIT_TESTS_FD_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_CommandQueue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueTrace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_EnqueueWriteBuffer_and_EnqueueReadBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/command_queue/test_HostAsyncCQ.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/compute/sfpu/sfpu_compute.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/test_device_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/test_eth_EnqueueProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multichip/test_eth_ring_gather_EnqueueProgram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pipelining/basic_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/streams/test_autonomous_relay_streams.cpp
)

add_executable(unit_tests_fast_dispatch ${UNIT_TESTS_FD_SRC} $<TARGET_OBJECTS:unit_tests_common_o>)

target_link_libraries(unit_tests_fast_dispatch PUBLIC test_metal_common_libs)
target_include_directories(unit_tests_fast_dispatch PRIVATE
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(unit_tests_fast_dispatch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
