set(PROFILER_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count/test_custom_cycle_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count_slow_dispatch/test_custom_cycle_count_slow_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op/test_multi_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op_buffer_overflow/test_multi_op_buffer_overflow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_realtime_profiler_csv/test_realtime_profiler_csv.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch_cores/test_dispatch_cores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events/test_timestamped_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_noc_event_profiler/test_noc_event_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_device_api_debugger/test_device_api_debugger.cpp
)

CREATE_PGM_EXAMPLES_EXE("${PROFILER_EXAMPLES_SRCS}" "profiler")

# X280 bare-metal counter example: built separately (needs internal Cluster
# headers, so it can't go through the public-API-only CREATE_PGM_EXAMPLES_EXE).
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_counter)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_poll_rate)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_dma_probe)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_grid_drain)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_grid_drain4)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_poll4)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_noc1_probe)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_poll4n1)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_poll6n1)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_pollmp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_gridilp)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_rdrbench)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_lossless)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_stream)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_d2hbw)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_profrelay)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_profcons)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_profsplit)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test_x280_realprof)

add_custom_target(
    profiler_examples
    DEPENDS
        ${PROGRAMMING_EXAMPLES_TEST_TARGETS}
        test_x280_counter
        test_x280_poll_rate
        test_x280_dma_probe
        test_x280_grid_drain
        test_x280_grid_drain4
        test_x280_poll4
        test_x280_noc1_probe
        test_x280_poll4n1
        test_x280_poll6n1
        test_x280_pollmp
        test_x280_gridilp
        test_x280_lossless
        test_x280_stream
        test_x280_d2hbw
        test_x280_profrelay
        test_x280_profcons
        test_x280_profsplit
        test_x280_realprof
)
