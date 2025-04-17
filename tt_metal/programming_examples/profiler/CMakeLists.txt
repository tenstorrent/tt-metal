set(PROFILER_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count/test_custom_cycle_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count_slow_dispatch/test_custom_cycle_count_slow_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op/test_multi_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch_cores/test_dispatch_cores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_timestamped_events/test_timestamped_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_noc_event_profiler/test_noc_event_profiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fabric_event_profiler/fabric_data_movement/test_fabric_event_profiler.cpp
)

CREATE_PGM_EXAMPLES_EXE("${PROFILER_EXAMPLES_SRCS}" "profiler")

add_custom_target(profiler_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})

# include directories required for fabric unit test
set(FABRIC_TEST_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/test_fabric_event_profiler/")
set(FABRIC_TEST_SRC "${FABRIC_TEST_ROOT_DIR}/fabric_data_movement/test_fabric_event_profiler.cpp")
get_filename_component(FABRIC_TEST_TARGET ${FABRIC_TEST_SRC} NAME_WE)
target_include_directories(
    ${FABRIC_TEST_TARGET}
    PRIVATE
        ${FABRIC_TEST_ROOT_DIR}/common
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
)
