
set(PROFILER_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count/test_custom_cycle_count.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count_slow_dispatch/test_custom_cycle_count_slow_dispatch.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer/test_full_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op/test_multi_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch_cores/test_dispatch_cores.cpp
)

CREATE_PGM_EXAMPLES_EXE("${PROFILER_EXAMPLES_SRCS}" "profiler")

add_custom_target(profiler_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})
