
set(PROFILER_EXAMPLES_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/test_custom_cycle_count/test_custom_cycle_count
    ${CMAKE_CURRENT_SOURCE_DIR}/test_full_buffer/test_full_buffer
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_op/test_multi_op
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dispatch_cores/test_dispatch_cores
)

CREATE_PGM_EXAMPLES_EXE("${PROFILER_EXAMPLES_SRCS}" "profiler")

add_custom_target(profiler_examples DEPENDS ${PROGRAMMING_EXAMPLES_TEST_TARGETS})

# should throw this into helper_functions.cmake
# foreach (TEST ${PROFILER_EXAMPLES_SRCS})
#     get_filename_component(TEST_TARGET ${TEST} NAME)

#     add_executable(${TEST_TARGET} ${TEST})
#     target_link_libraries(${TEST_TARGET} PUBLIC tt_metal stdc++fs yaml-cpp m)
#     target_include_directories(${TEST_TARGET} PRIVATE
#         ${UMD_HOME}
#         ${CMAKE_SOURCE_DIR}
#         ${CMAKE_SOURCE_DIR}/tt_metal
#         ${CMAKE_SOURCE_DIR}/tt_metal/common
#         ${CMAKE_CURRENT_SOURCE_DIR}
#     )
#     set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/programming_examples/profiler)
#     list(APPEND PROFILER_EXAMPLES_TEST_TARGETS ${TEST_TARGET})
# endforeach()

# add_custom_target(profiler_examples DEPENDS ${PROFILER_EXAMPLES_TEST_TARGETS})
