set(UNIT_TESTS_DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_distributed.cpp
)

add_executable(distributed_unit_tests ${UNIT_TESTS_DISTRIBUTED_SRC})
target_link_libraries(distributed_unit_tests PRIVATE tt_metal test_common_libs)

target_include_directories(distributed_unit_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tt_metal/distributed
)

set_target_properties(distributed_unit_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal/distributed)

gtest_discover_tests(distributed_unit_tests)
