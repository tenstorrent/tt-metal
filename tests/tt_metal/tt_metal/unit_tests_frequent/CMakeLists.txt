
set(UNIT_TESTS_FREQUENT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_many_times.cpp
)

add_executable(unit_tests_frequent ${UNIT_TESTS_FREQUENT_SRCS})

target_link_libraries(unit_tests_frequent PUBLIC test_metal_common_libs gtest gtest_main)
target_include_directories(unit_tests_frequent PRIVATE
    ${UMD_HOME}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/tt_metal
    ${PROJECT_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(unit_tests_frequent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/test/tt_metal)
