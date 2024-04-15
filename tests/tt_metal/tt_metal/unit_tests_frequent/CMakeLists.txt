
set(UNIT_TESTS_FREQUENT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_many_times.cpp
)

add_executable(unit_tests_frequent ${UNIT_TESTS_FREQUENT_SRCS})

target_link_libraries(unit_tests_frequent PUBLIC test_metal_common_libs)
target_include_directories(unit_tests_frequent PRIVATE
    ${UMD_HOME}
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/tt_metal
    ${CMAKE_SOURCE_DIR}/tests
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(unit_tests_frequent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/tt_metal)
