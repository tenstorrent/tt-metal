
set(UNIT_TESTS_FREQUENT_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_many_times.cpp
)

add_executable(unit_tests_frequent ${UNIT_TESTS_FREQUENT_SRCS})

target_link_libraries(unit_tests_frequent PUBLIC ${COMMON_LIBS} tt_metal_libs stdc++fs yaml-cpp gtest gtest_main m)
target_include_directories(unit_tests_frequent PRIVATE
                                        ${UMD_HOME}
                                        ${CMAKE_SOURCE_DIR}
                                        ${CMAKE_SOURCE_DIR}/tt_metal
                                        ${CMAKE_SOURCE_DIR}/tests
                                        ${CMAKE_CURRENT_SOURCE_DIR}
                                        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_compile_options(unit_tests_frequent PUBLIC ${COMPILE_OPTIONS})
set_target_properties(unit_tests_frequent PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/tt_metal)
