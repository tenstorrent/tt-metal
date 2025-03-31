set(UNIT_TESTS_NOC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_dynamic_noc.cpp)

add_executable(unit_tests_noc ${UNIT_TESTS_NOC_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_noc)

target_link_libraries(unit_tests_noc PUBLIC test_metal_common_libs)

target_include_directories(
    unit_tests_noc
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set_target_properties(
    unit_tests_noc
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
