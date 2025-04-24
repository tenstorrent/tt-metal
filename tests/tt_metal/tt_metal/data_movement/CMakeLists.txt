set(UNIT_TESTS_DATA_MOVEMENT_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/dm_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dram_unary/test_unary_dram.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/one_to_one/test_one_to_one.cpp
)

add_executable(unit_tests_data_movement ${UNIT_TESTS_DATA_MOVEMENT_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_data_movement)

target_link_libraries(unit_tests_data_movement PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_data_movement
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_data_movement
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
