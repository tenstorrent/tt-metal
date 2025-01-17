set(UNIT_TESTS_STL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_any_range.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_slotmap.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_strong_type.cpp
)

add_executable(unit_tests_stl ${UNIT_TESTS_STL_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_stl)

target_link_libraries(unit_tests_stl PUBLIC test_metal_common_libs)
target_include_directories(
    unit_tests_stl
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
set_target_properties(
    unit_tests_stl
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
