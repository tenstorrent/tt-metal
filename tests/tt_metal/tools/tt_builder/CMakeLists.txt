set(UNIT_TESTS_BUILDER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_builder.cpp)
add_executable(unit_tests_builder ${UNIT_TESTS_BUILDER_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_builder)

target_link_libraries(
    unit_tests_builder
    PUBLIC
        test_metal_common_libs
        tt_builder_lib
)
target_include_directories(
    unit_tests_builder
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/tools/tt_builder
        ${PROJECT_SOURCE_DIR}/tests/
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    unit_tests_builder
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/
)
