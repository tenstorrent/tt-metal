add_executable(unit_tests_sfpi)
set_target_properties(
    unit_tests_sfpi
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

target_sources(unit_tests_sfpi PRIVATE test_sfpi.cpp)
target_include_directories(
    unit_tests_sfpi
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(unit_tests_sfpi PRIVATE test_metal_common_libs)
