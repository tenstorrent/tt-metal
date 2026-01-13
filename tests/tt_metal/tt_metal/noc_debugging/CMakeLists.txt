set(UNIT_TESTS_NOC_DEBUGGING_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_reads_writes.cpp)

# Create the test executable
add_executable(unit_tests_noc_debugging ${UNIT_TESTS_NOC_DEBUGGING_SRC})

# Enable unity build for the executable
TT_ENABLE_UNITY_BUILD(unit_tests_noc_debugging)

# Link libraries
target_link_libraries(unit_tests_noc_debugging PRIVATE test_metal_common_libs)

# Set include directories
target_include_directories(
    unit_tests_noc_debugging
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set runtime output directory
set_target_properties(
    unit_tests_noc_debugging
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
