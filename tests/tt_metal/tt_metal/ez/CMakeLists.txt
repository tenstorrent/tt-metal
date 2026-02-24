# Include source file lists
include(sources.cmake)

# Create the test executable
add_executable(unit_tests_ez ${UNIT_TESTS_EZ_SRC})

# Link libraries
target_link_libraries(
    unit_tests_ez
    PRIVATE
        test_metal_common_libs
        tt_metal_ez
        Boost::smart_ptr
)

# Set include directories
target_include_directories(
    unit_tests_ez
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Set runtime output directory
set_target_properties(
    unit_tests_ez
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
