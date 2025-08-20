set(UNIT_TESTS_MISC_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_tilize_untilize.cpp)

# Create the test executable
add_executable(unit_tests_misc ${UNIT_TESTS_MISC_SRC})

# Enable unity build for the executable
TT_ENABLE_UNITY_BUILD(unit_tests_misc)

# Link libraries
target_link_libraries(
    unit_tests_misc
    PRIVATE
        test_metal_common_libs
        Boost::smart_ptr
)

# Set include directories
target_include_directories(
    unit_tests_misc
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# Set runtime output directory
set_target_properties(
    unit_tests_misc
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
