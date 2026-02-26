# Create object library first
add_library(unit_tests_api_lib OBJECT)
add_library(TT::Metalium::Test::API ALIAS unit_tests_api_lib)

# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Use target_sources to gather the necessary sources
target_sources(unit_tests_api_lib PRIVATE ${UNIT_TESTS_API_SOURCES})

target_include_directories(
    unit_tests_api_lib
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# Link libraries to the object library
target_link_libraries(
    unit_tests_api_lib
    PRIVATE
        test_metal_common_libs
        Boost::smart_ptr
)

TT_ENABLE_UNITY_BUILD(unit_tests_api_lib)
target_precompile_headers(unit_tests_api_lib REUSE_FROM metal_common_pch)

# Create the test executable
add_executable(unit_tests_api)

# Link libraries to the executable
target_link_libraries(unit_tests_api PRIVATE unit_tests_api_lib)

# Set runtime output directory
set_target_properties(
    unit_tests_api
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
