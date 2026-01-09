add_library(unit_tests_legacy_lib OBJECT)
add_library(TT::Metalium::Test::Legacy ALIAS unit_tests_legacy_lib)

target_sources(unit_tests_legacy_lib PRIVATE test_legacy.cpp)

target_include_directories(
    unit_tests_legacy_lib
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal
)

target_link_libraries(unit_tests_legacy_lib PRIVATE test_metal_common_libs)

# Create the test executable
add_executable(unit_tests_legacy)
target_link_libraries(unit_tests_legacy PRIVATE unit_tests_legacy_lib)

set_target_properties(
    unit_tests_legacy
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
