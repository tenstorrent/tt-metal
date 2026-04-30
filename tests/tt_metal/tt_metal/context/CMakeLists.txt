# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Smoke tests
add_library(unit_tests_context_smoke OBJECT)
add_library(TT::Metalium::Test::Context::Smoke ALIAS unit_tests_context_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_context_smoke)
target_sources(unit_tests_context_smoke PRIVATE ${UNIT_TESTS_CONTEXT_SMOKE_SOURCES})
target_include_directories(
    unit_tests_context_smoke
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(unit_tests_context_smoke PRIVATE test_metal_common_libs)

# Non-smoke tests
add_executable(unit_tests_context)
target_include_directories(
    unit_tests_context
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)
set_target_properties(
    unit_tests_context
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(
    unit_tests_context
    PRIVATE
        TT::Metalium::Test::Context::Smoke
        test_metal_common_libs
)
