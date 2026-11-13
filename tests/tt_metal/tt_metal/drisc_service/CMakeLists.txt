# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(unit_tests_drisc_service ${UNIT_TESTS_DRISC_SERVICE_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_drisc_service)
tt_reuse_precompile_headers(unit_tests_drisc_service metal_test_pch)

target_link_libraries(unit_tests_drisc_service PUBLIC test_metal_common_libs)
add_dependencies(unit_tests_drisc_service drisc_service_elf)

target_include_directories(
    unit_tests_drisc_service
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set_target_properties(
    unit_tests_drisc_service
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
