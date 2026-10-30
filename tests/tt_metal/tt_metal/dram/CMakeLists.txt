# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

add_executable(mimir_tests ${MIMIR_TESTS_SRC})
TT_ENABLE_UNITY_BUILD(mimir_tests)
target_precompile_headers(mimir_tests REUSE_FROM metal_test_pch)

target_link_libraries(mimir_tests PRIVATE test_metal_common_libs)
target_include_directories(
    mimir_tests
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-2xx/quasar
)
set_target_properties(
    mimir_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
