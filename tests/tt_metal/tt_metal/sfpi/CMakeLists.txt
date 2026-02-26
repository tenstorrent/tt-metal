# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

# Create the object library first
add_library(unit_tests_sfpi_lib OBJECT)
add_library(TT::Metalium::Test::SFPI ALIAS unit_tests_sfpi_lib)

target_sources(unit_tests_sfpi_lib PRIVATE ${UNIT_TESTS_SFPI_SOURCES})
target_precompile_headers(unit_tests_sfpi_lib REUSE_FROM metal_common_pch)
target_include_directories(
    unit_tests_sfpi_lib
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(unit_tests_sfpi_lib PRIVATE test_metal_common_libs)

# Create the executable using the object library
add_executable(unit_tests_sfpi)
set_target_properties(
    unit_tests_sfpi
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

target_link_libraries(unit_tests_sfpi PRIVATE unit_tests_sfpi_lib)
