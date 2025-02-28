# Smoke tests (fast, not necessarily thorough)
add_library(unit_tests_stl_smoke OBJECT)
add_library(TT::Metalium::Test::STL::Smoke ALIAS unit_tests_stl_smoke)
TT_ENABLE_UNITY_BUILD(unit_tests_stl_smoke)

target_sources(
    unit_tests_stl_smoke
    PRIVATE
        test_any_range.cpp
        test_indestructible.cpp
        test_slotmap.cpp
        test_strong_type.cpp
)
target_include_directories(
    unit_tests_stl_smoke
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(unit_tests_stl_smoke PRIVATE test_metal_common_libs)

# Target for all STL tests regardless of duration
add_executable(unit_tests_stl)
set_target_properties(
    unit_tests_stl
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(unit_tests_stl PRIVATE TT::Metalium::Test::STL::Smoke)
