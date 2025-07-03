# Smoke tests (fast, not necessarily thorough)
add_library(fault_tolerance OBJECT)
add_library(TT::Metalium::Test::Distributed::FaultTolerance ALIAS fault_tolerance)
TT_ENABLE_UNITY_BUILD(fault_tolerance)

target_sources(
    fault_tolerance
    PRIVATE
        ulfm_tests.cpp
        main.cpp
)
target_include_directories(
    fault_tolerance
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)
target_link_libraries(fault_tolerance PRIVATE test_metal_common_libs)

# Target for all STL tests regardless of duration
add_executable(fault_tolerance_tests)
set_target_properties(
    fault_tolerance_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(fault_tolerance_tests PRIVATE TT::Metalium::Test::Distributed::FaultTolerance)
