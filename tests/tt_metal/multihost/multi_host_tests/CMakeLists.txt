# Smoke tests (fast, not necessarily thorough)
add_library(multi_host OBJECT)
add_library(TT::Metalium::Test::Distributed::Multihost ALIAS multi_host)
TT_ENABLE_UNITY_BUILD(multi_host)

target_sources(multi_host PRIVATE main.cpp)
target_include_directories(
    multi_host
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)
target_link_libraries(multi_host PRIVATE test_metal_common_libs)

# Target for all STL tests regardless of duration
add_executable(multi_host_tests)
set_target_properties(
    multi_host_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(multi_host_tests PRIVATE TT::Metalium::Test::Distributed::Multihost)
