# Smoke tests (fast, not necessarily thorough)
add_library(single_host_mp OBJECT)
add_library(TT::Metalium::Test::Distributed::Multihost::mp ALIAS single_host_mp)
TT_ENABLE_UNITY_BUILD(single_host_mp)

target_sources(
    single_host_mp
    PRIVATE
        test_context.cpp
        main.cpp
)
target_include_directories(
    single_host_mp
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
)
target_link_libraries(single_host_mp PRIVATE test_metal_common_libs)

# Target for all STL tests regardless of duration
add_executable(single_host_mp_tests)
set_target_properties(
    single_host_mp_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
target_link_libraries(single_host_mp_tests PRIVATE TT::Metalium::Test::Distributed::Multihost::mp)
