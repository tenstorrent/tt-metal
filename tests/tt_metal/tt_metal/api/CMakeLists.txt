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
tt_reuse_precompile_headers(unit_tests_api_lib metal_test_pch)

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

# Runtime tensor tests build into their own executable (mirrors unit_tests_ttnn_tensor)
# so they stay out of the tt-metalium smoke binary. They run via unit_tests_tensor in
# build-and-unit-tests and the t3k unit-test suite.
add_library(unit_tests_tensor_lib OBJECT)
add_library(TT::Metalium::Test::Tensor ALIAS unit_tests_tensor_lib)
target_sources(unit_tests_tensor_lib PRIVATE ${UNIT_TESTS_API_TENSOR_SOURCES})
target_include_directories(
    unit_tests_tensor_lib
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(
    unit_tests_tensor_lib
    PRIVATE
        test_metal_common_libs
        Boost::smart_ptr
)
TT_ENABLE_UNITY_BUILD(unit_tests_tensor_lib)
tt_reuse_precompile_headers(unit_tests_tensor_lib metal_test_pch)

add_executable(unit_tests_tensor)
target_link_libraries(unit_tests_tensor PRIVATE unit_tests_tensor_lib)
set_target_properties(
    unit_tests_tensor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# One binary for every suite that needs TT_METAL_ALLOCATOR_MODE_HYBRID set before the
# RunTimeOptions singleton caches it, so they can all be run in one go.
add_executable(
    unit_tests_hybrid_allocator
    allocator/test_per_core_allocation.cpp
    allocator/test_range_lockstep_allocation.cpp
    allocator/test_range_lockstep_allocation_host.cpp
    allocator/test_uniform_per_core_allocation.cpp
)
target_include_directories(
    unit_tests_hybrid_allocator
    BEFORE
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${PROJECT_SOURCE_DIR}
)
target_link_libraries(unit_tests_hybrid_allocator PRIVATE test_metal_common_libs)
tt_reuse_precompile_headers(unit_tests_hybrid_allocator metal_test_pch)
set_target_properties(
    unit_tests_hybrid_allocator
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Quasar-only DFB init timing benchmarks (standalone; not linked into unit_tests_api).
add_executable(dfb_init_timing_bench dataflow_buffer/dfb_init_timing_bench.cpp)
target_include_directories(
    dfb_init_timing_bench
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(
    dfb_init_timing_bench
    PRIVATE
        test_metal_common_libs
        Boost::smart_ptr
)
tt_reuse_precompile_headers(dfb_init_timing_bench metal_test_pch)
set_target_properties(
    dfb_init_timing_bench
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
