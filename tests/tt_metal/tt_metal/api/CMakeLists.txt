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
target_precompile_headers(unit_tests_api_lib REUSE_FROM metal_common_pch)

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
target_precompile_headers(unit_tests_tensor_lib REUSE_FROM metal_common_pch)

add_executable(unit_tests_tensor)
target_link_libraries(unit_tests_tensor PRIVATE unit_tests_tensor_lib)
set_target_properties(
    unit_tests_tensor
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

# Per-core allocation tests need their own binary so TT_METAL_ALLOCATOR_MODE_HYBRID
# env var is set before the RunTimeOptions singleton caches it.
add_executable(unit_tests_per_core_allocation allocator/test_per_core_allocation.cpp)
target_include_directories(
    unit_tests_per_core_allocation
    BEFORE
    PRIVATE
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
)
target_link_libraries(unit_tests_per_core_allocation PRIVATE test_metal_common_libs)
target_precompile_headers(unit_tests_per_core_allocation REUSE_FROM metal_common_pch)
set_target_properties(
    unit_tests_per_core_allocation
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)

if(TT_METAL_USE_EMULE)
    target_link_options(unit_tests_api PRIVATE -rdynamic)

    # Per-fiber ASAN sanitizer-state isolation test. Its own binary because it
    # includes the emule kernel-side context header (jit_hw/internal/
    # emule_thread_ctx.h) directly, which isn't on the shared api test lib's
    # include path and shouldn't be dragged into its unity build / PCH. Sources come
    # from the emule-owned emule/sources.cmake (included via sources.cmake above).
    add_executable(unit_tests_fiber_asan ${UNIT_TESTS_FIBER_ASAN_SOURCES})
    target_include_directories(
        unit_tests_fiber_asan
        BEFORE
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
            ${TT_EMULE_PATH}/include
    )
    target_link_libraries(unit_tests_fiber_asan PRIVATE test_metal_common_libs)
    target_link_options(unit_tests_fiber_asan PRIVATE -rdynamic)
    set_target_properties(
        unit_tests_fiber_asan
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
endif()
