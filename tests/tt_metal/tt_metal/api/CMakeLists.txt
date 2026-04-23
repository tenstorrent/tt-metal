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

    # DST capacity tests depend on tt_emule headers (standalone DstRegisterFile
    # + jit_hw/compute/common.h), so only compile them when emule is enabled.
    # Skip unity build — jit_hw/compute/common.h redefines CBIndex/MathFidelity.
    target_sources(unit_tests_api_lib PRIVATE test_dst_capacity.cpp)
    set_source_files_properties(
        test_dst_capacity.cpp PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
    target_link_libraries(unit_tests_api_lib PRIVATE tt_emule_headers)
    get_target_property(_EMULE_JIT_DIR tt_emule_headers TT_EMULE_JIT_INCLUDE_DIR)
    target_include_directories(unit_tests_api_lib PRIVATE ${_EMULE_JIT_DIR})
endif()
