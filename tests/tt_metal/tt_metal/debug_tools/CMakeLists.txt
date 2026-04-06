# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

function(create_unit_test_executable)
    set(exec_name unit_tests_debug_tools)

    # Create the test executable
    add_executable(${exec_name} ${UNIT_TESTS_DEBUG_TOOLS_SRC})

    # Enable unity build for the executable
    TT_ENABLE_UNITY_BUILD(${exec_name})
    target_precompile_headers(${exec_name} REUSE_FROM metal_common_pch)

    # Link libraries
    target_link_libraries(${exec_name} PRIVATE test_metal_common_libs)

    # Set include directories
    target_include_directories(
        ${exec_name}
        BEFORE
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/common
    )

    # Set runtime output directory
    set_target_properties(
        ${exec_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal
    )
endfunction()

create_unit_test_executable()

# Inspector unit tests
# Create the test executable
add_executable(unit_tests_inspector ${UNIT_TESTS_INSPECTOR_SRC})

# Enable unity build for the executable
TT_ENABLE_UNITY_BUILD(unit_tests_inspector)
target_precompile_headers(unit_tests_inspector REUSE_FROM metal_common_pch)

# Link libraries
target_link_libraries(
    unit_tests_inspector
    PRIVATE
        test_metal_common_libs
        Metalium::Metal::Impl
        capnp
        capnp-rpc
        numa
)

# Set KJ_NO_EXCEPTIONS=0 to enable exceptions in Cap'n Proto code due to a detection bug with g++-12.
target_compile_options(unit_tests_inspector PRIVATE -DKJ_NO_EXCEPTIONS=0)

# Set include directories
target_include_directories(
    unit_tests_inspector
    BEFORE
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
        ${PROJECT_SOURCE_DIR}/tests/tt_metal/tt_metal/common
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)

# Set runtime output directory
set_target_properties(
    unit_tests_inspector
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
