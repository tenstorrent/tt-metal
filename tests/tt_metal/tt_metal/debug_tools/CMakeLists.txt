# Include source file lists (module owners maintain sources.cmake)
include(sources.cmake)

function(create_unit_test_executable)
    set(exec_name unit_tests_debug_tools)

    # Create the test executable
    add_executable(${exec_name} ${UNIT_TESTS_DEBUG_TOOLS_SRC})

    # Enable unity build for the executable
    TT_ENABLE_UNITY_BUILD(${exec_name})
    tt_reuse_precompile_headers(${exec_name} metal_test_pch)

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

    # ttexalens::elf is a static library that is already linked into libtt_metal.so, which this test links
    # through test_metal_common_libs. Linking it again would put a second copy of it in this executable, which
    # ASan reports as an ODR violation (e.g. on ttexalens::native_elf::details::SpanStreamBuf's vtable). Take only
    # its headers and resolve the symbols from libtt_metal.so.
    target_include_directories(
        ${exec_name}
        SYSTEM
        PRIVATE
            "$<TARGET_PROPERTY:ttexalens::elf,INTERFACE_INCLUDE_DIRECTORIES>"
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
tt_reuse_precompile_headers(unit_tests_inspector metal_test_pch)

# Link libraries
target_link_libraries(
    unit_tests_inspector
    PRIVATE
        test_metal_common_libs
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

# Metalium::Metal::Impl (an OBJECT library) and Cap'n Proto (a static library) are already inside libtt_metal.so, which
# this test links through test_metal_common_libs. Linking either of them here as well would embed a second copy in this
# executable (every impl object, including the generated inspector rpc.capnp.c++, and the capnp/kj runtime), which ASan
# reports as an ODR violation. Take only their include directories (Impl also provides the generated rpc headers) and
# resolve the symbols from libtt_metal.so.
target_include_directories(
    unit_tests_inspector
    SYSTEM
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal::Impl,INTERFACE_INCLUDE_DIRECTORIES>"
        "$<TARGET_PROPERTY:capnp,INTERFACE_INCLUDE_DIRECTORIES>"
        "$<TARGET_PROPERTY:capnp-rpc,INTERFACE_INCLUDE_DIRECTORIES>"
)

# Set runtime output directory
set_target_properties(
    unit_tests_inspector
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal
)
