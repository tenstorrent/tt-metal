set(UNIT_TESTS_DEBUG_TOOLS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_eth_cores.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_invalid_print_core.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_mute_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_mute_print_server.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_buffering.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_all_harts.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_before_finish.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_hanging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_prepend_device_core_risc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_tensix_dest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_tiles.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_raise_wait.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dprint/test_print_config_register.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_assert.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_link_training.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_noc_sanitize_delays.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_noc_sanitize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_pause.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_ringbuf.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/watcher/test_waypoint.cpp
)

# Define the function to create test executables for each architecture
function(create_unit_test_executable arch_name)
    # Define the test executable name using the architecture name
    set(exec_name unit_tests_debug_tools_${arch_name})
    string(REPLACE "wormhole" "wormhole_b0" exec_name ${exec_name})

    # Create the test executable
    add_executable(${exec_name} ${UNIT_TESTS_DEBUG_TOOLS_SRC})

    # Enable unity build for the executable
    TT_ENABLE_UNITY_BUILD(${exec_name})

    # Link libraries
    target_link_libraries(${exec_name} PRIVATE test_metal_common_libs)

    # Set include directories
    target_include_directories(
        ${exec_name}
        BEFORE
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${arch_name}
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

# Define the architectures for which to create test executables
set(ARCHITECTURES
    "grayskull"
    "wormhole"
    "blackhole"
)

# Create a test executable for each architecture
foreach(arch ${ARCHITECTURES})
    create_unit_test_executable(${arch})
endforeach()
