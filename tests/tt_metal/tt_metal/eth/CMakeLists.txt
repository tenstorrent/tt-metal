set(UNIT_TESTS_ETH_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_basic_eth.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_buffer_movement_kernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_erisc_app_direct_send.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_ring_gather_kernels.cpp
)

# Define the function to create test executables for each architecture
function(create_unit_test_executable arch_name)
    # Define the test executable name using the architecture name
    set(exec_name unit_tests_eth_${arch_name})
    string(REPLACE "wormhole" "wormhole_b0" exec_name ${exec_name})

    # Create the test executable
    add_executable(${exec_name} ${UNIT_TESTS_ETH_SRC})

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
    "wormhole"
    "blackhole"
)

# Create a test executable for each architecture
foreach(arch ${ARCHITECTURES})
    create_unit_test_executable(${arch})
endforeach()
