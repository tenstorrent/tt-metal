set(UNIT_TESTS_DISTRIBUTED_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_buffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_coord.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_device_reshape.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_workload.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_sub_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_allocator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_events.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mesh_trace.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

# Define the function to create test executables for each architecture
function(create_unit_test_executable arch_name)
    # Define the test executable name using the architecture name
    set(exec_name distributed_unit_tests_${arch_name})
    string(REPLACE "wormhole" "wormhole_b0" exec_name "${exec_name}")

    add_executable(${exec_name} ${UNIT_TESTS_DISTRIBUTED_SRC})

    target_link_libraries(
        ${exec_name}
        PRIVATE
            tt_metal
            test_common_libs
    )

    target_include_directories(
        ${exec_name}
        PRIVATE
            ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc/${arch_name}
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
    )

    set_target_properties(
        ${exec_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/distributed
    )
endfunction()

# Define the architectures for which to create test executables
set(ARCHITECTURES
    "wormhole"
    "blackhole"
)

# Create a test executable for each architecture
foreach(arch IN LISTS ARCHITECTURES)
    create_unit_test_executable(${arch})
endforeach()

# Dont do this for now
# When the test is probed something is constructed that tries to access a device
# Build machine might not have a device
# We don't use ctest in this project so we shouldn't need this yet
#gtest_discover_tests(distributed_unit_tests)
