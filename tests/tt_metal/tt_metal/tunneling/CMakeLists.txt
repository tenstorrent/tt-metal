set(UNIT_TESTS_TUNNELING_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_lite_fabric_init.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_lite_fabric.cpp
)

function(create_unit_test_executable)
    set(exec_name unit_tests_tunneling)

    # Create the test executable
    add_executable(${exec_name} ${UNIT_TESTS_TUNNELING_SRC})

    # Enable unity build for the executable
    TT_ENABLE_UNITY_BUILD(${exec_name})

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
