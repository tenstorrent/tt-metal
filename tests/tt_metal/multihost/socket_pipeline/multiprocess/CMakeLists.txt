# Metal-level only tests (no external dependencies)
function(setup_metal_pipeline_test target_name test_source)
    add_executable(${target_name})
    target_sources(
        ${target_name}
        PRIVATE
            main.cpp
            ${test_source}
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost/fabric_tests/intermesh_routing_test_utils.cpp
            # Metal-level socket operation implementations
            utils/mesh_socket_send_recv.cpp
            utils/mesh_socket_forward.cpp
    )
    set_target_properties(
        ${target_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/multihost/socket_pipeline/multiprocess
    )
    target_link_libraries(${target_name} PRIVATE test_metal_common_libs)
    target_include_directories(
        ${target_name}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
    )
endfunction()

# Metal-level single-host loopback pipeline test
setup_metal_pipeline_test(unit_tests_single_host_pipeline single_host_pipeline_tests.cpp)
# Metal-level multi-host pipeline test (closet box + single galaxy)
setup_metal_pipeline_test(unit_tests_pipeline multi_host_pipeline.cpp)
