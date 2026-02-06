# Tests that depend on ttnn APIs (send_recv_async, distributed tensor, etc.)
function(setup_multiprocess_unit_test target_name test_source)
    add_executable(${target_name})
    target_sources(
        ${target_name}
        PRIVATE
            main.cpp
            ${test_source}
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost/fabric_tests/intermesh_routing_test_utils.cpp
    )
    set_target_properties(
        ${target_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY
                ${PROJECT_BINARY_DIR}/test/tt_metal/multihost/socket_pipeline/multiprocess
    )
    target_link_libraries(
        ${target_name}
        PRIVATE
            test_common_libs
            TTNN::CPP
    )
    target_include_directories(
        ${target_name}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
    )
endfunction()

# Metal-level only tests (no ttnn dependency)
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

# Tests that use ttnn APIs and metal-level socket utils
function(setup_socket_pipeline_test target_name test_source)
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
    target_link_libraries(
        ${target_name}
        PRIVATE
            test_common_libs
            TTNN::CPP
    )
    target_include_directories(
        ${target_name}
        PRIVATE
            "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
            ${PROJECT_SOURCE_DIR}/tests
            ${PROJECT_SOURCE_DIR}/tests/tt_metal/multihost
    )
endfunction()

setup_multiprocess_unit_test(unit_tests_dual_rank_2x4 test_host_all_gather.cpp)
# test_send_recv_ops.cpp includes both ttnn-based and metal-level migrated tests
setup_socket_pipeline_test(unit_tests_dual_rank_2x2 test_send_recv_ops.cpp)
setup_multiprocess_unit_test(unit_tests_pipeline_2x4 test_send_recv_pipeline.cpp)
# socket_fwd_pipeline.cpp is pure metal-level (no ttnn dependency)
setup_metal_pipeline_test(unit_tests_pipeline socket_fwd_pipeline.cpp)
