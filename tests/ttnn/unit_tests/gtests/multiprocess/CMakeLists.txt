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
                ${PROJECT_BINARY_DIR}/test/ttnn/multiprocess
    )
    target_link_libraries(
        ${target_name}
        PRIVATE
            test_common_libs
            TTNN::CPP
    )
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

setup_multiprocess_unit_test(unit_tests_dual_rank_2x4 test_host_all_gather.cpp)
setup_multiprocess_unit_test(unit_tests_dual_rank_2x2 test_send_recv_ops.cpp)
