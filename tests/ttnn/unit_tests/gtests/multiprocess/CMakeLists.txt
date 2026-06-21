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
    tt_reuse_precompile_headers(${target_name} TTNN::PCHLean)
endfunction()

setup_multiprocess_unit_test(unit_tests_dual_rank_2x4 test_host_all_gather.cpp)
setup_multiprocess_unit_test(unit_tests_dual_rank_2x2 test_send_recv_ops.cpp)
setup_multiprocess_unit_test(unit_tests_pipeline_2x4 test_send_recv_pipeline.cpp)
setup_multiprocess_unit_test(unit_tests_cross_process_d2d test_cross_process_d2d_stream_service.cpp)

# Cross-process H2DStreamService test: owns its main() (MPI world setup), so it is a
# standalone target rather than using setup_multiprocess_unit_test. Launched with `mpirun -np 2`.
add_executable(cross_process_h2d_stream_service_test)
target_sources(cross_process_h2d_stream_service_test PRIVATE test_cross_process_h2d_stream_service.cpp)
target_link_libraries(
    cross_process_h2d_stream_service_test
    PRIVATE
        TTNN::CPP
        tt_metal
        test_common_libs
)
target_include_directories(
    cross_process_h2d_stream_service_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)
set_target_properties(
    cross_process_h2d_stream_service_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn/multiprocess
)

# Cross-process D2HStreamService test: owns its main() (MPI world setup), so it is a
# standalone target rather than using setup_multiprocess_unit_test. Launched with `mpirun -np 2`.
add_executable(cross_process_d2h_stream_service_test)
target_sources(cross_process_d2h_stream_service_test PRIVATE test_cross_process_d2h_stream_service.cpp)
target_link_libraries(
    cross_process_d2h_stream_service_test
    PRIVATE
        TTNN::CPP
        tt_metal
        test_common_libs
)
target_include_directories(
    cross_process_d2h_stream_service_test
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)
set_target_properties(
    cross_process_d2h_stream_service_test
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/ttnn/multiprocess
)
