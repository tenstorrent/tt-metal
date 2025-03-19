set(TTNN_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_basic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiprod_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cq_multi_dev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_capture_arguments_morehdot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_capture_arguments_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_query_op_constraints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_query_op_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reflect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_to_and_from_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sliding_window_infra.cpp
)

set(TTNN_CCL_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_erisc_data_mover_with_workers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_fabric_erisc_data_mover_loopback_with_workers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_commands.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_helpers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_tensor_slicers.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_sharded_address_generators_new.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_sharded_address_generators.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_ccl_reduce_scatter_host_helpers.cpp
)

set(TTNN_FABRIC_EDM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_fabric_edm.cpp)

set(TTNN_TENSOR_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/common_tensor_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_with_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_distributed_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_mesh_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_sharding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_vector_conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_xtensor_conversion.cpp
)

add_executable(unit_tests_ttnn ${TTNN_UNIT_TESTS_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn)
add_executable(unit_tests_ttnn_ccl ${TTNN_CCL_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_fabric_edm ${TTNN_FABRIC_EDM_SRC})
add_executable(unit_tests_ttnn_tensor ${TTNN_TENSOR_UNIT_TESTS_SRC})
add_executable(test_multi_device ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_device.cpp)
add_executable(galaxy_unit_tests_ttnn ${CMAKE_CURRENT_SOURCE_DIR}/test_ccl_on_galaxy.cpp)
add_executable(
    test_ccl_multi_cq_multi_device
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_thread/test_ccl_multi_cq_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_thread/test_utils.cpp
)

target_link_libraries(
    test_ccl_multi_cq_multi_device
    PRIVATE
        Boost::asio
        Boost::lockfree
)

# Set up properties for all targets
setup_ttnn_test_target(unit_tests_ttnn)
setup_ttnn_test_target(unit_tests_ttnn_ccl)
setup_ttnn_test_target(unit_tests_ttnn_fabric_edm)
setup_ttnn_test_target(unit_tests_ttnn_tensor)
setup_ttnn_test_target(test_multi_device)
setup_ttnn_test_target(galaxy_unit_tests_ttnn)
setup_ttnn_test_target(test_ccl_multi_cq_multi_device)
