set(TTNN_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/test_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_add.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_basic.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_async_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul_benchmark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multiprod_queue.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_multi_cq_multi_dev.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_launch_operation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_capture_arguments_morehdot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_capture_arguments_transpose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_query_op_constraints.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_query_op_runtime.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reflect.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_to_and_from_json.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_broadcast_to.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sliding_window_infra.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_conv2d.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/common_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_generic_op.cpp
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

set(TTNN_CCL_OP_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_persistent_fabric_ccl_ops.cpp)

set(TTNN_FABRIC_EDM_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_fabric_edm.cpp)
set(TTNN_CCL_MULTI_TENSOR_UNIT_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_multi_tensor_ccl.cpp)
set(TTNN_1D_FABRIC_LATENCY_TEST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/ccl/test_1d_fabric_loopback_latency.cpp)

set(TTNN_ACCESSOR_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/accessor/test_sharded_accessor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/accessor/test_sharded_accessor_on_device.cpp
)

set(TTNN_TENSOR_UNIT_TESTS_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/common_tensor_test_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_create_tensor_with_layout.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_distributed_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_mesh_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_serialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_sharding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_tensor_nd_sharding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_vector_conversion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tensor/test_xtensor_conversion.cpp
)

set(EMITC_UNIT_TESTS_SRC ${CMAKE_CURRENT_SOURCE_DIR}/emitc/test_sanity.cpp)

add_executable(unit_tests_ttnn ${TTNN_UNIT_TESTS_SRC})
TT_ENABLE_UNITY_BUILD(unit_tests_ttnn)
add_executable(unit_tests_ttnn_ccl ${TTNN_CCL_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_ccl_ops ${TTNN_CCL_OP_TESTS_SRC})
add_executable(unit_tests_ttnn_fabric_edm ${TTNN_FABRIC_EDM_SRC})
add_executable(unit_tests_ttnn_ccl_multi_tensor ${TTNN_CCL_MULTI_TENSOR_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_1d_fabric_latency ${TTNN_1D_FABRIC_LATENCY_TEST_SRC})
add_executable(unit_tests_ttnn_accessor ${TTNN_ACCESSOR_UNIT_TESTS_SRC})
add_executable(unit_tests_ttnn_tensor ${TTNN_TENSOR_UNIT_TESTS_SRC})
target_link_libraries(unit_tests_ttnn_tensor PRIVATE xtensor)
add_executable(
    test_ccl_multi_cq_multi_device
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_thread/test_ccl_multi_cq_multi_device.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/multi_thread/test_utils.cpp
)
add_executable(unit_tests_ttnn_emitc ${EMITC_UNIT_TESTS_SRC})

target_link_libraries(
    test_ccl_multi_cq_multi_device
    PRIVATE
        Boost::asio
        Boost::lockfree
)

# Set up properties for all targets
setup_ttnn_test_target(unit_tests_ttnn)
setup_ttnn_test_target(unit_tests_ttnn_ccl)
setup_ttnn_test_target(unit_tests_ttnn_ccl_ops)
setup_ttnn_test_target(unit_tests_ttnn_fabric_edm)
setup_ttnn_test_target(unit_tests_ttnn_ccl_multi_tensor)
setup_ttnn_test_target(unit_tests_ttnn_1d_fabric_latency)
setup_ttnn_test_target(unit_tests_ttnn_accessor)
setup_ttnn_test_target(unit_tests_ttnn_tensor)
setup_ttnn_test_target(test_ccl_multi_cq_multi_device)
setup_ttnn_test_target(unit_tests_ttnn_emitc)
