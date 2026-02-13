# Source files for ttnn unit tests
# Module owners should update this file when adding/removing/renaming source files

set(UNIT_TESTS_TTNN_SMOKE_SOURCES
    test_reflect.cpp
    test_to_and_from_json.cpp
    test_sliding_window_infra.cpp
    test_async_runtime.cpp
    test_conv2d.cpp
    test_multi_cq_multi_dev.cpp
    test_multiprod_queue.cpp
)

set(UNIT_TESTS_TTNN_BASIC_SOURCES
    test_add.cpp
    test_add_int.cpp
    test_bernoulli_descriptor_benchmark.cpp
    test_broadcast_to.cpp
    test_conv2d_descriptor_benchmark.cpp
    test_convert_to_hwc_gather.cpp
    test_generic_op.cpp
    test_graph_add.cpp
    test_graph_basic.cpp
    test_levelized_graph.cpp
    test_graph_capture_arguments_morehdot.cpp
    test_graph_capture_arguments_transpose.cpp
    test_graph_query_op_constraints.cpp
    test_graph_query_op_runtime.cpp
    test_launch_operation.cpp
    test_matmul_descriptor_benchmark.cpp
    test_relational_int.cpp
    test_rsub_int.cpp
    test_sub_int.cpp
)

set(UNIT_TESTS_TTNN_CCL_SOURCES
    ccl/test_ccl_commands.cpp
    ccl/test_ccl_helpers.cpp
    ccl/test_ccl_reduce_scatter_host_helpers.cpp
    ccl/test_ccl_tensor_slicers.cpp
    ccl/test_erisc_data_mover_with_workers.cpp
    ccl/test_fabric_erisc_data_mover_loopback_with_workers.cpp
    ccl/test_sharded_address_generators.cpp
    ccl/test_sharded_address_generators_new.cpp
)

set(UNIT_TESTS_TTNN_CCL_OPS_SOURCES
    ccl/test_persistent_fabric_ccl_ops.cpp
    ccl/test_send_recv_ops.cpp
)

set(UNIT_TESTS_TTNN_CCL_MULTI_TENSOR_SOURCES ccl/test_multi_tensor_ccl.cpp)

set(UNIT_TESTS_TTNN_ACCESSOR_SOURCES
    accessor/common.cpp
    accessor/test_accessor_benchmarks.cpp
    accessor/test_tensor_accessor.cpp
    accessor/test_tensor_accessor_on_device.cpp
)

set(UNIT_TESTS_TTNN_TENSOR_SOURCES
    tensor/common_tensor_test_utils.cpp
    tensor/test_create_tensor.cpp
    tensor/test_create_tensor_multi_device.cpp
    tensor/test_create_tensor_with_layout.cpp
    tensor/test_distributed_tensor.cpp
    tensor/test_tensor_topology.cpp
    tensor/test_mesh_tensor.cpp
    tensor/test_partition.cpp
    tensor/test_tensor_layout.cpp
    tensor/test_tensor_nd_sharding.cpp
    tensor/test_tensor_serialization.cpp
    tensor/test_unit_mesh_utils.cpp
    tensor/test_vector_conversion.cpp
    tensor/test_xtensor_adapter.cpp
    tensor/test_xtensor_conversion.cpp
)

set(TEST_CCL_MULTI_CQ_MULTI_DEVICE_SOURCES multi_thread/test_ccl_multi_cq_multi_device.cpp)

set(UNIT_TESTS_TTNN_EMITC_SOURCES emitc/test_sanity.cpp)
