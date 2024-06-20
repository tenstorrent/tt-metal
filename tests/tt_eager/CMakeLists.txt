
add_library(test_eager_common_libs INTERFACE)
target_link_libraries(test_eager_common_libs INTERFACE tt_eager test_common_libs)

set(TT_EAGER_TESTS_OPS
    ops/ccl/test_all_gather_utils
    ops/ccl/test_all_gather_sharded_indexing_helpers
    ops/ccl/test_ccl_helpers
    ops/test_average_pool
    ops/test_eltwise_binary_op
    ops/test_eltwise_unary_op
    ops/test_softmax_op
    ops/test_layernorm_op
    # ops/test_moreh_adam_op                  # <- not called in run_tt_eager.py
    # ops/test_moreh_matmul_op                # <- not called in run_tt_eager.py
    # ops/test_moreh_layernorm_op             # <- not called in run_tt_eager.py
    ops/test_multi_queue_api
    ops/test_transpose_op
    ops/test_transpose_wh_single_core
    ops/test_transpose_wh_multi_core
    ops/test_reduce_op
    ops/test_bcast_op
    ops/test_bmm_op
    # ops/test_pad_op                         # <- not called in run_tt_eager.py
    ops/test_tilize_op
    ops/test_tilize_zero_padding
    ops/test_tilize_op_channels_last
    ops/test_tilize_zero_padding_channels_last
    ops/test_sfpu
    ops/test_sliding_window_ops
    ops/test_fold_op
)

set(TT_EAGER_TESTS_TENSORS
    tensors/test_copy_and_move
    tensors/test_host_device_loopback
    tensors/test_raw_host_memory_pointer
    # tensors/test_sharded_loopback           # <- not called in run_tt_eager.py
    tensors/test_async_tensor_apis
)

set(TT_EAGER_TESTS_INTEGRATION
    integration_tests/test_bert
)

set(EAGER_TEST_TARGETS "")  # list of all eager test targets, used in CREATE_EAGER_TEST_EXE
include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

CREATE_EAGER_TEST_EXE("${TT_EAGER_TESTS_OPS}")
CREATE_EAGER_TEST_EXE("${TT_EAGER_TESTS_TENSORS}")
CREATE_EAGER_TEST_EXE("${TT_EAGER_TESTS_INTEGRATION}")

add_custom_target(eager_tests DEPENDS ${EAGER_TEST_TARGETS})
