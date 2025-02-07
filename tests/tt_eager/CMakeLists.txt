add_library(test_eager_common_libs INTERFACE)
target_link_libraries(test_eager_common_libs INTERFACE test_common_libs)

set(TT_EAGER_TESTS_OPS
    ops/test_average_pool.cpp
    ops/test_eltwise_binary_op.cpp
    ops/test_eltwise_unary_op.cpp
    ops/test_softmax_op.cpp
    ops/test_layernorm_op.cpp
    ops/test_bcast_op.cpp
    ops/test_bmm_op.cpp
    # ops/test_pad_op.cpp                     # <- not called in run_tt_eager.py
    ops/test_sfpu.cpp
    ops/test_sliding_window_ops.cpp
    ops/test_fold_op.cpp
)

set(TT_EAGER_TESTS_TENSORS
    tensors/test_copy_and_move.cpp
    tensors/test_host_device_loopback.cpp
    tensors/test_raw_host_memory_pointer.cpp
    # tensors/test_sharded_loopback.cpp       # <- not called in run_tt_eager.py
    tensors/test_async_tensor_apis.cpp
)

set(TT_EAGER_TESTS_INTEGRATION integration_tests/test_bert.cpp)

set(EAGER_TEST_TARGETS "") # list of all eager test targets, used in CREATE_EAGER_TEST_EXE
include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)

CREATE_EAGER_TEST_EXE("${TT_EAGER_TESTS_OPS}")
CREATE_EAGER_TEST_EXE("${TT_EAGER_TESTS_TENSORS}")
CREATE_EAGER_TEST_EXE("${TT_EAGER_TESTS_INTEGRATION}")

add_custom_target(eager_tests DEPENDS ${EAGER_TEST_TARGETS})
