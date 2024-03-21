# Every variable in subdir must be prefixed with subdir (emulating a namespace)
TT_EAGER_TESTS += \
		 tests/tt_eager/dtx/unit_tests \
		 tests/tt_eager/dtx/tensor \
		 tests/tt_eager/dtx/overlap \
		 tests/tt_eager/dtx/collapse_transformations \
		 tests/tt_eager/dtx/test_dtx \
		 tests/tt_eager/dtx/test_dtx_tilized_row_to_col_major \
		 tests/tt_eager/ops/ccl/test_all_gather_utils \
		 tests/tt_eager/ops/test_average_pool \
		 tests/tt_eager/ops/test_eltwise_binary_op \
		 tests/tt_eager/ops/test_eltwise_unary_op \
		 tests/tt_eager/ops/test_softmax_op \
		 tests/tt_eager/ops/test_layernorm_op \
		 tests/tt_eager/ops/test_moreh_adam_op \
		 tests/tt_eager/ops/test_moreh_matmul_op \
		 tests/tt_eager/ops/test_moreh_layernorm_op \
		 tests/tt_eager/ops/test_multi_queue_api \
		 tests/tt_eager/ops/test_transpose_op \
		 tests/tt_eager/ops/test_transpose_wh_single_core \
		 tests/tt_eager/ops/test_transpose_wh_multi_core \
		 tests/tt_eager/ops/test_reduce_op \
		 tests/tt_eager/ops/test_bcast_op \
		 tests/tt_eager/ops/test_bmm_op \
		 tests/tt_eager/ops/test_pad_op \
		 tests/tt_eager/ops/test_tilize_op \
		 tests/tt_eager/ops/test_tilize_zero_padding \
		 tests/tt_eager/ops/test_tilize_op_channels_last \
		 tests/tt_eager/ops/test_tilize_zero_padding_channels_last \
		 tests/tt_eager/ops/test_sfpu \
		 tests/tt_eager/ops/test_fold_op \
		 tests/tt_eager/tensors/test_copy_and_move \
		 tests/tt_eager/tensors/test_host_device_loopback \
		 tests/tt_eager/tensors/test_raw_host_memory_pointer \
		 tests/tt_eager/tensors/test_sharded_loopback \
		 tests/tt_eager/tensors/test_async_tensor_apis \
		 tests/tt_eager/integration_tests/test_bert \

TT_EAGER_TESTS_SRCS = $(addprefix tests/tt_eager/, $(addsuffix .cpp, $(TT_EAGER_TESTS:tests/%=%)))

TT_EAGER_TESTS_INCLUDES = $(TEST_INCLUDES) $(TT_EAGER_INCLUDES)
TT_EAGER_TESTS_LDFLAGS = $(TT_METAL_TESTS_LDFLAGS) $(TT_LIB_LDFLAGS) -lgtest -lgtest_main

TT_EAGER_TESTS_OBJS = $(addprefix $(OBJDIR)/, $(TT_EAGER_TESTS_SRCS:.cpp=.o))
TT_EAGER_TESTS_DEPS = $(addprefix $(OBJDIR)/, $(TT_EAGER_TESTS_SRCS:.cpp=.d))

-include $(TT_EAGER_TESTS_DEPS)

tests/tt_eager: $(TT_EAGER_TESTS)
tests/tt_eager/%: $(TESTDIR)/tt_eager/%;

.PRECIOUS: $(TESTDIR)/tt_eager/%
$(TESTDIR)/tt_eager/%: $(OBJDIR)/tt_eager/tests/%.o $(TT_DNN_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_EAGER_TESTS_INCLUDES) -o $@ $^ $(TT_EAGER_TESTS_LDFLAGS)

.PRECIOUS: $(OBJDIR)/tt_eager/tests/%.o
$(OBJDIR)/tt_eager/tests/%.o: tests/tt_eager/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(TT_EAGER_TESTS_INCLUDES) -c -o $@ $<
