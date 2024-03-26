TT_DNN_SRCS = \
	tt_eager/tt_dnn/op_library/auto_format.cpp \
	tt_eager/tt_dnn/op_library/data_transfer/data_transfer_op.cpp \
	tt_eager/tt_dnn/op_library/layout_conversion/layout_conversion_op.cpp \
	tt_eager/tt_dnn/op_library/all_gather/all_gather_op.cpp \
	tt_eager/tt_dnn/op_library/all_gather/multi_core/all_gather_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/sharded/sharded_op.cpp \
	tt_eager/tt_dnn/op_library/sharded/multi_core/sharded_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/sharded_partial/sharded_op_partial.cpp \
	tt_eager/tt_dnn/op_library/sharded_partial/multi_core/sharded_op_partial_multi_core.cpp \
	tt_eager/tt_dnn/op_library/copy/copy_op.cpp \
	tt_eager/tt_dnn/op_library/copy/single_core/copy_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/copy/multi_core/copy_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/move/move_op.cpp \
	tt_eager/tt_dnn/op_library/move/single_core/move_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/move/multi_core/move_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/move/multi_core/move_op_multi_core_overlap.cpp \
	tt_eager/tt_dnn/op_library/move/multi_core/move_op_multi_core_sharded.cpp \
	tt_eager/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.cpp \
	tt_eager/tt_dnn/op_library/eltwise_binary/single_core/eltwise_binary_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/eltwise_binary/multi_core/eltwise_binary_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.cpp \
	tt_eager/tt_dnn/op_library/eltwise_unary/single_core/eltwise_unary_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/eltwise_unary/multi_core/eltwise_unary_op_sharded.cpp \
	tt_eager/tt_dnn/op_library/pad/pad_op.cpp \
	tt_eager/tt_dnn/op_library/pad/pad_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/unpad/single_core/unpad_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/unpad/multi_core/unpad_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/unpad/unpad_op.cpp \
	tt_eager/tt_dnn/op_library/indexed_fill/multi_core/indexed_fill_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/indexed_fill/indexed_fill_op.cpp \
	tt_eager/tt_dnn/op_library/non_zero_indices/single_core/non_zero_indices_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/non_zero_indices/non_zero_indices_op.cpp \
	tt_eager/tt_dnn/op_library/fill_rm/fill_rm_op.cpp \
	tt_eager/tt_dnn/op_library/fully_connected/fully_connected_op.cpp \
	tt_eager/tt_dnn/op_library/pool/average_pool.cpp \
	tt_eager/tt_dnn/op_library/pool/max_pool.cpp \
	tt_eager/tt_dnn/op_library/pool/max_pool_single_core.cpp \
	tt_eager/tt_dnn/op_library/pool/max_pool_multi_core.cpp \
	tt_eager/tt_dnn/op_library/transpose/transpose_op.cpp \
	tt_eager/tt_dnn/op_library/transpose/wh_multi_core/transpose_wh_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/transpose/hc_multi_core/transpose_hc_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/transpose/single_core/transpose_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/reduce/reduce_op.cpp \
	tt_eager/tt_dnn/op_library/reduce/single_core/reduce_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/reduce/multi_core_h/reduce_op_multi_core_h.cpp \
	tt_eager/tt_dnn/op_library/reduce/multi_core_w/reduce_op_multi_core_w.cpp \
	tt_eager/tt_dnn/op_library/bcast/bcast_op.cpp \
	tt_eager/tt_dnn/op_library/bcast/single_core/bcast_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/bcast/multi_core_h/bcast_op_multi_core_h.cpp \
	tt_eager/tt_dnn/op_library/bcast/multi_core_w/bcast_op_multi_core_w.cpp \
	tt_eager/tt_dnn/op_library/bcast/multi_core_hw/bcast_op_multi_core_hw.cpp \
	tt_eager/tt_dnn/op_library/bmm/bmm_op.cpp \
	tt_eager/tt_dnn/op_library/bmm/single_core/bmm_op_single_core_tilize_untilize.cpp \
	tt_eager/tt_dnn/op_library/bmm/single_core/bmm_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/bmm/multi_core/bmm_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/bmm/multi_core_reuse/bmm_op_multi_core_reuse.cpp \
	tt_eager/tt_dnn/op_library/bmm/multi_core_reuse_padding/bmm_op_multi_core_reuse_padding.cpp \
	tt_eager/tt_dnn/op_library/bmm/multi_core_reuse_mcast_1d_optimized/bmm_op_multi_core_reuse_mcast_1d_optimized.cpp \
	tt_eager/tt_dnn/op_library/bmm/multi_core_reuse_mcast_2d_optimized/bmm_op_multi_core_reuse_mcast_2d_optimized.cpp \
	tt_eager/tt_dnn/op_library/bmm/multi_core_reuse_optimized/bmm_op_multi_core_reuse_optimized.cpp \
	tt_eager/tt_dnn/op_library/downsample/downsample_op.cpp \
	tt_eager/tt_dnn/op_library/conv/conv_op.cpp \
	tt_eager/tt_dnn/op_library/conv/optimized_conv_op.cpp \
	tt_eager/tt_dnn/op_library/conv/multi_core_optimized_conv/optimized_conv_op.cpp \
	tt_eager/tt_dnn/op_library/conv/multi_core_optimized_conv_sharded/optimized_conv_op_sharded.cpp \
	tt_eager/tt_dnn/op_library/conv/multi_core_optimized_conv_sharded/optimized_conv_op_sharded_v2.cpp \
	tt_eager/tt_dnn/op_library/tilize/tilize_multi_core/tilize_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/tilize/tilize_single_core/tilize_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/tilize/tilize_op.cpp \
	tt_eager/tt_dnn/op_library/untilize/multi_core/untilize_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/untilize/single_core/untilize_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/untilize/untilize_op.cpp \
	tt_eager/tt_dnn/op_library/untilize/untilize_with_halo_op.cpp \
	tt_eager/tt_dnn/op_library/untilize/untilize_with_halo_op_v2.cpp \
	tt_eager/tt_dnn/op_library/softmax/multi_core/softmax_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/softmax/softmax_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_helper_functions.cpp \
	tt_eager/tt_dnn/op_library/moreh_adam/moreh_adam.cpp \
	tt_eager/tt_dnn/op_library/moreh_adam/moreh_adam_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_adamw/moreh_adamw.cpp \
	tt_eager/tt_dnn/op_library/moreh_adamw/moreh_adamw_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_arange/moreh_arange_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step1/moreh_clip_grad_norm_step1.cpp \
	tt_eager/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step2/moreh_clip_grad_norm_step2.cpp \
	tt_eager/tt_dnn/op_library/moreh_clip_grad_norm/moreh_clip_grad_norm_step3/moreh_clip_grad_norm_step3.cpp \
  tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_op.cpp \
  tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step1/moreh_nll_loss_step1.cpp \
  tt_eager/tt_dnn/op_library/moreh_nll_loss/moreh_nll_loss_step2/moreh_nll_loss_step2.cpp \
  tt_eager/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward_op.cpp \
  tt_eager/tt_dnn/op_library/moreh_nll_loss_backward/moreh_nll_loss_backward/moreh_nll_loss_backward.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax/moreh_softmax_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax/softmax_w_small/softmax_w_small.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax/softmax_h_small/softmax_h_small.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax/softmax_w_large/softmax_w_large.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax/softmax_h_large/softmax_h_large.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax/softmax_c_large/softmax_c_large.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax_backward/moreh_softmax_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax_backward/softmax_backward_w_small/softmax_backward_w_small.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax_backward/softmax_backward_h_small/softmax_backward_h_small.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax_backward/softmax_backward_w_large/softmax_backward_w_large.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax_backward/softmax_backward_h_large/softmax_backward_h_large.cpp \
	tt_eager/tt_dnn/op_library/moreh_softmax_backward/softmax_backward_c_large/softmax_backward_c_large.cpp \
	tt_eager/tt_dnn/op_library/moreh_sum/moreh_sum_h/moreh_sum_h.cpp \
	tt_eager/tt_dnn/op_library/moreh_sum/moreh_sum_w/moreh_sum_w.cpp \
	tt_eager/tt_dnn/op_library/moreh_sum/moreh_sum_nc/moreh_sum_nc.cpp \
	tt_eager/tt_dnn/op_library/moreh_sum/moreh_sum_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward.cpp \
	tt_eager/tt_dnn/op_library/moreh_sum_backward/moreh_sum_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_mean/moreh_mean_h/moreh_mean_h.cpp \
	tt_eager/tt_dnn/op_library/moreh_mean/moreh_mean_w/moreh_mean_w.cpp \
	tt_eager/tt_dnn/op_library/moreh_mean/moreh_mean_nc/moreh_mean_nc.cpp \
	tt_eager/tt_dnn/op_library/moreh_mean/moreh_mean_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_mean_backward/moreh_mean_backward.cpp \
	tt_eager/tt_dnn/op_library/moreh_mean_backward/moreh_mean_backward_op.cpp \
	tt_eager/tt_dnn/op_library/layernorm/multi_core/layernorm_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/layernorm/layernorm_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_bmm/moreh_bmm_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_bmm_backward/moreh_bmm_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_linear/moreh_linear_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_linear_backward/moreh_linear_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_linear_backward/bias_backward_h/moreh_bias_backward_multi_core_h.cpp \
	tt_eager/tt_dnn/op_library/moreh_linear_backward/bias_backward_hw/moreh_bias_backward_single_core_hw.cpp \
	tt_eager/tt_dnn/op_library/moreh_matmul/multi_core/moreh_matmul_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/moreh_matmul/moreh_matmul_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_matmul_backward/moreh_matmul_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_h/moreh_norm_h.cpp \
	tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_w/moreh_norm_w.cpp \
	tt_eager/tt_dnn/op_library/moreh_norm/moreh_norm_other/moreh_norm_other.cpp \
	tt_eager/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_norm_backward/moreh_norm_backward.cpp \
	tt_eager/tt_dnn/op_library/moreh_dot/single_core/moreh_dot_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/moreh_dot/moreh_dot_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_dot_backward/single_core/moreh_dot_backward_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/moreh_dot_backward/moreh_dot_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_layernorm/moreh_layernorm_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_layernorm_backward/moreh_layernorm_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_layernorm_backward/input_grad/moreh_layernorm_backward_input_grad.cpp \
	tt_eager/tt_dnn/op_library/moreh_layernorm_backward/gamma_beta_grad/moreh_layernorm_backward_gamma_beta_grad.cpp \
	tt_eager/tt_dnn/op_library/moreh_groupnorm/moreh_groupnorm_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_groupnorm/moreh_groupnorm.cpp \
	tt_eager/tt_dnn/op_library/moreh_groupnorm_backward/moreh_groupnorm_backward_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_groupnorm_backward/input_grad/moreh_groupnorm_backward_input_grad.cpp \
	tt_eager/tt_dnn/op_library/moreh_groupnorm_backward/gamma_beta_grad/moreh_groupnorm_backward_gamma_beta_grad.cpp \
	tt_eager/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_nc/moreh_cumsum_nc.cpp \
	tt_eager/tt_dnn/op_library/moreh_cumsum/moreh_cumsum_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_sgd/moreh_sgd_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_sgd/moreh_sgd.cpp \
	tt_eager/tt_dnn/op_library/groupnorm/groupnorm_op.cpp \
	tt_eager/tt_dnn/op_library/reshape/reshape_op.cpp \
	tt_eager/tt_dnn/op_library/permute/permute_op.cpp \
	tt_eager/tt_dnn/op_library/composite/composite_ops.cpp\
	tt_eager/tt_dnn/op_library/backward/backward_ops.cpp\
	tt_eager/tt_dnn/op_library/optimizer/optimizer_ops.cpp\
	tt_eager/tt_dnn/op_library/complex/complex_ops.cpp\
	tt_eager/tt_dnn/op_library/loss/loss_op.cpp\
	tt_eager/tt_dnn/op_library/transformer_tms/transformer_tms.cpp \
	tt_eager/tt_dnn/op_library/transformer_tms/multi_core_split_query_key_value_and_split_heads/multi_core_split_query_key_value_and_split_heads.cpp \
	tt_eager/tt_dnn/op_library/transformer_tms/multi_core_concatenate_heads/multi_core_concatenate_heads.cpp \
	tt_eager/tt_dnn/op_library/transformer_tms/multi_core_attn_matmul/multi_core_attn_matmul.cpp \
	tt_eager/tt_dnn/op_library/transformer_tms/multi_core_group_attn_matmul/multi_core_group_attn_matmul.cpp \
	tt_eager/tt_dnn/op_library/run_operation.cpp \
	tt_eager/tt_dnn/op_library/split/split_tiled.cpp \
	tt_eager/tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.cpp \
	tt_eager/tt_dnn/op_library/operation_history.cpp \
	tt_eager/tt_dnn/op_library/concat/multi_core/concat_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/concat/single_core/concat_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/concat/concat_op.cpp \
	tt_eager/tt_dnn/op_library/repeat/multi_core/repeat_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/repeat/single_core/repeat_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/repeat/repeat_op.cpp \
	tt_eager/tt_dnn/op_library/nlp_tms/nlp_tms.cpp \
	tt_eager/tt_dnn/op_library/nlp_tms/nlp_create_qkv_heads_falcon7b.cpp \
	tt_eager/tt_dnn/op_library/nlp_tms/nlp_create_qkv_heads.cpp \
	tt_eager/tt_dnn/op_library/nlp_tms/nlp_concat_heads.cpp \
	tt_eager/tt_dnn/op_library/rotate_half/single_core/rotate_half_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/rotate_half/rotate_half_op.cpp \
	tt_eager/tt_dnn/op_library/rotary_embedding/multi_core/rotary_embedding_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/rotary_embedding/single_core/rotary_embedding_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/rotary_embedding/rotary_embedding_op.cpp \
	tt_eager/tt_dnn/op_library/embeddings/embeddings_op.cpp \
	tt_eager/tt_dnn/op_library/update_cache/multi_core/update_cache_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/update_cache/single_core/update_cache_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/update_cache/update_cache_op.cpp \
	tt_eager/tt_dnn/op_library/upsample/multi_core/upsample_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/upsample/single_core/upsample_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/upsample/upsample_op.cpp \
	tt_eager/tt_dnn/op_library/fold/fold_op.cpp \
	tt_eager/tt_dnn/op_library/fold/single_core/fold_op_single_core.cpp \
	tt_eager/tt_dnn/op_library/fold/multi_core/fold_op_multi_core.cpp \
	tt_eager/tt_dnn/op_library/moreh_getitem/moreh_getitem_op.cpp \
	tt_eager/tt_dnn/op_library/moreh_getitem/moreh_getitem_rm/moreh_getitem_rm.cpp \
	tt_eager/tt_dnn/op_library/moreh_getitem/moreh_getitem_tilized/moreh_getitem_tilized.cpp \

TT_DNN_LIB = $(LIBDIR)/libtt_dnn.a
TT_DNN_DEFINES =
TT_DNN_INCLUDES = $(TT_EAGER_INCLUDES)
TT_DNN_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_DNN_OBJS = $(addprefix $(OBJDIR)/, $(TT_DNN_SRCS:.cpp=.o))
TT_DNN_DEPS = $(addprefix $(OBJDIR)/, $(TT_DNN_SRCS:.cpp=.d))

-include $(TT_DNN_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
tt_eager/tt_dnn: $(TT_DNN_LIB)

$(TT_DNN_LIB): $(COMMON_LIB) $(DTX_LIB) $(TT_DNN_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $(TT_DNN_OBJS)

$(OBJDIR)/tt_eager/tt_dnn/%.o: tt_eager/tt_dnn/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_DNN_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_DNN_INCLUDES) $(TT_DNN_DEFINES) -c -o $@ $<
