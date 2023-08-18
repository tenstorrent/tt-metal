TT_DNN_SRCS = \
	libs/tt_dnn/op_library/auto_format.cpp \
	libs/tt_dnn/op_library/data_transfer/data_transfer_op.cpp \
	libs/tt_dnn/op_library/layout_conversion/layout_conversion_op.cpp \
	libs/tt_dnn/op_library/move/move_op.cpp \
	libs/tt_dnn/op_library/move/single_core/move_op_single_core.cpp \
	libs/tt_dnn/op_library/move/multi_core/move_op_multi_core.cpp \
	libs/tt_dnn/op_library/move/multi_core/move_op_multi_core_overlap.cpp \
	libs/tt_dnn/op_library/eltwise_binary/eltwise_binary_op.cpp \
	libs/tt_dnn/op_library/eltwise_binary/single_core/eltwise_binary_op_single_core.cpp \
	libs/tt_dnn/op_library/eltwise_binary/multi_core/eltwise_binary_op_multi_core.cpp \
	libs/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.cpp \
	libs/tt_dnn/op_library/eltwise_unary/single_core/eltwise_unary_op_single_core.cpp \
	libs/tt_dnn/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.cpp \
	libs/tt_dnn/op_library/pad/pad_op.cpp \
	libs/tt_dnn/op_library/unpad/unpad_op.cpp \
	libs/tt_dnn/op_library/fill_rm/fill_rm_op.cpp \
	libs/tt_dnn/op_library/fully_connected/fully_connected_op.cpp \
	libs/tt_dnn/op_library/pool/average_pool.cpp \
	libs/tt_dnn/op_library/pool/max_pool.cpp \
	libs/tt_dnn/op_library/transpose/transpose_op.cpp \
	libs/tt_dnn/op_library/transpose/wh_multi_core/transpose_wh_op_multi_core.cpp \
	libs/tt_dnn/op_library/transpose/hc_multi_core/transpose_hc_op_multi_core.cpp \
	libs/tt_dnn/op_library/transpose/single_core/transpose_op_single_core.cpp \
	libs/tt_dnn/op_library/reduce/reduce_op.cpp \
	libs/tt_dnn/op_library/reduce/single_core/reduce_op_single_core.cpp \
	libs/tt_dnn/op_library/reduce/multi_core_h/reduce_op_multi_core_h.cpp \
	libs/tt_dnn/op_library/reduce/multi_core_w/reduce_op_multi_core_w.cpp \
	libs/tt_dnn/op_library/bcast/bcast_op.cpp \
	libs/tt_dnn/op_library/bcast/single_core/bcast_op_single_core.cpp \
	libs/tt_dnn/op_library/bcast/multi_core_h/bcast_op_multi_core_h.cpp \
	libs/tt_dnn/op_library/bcast/multi_core_w/bcast_op_multi_core_w.cpp \
	libs/tt_dnn/op_library/bcast/multi_core_hw/bcast_op_multi_core_hw.cpp \
	libs/tt_dnn/op_library/bmm/bmm_op.cpp \
	libs/tt_dnn/op_library/bmm/single_core/bmm_op_single_core_tilize_untilize.cpp \
	libs/tt_dnn/op_library/bmm/single_core/bmm_op_single_core.cpp \
	libs/tt_dnn/op_library/bmm/multi_core/bmm_op_multi_core.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse/bmm_op_multi_core_reuse.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast/bmm_op_multi_core_reuse_mcast.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_generalized/bmm_op_multi_core_reuse_generalized.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_generalized/bmm_op_multi_core_reuse_mcast_generalized.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_padding/bmm_op_multi_core_reuse_padding.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_padding/bmm_op_multi_core_reuse_mcast_padding.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_1d_optimized/bmm_op_multi_core_reuse_mcast_1d_optimized.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_mcast_2d_optimized/bmm_op_multi_core_reuse_mcast_2d_optimized.cpp \
	libs/tt_dnn/op_library/bmm/multi_core_reuse_optimized/bmm_op_multi_core_reuse_optimized.cpp \
	libs/tt_dnn/op_library/conv/conv_op.cpp \
	libs/tt_dnn/op_library/tilize/tilize_op.cpp \
	libs/tt_dnn/op_library/untilize/untilize_op.cpp \
	libs/tt_dnn/op_library/softmax/softmax_op.cpp \
	libs/tt_dnn/op_library/layernorm/layernorm_op.cpp \
	libs/tt_dnn/op_library/groupnorm/groupnorm_op.cpp \
	libs/tt_dnn/op_library/reshape/reshape_op.cpp \
	libs/tt_dnn/op_library/permute/permute_op.cpp \
	libs/tt_dnn/op_library/composite/composite_ops.cpp\
	libs/tt_dnn/op_library/transformer_tms/transformer_tms.cpp \
	libs/tt_dnn/op_library/transformer_tms/multi_core_split_fused_qkv_and_split_heads/multi_core_split_fused_qkv_and_split_heads.cpp \
	libs/tt_dnn/op_library/transformer_tms/multi_core_concatenate_heads/multi_core_concatenate_heads.cpp \
	libs/tt_dnn/op_library/transformer_tms/multi_core_attn_matmul/multi_core_attn_matmul.cpp \
	libs/tt_dnn/op_library/run_operation.cpp \
	libs/tt_dnn/op_library/split/split_tiled.cpp \
	libs/tt_dnn/op_library/split/split_last_dim_two_chunks_tiled.cpp \
	libs/tt_dnn/op_library/operation_history.cpp \
	libs/tt_dnn/op_library/concat/concat_op.cpp \
	libs/tt_dnn/op_library/nlp_tms/nlp_tms.cpp \
	libs/tt_dnn/op_library/nlp_tms/nlp_create_qkv_heads.cpp \
	libs/tt_dnn/op_library/nlp_tms/nlp_concat_heads.cpp \
	libs/tt_dnn/op_library/rotate_half/single_core/rotate_half_op_single_core.cpp \
	libs/tt_dnn/op_library/rotate_half/rotate_half_op.cpp \
	libs/tt_dnn/op_library/rotary_embedding/multi_core/rotary_embedding_op_multi_core.cpp \
	libs/tt_dnn/op_library/rotary_embedding/single_core/rotary_embedding_op_single_core.cpp \
	libs/tt_dnn/op_library/rotary_embedding/rotary_embedding_op.cpp \
	libs/tt_dnn/op_library/embeddings/embeddings_op.cpp \
	libs/tt_dnn/op_library/update_cache/multi_core/update_cache_op_multi_core.cpp \
	libs/tt_dnn/op_library/update_cache/single_core/update_cache_op_single_core.cpp \
	libs/tt_dnn/op_library/update_cache/update_cache_op.cpp \


TT_DNN_LIB = $(LIBDIR)/libtt_dnn.a
TT_DNN_DEFINES =
TT_DNN_INCLUDES = $(LIBS_INCLUDES)
TT_DNN_LDFLAGS = -lcommon -lllrt -ltt_metal -ltensor -ldtx
TT_DNN_CFLAGS = $(CFLAGS) -Werror -Wno-int-to-pointer-cast

TT_DNN_OBJS = $(addprefix $(OBJDIR)/, $(TT_DNN_SRCS:.cpp=.o))
TT_DNN_DEPS = $(addprefix $(OBJDIR)/, $(TT_DNN_SRCS:.cpp=.d))

-include $(TT_DNN_DEPS)

# Each module has a top level target as the entrypoint which must match the subdir name
libs/tt_dnn: $(TT_DNN_LIB)

$(TT_DNN_LIB): $(COMMON_LIB) $(DTX_LIB) $(TT_DNN_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs -o $@ $(TT_DNN_OBJS)

$(OBJDIR)/libs/tt_dnn/%.o: libs/tt_dnn/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(TT_DNN_CFLAGS) $(CXXFLAGS) $(STATIC_LIB_FLAGS) $(TT_DNN_INCLUDES) $(TT_DNN_DEFINES) -c -o $@ $<
