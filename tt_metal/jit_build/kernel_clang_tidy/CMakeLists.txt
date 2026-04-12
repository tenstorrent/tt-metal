# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# CMake target for running clang-tidy on kernel (device) code.
#
# Kernel code is normally compiled at runtime via JIT with a RISC-V cross-compiler.
# This target re-compiles the same source files with the *host* clang compiler so
# that clang-tidy can analyse them.  The target is EXCLUDE_FROM_ALL — it is only
# built on demand (e.g. via `cmake --build <dir> --target kernel_clang_tidy`).
#
# Key differences from actual JIT compilation:
#   • Host clang instead of riscv-tt-elf-g++
#   • No RISC-V flags (-mcpu=tt-wh etc.)
#   • -Wno-unknown-attributes to tolerate RISC-V __attribute__ extensions
#   • A prelude header provides stub values for JIT-generated content
#   • Fake KERNEL_COMPILE_TIME_ARGS so get_compile_time_arg_val() resolves
#
# Because the goal is static analysis (not code generation), perfect compilation
# is not required — we just need clang's frontend to parse the code.

# ── Gather all kernel source files ──────────────────────────────────────────

set(ALL_KERNEL_FILES "")

file(
    GLOB_RECURSE TT_METAL_KERNELS
    "${CMAKE_SOURCE_DIR}/tt_metal/kernels/*.cpp"
    "${CMAKE_SOURCE_DIR}/tt_metal/kernels/*.cc"
)
file(
    GLOB_RECURSE TT_METAL_DISPATCH_KERNELS
    "${CMAKE_SOURCE_DIR}/tt_metal/impl/dispatch/kernels/*.cpp"
    "${CMAKE_SOURCE_DIR}/tt_metal/impl/dispatch/kernels/*.cc"
)
file(
    GLOB_RECURSE TT_METAL_FABRIC_KERNELS
    "${CMAKE_SOURCE_DIR}/tt_metal/fabric/impl/kernels/*.cpp"
    "${CMAKE_SOURCE_DIR}/tt_metal/fabric/impl/kernels/*.cc"
)

# Only include files from kernel/ or kernels_ng/ directories under operations
file(
    GLOB_RECURSE ALL_TTNN_FILES
    "${CMAKE_SOURCE_DIR}/ttnn/cpp/ttnn/operations/*.cpp"
    "${CMAKE_SOURCE_DIR}/ttnn/cpp/ttnn/operations/*.cc"
)
set(TTNN_OPERATION_KERNELS "")
foreach(file ${ALL_TTNN_FILES})
    if(file MATCHES ".*/kernels/|.*/kernels_ng/")
        list(APPEND TTNN_OPERATION_KERNELS ${file})
    endif()
endforeach()

list(
    APPEND
    ALL_KERNEL_FILES
    ${TT_METAL_KERNELS}
    ${TT_METAL_DISPATCH_KERNELS}
    ${TT_METAL_FABRIC_KERNELS}
    ${TTNN_OPERATION_KERNELS}
)

# ── Exclusion list ───────────────────────────────────────────────────────────
# Files excluded from clang-tidy analysis because they have deep structural
# issues that cannot be solved by stub values alone:
#
# Fabric EDM routers (edm_fabric/):
#   ChannelAllocations<0,1,1> template with KERNEL_COMPILE_TIME_ARGS=1 produces
#   an invalid instantiation (missing members, failing static_asserts on
#   MY_ERISC_ID < NUM_ACTIVE_ERISCS, etc.).  These kernels require a genuine
#   compile-time configuration that cannot be captured with dummy arg values.
#
# CCL all-to-all / broadcast / ccl_send kernels:
#   • all_to_all_{combine,dispatch}: DEST_CHIP_ID / DEST_MESH_ID / DIRECTIONS
#     are JIT-generated array-initialiser macros; even with stubs, TensorAccessor
#     template args from KERNEL_COMPILE_TIME_ARGS=1 produce an invalid
#     instantiation (TensorAccessorArgs<7U,4294967295U>) that no get_noc_addr
#     overload accepts.
#   • ccl_send: coord_t naming conflict between ttnn::ccl::coord_t and an
#     existing typedef, plus a coord_t size static_assert.
#   • ccl_send_reader_two_input: static_assert on is_sharded||is_interleaved
#     fires with dummy compile-time args.
#   • moe_utils.hpp (included by all_to_all_combine): auto return type deduced
#     as tuple<bool,uint32_t&> in one branch and tuple<bool,unsigned long> in
#     another — a real code-quality issue unrelated to our stubs.
set(KCT_EXCLUDE_PATTERNS
    # Fabric EDM routers
    ".*/edm_fabric/fabric_erisc_router\\.cpp$"
    ".*/edm_fabric/fabric_router_mux_extension\\.cpp$"
    ".*/edm_fabric/fabric_router_relay_extension\\.cpp$"
    ".*/edm_fabric/fabric_router_udm_mux_extension\\.cpp$"
    # CCL all-to-all
    ".*/ccl/all_to_all_combine/.*/reader_all_to_all_combine\\.cpp$"
    ".*/ccl/all_to_all_combine/.*/writer_all_to_all_combine\\.cpp$"
    ".*/ccl/all_to_all_dispatch/.*/reader_all_to_all_dispatch\\.cpp$"
    ".*/ccl/all_to_all_dispatch/.*/writer_all_to_all_dispatch\\.cpp$"
    # CCL broadcast / send
    ".*/ccl/broadcast/.*/broadcast_rm_reader\\.cpp$"
    ".*/ccl/common/kernels/ccl_send\\.cpp$"
    ".*/ccl/common/kernels/ccl_send_reader_two_input\\.cpp$"
    # Bcast compute kernels: require BCAST_LLKOP and BCAST_OP macros to be
    # defined at JIT compile time (-D flags). The KCT build cannot supply them.
    ".*/data_movement/bcast/device/kernels/compute/bcast_h\\.cpp$"
    ".*/data_movement/bcast/device/kernels/compute/bcast_h_sharded_optimised\\.cpp$"
    ".*/data_movement/bcast/device/kernels/compute/bcast_hw\\.cpp$"
    ".*/data_movement/bcast/device/kernels/compute/bcast_w\\.cpp$"
    # Typecast compute kernel: requires TYPECAST_LLK_INIT / TYPECAST_LLK macros
    ".*/copy/typecast/device/kernels/compute/eltwise_typecast\\.cpp$"
    # clone/read_kernel_rm: TensorAccessorArgs<2U,4294967295U> produces invalid
    # template instantiation when KERNEL_COMPILE_TIME_ARGS=1,... is used.
    ".*/clone/device/kernels/read_kernel_rm\\.cpp$"
    # conv2d kernels: static assertions (tile_size % 512, input_cb != output_cb,
    # constexpr ntile_width) fire with the default all-1 compile-time args.
    ".*/conv/conv2d/device/kernels/activation_reader_width_sharded\\.cpp$"
    ".*/conv/conv2d/device/kernels/compute_depthwise_conv1d\\.cpp$"
    ".*/conv/conv2d/device/kernels/reader_conv_activations_2d_mcast_padded_with_halo_3x3_weights_v2\\.cpp$"
    ".*/conv/conv2d/device/kernels/reader_conv_activations_padded_with_halo_3x3_weights_v2\\.cpp$"
    ".*/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks\\.cpp$"
    ".*/conv/conv2d/device/kernels/reader_writer_tiled_out_1d_mcast_sender_conv_weights_tiled_col_to_rm_blocks\\.cpp$"
    ".*/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_receiver_conv_weights_tiled_col_to_rm_blocks\\.cpp$"
    ".*/conv/conv2d/device/kernels/writer_tiled_out_2d_mcast_sender_conv_weights_tiled_col_to_rm_blocks\\.cpp$"
    # clone write_kernel_rm: uses TensorAccessorArgs without including dataflow_api.h;
    # the TensorAccessor-based get_noc_addr SFINAE also fails with all-1 CT args.
    ".*/clone/device/kernels/write_kernel_rm\\.cpp$"
    # concat kernels: include only <stdint.h> and rely on JIT-provided dataflow
    # API headers.  In the KCT build those functions are undeclared.
    ".*/concat/device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors\\.cpp$"
    ".*/concat/device/kernels/dataflow/reader_height_sharded_width_concat_two_tensors_tiled\\.cpp$"
    ".*/concat/device/kernels/dataflow/reader_s2s_tensor_concat\\.cpp$"
    ".*/concat/device/kernels/dataflow/writer_height_sharded_width_concat_two_tensors_tiled\\.cpp$"
    ".*/concat/device/kernels/dataflow/writer_s2i_width\\.cpp$"
    # Kernels that call get_noc_addr(id, TensorAccessor<sharded_DSpec>).
    # With KERNEL_COMPILE_TIME_ARGS=1,... the is_sharded flag evaluates to 1,
    # making TensorAccessor use the sharded specialisation (not InterleavedAddrGen).
    # The SFINAE resolver (decltype(addrgen.get_noc_addr()) with 0 args) then
    # finds no matching overload.  Fixing this requires valid sharding CT args.
    ".*/fill_pad/device/kernels/dataflow/fill_pad_writer\\.cpp$"
    ".*/fill_rm/device/kernels/dataflow/fill_rm_interleaved\\.cpp$"
    ".*/fold/device/kernels/dataflow/reader_dram2cb_for_rm_input\\.cpp$"
    ".*/fold/device/kernels/dataflow/writer_cb2dram_for_rm_input\\.cpp$"
    ".*/fold/device/kernels/dataflow/writer_cb2dram_for_tiled_input\\.cpp$"
    ".*/indexed_fill/device/kernels/dataflow/indexed_fill_reader\\.cpp$"
    ".*/moe_expert_token_remap/device/kernels/dataflow/writer_moe_expert_token_remap\\.cpp$"
    ".*/moe_routing_remap/device/kernels/dataflow/reader_moe_routing_remap\\.cpp$"
    ".*/moe_routing_remap/device/kernels/dataflow/writer_moe_routing_remap\\.cpp$"
    # Same TensorAccessor/get_noc_addr SFINAE failure as above.
    ".*/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved\\.cpp$"
    ".*/pad/device/kernels/dataflow/writer_pad_dims_rm_interleaved\\.cpp$"
    # writer_pad_dims_rm_sharded_stickwise: static_assert '1U == 2||4' fires
    # (padding_value_num_bytes) with all-1 CT args.
    ".*/pad/device/kernels/dataflow/writer_pad_dims_rm_sharded_stickwise\\.cpp$"
    # non_zero_indices_sc_reader: get_noc_addr SFINAE + secondary parse error
    # ('input_addr_ptr' undeclared — cascade from the first failure).
    ".*/non_zero_indices/device/kernels/dataflow/non_zero_indices_sc_reader\\.cpp$"
    # scatter kernels: df_to_std<DataFormat> primary has std_type=void; with
    # DATA_FORMATS_DEFINED and CB index 1 the format resolves to an unspecialised
    # DataFormat, yielding sizeof(void) errors.
    ".*/scatter/device/kernels/dataflow/reader_scatter\\.cpp$"
    ".*/scatter/device/kernels/dataflow/reader_bf16_reduction_scatter\\.cpp$"
    ".*/scatter/device/kernels/dataflow/writer_scatter\\.cpp$"
    ".*/scatter/device/kernels/dataflow/writer_bf16_reduction_scatter\\.cpp$"
    # reader_unary_sharded: includes only <stdint.h>, relies on JIT-injected
    # dataflow API headers (get_arg_val, cb_reserve_back etc. undeclared).
    ".*/sharded/device/kernels/dataflow/reader_unary_sharded\\.cpp$"
    # nd_reshard kernels: include only <cstdint> + api/tensor/tensor_accessor.h,
    # relying on JIT firmware to provide get_write_ptr, cb_wait_front, etc.
    # Those are only available via api/dataflow/dataflow_api.h which they omit.
    ".*/reshard/device/kernels/nd_reshard_copy_pages_reader\\.cpp$"
    ".*/reshard/device/kernels/nd_reshard_copy_pages_writer\\.cpp$"
    # tilize.cpp: static_assert fixed by tilize_helpers.hpp stub, but kept excluded
    # as a safety net until we verify no other failures remain in that file.
    ".*/tilize/device/kernels/compute/tilize\\.cpp$"
    # tilize_with_val_padding readers: static_assert val_size==2||4 fires with
    # val_size=1 (all-1 KERNEL_COMPILE_TIME_ARGS).
    ".*/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows_multicore\\.cpp$"
    ".*/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_multicore_both_dims\\.cpp$"
    # untilize kernels: same in-place static_assert as tilize; input_cb==output_cb==1
    ".*/untilize/device/kernels/compute/untilize\\.cpp$"
    ".*/untilize/device/kernels/compute/untilize_variable_num_blocks\\.cpp$"
    # eltwise/binary bcast compute kernels: same BCAST_LLKOP/BCAST_OP issue as
    # data_movement/bcast (macros must be defined at JIT compile time).
    ".*/eltwise/binary/device/kernels/compute/bcast_h\\.cpp$"
    ".*/eltwise/binary/device/kernels/compute/bcast_h_sharded_optimised\\.cpp$"
    ".*/eltwise/binary/device/kernels/compute/bcast_hw\\.cpp$"
    ".*/eltwise/binary/device/kernels/compute/bcast_w\\.cpp$"
    # eltwise_binary_sfpu_kernel: SFPI API issues in ckernel_sfpu_div_int32_floor.h
    # (float_to_uint16 renamed to float_to_int16, missing __vDReg::get,
    # SFPLOAD_ADDR_MODE_NOINC, and vInt/vUInt→vFloat conversion failures).
    # Also llk_math_eltwise_binary_sfpu_{gcd,lcm}.h call not found.
    # These are LLK/SFPI code-quality issues, not stub problems.
    ".*/eltwise/binary/device/kernels/compute/eltwise_binary_sfpu_kernel\\.cpp$"
    # eltwise/binary_ng ALL compute kernels: BINARY_OP / BINARY_OP_TYPE / BINARY_SFPU_INIT /
    # BINARY_SFPU_OP / PROCESS_LHS_ACTIVATIONS / PROCESS_RHS_ACTIVATIONS macros not defined
    # at KCT compile time; also SFPI API issues in row/col bcast variants.
    ".*/eltwise/binary_ng/device/kernels/compute/.*\\.cpp$"
    ".*/eltwise/binary_ng/device/kernels_ng/compute/.*\\.cpp$"
    # eltwise/binary_ng dataflow kernels_ng: FILL_TILE_WITH_FIRST_COLUMN_RM /
    # FILL_TILE_WITH_FIRST_ROW_RM macros are JIT-injected for bcast variants.
    ".*/eltwise/binary_ng/device/kernels_ng/dataflow/.*\\.cpp$"
    # eltwise/ternary compute kernels using TERNARY_SFPU_OP_INIT / TERNARY_SFPU_OP_FUNC
    # macros that are JIT-injected; ternary_addc_ops_fpu_rowbcast also uses them.
    ".*/eltwise/ternary/device/kernels/compute/ternary_sfpu_.*\\.cpp$"
    ".*/eltwise/ternary/device/kernels/compute/ternary_addc_ops_sfpu.*\\.cpp$"
    ".*/eltwise/ternary/device/kernels/compute/ternary_addc_ops_fpu_rowbcast\\.cpp$"
    # where_tss_kernel: SFPU_OP_CHAIN_0 macro is JIT-injected via get_block_defines,
    # not available at KCT compile time.
    ".*/eltwise/unary/device/kernels/compute/where_tss_kernel\\.cpp$"
    ".*/eltwise/unary_ng/device/kernels/compute/where_tss_kernel\\.cpp$"
    # embedding/tilize_chunked: cb_id_in0 == cb_id_out0 == 1 with all-1 CT args
    # triggers static_assert(input_cb != output_cb) in tilize_helpers.inl.
    ".*/embedding/device/kernels/compute/tilize_chunked\\.cpp$"
    # experimental CCL kernels — CT-arg-triggered failures not fixable by stubs:
    #   tilize_compute: same input_cb==output_cb static_assert as embedding/tilize_chunked
    #   dm_in0_sender: no matching function for 'compute_actual_k_block' (JIT-injected)
    #   all_to_all_sender_writer: '#error "Unsupported Topology Type"' fires with CT args=1
    ".*/all_gather_concat_heads_fused/device/kernels/tilize_compute\\.cpp$"
    ".*/all_gather_minimal_matmul_async/device/kernels/dm_in0_sender\\.cpp$"
    ".*/all_to_all_async_generic/device/kernels/all_to_all_sender_writer\\.cpp$"
    # all_to_all_dispatch_metadata: JIT-injected DEST_CHIP_ID / DEST_MESH_ID / DIRECTIONS /
    # AXIS / DISPATCH_ALGORITHM macros; template argument failures in moe_utils.hpp.
    ".*/all_to_all_dispatch_metadata/device/kernels/dataflow/writer_all_to_all_dispatch_metadata\\.cpp$"
    # llama_reduce_scatter / llama_reduce_scatter_create_heads: JIT-injected
    # DEVICE_ORDER / INPUT_CORE_XY / OUTPUT_CORE_XY macros (coordinate arrays).
    ".*/llama_reduce_scatter/device/kernels/dataflow/reader_llama_reduce_scatter\\.cpp$"
    ".*/llama_reduce_scatter/device/kernels/dataflow/writer_llama_reduce_scatter\\.cpp$"
    ".*/llama_reduce_scatter_create_heads/device/kernels/dataflow/reader_llama_reduce_scatter\\.cpp$"
    ".*/llama_reduce_scatter_create_heads/device/kernels/dataflow/writer_llama_reduce_scatter\\.cpp$"
    # moe/selective_reduce_combine: JIT-injected DEST_CHIP_ID/DEST_MESH_ID/DIRECTIONS/
    # REPLICATE_GROUP_AXIS macros; template failures in moe_utils.hpp.
    ".*/moe/selective_reduce_combine/device/kernels/dataflow/writer\\.cpp$"
    # moe_compute / moe_gpt: compute_dummy.cpp can't find compute_kernel_api.h
    # (non-standard layout); dm1.cpp uses JIT-injected OUTPUT_SHARD_CORE_MAP macro.
    ".*/moe_compute/device/kernels/compute_dummy\\.cpp$"
    ".*/moe_compute/device/kernels/dm1\\.cpp$"
    ".*/moe_gpt/device/kernels/dm1\\.cpp$"
    # send_recv_async: use PACKET_HEADER_TYPE (defined in tt_fabric_api.h behind
    # fabric conditionals not resolved at KCT time) and tt::tt_fabric namespace
    # members that are unavailable without the full fabric header chain.
    ".*/send_recv_async/recv_async/device/kernels/receiver_inplace_writer\\.cpp$"
    ".*/send_recv_async/recv_async/device/kernels/receiver_reader\\.cpp$"
    ".*/send_recv_async/send_async/device/kernels/sender_writer\\.cpp$"
    # experimental/conv3d/reader_vol2col: constexpr-if failures and CB page size
    # static_assert with CT args=1 (unrelated to tilize — not fixed by the stub).
    ".*/experimental/conv3d/device/kernels/reader_vol2col\\.cpp$"
    # strided_all_gather_async/minimal_default_writer: TensorAccessor<sharded_DSpec>
    # has no 'page_size' member — same sharded TensorAccessor SFINAE failure as
    # the fill_pad/fold/indexed_fill exclusions above.
    ".*/strided_all_gather_async/device/kernels/minimal_default_writer\\.cpp$"
    # deepseek/mla/matmul_wo/dm1: JIT-injected COLLECTOR_CORE_COORDS macro +
    # experimental::noc static_assert on sizeof(CoreLocalMem) with CT args=1.
    ".*/deepseek/mla/matmul_wo/device/kernels/dm1\\.cpp$"
    # experimental/matmul/attn_matmul: uses async_read() not available in the
    # KCT dataflow stub (non-standard InterleavedAddrGenFast extension).
    ".*/experimental/matmul/attn_matmul/device/kernels/dataflow/reader_transformer_attn_matmul\\.cpp$"
    # experimental/isin: isin_common.hpp includes only <cstdint> and relies on
    # JIT-provided dataflow and TensorAccessor headers (cb_reserve_back, etc.
    # are undeclared); also uses 'elements_number_type' without defining it.
    ".*/experimental/isin/device/kernels/dataflow/isin_reader\\.cpp$"
    # group_attn_matmul dataflow: Noc::async_read()/async_write_multicast() fail
    # because TensorAccessor<CT_args=1> doesn't produce a valid noc_traits_t
    # specialisation — 'no matching member function' after noc.h stub suppresses
    # the static_assert. Structural failure; CT args=1 → invalid template args.
    ".*/group_attn_matmul/device/kernels/dataflow/writer_transformer_group_attn_matmul\\.cpp$"
    ".*/group_attn_matmul/device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul\\.cpp$"
    # deepseek_grouped_gate: dataflow kernels include only <cstdint> and use
    # TensorAccessorArgs/cb_reserve_back/get_arg_val without including dataflow_api.h.
    ".*/deepseek_grouped_gate/device/kernels/dataflow/reader_deepseek_grouped_gate\\.cpp$"
    ".*/deepseek_grouped_gate/device/kernels/dataflow/writer_deepseek_grouped_gate\\.cpp$"
    # integral_image/intimg_reader: template instantiated with input_number_t=void
    # because df_to_std<DataFormat> returns void for the DataFormat value that
    # results from DATA_FORMATS_DEFINED + CB index with all-1 CT args.
    # sizeof(void) is then invalid. Same pattern as the scatter kernels.
    ".*/integral_image/device/kernels/intimg_reader\\.cpp$"
    # topk_router_gpt/compute + fused_distributed_rmsnorm: call llk_math_reduce
    # with enforce_fp32_accumulation=true, which static_asserts that
    # is_fp32_dest_acc_en (=DST_ACCUM_MODE) is true — but our KCT prelude sets
    # DST_ACCUM_MODE=false globally. Changing it globally risks other failures.
    ".*/topk_router_gpt/device/kernels/compute\\.cpp$"
    ".*/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_post_allgather\\.cpp$"
    ".*/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_pre_allgather\\.cpp$"
    # split_qkv: writer includes only <stdint.h> and relies on JIT-provided dataflow
    # headers (get_tile_size, cb_reserve_back, etc. undeclared).
    ".*/split_query_key_value_and_split_heads/device/kernels/dataflow/writer_tm_tile_layout_create_qkv_heads_sharded\\.cpp$"
    # index_fill/writer_index_fill: static_assert 'elem_size==2||4' fires because
    # elem_size CT arg evaluates to 1 with KERNEL_COMPILE_TIME_ARGS=1,...
    ".*/index_fill/device/kernels/writer_index_fill\\.cpp$"

    # matmul ring-all-gather / sharded dataflow kernels: use TensorAccessor<CoreLocalMem<...>>
    # types that have no noc_traits_t specialisation with CT args=1.  The primary template
    # (whose static_assert we suppress in experimental/noc.h) ends up as an empty struct with
    # no src_args_type / dst_args_type aliases, so Noc::async_read/async_write substitution
    # fails.  Only 6 of 111 experimental/noc.h-using kernels hit this — the others use
    # accessor types that do have valid noc_traits_t specialisations.
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather\\.cpp$"
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather\\.cpp$"
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded\\.cpp$"
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded\\.cpp$"
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded\\.cpp$"
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding\\.cpp$"
    # Additional matmul sender-writer that also uses noc_traits_t on bad CT-arg accessor types
    ".*/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding\\.cpp$"

    # (moreh_dot was previously excluded for missing REDUCE_OP/REDUCE_DIM but is now
    # covered by the prelude fix that defines those macros.)

    # layernorm compute kernels: call reduce_init<..., enforce_fp32_accumulation=true>
    # via the FLOAT32_REDUCTION CT arg (=1 under KERNEL_COMPILE_TIME_ARGS=1,...), but
    # DST_ACCUM_MODE=false in the KCT prelude.  llk_math_reduce.h:307 then fires
    # static_assert(is_fp32_dest_acc_en) — the same conflict as topk_router_gpt and
    # fused_distributed_rmsnorm.  Fixing it globally by setting DST_ACCUM_MODE=true
    # risks breaking other kernels that rely on DST_ACCUM_MODE=false.
    ".*/layernorm/device/kernels/compute/layernorm\\.cpp$"
    ".*/layernorm/device/kernels/compute/layernorm_large_tensor\\.cpp$"
    ".*/layernorm/device/kernels/compute/layernorm_sharded\\.cpp$"
    ".*/layernorm/device/kernels/compute/layernorm_sharded_post_allgather\\.cpp$"
    ".*/layernorm/device/kernels/compute/layernorm_sharded_pre_allgather\\.cpp$"
    # layernorm_distributed compute kernels: same FLOAT32_REDUCTION=1 vs
    # DST_ACCUM_MODE=false conflict.  layernorm_pre_allgather_welford also has a
    # _bit_cast_ function redefinition vs ckernel_sfpu_welfords.h (code-quality
    # issue independent of KCT stubs).
    ".*/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather\\.cpp$"
    ".*/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_2d\\.cpp$"
    ".*/layernorm_distributed/device/kernels/compute/layernorm_pre_allgather_welford\\.cpp$"
    ".*/layernorm_distributed/device/kernels/compute/layernorm_post_allgather\\.cpp$"
    # rmsnorm_distributed compute kernels: same FLOAT32_REDUCTION=1 vs
    # DST_ACCUM_MODE=false conflict as layernorm_distributed and layernorm.
    ".*/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather\\.cpp$"
    ".*/rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather_2d\\.cpp$"
    ".*/rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather\\.cpp$"
    # pool/generic/compute_mpwi: static_assert(REDUCE_OP == MAX) fires because the
    # KCT prelude defaults REDUCE_OP=SUM.  This kernel requires MAX at JIT time.
    ".*/pool/generic/device/kernels/compute/compute_mpwi\\.cpp$"
    # pool/grid_sample and pool/rotate: grid_sample_reader_common.hpp:269 uses
    # '::template read_grid_point(...)' without explicit template args, which
    # clang-20 rejects as [-Wmissing-template-arg-list-after-template-kw].
    # This is a real code-quality issue that requires fixing upstream.
    ".*/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_interleaved_start_id\\.cpp$"
    ".*/pool/grid_sample/device/kernels/dataflow/reader_grid_sample_sharded\\.cpp$"
    ".*/pool/grid_sample/device/kernels/dataflow/writer_grid_sample_nearest_sharded\\.cpp$"
    ".*/pool/rotate/device/kernels/dataflow/reader_rotate_bilinear_interleaved\\.cpp$"
    # reduction/argmax: df_to_std<DataFormat> returns void for the DataFormat value
    # that results from DATA_FORMATS_DEFINED + CB index with all-1 CT args.
    # sizeof(void) cascades into multiple 'incomplete type void' errors.
    # Same pattern as scatter and integral_image kernels.
    ".*/reduction/argmax/device/kernels/reader_argmax_interleaved\\.cpp$"
    ".*/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore\\.cpp$"
    ".*/reduction/argmax/device/kernels/reader_argmax_tile_layout\\.cpp$"
    # reduction/accumulation/compute: BINARY_OP_INIT / FILL_TILE / BINARY_OP are
    # JIT-injected macros not available at KCT compile time.  Same pattern as the
    # binary_ng and bcast compute kernels.
    ".*/reduction/accumulation/device/kernels/compute/accumulation_compute\\.cpp$"
    # prefetcher/writer_l1: uses experimental::remote_cb_* functions not exposed
    # by the experimental circular_buffer stub (remote circular buffer API).
    ".*/prefetcher/prefetcher/device/kernels/writer_l1\\.cpp$"
    # reduction/topk/reader_create_index_*: include only topk_dataflow_common.hpp
    # and <stdint.h>, relying on JIT firmware to provide cb_reserve_back, get_write_ptr,
    # cb_push_back, etc. (same pattern as reader_unary_sharded, nd_reshard).
    ".*/reduction/topk/device/kernels/dataflow/reader_create_index_local_topk\\.cpp$"
    ".*/reduction/topk/device/kernels/dataflow/reader_create_index_tensor\\.cpp$"
    # sliding_window/halo/compute/pack_untilize: untilize_helpers.inl static_assert
    # fires ('input_cb != output_cb') because CB indices both evaluate to 1 with
    # all-1 KERNEL_COMPILE_TIME_ARGS.
    ".*/sliding_window/halo/device/kernels/compute/pack_untilize\\.cpp$"
    # sliding_window/halo/dataflow/halo_gather: two static_asserts fire with CT
    # args=1: 'Cannot be block and width sharding' (both flags true) and 'Remote
    # read is not supported in this kernel'.  Structural CT-arg conflict.
    ".*/sliding_window/halo/device/kernels/dataflow/halo_gather\\.cpp$"
)

set(ALL_KERNEL_FILES_FILTERED "")
foreach(file ${ALL_KERNEL_FILES})
    set(_excluded FALSE)
    foreach(pattern ${KCT_EXCLUDE_PATTERNS})
        if(file MATCHES "${pattern}")
            set(_excluded TRUE)
            message(STATUS "[kernel_clang_tidy] Excluding (structural): ${file}")
            break()
        endif()
    endforeach()
    if(NOT _excluded)
        list(APPEND ALL_KERNEL_FILES_FILTERED ${file})
    endif()
endforeach()
set(ALL_KERNEL_FILES ${ALL_KERNEL_FILES_FILTERED})

list(LENGTH ALL_KERNEL_FILES ALL_KERNEL_FILES_LEN)
message(STATUS "[kernel_clang_tidy] Found ${ALL_KERNEL_FILES_LEN} kernel files (after exclusions)")

# ── Architecture selection (default: wormhole_b0) ──────────────────────────

if(NOT DEFINED ARCH_NAME)
    set(ARCH_NAME "wormhole_b0")
endif()

string(TOLOWER "${ARCH_NAME}" ARCH_NAME_LOWER)

if(ARCH_NAME_LOWER MATCHES "wormhole")
    set(KCT_ARCH_PREFIX "wormhole")
    set(KCT_ARCH_ALIAS "wormhole_b0")
    set(KCT_ARCH_DEFINE "ARCH_WORMHOLE")
    set(KCT_NUM_DRAM_BANKS 12)
    set(KCT_NUM_L1_BANKS 64)
elseif(ARCH_NAME_LOWER MATCHES "blackhole")
    set(KCT_ARCH_PREFIX "blackhole")
    set(KCT_ARCH_ALIAS "blackhole")
    set(KCT_ARCH_DEFINE "ARCH_BLACKHOLE")
    set(KCT_NUM_DRAM_BANKS 8)
    set(KCT_NUM_L1_BANKS 140)
else()
    message(WARNING "[kernel_clang_tidy] Unsupported ARCH_NAME=${ARCH_NAME}, defaulting to wormhole_b0")
    set(KCT_ARCH_PREFIX "wormhole")
    set(KCT_ARCH_ALIAS "wormhole_b0")
    set(KCT_ARCH_DEFINE "ARCH_WORMHOLE")
    set(KCT_NUM_DRAM_BANKS 12)
    set(KCT_NUM_L1_BANKS 64)
endif()

# ── Strip inherited host compile options ────────────────────────────────────
# The parent project adds host-specific flags via add_compile_options()
# (e.g. -march=x86-64-v3, -Wall, -Werror) that conflict with our RISC-V
# cross-compilation target.  Clear them; we set our own complete flag set
# via target_compile_options() below.
set_directory_properties(
    PROPERTIES
        COMPILE_OPTIONS
            ""
)

# ── Detect SFPI toolchain location ──────────────────────────────────────────
# Mirrors the fallback logic in tt_metal/jit_build/build.cpp (JitBuildEnv ctor).
# SFPI is not a git submodule — it is either checked out locally at
# runtime/sfpi/ (for development) or provided by the CI Docker image at
# /opt/tenstorrent/sfpi/.

set(KCT_SFPI_ROOT "")
foreach(_sfpi_candidate "${CMAKE_SOURCE_DIR}/runtime/sfpi" "/opt/tenstorrent/sfpi")
    if(EXISTS "${_sfpi_candidate}/compiler/bin/riscv-tt-elf-g++")
        set(KCT_SFPI_ROOT "${_sfpi_candidate}")
        break()
    endif()
endforeach()

if(KCT_SFPI_ROOT)
    # Detect the GCC version from the directory structure rather than
    # hardcoding "15.1.0" — avoids breakage when SFPI updates its GCC.
    file(GLOB _sfpi_cxx_dirs "${KCT_SFPI_ROOT}/compiler/riscv-tt-elf/include/c++/*")
    if(_sfpi_cxx_dirs)
        list(GET _sfpi_cxx_dirs 0 _sfpi_cxx_dir)
        get_filename_component(KCT_SFPI_GCC_VER "${_sfpi_cxx_dir}" NAME)
    else()
        set(KCT_SFPI_GCC_VER "15.1.0") # fallback if glob finds nothing
    endif()
    message(STATUS "[kernel_clang_tidy] SFPI found at ${KCT_SFPI_ROOT} (GCC ${KCT_SFPI_GCC_VER})")
    set(KCT_HAVE_SFPI TRUE)
else()
    message(
        WARNING
        "[kernel_clang_tidy] SFPI not found at runtime/sfpi/ or /opt/tenstorrent/sfpi/. "
        "Falling back to system C++ headers (host target instead of riscv32-unknown-elf). "
        "SFPI-dependent compute kernels will have reduced analysis coverage."
    )
    set(KCT_HAVE_SFPI FALSE)
endif()

# ── Create the OBJECT library ───────────────────────────────────────────────

add_library(kernel_clang_tidy OBJECT EXCLUDE_FROM_ALL ${ALL_KERNEL_FILES})

# ── Include paths ───────────────────────────────────────────────────────────
# Mirrors the paths from:
#   • JitBuildEnv constructor (tt_metal/jit_build/build.cpp)
#   • HalJitBuildQuery{Wormhole,BlackHole}::includes() in tt_metal/llrt/hal/tt-1xx/

target_include_directories(
    kernel_clang_tidy
    PRIVATE
        # -- JIT-generated file stubs (must come first to shadow real generated files) --
        "${CMAKE_CURRENT_SOURCE_DIR}/jit_stubs"
        # -- Common includes (from JitBuildEnv) --
        "${CMAKE_SOURCE_DIR}"
        "${CMAKE_SOURCE_DIR}/ttnn"
        "${CMAKE_SOURCE_DIR}/ttnn/cpp"
        "${CMAKE_SOURCE_DIR}/tt_metal"
        "${CMAKE_SOURCE_DIR}/tt_metal/include"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc"
        "${CMAKE_SOURCE_DIR}/tt_metal/hostdevcommon/api"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/debug"
        "${CMAKE_SOURCE_DIR}/tt_metal/api/"
        "${CMAKE_SOURCE_DIR}/tt_metal/api/tt-metalium/"
        # -- Architecture-specific includes (from HAL) --
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/common"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/llk_io"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${KCT_ARCH_PREFIX}"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${KCT_ARCH_PREFIX}/${KCT_ARCH_ALIAS}_defines"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/inc/internal/tt-1xx/${KCT_ARCH_PREFIX}/noc"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/common"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${KCT_ARCH_ALIAS}/common/inc"
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${KCT_ARCH_ALIAS}/llk_lib"
        # -- Compute kernel includes (LLK API + SFPU) --
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/llk_api"
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/ckernels/${KCT_ARCH_ALIAS}/metal/llk_api/llk_sfpu"
        # -- Firmware source (for headers referenced via firmware paths) --
        "${CMAKE_SOURCE_DIR}/tt_metal/hw/firmware/src/tt-1xx"
        # -- SFPU includes --
        "${CMAKE_SOURCE_DIR}/tt_metal/third_party/tt_llk/tt_llk_${KCT_ARCH_ALIAS}/common/inc/sfpu"
        # -- SFPI headers --
        # sfpi.h requires TT RISC-V compiler builtins that clang doesn't have.
        # Compute kernels using SFPI will hit a compile error in sfpi.h, but
        # clang-tidy still analyses the rest of their code (with -ferror-limit=0).
        $<$<BOOL:${KCT_HAVE_SFPI}>:${KCT_SFPI_ROOT}/include>
)

# ── Compiler flags ──────────────────────────────────────────────────────────
# We intentionally omit RISC-V flags (-mcpu=tt-wh, -flto, etc.) and instead
# use host-compatible flags.  -Wno-unknown-attributes silences errors from
# RISC-V-specific __attribute__ extensions (rvtt_l1_ptr, rvtt_reg_ptr).

target_compile_options(
    kernel_clang_tidy
    PRIVATE
        -std=c++17
        # When SFPI is available: parse as 32-bit RISC-V (matching the real JIT
        # target) and use the SFPI toolchain's C++ standard library headers.
        # When SFPI is absent: fall back to host target + system headers so that
        # local development still runs clang-tidy on the subset of kernel code
        # that doesn't need RISC-V-specific headers.
        $<IF:$<BOOL:${KCT_HAVE_SFPI}>,--target=riscv32-unknown-elf,>
        $<IF:$<BOOL:${KCT_HAVE_SFPI}>,-nostdinc++,>
        $<$<BOOL:${KCT_HAVE_SFPI}>:-isystem${KCT_SFPI_ROOT}/compiler/riscv-tt-elf/include/c++/${KCT_SFPI_GCC_VER}>
        $<$<BOOL:${KCT_HAVE_SFPI}>:-isystem${KCT_SFPI_ROOT}/compiler/riscv-tt-elf/include/c++/${KCT_SFPI_GCC_VER}/riscv-tt-elf>
        $<$<BOOL:${KCT_HAVE_SFPI}>:-isystem${KCT_SFPI_ROOT}/compiler/riscv-tt-elf/include>
        $<$<BOOL:${KCT_HAVE_SFPI}>:-isystem${KCT_SFPI_ROOT}/compiler/lib/gcc/riscv-tt-elf/${KCT_SFPI_GCC_VER}/include>
        -O0 # No optimisation — just parse
        -fno-exceptions
        -fno-integrated-as  # LLK uses .ttinsn (Tensix asm directive) which clang's
                            # integrated assembler doesn't recognise.  This flag
                            # delegates asm blocks to the external assembler (or
                            # skips validation), silencing "unknown directive" and
                            # "invalid constraint 'i'" errors from INSTRUCTION_WORD().
        -ferror-limit=0 # Don't stop after 20 errors per file
        -Wno-unknown-attributes # Tolerate __attribute__((rvtt_l1_ptr)) etc.
        -Wno-c++11-narrowing # Suppress narrowing (expected: 32-bit target vs host types)
        -Wno-unknown-pragmas
        -Wno-deprecated-declarations
        -Wno-unused-variable
        -Wno-unused-function
        -Wno-error
        # Force-include the prelude that provides JIT-generated stubs
        -include
        "${CMAKE_CURRENT_SOURCE_DIR}/kernel_clang_tidy_prelude.h"
)

# ── Compile definitions ────────────────────────────────────────────────────
# Mirrors defines from JitBuildEnv + HAL.  We set sensible defaults for
# device-specific values (PCIE_NOC_X, NUM_DRAM_BANKS, etc.) and provide
# dummy KERNEL_COMPILE_TIME_ARGS so that get_compile_time_arg_val() resolves.

target_compile_definitions(
    kernel_clang_tidy
    PRIVATE
        # -- Core build mode --
        TENSIX_FIRMWARE
        LOCAL_MEM_EN=0
        KERNEL_BUILD
        # -- Processor identification --
        # COMPILE_FOR_TRISC is required to match the real TRISC kernel build:
        #   • risc_common.h gates its global reg_read() under !COMPILE_FOR_TRISC;
        #     omitting it causes an ambiguity with ckernel::reg_read.
        #   • experimental/circular_buffer.h takes the LLK code path (no NOC
        #     includes) when COMPILE_FOR_TRISC is set, avoiding dataflow_api.h
        #     conflicts with api/compute/common.h.
        #   • firmware_common.h skips api/dataflow/dataflow_api.h (the
        #     jit_stubs/internal/firmware_common.h wrapper handles both cases).
        COMPILE_FOR_TRISC
        UCK_CHLKC_MATH
        NAMESPACE=chlkc_math
        PROCESSOR_INDEX=1
        TRISC_MATH
        # -- NOC configuration --
        NOC_INDEX=0
        NOC_MODE=0
        PCIE_NOC_X=0
        PCIE_NOC_Y=3
        ROUTING_FW_ENABLED
        # -- Device dimensions (reasonable defaults) --
        NUM_DRAM_BANKS=${KCT_NUM_DRAM_BANKS}
        NUM_L1_BANKS=${KCT_NUM_L1_BANKS}
        # -- Bank mapping (used by InterleavedAddrGen in dataflow_api_addrgen.h) --
        LOG_BASE_2_OF_NUM_DRAM_BANKS=4
        LOG_BASE_2_OF_NUM_L1_BANKS=6
        # -- Dispatch constants (normally set by JIT) --
        DISPATCH_MESSAGE_ADDR=0
        # -- Data format stub --
        DATA_FORMATS_DEFINED
        # -- Architecture --
        ${KCT_ARCH_DEFINE}
        # -- Compile-time arguments (dummy values so kernels parse) --
        # Provide enough entries (128) to satisfy all get_compile_time_arg_val() indices
        "KERNEL_COMPILE_TIME_ARGS=\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"
    # Named compile-time args: get_named_compile_time_arg_val() is provided
    # as a macro in kernel_clang_tidy_prelude.h (force-included) — see that
    # file for details.  We intentionally do NOT define
    # KERNEL_COMPILE_TIME_ARG_MAP here.
)
