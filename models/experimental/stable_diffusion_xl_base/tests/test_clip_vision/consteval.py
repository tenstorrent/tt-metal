# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for CLIP Resampler model."""

import utils

import ttnn


# =============================================================================
# Helper functions for duplicate const-eval patterns
# =============================================================================


def _single_weight_reshape_repeat_1280(weights, device):
    """Group 1: Process single weight through reshape [1,1,1280] and repeat [1,257,1]."""
    t = ttnn.to_device(weights[0], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t2 = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    t3 = ttnn.reshape(t2, [1, 1, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    t4 = ttnn.repeat(t3, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(t3, False)
    return [t4]


def _three_weight_concat_dim0(weights, device):
    """Group 2: Concatenate three weights on dimension 0."""
    t0 = ttnn.to_device(weights[2], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l0 = ttnn.to_layout(t0, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t0, False)
    t1 = ttnn.to_device(weights[1], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l1 = ttnn.to_layout(t1, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t1, False)
    t2 = ttnn.to_device(weights[0], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l2 = ttnn.to_layout(t2, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    result = ttnn.concat([l0, l1, l2], 0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(l2, False)
    ttnn.deallocate(l1, False)
    ttnn.deallocate(l0, False)
    return [result]


def _single_weight_reshape_repeat_5120(weights, device):
    """Group 3: Process single weight through reshape [1,1,5120] and repeat [1,257,1]."""
    t = ttnn.to_device(weights[0], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    t2 = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t, False)
    t3 = ttnn.reshape(t2, [1, 1, 5120], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    t4 = ttnn.repeat(t3, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(t3, False)
    return [t4]


def _three_weight_reshape_repeat_concat_dim2(weights, device):
    """Group 4: Reshape, repeat each of three weights, then concatenate on dim 2."""
    # Process weights[2]
    t0 = ttnn.to_device(weights[2], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l0 = ttnn.to_layout(t0, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t0, False)
    r0 = ttnn.reshape(l0, [1, 1, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(l0, False)
    p0 = ttnn.repeat(r0, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(r0, False)
    # Process weights[1]
    t1 = ttnn.to_device(weights[1], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l1 = ttnn.to_layout(t1, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t1, False)
    r1 = ttnn.reshape(l1, [1, 1, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(l1, False)
    p1 = ttnn.repeat(r1, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(r1, False)
    # Process weights[0]
    t2 = ttnn.to_device(weights[0], device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    l2 = ttnn.to_layout(t2, ttnn.Layout.TILE, None, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(t2, False)
    r2 = ttnn.reshape(l2, [1, 1, 1280], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(l2, False)
    p2 = ttnn.repeat(r2, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(r2, False)
    # Concat
    result = ttnn.concat([p0, p1, p2], 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(p2, False)
    ttnn.deallocate(p1, False)
    ttnn.deallocate(p0, False)
    return [result]


# =============================================================================
# Const-eval functions
# =============================================================================


def _full_tensor(device, shape, fill_value, dtype):
    """Create a full tensor with the given shape, fill value, and dtype."""
    return [
        ttnn.full(
            shape=ttnn.Shape(shape),
            fill_value=fill_value,
            dtype=dtype,
            layout=ttnn.Layout.TILE,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    ]


def main_const_eval_0(device):
    return _full_tensor(device, [1, 20, 16], 0.0, ttnn.DataType.BFLOAT16)


def main_const_eval_22(device):
    return _full_tensor(device, [1, 16, 1280], 1.0, ttnn.DataType.BFLOAT16)


def main_const_eval_32(device):
    return _full_tensor(device, [1, 16, 80, 257], 0.33437016606330872, ttnn.DataType.FLOAT32)


def main_const_eval_49(weights, device):
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=weights[0],
        input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        input_layout=ttnn.Layout.TILE,
        weights_format="OIHW",
        in_channels=3,
        out_channels=1280,
        batch_size=1,
        input_height=224,
        input_width=224,
        kernel_size=[14, 14],
        stride=[14, 14],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=False,
        groups=1,
        device=device,
        input_dtype=ttnn.DataType.BFLOAT16,
        output_dtype=ttnn.DataType.BFLOAT16,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=True,
            config_tensors_in_dram=True,
            act_block_h_override=0,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
    )
    util_create_list_72 = [ttnn_prepare_conv_weights_0]
    return util_create_list_72


def main_const_eval_51(weights, device):
    ttnn_to_device_95 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_95 = ttnn.to_layout(
        ttnn_to_device_95,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_device_95, False)
    ttnn_reshape_59 = ttnn.reshape(
        ttnn_to_layout_95,
        [1, 1, 2048],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_layout_95, False)
    ttnn_repeat_59 = ttnn.repeat(ttnn_reshape_59, ttnn.Shape([1, 16, 1]))
    ttnn.deallocate(ttnn_reshape_59, False)
    util_create_list_75 = [ttnn_repeat_59]
    return util_create_list_75


def main_const_eval_53(device):
    return _full_tensor(device, [1, 20, 16, 273], float("-inf"), ttnn.DataType.FLOAT32)


def main_const_eval_57(weights, device):
    ttnn_to_device_102 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_102 = ttnn.to_layout(
        ttnn_to_device_102,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_device_102, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_to_layout_102,
        ttnn.DataType.UINT32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_layout_102, False)
    ttnn_to_layout_103 = ttnn.to_layout(
        ttnn_typecast_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_to_device_103 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_embedding_0 = ttnn.embedding(ttnn_to_layout_103, ttnn_to_device_103, layout=ttnn.Layout.TILE)
    ttnn.deallocate(ttnn_to_device_103, False)
    ttnn.deallocate(ttnn_to_layout_103, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_embedding_0,
        [0, 2, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    util_create_list_82 = [ttnn_permute_0]
    return util_create_list_82


def main_const_eval_59(device):
    return _full_tensor(device, [1, 16, 257, 257], 0.0, ttnn.DataType.FLOAT32)


def main_const_eval_101(device):
    return _full_tensor(device, [1, 16, 257, 257], float("-inf"), ttnn.DataType.FLOAT32)


def main_const_eval_122(weights, device):
    ttnn_to_device_212 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_212 = ttnn.to_layout(
        ttnn_to_device_212,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_device_212, False)
    ttnn_reshape_135 = ttnn.reshape(
        ttnn_to_layout_212,
        [1, 1, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_layout_212, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_135,
        [0, 2, 1],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_135, False)
    util_create_list_170 = [ttnn_permute_1]
    return util_create_list_170


def main_const_eval_124(device):
    return _full_tensor(device, [1, 20, 16, 273], 0.0, ttnn.DataType.FLOAT32)


def main_const_eval_136(device):
    return _full_tensor(device, [1, 16, 257], 0.0, ttnn.DataType.BFLOAT16)


def main_const_eval_145(weights, device):
    ttnn_to_device_247 = ttnn.to_device(
        weights[3],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_247 = ttnn.to_layout(
        ttnn_to_device_247,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_device_247, False)
    ttnn_to_device_248 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_248 = ttnn.to_layout(
        ttnn_to_device_248,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_device_248, False)
    ttnn_to_device_249 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_to_layout_249 = ttnn.to_layout(
        ttnn_to_device_249,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_to_device_249, False)
    ttnn_full_8 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 64]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_layer_norm_0 = ttnn.layer_norm(
        weights[0],
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_layout_248,
        bias=ttnn_to_layout_249,
        residual_input_tensor=None,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=None,
    )
    ttnn.deallocate(ttnn_to_layout_249, False)
    ttnn.deallocate(ttnn_to_layout_248, False)
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_161,
        ttnn_to_layout_247,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_161, False)
    ttnn.deallocate(ttnn_to_layout_247, False)
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 16, 20, 64],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_162,
        [0, 2, 1, 3],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_162, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_permute_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_typecast_1,
        ttnn_full_8,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_reshape_163 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(ttnn_layer_norm_0, False)
    util_create_list_200 = [ttnn_full_8, ttnn_multiply_0, ttnn_reshape_163]
    return util_create_list_200


def main_const_eval_151(device):
    return _full_tensor(device, [1, 16, 257, 80], 0.33437016606330872, ttnn.DataType.FLOAT32)


def main_const_eval_165(device):
    return _full_tensor(device, [1, 20, 64, 273], 0.35355338454246521, ttnn.DataType.FLOAT32)


def run_const_evals(weights, cache, device):
    """Run all const-eval functions and return their results."""
    # fmt: off
    cez_0_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_0, cache, "main_const_eval_0", device)[0]
    ce_0_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[289]], cache, "main_const_eval_1", device)[0]
    ce_1_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[231]], cache, "main_const_eval_2", device)[0]
    ce_2_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[61]], cache, "main_const_eval_3", device)[0]
    ce_3_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[22], weights[513], weights[515]], cache, "main_const_eval_4", device)[0]
    ce_4_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[13]], cache, "main_const_eval_5", device)[0]
    ce_5_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[118], weights[481], weights[483]], cache, "main_const_eval_6", device)[0]
    ce_6_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[334], weights[409], weights[411]], cache, "main_const_eval_7", device)[0]
    ce_7_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[274], weights[429], weights[431]], cache, "main_const_eval_8", device)[0]
    ce_8_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[67]], cache, "main_const_eval_9", device)[0]
    ce_9_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[249], weights[436], weights[438]], cache, "main_const_eval_10", device)[0]
    ce_10_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[202], weights[453], weights[455]], cache, "main_const_eval_11", device)[0]
    ce_11_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[343]], cache, "main_const_eval_12", device)[0]
    ce_12_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[213], weights[448], weights[450]], cache, "main_const_eval_13", device)[0]
    ce_13_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[285], weights[424], weights[426]], cache, "main_const_eval_14", device)[0]
    ce_14_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[33], weights[508], weights[510]], cache, "main_const_eval_15", device)[0]
    ce_15_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[46], weights[505], weights[507]], cache, "main_const_eval_16", device)[0]
    ce_16_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[106], weights[485], weights[487]], cache, "main_const_eval_17", device)[0]
    ce_17_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[165], weights[464], weights[466]], cache, "main_const_eval_18", device)[0]
    ce_18_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[298], weights[421], weights[423]], cache, "main_const_eval_19", device)[0]
    ce_19_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[211]], cache, "main_const_eval_20", device)[0]
    ce_20_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[253]], cache, "main_const_eval_21", device)[0]
    cez_1_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_22, cache, "main_const_eval_22", device)[0]
    ce_21_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[189], weights[456], weights[458]], cache, "main_const_eval_23", device)[0]
    ce_22_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[103]], cache, "main_const_eval_24", device)[0]
    ce_23_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[267]], cache, "main_const_eval_25", device)[0]
    ce_24_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[286], weights[425], weights[427]], cache, "main_const_eval_26", device)[0]
    ce_25_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[361]], cache, "main_const_eval_27", device)[0]
    ce_26_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[93], weights[488], weights[490]], cache, "main_const_eval_28", device)[0]
    ce_27_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[217]], cache, "main_const_eval_29", device)[0]
    ce_28_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[69], weights[496], weights[498]], cache, "main_const_eval_30", device)[0]
    ce_29_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[55]], cache, "main_const_eval_31", device)[0]
    cez_2_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_32, cache, "main_const_eval_32", device)[0]
    ce_30_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[81], weights[492], weights[494]], cache, "main_const_eval_33", device)[0]
    ce_31_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[163]], cache, "main_const_eval_34", device)[0]
    ce_32_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[117], weights[480], weights[482]], cache, "main_const_eval_35", device)[0]
    ce_33_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[58], weights[501], weights[503]], cache, "main_const_eval_36", device)[0]
    ce_34_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[151]], cache, "main_const_eval_37", device)[0]
    ce_35_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[79]], cache, "main_const_eval_38", device)[0]
    ce_36_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[133]], cache, "main_const_eval_39", device)[0]
    ce_37_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[207]], cache, "main_const_eval_40", device)[0]
    ce_38_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[205]], cache, "main_const_eval_41", device)[0]
    ce_39_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[15]], cache, "main_const_eval_42", device)[0]
    ce_40_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[201], weights[452], weights[454]], cache, "main_const_eval_43", device)[0]
    ce_41_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[45], weights[504], weights[506]], cache, "main_const_eval_44", device)[0]
    ce_42_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[94], weights[489], weights[491]], cache, "main_const_eval_45", device)[0]
    ce_43_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[63]], cache, "main_const_eval_46", device)[0]
    ce_44_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[171]], cache, "main_const_eval_47", device)[0]
    ce_45_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[339]], cache, "main_const_eval_48", device)[0]
    ce_46_0 = utils.constEvalFuncWrapper(main_const_eval_49, [weights[389]], cache, "main_const_eval_49", device)[0]
    ce_47_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[154], weights[469], weights[471]], cache, "main_const_eval_50", device)[0]
    ce_48_0 = utils.constEvalFuncWrapper(main_const_eval_51, [weights[2]], cache, "main_const_eval_51", device)[0]
    ce_49_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[51]], cache, "main_const_eval_52", device)[0]
    cez_3_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_53, cache, "main_const_eval_53", device)[0]
    ce_50_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[21], weights[512], weights[514]], cache, "main_const_eval_54", device)[0]
    ce_51_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[235]], cache, "main_const_eval_55", device)[0]
    ce_52_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[183]], cache, "main_const_eval_56", device)[0]
    ce_53_0 = utils.constEvalFuncWrapper(main_const_eval_57, [weights[387], weights[388]], cache, "main_const_eval_57", device)[0]
    ce_54_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[153], weights[468], weights[470]], cache, "main_const_eval_58", device)[0]
    cez_4_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_59, cache, "main_const_eval_59", device)[0]
    ce_55_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[130], weights[477], weights[479]], cache, "main_const_eval_60", device)[0]
    ce_56_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[123]], cache, "main_const_eval_61", device)[0]
    ce_57_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[43]], cache, "main_const_eval_62", device)[0]
    ce_58_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[315]], cache, "main_const_eval_63", device)[0]
    ce_59_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[37]], cache, "main_const_eval_64", device)[0]
    ce_60_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[99]], cache, "main_const_eval_65", device)[0]
    ce_61_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[291]], cache, "main_const_eval_66", device)[0]
    ce_62_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[295]], cache, "main_const_eval_67", device)[0]
    ce_63_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[375]], cache, "main_const_eval_68", device)[0]
    ce_64_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[27]], cache, "main_const_eval_69", device)[0]
    ce_65_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[261], weights[432], weights[434]], cache, "main_const_eval_70", device)[0]
    ce_66_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[219]], cache, "main_const_eval_71", device)[0]
    ce_67_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[237], weights[440], weights[442]], cache, "main_const_eval_72", device)[0]
    ce_68_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[127]], cache, "main_const_eval_73", device)[0]
    ce_69_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[34], weights[509], weights[511]], cache, "main_const_eval_74", device)[0]
    ce_70_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[319]], cache, "main_const_eval_75", device)[0]
    ce_71_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[321], weights[412], weights[414]], cache, "main_const_eval_76", device)[0]
    ce_72_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[166], weights[465], weights[467]], cache, "main_const_eval_77", device)[0]
    ce_73_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[57], weights[500], weights[502]], cache, "main_const_eval_78", device)[0]
    ce_74_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[337]], cache, "main_const_eval_79", device)[0]
    ce_75_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[333], weights[408], weights[410]], cache, "main_const_eval_80", device)[0]
    ce_76_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[379]], cache, "main_const_eval_81", device)[0]
    ce_77_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[283]], cache, "main_const_eval_82", device)[0]
    ce_78_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[19]], cache, "main_const_eval_83", device)[0]
    ce_79_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[75]], cache, "main_const_eval_84", device)[0]
    ce_80_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[85]], cache, "main_const_eval_85", device)[0]
    ce_81_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[255]], cache, "main_const_eval_86", device)[0]
    ce_82_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[271]], cache, "main_const_eval_87", device)[0]
    ce_83_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[159]], cache, "main_const_eval_88", device)[0]
    ce_84_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[358], weights[401], weights[403]], cache, "main_const_eval_89", device)[0]
    ce_85_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[139]], cache, "main_const_eval_90", device)[0]
    ce_86_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[277]], cache, "main_const_eval_91", device)[0]
    ce_87_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[303]], cache, "main_const_eval_92", device)[0]
    ce_88_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[370], weights[397], weights[399]], cache, "main_const_eval_93", device)[0]
    ce_89_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[331]], cache, "main_const_eval_94", device)[0]
    ce_90_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[157]], cache, "main_const_eval_95", device)[0]
    ce_91_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[327]], cache, "main_const_eval_96", device)[0]
    ce_92_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[193]], cache, "main_const_eval_97", device)[0]
    ce_93_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[70], weights[497], weights[499]], cache, "main_const_eval_98", device)[0]
    ce_94_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[322], weights[413], weights[415]], cache, "main_const_eval_99", device)[0]
    ce_95_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[226], weights[445], weights[447]], cache, "main_const_eval_100", device)[0]
    cez_5_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_101, cache, "main_const_eval_101", device)[0]
    ce_96_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[297], weights[420], weights[422]], cache, "main_const_eval_102", device)[0]
    ce_97_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[349]], cache, "main_const_eval_103", device)[0]
    ce_98_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[279]], cache, "main_const_eval_104", device)[0]
    ce_99_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[325]], cache, "main_const_eval_105", device)[0]
    ce_100_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[178], weights[461], weights[463]], cache, "main_const_eval_106", device)[0]
    ce_101_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[262], weights[433], weights[435]], cache, "main_const_eval_107", device)[0]
    ce_102_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[142], weights[473], weights[475]], cache, "main_const_eval_108", device)[0]
    ce_103_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[105], weights[484], weights[486]], cache, "main_const_eval_109", device)[0]
    ce_104_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[175]], cache, "main_const_eval_110", device)[0]
    ce_105_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[229]], cache, "main_const_eval_111", device)[0]
    ce_106_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[190], weights[457], weights[459]], cache, "main_const_eval_112", device)[0]
    ce_107_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[369], weights[396], weights[398]], cache, "main_const_eval_113", device)[0]
    ce_108_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[87]], cache, "main_const_eval_114", device)[0]
    ce_109_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[223]], cache, "main_const_eval_115", device)[0]
    ce_110_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[346], weights[405], weights[407]], cache, "main_const_eval_116", device)[0]
    ce_111_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[345], weights[404], weights[406]], cache, "main_const_eval_117", device)[0]
    ce_112_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[307]], cache, "main_const_eval_118", device)[0]
    ce_113_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[145]], cache, "main_const_eval_119", device)[0]
    ce_114_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[135]], cache, "main_const_eval_120", device)[0]
    ce_115_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[169]], cache, "main_const_eval_121", device)[0]
    ce_116_0 = utils.constEvalFuncWrapper(main_const_eval_122, [weights[391]], cache, "main_const_eval_122", device)[0]
    ce_117_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[73]], cache, "main_const_eval_123", device)[0]
    cez_6_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_124, cache, "main_const_eval_124", device)[0]
    ce_118_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[351]], cache, "main_const_eval_125", device)[0]
    ce_119_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[82], weights[493], weights[495]], cache, "main_const_eval_126", device)[0]
    ce_120_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[310], weights[417], weights[419]], cache, "main_const_eval_127", device)[0]
    ce_121_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[91]], cache, "main_const_eval_128", device)[0]
    ce_122_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[177], weights[460], weights[462]], cache, "main_const_eval_129", device)[0]
    ce_123_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[382], weights[393], weights[395]], cache, "main_const_eval_130", device)[0]
    ce_124_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[357], weights[400], weights[402]], cache, "main_const_eval_131", device)[0]
    ce_125_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[195]], cache, "main_const_eval_132", device)[0]
    ce_126_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[11]], cache, "main_const_eval_133", device)[0]
    ce_127_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[129], weights[476], weights[478]], cache, "main_const_eval_134", device)[0]
    ce_128_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[225], weights[444], weights[446]], cache, "main_const_eval_135", device)[0]
    cez_7_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_136, cache, "main_const_eval_136", device)[0]
    ce_129_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[259]], cache, "main_const_eval_137", device)[0]
    ce_130_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[199]], cache, "main_const_eval_138", device)[0]
    ce_131_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[31]], cache, "main_const_eval_139", device)[0]
    ce_132_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[241]], cache, "main_const_eval_140", device)[0]
    ce_133_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[121]], cache, "main_const_eval_141", device)[0]
    ce_134_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[181]], cache, "main_const_eval_142", device)[0]
    ce_135_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[115]], cache, "main_const_eval_143", device)[0]
    ce_136_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[39]], cache, "main_const_eval_144", device)[0]
    ce_137 = utils.constEvalFuncWrapper(main_const_eval_145, [weights[4], weights[7], weights[8], weights[517]], cache, "main_const_eval_145", device)
    ce_137_0 = ce_137[0]
    ce_137_1 = ce_137[1]
    ce_137_2 = ce_137[2]
    ce_138_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[381], weights[392], weights[394]], cache, "main_const_eval_146", device)[0]
    ce_139_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[25]], cache, "main_const_eval_147", device)[0]
    ce_140_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[97]], cache, "main_const_eval_148", device)[0]
    ce_141_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[243]], cache, "main_const_eval_149", device)[0]
    ce_142_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[214], weights[449], weights[451]], cache, "main_const_eval_150", device)[0]
    cez_8_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_151, cache, "main_const_eval_151", device)[0]
    ce_143_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[367]], cache, "main_const_eval_152", device)[0]
    ce_144_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[273], weights[428], weights[430]], cache, "main_const_eval_153", device)[0]
    ce_145_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[49]], cache, "main_const_eval_154", device)[0]
    ce_146_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[187]], cache, "main_const_eval_155", device)[0]
    ce_147_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[355]], cache, "main_const_eval_156", device)[0]
    ce_148_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[363]], cache, "main_const_eval_157", device)[0]
    ce_149_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[313]], cache, "main_const_eval_158", device)[0]
    ce_150_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[111]], cache, "main_const_eval_159", device)[0]
    ce_151_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[247]], cache, "main_const_eval_160", device)[0]
    ce_152_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[265]], cache, "main_const_eval_161", device)[0]
    ce_153_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[309], weights[416], weights[418]], cache, "main_const_eval_162", device)[0]
    ce_154_0 = utils.constEvalFuncWrapper(_three_weight_reshape_repeat_concat_dim2, [weights[141], weights[472], weights[474]], cache, "main_const_eval_163", device)[0]
    ce_155_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[109]], cache, "main_const_eval_164", device)[0]
    cez_9_0 = utils.constEvalFuncWrapperZeroArg(main_const_eval_165, cache, "main_const_eval_165", device)[0]
    ce_156_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[250], weights[437], weights[439]], cache, "main_const_eval_166", device)[0]
    ce_157_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[373]], cache, "main_const_eval_167", device)[0]
    ce_158_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_5120, [weights[147]], cache, "main_const_eval_168", device)[0]
    ce_159_0 = utils.constEvalFuncWrapper(_three_weight_concat_dim0, [weights[238], weights[441], weights[443]], cache, "main_const_eval_169", device)[0]
    ce_160_0 = utils.constEvalFuncWrapper(_single_weight_reshape_repeat_1280, [weights[301]], cache, "main_const_eval_170", device)[0]
    # fmt: on

    return {
        "cez_0_0": cez_0_0,
        "cez_1_0": cez_1_0,
        "cez_2_0": cez_2_0,
        "cez_3_0": cez_3_0,
        "cez_4_0": cez_4_0,
        "cez_5_0": cez_5_0,
        "cez_6_0": cez_6_0,
        "cez_7_0": cez_7_0,
        "cez_8_0": cez_8_0,
        "cez_9_0": cez_9_0,
        "ce_0_0": ce_0_0,
        "ce_1_0": ce_1_0,
        "ce_2_0": ce_2_0,
        "ce_3_0": ce_3_0,
        "ce_4_0": ce_4_0,
        "ce_5_0": ce_5_0,
        "ce_6_0": ce_6_0,
        "ce_7_0": ce_7_0,
        "ce_8_0": ce_8_0,
        "ce_9_0": ce_9_0,
        "ce_10_0": ce_10_0,
        "ce_11_0": ce_11_0,
        "ce_12_0": ce_12_0,
        "ce_13_0": ce_13_0,
        "ce_14_0": ce_14_0,
        "ce_15_0": ce_15_0,
        "ce_16_0": ce_16_0,
        "ce_17_0": ce_17_0,
        "ce_18_0": ce_18_0,
        "ce_19_0": ce_19_0,
        "ce_20_0": ce_20_0,
        "ce_21_0": ce_21_0,
        "ce_22_0": ce_22_0,
        "ce_23_0": ce_23_0,
        "ce_24_0": ce_24_0,
        "ce_25_0": ce_25_0,
        "ce_26_0": ce_26_0,
        "ce_27_0": ce_27_0,
        "ce_28_0": ce_28_0,
        "ce_29_0": ce_29_0,
        "ce_30_0": ce_30_0,
        "ce_31_0": ce_31_0,
        "ce_32_0": ce_32_0,
        "ce_33_0": ce_33_0,
        "ce_34_0": ce_34_0,
        "ce_35_0": ce_35_0,
        "ce_36_0": ce_36_0,
        "ce_37_0": ce_37_0,
        "ce_38_0": ce_38_0,
        "ce_39_0": ce_39_0,
        "ce_40_0": ce_40_0,
        "ce_41_0": ce_41_0,
        "ce_42_0": ce_42_0,
        "ce_43_0": ce_43_0,
        "ce_44_0": ce_44_0,
        "ce_45_0": ce_45_0,
        "ce_46_0": ce_46_0,
        "ce_47_0": ce_47_0,
        "ce_48_0": ce_48_0,
        "ce_49_0": ce_49_0,
        "ce_50_0": ce_50_0,
        "ce_51_0": ce_51_0,
        "ce_52_0": ce_52_0,
        "ce_53_0": ce_53_0,
        "ce_54_0": ce_54_0,
        "ce_55_0": ce_55_0,
        "ce_56_0": ce_56_0,
        "ce_57_0": ce_57_0,
        "ce_58_0": ce_58_0,
        "ce_59_0": ce_59_0,
        "ce_60_0": ce_60_0,
        "ce_61_0": ce_61_0,
        "ce_62_0": ce_62_0,
        "ce_63_0": ce_63_0,
        "ce_64_0": ce_64_0,
        "ce_65_0": ce_65_0,
        "ce_66_0": ce_66_0,
        "ce_67_0": ce_67_0,
        "ce_68_0": ce_68_0,
        "ce_69_0": ce_69_0,
        "ce_70_0": ce_70_0,
        "ce_71_0": ce_71_0,
        "ce_72_0": ce_72_0,
        "ce_73_0": ce_73_0,
        "ce_74_0": ce_74_0,
        "ce_75_0": ce_75_0,
        "ce_76_0": ce_76_0,
        "ce_77_0": ce_77_0,
        "ce_78_0": ce_78_0,
        "ce_79_0": ce_79_0,
        "ce_80_0": ce_80_0,
        "ce_81_0": ce_81_0,
        "ce_82_0": ce_82_0,
        "ce_83_0": ce_83_0,
        "ce_84_0": ce_84_0,
        "ce_85_0": ce_85_0,
        "ce_86_0": ce_86_0,
        "ce_87_0": ce_87_0,
        "ce_88_0": ce_88_0,
        "ce_89_0": ce_89_0,
        "ce_90_0": ce_90_0,
        "ce_91_0": ce_91_0,
        "ce_92_0": ce_92_0,
        "ce_93_0": ce_93_0,
        "ce_94_0": ce_94_0,
        "ce_95_0": ce_95_0,
        "ce_96_0": ce_96_0,
        "ce_97_0": ce_97_0,
        "ce_98_0": ce_98_0,
        "ce_99_0": ce_99_0,
        "ce_100_0": ce_100_0,
        "ce_101_0": ce_101_0,
        "ce_102_0": ce_102_0,
        "ce_103_0": ce_103_0,
        "ce_104_0": ce_104_0,
        "ce_105_0": ce_105_0,
        "ce_106_0": ce_106_0,
        "ce_107_0": ce_107_0,
        "ce_108_0": ce_108_0,
        "ce_109_0": ce_109_0,
        "ce_110_0": ce_110_0,
        "ce_111_0": ce_111_0,
        "ce_112_0": ce_112_0,
        "ce_113_0": ce_113_0,
        "ce_114_0": ce_114_0,
        "ce_115_0": ce_115_0,
        "ce_116_0": ce_116_0,
        "ce_117_0": ce_117_0,
        "ce_118_0": ce_118_0,
        "ce_119_0": ce_119_0,
        "ce_120_0": ce_120_0,
        "ce_121_0": ce_121_0,
        "ce_122_0": ce_122_0,
        "ce_123_0": ce_123_0,
        "ce_124_0": ce_124_0,
        "ce_125_0": ce_125_0,
        "ce_126_0": ce_126_0,
        "ce_127_0": ce_127_0,
        "ce_128_0": ce_128_0,
        "ce_129_0": ce_129_0,
        "ce_130_0": ce_130_0,
        "ce_131_0": ce_131_0,
        "ce_132_0": ce_132_0,
        "ce_133_0": ce_133_0,
        "ce_134_0": ce_134_0,
        "ce_135_0": ce_135_0,
        "ce_136_0": ce_136_0,
        "ce_137_0": ce_137_0,
        "ce_137_1": ce_137_1,
        "ce_137_2": ce_137_2,
        "ce_138_0": ce_138_0,
        "ce_139_0": ce_139_0,
        "ce_140_0": ce_140_0,
        "ce_141_0": ce_141_0,
        "ce_142_0": ce_142_0,
        "ce_143_0": ce_143_0,
        "ce_144_0": ce_144_0,
        "ce_145_0": ce_145_0,
        "ce_146_0": ce_146_0,
        "ce_147_0": ce_147_0,
        "ce_148_0": ce_148_0,
        "ce_149_0": ce_149_0,
        "ce_150_0": ce_150_0,
        "ce_151_0": ce_151_0,
        "ce_152_0": ce_152_0,
        "ce_153_0": ce_153_0,
        "ce_154_0": ce_154_0,
        "ce_155_0": ce_155_0,
        "ce_156_0": ce_156_0,
        "ce_157_0": ce_157_0,
        "ce_158_0": ce_158_0,
        "ce_159_0": ce_159_0,
        "ce_160_0": ce_160_0,
    }
