import ttnn
import utils


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([2, 1]),
        fill_value=1.0013580322265625e-05,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    ttnn_reshape_0 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_0, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_repeat_0,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [ttnn_reshape_1]
    return util_create_list_1


def main_const_eval_2(input):
    input_0 = input[0]
    ttnn_reshape_2 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_reshape_2,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_1 = ttnn.repeat(ttnn_permute_0, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_0 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_0,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_2 = [ttnn_reshape_3]
    return util_create_list_2


def main_const_eval_3(input):
    input_0 = input[0]
    ttnn_reshape_4 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_3 = [ttnn_reshape_4]
    return util_create_list_3


def main_const_eval_4(input):
    input_0 = input[0]
    ttnn_reshape_5 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_2 = ttnn.repeat(ttnn_reshape_5, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_repeat_2,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_4 = [ttnn_reshape_6]
    return util_create_list_4


def main_const_eval_5(input):
    input_0 = input[0]
    ttnn_reshape_7 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_7,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_3 = ttnn.repeat(ttnn_permute_1, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_1 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_1,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_5 = [ttnn_reshape_8]
    return util_create_list_5


def main_const_eval_6(input):
    input_0 = input[0]
    ttnn_reshape_9 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_9,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_4 = ttnn.repeat(ttnn_permute_2, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_2 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_2,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_6 = [ttnn_reshape_10]
    return util_create_list_6


def main_const_eval_7(input):
    input_0 = input[0]
    ttnn_reshape_11 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_7 = [ttnn_reshape_11]
    return util_create_list_7


def main_const_eval_8(input):
    input_0 = input[0]
    ttnn_reshape_12 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_8 = [ttnn_reshape_12]
    return util_create_list_8


def main_const_eval_9(input):
    input_0 = input[0]
    ttnn_reshape_13 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_9 = [ttnn_reshape_13]
    return util_create_list_9


def main_const_eval_10(input):
    input_0 = input[0]
    ttnn_reshape_14 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_3 = ttnn.permute(
        ttnn_reshape_14,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_5 = ttnn.repeat(ttnn_permute_3, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_3 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_5,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_3,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_10 = [ttnn_reshape_15]
    return util_create_list_10


def main_const_eval_11():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([2, 50, 1]),
        fill_value=1.0013580322265625e-05,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_11 = [ttnn_full_1]
    return util_create_list_11


def main_const_eval_12(input):
    input_0 = input[0]
    ttnn_reshape_16 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_12 = [ttnn_reshape_16]
    return util_create_list_12


def main_const_eval_13(input):
    input_0 = input[0]
    ttnn_reshape_17 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_4 = ttnn.permute(
        ttnn_reshape_17,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_6 = ttnn.repeat(ttnn_permute_4, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_4 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_4,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_13 = [ttnn_reshape_18]
    return util_create_list_13


def main_const_eval_14(input):
    input_0 = input[0]
    ttnn_reshape_19 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_5 = ttnn.permute(
        ttnn_reshape_19,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_7 = ttnn.repeat(ttnn_permute_5, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_5 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_5,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_14 = [ttnn_reshape_20]
    return util_create_list_14


def main_const_eval_15(input):
    input_0 = input[0]
    ttnn_reshape_21 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_15 = [ttnn_reshape_21]
    return util_create_list_15


def main_const_eval_16(input):
    input_0 = input[0]
    ttnn_reshape_22 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_16 = [ttnn_reshape_22]
    return util_create_list_16


def main_const_eval_17(input):
    input_0 = input[0]
    ttnn_reshape_23 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_6 = ttnn.permute(
        ttnn_reshape_23,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_8 = ttnn.repeat(ttnn_permute_6, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_6 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_8,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_6,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_17 = [ttnn_reshape_24]
    return util_create_list_17


def main_const_eval_18(input):
    input_0 = input[0]
    ttnn_reshape_25 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_18 = [ttnn_reshape_25]
    return util_create_list_18


def main_const_eval_19(input):
    input_0 = input[0]
    ttnn_reshape_26 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_19 = [ttnn_reshape_26]
    return util_create_list_19


def main_const_eval_20(input):
    input_0 = input[0]
    ttnn_reshape_27 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_20 = [ttnn_reshape_27]
    return util_create_list_20


def main_const_eval_21(input):
    input_0 = input[0]
    ttnn_reshape_28 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_7 = ttnn.permute(
        ttnn_reshape_28,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_9 = ttnn.repeat(ttnn_permute_7, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_7 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_9,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_7,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_21 = [ttnn_reshape_29]
    return util_create_list_21


def main_const_eval_22(input):
    input_0 = input[0]
    ttnn_reshape_30 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_22 = [ttnn_reshape_30]
    return util_create_list_22


def main_const_eval_23(input):
    input_0 = input[0]
    ttnn_reshape_31 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_10 = ttnn.repeat(ttnn_reshape_31, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_repeat_10,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_23 = [ttnn_reshape_32]
    return util_create_list_23


def main_const_eval_24(input):
    input_0 = input[0]
    ttnn_reshape_33 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_8 = ttnn.permute(
        ttnn_reshape_33,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_11 = ttnn.repeat(ttnn_permute_8, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_8 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_11,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_8,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_24 = [ttnn_reshape_34]
    return util_create_list_24


def main_const_eval_25(input):
    input_0 = input[0]
    ttnn_reshape_35 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_25 = [ttnn_reshape_35]
    return util_create_list_25


def main_const_eval_26(input):
    input_0 = input[0]
    ttnn_reshape_36 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_26 = [ttnn_reshape_36]
    return util_create_list_26


def main_const_eval_27(input):
    input_0 = input[0]
    ttnn_reshape_37 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_9 = ttnn.permute(
        ttnn_reshape_37,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_12 = ttnn.repeat(ttnn_permute_9, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_9 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_12,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_38 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_9,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_27 = [ttnn_reshape_38]
    return util_create_list_27


def main_const_eval_28(input):
    input_0 = input[0]
    ttnn_reshape_39 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_28 = [ttnn_reshape_39]
    return util_create_list_28


def main_const_eval_29(input):
    input_0 = input[0]
    ttnn_reshape_40 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_10 = ttnn.permute(
        ttnn_reshape_40,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_13 = ttnn.repeat(ttnn_permute_10, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_11 = ttnn.permute(
        ttnn_repeat_13,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_41 = ttnn.reshape(
        ttnn_permute_11,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_29 = [ttnn_reshape_41]
    return util_create_list_29


def main_const_eval_30(input):
    input_0 = input[0]
    ttnn_reshape_42 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_30 = [ttnn_reshape_42]
    return util_create_list_30


def main_const_eval_31(input):
    input_0 = input[0]
    ttnn_reshape_43 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_14 = ttnn.repeat(ttnn_reshape_43, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_repeat_14,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_31 = [ttnn_reshape_44]
    return util_create_list_31


def main_const_eval_32(input):
    input_0 = input[0]
    ttnn_reshape_45 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_15 = ttnn.repeat(ttnn_reshape_45, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_repeat_15,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_32 = [ttnn_reshape_46]
    return util_create_list_32


def main_const_eval_33(input):
    input_0 = input[0]
    ttnn_reshape_47 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_16 = ttnn.repeat(ttnn_reshape_47, ttnn.Shape([2, 1, 1]))
    ttnn_permute_12 = ttnn.permute(
        ttnn_repeat_16,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    util_create_list_33 = [ttnn_permute_12]
    return util_create_list_33


def main_const_eval_34(input):
    input_0 = input[0]
    ttnn_reshape_48 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_34 = [ttnn_reshape_48]
    return util_create_list_34


def main_const_eval_35(input):
    input_0 = input[0]
    ttnn_reshape_49 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_13 = ttnn.permute(
        ttnn_reshape_49,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_17 = ttnn.repeat(ttnn_permute_13, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_14 = ttnn.permute(
        ttnn_repeat_17,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_50 = ttnn.reshape(
        ttnn_permute_14,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_35 = [ttnn_reshape_50]
    return util_create_list_35


def main_const_eval_36(input):
    input_0 = input[0]
    ttnn_reshape_51 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_36 = [ttnn_reshape_51]
    return util_create_list_36


def main_const_eval_37(input):
    input_0 = input[0]
    ttnn_reshape_52 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_37 = [ttnn_reshape_52]
    return util_create_list_37


def main_const_eval_38(input):
    input_0 = input[0]
    ttnn_reshape_53 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_38 = [ttnn_reshape_53]
    return util_create_list_38


def main_const_eval_39(input):
    input_0 = input[0]
    ttnn_reshape_54 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_39 = [ttnn_reshape_54]
    return util_create_list_39


def main_const_eval_40(input):
    input_0 = input[0]
    ttnn_reshape_55 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_18 = ttnn.repeat(ttnn_reshape_55, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_repeat_18,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_40 = [ttnn_reshape_56]
    return util_create_list_40


def main_const_eval_41(input):
    input_0 = input[0]
    ttnn_reshape_57 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_15 = ttnn.permute(
        ttnn_reshape_57,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_19 = ttnn.repeat(ttnn_permute_15, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_16 = ttnn.permute(
        ttnn_repeat_19,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_permute_16,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_41 = [ttnn_reshape_58]
    return util_create_list_41


def main_const_eval_42(input):
    input_0 = input[0]
    ttnn_reshape_59 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_42 = [ttnn_reshape_59]
    return util_create_list_42


def main_const_eval_43(input):
    input_0 = input[0]
    ttnn_reshape_60 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_43 = [ttnn_reshape_60]
    return util_create_list_43


def main_const_eval_44(input):
    input_0 = input[0]
    ttnn_reshape_61 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_17 = ttnn.permute(
        ttnn_reshape_61,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_20 = ttnn.repeat(ttnn_permute_17, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_10 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_20,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_10,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_44 = [ttnn_reshape_62]
    return util_create_list_44


def main_const_eval_45(input):
    input_0 = input[0]
    ttnn_reshape_63 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_45 = [ttnn_reshape_63]
    return util_create_list_45


def main_const_eval_46(input):
    input_0 = input[0]
    ttnn_reshape_64 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_18 = ttnn.permute(
        ttnn_reshape_64,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_21 = ttnn.repeat(ttnn_permute_18, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_11 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_21,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_11,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_46 = [ttnn_reshape_65]
    return util_create_list_46


def main_const_eval_47(input):
    input_0 = input[0]
    ttnn_reshape_66 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_19 = ttnn.permute(
        ttnn_reshape_66,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_22 = ttnn.repeat(ttnn_permute_19, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_12 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_22,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_12,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_47 = [ttnn_reshape_67]
    return util_create_list_47


def main_const_eval_48(input):
    input_0 = input[0]
    ttnn_reshape_68 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_48 = [ttnn_reshape_68]
    return util_create_list_48


def main_const_eval_49(input):
    input_0 = input[0]
    ttnn_reshape_69 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_49 = [ttnn_reshape_69]
    return util_create_list_49


def main_const_eval_50(input):
    input_0 = input[0]
    ttnn_reshape_70 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_50 = [ttnn_reshape_70]
    return util_create_list_50


def main_const_eval_51(input):
    input_0 = input[0]
    ttnn_reshape_71 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_23 = ttnn.repeat(ttnn_reshape_71, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_repeat_23,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_51 = [ttnn_reshape_72]
    return util_create_list_51


def main_const_eval_52(input):
    input_0 = input[0]
    ttnn_reshape_73 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_52 = [ttnn_reshape_73]
    return util_create_list_52


def main_const_eval_53(input):
    input_0 = input[0]
    ttnn_reshape_74 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_53 = [ttnn_reshape_74]
    return util_create_list_53


def main_const_eval_54(input):
    input_0 = input[0]
    ttnn_reshape_75 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_20 = ttnn.permute(
        ttnn_reshape_75,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_24 = ttnn.repeat(ttnn_permute_20, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_13 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_24,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_76 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_13,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_54 = [ttnn_reshape_76]
    return util_create_list_54


def main_const_eval_55(input):
    input_0 = input[0]
    ttnn_reshape_77 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_55 = [ttnn_reshape_77]
    return util_create_list_55


def main_const_eval_56(input):
    input_0 = input[0]
    ttnn_reshape_78 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_56 = [ttnn_reshape_78]
    return util_create_list_56


def main_const_eval_57(input):
    input_0 = input[0]
    ttnn_reshape_79 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_25 = ttnn.repeat(ttnn_reshape_79, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_80 = ttnn.reshape(
        ttnn_repeat_25,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_57 = [ttnn_reshape_80]
    return util_create_list_57


def main_const_eval_58(input):
    input_0 = input[0]
    ttnn_reshape_81 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_58 = [ttnn_reshape_81]
    return util_create_list_58


def main_const_eval_59(input):
    input_0 = input[0]
    ttnn_reshape_82 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_59 = [ttnn_reshape_82]
    return util_create_list_59


def main_const_eval_60(input):
    input_0 = input[0]
    ttnn_reshape_83 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_26 = ttnn.repeat(ttnn_reshape_83, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_84 = ttnn.reshape(
        ttnn_repeat_26,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_60 = [ttnn_reshape_84]
    return util_create_list_60


def main_const_eval_61(input):
    input_0 = input[0]
    ttnn_reshape_85 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_27 = ttnn.repeat(ttnn_reshape_85, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_86 = ttnn.reshape(
        ttnn_repeat_27,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_61 = [ttnn_reshape_86]
    return util_create_list_61


def main_const_eval_62(input):
    input_0 = input[0]
    ttnn_reshape_87 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_62 = [ttnn_reshape_87]
    return util_create_list_62


def main_const_eval_63(input):
    input_0 = input[0]
    ttnn_reshape_88 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_28 = ttnn.repeat(ttnn_reshape_88, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_89 = ttnn.reshape(
        ttnn_repeat_28,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_63 = [ttnn_reshape_89]
    return util_create_list_63


def main_const_eval_64(input):
    input_0 = input[0]
    ttnn_reshape_90 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_21 = ttnn.permute(
        ttnn_reshape_90,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_29 = ttnn.repeat(ttnn_permute_21, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_14 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_29,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_91 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_14,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_64 = [ttnn_reshape_91]
    return util_create_list_64


def main_const_eval_65(input):
    input_0 = input[0]
    ttnn_reshape_92 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_65 = [ttnn_reshape_92]
    return util_create_list_65


def main_const_eval_66(input):
    input_0 = input[0]
    ttnn_reshape_93 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_30 = ttnn.repeat(ttnn_reshape_93, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_repeat_30,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_66 = [ttnn_reshape_94]
    return util_create_list_66


def main_const_eval_67(input):
    input_0 = input[0]
    ttnn_reshape_95 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_31 = ttnn.repeat(ttnn_reshape_95, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_repeat_31,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_67 = [ttnn_reshape_96]
    return util_create_list_67


def main_const_eval_68(input):
    input_0 = input[0]
    ttnn_reshape_97 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_68 = [ttnn_reshape_97]
    return util_create_list_68


def main_const_eval_69(input):
    input_0 = input[0]
    input_1 = input[1]
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_typecast_0 = ttnn.typecast(
        input_0,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_from_device_0 = ttnn.from_device(ttnn_typecast_0)
    ttnn_to_layout_0 = ttnn.to_layout(ttnn_from_device_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn_to_device_0 = ttnn.to_device(
        ttnn_to_layout_0,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_embedding_0 = ttnn.embedding(ttnn_to_device_0, input_1)
    ttnn_repeat_32 = ttnn.repeat(ttnn_embedding_0, ttnn.Shape([2, 1, 1]))
    ttnn_permute_22 = ttnn.permute(
        ttnn_repeat_32,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    util_create_list_69 = [ttnn_permute_22]
    return util_create_list_69


def main_const_eval_70(input):
    input_0 = input[0]
    ttnn_reshape_98 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_23 = ttnn.permute(
        ttnn_reshape_98,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_33 = ttnn.repeat(ttnn_permute_23, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_24 = ttnn.permute(
        ttnn_repeat_33,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_99 = ttnn.reshape(
        ttnn_permute_24,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_70 = [ttnn_reshape_99]
    return util_create_list_70


def main_const_eval_71(input):
    input_0 = input[0]
    ttnn_reshape_100 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_25 = ttnn.permute(
        ttnn_reshape_100,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_34 = ttnn.repeat(ttnn_permute_25, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_15 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_34,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_101 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_15,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_71 = [ttnn_reshape_101]
    return util_create_list_71


def main_const_eval_72(input):
    input_0 = input[0]
    ttnn_reshape_102 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_72 = [ttnn_reshape_102]
    return util_create_list_72


def main_const_eval_73(input):
    input_0 = input[0]
    ttnn_reshape_103 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_26 = ttnn.permute(
        ttnn_reshape_103,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_35 = ttnn.repeat(ttnn_permute_26, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_27 = ttnn.permute(
        ttnn_repeat_35,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_104 = ttnn.reshape(
        ttnn_permute_27,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_73 = [ttnn_reshape_104]
    return util_create_list_73


def main_const_eval_74(input):
    input_0 = input[0]
    ttnn_reshape_105 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_28 = ttnn.permute(
        ttnn_reshape_105,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_36 = ttnn.repeat(ttnn_permute_28, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_29 = ttnn.permute(
        ttnn_repeat_36,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_106 = ttnn.reshape(
        ttnn_permute_29,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_74 = [ttnn_reshape_106]
    return util_create_list_74


def main_const_eval_75(input):
    input_0 = input[0]
    ttnn_reshape_107 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_75 = [ttnn_reshape_107]
    return util_create_list_75


def main_const_eval_76(input):
    input_0 = input[0]
    ttnn_reshape_108 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_76 = [ttnn_reshape_108]
    return util_create_list_76


def main_const_eval_77(input):
    input_0 = input[0]
    ttnn_reshape_109 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_30 = ttnn.permute(
        ttnn_reshape_109,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_37 = ttnn.repeat(ttnn_permute_30, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_31 = ttnn.permute(
        ttnn_repeat_37,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_110 = ttnn.reshape(
        ttnn_permute_31,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_77 = [ttnn_reshape_110]
    return util_create_list_77


def main_const_eval_78(input):
    input_0 = input[0]
    ttnn_reshape_111 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_78 = [ttnn_reshape_111]
    return util_create_list_78


def main_const_eval_79(input):
    input_0 = input[0]
    ttnn_reshape_112 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_79 = [ttnn_reshape_112]
    return util_create_list_79


def main_const_eval_80(input):
    input_0 = input[0]
    ttnn_reshape_113 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_38 = ttnn.repeat(ttnn_reshape_113, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_repeat_38,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_80 = [ttnn_reshape_114]
    return util_create_list_80


def main_const_eval_81(input):
    input_0 = input[0]
    ttnn_reshape_115 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_32 = ttnn.permute(
        ttnn_reshape_115,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_39 = ttnn.repeat(ttnn_permute_32, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_16 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_39,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_16,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_81 = [ttnn_reshape_116]
    return util_create_list_81


def main_const_eval_82():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([2, 50, 3072]),
        fill_value=1.703125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_117 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_123 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_124 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_125 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_126 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_127 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_128 = ttnn.reshape(
        ttnn_full_2,
        [100, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_82 = [
        ttnn_reshape_117,
        ttnn_reshape_118,
        ttnn_reshape_119,
        ttnn_reshape_120,
        ttnn_reshape_121,
        ttnn_reshape_122,
        ttnn_reshape_123,
        ttnn_reshape_124,
        ttnn_reshape_125,
        ttnn_reshape_126,
        ttnn_reshape_127,
        ttnn_reshape_128,
    ]
    return util_create_list_82


def main_const_eval_83(input):
    input_0 = input[0]
    ttnn_reshape_129 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_83 = [ttnn_reshape_129]
    return util_create_list_83


def main_const_eval_84(input):
    input_0 = input[0]
    ttnn_reshape_130 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_33 = ttnn.permute(
        ttnn_reshape_130,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_40 = ttnn.repeat(ttnn_permute_33, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_34 = ttnn.permute(
        ttnn_repeat_40,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_131 = ttnn.reshape(
        ttnn_permute_34,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_84 = [ttnn_reshape_131]
    return util_create_list_84


def main_const_eval_85(input):
    input_0 = input[0]
    ttnn_reshape_132 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_85 = [ttnn_reshape_132]
    return util_create_list_85


def main_const_eval_86(input):
    input_0 = input[0]
    ttnn_reshape_133 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_86 = [ttnn_reshape_133]
    return util_create_list_86


def main_const_eval_87(input):
    input_0 = input[0]
    ttnn_reshape_134 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_87 = [ttnn_reshape_134]
    return util_create_list_87


def main_const_eval_88(input):
    input_0 = input[0]
    ttnn_reshape_135 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_41 = ttnn.repeat(ttnn_reshape_135, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_136 = ttnn.reshape(
        ttnn_repeat_41,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_88 = [ttnn_reshape_136]
    return util_create_list_88


def main_const_eval_89(input):
    input_0 = input[0]
    ttnn_reshape_137 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_89 = [ttnn_reshape_137]
    return util_create_list_89


def main_const_eval_90(input):
    input_0 = input[0]
    ttnn_reshape_138 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_90 = [ttnn_reshape_138]
    return util_create_list_90


def main_const_eval_91(input):
    input_0 = input[0]
    ttnn_reshape_139 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_91 = [ttnn_reshape_139]
    return util_create_list_91


def main_const_eval_92(input):
    input_0 = input[0]
    ttnn_reshape_140 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_35 = ttnn.permute(
        ttnn_reshape_140,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_42 = ttnn.repeat(ttnn_permute_35, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_17 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_42,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_17,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_92 = [ttnn_reshape_141]
    return util_create_list_92


def main_const_eval_93(input):
    input_0 = input[0]
    ttnn_reshape_142 = ttnn.reshape(
        input_0,
        [1, 3072],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_93 = [ttnn_reshape_142]
    return util_create_list_93


def main_const_eval_94(input):
    input_0 = input[0]
    ttnn_reshape_143 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_94 = [ttnn_reshape_143]
    return util_create_list_94


def main_const_eval_95(input):
    input_0 = input[0]
    ttnn_reshape_144 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_36 = ttnn.permute(
        ttnn_reshape_144,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_43 = ttnn.repeat(ttnn_permute_36, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_18 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_43,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_18,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_95 = [ttnn_reshape_145]
    return util_create_list_95


def main_const_eval_96(input):
    input_0 = input[0]
    ttnn_reshape_146 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_96 = [ttnn_reshape_146]
    return util_create_list_96


def main_const_eval_97(input):
    input_0 = input[0]
    ttnn_reshape_147 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_97 = [ttnn_reshape_147]
    return util_create_list_97


def main_const_eval_98(input):
    input_0 = input[0]
    ttnn_reshape_148 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_98 = [ttnn_reshape_148]
    return util_create_list_98


def main_const_eval_99(input):
    input_0 = input[0]
    ttnn_reshape_149 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_44 = ttnn.repeat(ttnn_reshape_149, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_repeat_44,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_99 = [ttnn_reshape_150]
    return util_create_list_99


def main_const_eval_100(input):
    input_0 = input[0]
    ttnn_reshape_151 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_45 = ttnn.repeat(ttnn_reshape_151, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_repeat_45,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_100 = [ttnn_reshape_152]
    return util_create_list_100


def main_const_eval_101(input):
    input_0 = input[0]
    ttnn_reshape_153 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_101 = [ttnn_reshape_153]
    return util_create_list_101


def main_const_eval_102(input):
    input_0 = input[0]
    ttnn_reshape_154 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_102 = [ttnn_reshape_154]
    return util_create_list_102


def main_const_eval_103(input):
    input_0 = input[0]
    ttnn_reshape_155 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_37 = ttnn.permute(
        ttnn_reshape_155,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_46 = ttnn.repeat(ttnn_permute_37, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_19 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_46,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_19,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_103 = [ttnn_reshape_156]
    return util_create_list_103


def main_const_eval_104(input):
    input_0 = input[0]
    ttnn_reshape_157 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_38 = ttnn.permute(
        ttnn_reshape_157,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_47 = ttnn.repeat(ttnn_permute_38, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_20 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_47,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_20,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_104 = [ttnn_reshape_158]
    return util_create_list_104


def main_const_eval_105(input):
    input_0 = input[0]
    ttnn_reshape_159 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_105 = [ttnn_reshape_159]
    return util_create_list_105


def main_const_eval_106(input):
    input_0 = input[0]
    ttnn_reshape_160 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_48 = ttnn.repeat(ttnn_reshape_160, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_repeat_48,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_106 = [ttnn_reshape_161]
    return util_create_list_106


def main_const_eval_107(input):
    input_0 = input[0]
    ttnn_reshape_162 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_107 = [ttnn_reshape_162]
    return util_create_list_107


def main_const_eval_108(input):
    input_0 = input[0]
    ttnn_reshape_163 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_39 = ttnn.permute(
        ttnn_reshape_163,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_49 = ttnn.repeat(ttnn_permute_39, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_21 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_49,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_164 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_21,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_108 = [ttnn_reshape_164]
    return util_create_list_108


def main_const_eval_109(input):
    input_0 = input[0]
    ttnn_reshape_165 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_109 = [ttnn_reshape_165]
    return util_create_list_109


def main_const_eval_110(input):
    input_0 = input[0]
    ttnn_reshape_166 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_50 = ttnn.repeat(ttnn_reshape_166, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_167 = ttnn.reshape(
        ttnn_repeat_50,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_110 = [ttnn_reshape_167]
    return util_create_list_110


def main_const_eval_111(input):
    input_0 = input[0]
    ttnn_reshape_168 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_40 = ttnn.permute(
        ttnn_reshape_168,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_51 = ttnn.repeat(ttnn_permute_40, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_41 = ttnn.permute(
        ttnn_repeat_51,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_169 = ttnn.reshape(
        ttnn_permute_41,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_111 = [ttnn_reshape_169]
    return util_create_list_111


def main_const_eval_112(input):
    input_0 = input[0]
    ttnn_reshape_170 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_112 = [ttnn_reshape_170]
    return util_create_list_112


def main_const_eval_113(input):
    input_0 = input[0]
    ttnn_reshape_171 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_52 = ttnn.repeat(ttnn_reshape_171, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_172 = ttnn.reshape(
        ttnn_repeat_52,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_113 = [ttnn_reshape_172]
    return util_create_list_113


def main_const_eval_114(input):
    input_0 = input[0]
    ttnn_reshape_173 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_53 = ttnn.repeat(ttnn_reshape_173, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_174 = ttnn.reshape(
        ttnn_repeat_53,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_114 = [ttnn_reshape_174]
    return util_create_list_114


def main_const_eval_115(input):
    input_0 = input[0]
    ttnn_reshape_175 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_115 = [ttnn_reshape_175]
    return util_create_list_115


def main_const_eval_116(input):
    input_0 = input[0]
    ttnn_reshape_176 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_42 = ttnn.permute(
        ttnn_reshape_176,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_54 = ttnn.repeat(ttnn_permute_42, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_43 = ttnn.permute(
        ttnn_repeat_54,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_177 = ttnn.reshape(
        ttnn_permute_43,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_116 = [ttnn_reshape_177]
    return util_create_list_116


def main_const_eval_117(input):
    input_0 = input[0]
    ttnn_reshape_178 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_117 = [ttnn_reshape_178]
    return util_create_list_117


def main_const_eval_118(input):
    input_0 = input[0]
    ttnn_reshape_179 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_55 = ttnn.repeat(ttnn_reshape_179, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_180 = ttnn.reshape(
        ttnn_repeat_55,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_118 = [ttnn_reshape_180]
    return util_create_list_118


def main_const_eval_119(input):
    input_0 = input[0]
    ttnn_reshape_181 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_56 = ttnn.repeat(ttnn_reshape_181, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_182 = ttnn.reshape(
        ttnn_repeat_56,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_119 = [ttnn_reshape_182]
    return util_create_list_119


def main_const_eval_120():
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([2, 12, 50, 50]),
        fill_value=0.125,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_120 = [ttnn_full_3]
    return util_create_list_120


def main_const_eval_121(input):
    input_0 = input[0]
    ttnn_reshape_183 = ttnn.reshape(
        input_0,
        [1, 1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_repeat_57 = ttnn.repeat(ttnn_reshape_183, ttnn.Shape([2, 50, 1]))
    ttnn_reshape_184 = ttnn.reshape(
        ttnn_repeat_57,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_121 = [ttnn_reshape_184]
    return util_create_list_121


def main_const_eval_122(input):
    input_0 = input[0]
    ttnn_reshape_185 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_44 = ttnn.permute(
        ttnn_reshape_185,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_58 = ttnn.repeat(ttnn_permute_44, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_45 = ttnn.permute(
        ttnn_repeat_58,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_186 = ttnn.reshape(
        ttnn_permute_45,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_122 = [ttnn_reshape_186]
    return util_create_list_122


def main_const_eval_123(input):
    input_0 = input[0]
    ttnn_reshape_187 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_123 = [ttnn_reshape_187]
    return util_create_list_123


def main_const_eval_124(input):
    input_0 = input[0]
    ttnn_reshape_188 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_124 = [ttnn_reshape_188]
    return util_create_list_124


def main_const_eval_125(input):
    input_0 = input[0]
    ttnn_reshape_189 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_46 = ttnn.permute(
        ttnn_reshape_189,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_59 = ttnn.repeat(ttnn_permute_46, ttnn.Shape([2, 1, 1, 50]))
    ttnn_permute_47 = ttnn.permute(
        ttnn_repeat_59,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_permute_47,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_125 = [ttnn_reshape_190]
    return util_create_list_125


def main_const_eval_126(input):
    input_0 = input[0]
    ttnn_reshape_191 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_48 = ttnn.permute(
        ttnn_reshape_191,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_60 = ttnn.repeat(ttnn_permute_48, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_22 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_60,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_192 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_22,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_126 = [ttnn_reshape_192]
    return util_create_list_126


def main_const_eval_127(input):
    input_0 = input[0]
    ttnn_reshape_193 = ttnn.reshape(
        input_0,
        [1, 1, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_49 = ttnn.permute(
        ttnn_reshape_193,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_repeat_61 = ttnn.repeat(ttnn_permute_49, ttnn.Shape([2, 1, 50, 1]))
    ttnn_transformer_concatenate_heads_23 = ttnn.transformer.concatenate_heads(
        ttnn_repeat_61,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_194 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_23,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_127 = [ttnn_reshape_194]
    return util_create_list_127


def main_const_eval_128(input):
    input_0 = input[0]
    ttnn_reshape_195 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_128 = [ttnn_reshape_195]
    return util_create_list_128


def main_const_eval_129(input):
    input_0 = input[0]
    ttnn_reshape_196 = ttnn.reshape(
        input_0,
        [1, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_129 = [ttnn_reshape_196]
    return util_create_list_129


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None
CACHED_main_const_eval_3 = None
CACHED_main_const_eval_4 = None
CACHED_main_const_eval_5 = None
CACHED_main_const_eval_6 = None
CACHED_main_const_eval_7 = None
CACHED_main_const_eval_8 = None
CACHED_main_const_eval_9 = None
CACHED_main_const_eval_10 = None
CACHED_main_const_eval_11 = None
CACHED_main_const_eval_12 = None
CACHED_main_const_eval_13 = None
CACHED_main_const_eval_14 = None
CACHED_main_const_eval_15 = None
CACHED_main_const_eval_16 = None
CACHED_main_const_eval_17 = None
CACHED_main_const_eval_18 = None
CACHED_main_const_eval_19 = None
CACHED_main_const_eval_20 = None
CACHED_main_const_eval_21 = None
CACHED_main_const_eval_22 = None
CACHED_main_const_eval_23 = None
CACHED_main_const_eval_24 = None
CACHED_main_const_eval_25 = None
CACHED_main_const_eval_26 = None
CACHED_main_const_eval_27 = None
CACHED_main_const_eval_28 = None
CACHED_main_const_eval_29 = None
CACHED_main_const_eval_30 = None
CACHED_main_const_eval_31 = None
CACHED_main_const_eval_32 = None
CACHED_main_const_eval_33 = None
CACHED_main_const_eval_34 = None
CACHED_main_const_eval_35 = None
CACHED_main_const_eval_36 = None
CACHED_main_const_eval_37 = None
CACHED_main_const_eval_38 = None
CACHED_main_const_eval_39 = None
CACHED_main_const_eval_40 = None
CACHED_main_const_eval_41 = None
CACHED_main_const_eval_42 = None
CACHED_main_const_eval_43 = None
CACHED_main_const_eval_44 = None
CACHED_main_const_eval_45 = None
CACHED_main_const_eval_46 = None
CACHED_main_const_eval_47 = None
CACHED_main_const_eval_48 = None
CACHED_main_const_eval_49 = None
CACHED_main_const_eval_50 = None
CACHED_main_const_eval_51 = None
CACHED_main_const_eval_52 = None
CACHED_main_const_eval_53 = None
CACHED_main_const_eval_54 = None
CACHED_main_const_eval_55 = None
CACHED_main_const_eval_56 = None
CACHED_main_const_eval_57 = None
CACHED_main_const_eval_58 = None
CACHED_main_const_eval_59 = None
CACHED_main_const_eval_60 = None
CACHED_main_const_eval_61 = None
CACHED_main_const_eval_62 = None
CACHED_main_const_eval_63 = None
CACHED_main_const_eval_64 = None
CACHED_main_const_eval_65 = None
CACHED_main_const_eval_66 = None
CACHED_main_const_eval_67 = None
CACHED_main_const_eval_68 = None
CACHED_main_const_eval_69 = None
CACHED_main_const_eval_70 = None
CACHED_main_const_eval_71 = None
CACHED_main_const_eval_72 = None
CACHED_main_const_eval_73 = None
CACHED_main_const_eval_74 = None
CACHED_main_const_eval_75 = None
CACHED_main_const_eval_76 = None
CACHED_main_const_eval_77 = None
CACHED_main_const_eval_78 = None
CACHED_main_const_eval_79 = None
CACHED_main_const_eval_80 = None
CACHED_main_const_eval_81 = None
CACHED_main_const_eval_82 = None
CACHED_main_const_eval_83 = None
CACHED_main_const_eval_84 = None
CACHED_main_const_eval_85 = None
CACHED_main_const_eval_86 = None
CACHED_main_const_eval_87 = None
CACHED_main_const_eval_88 = None
CACHED_main_const_eval_89 = None
CACHED_main_const_eval_90 = None
CACHED_main_const_eval_91 = None
CACHED_main_const_eval_92 = None
CACHED_main_const_eval_93 = None
CACHED_main_const_eval_94 = None
CACHED_main_const_eval_95 = None
CACHED_main_const_eval_96 = None
CACHED_main_const_eval_97 = None
CACHED_main_const_eval_98 = None
CACHED_main_const_eval_99 = None
CACHED_main_const_eval_100 = None
CACHED_main_const_eval_101 = None
CACHED_main_const_eval_102 = None
CACHED_main_const_eval_103 = None
CACHED_main_const_eval_104 = None
CACHED_main_const_eval_105 = None
CACHED_main_const_eval_106 = None
CACHED_main_const_eval_107 = None
CACHED_main_const_eval_108 = None
CACHED_main_const_eval_109 = None
CACHED_main_const_eval_110 = None
CACHED_main_const_eval_111 = None
CACHED_main_const_eval_112 = None
CACHED_main_const_eval_113 = None
CACHED_main_const_eval_114 = None
CACHED_main_const_eval_115 = None
CACHED_main_const_eval_116 = None
CACHED_main_const_eval_117 = None
CACHED_main_const_eval_118 = None
CACHED_main_const_eval_119 = None
CACHED_main_const_eval_120 = None
CACHED_main_const_eval_121 = None
CACHED_main_const_eval_122 = None
CACHED_main_const_eval_123 = None
CACHED_main_const_eval_124 = None
CACHED_main_const_eval_125 = None
CACHED_main_const_eval_126 = None
CACHED_main_const_eval_127 = None
CACHED_main_const_eval_128 = None
CACHED_main_const_eval_129 = None


def _main(input):
    global CACHED_main_const_eval_129
    global CACHED_main_const_eval_128
    global CACHED_main_const_eval_127
    global CACHED_main_const_eval_126
    global CACHED_main_const_eval_125
    global CACHED_main_const_eval_124
    global CACHED_main_const_eval_123
    global CACHED_main_const_eval_122
    global CACHED_main_const_eval_121
    global CACHED_main_const_eval_120
    global CACHED_main_const_eval_119
    global CACHED_main_const_eval_118
    global CACHED_main_const_eval_117
    global CACHED_main_const_eval_116
    global CACHED_main_const_eval_115
    global CACHED_main_const_eval_114
    global CACHED_main_const_eval_113
    global CACHED_main_const_eval_112
    global CACHED_main_const_eval_111
    global CACHED_main_const_eval_110
    global CACHED_main_const_eval_109
    global CACHED_main_const_eval_108
    global CACHED_main_const_eval_107
    global CACHED_main_const_eval_106
    global CACHED_main_const_eval_105
    global CACHED_main_const_eval_104
    global CACHED_main_const_eval_103
    global CACHED_main_const_eval_102
    global CACHED_main_const_eval_101
    global CACHED_main_const_eval_100
    global CACHED_main_const_eval_99
    global CACHED_main_const_eval_98
    global CACHED_main_const_eval_97
    global CACHED_main_const_eval_96
    global CACHED_main_const_eval_95
    global CACHED_main_const_eval_94
    global CACHED_main_const_eval_93
    global CACHED_main_const_eval_92
    global CACHED_main_const_eval_91
    global CACHED_main_const_eval_90
    global CACHED_main_const_eval_89
    global CACHED_main_const_eval_88
    global CACHED_main_const_eval_87
    global CACHED_main_const_eval_86
    global CACHED_main_const_eval_85
    global CACHED_main_const_eval_84
    global CACHED_main_const_eval_83
    global CACHED_main_const_eval_82
    global CACHED_main_const_eval_81
    global CACHED_main_const_eval_80
    global CACHED_main_const_eval_79
    global CACHED_main_const_eval_78
    global CACHED_main_const_eval_77
    global CACHED_main_const_eval_76
    global CACHED_main_const_eval_75
    global CACHED_main_const_eval_74
    global CACHED_main_const_eval_73
    global CACHED_main_const_eval_72
    global CACHED_main_const_eval_71
    global CACHED_main_const_eval_70
    global CACHED_main_const_eval_69
    global CACHED_main_const_eval_68
    global CACHED_main_const_eval_67
    global CACHED_main_const_eval_66
    global CACHED_main_const_eval_65
    global CACHED_main_const_eval_64
    global CACHED_main_const_eval_63
    global CACHED_main_const_eval_62
    global CACHED_main_const_eval_61
    global CACHED_main_const_eval_60
    global CACHED_main_const_eval_59
    global CACHED_main_const_eval_58
    global CACHED_main_const_eval_57
    global CACHED_main_const_eval_56
    global CACHED_main_const_eval_55
    global CACHED_main_const_eval_54
    global CACHED_main_const_eval_53
    global CACHED_main_const_eval_52
    global CACHED_main_const_eval_51
    global CACHED_main_const_eval_50
    global CACHED_main_const_eval_49
    global CACHED_main_const_eval_48
    global CACHED_main_const_eval_47
    global CACHED_main_const_eval_46
    global CACHED_main_const_eval_45
    global CACHED_main_const_eval_44
    global CACHED_main_const_eval_43
    global CACHED_main_const_eval_42
    global CACHED_main_const_eval_41
    global CACHED_main_const_eval_40
    global CACHED_main_const_eval_39
    global CACHED_main_const_eval_38
    global CACHED_main_const_eval_37
    global CACHED_main_const_eval_36
    global CACHED_main_const_eval_35
    global CACHED_main_const_eval_34
    global CACHED_main_const_eval_33
    global CACHED_main_const_eval_32
    global CACHED_main_const_eval_31
    global CACHED_main_const_eval_30
    global CACHED_main_const_eval_29
    global CACHED_main_const_eval_28
    global CACHED_main_const_eval_27
    global CACHED_main_const_eval_26
    global CACHED_main_const_eval_25
    global CACHED_main_const_eval_24
    global CACHED_main_const_eval_23
    global CACHED_main_const_eval_22
    global CACHED_main_const_eval_21
    global CACHED_main_const_eval_20
    global CACHED_main_const_eval_19
    global CACHED_main_const_eval_18
    global CACHED_main_const_eval_17
    global CACHED_main_const_eval_16
    global CACHED_main_const_eval_15
    global CACHED_main_const_eval_14
    global CACHED_main_const_eval_13
    global CACHED_main_const_eval_12
    global CACHED_main_const_eval_11
    global CACHED_main_const_eval_10
    global CACHED_main_const_eval_9
    global CACHED_main_const_eval_8
    global CACHED_main_const_eval_7
    global CACHED_main_const_eval_6
    global CACHED_main_const_eval_5
    global CACHED_main_const_eval_4
    global CACHED_main_const_eval_3
    global CACHED_main_const_eval_2
    global CACHED_main_const_eval_1
    global CACHED_main_const_eval_0
    input_0 = input[0]
    input_1 = input[1]
    input_2 = input[2]
    input_3 = input[3]
    input_4 = input[4]
    input_5 = input[5]
    input_6 = input[6]
    input_7 = input[7]
    input_8 = input[8]
    input_9 = input[9]
    input_10 = input[10]
    input_11 = input[11]
    input_12 = input[12]
    input_13 = input[13]
    input_14 = input[14]
    input_15 = input[15]
    input_16 = input[16]
    input_17 = input[17]
    input_18 = input[18]
    input_19 = input[19]
    input_20 = input[20]
    input_21 = input[21]
    input_22 = input[22]
    input_23 = input[23]
    input_24 = input[24]
    input_25 = input[25]
    input_26 = input[26]
    input_27 = input[27]
    input_28 = input[28]
    input_29 = input[29]
    input_30 = input[30]
    input_31 = input[31]
    input_32 = input[32]
    input_33 = input[33]
    input_34 = input[34]
    input_35 = input[35]
    input_36 = input[36]
    input_37 = input[37]
    input_38 = input[38]
    input_39 = input[39]
    input_40 = input[40]
    input_41 = input[41]
    input_42 = input[42]
    input_43 = input[43]
    input_44 = input[44]
    input_45 = input[45]
    input_46 = input[46]
    input_47 = input[47]
    input_48 = input[48]
    input_49 = input[49]
    input_50 = input[50]
    input_51 = input[51]
    input_52 = input[52]
    input_53 = input[53]
    input_54 = input[54]
    input_55 = input[55]
    input_56 = input[56]
    input_57 = input[57]
    input_58 = input[58]
    input_59 = input[59]
    input_60 = input[60]
    input_61 = input[61]
    input_62 = input[62]
    input_63 = input[63]
    input_64 = input[64]
    input_65 = input[65]
    input_66 = input[66]
    input_67 = input[67]
    input_68 = input[68]
    input_69 = input[69]
    input_70 = input[70]
    input_71 = input[71]
    input_72 = input[72]
    input_73 = input[73]
    input_74 = input[74]
    input_75 = input[75]
    input_76 = input[76]
    input_77 = input[77]
    input_78 = input[78]
    input_79 = input[79]
    input_80 = input[80]
    input_81 = input[81]
    input_82 = input[82]
    input_83 = input[83]
    input_84 = input[84]
    input_85 = input[85]
    input_86 = input[86]
    input_87 = input[87]
    input_88 = input[88]
    input_89 = input[89]
    input_90 = input[90]
    input_91 = input[91]
    input_92 = input[92]
    input_93 = input[93]
    input_94 = input[94]
    input_95 = input[95]
    input_96 = input[96]
    input_97 = input[97]
    input_98 = input[98]
    input_99 = input[99]
    input_100 = input[100]
    input_101 = input[101]
    input_102 = input[102]
    input_103 = input[103]
    input_104 = input[104]
    input_105 = input[105]
    input_106 = input[106]
    input_107 = input[107]
    input_108 = input[108]
    input_109 = input[109]
    input_110 = input[110]
    input_111 = input[111]
    input_112 = input[112]
    input_113 = input[113]
    input_114 = input[114]
    input_115 = input[115]
    input_116 = input[116]
    input_117 = input[117]
    input_118 = input[118]
    input_119 = input[119]
    input_120 = input[120]
    input_121 = input[121]
    input_122 = input[122]
    input_123 = input[123]
    input_124 = input[124]
    input_125 = input[125]
    input_126 = input[126]
    input_127 = input[127]
    input_128 = input[128]
    input_129 = input[129]
    input_130 = input[130]
    input_131 = input[131]
    input_132 = input[132]
    input_133 = input[133]
    input_134 = input[134]
    input_135 = input[135]
    input_136 = input[136]
    input_137 = input[137]
    input_138 = input[138]
    input_139 = input[139]
    input_140 = input[140]
    input_141 = input[141]
    input_142 = input[142]
    input_143 = input[143]
    input_144 = input[144]
    input_145 = input[145]
    input_146 = input[146]
    input_147 = input[147]
    input_148 = input[148]
    input_149 = input[149]
    input_150 = input[150]
    input_151 = input[151]
    input_152 = input[152]
    input_153 = input[153]
    input_154 = input[154]
    input_155 = input[155]
    input_156 = input[156]
    input_157 = input[157]
    input_158 = input[158]
    input_159 = input[159]
    input_160 = input[160]
    input_161 = input[161]
    input_162 = input[162]
    input_163 = input[163]
    input_164 = input[164]
    input_165 = input[165]
    input_166 = input[166]
    input_167 = input[167]
    input_168 = input[168]
    input_169 = input[169]
    input_170 = input[170]
    input_171 = input[171]
    input_172 = input[172]
    input_173 = input[173]
    input_174 = input[174]
    input_175 = input[175]
    input_176 = input[176]
    input_177 = input[177]
    input_178 = input[178]
    input_179 = input[179]
    input_180 = input[180]
    input_181 = input[181]
    input_182 = input[182]
    input_183 = input[183]
    input_184 = input[184]
    input_185 = input[185]
    input_186 = input[186]
    input_187 = input[187]
    input_188 = input[188]
    input_189 = input[189]
    input_190 = input[190]
    input_191 = input[191]
    input_192 = input[192]
    input_193 = input[193]
    input_194 = input[194]
    input_195 = input[195]
    input_196 = input[196]
    input_197 = input[197]
    input_198 = input[198]
    input_199 = input[199]
    input_200 = input[200]
    input_201 = input[201]
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(const_0, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    util_create_list_130 = [input_15]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(const_1, util_create_list_130, CACHED_main_const_eval_1)
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_2
    util_create_list_131 = [input_107]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(const_2, util_create_list_131, CACHED_main_const_eval_2)
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_3 = main_const_eval_3
    util_create_list_132 = [input_43]
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(const_3, util_create_list_132, CACHED_main_const_eval_3)
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapper_2
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_4 = main_const_eval_4
    util_create_list_133 = [input_57]
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(const_4, util_create_list_133, CACHED_main_const_eval_4)
    CACHED_main_const_eval_4 = utils_constEvalFuncWrapper_3
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_5 = main_const_eval_5
    util_create_list_134 = [input_23]
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(const_5, util_create_list_134, CACHED_main_const_eval_5)
    CACHED_main_const_eval_5 = utils_constEvalFuncWrapper_4
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_6 = main_const_eval_6
    util_create_list_135 = [input_119]
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(const_6, util_create_list_135, CACHED_main_const_eval_6)
    CACHED_main_const_eval_6 = utils_constEvalFuncWrapper_5
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_7 = main_const_eval_7
    util_create_list_136 = [input_41]
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(const_7, util_create_list_136, CACHED_main_const_eval_7)
    CACHED_main_const_eval_7 = utils_constEvalFuncWrapper_6
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_8 = main_const_eval_8
    util_create_list_137 = [input_20]
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(const_8, util_create_list_137, CACHED_main_const_eval_8)
    CACHED_main_const_eval_8 = utils_constEvalFuncWrapper_7
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_9 = main_const_eval_9
    util_create_list_138 = [input_104]
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(const_9, util_create_list_138, CACHED_main_const_eval_9)
    CACHED_main_const_eval_9 = utils_constEvalFuncWrapper_8
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_10 = main_const_eval_10
    util_create_list_139 = [input_160]
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(const_10, util_create_list_139, CACHED_main_const_eval_10)
    CACHED_main_const_eval_10 = utils_constEvalFuncWrapper_9
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_11 = main_const_eval_11
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(const_11, CACHED_main_const_eval_11)
    CACHED_main_const_eval_11 = utils_constEvalFuncWrapperZeroArg_1
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_12 = main_const_eval_12
    util_create_list_140 = [input_109]
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_12, util_create_list_140, CACHED_main_const_eval_12
    )
    CACHED_main_const_eval_12 = utils_constEvalFuncWrapper_10
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_13 = main_const_eval_13
    util_create_list_141 = [input_11]
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_13, util_create_list_141, CACHED_main_const_eval_13
    )
    CACHED_main_const_eval_13 = utils_constEvalFuncWrapper_11
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_14 = main_const_eval_14
    util_create_list_142 = [input_184]
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_14, util_create_list_142, CACHED_main_const_eval_14
    )
    CACHED_main_const_eval_14 = utils_constEvalFuncWrapper_12
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_15 = main_const_eval_15
    util_create_list_143 = [input_139]
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_15, util_create_list_143, CACHED_main_const_eval_15
    )
    CACHED_main_const_eval_15 = utils_constEvalFuncWrapper_13
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_16 = main_const_eval_16
    util_create_list_144 = [input_86]
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_16, util_create_list_144, CACHED_main_const_eval_16
    )
    CACHED_main_const_eval_16 = utils_constEvalFuncWrapper_14
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_17 = main_const_eval_17
    util_create_list_145 = [input_143]
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_17, util_create_list_145, CACHED_main_const_eval_17
    )
    CACHED_main_const_eval_17 = utils_constEvalFuncWrapper_15
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_18 = main_const_eval_18
    util_create_list_146 = [input_25]
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_18, util_create_list_146, CACHED_main_const_eval_18
    )
    CACHED_main_const_eval_18 = utils_constEvalFuncWrapper_16
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_19 = main_const_eval_19
    util_create_list_147 = [input_110]
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_19, util_create_list_147, CACHED_main_const_eval_19
    )
    CACHED_main_const_eval_19 = utils_constEvalFuncWrapper_17
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_20 = main_const_eval_20
    util_create_list_148 = [input_8]
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_20, util_create_list_148, CACHED_main_const_eval_20
    )
    CACHED_main_const_eval_20 = utils_constEvalFuncWrapper_18
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_21 = main_const_eval_21
    util_create_list_149 = [input_59]
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_21, util_create_list_149, CACHED_main_const_eval_21
    )
    CACHED_main_const_eval_21 = utils_constEvalFuncWrapper_19
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_22 = main_const_eval_22
    util_create_list_150 = [input_103]
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_22, util_create_list_150, CACHED_main_const_eval_22
    )
    CACHED_main_const_eval_22 = utils_constEvalFuncWrapper_20
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_23 = main_const_eval_23
    util_create_list_151 = [input_99]
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_23, util_create_list_151, CACHED_main_const_eval_23
    )
    CACHED_main_const_eval_23 = utils_constEvalFuncWrapper_21
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_24 = main_const_eval_24
    util_create_list_152 = [input_200]
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_24, util_create_list_152, CACHED_main_const_eval_24
    )
    CACHED_main_const_eval_24 = utils_constEvalFuncWrapper_22
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_25 = main_const_eval_25
    util_create_list_153 = [input_62]
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_25, util_create_list_153, CACHED_main_const_eval_25
    )
    CACHED_main_const_eval_25 = utils_constEvalFuncWrapper_23
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_26 = main_const_eval_26
    util_create_list_154 = [input_65]
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_26, util_create_list_154, CACHED_main_const_eval_26
    )
    CACHED_main_const_eval_26 = utils_constEvalFuncWrapper_24
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_27 = main_const_eval_27
    util_create_list_155 = [input_176]
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_27, util_create_list_155, CACHED_main_const_eval_27
    )
    CACHED_main_const_eval_27 = utils_constEvalFuncWrapper_25
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_28 = main_const_eval_28
    util_create_list_156 = [input_121]
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_28, util_create_list_156, CACHED_main_const_eval_28
    )
    CACHED_main_const_eval_28 = utils_constEvalFuncWrapper_26
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_29 = main_const_eval_29
    util_create_list_157 = [input_190]
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_29, util_create_list_157, CACHED_main_const_eval_29
    )
    CACHED_main_const_eval_29 = utils_constEvalFuncWrapper_27
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_30 = main_const_eval_30
    util_create_list_158 = [input_140]
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_30, util_create_list_158, CACHED_main_const_eval_30
    )
    CACHED_main_const_eval_30 = utils_constEvalFuncWrapper_28
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_31 = main_const_eval_31
    util_create_list_159 = [input_33]
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_31, util_create_list_159, CACHED_main_const_eval_31
    )
    CACHED_main_const_eval_31 = utils_constEvalFuncWrapper_29
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_32 = main_const_eval_32
    util_create_list_160 = [input_9]
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_32, util_create_list_160, CACHED_main_const_eval_32
    )
    CACHED_main_const_eval_32 = utils_constEvalFuncWrapper_30
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    const_33 = main_const_eval_33
    util_create_list_161 = [input_153]
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_33, util_create_list_161, CACHED_main_const_eval_33
    )
    CACHED_main_const_eval_33 = utils_constEvalFuncWrapper_31
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_34 = main_const_eval_34
    util_create_list_162 = [input_50]
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_34, util_create_list_162, CACHED_main_const_eval_34
    )
    CACHED_main_const_eval_34 = utils_constEvalFuncWrapper_32
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_35 = main_const_eval_35
    util_create_list_163 = [input_158]
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_35, util_create_list_163, CACHED_main_const_eval_35
    )
    CACHED_main_const_eval_35 = utils_constEvalFuncWrapper_33
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_36 = main_const_eval_36
    util_create_list_164 = [input_92]
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_36, util_create_list_164, CACHED_main_const_eval_36
    )
    CACHED_main_const_eval_36 = utils_constEvalFuncWrapper_34
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_37 = main_const_eval_37
    util_create_list_165 = [input_17]
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_37, util_create_list_165, CACHED_main_const_eval_37
    )
    CACHED_main_const_eval_37 = utils_constEvalFuncWrapper_35
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_38 = main_const_eval_38
    util_create_list_166 = [input_134]
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_38, util_create_list_166, CACHED_main_const_eval_38
    )
    CACHED_main_const_eval_38 = utils_constEvalFuncWrapper_36
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_39 = main_const_eval_39
    util_create_list_167 = [input_125]
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_39, util_create_list_167, CACHED_main_const_eval_39
    )
    CACHED_main_const_eval_39 = utils_constEvalFuncWrapper_37
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_40 = main_const_eval_40
    util_create_list_168 = [input_105]
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_40, util_create_list_168, CACHED_main_const_eval_40
    )
    CACHED_main_const_eval_40 = utils_constEvalFuncWrapper_38
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_41 = main_const_eval_41
    util_create_list_169 = [input_166]
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_41, util_create_list_169, CACHED_main_const_eval_41
    )
    CACHED_main_const_eval_41 = utils_constEvalFuncWrapper_39
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_42 = main_const_eval_42
    util_create_list_170 = [input_77]
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_42, util_create_list_170, CACHED_main_const_eval_42
    )
    CACHED_main_const_eval_42 = utils_constEvalFuncWrapper_40
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_43 = main_const_eval_43
    util_create_list_171 = [input_89]
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_43, util_create_list_171, CACHED_main_const_eval_43
    )
    CACHED_main_const_eval_43 = utils_constEvalFuncWrapper_41
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_44 = main_const_eval_44
    util_create_list_172 = [input_131]
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_44, util_create_list_172, CACHED_main_const_eval_44
    )
    CACHED_main_const_eval_44 = utils_constEvalFuncWrapper_42
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_45 = main_const_eval_45
    util_create_list_173 = [input_127]
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_45, util_create_list_173, CACHED_main_const_eval_45
    )
    CACHED_main_const_eval_45 = utils_constEvalFuncWrapper_43
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_46 = main_const_eval_46
    util_create_list_174 = [input_156]
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_46, util_create_list_174, CACHED_main_const_eval_46
    )
    CACHED_main_const_eval_46 = utils_constEvalFuncWrapper_44
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_47 = main_const_eval_47
    util_create_list_175 = [input_71]
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_47, util_create_list_175, CACHED_main_const_eval_47
    )
    CACHED_main_const_eval_47 = utils_constEvalFuncWrapper_45
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_48 = main_const_eval_48
    util_create_list_176 = [input_137]
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_48, util_create_list_176, CACHED_main_const_eval_48
    )
    CACHED_main_const_eval_48 = utils_constEvalFuncWrapper_46
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_49 = main_const_eval_49
    util_create_list_177 = [input_5]
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_49, util_create_list_177, CACHED_main_const_eval_49
    )
    CACHED_main_const_eval_49 = utils_constEvalFuncWrapper_47
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_50 = main_const_eval_50
    util_create_list_178 = [input_14]
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_50, util_create_list_178, CACHED_main_const_eval_50
    )
    CACHED_main_const_eval_50 = utils_constEvalFuncWrapper_48
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_51 = main_const_eval_51
    util_create_list_179 = [input_129]
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_51, util_create_list_179, CACHED_main_const_eval_51
    )
    CACHED_main_const_eval_51 = utils_constEvalFuncWrapper_49
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_52 = main_const_eval_52
    util_create_list_180 = [input_74]
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_52, util_create_list_180, CACHED_main_const_eval_52
    )
    CACHED_main_const_eval_52 = utils_constEvalFuncWrapper_50
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_53 = main_const_eval_53
    util_create_list_181 = [input_147]
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_53, util_create_list_181, CACHED_main_const_eval_53
    )
    CACHED_main_const_eval_53 = utils_constEvalFuncWrapper_51
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_54 = main_const_eval_54
    util_create_list_182 = [input_164]
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_54, util_create_list_182, CACHED_main_const_eval_54
    )
    CACHED_main_const_eval_54 = utils_constEvalFuncWrapper_52
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_55 = main_const_eval_55
    util_create_list_183 = [input_32]
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_55, util_create_list_183, CACHED_main_const_eval_55
    )
    CACHED_main_const_eval_55 = utils_constEvalFuncWrapper_53
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_56 = main_const_eval_56
    util_create_list_184 = [input_61]
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_56, util_create_list_184, CACHED_main_const_eval_56
    )
    CACHED_main_const_eval_56 = utils_constEvalFuncWrapper_54
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_57 = main_const_eval_57
    util_create_list_185 = [input_87]
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_57, util_create_list_185, CACHED_main_const_eval_57
    )
    CACHED_main_const_eval_57 = utils_constEvalFuncWrapper_55
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_58 = main_const_eval_58
    util_create_list_186 = [input_13]
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_58, util_create_list_186, CACHED_main_const_eval_58
    )
    CACHED_main_const_eval_58 = utils_constEvalFuncWrapper_56
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_59 = main_const_eval_59
    util_create_list_187 = [input_26]
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_59, util_create_list_187, CACHED_main_const_eval_59
    )
    CACHED_main_const_eval_59 = utils_constEvalFuncWrapper_57
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_60 = main_const_eval_60
    util_create_list_188 = [input_39]
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_60, util_create_list_188, CACHED_main_const_eval_60
    )
    CACHED_main_const_eval_60 = utils_constEvalFuncWrapper_58
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_61 = main_const_eval_61
    util_create_list_189 = [input_123]
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_61, util_create_list_189, CACHED_main_const_eval_61
    )
    CACHED_main_const_eval_61 = utils_constEvalFuncWrapper_59
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_62 = main_const_eval_62
    util_create_list_190 = [input_128]
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_62, util_create_list_190, CACHED_main_const_eval_62
    )
    CACHED_main_const_eval_62 = utils_constEvalFuncWrapper_60
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_63 = main_const_eval_63
    util_create_list_191 = [input_45]
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_63, util_create_list_191, CACHED_main_const_eval_63
    )
    CACHED_main_const_eval_63 = utils_constEvalFuncWrapper_61
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_64 = main_const_eval_64
    util_create_list_192 = [input_95]
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_64, util_create_list_192, CACHED_main_const_eval_64
    )
    CACHED_main_const_eval_64 = utils_constEvalFuncWrapper_62
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_65 = main_const_eval_65
    util_create_list_193 = [input_115]
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_65, util_create_list_193, CACHED_main_const_eval_65
    )
    CACHED_main_const_eval_65 = utils_constEvalFuncWrapper_63
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_66 = main_const_eval_66
    util_create_list_194 = [input_117]
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_66, util_create_list_194, CACHED_main_const_eval_66
    )
    CACHED_main_const_eval_66 = utils_constEvalFuncWrapper_64
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_67 = main_const_eval_67
    util_create_list_195 = [input_81]
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_67, util_create_list_195, CACHED_main_const_eval_67
    )
    CACHED_main_const_eval_67 = utils_constEvalFuncWrapper_65
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_68 = main_const_eval_68
    util_create_list_196 = [input_145]
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_68, util_create_list_196, CACHED_main_const_eval_68
    )
    CACHED_main_const_eval_68 = utils_constEvalFuncWrapper_66
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_69 = main_const_eval_69
    util_create_list_197 = [input_149, input_150]
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_69, util_create_list_197, CACHED_main_const_eval_69
    )
    CACHED_main_const_eval_69 = utils_constEvalFuncWrapper_67
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_70 = main_const_eval_70
    util_create_list_198 = [input_162]
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_70, util_create_list_198, CACHED_main_const_eval_70
    )
    CACHED_main_const_eval_70 = utils_constEvalFuncWrapper_68
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_71 = main_const_eval_71
    util_create_list_199 = [input_188]
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_71, util_create_list_199, CACHED_main_const_eval_71
    )
    CACHED_main_const_eval_71 = utils_constEvalFuncWrapper_69
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_72 = main_const_eval_72
    util_create_list_200 = [input_44]
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_72, util_create_list_200, CACHED_main_const_eval_72
    )
    CACHED_main_const_eval_72 = utils_constEvalFuncWrapper_70
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_73 = main_const_eval_73
    util_create_list_201 = [input_182]
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_73, util_create_list_201, CACHED_main_const_eval_73
    )
    CACHED_main_const_eval_73 = utils_constEvalFuncWrapper_71
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_74 = main_const_eval_74
    util_create_list_202 = [input_174]
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_74, util_create_list_202, CACHED_main_const_eval_74
    )
    CACHED_main_const_eval_74 = utils_constEvalFuncWrapper_72
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_75 = main_const_eval_75
    util_create_list_203 = [input_55]
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_75, util_create_list_203, CACHED_main_const_eval_75
    )
    CACHED_main_const_eval_75 = utils_constEvalFuncWrapper_73
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_76 = main_const_eval_76
    util_create_list_204 = [input_146]
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_76, util_create_list_204, CACHED_main_const_eval_76
    )
    CACHED_main_const_eval_76 = utils_constEvalFuncWrapper_74
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_77 = main_const_eval_77
    util_create_list_205 = [input_198]
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_77, util_create_list_205, CACHED_main_const_eval_77
    )
    CACHED_main_const_eval_77 = utils_constEvalFuncWrapper_75
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_78 = main_const_eval_78
    util_create_list_206 = [input_68]
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_78, util_create_list_206, CACHED_main_const_eval_78
    )
    CACHED_main_const_eval_78 = utils_constEvalFuncWrapper_76
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_79 = main_const_eval_79
    util_create_list_207 = [input_97]
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_79, util_create_list_207, CACHED_main_const_eval_79
    )
    CACHED_main_const_eval_79 = utils_constEvalFuncWrapper_77
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_80 = main_const_eval_80
    util_create_list_208 = [input_21]
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_80, util_create_list_208, CACHED_main_const_eval_80
    )
    CACHED_main_const_eval_80 = utils_constEvalFuncWrapper_78
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_81 = main_const_eval_81
    util_create_list_209 = [input_35]
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_81, util_create_list_209, CACHED_main_const_eval_81
    )
    CACHED_main_const_eval_81 = utils_constEvalFuncWrapper_79
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_82 = main_const_eval_82
    utils_constEvalFuncWrapperZeroArg_2 = utils.constEvalFuncWrapperZeroArg(const_82, CACHED_main_const_eval_82)
    CACHED_main_const_eval_82 = utils_constEvalFuncWrapperZeroArg_2
    utils_constEvalFuncWrapperZeroArg_2_0 = utils_constEvalFuncWrapperZeroArg_2[0]
    utils_constEvalFuncWrapperZeroArg_2_1 = utils_constEvalFuncWrapperZeroArg_2[1]
    utils_constEvalFuncWrapperZeroArg_2_2 = utils_constEvalFuncWrapperZeroArg_2[2]
    utils_constEvalFuncWrapperZeroArg_2_3 = utils_constEvalFuncWrapperZeroArg_2[3]
    utils_constEvalFuncWrapperZeroArg_2_4 = utils_constEvalFuncWrapperZeroArg_2[4]
    utils_constEvalFuncWrapperZeroArg_2_5 = utils_constEvalFuncWrapperZeroArg_2[5]
    utils_constEvalFuncWrapperZeroArg_2_6 = utils_constEvalFuncWrapperZeroArg_2[6]
    utils_constEvalFuncWrapperZeroArg_2_7 = utils_constEvalFuncWrapperZeroArg_2[7]
    utils_constEvalFuncWrapperZeroArg_2_8 = utils_constEvalFuncWrapperZeroArg_2[8]
    utils_constEvalFuncWrapperZeroArg_2_9 = utils_constEvalFuncWrapperZeroArg_2[9]
    utils_constEvalFuncWrapperZeroArg_2_10 = utils_constEvalFuncWrapperZeroArg_2[10]
    utils_constEvalFuncWrapperZeroArg_2_11 = utils_constEvalFuncWrapperZeroArg_2[11]
    const_83 = main_const_eval_83
    util_create_list_210 = [input_80]
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_83, util_create_list_210, CACHED_main_const_eval_83
    )
    CACHED_main_const_eval_83 = utils_constEvalFuncWrapper_80
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_84 = main_const_eval_84
    util_create_list_211 = [input_170]
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_84, util_create_list_211, CACHED_main_const_eval_84
    )
    CACHED_main_const_eval_84 = utils_constEvalFuncWrapper_81
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_85 = main_const_eval_85
    util_create_list_212 = [input_101]
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_85, util_create_list_212, CACHED_main_const_eval_85
    )
    CACHED_main_const_eval_85 = utils_constEvalFuncWrapper_82
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_86 = main_const_eval_86
    util_create_list_213 = [input_53]
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_86, util_create_list_213, CACHED_main_const_eval_86
    )
    CACHED_main_const_eval_86 = utils_constEvalFuncWrapper_83
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_87 = main_const_eval_87
    util_create_list_214 = [input_67]
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_87, util_create_list_214, CACHED_main_const_eval_87
    )
    CACHED_main_const_eval_87 = utils_constEvalFuncWrapper_84
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_88 = main_const_eval_88
    util_create_list_215 = [input_93]
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_88, util_create_list_215, CACHED_main_const_eval_88
    )
    CACHED_main_const_eval_88 = utils_constEvalFuncWrapper_85
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_89 = main_const_eval_89
    util_create_list_216 = [input_113]
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_89, util_create_list_216, CACHED_main_const_eval_89
    )
    CACHED_main_const_eval_89 = utils_constEvalFuncWrapper_86
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_90 = main_const_eval_90
    util_create_list_217 = [input_148]
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_90, util_create_list_217, CACHED_main_const_eval_90
    )
    CACHED_main_const_eval_90 = utils_constEvalFuncWrapper_87
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_91 = main_const_eval_91
    util_create_list_218 = [input_38]
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_91, util_create_list_218, CACHED_main_const_eval_91
    )
    CACHED_main_const_eval_91 = utils_constEvalFuncWrapper_88
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_92 = main_const_eval_92
    util_create_list_219 = [input_172]
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_92, util_create_list_219, CACHED_main_const_eval_92
    )
    CACHED_main_const_eval_92 = utils_constEvalFuncWrapper_89
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_93 = main_const_eval_93
    util_create_list_220 = [input_29]
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_93, util_create_list_220, CACHED_main_const_eval_93
    )
    CACHED_main_const_eval_93 = utils_constEvalFuncWrapper_90
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_94 = main_const_eval_94
    util_create_list_221 = [input_7]
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_94, util_create_list_221, CACHED_main_const_eval_94
    )
    CACHED_main_const_eval_94 = utils_constEvalFuncWrapper_91
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_95 = main_const_eval_95
    util_create_list_222 = [input_83]
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_95, util_create_list_222, CACHED_main_const_eval_95
    )
    CACHED_main_const_eval_95 = utils_constEvalFuncWrapper_92
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_96 = main_const_eval_96
    util_create_list_223 = [input_31]
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_96, util_create_list_223, CACHED_main_const_eval_96
    )
    CACHED_main_const_eval_96 = utils_constEvalFuncWrapper_93
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_97 = main_const_eval_97
    util_create_list_224 = [input_1]
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_97, util_create_list_224, CACHED_main_const_eval_97
    )
    CACHED_main_const_eval_97 = utils_constEvalFuncWrapper_94
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_98 = main_const_eval_98
    util_create_list_225 = [input_116]
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_98, util_create_list_225, CACHED_main_const_eval_98
    )
    CACHED_main_const_eval_98 = utils_constEvalFuncWrapper_95
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    const_99 = main_const_eval_99
    util_create_list_226 = [input_63]
    utils_constEvalFuncWrapper_96 = utils.constEvalFuncWrapper(
        const_99, util_create_list_226, CACHED_main_const_eval_99
    )
    CACHED_main_const_eval_99 = utils_constEvalFuncWrapper_96
    utils_constEvalFuncWrapper_96_0 = utils_constEvalFuncWrapper_96[0]
    const_100 = main_const_eval_100
    util_create_list_227 = [input_27]
    utils_constEvalFuncWrapper_97 = utils.constEvalFuncWrapper(
        const_100, util_create_list_227, CACHED_main_const_eval_100
    )
    CACHED_main_const_eval_100 = utils_constEvalFuncWrapper_97
    utils_constEvalFuncWrapper_97_0 = utils_constEvalFuncWrapper_97[0]
    const_101 = main_const_eval_101
    util_create_list_228 = [input_49]
    utils_constEvalFuncWrapper_98 = utils.constEvalFuncWrapper(
        const_101, util_create_list_228, CACHED_main_const_eval_101
    )
    CACHED_main_const_eval_101 = utils_constEvalFuncWrapper_98
    utils_constEvalFuncWrapper_98_0 = utils_constEvalFuncWrapper_98[0]
    const_102 = main_const_eval_102
    util_create_list_229 = [input_56]
    utils_constEvalFuncWrapper_99 = utils.constEvalFuncWrapper(
        const_102, util_create_list_229, CACHED_main_const_eval_102
    )
    CACHED_main_const_eval_102 = utils_constEvalFuncWrapper_99
    utils_constEvalFuncWrapper_99_0 = utils_constEvalFuncWrapper_99[0]
    const_103 = main_const_eval_103
    util_create_list_230 = [input_192]
    utils_constEvalFuncWrapper_100 = utils.constEvalFuncWrapper(
        const_103, util_create_list_230, CACHED_main_const_eval_103
    )
    CACHED_main_const_eval_103 = utils_constEvalFuncWrapper_100
    utils_constEvalFuncWrapper_100_0 = utils_constEvalFuncWrapper_100[0]
    const_104 = main_const_eval_104
    util_create_list_231 = [input_196]
    utils_constEvalFuncWrapper_101 = utils.constEvalFuncWrapper(
        const_104, util_create_list_231, CACHED_main_const_eval_104
    )
    CACHED_main_const_eval_104 = utils_constEvalFuncWrapper_101
    utils_constEvalFuncWrapper_101_0 = utils_constEvalFuncWrapper_101[0]
    const_105 = main_const_eval_105
    util_create_list_232 = [input_85]
    utils_constEvalFuncWrapper_102 = utils.constEvalFuncWrapper(
        const_105, util_create_list_232, CACHED_main_const_eval_105
    )
    CACHED_main_const_eval_105 = utils_constEvalFuncWrapper_102
    utils_constEvalFuncWrapper_102_0 = utils_constEvalFuncWrapper_102[0]
    const_106 = main_const_eval_106
    util_create_list_233 = [input_69]
    utils_constEvalFuncWrapper_103 = utils.constEvalFuncWrapper(
        const_106, util_create_list_233, CACHED_main_const_eval_106
    )
    CACHED_main_const_eval_106 = utils_constEvalFuncWrapper_103
    utils_constEvalFuncWrapper_103_0 = utils_constEvalFuncWrapper_103[0]
    const_107 = main_const_eval_107
    util_create_list_234 = [input_133]
    utils_constEvalFuncWrapper_104 = utils.constEvalFuncWrapper(
        const_107, util_create_list_234, CACHED_main_const_eval_107
    )
    CACHED_main_const_eval_107 = utils_constEvalFuncWrapper_104
    utils_constEvalFuncWrapper_104_0 = utils_constEvalFuncWrapper_104[0]
    const_108 = main_const_eval_108
    util_create_list_235 = [input_168]
    utils_constEvalFuncWrapper_105 = utils.constEvalFuncWrapper(
        const_108, util_create_list_235, CACHED_main_const_eval_108
    )
    CACHED_main_const_eval_108 = utils_constEvalFuncWrapper_105
    utils_constEvalFuncWrapper_105_0 = utils_constEvalFuncWrapper_105[0]
    const_109 = main_const_eval_109
    util_create_list_236 = [input_98]
    utils_constEvalFuncWrapper_106 = utils.constEvalFuncWrapper(
        const_109, util_create_list_236, CACHED_main_const_eval_109
    )
    CACHED_main_const_eval_109 = utils_constEvalFuncWrapper_106
    utils_constEvalFuncWrapper_106_0 = utils_constEvalFuncWrapper_106[0]
    const_110 = main_const_eval_110
    util_create_list_237 = [input_51]
    utils_constEvalFuncWrapper_107 = utils.constEvalFuncWrapper(
        const_110, util_create_list_237, CACHED_main_const_eval_110
    )
    CACHED_main_const_eval_110 = utils_constEvalFuncWrapper_107
    utils_constEvalFuncWrapper_107_0 = utils_constEvalFuncWrapper_107[0]
    const_111 = main_const_eval_111
    util_create_list_238 = [input_154]
    utils_constEvalFuncWrapper_108 = utils.constEvalFuncWrapper(
        const_111, util_create_list_238, CACHED_main_const_eval_111
    )
    CACHED_main_const_eval_111 = utils_constEvalFuncWrapper_108
    utils_constEvalFuncWrapper_108_0 = utils_constEvalFuncWrapper_108[0]
    const_112 = main_const_eval_112
    util_create_list_239 = [input_73]
    utils_constEvalFuncWrapper_109 = utils.constEvalFuncWrapper(
        const_112, util_create_list_239, CACHED_main_const_eval_112
    )
    CACHED_main_const_eval_112 = utils_constEvalFuncWrapper_109
    utils_constEvalFuncWrapper_109_0 = utils_constEvalFuncWrapper_109[0]
    const_113 = main_const_eval_113
    util_create_list_240 = [input_111]
    utils_constEvalFuncWrapper_110 = utils.constEvalFuncWrapper(
        const_113, util_create_list_240, CACHED_main_const_eval_113
    )
    CACHED_main_const_eval_113 = utils_constEvalFuncWrapper_110
    utils_constEvalFuncWrapper_110_0 = utils_constEvalFuncWrapper_110[0]
    const_114 = main_const_eval_114
    util_create_list_241 = [input_135]
    utils_constEvalFuncWrapper_111 = utils.constEvalFuncWrapper(
        const_114, util_create_list_241, CACHED_main_const_eval_114
    )
    CACHED_main_const_eval_114 = utils_constEvalFuncWrapper_111
    utils_constEvalFuncWrapper_111_0 = utils_constEvalFuncWrapper_111[0]
    const_115 = main_const_eval_115
    util_create_list_242 = [input_2]
    utils_constEvalFuncWrapper_112 = utils.constEvalFuncWrapper(
        const_115, util_create_list_242, CACHED_main_const_eval_115
    )
    CACHED_main_const_eval_115 = utils_constEvalFuncWrapper_112
    utils_constEvalFuncWrapper_112_0 = utils_constEvalFuncWrapper_112[0]
    const_116 = main_const_eval_116
    util_create_list_243 = [input_178]
    utils_constEvalFuncWrapper_113 = utils.constEvalFuncWrapper(
        const_116, util_create_list_243, CACHED_main_const_eval_116
    )
    CACHED_main_const_eval_116 = utils_constEvalFuncWrapper_113
    utils_constEvalFuncWrapper_113_0 = utils_constEvalFuncWrapper_113[0]
    const_117 = main_const_eval_117
    util_create_list_244 = [input_91]
    utils_constEvalFuncWrapper_114 = utils.constEvalFuncWrapper(
        const_117, util_create_list_244, CACHED_main_const_eval_117
    )
    CACHED_main_const_eval_117 = utils_constEvalFuncWrapper_114
    utils_constEvalFuncWrapper_114_0 = utils_constEvalFuncWrapper_114[0]
    const_118 = main_const_eval_118
    util_create_list_245 = [input_141]
    utils_constEvalFuncWrapper_115 = utils.constEvalFuncWrapper(
        const_118, util_create_list_245, CACHED_main_const_eval_118
    )
    CACHED_main_const_eval_118 = utils_constEvalFuncWrapper_115
    utils_constEvalFuncWrapper_115_0 = utils_constEvalFuncWrapper_115[0]
    const_119 = main_const_eval_119
    util_create_list_246 = [input_75]
    utils_constEvalFuncWrapper_116 = utils.constEvalFuncWrapper(
        const_119, util_create_list_246, CACHED_main_const_eval_119
    )
    CACHED_main_const_eval_119 = utils_constEvalFuncWrapper_116
    utils_constEvalFuncWrapper_116_0 = utils_constEvalFuncWrapper_116[0]
    const_120 = main_const_eval_120
    utils_constEvalFuncWrapperZeroArg_3 = utils.constEvalFuncWrapperZeroArg(const_120, CACHED_main_const_eval_120)
    CACHED_main_const_eval_120 = utils_constEvalFuncWrapperZeroArg_3
    utils_constEvalFuncWrapperZeroArg_3_0 = utils_constEvalFuncWrapperZeroArg_3[0]
    const_121 = main_const_eval_121
    util_create_list_247 = [input_3]
    utils_constEvalFuncWrapper_117 = utils.constEvalFuncWrapper(
        const_121, util_create_list_247, CACHED_main_const_eval_121
    )
    CACHED_main_const_eval_121 = utils_constEvalFuncWrapper_117
    utils_constEvalFuncWrapper_117_0 = utils_constEvalFuncWrapper_117[0]
    const_122 = main_const_eval_122
    util_create_list_248 = [input_186]
    utils_constEvalFuncWrapper_118 = utils.constEvalFuncWrapper(
        const_122, util_create_list_248, CACHED_main_const_eval_122
    )
    CACHED_main_const_eval_122 = utils_constEvalFuncWrapper_118
    utils_constEvalFuncWrapper_118_0 = utils_constEvalFuncWrapper_118[0]
    const_123 = main_const_eval_123
    util_create_list_249 = [input_19]
    utils_constEvalFuncWrapper_119 = utils.constEvalFuncWrapper(
        const_123, util_create_list_249, CACHED_main_const_eval_123
    )
    CACHED_main_const_eval_123 = utils_constEvalFuncWrapper_119
    utils_constEvalFuncWrapper_119_0 = utils_constEvalFuncWrapper_119[0]
    const_124 = main_const_eval_124
    util_create_list_250 = [input_37]
    utils_constEvalFuncWrapper_120 = utils.constEvalFuncWrapper(
        const_124, util_create_list_250, CACHED_main_const_eval_124
    )
    CACHED_main_const_eval_124 = utils_constEvalFuncWrapper_120
    utils_constEvalFuncWrapper_120_0 = utils_constEvalFuncWrapper_120[0]
    const_125 = main_const_eval_125
    util_create_list_251 = [input_194]
    utils_constEvalFuncWrapper_121 = utils.constEvalFuncWrapper(
        const_125, util_create_list_251, CACHED_main_const_eval_125
    )
    CACHED_main_const_eval_125 = utils_constEvalFuncWrapper_121
    utils_constEvalFuncWrapper_121_0 = utils_constEvalFuncWrapper_121[0]
    const_126 = main_const_eval_126
    util_create_list_252 = [input_47]
    utils_constEvalFuncWrapper_122 = utils.constEvalFuncWrapper(
        const_126, util_create_list_252, CACHED_main_const_eval_126
    )
    CACHED_main_const_eval_126 = utils_constEvalFuncWrapper_122
    utils_constEvalFuncWrapper_122_0 = utils_constEvalFuncWrapper_122[0]
    const_127 = main_const_eval_127
    util_create_list_253 = [input_180]
    utils_constEvalFuncWrapper_123 = utils.constEvalFuncWrapper(
        const_127, util_create_list_253, CACHED_main_const_eval_127
    )
    CACHED_main_const_eval_127 = utils_constEvalFuncWrapper_123
    utils_constEvalFuncWrapper_123_0 = utils_constEvalFuncWrapper_123[0]
    const_128 = main_const_eval_128
    util_create_list_254 = [input_122]
    utils_constEvalFuncWrapper_124 = utils.constEvalFuncWrapper(
        const_128, util_create_list_254, CACHED_main_const_eval_128
    )
    CACHED_main_const_eval_128 = utils_constEvalFuncWrapper_124
    utils_constEvalFuncWrapper_124_0 = utils_constEvalFuncWrapper_124[0]
    const_129 = main_const_eval_129
    util_create_list_255 = [input_79]
    utils_constEvalFuncWrapper_125 = utils.constEvalFuncWrapper(
        const_129, util_create_list_255, CACHED_main_const_eval_129
    )
    CACHED_main_const_eval_129 = utils_constEvalFuncWrapper_125
    utils_constEvalFuncWrapper_125_0 = utils_constEvalFuncWrapper_125[0]
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    CLIPVisionEmbeddings_0_0_0 = CLIPVisionEmbeddings_0_0(
        utils_constEvalFuncWrapper_31_0,
        input_152,
        utils_DeviceGetter_get_device_5,
        utils_constEvalFuncWrapper_67_0,
        input_151,
    )
    LayerNorm_1_0_0 = LayerNorm_1_0(
        CLIPVisionEmbeddings_0_0_0,
        utils_constEvalFuncWrapper_51_0,
        utils_constEvalFuncWrapper_87_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
    )
    v_343, v_344 = CLIPEncoderLayer_2_0(utils_constEvalFuncWrapperZeroArg_1_0, LayerNorm_1_0_0)
    Linear_3_0_0 = Linear_3_0(v_344)
    LayerNorm_4_0_0 = LayerNorm_4_0(
        utils_constEvalFuncWrapper_66_0,
        v_343,
        Linear_3_0_0,
        utils_constEvalFuncWrapper_74_0,
    )
    v_345, v_346, v_347 = Linear_5_0(
        utils_constEvalFuncWrapper_108_0,
        utils_constEvalFuncWrapper_15_0,
        input_157,
        input_144,
        LayerNorm_4_0_0,
        utils_constEvalFuncWrapper_44_0,
        input_155,
    )
    CLIPAttention_6_0_0 = CLIPAttention_6_0(utils_constEvalFuncWrapperZeroArg_3_0, v_345, v_346, v_347)
    Linear_7_0_0 = Linear_7_0(utils_constEvalFuncWrapper_115_0, CLIPAttention_6_0_0, input_142)
    v_348, v_349, v_350 = CLIPEncoderLayer_8_0(Linear_7_0_0, utils_constEvalFuncWrapperZeroArg_1_0, LayerNorm_1_0_0)
    Linear_9_0_0 = Linear_9_0(v_350)
    LayerNorm_10_0_0 = LayerNorm_10_0(
        Linear_9_0_0,
        utils_constEvalFuncWrapper_13_0,
        utils_constEvalFuncWrapper_28_0,
        v_349,
    )
    Linear_11_0_0 = Linear_11_0(LayerNorm_10_0_0, utils_constEvalFuncWrapper_46_0, input_138)
    QuickGELUActivation_12_0_0 = QuickGELUActivation_12_0(Linear_11_0_0, utils_constEvalFuncWrapperZeroArg_2_0)
    Linear_13_0_0 = Linear_13_0(utils_constEvalFuncWrapper_111_0, QuickGELUActivation_12_0_0, input_136)
    v_351, v_352, v_353 = CLIPEncoderLayer_14_0(v_348, utils_constEvalFuncWrapperZeroArg_1_0, Linear_13_0_0)
    Linear_15_0_0 = Linear_15_0(v_353)
    LayerNorm_16_0_0 = LayerNorm_16_0(
        Linear_15_0_0,
        utils_constEvalFuncWrapper_104_0,
        utils_constEvalFuncWrapper_36_0,
        v_352,
    )
    v_354, v_355, v_356 = Linear_17_0(
        utils_constEvalFuncWrapper_9_0,
        LayerNorm_16_0_0,
        input_161,
        input_159,
        input_132,
        utils_constEvalFuncWrapper_42_0,
        utils_constEvalFuncWrapper_33_0,
    )
    CLIPAttention_18_0_0 = CLIPAttention_18_0(utils_constEvalFuncWrapperZeroArg_3_0, v_354, v_355, v_356)
    Linear_19_0_0 = Linear_19_0(input_130, CLIPAttention_18_0_0, utils_constEvalFuncWrapper_49_0)
    v_357, v_358, v_359 = CLIPEncoderLayer_20_0(Linear_19_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_351)
    Linear_21_0_0 = Linear_21_0(v_358)
    LayerNorm_22_0_0 = LayerNorm_22_0(
        utils_constEvalFuncWrapper_60_0,
        utils_constEvalFuncWrapper_43_0,
        v_359,
        Linear_21_0_0,
    )
    Linear_23_0_0 = Linear_23_0(LayerNorm_22_0_0, input_126, utils_constEvalFuncWrapper_37_0)
    QuickGELUActivation_24_0_0 = QuickGELUActivation_24_0(utils_constEvalFuncWrapperZeroArg_2_1, Linear_23_0_0)
    Linear_25_0_0 = Linear_25_0(input_124, QuickGELUActivation_24_0_0, utils_constEvalFuncWrapper_59_0)
    v_360, v_361, v_362 = CLIPEncoderLayer_26_0(Linear_25_0_0, v_357, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_27_0_0 = Linear_27_0(v_360)
    LayerNorm_28_0_0 = LayerNorm_28_0(
        Linear_27_0_0,
        utils_constEvalFuncWrapper_26_0,
        utils_constEvalFuncWrapper_124_0,
        v_362,
    )
    v_363, v_364, v_365 = Linear_29_0(
        input_165,
        utils_constEvalFuncWrapper_52_0,
        utils_constEvalFuncWrapper_5_0,
        LayerNorm_28_0_0,
        utils_constEvalFuncWrapper_68_0,
        input_163,
        input_120,
    )
    CLIPAttention_30_0_0 = CLIPAttention_30_0(utils_constEvalFuncWrapperZeroArg_3_0, v_363, v_364, v_365)
    Linear_31_0_0 = Linear_31_0(CLIPAttention_30_0_0, input_118, utils_constEvalFuncWrapper_64_0)
    v_366, v_367, v_368 = CLIPEncoderLayer_32_0(Linear_31_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_361)
    Linear_33_0_0 = Linear_33_0(v_368)
    LayerNorm_34_0_0 = LayerNorm_34_0(
        Linear_33_0_0,
        v_366,
        utils_constEvalFuncWrapper_63_0,
        utils_constEvalFuncWrapper_95_0,
    )
    Linear_35_0_0 = Linear_35_0(LayerNorm_34_0_0, input_114, utils_constEvalFuncWrapper_86_0)
    QuickGELUActivation_36_0_0 = QuickGELUActivation_36_0(Linear_35_0_0, utils_constEvalFuncWrapperZeroArg_2_2)
    Linear_37_0_0 = Linear_37_0(utils_constEvalFuncWrapper_110_0, input_112, QuickGELUActivation_36_0_0)
    v_369, v_370, v_371 = CLIPEncoderLayer_38_0(v_367, Linear_37_0_0, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_39_0_0 = Linear_39_0(v_371)
    LayerNorm_40_0_0 = LayerNorm_40_0(
        v_369,
        utils_constEvalFuncWrapper_17_0,
        Linear_39_0_0,
        utils_constEvalFuncWrapper_10_0,
    )
    v_372, v_373, v_374 = Linear_41_0(
        LayerNorm_40_0_0,
        input_167,
        input_169,
        input_108,
        utils_constEvalFuncWrapper_1_0,
        utils_constEvalFuncWrapper_39_0,
        utils_constEvalFuncWrapper_105_0,
    )
    CLIPAttention_42_0_0 = CLIPAttention_42_0(utils_constEvalFuncWrapperZeroArg_3_0, v_372, v_373, v_374)
    Linear_43_0_0 = Linear_43_0(utils_constEvalFuncWrapper_38_0, input_106, CLIPAttention_42_0_0)
    v_375, v_376, v_377 = CLIPEncoderLayer_44_0(v_370, utils_constEvalFuncWrapperZeroArg_1_0, Linear_43_0_0)
    Linear_45_0_0 = Linear_45_0(v_375)
    LayerNorm_46_0_0 = LayerNorm_46_0(
        utils_constEvalFuncWrapper_8_0,
        Linear_45_0_0,
        utils_constEvalFuncWrapper_20_0,
        v_377,
    )
    Linear_47_0_0 = Linear_47_0(input_102, utils_constEvalFuncWrapper_82_0, LayerNorm_46_0_0)
    QuickGELUActivation_48_0_0 = QuickGELUActivation_48_0(utils_constEvalFuncWrapperZeroArg_2_3, Linear_47_0_0)
    Linear_49_0_0 = Linear_49_0(input_100, QuickGELUActivation_48_0_0, utils_constEvalFuncWrapper_21_0)
    v_378, v_379, v_380 = CLIPEncoderLayer_50_0(Linear_49_0_0, v_376, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_51_0_0 = Linear_51_0(v_380)
    LayerNorm_52_0_0 = LayerNorm_52_0(
        utils_constEvalFuncWrapper_106_0,
        utils_constEvalFuncWrapper_77_0,
        v_378,
        Linear_51_0_0,
    )
    v_381, v_382, v_383 = Linear_53_0(
        utils_constEvalFuncWrapper_89_0,
        LayerNorm_52_0_0,
        input_96,
        input_171,
        utils_constEvalFuncWrapper_81_0,
        input_173,
        utils_constEvalFuncWrapper_62_0,
    )
    CLIPAttention_54_0_0 = CLIPAttention_54_0(utils_constEvalFuncWrapperZeroArg_3_0, v_381, v_382, v_383)
    Linear_55_0_0 = Linear_55_0(utils_constEvalFuncWrapper_85_0, input_94, CLIPAttention_54_0_0)
    v_384, v_385, v_386 = CLIPEncoderLayer_56_0(Linear_55_0_0, v_379, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_57_0_0 = Linear_57_0(v_384)
    LayerNorm_58_0_0 = LayerNorm_58_0(
        Linear_57_0_0,
        v_385,
        utils_constEvalFuncWrapper_34_0,
        utils_constEvalFuncWrapper_114_0,
    )
    Linear_59_0_0 = Linear_59_0(LayerNorm_58_0_0, input_90, utils_constEvalFuncWrapper_41_0)
    QuickGELUActivation_60_0_0 = QuickGELUActivation_60_0(utils_constEvalFuncWrapperZeroArg_2_4, Linear_59_0_0)
    Linear_61_0_0 = Linear_61_0(utils_constEvalFuncWrapper_55_0, input_88, QuickGELUActivation_60_0_0)
    v_387, v_388, v_389 = CLIPEncoderLayer_62_0(Linear_61_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_386)
    Linear_63_0_0 = Linear_63_0(v_388)
    LayerNorm_64_0_0 = LayerNorm_64_0(
        Linear_63_0_0,
        utils_constEvalFuncWrapper_102_0,
        utils_constEvalFuncWrapper_14_0,
        v_389,
    )
    v_390, v_391, v_392 = Linear_65_0(
        utils_constEvalFuncWrapper_25_0,
        LayerNorm_64_0_0,
        utils_constEvalFuncWrapper_92_0,
        input_175,
        input_177,
        input_84,
        utils_constEvalFuncWrapper_72_0,
    )
    CLIPAttention_66_0_0 = CLIPAttention_66_0(utils_constEvalFuncWrapperZeroArg_3_0, v_390, v_391, v_392)
    Linear_67_0_0 = Linear_67_0(CLIPAttention_66_0_0, utils_constEvalFuncWrapper_65_0, input_82)
    v_393, v_394, v_395 = CLIPEncoderLayer_68_0(Linear_67_0_0, v_387, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_69_0_0 = Linear_69_0(v_394)
    LayerNorm_70_0_0 = LayerNorm_70_0(
        utils_constEvalFuncWrapper_125_0,
        v_393,
        Linear_69_0_0,
        utils_constEvalFuncWrapper_80_0,
    )
    Linear_71_0_0 = Linear_71_0(input_78, utils_constEvalFuncWrapper_40_0, LayerNorm_70_0_0)
    QuickGELUActivation_72_0_0 = QuickGELUActivation_72_0(Linear_71_0_0, utils_constEvalFuncWrapperZeroArg_2_5)
    Linear_73_0_0 = Linear_73_0(input_76, utils_constEvalFuncWrapper_116_0, QuickGELUActivation_72_0_0)
    v_396, v_397, v_398 = CLIPEncoderLayer_74_0(Linear_73_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_395)
    Linear_75_0_0 = Linear_75_0(v_397)
    LayerNorm_76_0_0 = LayerNorm_76_0(
        v_396,
        Linear_75_0_0,
        utils_constEvalFuncWrapper_109_0,
        utils_constEvalFuncWrapper_50_0,
    )
    v_399, v_400, v_401 = Linear_77_0(
        utils_constEvalFuncWrapper_113_0,
        input_179,
        utils_constEvalFuncWrapper_123_0,
        LayerNorm_76_0_0,
        input_181,
        input_72,
        utils_constEvalFuncWrapper_45_0,
    )
    CLIPAttention_78_0_0 = CLIPAttention_78_0(utils_constEvalFuncWrapperZeroArg_3_0, v_399, v_400, v_401)
    Linear_79_0_0 = Linear_79_0(CLIPAttention_78_0_0, input_70, utils_constEvalFuncWrapper_103_0)
    v_402, v_403, v_404 = CLIPEncoderLayer_80_0(v_398, Linear_79_0_0, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_81_0_0 = Linear_81_0(v_404)
    LayerNorm_82_0_0 = LayerNorm_82_0(
        v_402,
        Linear_81_0_0,
        utils_constEvalFuncWrapper_76_0,
        utils_constEvalFuncWrapper_84_0,
    )
    Linear_83_0_0 = Linear_83_0(LayerNorm_82_0_0, input_66, utils_constEvalFuncWrapper_24_0)
    QuickGELUActivation_84_0_0 = QuickGELUActivation_84_0(utils_constEvalFuncWrapperZeroArg_2_6, Linear_83_0_0)
    Linear_85_0_0 = Linear_85_0(utils_constEvalFuncWrapper_96_0, QuickGELUActivation_84_0_0, input_64)
    v_405, v_406, v_407 = CLIPEncoderLayer_86_0(v_403, utils_constEvalFuncWrapperZeroArg_1_0, Linear_85_0_0)
    Linear_87_0_0 = Linear_87_0(v_405)
    LayerNorm_88_0_0 = LayerNorm_88_0(
        Linear_87_0_0,
        v_406,
        utils_constEvalFuncWrapper_54_0,
        utils_constEvalFuncWrapper_23_0,
    )
    v_408, v_409, v_410 = Linear_89_0(
        input_183,
        utils_constEvalFuncWrapper_71_0,
        utils_constEvalFuncWrapper_19_0,
        LayerNorm_88_0_0,
        input_185,
        input_60,
        utils_constEvalFuncWrapper_12_0,
    )
    CLIPAttention_90_0_0 = CLIPAttention_90_0(utils_constEvalFuncWrapperZeroArg_3_0, v_408, v_409, v_410)
    Linear_91_0_0 = Linear_91_0(input_58, utils_constEvalFuncWrapper_3_0, CLIPAttention_90_0_0)
    v_411, v_412, v_413 = CLIPEncoderLayer_92_0(Linear_91_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_407)
    Linear_93_0_0 = Linear_93_0(v_412)
    LayerNorm_94_0_0 = LayerNorm_94_0(
        v_411,
        utils_constEvalFuncWrapper_73_0,
        utils_constEvalFuncWrapper_99_0,
        Linear_93_0_0,
    )
    Linear_95_0_0 = Linear_95_0(utils_constEvalFuncWrapper_83_0, input_54, LayerNorm_94_0_0)
    QuickGELUActivation_96_0_0 = QuickGELUActivation_96_0(Linear_95_0_0, utils_constEvalFuncWrapperZeroArg_2_7)
    Linear_97_0_0 = Linear_97_0(QuickGELUActivation_96_0_0, utils_constEvalFuncWrapper_107_0, input_52)
    v_414, v_415, v_416 = CLIPEncoderLayer_98_0(Linear_97_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_413)
    Linear_99_0_0 = Linear_99_0(v_414)
    LayerNorm_100_0_0 = LayerNorm_100_0(
        Linear_99_0_0,
        v_415,
        utils_constEvalFuncWrapper_32_0,
        utils_constEvalFuncWrapper_98_0,
    )
    v_417, v_418, v_419 = Linear_101_0(
        utils_constEvalFuncWrapper_69_0,
        input_187,
        LayerNorm_100_0_0,
        utils_constEvalFuncWrapper_122_0,
        input_189,
        utils_constEvalFuncWrapper_118_0,
        input_48,
    )
    CLIPAttention_102_0_0 = CLIPAttention_102_0(utils_constEvalFuncWrapperZeroArg_3_0, v_417, v_418, v_419)
    Linear_103_0_0 = Linear_103_0(CLIPAttention_102_0_0, input_46, utils_constEvalFuncWrapper_61_0)
    v_420, v_421, v_422 = CLIPEncoderLayer_104_0(v_416, Linear_103_0_0, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_105_0_0 = Linear_105_0(v_421)
    LayerNorm_106_0_0 = LayerNorm_106_0(
        utils_constEvalFuncWrapper_70_0,
        Linear_105_0_0,
        utils_constEvalFuncWrapper_2_0,
        v_422,
    )
    Linear_107_0_0 = Linear_107_0(utils_constEvalFuncWrapper_6_0, LayerNorm_106_0_0, input_42)
    QuickGELUActivation_108_0_0 = QuickGELUActivation_108_0(Linear_107_0_0, utils_constEvalFuncWrapperZeroArg_2_8)
    Linear_109_0_0 = Linear_109_0(QuickGELUActivation_108_0_0, input_40, utils_constEvalFuncWrapper_58_0)
    v_423, v_424, v_425 = CLIPEncoderLayer_110_0(Linear_109_0_0, v_420, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_111_0_0 = Linear_111_0(v_425)
    LayerNorm_112_0_0 = LayerNorm_112_0(
        v_423,
        Linear_111_0_0,
        utils_constEvalFuncWrapper_120_0,
        utils_constEvalFuncWrapper_88_0,
    )
    v_426, v_427, v_428 = Linear_113_0(
        utils_constEvalFuncWrapper_27_0,
        input_193,
        utils_constEvalFuncWrapper_79_0,
        LayerNorm_112_0_0,
        utils_constEvalFuncWrapper_100_0,
        input_191,
        input_36,
    )
    CLIPAttention_114_0_0 = CLIPAttention_114_0(utils_constEvalFuncWrapperZeroArg_3_0, v_426, v_427, v_428)
    Linear_115_0_0 = Linear_115_0(utils_constEvalFuncWrapper_29_0, input_34, CLIPAttention_114_0_0)
    v_429, v_430, v_431 = CLIPEncoderLayer_116_0(v_424, Linear_115_0_0, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_117_0_0 = Linear_117_0(v_429)
    LayerNorm_118_0_0 = LayerNorm_118_0(
        utils_constEvalFuncWrapper_93_0,
        Linear_117_0_0,
        utils_constEvalFuncWrapper_53_0,
        v_430,
    )
    Linear_119_0_0 = Linear_119_0(LayerNorm_118_0_0, input_30, utils_constEvalFuncWrapper_90_0)
    QuickGELUActivation_120_0_0 = QuickGELUActivation_120_0(utils_constEvalFuncWrapperZeroArg_2_9, Linear_119_0_0)
    Linear_121_0_0 = Linear_121_0(utils_constEvalFuncWrapper_97_0, input_28, QuickGELUActivation_120_0_0)
    v_432, v_433, v_434 = CLIPEncoderLayer_122_0(Linear_121_0_0, utils_constEvalFuncWrapperZeroArg_1_0, v_431)
    Linear_123_0_0 = Linear_123_0(v_434)
    LayerNorm_124_0_0 = LayerNorm_124_0(
        utils_constEvalFuncWrapper_57_0,
        v_433,
        utils_constEvalFuncWrapper_16_0,
        Linear_123_0_0,
    )
    v_435, v_436, v_437 = Linear_125_0(
        utils_constEvalFuncWrapper_4_0,
        utils_constEvalFuncWrapper_101_0,
        input_195,
        input_24,
        utils_constEvalFuncWrapper_121_0,
        input_197,
        LayerNorm_124_0_0,
    )
    CLIPAttention_126_0_0 = CLIPAttention_126_0(utils_constEvalFuncWrapperZeroArg_3_0, v_435, v_436, v_437)
    Linear_127_0_0 = Linear_127_0(CLIPAttention_126_0_0, input_22, utils_constEvalFuncWrapper_78_0)
    v_438, v_439, v_440 = CLIPEncoderLayer_128_0(Linear_127_0_0, v_432, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_129_0_0 = Linear_129_0(v_438)
    LayerNorm_130_0_0 = LayerNorm_130_0(
        Linear_129_0_0,
        utils_constEvalFuncWrapper_119_0,
        utils_constEvalFuncWrapper_7_0,
        v_440,
    )
    Linear_131_0_0 = Linear_131_0(utils_constEvalFuncWrapper_35_0, input_18, LayerNorm_130_0_0)
    QuickGELUActivation_132_0_0 = QuickGELUActivation_132_0(Linear_131_0_0, utils_constEvalFuncWrapperZeroArg_2_10)
    Linear_133_0_0 = Linear_133_0(input_16, QuickGELUActivation_132_0_0, utils_constEvalFuncWrapper_0_0)
    v_441, v_442, v_443 = CLIPEncoderLayer_134_0(Linear_133_0_0, v_439, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_135_0_0 = Linear_135_0(v_441)
    LayerNorm_136_0_0 = LayerNorm_136_0(
        Linear_135_0_0,
        v_442,
        utils_constEvalFuncWrapper_48_0,
        utils_constEvalFuncWrapper_56_0,
    )
    v_444, v_445, v_446 = Linear_137_0(
        LayerNorm_136_0_0,
        utils_constEvalFuncWrapper_11_0,
        utils_constEvalFuncWrapper_75_0,
        input_199,
        input_201,
        utils_constEvalFuncWrapper_22_0,
        input_12,
    )
    CLIPAttention_138_0_0 = CLIPAttention_138_0(utils_constEvalFuncWrapperZeroArg_3_0, v_444, v_445, v_446)
    Linear_139_0_0 = Linear_139_0(CLIPAttention_138_0_0, utils_constEvalFuncWrapper_30_0, input_10)
    v_447, v_448, v_449 = CLIPEncoderLayer_140_0(Linear_139_0_0, v_443, utils_constEvalFuncWrapperZeroArg_1_0)
    Linear_141_0_0 = Linear_141_0(v_449)
    LayerNorm_142_0_0 = LayerNorm_142_0(
        v_447,
        Linear_141_0_0,
        utils_constEvalFuncWrapper_91_0,
        utils_constEvalFuncWrapper_18_0,
    )
    Linear_143_0_0 = Linear_143_0(input_6, utils_constEvalFuncWrapper_47_0, LayerNorm_142_0_0)
    QuickGELUActivation_144_0_0 = QuickGELUActivation_144_0(Linear_143_0_0, utils_constEvalFuncWrapperZeroArg_2_11)
    Linear_145_0_0 = Linear_145_0(utils_constEvalFuncWrapper_117_0, QuickGELUActivation_144_0_0, input_4)
    CLIPEncoderLayer_146_0_0 = CLIPEncoderLayer_146_0(v_448, Linear_145_0_0)
    CLIPVisionTransformer_147_0_0 = CLIPVisionTransformer_147_0(
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapper_94_0,
        utils_constEvalFuncWrapper_112_0,
        CLIPEncoderLayer_146_0_0,
    )
    Linear_148_0_0 = Linear_148_0(CLIPVisionTransformer_147_0_0, input_0)
    util_create_list_256 = [Linear_148_0_0, CLIPEncoderLayer_146_0_0]
    return util_create_list_256


def Linear_141_0(input):
    ttnn_reshape_197 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_197


def LayerNorm_22_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_0 = ttnn.multiply(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_multiply_0,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_0 = ttnn.add(
        ttnn_multiply_1,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_0


def CLIPAttention_126_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_0 = ttnn.matmul(
        input_1,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_198 = ttnn.reshape(
        ttnn_matmul_0,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_reshape_198,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_multiply_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_0 = ttnn.softmax(
        ttnn_typecast_1,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_softmax_0,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_199 = ttnn.reshape(
        ttnn_typecast_2,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_reshape_199,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_200 = ttnn.reshape(
        ttnn_matmul_1,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_200


def CLIPEncoderLayer_56_0(input_0, input_1, input_2):
    ttnn_add_1 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_0 = ttnn.mean(
        ttnn_add_1,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_0 = ttnn.neg(
        ttnn_mean_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_2 = ttnn.add(
        ttnn_add_1,
        ttnn_neg_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_add_2,
        ttnn_add_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_1 = ttnn.mean(
        ttnn_multiply_3,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_3 = ttnn.add(
        ttnn_mean_1,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_0 = ttnn.rsqrt(
        ttnn_add_3,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_201 = ttnn.reshape(
        ttnn_rsqrt_0,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_2, ttnn_reshape_201, ttnn_add_1


def Linear_37_0(input_0, input_1, input_2):
    ttnn_matmul_2 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_4 = ttnn.add(
        ttnn_matmul_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_202 = ttnn.reshape(
        ttnn_add_4,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_202


def CLIPEncoderLayer_38_0(input_0, input_1, input_2):
    ttnn_add_5 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_2 = ttnn.mean(
        ttnn_add_5,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_1 = ttnn.neg(
        ttnn_mean_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_6 = ttnn.add(
        ttnn_add_5,
        ttnn_neg_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_add_6,
        ttnn_add_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_3 = ttnn.mean(
        ttnn_multiply_4,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_7 = ttnn.add(
        ttnn_mean_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_1 = ttnn.rsqrt(
        ttnn_add_7,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_203 = ttnn.reshape(
        ttnn_rsqrt_1,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_203, ttnn_add_5, ttnn_add_6


def QuickGELUActivation_60_0(input_0, input_1):
    ttnn_multiply_5 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_0 = ttnn.sigmoid(
        ttnn_multiply_5,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_6 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_6


def LayerNorm_70_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_7 = ttnn.multiply(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_multiply_7,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_8 = ttnn.add(
        ttnn_multiply_8,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_8


def LayerNorm_34_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_9 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_10 = ttnn.multiply(
        ttnn_multiply_9,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_9 = ttnn.add(
        ttnn_multiply_10,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_9


def CLIPEncoderLayer_2_0(input_0, input_1):
    ttnn_mean_4 = ttnn.mean(
        input_1,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_2 = ttnn.neg(
        ttnn_mean_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_10 = ttnn.add(
        input_1,
        ttnn_neg_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_11 = ttnn.multiply(
        ttnn_add_10,
        ttnn_add_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_5 = ttnn.mean(
        ttnn_multiply_11,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_11 = ttnn.add(
        ttnn_mean_5,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_2 = ttnn.rsqrt(
        ttnn_add_11,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_204 = ttnn.reshape(
        ttnn_rsqrt_2,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_204, ttnn_add_10


def Linear_33_0(input):
    ttnn_reshape_205 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_205


def LayerNorm_28_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_12 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_13 = ttnn.multiply(
        ttnn_multiply_12,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_12 = ttnn.add(
        ttnn_multiply_13,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_12


def Linear_119_0(input_0, input_1, input_2):
    ttnn_linear_0 = ttnn.linear(
        input_0,
        input_1,
        bias=input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_0


def Linear_65_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_3 = ttnn.matmul(
        input_1,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_4 = ttnn.matmul(
        input_1,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_5 = ttnn.matmul(
        input_1,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_13 = ttnn.add(
        ttnn_matmul_4,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_14 = ttnn.add(
        ttnn_matmul_5,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_15 = ttnn.add(
        ttnn_matmul_3,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_206 = ttnn.reshape(
        ttnn_add_13,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_207 = ttnn.reshape(
        ttnn_add_14,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_208 = ttnn.reshape(
        ttnn_add_15,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_50 = ttnn.permute(
        ttnn_reshape_207,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_51 = ttnn.permute(
        ttnn_reshape_208,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_52 = ttnn.permute(
        ttnn_reshape_206,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_209 = ttnn.reshape(
        ttnn_permute_50,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_210 = ttnn.reshape(
        ttnn_permute_51,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_211 = ttnn.reshape(
        ttnn_permute_52,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_209, ttnn_reshape_211, ttnn_reshape_210


def LayerNorm_4_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_14 = ttnn.multiply(
        input_2,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_15 = ttnn.multiply(
        ttnn_multiply_14,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_16 = ttnn.add(
        ttnn_multiply_15,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_16


def QuickGELUActivation_12_0(input_0, input_1):
    ttnn_multiply_16 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_1 = ttnn.sigmoid(
        ttnn_multiply_16,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_17 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_17


def Linear_137_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_6 = ttnn.matmul(
        input_0,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_7 = ttnn.matmul(
        input_0,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_8 = ttnn.matmul(
        input_0,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_17 = ttnn.add(
        ttnn_matmul_7,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_18 = ttnn.add(
        ttnn_matmul_8,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_19 = ttnn.add(
        ttnn_matmul_6,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_212 = ttnn.reshape(
        ttnn_add_17,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_213 = ttnn.reshape(
        ttnn_add_18,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_214 = ttnn.reshape(
        ttnn_add_19,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_53 = ttnn.permute(
        ttnn_reshape_213,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_54 = ttnn.permute(
        ttnn_reshape_214,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_55 = ttnn.permute(
        ttnn_reshape_212,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_215 = ttnn.reshape(
        ttnn_permute_53,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_216 = ttnn.reshape(
        ttnn_permute_54,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_217 = ttnn.reshape(
        ttnn_permute_55,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_216, ttnn_reshape_217, ttnn_reshape_215


def Linear_139_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_24 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_218 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_24,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_9 = ttnn.matmul(
        ttnn_reshape_218,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_20 = ttnn.add(
        ttnn_matmul_9,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_219 = ttnn.reshape(
        ttnn_add_20,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_219


def CLIPEncoderLayer_122_0(input_0, input_1, input_2):
    ttnn_add_21 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_6 = ttnn.mean(
        ttnn_add_21,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_3 = ttnn.neg(
        ttnn_mean_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_22 = ttnn.add(
        ttnn_add_21,
        ttnn_neg_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_18 = ttnn.multiply(
        ttnn_add_22,
        ttnn_add_22,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_7 = ttnn.mean(
        ttnn_multiply_18,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_23 = ttnn.add(
        ttnn_mean_7,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_add_23,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_220 = ttnn.reshape(
        ttnn_rsqrt_3,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_21, ttnn_reshape_220, ttnn_add_22


def LayerNorm_118_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_19 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_20 = ttnn.multiply(
        ttnn_multiply_19,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_24 = ttnn.add(
        ttnn_multiply_20,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_24


def CLIPEncoderLayer_14_0(input_0, input_1, input_2):
    ttnn_add_25 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_8 = ttnn.mean(
        ttnn_add_25,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_4 = ttnn.neg(
        ttnn_mean_8,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_26 = ttnn.add(
        ttnn_add_25,
        ttnn_neg_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_21 = ttnn.multiply(
        ttnn_add_26,
        ttnn_add_26,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_9 = ttnn.mean(
        ttnn_multiply_21,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_27 = ttnn.add(
        ttnn_mean_9,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_4 = ttnn.rsqrt(
        ttnn_add_27,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_221 = ttnn.reshape(
        ttnn_rsqrt_4,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_25, ttnn_reshape_221, ttnn_add_26


def Linear_107_0(input_0, input_1, input_2):
    ttnn_linear_1 = ttnn.linear(
        input_1,
        input_2,
        bias=input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_1


def Linear_83_0(input_0, input_1, input_2):
    ttnn_linear_2 = ttnn.linear(
        input_0,
        input_1,
        bias=input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_2


def Linear_17_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_10 = ttnn.matmul(
        input_1,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_11 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_12 = ttnn.matmul(
        input_1,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_28 = ttnn.add(
        ttnn_matmul_11,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_29 = ttnn.add(
        ttnn_matmul_12,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_30 = ttnn.add(
        ttnn_matmul_10,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_222 = ttnn.reshape(
        ttnn_add_28,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_223 = ttnn.reshape(
        ttnn_add_29,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_224 = ttnn.reshape(
        ttnn_add_30,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_56 = ttnn.permute(
        ttnn_reshape_223,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_57 = ttnn.permute(
        ttnn_reshape_224,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_58 = ttnn.permute(
        ttnn_reshape_222,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_225 = ttnn.reshape(
        ttnn_permute_56,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_226 = ttnn.reshape(
        ttnn_permute_57,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_227 = ttnn.reshape(
        ttnn_permute_58,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_226, ttnn_reshape_225, ttnn_reshape_227


def Linear_101_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_13 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_14 = ttnn.matmul(
        input_2,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_15 = ttnn.matmul(
        input_2,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_31 = ttnn.add(
        ttnn_matmul_14,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_32 = ttnn.add(
        ttnn_matmul_15,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_33 = ttnn.add(
        ttnn_matmul_13,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_228 = ttnn.reshape(
        ttnn_add_31,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_229 = ttnn.reshape(
        ttnn_add_32,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_230 = ttnn.reshape(
        ttnn_add_33,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_59 = ttnn.permute(
        ttnn_reshape_229,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_60 = ttnn.permute(
        ttnn_reshape_230,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_61 = ttnn.permute(
        ttnn_reshape_228,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_231 = ttnn.reshape(
        ttnn_permute_59,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_232 = ttnn.reshape(
        ttnn_permute_60,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_233 = ttnn.reshape(
        ttnn_permute_61,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_232, ttnn_reshape_233, ttnn_reshape_231


def CLIPVisionEmbeddings_0_0(input_0, input_1, input_2, input_3, input_4):
    ttnn_permute_62 = ttnn.permute(
        input_1,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_234 = ttnn.reshape(
        ttnn_permute_62,
        [1, 1, 100352, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_reshape_234,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_layout_1,
        weight_tensor=input_4,
        device=input_2,
        in_channels=3,
        out_channels=768,
        batch_size=2,
        input_height=224,
        input_width=224,
        kernel_size=[32, 32],
        stride=[32, 32],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT16,
            deallocate_activation=False,
            reallocate_halo_output=False,
            act_block_h_override=0,
            act_block_w_div=1,
            reshard_if_not_optimal=False,
            override_sharding_config=False,
            transpose_shards=False,
            output_layout=ttnn.Layout.TILE,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            in_place=False,
            enable_kernel_stride_folding=False,
        ),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_235 = ttnn.reshape(
        ttnn_conv2d_0,
        [2, 7, 7, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_63 = ttnn.permute(
        ttnn_reshape_235,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_236 = ttnn.reshape(
        ttnn_permute_63,
        [2, 768, 49],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_257 = [input_0, ttnn_reshape_236]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_257,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_34 = ttnn.add(
        ttnn_concat_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_64 = ttnn.permute(
        ttnn_add_34,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    return ttnn_permute_64


def LayerNorm_106_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_22 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_23 = ttnn.multiply(
        ttnn_multiply_22,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_35 = ttnn.add(
        ttnn_multiply_23,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_35


def Linear_73_0(input_0, input_1, input_2):
    ttnn_matmul_16 = ttnn.matmul(
        input_2,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_36 = ttnn.add(
        ttnn_matmul_16,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_237 = ttnn.reshape(
        ttnn_add_36,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_237


def Linear_121_0(input_0, input_1, input_2):
    ttnn_matmul_17 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_37 = ttnn.add(
        ttnn_matmul_17,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_238 = ttnn.reshape(
        ttnn_add_37,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_238


def Linear_105_0(input):
    ttnn_reshape_239 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_239


def Linear_133_0(input_0, input_1, input_2):
    ttnn_matmul_18 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_38 = ttnn.add(
        ttnn_matmul_18,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_240 = ttnn.reshape(
        ttnn_add_38,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_240


def Linear_131_0(input_0, input_1, input_2):
    ttnn_linear_3 = ttnn.linear(
        input_2,
        input_1,
        bias=input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_3


def Linear_7_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_25 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_241 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_25,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_19 = ttnn.matmul(
        ttnn_reshape_241,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_39 = ttnn.add(
        ttnn_matmul_19,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_242 = ttnn.reshape(
        ttnn_add_39,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_242


def QuickGELUActivation_96_0(input_0, input_1):
    ttnn_multiply_24 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_2 = ttnn.sigmoid(
        ttnn_multiply_24,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_25 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_25


def QuickGELUActivation_132_0(input_0, input_1):
    ttnn_multiply_26 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_3 = ttnn.sigmoid(
        ttnn_multiply_26,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_27 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_27


def QuickGELUActivation_48_0(input_0, input_1):
    ttnn_multiply_28 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_4 = ttnn.sigmoid(
        ttnn_multiply_28,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_29 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_29


def CLIPEncoderLayer_80_0(input_0, input_1, input_2):
    ttnn_add_40 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_10 = ttnn.mean(
        ttnn_add_40,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_5 = ttnn.neg(
        ttnn_mean_10,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_41 = ttnn.add(
        ttnn_add_40,
        ttnn_neg_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_30 = ttnn.multiply(
        ttnn_add_41,
        ttnn_add_41,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_11 = ttnn.mean(
        ttnn_multiply_30,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_42 = ttnn.add(
        ttnn_mean_11,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_5 = ttnn.rsqrt(
        ttnn_add_42,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_243 = ttnn.reshape(
        ttnn_rsqrt_5,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_243, ttnn_add_40, ttnn_add_41


def QuickGELUActivation_84_0(input_0, input_1):
    ttnn_multiply_31 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_5 = ttnn.sigmoid(
        ttnn_multiply_31,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_32 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_32


def LayerNorm_58_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_33 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_34 = ttnn.multiply(
        ttnn_multiply_33,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_43 = ttnn.add(
        ttnn_multiply_34,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_43


def CLIPEncoderLayer_26_0(input_0, input_1, input_2):
    ttnn_add_44 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_12 = ttnn.mean(
        ttnn_add_44,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_6 = ttnn.neg(
        ttnn_mean_12,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_45 = ttnn.add(
        ttnn_add_44,
        ttnn_neg_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_35 = ttnn.multiply(
        ttnn_add_45,
        ttnn_add_45,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_13 = ttnn.mean(
        ttnn_multiply_35,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_46 = ttnn.add(
        ttnn_mean_13,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_6 = ttnn.rsqrt(
        ttnn_add_46,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_244 = ttnn.reshape(
        ttnn_rsqrt_6,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_45, ttnn_add_44, ttnn_reshape_244


def Linear_67_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_26 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_245 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_26,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_20 = ttnn.matmul(
        ttnn_reshape_245,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_47 = ttnn.add(
        ttnn_matmul_20,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_246 = ttnn.reshape(
        ttnn_add_47,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_246


def LayerNorm_40_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_36 = ttnn.multiply(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_37 = ttnn.multiply(
        ttnn_multiply_36,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_48 = ttnn.add(
        ttnn_multiply_37,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_48


def CLIPEncoderLayer_32_0(input_0, input_1, input_2):
    ttnn_add_49 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_14 = ttnn.mean(
        ttnn_add_49,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_7 = ttnn.neg(
        ttnn_mean_14,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_50 = ttnn.add(
        ttnn_add_49,
        ttnn_neg_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_38 = ttnn.multiply(
        ttnn_add_50,
        ttnn_add_50,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_15 = ttnn.mean(
        ttnn_multiply_38,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_51 = ttnn.add(
        ttnn_mean_15,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_7 = ttnn.rsqrt(
        ttnn_add_51,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_247 = ttnn.reshape(
        ttnn_rsqrt_7,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_247, ttnn_add_49, ttnn_add_50


def Linear_19_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_27 = ttnn.transformer.concatenate_heads(
        input_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_248 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_27,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_21 = ttnn.matmul(
        ttnn_reshape_248,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_52 = ttnn.add(
        ttnn_matmul_21,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_249 = ttnn.reshape(
        ttnn_add_52,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_249


def CLIPEncoderLayer_50_0(input_0, input_1, input_2):
    ttnn_add_53 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_16 = ttnn.mean(
        ttnn_add_53,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_8 = ttnn.neg(
        ttnn_mean_16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_54 = ttnn.add(
        ttnn_add_53,
        ttnn_neg_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_39 = ttnn.multiply(
        ttnn_add_54,
        ttnn_add_54,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_17 = ttnn.mean(
        ttnn_multiply_39,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_55 = ttnn.add(
        ttnn_mean_17,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_8 = ttnn.rsqrt(
        ttnn_add_55,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_250 = ttnn.reshape(
        ttnn_rsqrt_8,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_250, ttnn_add_53, ttnn_add_54


def Linear_29_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_22 = ttnn.matmul(
        input_3,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_23 = ttnn.matmul(
        input_3,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_24 = ttnn.matmul(
        input_3,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_56 = ttnn.add(
        ttnn_matmul_23,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_57 = ttnn.add(
        ttnn_matmul_24,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_58 = ttnn.add(
        ttnn_matmul_22,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_251 = ttnn.reshape(
        ttnn_add_56,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_252 = ttnn.reshape(
        ttnn_add_57,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_253 = ttnn.reshape(
        ttnn_add_58,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_65 = ttnn.permute(
        ttnn_reshape_252,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_66 = ttnn.permute(
        ttnn_reshape_253,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_67 = ttnn.permute(
        ttnn_reshape_251,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_254 = ttnn.reshape(
        ttnn_permute_65,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_255 = ttnn.reshape(
        ttnn_permute_66,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_256 = ttnn.reshape(
        ttnn_permute_67,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_255, ttnn_reshape_256, ttnn_reshape_254


def Linear_61_0(input_0, input_1, input_2):
    ttnn_matmul_25 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_59 = ttnn.add(
        ttnn_matmul_25,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_257 = ttnn.reshape(
        ttnn_add_59,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_257


def CLIPAttention_30_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_26 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_258 = ttnn.reshape(
        ttnn_matmul_26,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_40 = ttnn.multiply(
        ttnn_reshape_258,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_multiply_40,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_1 = ttnn.softmax(
        ttnn_typecast_3,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_softmax_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_259 = ttnn.reshape(
        ttnn_typecast_4,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_reshape_259,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_260 = ttnn.reshape(
        ttnn_matmul_27,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_260


def Linear_41_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_28 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_29 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_30 = ttnn.matmul(
        input_0,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_60 = ttnn.add(
        ttnn_matmul_29,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_61 = ttnn.add(
        ttnn_matmul_30,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_62 = ttnn.add(
        ttnn_matmul_28,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_261 = ttnn.reshape(
        ttnn_add_60,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_262 = ttnn.reshape(
        ttnn_add_61,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_263 = ttnn.reshape(
        ttnn_add_62,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_68 = ttnn.permute(
        ttnn_reshape_262,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_69 = ttnn.permute(
        ttnn_reshape_263,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_70 = ttnn.permute(
        ttnn_reshape_261,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_264 = ttnn.reshape(
        ttnn_permute_68,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_265 = ttnn.reshape(
        ttnn_permute_69,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_266 = ttnn.reshape(
        ttnn_permute_70,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_265, ttnn_reshape_264, ttnn_reshape_266


def Linear_97_0(input_0, input_1, input_2):
    ttnn_matmul_31 = ttnn.matmul(
        input_0,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_63 = ttnn.add(
        ttnn_matmul_31,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_267 = ttnn.reshape(
        ttnn_add_63,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_267


def CLIPEncoderLayer_44_0(input_0, input_1, input_2):
    ttnn_add_64 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_18 = ttnn.mean(
        ttnn_add_64,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_9 = ttnn.neg(
        ttnn_mean_18,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_65 = ttnn.add(
        ttnn_add_64,
        ttnn_neg_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_41 = ttnn.multiply(
        ttnn_add_65,
        ttnn_add_65,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_19 = ttnn.mean(
        ttnn_multiply_41,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_66 = ttnn.add(
        ttnn_mean_19,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_9 = ttnn.rsqrt(
        ttnn_add_66,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_268 = ttnn.reshape(
        ttnn_rsqrt_9,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_65, ttnn_add_64, ttnn_reshape_268


def CLIPEncoderLayer_68_0(input_0, input_1, input_2):
    ttnn_add_67 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_20 = ttnn.mean(
        ttnn_add_67,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_10 = ttnn.neg(
        ttnn_mean_20,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_68 = ttnn.add(
        ttnn_add_67,
        ttnn_neg_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_42 = ttnn.multiply(
        ttnn_add_68,
        ttnn_add_68,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_21 = ttnn.mean(
        ttnn_multiply_42,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_69 = ttnn.add(
        ttnn_mean_21,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_10 = ttnn.rsqrt(
        ttnn_add_69,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_269 = ttnn.reshape(
        ttnn_rsqrt_10,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_269, ttnn_add_68, ttnn_add_67


def CLIPEncoderLayer_116_0(input_0, input_1, input_2):
    ttnn_add_70 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_22 = ttnn.mean(
        ttnn_add_70,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_11 = ttnn.neg(
        ttnn_mean_22,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_71 = ttnn.add(
        ttnn_add_70,
        ttnn_neg_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_43 = ttnn.multiply(
        ttnn_add_71,
        ttnn_add_71,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_23 = ttnn.mean(
        ttnn_multiply_43,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_72 = ttnn.add(
        ttnn_mean_23,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_11 = ttnn.rsqrt(
        ttnn_add_72,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_270 = ttnn.reshape(
        ttnn_rsqrt_11,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_71, ttnn_reshape_270, ttnn_add_70


def Linear_5_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_32 = ttnn.matmul(
        input_4,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_33 = ttnn.matmul(
        input_4,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_34 = ttnn.matmul(
        input_4,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_73 = ttnn.add(
        ttnn_matmul_33,
        input_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_74 = ttnn.add(
        ttnn_matmul_34,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_75 = ttnn.add(
        ttnn_matmul_32,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_271 = ttnn.reshape(
        ttnn_add_73,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_272 = ttnn.reshape(
        ttnn_add_74,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_273 = ttnn.reshape(
        ttnn_add_75,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_71 = ttnn.permute(
        ttnn_reshape_272,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_72 = ttnn.permute(
        ttnn_reshape_273,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_73 = ttnn.permute(
        ttnn_reshape_271,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_274 = ttnn.reshape(
        ttnn_permute_71,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_275 = ttnn.reshape(
        ttnn_permute_72,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_276 = ttnn.reshape(
        ttnn_permute_73,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_274, ttnn_reshape_276, ttnn_reshape_275


def LayerNorm_10_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_44 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_45 = ttnn.multiply(
        ttnn_multiply_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_76 = ttnn.add(
        ttnn_multiply_45,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_76


def Linear_93_0(input):
    ttnn_reshape_277 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_277


def Linear_115_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_28 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_278 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_28,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_35 = ttnn.matmul(
        ttnn_reshape_278,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_77 = ttnn.add(
        ttnn_matmul_35,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_279 = ttnn.reshape(
        ttnn_add_77,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_279


def CLIPEncoderLayer_110_0(input_0, input_1, input_2):
    ttnn_add_78 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_24 = ttnn.mean(
        ttnn_add_78,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_12 = ttnn.neg(
        ttnn_mean_24,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_79 = ttnn.add(
        ttnn_add_78,
        ttnn_neg_12,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_46 = ttnn.multiply(
        ttnn_add_79,
        ttnn_add_79,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_25 = ttnn.mean(
        ttnn_multiply_46,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_80 = ttnn.add(
        ttnn_mean_25,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_12 = ttnn.rsqrt(
        ttnn_add_80,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_280 = ttnn.reshape(
        ttnn_rsqrt_12,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_280, ttnn_add_78, ttnn_add_79


def Linear_71_0(input_0, input_1, input_2):
    ttnn_linear_4 = ttnn.linear(
        input_2,
        input_0,
        bias=input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_4


def QuickGELUActivation_120_0(input_0, input_1):
    ttnn_multiply_47 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_6 = ttnn.sigmoid(
        ttnn_multiply_47,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_48 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_48


def Linear_59_0(input_0, input_1, input_2):
    ttnn_linear_5 = ttnn.linear(
        input_0,
        input_1,
        bias=input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_5


def Linear_129_0(input):
    ttnn_reshape_281 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_281


def CLIPAttention_102_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_36 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_282 = ttnn.reshape(
        ttnn_matmul_36,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_49 = ttnn.multiply(
        ttnn_reshape_282,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_multiply_49,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_2 = ttnn.softmax(
        ttnn_typecast_5,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_softmax_2,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_283 = ttnn.reshape(
        ttnn_typecast_6,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_37 = ttnn.matmul(
        ttnn_reshape_283,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_284 = ttnn.reshape(
        ttnn_matmul_37,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_284


def Linear_13_0(input_0, input_1, input_2):
    ttnn_matmul_38 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_81 = ttnn.add(
        ttnn_matmul_38,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_285 = ttnn.reshape(
        ttnn_add_81,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_285


def Linear_11_0(input_0, input_1, input_2):
    ttnn_linear_6 = ttnn.linear(
        input_0,
        input_2,
        bias=input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_6


def Linear_125_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_39 = ttnn.matmul(
        input_6,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_40 = ttnn.matmul(
        input_6,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_41 = ttnn.matmul(
        input_6,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_82 = ttnn.add(
        ttnn_matmul_40,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_83 = ttnn.add(
        ttnn_matmul_41,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_84 = ttnn.add(
        ttnn_matmul_39,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_286 = ttnn.reshape(
        ttnn_add_82,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_287 = ttnn.reshape(
        ttnn_add_83,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_288 = ttnn.reshape(
        ttnn_add_84,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_74 = ttnn.permute(
        ttnn_reshape_287,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_75 = ttnn.permute(
        ttnn_reshape_288,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_76 = ttnn.permute(
        ttnn_reshape_286,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_289 = ttnn.reshape(
        ttnn_permute_74,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_290 = ttnn.reshape(
        ttnn_permute_75,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_291 = ttnn.reshape(
        ttnn_permute_76,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_291, ttnn_reshape_289, ttnn_reshape_290


def Linear_87_0(input):
    ttnn_reshape_292 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_292


def CLIPAttention_78_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_42 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_293 = ttnn.reshape(
        ttnn_matmul_42,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_50 = ttnn.multiply(
        ttnn_reshape_293,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_multiply_50,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_3 = ttnn.softmax(
        ttnn_typecast_7,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_softmax_3,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_294 = ttnn.reshape(
        ttnn_typecast_8,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_43 = ttnn.matmul(
        ttnn_reshape_294,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_295 = ttnn.reshape(
        ttnn_matmul_43,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_295


def Linear_63_0(input):
    ttnn_reshape_296 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_296


def Linear_75_0(input):
    ttnn_reshape_297 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_297


def LayerNorm_88_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_51 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_52 = ttnn.multiply(
        ttnn_multiply_51,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_85 = ttnn.add(
        ttnn_multiply_52,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_85


def LayerNorm_1_0(input_0, input_1, input_2, input_3):
    ttnn_mean_26 = ttnn.mean(
        input_0,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_13 = ttnn.neg(
        ttnn_mean_26,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_86 = ttnn.add(
        input_0,
        ttnn_neg_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_53 = ttnn.multiply(
        ttnn_add_86,
        ttnn_add_86,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_27 = ttnn.mean(
        ttnn_multiply_53,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_87 = ttnn.add(
        ttnn_mean_27,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_13 = ttnn.rsqrt(
        ttnn_add_87,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_54 = ttnn.multiply(
        ttnn_add_86,
        ttnn_rsqrt_13,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_55 = ttnn.multiply(
        ttnn_multiply_54,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_88 = ttnn.add(
        ttnn_multiply_55,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_88


def Linear_3_0(input):
    ttnn_reshape_298 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_298


def Linear_117_0(input):
    ttnn_reshape_299 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_299


def Linear_103_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_29 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_300 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_29,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_44 = ttnn.matmul(
        ttnn_reshape_300,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_89 = ttnn.add(
        ttnn_matmul_44,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_301 = ttnn.reshape(
        ttnn_add_89,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_301


def Linear_123_0(input):
    ttnn_reshape_302 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_302


def LayerNorm_64_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_56 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_57 = ttnn.multiply(
        ttnn_multiply_56,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_90 = ttnn.add(
        ttnn_multiply_57,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_90


def CLIPEncoderLayer_62_0(input_0, input_1, input_2):
    ttnn_add_91 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_28 = ttnn.mean(
        ttnn_add_91,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_14 = ttnn.neg(
        ttnn_mean_28,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_92 = ttnn.add(
        ttnn_add_91,
        ttnn_neg_14,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_58 = ttnn.multiply(
        ttnn_add_92,
        ttnn_add_92,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_29 = ttnn.mean(
        ttnn_multiply_58,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_93 = ttnn.add(
        ttnn_mean_29,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_14 = ttnn.rsqrt(
        ttnn_add_93,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_303 = ttnn.reshape(
        ttnn_rsqrt_14,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_91, ttnn_add_92, ttnn_reshape_303


def CLIPEncoderLayer_86_0(input_0, input_1, input_2):
    ttnn_add_94 = ttnn.add(
        input_0,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_30 = ttnn.mean(
        ttnn_add_94,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_15 = ttnn.neg(
        ttnn_mean_30,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_95 = ttnn.add(
        ttnn_add_94,
        ttnn_neg_15,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_59 = ttnn.multiply(
        ttnn_add_95,
        ttnn_add_95,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_31 = ttnn.mean(
        ttnn_multiply_59,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_96 = ttnn.add(
        ttnn_mean_31,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_15 = ttnn.rsqrt(
        ttnn_add_96,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_304 = ttnn.reshape(
        ttnn_rsqrt_15,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_95, ttnn_reshape_304, ttnn_add_94


def QuickGELUActivation_108_0(input_0, input_1):
    ttnn_multiply_60 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_7 = ttnn.sigmoid(
        ttnn_multiply_60,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_61 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_7,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_61


def Linear_21_0(input):
    ttnn_reshape_305 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_305


def QuickGELUActivation_144_0(input_0, input_1):
    ttnn_multiply_62 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_8 = ttnn.sigmoid(
        ttnn_multiply_62,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_63 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_63


def CLIPEncoderLayer_104_0(input_0, input_1, input_2):
    ttnn_add_97 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_32 = ttnn.mean(
        ttnn_add_97,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_16 = ttnn.neg(
        ttnn_mean_32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_98 = ttnn.add(
        ttnn_add_97,
        ttnn_neg_16,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_64 = ttnn.multiply(
        ttnn_add_98,
        ttnn_add_98,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_33 = ttnn.mean(
        ttnn_multiply_64,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_99 = ttnn.add(
        ttnn_mean_33,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_16 = ttnn.rsqrt(
        ttnn_add_99,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_306 = ttnn.reshape(
        ttnn_rsqrt_16,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_97, ttnn_add_98, ttnn_reshape_306


def Linear_25_0(input_0, input_1, input_2):
    ttnn_matmul_45 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_100 = ttnn.add(
        ttnn_matmul_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_307 = ttnn.reshape(
        ttnn_add_100,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_307


def LayerNorm_94_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_65 = ttnn.multiply(
        input_3,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_66 = ttnn.multiply(
        ttnn_multiply_65,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_101 = ttnn.add(
        ttnn_multiply_66,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_101


def Linear_57_0(input):
    ttnn_reshape_308 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_308


def Linear_35_0(input_0, input_1, input_2):
    ttnn_linear_7 = ttnn.linear(
        input_0,
        input_1,
        bias=input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_7


def Linear_148_0(input_0, input_1):
    ttnn_matmul_46 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_matmul_46


def Linear_45_0(input):
    ttnn_reshape_309 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_309


def LayerNorm_112_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_67 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_68 = ttnn.multiply(
        ttnn_multiply_67,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_102 = ttnn.add(
        ttnn_multiply_68,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_102


def CLIPEncoderLayer_92_0(input_0, input_1, input_2):
    ttnn_add_103 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_34 = ttnn.mean(
        ttnn_add_103,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_17 = ttnn.neg(
        ttnn_mean_34,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_104 = ttnn.add(
        ttnn_add_103,
        ttnn_neg_17,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_69 = ttnn.multiply(
        ttnn_add_104,
        ttnn_add_104,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_35 = ttnn.mean(
        ttnn_multiply_69,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_105 = ttnn.add(
        ttnn_mean_35,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_17 = ttnn.rsqrt(
        ttnn_add_105,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_310 = ttnn.reshape(
        ttnn_rsqrt_17,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_310, ttnn_add_104, ttnn_add_103


def Linear_27_0(input):
    ttnn_reshape_311 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_311


def LayerNorm_124_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_70 = ttnn.multiply(
        input_3,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_71 = ttnn.multiply(
        ttnn_multiply_70,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_106 = ttnn.add(
        ttnn_multiply_71,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_106


def Linear_143_0(input_0, input_1, input_2):
    ttnn_linear_8 = ttnn.linear(
        input_2,
        input_0,
        bias=input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_8


def CLIPAttention_18_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_47 = ttnn.matmul(
        input_3,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_312 = ttnn.reshape(
        ttnn_matmul_47,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_72 = ttnn.multiply(
        ttnn_reshape_312,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_multiply_72,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_4 = ttnn.softmax(
        ttnn_typecast_9,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_10 = ttnn.typecast(
        ttnn_softmax_4,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_313 = ttnn.reshape(
        ttnn_typecast_10,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_48 = ttnn.matmul(
        ttnn_reshape_313,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_314 = ttnn.reshape(
        ttnn_matmul_48,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_314


def Linear_145_0(input_0, input_1, input_2):
    ttnn_matmul_49 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_107 = ttnn.add(
        ttnn_matmul_49,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_315 = ttnn.reshape(
        ttnn_add_107,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_315


def Linear_49_0(input_0, input_1, input_2):
    ttnn_matmul_50 = ttnn.matmul(
        input_1,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_108 = ttnn.add(
        ttnn_matmul_50,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_316 = ttnn.reshape(
        ttnn_add_108,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_316


def LayerNorm_100_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_73 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_74 = ttnn.multiply(
        ttnn_multiply_73,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_109 = ttnn.add(
        ttnn_multiply_74,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_109


def Linear_91_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_30 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_317 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_30,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_51 = ttnn.matmul(
        ttnn_reshape_317,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_110 = ttnn.add(
        ttnn_matmul_51,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_318 = ttnn.reshape(
        ttnn_add_110,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_318


def Linear_39_0(input):
    ttnn_reshape_319 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_319


def CLIPAttention_54_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_52 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_320 = ttnn.reshape(
        ttnn_matmul_52,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_75 = ttnn.multiply(
        ttnn_reshape_320,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_multiply_75,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_5 = ttnn.softmax(
        ttnn_typecast_11,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_softmax_5,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_321 = ttnn.reshape(
        ttnn_typecast_12,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_53 = ttnn.matmul(
        ttnn_reshape_321,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_322 = ttnn.reshape(
        ttnn_matmul_53,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_322


def QuickGELUActivation_72_0(input_0, input_1):
    ttnn_multiply_76 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_9 = ttnn.sigmoid(
        ttnn_multiply_76,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_77 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_9,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_77


def Linear_77_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_54 = ttnn.matmul(
        input_3,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_55 = ttnn.matmul(
        input_3,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_56 = ttnn.matmul(
        input_3,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_111 = ttnn.add(
        ttnn_matmul_55,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_112 = ttnn.add(
        ttnn_matmul_56,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_113 = ttnn.add(
        ttnn_matmul_54,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_323 = ttnn.reshape(
        ttnn_add_111,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_324 = ttnn.reshape(
        ttnn_add_112,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_325 = ttnn.reshape(
        ttnn_add_113,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_77 = ttnn.permute(
        ttnn_reshape_324,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_78 = ttnn.permute(
        ttnn_reshape_325,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_79 = ttnn.permute(
        ttnn_reshape_323,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_326 = ttnn.reshape(
        ttnn_permute_77,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_327 = ttnn.reshape(
        ttnn_permute_78,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_328 = ttnn.reshape(
        ttnn_permute_79,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_326, ttnn_reshape_328, ttnn_reshape_327


def Linear_47_0(input_0, input_1, input_2):
    ttnn_linear_9 = ttnn.linear(
        input_2,
        input_0,
        bias=input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_9


def CLIPAttention_90_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_57 = ttnn.matmul(
        input_3,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_329 = ttnn.reshape(
        ttnn_matmul_57,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_78 = ttnn.multiply(
        ttnn_reshape_329,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_multiply_78,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_6 = ttnn.softmax(
        ttnn_typecast_13,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_softmax_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_330 = ttnn.reshape(
        ttnn_typecast_14,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_58 = ttnn.matmul(
        ttnn_reshape_330,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_331 = ttnn.reshape(
        ttnn_matmul_58,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_331


def LayerNorm_82_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_79 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_80 = ttnn.multiply(
        ttnn_multiply_79,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_114 = ttnn.add(
        ttnn_multiply_80,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_114


def LayerNorm_136_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_81 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_82 = ttnn.multiply(
        ttnn_multiply_81,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_115 = ttnn.add(
        ttnn_multiply_82,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_115


def LayerNorm_76_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_83 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_84 = ttnn.multiply(
        ttnn_multiply_83,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_116 = ttnn.add(
        ttnn_multiply_84,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_116


def Linear_89_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_59 = ttnn.matmul(
        input_3,
        input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_60 = ttnn.matmul(
        input_3,
        input_4,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_61 = ttnn.matmul(
        input_3,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_117 = ttnn.add(
        ttnn_matmul_60,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_118 = ttnn.add(
        ttnn_matmul_61,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_119 = ttnn.add(
        ttnn_matmul_59,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_332 = ttnn.reshape(
        ttnn_add_117,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_333 = ttnn.reshape(
        ttnn_add_118,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_334 = ttnn.reshape(
        ttnn_add_119,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_80 = ttnn.permute(
        ttnn_reshape_333,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_81 = ttnn.permute(
        ttnn_reshape_334,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_82 = ttnn.permute(
        ttnn_reshape_332,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_335 = ttnn.reshape(
        ttnn_permute_80,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_336 = ttnn.reshape(
        ttnn_permute_81,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_337 = ttnn.reshape(
        ttnn_permute_82,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_335, ttnn_reshape_336, ttnn_reshape_337


def LayerNorm_52_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_85 = ttnn.multiply(
        input_3,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_86 = ttnn.multiply(
        ttnn_multiply_85,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_120 = ttnn.add(
        ttnn_multiply_86,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_120


def Linear_69_0(input):
    ttnn_reshape_338 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_338


def CLIPEncoderLayer_140_0(input_0, input_1, input_2):
    ttnn_add_121 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_36 = ttnn.mean(
        ttnn_add_121,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_18 = ttnn.neg(
        ttnn_mean_36,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_122 = ttnn.add(
        ttnn_add_121,
        ttnn_neg_18,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_87 = ttnn.multiply(
        ttnn_add_122,
        ttnn_add_122,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_37 = ttnn.mean(
        ttnn_multiply_87,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_123 = ttnn.add(
        ttnn_mean_37,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_18 = ttnn.rsqrt(
        ttnn_add_123,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_339 = ttnn.reshape(
        ttnn_rsqrt_18,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_339, ttnn_add_121, ttnn_add_122


def CLIPEncoderLayer_146_0(input_0, input_1):
    ttnn_add_124 = ttnn.add(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_124


def Linear_55_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_31 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_340 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_31,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_62 = ttnn.matmul(
        ttnn_reshape_340,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_125 = ttnn.add(
        ttnn_matmul_62,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_341 = ttnn.reshape(
        ttnn_add_125,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_341


def Linear_127_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_32 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_342 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_32,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_63 = ttnn.matmul(
        ttnn_reshape_342,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_126 = ttnn.add(
        ttnn_matmul_63,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_343 = ttnn.reshape(
        ttnn_add_126,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_343


def Linear_95_0(input_0, input_1, input_2):
    ttnn_linear_10 = ttnn.linear(
        input_2,
        input_1,
        bias=input_0,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_10


def QuickGELUActivation_24_0(input_0, input_1):
    ttnn_multiply_88 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_10 = ttnn.sigmoid(
        ttnn_multiply_88,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_89 = ttnn.multiply(
        input_1,
        ttnn_sigmoid_10,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_89


def Linear_113_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_64 = ttnn.matmul(
        input_3,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_65 = ttnn.matmul(
        input_3,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_66 = ttnn.matmul(
        input_3,
        input_6,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_127 = ttnn.add(
        ttnn_matmul_65,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_128 = ttnn.add(
        ttnn_matmul_66,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_129 = ttnn.add(
        ttnn_matmul_64,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_344 = ttnn.reshape(
        ttnn_add_127,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_345 = ttnn.reshape(
        ttnn_add_128,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_346 = ttnn.reshape(
        ttnn_add_129,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_83 = ttnn.permute(
        ttnn_reshape_345,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_84 = ttnn.permute(
        ttnn_reshape_346,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_85 = ttnn.permute(
        ttnn_reshape_344,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_347 = ttnn.reshape(
        ttnn_permute_83,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_348 = ttnn.reshape(
        ttnn_permute_84,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_349 = ttnn.reshape(
        ttnn_permute_85,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_347, ttnn_reshape_348, ttnn_reshape_349


def LayerNorm_130_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_90 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_91 = ttnn.multiply(
        ttnn_multiply_90,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_130 = ttnn.add(
        ttnn_multiply_91,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_130


def CLIPAttention_42_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_67 = ttnn.matmul(
        input_3,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_350 = ttnn.reshape(
        ttnn_matmul_67,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_92 = ttnn.multiply(
        ttnn_reshape_350,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_multiply_92,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_7 = ttnn.softmax(
        ttnn_typecast_15,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_softmax_7,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_351 = ttnn.reshape(
        ttnn_typecast_16,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_68 = ttnn.matmul(
        ttnn_reshape_351,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_352 = ttnn.reshape(
        ttnn_matmul_68,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_352


def CLIPEncoderLayer_8_0(input_0, input_1, input_2):
    ttnn_add_131 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_38 = ttnn.mean(
        ttnn_add_131,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_19 = ttnn.neg(
        ttnn_mean_38,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_132 = ttnn.add(
        ttnn_add_131,
        ttnn_neg_19,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_93 = ttnn.multiply(
        ttnn_add_132,
        ttnn_add_132,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_39 = ttnn.mean(
        ttnn_multiply_93,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_133 = ttnn.add(
        ttnn_mean_39,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_19 = ttnn.rsqrt(
        ttnn_add_133,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_353 = ttnn.reshape(
        ttnn_rsqrt_19,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_131, ttnn_reshape_353, ttnn_add_132


def Linear_23_0(input_0, input_1, input_2):
    ttnn_linear_11 = ttnn.linear(
        input_0,
        input_1,
        bias=input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    return ttnn_linear_11


def Linear_99_0(input):
    ttnn_reshape_354 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_354


def Linear_79_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_33 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_355 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_33,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_69 = ttnn.matmul(
        ttnn_reshape_355,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_134 = ttnn.add(
        ttnn_matmul_69,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_356 = ttnn.reshape(
        ttnn_add_134,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_356


def CLIPAttention_114_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_70 = ttnn.matmul(
        input_3,
        input_2,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_357 = ttnn.reshape(
        ttnn_matmul_70,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_94 = ttnn.multiply(
        ttnn_reshape_357,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_multiply_94,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_8 = ttnn.softmax(
        ttnn_typecast_17,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_softmax_8,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_358 = ttnn.reshape(
        ttnn_typecast_18,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_71 = ttnn.matmul(
        ttnn_reshape_358,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_359 = ttnn.reshape(
        ttnn_matmul_71,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_359


def LayerNorm_16_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_95 = ttnn.multiply(
        input_0,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_96 = ttnn.multiply(
        ttnn_multiply_95,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_135 = ttnn.add(
        ttnn_multiply_96,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_135


def Linear_9_0(input):
    ttnn_reshape_360 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_360


def Linear_81_0(input):
    ttnn_reshape_361 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_361


def LayerNorm_46_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_97 = ttnn.multiply(
        input_1,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_98 = ttnn.multiply(
        ttnn_multiply_97,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_136 = ttnn.add(
        ttnn_multiply_98,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_136


def CLIPEncoderLayer_98_0(input_0, input_1, input_2):
    ttnn_add_137 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_40 = ttnn.mean(
        ttnn_add_137,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_20 = ttnn.neg(
        ttnn_mean_40,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_138 = ttnn.add(
        ttnn_add_137,
        ttnn_neg_20,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_99 = ttnn.multiply(
        ttnn_add_138,
        ttnn_add_138,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_41 = ttnn.mean(
        ttnn_multiply_99,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_139 = ttnn.add(
        ttnn_mean_41,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_20 = ttnn.rsqrt(
        ttnn_add_139,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_362 = ttnn.reshape(
        ttnn_rsqrt_20,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_138, ttnn_reshape_362, ttnn_add_137


def Linear_109_0(input_0, input_1, input_2):
    ttnn_matmul_72 = ttnn.matmul(
        input_0,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_140 = ttnn.add(
        ttnn_matmul_72,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_363 = ttnn.reshape(
        ttnn_add_140,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_363


def CLIPEncoderLayer_74_0(input_0, input_1, input_2):
    ttnn_add_141 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_42 = ttnn.mean(
        ttnn_add_141,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_21 = ttnn.neg(
        ttnn_mean_42,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_142 = ttnn.add(
        ttnn_add_141,
        ttnn_neg_21,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_100 = ttnn.multiply(
        ttnn_add_142,
        ttnn_add_142,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_43 = ttnn.mean(
        ttnn_multiply_100,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_143 = ttnn.add(
        ttnn_mean_43,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_21 = ttnn.rsqrt(
        ttnn_add_143,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_364 = ttnn.reshape(
        ttnn_rsqrt_21,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_364, ttnn_add_142, ttnn_add_141


def CLIPEncoderLayer_128_0(input_0, input_1, input_2):
    ttnn_add_144 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_44 = ttnn.mean(
        ttnn_add_144,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_22 = ttnn.neg(
        ttnn_mean_44,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_145 = ttnn.add(
        ttnn_add_144,
        ttnn_neg_22,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_101 = ttnn.multiply(
        ttnn_add_145,
        ttnn_add_145,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_45 = ttnn.mean(
        ttnn_multiply_101,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_146 = ttnn.add(
        ttnn_mean_45,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_22 = ttnn.rsqrt(
        ttnn_add_146,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_365 = ttnn.reshape(
        ttnn_rsqrt_22,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_145, ttnn_add_144, ttnn_reshape_365


def Linear_43_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_34 = ttnn.transformer.concatenate_heads(
        input_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_366 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_34,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_73 = ttnn.matmul(
        ttnn_reshape_366,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_147 = ttnn.add(
        ttnn_matmul_73,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_367 = ttnn.reshape(
        ttnn_add_147,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_367


def CLIPAttention_138_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_74 = ttnn.matmul(
        input_2,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_368 = ttnn.reshape(
        ttnn_matmul_74,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_102 = ttnn.multiply(
        ttnn_reshape_368,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_multiply_102,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_9 = ttnn.softmax(
        ttnn_typecast_19,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_softmax_9,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_369 = ttnn.reshape(
        ttnn_typecast_20,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_75 = ttnn.matmul(
        ttnn_reshape_369,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_370 = ttnn.reshape(
        ttnn_matmul_75,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_370


def Linear_85_0(input_0, input_1, input_2):
    ttnn_matmul_76 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_148 = ttnn.add(
        ttnn_matmul_76,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_371 = ttnn.reshape(
        ttnn_add_148,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_371


def CLIPAttention_6_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_77 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_372 = ttnn.reshape(
        ttnn_matmul_77,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_103 = ttnn.multiply(
        ttnn_reshape_372,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_21 = ttnn.typecast(
        ttnn_multiply_103,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_10 = ttnn.softmax(
        ttnn_typecast_21,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_softmax_10,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_373 = ttnn.reshape(
        ttnn_typecast_22,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_78 = ttnn.matmul(
        ttnn_reshape_373,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_374 = ttnn.reshape(
        ttnn_matmul_78,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_374


def Linear_51_0(input):
    ttnn_reshape_375 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_375


def Linear_31_0(input_0, input_1, input_2):
    ttnn_transformer_concatenate_heads_35 = ttnn.transformer.concatenate_heads(
        input_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_376 = ttnn.reshape(
        ttnn_transformer_concatenate_heads_35,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_79 = ttnn.matmul(
        ttnn_reshape_376,
        input_1,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_149 = ttnn.add(
        ttnn_matmul_79,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_377 = ttnn.reshape(
        ttnn_add_149,
        [2, 50, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_377


def Linear_15_0(input):
    ttnn_reshape_378 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_378


def CLIPAttention_66_0(input_0, input_1, input_2, input_3):
    ttnn_matmul_80 = ttnn.matmul(
        input_2,
        input_3,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_379 = ttnn.reshape(
        ttnn_matmul_80,
        [2, 12, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_104 = ttnn.multiply(
        ttnn_reshape_379,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_multiply_104,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_softmax_11 = ttnn.softmax(
        ttnn_typecast_23,
        3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_24 = ttnn.typecast(
        ttnn_softmax_11,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_380 = ttnn.reshape(
        ttnn_typecast_24,
        [24, 50, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_81 = ttnn.matmul(
        ttnn_reshape_380,
        input_1,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_reshape_381 = ttnn.reshape(
        ttnn_matmul_81,
        [2, 12, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_381


def Linear_111_0(input):
    ttnn_reshape_382 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_382


def LayerNorm_142_0(input_0, input_1, input_2, input_3):
    ttnn_multiply_105 = ttnn.multiply(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_106 = ttnn.multiply(
        ttnn_multiply_105,
        input_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_150 = ttnn.add(
        ttnn_multiply_106,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_150


def CLIPEncoderLayer_134_0(input_0, input_1, input_2):
    ttnn_add_151 = ttnn.add(
        input_1,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_46 = ttnn.mean(
        ttnn_add_151,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_23 = ttnn.neg(
        ttnn_mean_46,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_152 = ttnn.add(
        ttnn_add_151,
        ttnn_neg_23,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_107 = ttnn.multiply(
        ttnn_add_152,
        ttnn_add_152,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_47 = ttnn.mean(
        ttnn_multiply_107,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_153 = ttnn.add(
        ttnn_mean_47,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_23 = ttnn.rsqrt(
        ttnn_add_153,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_383 = ttnn.reshape(
        ttnn_rsqrt_23,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_152, ttnn_reshape_383, ttnn_add_151


def CLIPEncoderLayer_20_0(input_0, input_1, input_2):
    ttnn_add_154 = ttnn.add(
        input_2,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_48 = ttnn.mean(
        ttnn_add_154,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_24 = ttnn.neg(
        ttnn_mean_48,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_155 = ttnn.add(
        ttnn_add_154,
        ttnn_neg_24,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_108 = ttnn.multiply(
        ttnn_add_155,
        ttnn_add_155,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_49 = ttnn.mean(
        ttnn_multiply_108,
        [2],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_156 = ttnn.add(
        ttnn_mean_49,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_24 = ttnn.rsqrt(
        ttnn_add_156,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_384 = ttnn.reshape(
        ttnn_rsqrt_24,
        [100, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_154, ttnn_add_155, ttnn_reshape_384


def Linear_53_0(input_0, input_1, input_2, input_3, input_4, input_5, input_6):
    ttnn_matmul_82 = ttnn.matmul(
        input_1,
        input_3,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_83 = ttnn.matmul(
        input_1,
        input_5,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_matmul_84 = ttnn.matmul(
        input_1,
        input_2,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn_add_157 = ttnn.add(
        ttnn_matmul_83,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_158 = ttnn.add(
        ttnn_matmul_84,
        input_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_159 = ttnn.add(
        ttnn_matmul_82,
        input_4,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_385 = ttnn.reshape(
        ttnn_add_157,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_386 = ttnn.reshape(
        ttnn_add_158,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_387 = ttnn.reshape(
        ttnn_add_159,
        [2, 50, 12, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_86 = ttnn.permute(
        ttnn_reshape_386,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_87 = ttnn.permute(
        ttnn_reshape_387,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_permute_88 = ttnn.permute(
        ttnn_reshape_385,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn_reshape_388 = ttnn.reshape(
        ttnn_permute_86,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_389 = ttnn.reshape(
        ttnn_permute_87,
        [24, 64, 50],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_390 = ttnn.reshape(
        ttnn_permute_88,
        [24, 50, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_390, ttnn_reshape_389, ttnn_reshape_388


def QuickGELUActivation_36_0(input_0, input_1):
    ttnn_multiply_109 = ttnn.multiply(
        input_0,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_sigmoid_11 = ttnn.sigmoid(
        ttnn_multiply_109,
        vector_mode=4,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_110 = ttnn.multiply(
        input_0,
        ttnn_sigmoid_11,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_multiply_110


def CLIPVisionTransformer_147_0(input_0, input_1, input_2, input_3):
    ttnn_slice_0 = ttnn.slice(
        input_3,
        [0, 0, 0],
        [2, 1, 768],
        [1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_391 = ttnn.reshape(
        ttnn_slice_0,
        [2, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_50 = ttnn.mean(
        ttnn_reshape_391,
        [1],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_neg_25 = ttnn.neg(
        ttnn_mean_50,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_160 = ttnn.add(
        ttnn_reshape_391,
        ttnn_neg_25,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_111 = ttnn.multiply(
        ttnn_add_160,
        ttnn_add_160,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_mean_51 = ttnn.mean(
        ttnn_multiply_111,
        [1],
        True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_161 = ttnn.add(
        ttnn_mean_51,
        input_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_rsqrt_25 = ttnn.rsqrt(
        ttnn_add_161,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_112 = ttnn.multiply(
        ttnn_add_160,
        ttnn_rsqrt_25,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_113 = ttnn.multiply(
        ttnn_multiply_112,
        input_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_162 = ttnn.add(
        ttnn_multiply_113,
        input_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_162


def Linear_135_0(input):
    ttnn_reshape_392 = ttnn.reshape(
        input,
        [100, 768],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_reshape_392


def create_inputs_for__main():
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([512, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_1 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_2 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_3 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_4 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_5 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_6 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_7 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_8 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_9 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_10 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_11 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_12 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_13 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_14 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_15 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_16 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_17 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_18 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_19 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_20 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_21 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_22 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_23 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_24 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_25 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_26 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_27 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_28 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_29 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_30 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_31 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_32 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_33 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_34 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_35 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_36 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_37 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_38 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_39 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_40 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_41 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_42 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_43 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_44 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_45 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_46 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_47 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_48 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_49 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_50 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_51 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_52 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_53 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_54 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_55 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_56 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_57 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_58 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_59 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_60 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_61 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_62 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_63 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_64 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_65 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_66 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_67 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_68 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_69 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_70 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_71 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_72 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_73 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_74 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_75 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_76 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_77 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_78 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_79 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_80 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_81 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_82 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_83 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_84 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_85 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_86 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_87 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_88 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_89 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_90 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_91 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_92 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_93 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_94 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_95 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_96 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_97 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_98 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_99 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_100 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_101 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_102 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_103 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_104 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_105 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_106 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_107 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_108 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_109 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_110 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_111 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_112 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_113 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_114 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_115 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_116 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_117 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_118 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_119 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_120 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_121 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_122 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_123 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_124 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_125 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_126 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_127 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_128 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_129 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_130 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_131 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_132 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_133 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_134 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_135 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_136 = ttnn.ones(
        shape=ttnn.Shape([768, 3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_137 = ttnn.ones(
        shape=ttnn.Shape([3072]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_138 = ttnn.ones(
        shape=ttnn.Shape([3072, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_139 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_140 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_141 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_142 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_143 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_144 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_145 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_146 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_147 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_148 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_149 = ttnn.ones(
        shape=ttnn.Shape([1, 50]),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_150 = ttnn.ones(
        shape=ttnn.Shape([50, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_151 = ttnn.ones(
        shape=ttnn.Shape([768, 3, 32, 32]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=None,
    )
    ttnn_ones_152 = ttnn.ones(
        shape=ttnn.Shape([2, 3, 224, 224]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_153 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_154 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_155 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_156 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_157 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_158 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_159 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_160 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_161 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_162 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_163 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_164 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_165 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_166 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_167 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_168 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_169 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_170 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_171 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_172 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_173 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_174 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_175 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_176 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_177 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_178 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_179 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_180 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_181 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_182 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_183 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_184 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_185 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_186 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_187 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_188 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_189 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_190 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_191 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_192 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_193 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_194 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_195 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_196 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_197 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_198 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_199 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_200 = ttnn.ones(
        shape=ttnn.Shape([768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    ttnn_ones_201 = ttnn.ones(
        shape=ttnn.Shape([768, 768]),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
    )
    util_create_list_258 = [
        ttnn_ones_0,
        ttnn_ones_1,
        ttnn_ones_2,
        ttnn_ones_3,
        ttnn_ones_4,
        ttnn_ones_5,
        ttnn_ones_6,
        ttnn_ones_7,
        ttnn_ones_8,
        ttnn_ones_9,
        ttnn_ones_10,
        ttnn_ones_11,
        ttnn_ones_12,
        ttnn_ones_13,
        ttnn_ones_14,
        ttnn_ones_15,
        ttnn_ones_16,
        ttnn_ones_17,
        ttnn_ones_18,
        ttnn_ones_19,
        ttnn_ones_20,
        ttnn_ones_21,
        ttnn_ones_22,
        ttnn_ones_23,
        ttnn_ones_24,
        ttnn_ones_25,
        ttnn_ones_26,
        ttnn_ones_27,
        ttnn_ones_28,
        ttnn_ones_29,
        ttnn_ones_30,
        ttnn_ones_31,
        ttnn_ones_32,
        ttnn_ones_33,
        ttnn_ones_34,
        ttnn_ones_35,
        ttnn_ones_36,
        ttnn_ones_37,
        ttnn_ones_38,
        ttnn_ones_39,
        ttnn_ones_40,
        ttnn_ones_41,
        ttnn_ones_42,
        ttnn_ones_43,
        ttnn_ones_44,
        ttnn_ones_45,
        ttnn_ones_46,
        ttnn_ones_47,
        ttnn_ones_48,
        ttnn_ones_49,
        ttnn_ones_50,
        ttnn_ones_51,
        ttnn_ones_52,
        ttnn_ones_53,
        ttnn_ones_54,
        ttnn_ones_55,
        ttnn_ones_56,
        ttnn_ones_57,
        ttnn_ones_58,
        ttnn_ones_59,
        ttnn_ones_60,
        ttnn_ones_61,
        ttnn_ones_62,
        ttnn_ones_63,
        ttnn_ones_64,
        ttnn_ones_65,
        ttnn_ones_66,
        ttnn_ones_67,
        ttnn_ones_68,
        ttnn_ones_69,
        ttnn_ones_70,
        ttnn_ones_71,
        ttnn_ones_72,
        ttnn_ones_73,
        ttnn_ones_74,
        ttnn_ones_75,
        ttnn_ones_76,
        ttnn_ones_77,
        ttnn_ones_78,
        ttnn_ones_79,
        ttnn_ones_80,
        ttnn_ones_81,
        ttnn_ones_82,
        ttnn_ones_83,
        ttnn_ones_84,
        ttnn_ones_85,
        ttnn_ones_86,
        ttnn_ones_87,
        ttnn_ones_88,
        ttnn_ones_89,
        ttnn_ones_90,
        ttnn_ones_91,
        ttnn_ones_92,
        ttnn_ones_93,
        ttnn_ones_94,
        ttnn_ones_95,
        ttnn_ones_96,
        ttnn_ones_97,
        ttnn_ones_98,
        ttnn_ones_99,
        ttnn_ones_100,
        ttnn_ones_101,
        ttnn_ones_102,
        ttnn_ones_103,
        ttnn_ones_104,
        ttnn_ones_105,
        ttnn_ones_106,
        ttnn_ones_107,
        ttnn_ones_108,
        ttnn_ones_109,
        ttnn_ones_110,
        ttnn_ones_111,
        ttnn_ones_112,
        ttnn_ones_113,
        ttnn_ones_114,
        ttnn_ones_115,
        ttnn_ones_116,
        ttnn_ones_117,
        ttnn_ones_118,
        ttnn_ones_119,
        ttnn_ones_120,
        ttnn_ones_121,
        ttnn_ones_122,
        ttnn_ones_123,
        ttnn_ones_124,
        ttnn_ones_125,
        ttnn_ones_126,
        ttnn_ones_127,
        ttnn_ones_128,
        ttnn_ones_129,
        ttnn_ones_130,
        ttnn_ones_131,
        ttnn_ones_132,
        ttnn_ones_133,
        ttnn_ones_134,
        ttnn_ones_135,
        ttnn_ones_136,
        ttnn_ones_137,
        ttnn_ones_138,
        ttnn_ones_139,
        ttnn_ones_140,
        ttnn_ones_141,
        ttnn_ones_142,
        ttnn_ones_143,
        ttnn_ones_144,
        ttnn_ones_145,
        ttnn_ones_146,
        ttnn_ones_147,
        ttnn_ones_148,
        ttnn_ones_149,
        ttnn_ones_150,
        ttnn_ones_151,
        ttnn_ones_152,
        ttnn_ones_153,
        ttnn_ones_154,
        ttnn_ones_155,
        ttnn_ones_156,
        ttnn_ones_157,
        ttnn_ones_158,
        ttnn_ones_159,
        ttnn_ones_160,
        ttnn_ones_161,
        ttnn_ones_162,
        ttnn_ones_163,
        ttnn_ones_164,
        ttnn_ones_165,
        ttnn_ones_166,
        ttnn_ones_167,
        ttnn_ones_168,
        ttnn_ones_169,
        ttnn_ones_170,
        ttnn_ones_171,
        ttnn_ones_172,
        ttnn_ones_173,
        ttnn_ones_174,
        ttnn_ones_175,
        ttnn_ones_176,
        ttnn_ones_177,
        ttnn_ones_178,
        ttnn_ones_179,
        ttnn_ones_180,
        ttnn_ones_181,
        ttnn_ones_182,
        ttnn_ones_183,
        ttnn_ones_184,
        ttnn_ones_185,
        ttnn_ones_186,
        ttnn_ones_187,
        ttnn_ones_188,
        ttnn_ones_189,
        ttnn_ones_190,
        ttnn_ones_191,
        ttnn_ones_192,
        ttnn_ones_193,
        ttnn_ones_194,
        ttnn_ones_195,
        ttnn_ones_196,
        ttnn_ones_197,
        ttnn_ones_198,
        ttnn_ones_199,
        ttnn_ones_200,
        ttnn_ones_201,
    ]
    return util_create_list_258


def main():
    create_inputs_for__main_0 = create_inputs_for__main()
    _main_0 = _main(create_inputs_for__main_0)
    const0_0 = 0
    return const0_0


if __name__ == "__main__":
    main()
