import ttnn
import tests.ttnn.unit_tests.operations.eltwise.utils as utils
from loguru import logger

import torch.nn as nn
import torch

from scipy.stats import pearsonr
from transformers import AutoModelForImageClassification, AutoImageProcessor
from datasets import load_dataset


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([0]),
        fill_value=0.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def main_const_eval_1(input):
    input_0 = input[0]
    ttnn_reshape_0 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_1 = [ttnn_reshape_0]
    return util_create_list_1


def main_const_eval_2(input):
    input_0 = input[0]
    ttnn_reshape_1 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_2 = [ttnn_reshape_1]
    return util_create_list_2


def main_const_eval_3(input):
    input_0 = input[0]
    ttnn_reshape_2 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_3 = [ttnn_reshape_2]
    return util_create_list_3


def main_const_eval_4(input):
    input_0 = input[0]
    ttnn_reshape_3 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_4 = [ttnn_reshape_3]
    return util_create_list_4


def main_const_eval_5(input):
    input_0 = input[0]
    ttnn_reshape_4 = ttnn.reshape(
        input_0,
        [1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_5 = [ttnn_reshape_4]
    return util_create_list_5


def main_const_eval_6(input):
    input_0 = input[0]
    ttnn_reshape_5 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_6 = [ttnn_reshape_5]
    return util_create_list_6


def main_const_eval_7(input):
    input_0 = input[0]
    ttnn_reshape_6 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_7 = [ttnn_reshape_6]
    return util_create_list_7


def main_const_eval_8(input):
    input_0 = input[0]
    ttnn_reshape_7 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_8 = [ttnn_reshape_7]
    return util_create_list_8


def main_const_eval_9(input):
    input_0 = input[0]
    ttnn_reshape_8 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_9 = [ttnn_reshape_8]
    return util_create_list_9


def main_const_eval_10(input):
    input_0 = input[0]
    ttnn_reshape_9 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_10 = [ttnn_reshape_9]
    return util_create_list_10


def main_const_eval_11(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_0 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_0 = ttnn.to_device(
        ttnn_to_layout_0,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    util_create_list_11 = [ttnn_to_device_0]
    return util_create_list_11


def main_const_eval_12(input):
    input_0 = input[0]
    ttnn_reshape_10 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_12 = [ttnn_reshape_10]
    return util_create_list_12


def main_const_eval_13(input):
    input_0 = input[0]
    ttnn_reshape_11 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_13 = [ttnn_reshape_11]
    return util_create_list_13


def main_const_eval_14(input):
    input_0 = input[0]
    ttnn_reshape_12 = ttnn.reshape(
        input_0,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_14 = [ttnn_reshape_12]
    return util_create_list_14


def main_const_eval_15(input):
    input_0 = input[0]
    ttnn_reshape_13 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_15 = [ttnn_reshape_13]
    return util_create_list_15


def main_const_eval_16(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_1 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_1 = ttnn.to_device(
        ttnn_to_layout_1,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    util_create_list_16 = [ttnn_to_device_1]
    return util_create_list_16


def main_const_eval_17(input):
    input_0 = input[0]
    ttnn_reshape_14 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_17 = [ttnn_reshape_14]
    return util_create_list_17


def main_const_eval_18(input):
    input_0 = input[0]
    ttnn_reshape_15 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_18 = [ttnn_reshape_15]
    return util_create_list_18


def main_const_eval_19(input):
    input_0 = input[0]
    ttnn_reshape_16 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_19 = [ttnn_reshape_16]
    return util_create_list_19


def main_const_eval_20(input):
    input_0 = input[0]
    ttnn_reshape_17 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_20 = [ttnn_reshape_17]
    return util_create_list_20


def main_const_eval_21(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_2 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_2 = ttnn.to_device(
        ttnn_to_layout_2,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    util_create_list_21 = [ttnn_to_device_2]
    return util_create_list_21


def main_const_eval_22(input):
    input_0 = input[0]
    ttnn_reshape_18 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_22 = [ttnn_reshape_18]
    return util_create_list_22


def main_const_eval_23(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_3 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_3 = ttnn.to_device(
        ttnn_to_layout_3,
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_3, False)
    util_create_list_23 = [ttnn_to_device_3]
    return util_create_list_23


def main_const_eval_24(input):
    input_0 = input[0]
    ttnn_reshape_19 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_24 = [ttnn_reshape_19]
    return util_create_list_24


def main_const_eval_25(input):
    input_0 = input[0]
    ttnn_reshape_20 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_25 = [ttnn_reshape_20]
    return util_create_list_25


def main_const_eval_26(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_4 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_4 = ttnn.to_device(
        ttnn_to_layout_4,
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    util_create_list_26 = [ttnn_to_device_4]
    return util_create_list_26


def main_const_eval_27(input):
    input_0 = input[0]
    ttnn_reshape_21 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_27 = [ttnn_reshape_21]
    return util_create_list_27


def main_const_eval_28(input):
    input_0 = input[0]
    ttnn_permute_0 = ttnn.permute(
        input_0,
        [1, 0],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    util_create_list_28 = [ttnn_permute_0]
    return util_create_list_28


def main_const_eval_29(input):
    input_0 = input[0]
    ttnn_reshape_22 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_29 = [ttnn_reshape_22]
    return util_create_list_29


def main_const_eval_30(input):
    input_0 = input[0]
    ttnn_reshape_23 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_30 = [ttnn_reshape_23]
    return util_create_list_30


def main_const_eval_31(input):
    input_0 = input[0]
    ttnn_reshape_24 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_31 = [ttnn_reshape_24]
    return util_create_list_31


def main_const_eval_32(input):
    input_0 = input[0]
    ttnn_reshape_25 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_32 = [ttnn_reshape_25]
    return util_create_list_32


def main_const_eval_33(input):
    input_0 = input[0]
    ttnn_reshape_26 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_33 = [ttnn_reshape_26]
    return util_create_list_33


def main_const_eval_34(input):
    input_0 = input[0]
    ttnn_reshape_27 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_34 = [ttnn_reshape_27]
    return util_create_list_34


def main_const_eval_35(input):
    input_0 = input[0]
    ttnn_reshape_28 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_35 = [ttnn_reshape_28]
    return util_create_list_35


def main_const_eval_36(input):
    input_0 = input[0]
    ttnn_reshape_29 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_36 = [ttnn_reshape_29]
    return util_create_list_36


def main_const_eval_37(input):
    input_0 = input[0]
    ttnn_reshape_30 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_37 = [ttnn_reshape_30]
    return util_create_list_37


def main_const_eval_38(input):
    input_0 = input[0]
    ttnn_reshape_31 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_38 = [ttnn_reshape_31]
    return util_create_list_38


def main_const_eval_39(input):
    input_0 = input[0]
    ttnn_reshape_32 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_39 = [ttnn_reshape_32]
    return util_create_list_39


def main_const_eval_40(input):
    input_0 = input[0]
    ttnn_reshape_33 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_40 = [ttnn_reshape_33]
    return util_create_list_40


def main_const_eval_41(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_5 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_5 = ttnn.to_device(
        ttnn_to_layout_5,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    util_create_list_41 = [ttnn_to_device_5]
    return util_create_list_41


def main_const_eval_42(input):
    input_0 = input[0]
    ttnn_reshape_34 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_42 = [ttnn_reshape_34]
    return util_create_list_42


def main_const_eval_43(input):
    input_0 = input[0]
    ttnn_reshape_35 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_43 = [ttnn_reshape_35]
    return util_create_list_43


def main_const_eval_44(input):
    input_0 = input[0]
    ttnn_reshape_36 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_44 = [ttnn_reshape_36]
    return util_create_list_44


def main_const_eval_45(input):
    input_0 = input[0]
    ttnn_reshape_37 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_45 = [ttnn_reshape_37]
    return util_create_list_45


def main_const_eval_46(input):
    input_0 = input[0]
    ttnn_reshape_38 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_46 = [ttnn_reshape_38]
    return util_create_list_46


def main_const_eval_47(input):
    input_0 = input[0]
    ttnn_reshape_39 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_47 = [ttnn_reshape_39]
    return util_create_list_47


def main_const_eval_48(input):
    input_0 = input[0]
    ttnn_reshape_40 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_48 = [ttnn_reshape_40]
    return util_create_list_48


def main_const_eval_49(input):
    input_0 = input[0]
    ttnn_reshape_41 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_49 = [ttnn_reshape_41]
    return util_create_list_49


def main_const_eval_50(input):
    input_0 = input[0]
    ttnn_reshape_42 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_50 = [ttnn_reshape_42]
    return util_create_list_50


def main_const_eval_51(input):
    input_0 = input[0]
    ttnn_reshape_43 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_51 = [ttnn_reshape_43]
    return util_create_list_51


def main_const_eval_52(input):
    input_0 = input[0]
    ttnn_reshape_44 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_52 = [ttnn_reshape_44]
    return util_create_list_52


def main_const_eval_53(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_6 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_6 = ttnn.to_device(
        ttnn_to_layout_6,
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    util_create_list_53 = [ttnn_to_device_6]
    return util_create_list_53


def main_const_eval_54(input):
    input_0 = input[0]
    ttnn_reshape_45 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_54 = [ttnn_reshape_45]
    return util_create_list_54


def main_const_eval_55(input):
    input_0 = input[0]
    ttnn_reshape_46 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_55 = [ttnn_reshape_46]
    return util_create_list_55


def main_const_eval_56(input):
    input_0 = input[0]
    ttnn_reshape_47 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_56 = [ttnn_reshape_47]
    return util_create_list_56


def main_const_eval_57(input):
    input_0 = input[0]
    ttnn_reshape_48 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_57 = [ttnn_reshape_48]
    return util_create_list_57


def main_const_eval_58(input):
    input_0 = input[0]
    ttnn_reshape_49 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_58 = [ttnn_reshape_49]
    return util_create_list_58


def main_const_eval_59(input):
    input_0 = input[0]
    ttnn_reshape_50 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_59 = [ttnn_reshape_50]
    return util_create_list_59


def main_const_eval_60(input):
    input_0 = input[0]
    ttnn_reshape_51 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_60 = [ttnn_reshape_51]
    return util_create_list_60


def main_const_eval_61(input):
    input_0 = input[0]
    ttnn_reshape_52 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_61 = [ttnn_reshape_52]
    return util_create_list_61


def main_const_eval_62(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_7 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_7 = ttnn.to_device(
        ttnn_to_layout_7,
        device=utils_DeviceGetter_get_device_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    util_create_list_62 = [ttnn_to_device_7]
    return util_create_list_62


def main_const_eval_63(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_9 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_8 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_8 = ttnn.to_device(
        ttnn_to_layout_8,
        device=utils_DeviceGetter_get_device_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_8, False)
    util_create_list_63 = [ttnn_to_device_8]
    return util_create_list_63


def main_const_eval_64(input):
    input_0 = input[0]
    ttnn_reshape_53 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_64 = [ttnn_reshape_53]
    return util_create_list_64


def main_const_eval_65(input):
    input_0 = input[0]
    ttnn_reshape_54 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_65 = [ttnn_reshape_54]
    return util_create_list_65


def main_const_eval_66(input):
    input_0 = input[0]
    ttnn_reshape_55 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_66 = [ttnn_reshape_55]
    return util_create_list_66


def main_const_eval_67(input):
    input_0 = input[0]
    ttnn_reshape_56 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_67 = [ttnn_reshape_56]
    return util_create_list_67


def main_const_eval_68(input):
    input_0 = input[0]
    ttnn_reshape_57 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_68 = [ttnn_reshape_57]
    return util_create_list_68


def main_const_eval_69(input):
    input_0 = input[0]
    ttnn_reshape_58 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_69 = [ttnn_reshape_58]
    return util_create_list_69


def main_const_eval_70():
    utils_DeviceGetter_get_device_10 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_70 = [ttnn_full_1]
    return util_create_list_70


def main_const_eval_71(input):
    input_0 = input[0]
    ttnn_reshape_59 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_71 = [ttnn_reshape_59]
    return util_create_list_71


def main_const_eval_72(input):
    input_0 = input[0]
    ttnn_reshape_60 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_72 = [ttnn_reshape_60]
    return util_create_list_72


def main_const_eval_73(input):
    input_0 = input[0]
    ttnn_reshape_61 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_73 = [ttnn_reshape_61]
    return util_create_list_73


def main_const_eval_74(input):
    input_0 = input[0]
    ttnn_reshape_62 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_74 = [ttnn_reshape_62]
    return util_create_list_74


def main_const_eval_75(input):
    input_0 = input[0]
    ttnn_reshape_63 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_75 = [ttnn_reshape_63]
    return util_create_list_75


def main_const_eval_76(input):
    input_0 = input[0]
    ttnn_reshape_64 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_76 = [ttnn_reshape_64]
    return util_create_list_76


def main_const_eval_77(input):
    input_0 = input[0]
    ttnn_reshape_65 = ttnn.reshape(
        input_0,
        [1, 9, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_77 = [ttnn_reshape_65]
    return util_create_list_77


def main_const_eval_78(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_11 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_9 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_9 = ttnn.to_device(
        ttnn_to_layout_9,
        device=utils_DeviceGetter_get_device_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    util_create_list_78 = [ttnn_to_device_9]
    return util_create_list_78


def main_const_eval_79(input):
    input_0 = input[0]
    ttnn_reshape_66 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_79 = [ttnn_reshape_66]
    return util_create_list_79


def main_const_eval_80(input):
    input_0 = input[0]
    ttnn_reshape_67 = ttnn.reshape(
        input_0,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_80 = [ttnn_reshape_67]
    return util_create_list_80


def main_const_eval_81(input):
    input_0 = input[0]
    ttnn_reshape_68 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_81 = [ttnn_reshape_68]
    return util_create_list_81


def main_const_eval_82(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_12 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_10 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_10 = ttnn.to_device(
        ttnn_to_layout_10,
        device=utils_DeviceGetter_get_device_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    util_create_list_82 = [ttnn_to_device_10]
    return util_create_list_82


def main_const_eval_83(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_13 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_11 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_11 = ttnn.to_device(
        ttnn_to_layout_11,
        device=utils_DeviceGetter_get_device_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_11, False)
    util_create_list_83 = [ttnn_to_device_11]
    return util_create_list_83


def main_const_eval_84(input):
    input_0 = input[0]
    ttnn_reshape_69 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_84 = [ttnn_reshape_69]
    return util_create_list_84


def main_const_eval_85(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_14 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_12 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_12 = ttnn.to_device(
        ttnn_to_layout_12,
        device=utils_DeviceGetter_get_device_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_12, False)
    util_create_list_85 = [ttnn_to_device_12]
    return util_create_list_85


def main_const_eval_86(input):
    input_0 = input[0]
    ttnn_reshape_70 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_86 = [ttnn_reshape_70]
    return util_create_list_86


def main_const_eval_87(input):
    input_0 = input[0]
    ttnn_reshape_71 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_87 = [ttnn_reshape_71]
    return util_create_list_87


def main_const_eval_88(input):
    input_0 = input[0]
    ttnn_reshape_72 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_88 = [ttnn_reshape_72]
    return util_create_list_88


def main_const_eval_89(input):
    input_0 = input[0]
    ttnn_reshape_73 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_89 = [ttnn_reshape_73]
    return util_create_list_89


def main_const_eval_90(input):
    input_0 = input[0]
    ttnn_reshape_74 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_90 = [ttnn_reshape_74]
    return util_create_list_90


def main_const_eval_91(input):
    input_0 = input[0]
    ttnn_reshape_75 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_91 = [ttnn_reshape_75]
    return util_create_list_91


def main_const_eval_92(input):
    input_0 = input[0]
    ttnn_reshape_76 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_92 = [ttnn_reshape_76]
    return util_create_list_92


def main_const_eval_93(input):
    input_0 = input[0]
    ttnn_reshape_77 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_93 = [ttnn_reshape_77]
    return util_create_list_93


def main_const_eval_94(input):
    input_0 = input[0]
    ttnn_reshape_78 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_94 = [ttnn_reshape_78]
    return util_create_list_94


def main_const_eval_95(input):
    input_0 = input[0]
    ttnn_reshape_79 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_95 = [ttnn_reshape_79]
    return util_create_list_95


def main_const_eval_96(input):
    input_0 = input[0]
    ttnn_reshape_80 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_96 = [ttnn_reshape_80]
    return util_create_list_96


def main_const_eval_97(input):
    input_0 = input[0]
    ttnn_reshape_81 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_97 = [ttnn_reshape_81]
    return util_create_list_97


def main_const_eval_98(input):
    input_0 = input[0]
    ttnn_reshape_82 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_98 = [ttnn_reshape_82]
    return util_create_list_98


def main_const_eval_99(input):
    input_0 = input[0]
    ttnn_reshape_83 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_99 = [ttnn_reshape_83]
    return util_create_list_99


def main_const_eval_100(input):
    input_0 = input[0]
    ttnn_reshape_84 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_100 = [ttnn_reshape_84]
    return util_create_list_100


def main_const_eval_101(input):
    input_0 = input[0]
    ttnn_reshape_85 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_101 = [ttnn_reshape_85]
    return util_create_list_101


def main_const_eval_102(input):
    input_0 = input[0]
    ttnn_reshape_86 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_102 = [ttnn_reshape_86]
    return util_create_list_102


def main_const_eval_103(input):
    input_0 = input[0]
    ttnn_reshape_87 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_103 = [ttnn_reshape_87]
    return util_create_list_103


def main_const_eval_104(input):
    input_0 = input[0]
    ttnn_reshape_88 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_104 = [ttnn_reshape_88]
    return util_create_list_104


def main_const_eval_105(input):
    input_0 = input[0]
    ttnn_reshape_89 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_105 = [ttnn_reshape_89]
    return util_create_list_105


def main_const_eval_106(input):
    input_0 = input[0]
    ttnn_reshape_90 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_106 = [ttnn_reshape_90]
    return util_create_list_106


def main_const_eval_107(input):
    input_0 = input[0]
    ttnn_reshape_91 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_107 = [ttnn_reshape_91]
    return util_create_list_107


def main_const_eval_108(input):
    input_0 = input[0]
    ttnn_reshape_92 = ttnn.reshape(
        input_0,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_108 = [ttnn_reshape_92]
    return util_create_list_108


def main_const_eval_109(input):
    input_0 = input[0]
    ttnn_reshape_93 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_109 = [ttnn_reshape_93]
    return util_create_list_109


def main_const_eval_110(input):
    input_0 = input[0]
    ttnn_reshape_94 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_110 = [ttnn_reshape_94]
    return util_create_list_110


def main_const_eval_111(input):
    input_0 = input[0]
    ttnn_reshape_95 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_111 = [ttnn_reshape_95]
    return util_create_list_111


def main_const_eval_112(input):
    input_0 = input[0]
    ttnn_reshape_96 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_112 = [ttnn_reshape_96]
    return util_create_list_112


def main_const_eval_113(input):
    input_0 = input[0]
    ttnn_reshape_97 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_113 = [ttnn_reshape_97]
    return util_create_list_113


def main_const_eval_114(input):
    input_0 = input[0]
    ttnn_reshape_98 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_114 = [ttnn_reshape_98]
    return util_create_list_114


def main_const_eval_115(input):
    input_0 = input[0]
    ttnn_reshape_99 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_115 = [ttnn_reshape_99]
    return util_create_list_115


def main_const_eval_116(input):
    input_0 = input[0]
    ttnn_reshape_100 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_116 = [ttnn_reshape_100]
    return util_create_list_116


def main_const_eval_117(input):
    input_0 = input[0]
    ttnn_reshape_101 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_117 = [ttnn_reshape_101]
    return util_create_list_117


def main_const_eval_118(input):
    input_0 = input[0]
    ttnn_reshape_102 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_118 = [ttnn_reshape_102]
    return util_create_list_118


def main_const_eval_119(input):
    input_0 = input[0]
    ttnn_reshape_103 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_119 = [ttnn_reshape_103]
    return util_create_list_119


def main_const_eval_120(input):
    input_0 = input[0]
    ttnn_reshape_104 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_120 = [ttnn_reshape_104]
    return util_create_list_120


def main_const_eval_121(input):
    input_0 = input[0]
    ttnn_reshape_105 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_121 = [ttnn_reshape_105]
    return util_create_list_121


def main_const_eval_122(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_15 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_13 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_13 = ttnn.to_device(
        ttnn_to_layout_13,
        device=utils_DeviceGetter_get_device_15,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_13, False)
    util_create_list_122 = [ttnn_to_device_13]
    return util_create_list_122


def main_const_eval_123(input):
    input_0 = input[0]
    ttnn_reshape_106 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_123 = [ttnn_reshape_106]
    return util_create_list_123


def main_const_eval_124(input):
    input_0 = input[0]
    ttnn_reshape_107 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_124 = [ttnn_reshape_107]
    return util_create_list_124


def main_const_eval_125(input):
    input_0 = input[0]
    ttnn_reshape_108 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_125 = [ttnn_reshape_108]
    return util_create_list_125


def main_const_eval_126(input):
    input_0 = input[0]
    ttnn_reshape_109 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_126 = [ttnn_reshape_109]
    return util_create_list_126


def main_const_eval_127(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_16 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_14 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_14 = ttnn.to_device(
        ttnn_to_layout_14,
        device=utils_DeviceGetter_get_device_16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_14, False)
    util_create_list_127 = [ttnn_to_device_14]
    return util_create_list_127


def main_const_eval_128(input):
    input_0 = input[0]
    ttnn_reshape_110 = ttnn.reshape(
        input_0,
        [1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_128 = [ttnn_reshape_110]
    return util_create_list_128


def main_const_eval_129(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_17 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_15 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_15 = ttnn.to_device(
        ttnn_to_layout_15,
        device=utils_DeviceGetter_get_device_17,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    util_create_list_129 = [ttnn_to_device_15]
    return util_create_list_129


def main_const_eval_130(input):
    input_0 = input[0]
    ttnn_reshape_111 = ttnn.reshape(
        input_0,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_130 = [ttnn_reshape_111]
    return util_create_list_130


def main_const_eval_131(input):
    input_0 = input[0]
    ttnn_reshape_112 = ttnn.reshape(
        input_0,
        [1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_131 = [ttnn_reshape_112]
    return util_create_list_131


def main_const_eval_132(input):
    input_0 = input[0]
    ttnn_reshape_113 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_132 = [ttnn_reshape_113]
    return util_create_list_132


def main_const_eval_133(input):
    input_0 = input[0]
    ttnn_reshape_114 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_133 = [ttnn_reshape_114]
    return util_create_list_133


def main_const_eval_134(input):
    input_0 = input[0]
    ttnn_reshape_115 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_134 = [ttnn_reshape_115]
    return util_create_list_134


def main_const_eval_135(input):
    input_0 = input[0]
    ttnn_reshape_116 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_135 = [ttnn_reshape_116]
    return util_create_list_135


def main_const_eval_136(input):
    input_0 = input[0]
    ttnn_reshape_117 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_136 = [ttnn_reshape_117]
    return util_create_list_136


def main_const_eval_137(input):
    input_0 = input[0]
    ttnn_reshape_118 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_137 = [ttnn_reshape_118]
    return util_create_list_137


def main_const_eval_138(input):
    input_0 = input[0]
    ttnn_reshape_119 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_138 = [ttnn_reshape_119]
    return util_create_list_138


def main_const_eval_139(input):
    input_0 = input[0]
    ttnn_reshape_120 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_139 = [ttnn_reshape_120]
    return util_create_list_139


def main_const_eval_140(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_18 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_16 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_16 = ttnn.to_device(
        ttnn_to_layout_16,
        device=utils_DeviceGetter_get_device_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_16, False)
    util_create_list_140 = [ttnn_to_device_16]
    return util_create_list_140


def main_const_eval_141(input):
    input_0 = input[0]
    ttnn_reshape_121 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_141 = [ttnn_reshape_121]
    return util_create_list_141


def main_const_eval_142(input):
    input_0 = input[0]
    ttnn_reshape_122 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_142 = [ttnn_reshape_122]
    return util_create_list_142


def main_const_eval_143(input):
    input_0 = input[0]
    ttnn_reshape_123 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_143 = [ttnn_reshape_123]
    return util_create_list_143


def main_const_eval_144(input):
    input_0 = input[0]
    ttnn_reshape_124 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_144 = [ttnn_reshape_124]
    return util_create_list_144


def main_const_eval_145(input):
    input_0 = input[0]
    ttnn_reshape_125 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_145 = [ttnn_reshape_125]
    return util_create_list_145


def main_const_eval_146(input):
    input_0 = input[0]
    ttnn_reshape_126 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_146 = [ttnn_reshape_126]
    return util_create_list_146


def main_const_eval_147(input):
    input_0 = input[0]
    ttnn_reshape_127 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_147 = [ttnn_reshape_127]
    return util_create_list_147


def main_const_eval_148(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_19 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_17 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_17 = ttnn.to_device(
        ttnn_to_layout_17,
        device=utils_DeviceGetter_get_device_19,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_17, False)
    util_create_list_148 = [ttnn_to_device_17]
    return util_create_list_148


def main_const_eval_149(input):
    input_0 = input[0]
    ttnn_reshape_128 = ttnn.reshape(
        input_0,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_149 = [ttnn_reshape_128]
    return util_create_list_149


def main_const_eval_150(input):
    input_0 = input[0]
    ttnn_reshape_129 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_150 = [ttnn_reshape_129]
    return util_create_list_150


def main_const_eval_151(input):
    input_0 = input[0]
    ttnn_reshape_130 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_151 = [ttnn_reshape_130]
    return util_create_list_151


def main_const_eval_152(input):
    input_0 = input[0]
    ttnn_reshape_131 = ttnn.reshape(
        input_0,
        [1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_152 = [ttnn_reshape_131]
    return util_create_list_152


def main_const_eval_153(input):
    input_0 = input[0]
    ttnn_reshape_132 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_153 = [ttnn_reshape_132]
    return util_create_list_153


def main_const_eval_154(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_20 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_18 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_18 = ttnn.to_device(
        ttnn_to_layout_18,
        device=utils_DeviceGetter_get_device_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    util_create_list_154 = [ttnn_to_device_18]
    return util_create_list_154


def main_const_eval_155(input):
    input_0 = input[0]
    ttnn_reshape_133 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_155 = [ttnn_reshape_133]
    return util_create_list_155


def main_const_eval_156(input):
    input_0 = input[0]
    ttnn_reshape_134 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_156 = [ttnn_reshape_134]
    return util_create_list_156


def main_const_eval_157(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_21 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_19 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_to_layout_19,
        device=utils_DeviceGetter_get_device_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_19, False)
    util_create_list_157 = [ttnn_to_device_19]
    return util_create_list_157


def main_const_eval_158(input):
    input_0 = input[0]
    ttnn_reshape_135 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_158 = [ttnn_reshape_135]
    return util_create_list_158


def main_const_eval_159(input):
    input_0 = input[0]
    ttnn_reshape_136 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_159 = [ttnn_reshape_136]
    return util_create_list_159


def main_const_eval_160(input):
    input_0 = input[0]
    ttnn_reshape_137 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_160 = [ttnn_reshape_137]
    return util_create_list_160


def main_const_eval_161(input):
    input_0 = input[0]
    ttnn_reshape_138 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_161 = [ttnn_reshape_138]
    return util_create_list_161


def main_const_eval_162(input):
    input_0 = input[0]
    ttnn_reshape_139 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_162 = [ttnn_reshape_139]
    return util_create_list_162


def main_const_eval_163(input):
    input_0 = input[0]
    ttnn_reshape_140 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_163 = [ttnn_reshape_140]
    return util_create_list_163


def main_const_eval_164(input):
    input_0 = input[0]
    ttnn_reshape_141 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_164 = [ttnn_reshape_141]
    return util_create_list_164


def main_const_eval_165(input):
    input_0 = input[0]
    ttnn_reshape_142 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_165 = [ttnn_reshape_142]
    return util_create_list_165


def main_const_eval_166(input):
    input_0 = input[0]
    ttnn_reshape_143 = ttnn.reshape(
        input_0,
        [1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_166 = [ttnn_reshape_143]
    return util_create_list_166


def main_const_eval_167(input):
    input_0 = input[0]
    ttnn_reshape_144 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_167 = [ttnn_reshape_144]
    return util_create_list_167


def main_const_eval_168(input):
    input_0 = input[0]
    ttnn_reshape_145 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_168 = [ttnn_reshape_145]
    return util_create_list_168


def main_const_eval_169(input):
    input_0 = input[0]
    ttnn_reshape_146 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_169 = [ttnn_reshape_146]
    return util_create_list_169


def main_const_eval_170(input):
    input_0 = input[0]
    ttnn_reshape_147 = ttnn.reshape(
        input_0,
        [1, 1, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_170 = [ttnn_reshape_147]
    return util_create_list_170


def main_const_eval_171(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_22 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_20 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_20 = ttnn.to_device(
        ttnn_to_layout_20,
        device=utils_DeviceGetter_get_device_22,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_20, False)
    util_create_list_171 = [ttnn_to_device_20]
    return util_create_list_171


def main_const_eval_172(input):
    input_0 = input[0]
    ttnn_reshape_148 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_172 = [ttnn_reshape_148]
    return util_create_list_172


def main_const_eval_173(input):
    input_0 = input[0]
    ttnn_reshape_149 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_173 = [ttnn_reshape_149]
    return util_create_list_173


def main_const_eval_174(input):
    input_0 = input[0]
    ttnn_reshape_150 = ttnn.reshape(
        input_0,
        [1, 1, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_174 = [ttnn_reshape_150]
    return util_create_list_174


def main_const_eval_175(input):
    input_0 = input[0]
    ttnn_reshape_151 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_175 = [ttnn_reshape_151]
    return util_create_list_175


def main_const_eval_176(input):
    input_0 = input[0]
    ttnn_reshape_152 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_176 = [ttnn_reshape_152]
    return util_create_list_176


def main_const_eval_177(input):
    input_0 = input[0]
    ttnn_reshape_153 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_177 = [ttnn_reshape_153]
    return util_create_list_177


def main_const_eval_178(input):
    input_0 = input[0]
    ttnn_reshape_154 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_178 = [ttnn_reshape_154]
    return util_create_list_178


def main_const_eval_179(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_23 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_21 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_to_layout_21,
        device=utils_DeviceGetter_get_device_23,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_21, False)
    util_create_list_179 = [ttnn_to_device_21]
    return util_create_list_179


def main_const_eval_180(input):
    input_0 = input[0]
    ttnn_reshape_155 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_180 = [ttnn_reshape_155]
    return util_create_list_180


def main_const_eval_181(input):
    input_0 = input[0]
    ttnn_reshape_156 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_181 = [ttnn_reshape_156]
    return util_create_list_181


def main_const_eval_182(input):
    input_0 = input[0]
    ttnn_reshape_157 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_182 = [ttnn_reshape_157]
    return util_create_list_182


def main_const_eval_183(input):
    input_0 = input[0]
    ttnn_reshape_158 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_183 = [ttnn_reshape_158]
    return util_create_list_183


def main_const_eval_184(input):
    input_0 = input[0]
    ttnn_reshape_159 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_184 = [ttnn_reshape_159]
    return util_create_list_184


def main_const_eval_185(input):
    input_0 = input[0]
    ttnn_reshape_160 = ttnn.reshape(
        input_0,
        [1, 1, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_185 = [ttnn_reshape_160]
    return util_create_list_185


def main_const_eval_186(input):
    input_0 = input[0]
    ttnn_reshape_161 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_186 = [ttnn_reshape_161]
    return util_create_list_186


def main_const_eval_187(input):
    input_0 = input[0]
    ttnn_reshape_162 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_187 = [ttnn_reshape_162]
    return util_create_list_187


def main_const_eval_188(input):
    input_0 = input[0]
    ttnn_reshape_163 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_188 = [ttnn_reshape_163]
    return util_create_list_188


def main_const_eval_189(input):
    input_0 = input[0]
    ttnn_reshape_164 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_189 = [ttnn_reshape_164]
    return util_create_list_189


def main_const_eval_190(input):
    input_0 = input[0]
    ttnn_reshape_165 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_190 = [ttnn_reshape_165]
    return util_create_list_190


def main_const_eval_191(input):
    input_0 = input[0]
    ttnn_reshape_166 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_191 = [ttnn_reshape_166]
    return util_create_list_191


def main_const_eval_192(input):
    input_0 = input[0]
    ttnn_reshape_167 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_192 = [ttnn_reshape_167]
    return util_create_list_192


def main_const_eval_193(input):
    input_0 = input[0]
    ttnn_reshape_168 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_193 = [ttnn_reshape_168]
    return util_create_list_193


def main_const_eval_194(input):
    input_0 = input[0]
    ttnn_reshape_169 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_194 = [ttnn_reshape_169]
    return util_create_list_194


def main_const_eval_195(input):
    input_0 = input[0]
    ttnn_reshape_170 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_195 = [ttnn_reshape_170]
    return util_create_list_195


def main_const_eval_196(input):
    input_0 = input[0]
    ttnn_reshape_171 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_196 = [ttnn_reshape_171]
    return util_create_list_196


def main_const_eval_197(input):
    input_0 = input[0]
    ttnn_reshape_172 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_197 = [ttnn_reshape_172]
    return util_create_list_197


def main_const_eval_198(input):
    input_0 = input[0]
    ttnn_reshape_173 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_198 = [ttnn_reshape_173]
    return util_create_list_198


def main_const_eval_199(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_24 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_22 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_22 = ttnn.to_device(
        ttnn_to_layout_22,
        device=utils_DeviceGetter_get_device_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    util_create_list_199 = [ttnn_to_device_22]
    return util_create_list_199


def main_const_eval_200(input):
    input_0 = input[0]
    ttnn_reshape_174 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_200 = [ttnn_reshape_174]
    return util_create_list_200


def main_const_eval_201(input):
    input_0 = input[0]
    ttnn_reshape_175 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_201 = [ttnn_reshape_175]
    return util_create_list_201


def main_const_eval_202(input):
    input_0 = input[0]
    ttnn_reshape_176 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_202 = [ttnn_reshape_176]
    return util_create_list_202


def main_const_eval_203(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_25 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_23 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_to_layout_23,
        device=utils_DeviceGetter_get_device_25,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    util_create_list_203 = [ttnn_to_device_23]
    return util_create_list_203


def main_const_eval_204(input):
    input_0 = input[0]
    ttnn_reshape_177 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_204 = [ttnn_reshape_177]
    return util_create_list_204


def main_const_eval_205(input):
    input_0 = input[0]
    ttnn_reshape_178 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_205 = [ttnn_reshape_178]
    return util_create_list_205


def main_const_eval_206(input):
    input_0 = input[0]
    ttnn_reshape_179 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_206 = [ttnn_reshape_179]
    return util_create_list_206


def main_const_eval_207(input):
    input_0 = input[0]
    ttnn_reshape_180 = ttnn.reshape(
        input_0,
        [1, 1, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_207 = [ttnn_reshape_180]
    return util_create_list_207


def main_const_eval_208(input):
    input_0 = input[0]
    ttnn_reshape_181 = ttnn.reshape(
        input_0,
        [1, 1, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_208 = [ttnn_reshape_181]
    return util_create_list_208


def main_const_eval_209(input):
    input_0 = input[0]
    ttnn_reshape_182 = ttnn.reshape(
        input_0,
        [1, 32, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_209 = [ttnn_reshape_182]
    return util_create_list_209


def main_const_eval_210(input):
    input_0 = input[0]
    ttnn_reshape_183 = ttnn.reshape(
        input_0,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_210 = [ttnn_reshape_183]
    return util_create_list_210


def main_const_eval_211(input):
    input_0 = input[0]
    ttnn_reshape_184 = ttnn.reshape(
        input_0,
        [1, 1, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_211 = [ttnn_reshape_184]
    return util_create_list_211


def main_const_eval_212(input):
    input_0 = input[0]
    ttnn_reshape_185 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_212 = [ttnn_reshape_185]
    return util_create_list_212


def main_const_eval_213(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_26 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_24 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_24 = ttnn.to_device(
        ttnn_to_layout_24,
        device=utils_DeviceGetter_get_device_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    util_create_list_213 = [ttnn_to_device_24]
    return util_create_list_213


def main_const_eval_214(input):
    input_0 = input[0]
    ttnn_reshape_186 = ttnn.reshape(
        input_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_214 = [ttnn_reshape_186]
    return util_create_list_214


def main_const_eval_215(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_27 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_25 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_25 = ttnn.to_device(
        ttnn_to_layout_25,
        device=utils_DeviceGetter_get_device_27,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    util_create_list_215 = [ttnn_to_device_25]
    return util_create_list_215


def main_const_eval_216(input):
    input_0 = input[0]
    ttnn_reshape_187 = ttnn.reshape(
        input_0,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_216 = [ttnn_reshape_187]
    return util_create_list_216


def main_const_eval_217(input):
    input_0 = input[0]
    ttnn_reshape_188 = ttnn.reshape(
        input_0,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_217 = [ttnn_reshape_188]
    return util_create_list_217


def main_const_eval_218(input):
    input_0 = input[0]
    utils_DeviceGetter_get_device_28 = utils.DeviceGetter.get_device((1, 1))
    ttnn_to_layout_26 = ttnn.to_layout(
        input_0, ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn_to_device_26 = ttnn.to_device(
        ttnn_to_layout_26,
        device=utils_DeviceGetter_get_device_28,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    util_create_list_218 = [ttnn_to_device_26]
    return util_create_list_218


def main_const_eval_219(input):
    input_0 = input[0]
    ttnn_reshape_189 = ttnn.reshape(
        input_0,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_219 = [ttnn_reshape_189]
    return util_create_list_219


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
CACHED_main_const_eval_130 = None
CACHED_main_const_eval_131 = None
CACHED_main_const_eval_132 = None
CACHED_main_const_eval_133 = None
CACHED_main_const_eval_134 = None
CACHED_main_const_eval_135 = None
CACHED_main_const_eval_136 = None
CACHED_main_const_eval_137 = None
CACHED_main_const_eval_138 = None
CACHED_main_const_eval_139 = None
CACHED_main_const_eval_140 = None
CACHED_main_const_eval_141 = None
CACHED_main_const_eval_142 = None
CACHED_main_const_eval_143 = None
CACHED_main_const_eval_144 = None
CACHED_main_const_eval_145 = None
CACHED_main_const_eval_146 = None
CACHED_main_const_eval_147 = None
CACHED_main_const_eval_148 = None
CACHED_main_const_eval_149 = None
CACHED_main_const_eval_150 = None
CACHED_main_const_eval_151 = None
CACHED_main_const_eval_152 = None
CACHED_main_const_eval_153 = None
CACHED_main_const_eval_154 = None
CACHED_main_const_eval_155 = None
CACHED_main_const_eval_156 = None
CACHED_main_const_eval_157 = None
CACHED_main_const_eval_158 = None
CACHED_main_const_eval_159 = None
CACHED_main_const_eval_160 = None
CACHED_main_const_eval_161 = None
CACHED_main_const_eval_162 = None
CACHED_main_const_eval_163 = None
CACHED_main_const_eval_164 = None
CACHED_main_const_eval_165 = None
CACHED_main_const_eval_166 = None
CACHED_main_const_eval_167 = None
CACHED_main_const_eval_168 = None
CACHED_main_const_eval_169 = None
CACHED_main_const_eval_170 = None
CACHED_main_const_eval_171 = None
CACHED_main_const_eval_172 = None
CACHED_main_const_eval_173 = None
CACHED_main_const_eval_174 = None
CACHED_main_const_eval_175 = None
CACHED_main_const_eval_176 = None
CACHED_main_const_eval_177 = None
CACHED_main_const_eval_178 = None
CACHED_main_const_eval_179 = None
CACHED_main_const_eval_180 = None
CACHED_main_const_eval_181 = None
CACHED_main_const_eval_182 = None
CACHED_main_const_eval_183 = None
CACHED_main_const_eval_184 = None
CACHED_main_const_eval_185 = None
CACHED_main_const_eval_186 = None
CACHED_main_const_eval_187 = None
CACHED_main_const_eval_188 = None
CACHED_main_const_eval_189 = None
CACHED_main_const_eval_190 = None
CACHED_main_const_eval_191 = None
CACHED_main_const_eval_192 = None
CACHED_main_const_eval_193 = None
CACHED_main_const_eval_194 = None
CACHED_main_const_eval_195 = None
CACHED_main_const_eval_196 = None
CACHED_main_const_eval_197 = None
CACHED_main_const_eval_198 = None
CACHED_main_const_eval_199 = None
CACHED_main_const_eval_200 = None
CACHED_main_const_eval_201 = None
CACHED_main_const_eval_202 = None
CACHED_main_const_eval_203 = None
CACHED_main_const_eval_204 = None
CACHED_main_const_eval_205 = None
CACHED_main_const_eval_206 = None
CACHED_main_const_eval_207 = None
CACHED_main_const_eval_208 = None
CACHED_main_const_eval_209 = None
CACHED_main_const_eval_210 = None
CACHED_main_const_eval_211 = None
CACHED_main_const_eval_212 = None
CACHED_main_const_eval_213 = None
CACHED_main_const_eval_214 = None
CACHED_main_const_eval_215 = None
CACHED_main_const_eval_216 = None
CACHED_main_const_eval_217 = None
CACHED_main_const_eval_218 = None
CACHED_main_const_eval_219 = None


def _main(input):
    global CACHED_main_const_eval_219
    global CACHED_main_const_eval_218
    global CACHED_main_const_eval_217
    global CACHED_main_const_eval_216
    global CACHED_main_const_eval_215
    global CACHED_main_const_eval_214
    global CACHED_main_const_eval_213
    global CACHED_main_const_eval_212
    global CACHED_main_const_eval_211
    global CACHED_main_const_eval_210
    global CACHED_main_const_eval_209
    global CACHED_main_const_eval_208
    global CACHED_main_const_eval_207
    global CACHED_main_const_eval_206
    global CACHED_main_const_eval_205
    global CACHED_main_const_eval_204
    global CACHED_main_const_eval_203
    global CACHED_main_const_eval_202
    global CACHED_main_const_eval_201
    global CACHED_main_const_eval_200
    global CACHED_main_const_eval_199
    global CACHED_main_const_eval_198
    global CACHED_main_const_eval_197
    global CACHED_main_const_eval_196
    global CACHED_main_const_eval_195
    global CACHED_main_const_eval_194
    global CACHED_main_const_eval_193
    global CACHED_main_const_eval_192
    global CACHED_main_const_eval_191
    global CACHED_main_const_eval_190
    global CACHED_main_const_eval_189
    global CACHED_main_const_eval_188
    global CACHED_main_const_eval_187
    global CACHED_main_const_eval_186
    global CACHED_main_const_eval_185
    global CACHED_main_const_eval_184
    global CACHED_main_const_eval_183
    global CACHED_main_const_eval_182
    global CACHED_main_const_eval_181
    global CACHED_main_const_eval_180
    global CACHED_main_const_eval_179
    global CACHED_main_const_eval_178
    global CACHED_main_const_eval_177
    global CACHED_main_const_eval_176
    global CACHED_main_const_eval_175
    global CACHED_main_const_eval_174
    global CACHED_main_const_eval_173
    global CACHED_main_const_eval_172
    global CACHED_main_const_eval_171
    global CACHED_main_const_eval_170
    global CACHED_main_const_eval_169
    global CACHED_main_const_eval_168
    global CACHED_main_const_eval_167
    global CACHED_main_const_eval_166
    global CACHED_main_const_eval_165
    global CACHED_main_const_eval_164
    global CACHED_main_const_eval_163
    global CACHED_main_const_eval_162
    global CACHED_main_const_eval_161
    global CACHED_main_const_eval_160
    global CACHED_main_const_eval_159
    global CACHED_main_const_eval_158
    global CACHED_main_const_eval_157
    global CACHED_main_const_eval_156
    global CACHED_main_const_eval_155
    global CACHED_main_const_eval_154
    global CACHED_main_const_eval_153
    global CACHED_main_const_eval_152
    global CACHED_main_const_eval_151
    global CACHED_main_const_eval_150
    global CACHED_main_const_eval_149
    global CACHED_main_const_eval_148
    global CACHED_main_const_eval_147
    global CACHED_main_const_eval_146
    global CACHED_main_const_eval_145
    global CACHED_main_const_eval_144
    global CACHED_main_const_eval_143
    global CACHED_main_const_eval_142
    global CACHED_main_const_eval_141
    global CACHED_main_const_eval_140
    global CACHED_main_const_eval_139
    global CACHED_main_const_eval_138
    global CACHED_main_const_eval_137
    global CACHED_main_const_eval_136
    global CACHED_main_const_eval_135
    global CACHED_main_const_eval_134
    global CACHED_main_const_eval_133
    global CACHED_main_const_eval_132
    global CACHED_main_const_eval_131
    global CACHED_main_const_eval_130
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
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(
        const_0, CACHED_main_const_eval_0
    )
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    util_create_list_220 = [input_67]
    utils_constEvalFuncWrapper_0 = utils.constEvalFuncWrapper(
        const_1, util_create_list_220, CACHED_main_const_eval_1
    )
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapper_0
    utils_constEvalFuncWrapper_0_0 = utils_constEvalFuncWrapper_0[0]
    const_2 = main_const_eval_2
    util_create_list_221 = [input_17]
    utils_constEvalFuncWrapper_1 = utils.constEvalFuncWrapper(
        const_2, util_create_list_221, CACHED_main_const_eval_2
    )
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapper_1
    utils_constEvalFuncWrapper_1_0 = utils_constEvalFuncWrapper_1[0]
    const_3 = main_const_eval_3
    util_create_list_222 = [input_128]
    utils_constEvalFuncWrapper_2 = utils.constEvalFuncWrapper(
        const_3, util_create_list_222, CACHED_main_const_eval_3
    )
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapper_2
    utils_constEvalFuncWrapper_2_0 = utils_constEvalFuncWrapper_2[0]
    const_4 = main_const_eval_4
    util_create_list_223 = [input_13]
    utils_constEvalFuncWrapper_3 = utils.constEvalFuncWrapper(
        const_4, util_create_list_223, CACHED_main_const_eval_4
    )
    CACHED_main_const_eval_4 = utils_constEvalFuncWrapper_3
    utils_constEvalFuncWrapper_3_0 = utils_constEvalFuncWrapper_3[0]
    const_5 = main_const_eval_5
    util_create_list_224 = [input_55]
    utils_constEvalFuncWrapper_4 = utils.constEvalFuncWrapper(
        const_5, util_create_list_224, CACHED_main_const_eval_5
    )
    CACHED_main_const_eval_5 = utils_constEvalFuncWrapper_4
    utils_constEvalFuncWrapper_4_0 = utils_constEvalFuncWrapper_4[0]
    const_6 = main_const_eval_6
    util_create_list_225 = [input_85]
    utils_constEvalFuncWrapper_5 = utils.constEvalFuncWrapper(
        const_6, util_create_list_225, CACHED_main_const_eval_6
    )
    CACHED_main_const_eval_6 = utils_constEvalFuncWrapper_5
    utils_constEvalFuncWrapper_5_0 = utils_constEvalFuncWrapper_5[0]
    const_7 = main_const_eval_7
    util_create_list_226 = [input_77]
    utils_constEvalFuncWrapper_6 = utils.constEvalFuncWrapper(
        const_7, util_create_list_226, CACHED_main_const_eval_7
    )
    CACHED_main_const_eval_7 = utils_constEvalFuncWrapper_6
    utils_constEvalFuncWrapper_6_0 = utils_constEvalFuncWrapper_6[0]
    const_8 = main_const_eval_8
    util_create_list_227 = [input_97]
    utils_constEvalFuncWrapper_7 = utils.constEvalFuncWrapper(
        const_8, util_create_list_227, CACHED_main_const_eval_8
    )
    CACHED_main_const_eval_8 = utils_constEvalFuncWrapper_7
    utils_constEvalFuncWrapper_7_0 = utils_constEvalFuncWrapper_7[0]
    const_9 = main_const_eval_9
    util_create_list_228 = [input_96]
    utils_constEvalFuncWrapper_8 = utils.constEvalFuncWrapper(
        const_9, util_create_list_228, CACHED_main_const_eval_9
    )
    CACHED_main_const_eval_9 = utils_constEvalFuncWrapper_8
    utils_constEvalFuncWrapper_8_0 = utils_constEvalFuncWrapper_8[0]
    const_10 = main_const_eval_10
    util_create_list_229 = [input_83]
    utils_constEvalFuncWrapper_9 = utils.constEvalFuncWrapper(
        const_10, util_create_list_229, CACHED_main_const_eval_10
    )
    CACHED_main_const_eval_10 = utils_constEvalFuncWrapper_9
    utils_constEvalFuncWrapper_9_0 = utils_constEvalFuncWrapper_9[0]
    const_11 = main_const_eval_11
    util_create_list_230 = [input_26]
    utils_constEvalFuncWrapper_10 = utils.constEvalFuncWrapper(
        const_11, util_create_list_230, CACHED_main_const_eval_11
    )
    CACHED_main_const_eval_11 = utils_constEvalFuncWrapper_10
    utils_constEvalFuncWrapper_10_0 = utils_constEvalFuncWrapper_10[0]
    const_12 = main_const_eval_12
    util_create_list_231 = [input_70]
    utils_constEvalFuncWrapper_11 = utils.constEvalFuncWrapper(
        const_12, util_create_list_231, CACHED_main_const_eval_12
    )
    CACHED_main_const_eval_12 = utils_constEvalFuncWrapper_11
    utils_constEvalFuncWrapper_11_0 = utils_constEvalFuncWrapper_11[0]
    const_13 = main_const_eval_13
    util_create_list_232 = [input_123]
    utils_constEvalFuncWrapper_12 = utils.constEvalFuncWrapper(
        const_13, util_create_list_232, CACHED_main_const_eval_13
    )
    CACHED_main_const_eval_13 = utils_constEvalFuncWrapper_12
    utils_constEvalFuncWrapper_12_0 = utils_constEvalFuncWrapper_12[0]
    const_14 = main_const_eval_14
    util_create_list_233 = [input_59]
    utils_constEvalFuncWrapper_13 = utils.constEvalFuncWrapper(
        const_14, util_create_list_233, CACHED_main_const_eval_14
    )
    CACHED_main_const_eval_14 = utils_constEvalFuncWrapper_13
    utils_constEvalFuncWrapper_13_0 = utils_constEvalFuncWrapper_13[0]
    const_15 = main_const_eval_15
    util_create_list_234 = [input_88]
    utils_constEvalFuncWrapper_14 = utils.constEvalFuncWrapper(
        const_15, util_create_list_234, CACHED_main_const_eval_15
    )
    CACHED_main_const_eval_15 = utils_constEvalFuncWrapper_14
    utils_constEvalFuncWrapper_14_0 = utils_constEvalFuncWrapper_14[0]
    const_16 = main_const_eval_16
    util_create_list_235 = [input_20]
    utils_constEvalFuncWrapper_15 = utils.constEvalFuncWrapper(
        const_16, util_create_list_235, CACHED_main_const_eval_16
    )
    CACHED_main_const_eval_16 = utils_constEvalFuncWrapper_15
    utils_constEvalFuncWrapper_15_0 = utils_constEvalFuncWrapper_15[0]
    const_17 = main_const_eval_17
    util_create_list_236 = [input_61]
    utils_constEvalFuncWrapper_16 = utils.constEvalFuncWrapper(
        const_17, util_create_list_236, CACHED_main_const_eval_17
    )
    CACHED_main_const_eval_17 = utils_constEvalFuncWrapper_16
    utils_constEvalFuncWrapper_16_0 = utils_constEvalFuncWrapper_16[0]
    const_18 = main_const_eval_18
    util_create_list_237 = [input_65]
    utils_constEvalFuncWrapper_17 = utils.constEvalFuncWrapper(
        const_18, util_create_list_237, CACHED_main_const_eval_18
    )
    CACHED_main_const_eval_18 = utils_constEvalFuncWrapper_17
    utils_constEvalFuncWrapper_17_0 = utils_constEvalFuncWrapper_17[0]
    const_19 = main_const_eval_19
    util_create_list_238 = [input_127]
    utils_constEvalFuncWrapper_18 = utils.constEvalFuncWrapper(
        const_19, util_create_list_238, CACHED_main_const_eval_19
    )
    CACHED_main_const_eval_19 = utils_constEvalFuncWrapper_18
    utils_constEvalFuncWrapper_18_0 = utils_constEvalFuncWrapper_18[0]
    const_20 = main_const_eval_20
    util_create_list_239 = [input_65]
    utils_constEvalFuncWrapper_19 = utils.constEvalFuncWrapper(
        const_20, util_create_list_239, CACHED_main_const_eval_20
    )
    CACHED_main_const_eval_20 = utils_constEvalFuncWrapper_19
    utils_constEvalFuncWrapper_19_0 = utils_constEvalFuncWrapper_19[0]
    const_21 = main_const_eval_21
    util_create_list_240 = [input_16]
    utils_constEvalFuncWrapper_20 = utils.constEvalFuncWrapper(
        const_21, util_create_list_240, CACHED_main_const_eval_21
    )
    CACHED_main_const_eval_21 = utils_constEvalFuncWrapper_20
    utils_constEvalFuncWrapper_20_0 = utils_constEvalFuncWrapper_20[0]
    const_22 = main_const_eval_22
    util_create_list_241 = [input_97]
    utils_constEvalFuncWrapper_21 = utils.constEvalFuncWrapper(
        const_22, util_create_list_241, CACHED_main_const_eval_22
    )
    CACHED_main_const_eval_22 = utils_constEvalFuncWrapper_21
    utils_constEvalFuncWrapper_21_0 = utils_constEvalFuncWrapper_21[0]
    const_23 = main_const_eval_23
    util_create_list_242 = [input_50]
    utils_constEvalFuncWrapper_22 = utils.constEvalFuncWrapper(
        const_23, util_create_list_242, CACHED_main_const_eval_23
    )
    CACHED_main_const_eval_23 = utils_constEvalFuncWrapper_22
    utils_constEvalFuncWrapper_22_0 = utils_constEvalFuncWrapper_22[0]
    const_24 = main_const_eval_24
    util_create_list_243 = [input_31]
    utils_constEvalFuncWrapper_23 = utils.constEvalFuncWrapper(
        const_24, util_create_list_243, CACHED_main_const_eval_24
    )
    CACHED_main_const_eval_24 = utils_constEvalFuncWrapper_23
    utils_constEvalFuncWrapper_23_0 = utils_constEvalFuncWrapper_23[0]
    const_25 = main_const_eval_25
    util_create_list_244 = [input_100]
    utils_constEvalFuncWrapper_24 = utils.constEvalFuncWrapper(
        const_25, util_create_list_244, CACHED_main_const_eval_25
    )
    CACHED_main_const_eval_25 = utils_constEvalFuncWrapper_24
    utils_constEvalFuncWrapper_24_0 = utils_constEvalFuncWrapper_24[0]
    const_26 = main_const_eval_26
    util_create_list_245 = [input_34]
    utils_constEvalFuncWrapper_25 = utils.constEvalFuncWrapper(
        const_26, util_create_list_245, CACHED_main_const_eval_26
    )
    CACHED_main_const_eval_26 = utils_constEvalFuncWrapper_25
    utils_constEvalFuncWrapper_25_0 = utils_constEvalFuncWrapper_25[0]
    const_27 = main_const_eval_27
    util_create_list_246 = [input_31]
    utils_constEvalFuncWrapper_26 = utils.constEvalFuncWrapper(
        const_27, util_create_list_246, CACHED_main_const_eval_27
    )
    CACHED_main_const_eval_27 = utils_constEvalFuncWrapper_26
    utils_constEvalFuncWrapper_26_0 = utils_constEvalFuncWrapper_26[0]
    const_28 = main_const_eval_28
    util_create_list_247 = [input_54]
    utils_constEvalFuncWrapper_27 = utils.constEvalFuncWrapper(
        const_28, util_create_list_247, CACHED_main_const_eval_28
    )
    CACHED_main_const_eval_28 = utils_constEvalFuncWrapper_27
    utils_constEvalFuncWrapper_27_0 = utils_constEvalFuncWrapper_27[0]
    const_29 = main_const_eval_29
    util_create_list_248 = [input_103]
    utils_constEvalFuncWrapper_28 = utils.constEvalFuncWrapper(
        const_29, util_create_list_248, CACHED_main_const_eval_29
    )
    CACHED_main_const_eval_29 = utils_constEvalFuncWrapper_28
    utils_constEvalFuncWrapper_28_0 = utils_constEvalFuncWrapper_28[0]
    const_30 = main_const_eval_30
    util_create_list_249 = [input_11]
    utils_constEvalFuncWrapper_29 = utils.constEvalFuncWrapper(
        const_30, util_create_list_249, CACHED_main_const_eval_30
    )
    CACHED_main_const_eval_30 = utils_constEvalFuncWrapper_29
    utils_constEvalFuncWrapper_29_0 = utils_constEvalFuncWrapper_29[0]
    const_31 = main_const_eval_31
    util_create_list_250 = [input_21]
    utils_constEvalFuncWrapper_30 = utils.constEvalFuncWrapper(
        const_31, util_create_list_250, CACHED_main_const_eval_31
    )
    CACHED_main_const_eval_31 = utils_constEvalFuncWrapper_30
    utils_constEvalFuncWrapper_30_0 = utils_constEvalFuncWrapper_30[0]
    const_32 = main_const_eval_32
    util_create_list_251 = [input_111]
    utils_constEvalFuncWrapper_31 = utils.constEvalFuncWrapper(
        const_32, util_create_list_251, CACHED_main_const_eval_32
    )
    CACHED_main_const_eval_32 = utils_constEvalFuncWrapper_31
    utils_constEvalFuncWrapper_31_0 = utils_constEvalFuncWrapper_31[0]
    const_33 = main_const_eval_33
    util_create_list_252 = [input_70]
    utils_constEvalFuncWrapper_32 = utils.constEvalFuncWrapper(
        const_33, util_create_list_252, CACHED_main_const_eval_33
    )
    CACHED_main_const_eval_33 = utils_constEvalFuncWrapper_32
    utils_constEvalFuncWrapper_32_0 = utils_constEvalFuncWrapper_32[0]
    const_34 = main_const_eval_34
    util_create_list_253 = [input_75]
    utils_constEvalFuncWrapper_33 = utils.constEvalFuncWrapper(
        const_34, util_create_list_253, CACHED_main_const_eval_34
    )
    CACHED_main_const_eval_34 = utils_constEvalFuncWrapper_33
    utils_constEvalFuncWrapper_33_0 = utils_constEvalFuncWrapper_33[0]
    const_35 = main_const_eval_35
    util_create_list_254 = [input_120]
    utils_constEvalFuncWrapper_34 = utils.constEvalFuncWrapper(
        const_35, util_create_list_254, CACHED_main_const_eval_35
    )
    CACHED_main_const_eval_35 = utils_constEvalFuncWrapper_34
    utils_constEvalFuncWrapper_34_0 = utils_constEvalFuncWrapper_34[0]
    const_36 = main_const_eval_36
    util_create_list_255 = [input_129]
    utils_constEvalFuncWrapper_35 = utils.constEvalFuncWrapper(
        const_36, util_create_list_255, CACHED_main_const_eval_36
    )
    CACHED_main_const_eval_36 = utils_constEvalFuncWrapper_35
    utils_constEvalFuncWrapper_35_0 = utils_constEvalFuncWrapper_35[0]
    const_37 = main_const_eval_37
    util_create_list_256 = [input_87]
    utils_constEvalFuncWrapper_36 = utils.constEvalFuncWrapper(
        const_37, util_create_list_256, CACHED_main_const_eval_37
    )
    CACHED_main_const_eval_37 = utils_constEvalFuncWrapper_36
    utils_constEvalFuncWrapper_36_0 = utils_constEvalFuncWrapper_36[0]
    const_38 = main_const_eval_38
    util_create_list_257 = [input_37]
    utils_constEvalFuncWrapper_37 = utils.constEvalFuncWrapper(
        const_38, util_create_list_257, CACHED_main_const_eval_38
    )
    CACHED_main_const_eval_38 = utils_constEvalFuncWrapper_37
    utils_constEvalFuncWrapper_37_0 = utils_constEvalFuncWrapper_37[0]
    const_39 = main_const_eval_39
    util_create_list_258 = [input_69]
    utils_constEvalFuncWrapper_38 = utils.constEvalFuncWrapper(
        const_39, util_create_list_258, CACHED_main_const_eval_39
    )
    CACHED_main_const_eval_39 = utils_constEvalFuncWrapper_38
    utils_constEvalFuncWrapper_38_0 = utils_constEvalFuncWrapper_38[0]
    const_40 = main_const_eval_40
    util_create_list_259 = [input_122]
    utils_constEvalFuncWrapper_39 = utils.constEvalFuncWrapper(
        const_40, util_create_list_259, CACHED_main_const_eval_40
    )
    CACHED_main_const_eval_40 = utils_constEvalFuncWrapper_39
    utils_constEvalFuncWrapper_39_0 = utils_constEvalFuncWrapper_39[0]
    const_41 = main_const_eval_41
    util_create_list_260 = [input_48]
    utils_constEvalFuncWrapper_40 = utils.constEvalFuncWrapper(
        const_41, util_create_list_260, CACHED_main_const_eval_41
    )
    CACHED_main_const_eval_41 = utils_constEvalFuncWrapper_40
    utils_constEvalFuncWrapper_40_0 = utils_constEvalFuncWrapper_40[0]
    const_42 = main_const_eval_42
    util_create_list_261 = [input_86]
    utils_constEvalFuncWrapper_41 = utils.constEvalFuncWrapper(
        const_42, util_create_list_261, CACHED_main_const_eval_42
    )
    CACHED_main_const_eval_42 = utils_constEvalFuncWrapper_41
    utils_constEvalFuncWrapper_41_0 = utils_constEvalFuncWrapper_41[0]
    const_43 = main_const_eval_43
    util_create_list_262 = [input_92]
    utils_constEvalFuncWrapper_42 = utils.constEvalFuncWrapper(
        const_43, util_create_list_262, CACHED_main_const_eval_43
    )
    CACHED_main_const_eval_43 = utils_constEvalFuncWrapper_42
    utils_constEvalFuncWrapper_42_0 = utils_constEvalFuncWrapper_42[0]
    const_44 = main_const_eval_44
    util_create_list_263 = [input_39]
    utils_constEvalFuncWrapper_43 = utils.constEvalFuncWrapper(
        const_44, util_create_list_263, CACHED_main_const_eval_44
    )
    CACHED_main_const_eval_44 = utils_constEvalFuncWrapper_43
    utils_constEvalFuncWrapper_43_0 = utils_constEvalFuncWrapper_43[0]
    const_45 = main_const_eval_45
    util_create_list_264 = [input_15]
    utils_constEvalFuncWrapper_44 = utils.constEvalFuncWrapper(
        const_45, util_create_list_264, CACHED_main_const_eval_45
    )
    CACHED_main_const_eval_45 = utils_constEvalFuncWrapper_44
    utils_constEvalFuncWrapper_44_0 = utils_constEvalFuncWrapper_44[0]
    const_46 = main_const_eval_46
    util_create_list_265 = [input_78]
    utils_constEvalFuncWrapper_45 = utils.constEvalFuncWrapper(
        const_46, util_create_list_265, CACHED_main_const_eval_46
    )
    CACHED_main_const_eval_46 = utils_constEvalFuncWrapper_45
    utils_constEvalFuncWrapper_45_0 = utils_constEvalFuncWrapper_45[0]
    const_47 = main_const_eval_47
    util_create_list_266 = [input_131]
    utils_constEvalFuncWrapper_46 = utils.constEvalFuncWrapper(
        const_47, util_create_list_266, CACHED_main_const_eval_47
    )
    CACHED_main_const_eval_47 = utils_constEvalFuncWrapper_46
    utils_constEvalFuncWrapper_46_0 = utils_constEvalFuncWrapper_46[0]
    const_48 = main_const_eval_48
    util_create_list_267 = [input_105]
    utils_constEvalFuncWrapper_47 = utils.constEvalFuncWrapper(
        const_48, util_create_list_267, CACHED_main_const_eval_48
    )
    CACHED_main_const_eval_48 = utils_constEvalFuncWrapper_47
    utils_constEvalFuncWrapper_47_0 = utils_constEvalFuncWrapper_47[0]
    const_49 = main_const_eval_49
    util_create_list_268 = [input_106]
    utils_constEvalFuncWrapper_48 = utils.constEvalFuncWrapper(
        const_49, util_create_list_268, CACHED_main_const_eval_49
    )
    CACHED_main_const_eval_49 = utils_constEvalFuncWrapper_48
    utils_constEvalFuncWrapper_48_0 = utils_constEvalFuncWrapper_48[0]
    const_50 = main_const_eval_50
    util_create_list_269 = [input_71]
    utils_constEvalFuncWrapper_49 = utils.constEvalFuncWrapper(
        const_50, util_create_list_269, CACHED_main_const_eval_50
    )
    CACHED_main_const_eval_50 = utils_constEvalFuncWrapper_49
    utils_constEvalFuncWrapper_49_0 = utils_constEvalFuncWrapper_49[0]
    const_51 = main_const_eval_51
    util_create_list_270 = [input_95]
    utils_constEvalFuncWrapper_50 = utils.constEvalFuncWrapper(
        const_51, util_create_list_270, CACHED_main_const_eval_51
    )
    CACHED_main_const_eval_51 = utils_constEvalFuncWrapper_50
    utils_constEvalFuncWrapper_50_0 = utils_constEvalFuncWrapper_50[0]
    const_52 = main_const_eval_52
    util_create_list_271 = [input_71]
    utils_constEvalFuncWrapper_51 = utils.constEvalFuncWrapper(
        const_52, util_create_list_271, CACHED_main_const_eval_52
    )
    CACHED_main_const_eval_52 = utils_constEvalFuncWrapper_51
    utils_constEvalFuncWrapper_51_0 = utils_constEvalFuncWrapper_51[0]
    const_53 = main_const_eval_53
    util_create_list_272 = [input_12]
    utils_constEvalFuncWrapper_52 = utils.constEvalFuncWrapper(
        const_53, util_create_list_272, CACHED_main_const_eval_53
    )
    CACHED_main_const_eval_53 = utils_constEvalFuncWrapper_52
    utils_constEvalFuncWrapper_52_0 = utils_constEvalFuncWrapper_52[0]
    const_54 = main_const_eval_54
    util_create_list_273 = [input_134]
    utils_constEvalFuncWrapper_53 = utils.constEvalFuncWrapper(
        const_54, util_create_list_273, CACHED_main_const_eval_54
    )
    CACHED_main_const_eval_54 = utils_constEvalFuncWrapper_53
    utils_constEvalFuncWrapper_53_0 = utils_constEvalFuncWrapper_53[0]
    const_55 = main_const_eval_55
    util_create_list_274 = [input_100]
    utils_constEvalFuncWrapper_54 = utils.constEvalFuncWrapper(
        const_55, util_create_list_274, CACHED_main_const_eval_55
    )
    CACHED_main_const_eval_55 = utils_constEvalFuncWrapper_54
    utils_constEvalFuncWrapper_54_0 = utils_constEvalFuncWrapper_54[0]
    const_56 = main_const_eval_56
    util_create_list_275 = [input_75]
    utils_constEvalFuncWrapper_55 = utils.constEvalFuncWrapper(
        const_56, util_create_list_275, CACHED_main_const_eval_56
    )
    CACHED_main_const_eval_56 = utils_constEvalFuncWrapper_55
    utils_constEvalFuncWrapper_55_0 = utils_constEvalFuncWrapper_55[0]
    const_57 = main_const_eval_57
    util_create_list_276 = [input_72]
    utils_constEvalFuncWrapper_56 = utils.constEvalFuncWrapper(
        const_57, util_create_list_276, CACHED_main_const_eval_57
    )
    CACHED_main_const_eval_57 = utils_constEvalFuncWrapper_56
    utils_constEvalFuncWrapper_56_0 = utils_constEvalFuncWrapper_56[0]
    const_58 = main_const_eval_58
    util_create_list_277 = [input_79]
    utils_constEvalFuncWrapper_57 = utils.constEvalFuncWrapper(
        const_58, util_create_list_277, CACHED_main_const_eval_58
    )
    CACHED_main_const_eval_58 = utils_constEvalFuncWrapper_57
    utils_constEvalFuncWrapper_57_0 = utils_constEvalFuncWrapper_57[0]
    const_59 = main_const_eval_59
    util_create_list_278 = [input_69]
    utils_constEvalFuncWrapper_58 = utils.constEvalFuncWrapper(
        const_59, util_create_list_278, CACHED_main_const_eval_59
    )
    CACHED_main_const_eval_59 = utils_constEvalFuncWrapper_58
    utils_constEvalFuncWrapper_58_0 = utils_constEvalFuncWrapper_58[0]
    const_60 = main_const_eval_60
    util_create_list_279 = [input_45]
    utils_constEvalFuncWrapper_59 = utils.constEvalFuncWrapper(
        const_60, util_create_list_279, CACHED_main_const_eval_60
    )
    CACHED_main_const_eval_60 = utils_constEvalFuncWrapper_59
    utils_constEvalFuncWrapper_59_0 = utils_constEvalFuncWrapper_59[0]
    const_61 = main_const_eval_61
    util_create_list_280 = [input_67]
    utils_constEvalFuncWrapper_60 = utils.constEvalFuncWrapper(
        const_61, util_create_list_280, CACHED_main_const_eval_61
    )
    CACHED_main_const_eval_61 = utils_constEvalFuncWrapper_60
    utils_constEvalFuncWrapper_60_0 = utils_constEvalFuncWrapper_60[0]
    const_62 = main_const_eval_62
    util_create_list_281 = [input_38]
    utils_constEvalFuncWrapper_61 = utils.constEvalFuncWrapper(
        const_62, util_create_list_281, CACHED_main_const_eval_62
    )
    CACHED_main_const_eval_62 = utils_constEvalFuncWrapper_61
    utils_constEvalFuncWrapper_61_0 = utils_constEvalFuncWrapper_61[0]
    const_63 = main_const_eval_63
    util_create_list_282 = [input_22]
    utils_constEvalFuncWrapper_62 = utils.constEvalFuncWrapper(
        const_63, util_create_list_282, CACHED_main_const_eval_63
    )
    CACHED_main_const_eval_63 = utils_constEvalFuncWrapper_62
    utils_constEvalFuncWrapper_62_0 = utils_constEvalFuncWrapper_62[0]
    const_64 = main_const_eval_64
    util_create_list_283 = [input_133]
    utils_constEvalFuncWrapper_63 = utils.constEvalFuncWrapper(
        const_64, util_create_list_283, CACHED_main_const_eval_64
    )
    CACHED_main_const_eval_64 = utils_constEvalFuncWrapper_63
    utils_constEvalFuncWrapper_63_0 = utils_constEvalFuncWrapper_63[0]
    const_65 = main_const_eval_65
    util_create_list_284 = [input_83]
    utils_constEvalFuncWrapper_64 = utils.constEvalFuncWrapper(
        const_65, util_create_list_284, CACHED_main_const_eval_65
    )
    CACHED_main_const_eval_65 = utils_constEvalFuncWrapper_64
    utils_constEvalFuncWrapper_64_0 = utils_constEvalFuncWrapper_64[0]
    const_66 = main_const_eval_66
    util_create_list_285 = [input_113]
    utils_constEvalFuncWrapper_65 = utils.constEvalFuncWrapper(
        const_66, util_create_list_285, CACHED_main_const_eval_66
    )
    CACHED_main_const_eval_66 = utils_constEvalFuncWrapper_65
    utils_constEvalFuncWrapper_65_0 = utils_constEvalFuncWrapper_65[0]
    const_67 = main_const_eval_67
    util_create_list_286 = [input_11]
    utils_constEvalFuncWrapper_66 = utils.constEvalFuncWrapper(
        const_67, util_create_list_286, CACHED_main_const_eval_67
    )
    CACHED_main_const_eval_67 = utils_constEvalFuncWrapper_66
    utils_constEvalFuncWrapper_66_0 = utils_constEvalFuncWrapper_66[0]
    const_68 = main_const_eval_68
    util_create_list_287 = [input_116]
    utils_constEvalFuncWrapper_67 = utils.constEvalFuncWrapper(
        const_68, util_create_list_287, CACHED_main_const_eval_68
    )
    CACHED_main_const_eval_68 = utils_constEvalFuncWrapper_67
    utils_constEvalFuncWrapper_67_0 = utils_constEvalFuncWrapper_67[0]
    const_69 = main_const_eval_69
    util_create_list_288 = [input_108]
    utils_constEvalFuncWrapper_68 = utils.constEvalFuncWrapper(
        const_69, util_create_list_288, CACHED_main_const_eval_69
    )
    CACHED_main_const_eval_69 = utils_constEvalFuncWrapper_68
    utils_constEvalFuncWrapper_68_0 = utils_constEvalFuncWrapper_68[0]
    const_70 = main_const_eval_70
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(
        const_70, CACHED_main_const_eval_70
    )
    CACHED_main_const_eval_70 = utils_constEvalFuncWrapperZeroArg_1
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_71 = main_const_eval_71
    util_create_list_289 = [input_45]
    utils_constEvalFuncWrapper_69 = utils.constEvalFuncWrapper(
        const_71, util_create_list_289, CACHED_main_const_eval_71
    )
    CACHED_main_const_eval_71 = utils_constEvalFuncWrapper_69
    utils_constEvalFuncWrapper_69_0 = utils_constEvalFuncWrapper_69[0]
    const_72 = main_const_eval_72
    util_create_list_290 = [input_5]
    utils_constEvalFuncWrapper_70 = utils.constEvalFuncWrapper(
        const_72, util_create_list_290, CACHED_main_const_eval_72
    )
    CACHED_main_const_eval_72 = utils_constEvalFuncWrapper_70
    utils_constEvalFuncWrapper_70_0 = utils_constEvalFuncWrapper_70[0]
    const_73 = main_const_eval_73
    util_create_list_291 = [input_17]
    utils_constEvalFuncWrapper_71 = utils.constEvalFuncWrapper(
        const_73, util_create_list_291, CACHED_main_const_eval_73
    )
    CACHED_main_const_eval_73 = utils_constEvalFuncWrapper_71
    utils_constEvalFuncWrapper_71_0 = utils_constEvalFuncWrapper_71[0]
    const_74 = main_const_eval_74
    util_create_list_292 = [input_102]
    utils_constEvalFuncWrapper_72 = utils.constEvalFuncWrapper(
        const_74, util_create_list_292, CACHED_main_const_eval_74
    )
    CACHED_main_const_eval_74 = utils_constEvalFuncWrapper_72
    utils_constEvalFuncWrapper_72_0 = utils_constEvalFuncWrapper_72[0]
    const_75 = main_const_eval_75
    util_create_list_293 = [input_94]
    utils_constEvalFuncWrapper_73 = utils.constEvalFuncWrapper(
        const_75, util_create_list_293, CACHED_main_const_eval_75
    )
    CACHED_main_const_eval_75 = utils_constEvalFuncWrapper_73
    utils_constEvalFuncWrapper_73_0 = utils_constEvalFuncWrapper_73[0]
    const_76 = main_const_eval_76
    util_create_list_294 = [input_87]
    utils_constEvalFuncWrapper_74 = utils.constEvalFuncWrapper(
        const_76, util_create_list_294, CACHED_main_const_eval_76
    )
    CACHED_main_const_eval_76 = utils_constEvalFuncWrapper_74
    utils_constEvalFuncWrapper_74_0 = utils_constEvalFuncWrapper_74[0]
    const_77 = main_const_eval_77
    util_create_list_295 = [input_54]
    utils_constEvalFuncWrapper_75 = utils.constEvalFuncWrapper(
        const_77, util_create_list_295, CACHED_main_const_eval_77
    )
    CACHED_main_const_eval_77 = utils_constEvalFuncWrapper_75
    utils_constEvalFuncWrapper_75_0 = utils_constEvalFuncWrapper_75[0]
    const_78 = main_const_eval_78
    util_create_list_296 = [input_10]
    utils_constEvalFuncWrapper_76 = utils.constEvalFuncWrapper(
        const_78, util_create_list_296, CACHED_main_const_eval_78
    )
    CACHED_main_const_eval_78 = utils_constEvalFuncWrapper_76
    utils_constEvalFuncWrapper_76_0 = utils_constEvalFuncWrapper_76[0]
    const_79 = main_const_eval_79
    util_create_list_297 = [input_106]
    utils_constEvalFuncWrapper_77 = utils.constEvalFuncWrapper(
        const_79, util_create_list_297, CACHED_main_const_eval_79
    )
    CACHED_main_const_eval_79 = utils_constEvalFuncWrapper_77
    utils_constEvalFuncWrapper_77_0 = utils_constEvalFuncWrapper_77[0]
    const_80 = main_const_eval_80
    util_create_list_298 = [input_61]
    utils_constEvalFuncWrapper_78 = utils.constEvalFuncWrapper(
        const_80, util_create_list_298, CACHED_main_const_eval_80
    )
    CACHED_main_const_eval_80 = utils_constEvalFuncWrapper_78
    utils_constEvalFuncWrapper_78_0 = utils_constEvalFuncWrapper_78[0]
    const_81 = main_const_eval_81
    util_create_list_299 = [input_41]
    utils_constEvalFuncWrapper_79 = utils.constEvalFuncWrapper(
        const_81, util_create_list_299, CACHED_main_const_eval_81
    )
    CACHED_main_const_eval_81 = utils_constEvalFuncWrapper_79
    utils_constEvalFuncWrapper_79_0 = utils_constEvalFuncWrapper_79[0]
    const_82 = main_const_eval_82
    util_create_list_300 = [input_14]
    utils_constEvalFuncWrapper_80 = utils.constEvalFuncWrapper(
        const_82, util_create_list_300, CACHED_main_const_eval_82
    )
    CACHED_main_const_eval_82 = utils_constEvalFuncWrapper_80
    utils_constEvalFuncWrapper_80_0 = utils_constEvalFuncWrapper_80[0]
    const_83 = main_const_eval_83
    util_create_list_301 = [input_52]
    utils_constEvalFuncWrapper_81 = utils.constEvalFuncWrapper(
        const_83, util_create_list_301, CACHED_main_const_eval_83
    )
    CACHED_main_const_eval_83 = utils_constEvalFuncWrapper_81
    utils_constEvalFuncWrapper_81_0 = utils_constEvalFuncWrapper_81[0]
    const_84 = main_const_eval_84
    util_create_list_302 = [input_98]
    utils_constEvalFuncWrapper_82 = utils.constEvalFuncWrapper(
        const_84, util_create_list_302, CACHED_main_const_eval_84
    )
    CACHED_main_const_eval_84 = utils_constEvalFuncWrapper_82
    utils_constEvalFuncWrapper_82_0 = utils_constEvalFuncWrapper_82[0]
    const_85 = main_const_eval_85
    util_create_list_303 = [input_36]
    utils_constEvalFuncWrapper_83 = utils.constEvalFuncWrapper(
        const_85, util_create_list_303, CACHED_main_const_eval_85
    )
    CACHED_main_const_eval_85 = utils_constEvalFuncWrapper_83
    utils_constEvalFuncWrapper_83_0 = utils_constEvalFuncWrapper_83[0]
    const_86 = main_const_eval_86
    util_create_list_304 = [input_99]
    utils_constEvalFuncWrapper_84 = utils.constEvalFuncWrapper(
        const_86, util_create_list_304, CACHED_main_const_eval_86
    )
    CACHED_main_const_eval_86 = utils_constEvalFuncWrapper_84
    utils_constEvalFuncWrapper_84_0 = utils_constEvalFuncWrapper_84[0]
    const_87 = main_const_eval_87
    util_create_list_305 = [input_130]
    utils_constEvalFuncWrapper_85 = utils.constEvalFuncWrapper(
        const_87, util_create_list_305, CACHED_main_const_eval_87
    )
    CACHED_main_const_eval_87 = utils_constEvalFuncWrapper_85
    utils_constEvalFuncWrapper_85_0 = utils_constEvalFuncWrapper_85[0]
    const_88 = main_const_eval_88
    util_create_list_306 = [input_92]
    utils_constEvalFuncWrapper_86 = utils.constEvalFuncWrapper(
        const_88, util_create_list_306, CACHED_main_const_eval_88
    )
    CACHED_main_const_eval_88 = utils_constEvalFuncWrapper_86
    utils_constEvalFuncWrapper_86_0 = utils_constEvalFuncWrapper_86[0]
    const_89 = main_const_eval_89
    util_create_list_307 = [input_57]
    utils_constEvalFuncWrapper_87 = utils.constEvalFuncWrapper(
        const_89, util_create_list_307, CACHED_main_const_eval_89
    )
    CACHED_main_const_eval_89 = utils_constEvalFuncWrapper_87
    utils_constEvalFuncWrapper_87_0 = utils_constEvalFuncWrapper_87[0]
    const_90 = main_const_eval_90
    util_create_list_308 = [input_15]
    utils_constEvalFuncWrapper_88 = utils.constEvalFuncWrapper(
        const_90, util_create_list_308, CACHED_main_const_eval_90
    )
    CACHED_main_const_eval_90 = utils_constEvalFuncWrapper_88
    utils_constEvalFuncWrapper_88_0 = utils_constEvalFuncWrapper_88[0]
    const_91 = main_const_eval_91
    util_create_list_309 = [input_29]
    utils_constEvalFuncWrapper_89 = utils.constEvalFuncWrapper(
        const_91, util_create_list_309, CACHED_main_const_eval_91
    )
    CACHED_main_const_eval_91 = utils_constEvalFuncWrapper_89
    utils_constEvalFuncWrapper_89_0 = utils_constEvalFuncWrapper_89[0]
    const_92 = main_const_eval_92
    util_create_list_310 = [input_49]
    utils_constEvalFuncWrapper_90 = utils.constEvalFuncWrapper(
        const_92, util_create_list_310, CACHED_main_const_eval_92
    )
    CACHED_main_const_eval_92 = utils_constEvalFuncWrapper_90
    utils_constEvalFuncWrapper_90_0 = utils_constEvalFuncWrapper_90[0]
    const_93 = main_const_eval_93
    util_create_list_311 = [input_64]
    utils_constEvalFuncWrapper_91 = utils.constEvalFuncWrapper(
        const_93, util_create_list_311, CACHED_main_const_eval_93
    )
    CACHED_main_const_eval_93 = utils_constEvalFuncWrapper_91
    utils_constEvalFuncWrapper_91_0 = utils_constEvalFuncWrapper_91[0]
    const_94 = main_const_eval_94
    util_create_list_312 = [input_29]
    utils_constEvalFuncWrapper_92 = utils.constEvalFuncWrapper(
        const_94, util_create_list_312, CACHED_main_const_eval_94
    )
    CACHED_main_const_eval_94 = utils_constEvalFuncWrapper_92
    utils_constEvalFuncWrapper_92_0 = utils_constEvalFuncWrapper_92[0]
    const_95 = main_const_eval_95
    util_create_list_313 = [input_79]
    utils_constEvalFuncWrapper_93 = utils.constEvalFuncWrapper(
        const_95, util_create_list_313, CACHED_main_const_eval_95
    )
    CACHED_main_const_eval_95 = utils_constEvalFuncWrapper_93
    utils_constEvalFuncWrapper_93_0 = utils_constEvalFuncWrapper_93[0]
    const_96 = main_const_eval_96
    util_create_list_314 = [input_7]
    utils_constEvalFuncWrapper_94 = utils.constEvalFuncWrapper(
        const_96, util_create_list_314, CACHED_main_const_eval_96
    )
    CACHED_main_const_eval_96 = utils_constEvalFuncWrapper_94
    utils_constEvalFuncWrapper_94_0 = utils_constEvalFuncWrapper_94[0]
    const_97 = main_const_eval_97
    util_create_list_315 = [input_107]
    utils_constEvalFuncWrapper_95 = utils.constEvalFuncWrapper(
        const_97, util_create_list_315, CACHED_main_const_eval_97
    )
    CACHED_main_const_eval_97 = utils_constEvalFuncWrapper_95
    utils_constEvalFuncWrapper_95_0 = utils_constEvalFuncWrapper_95[0]
    const_98 = main_const_eval_98
    util_create_list_316 = [input_64]
    utils_constEvalFuncWrapper_96 = utils.constEvalFuncWrapper(
        const_98, util_create_list_316, CACHED_main_const_eval_98
    )
    CACHED_main_const_eval_98 = utils_constEvalFuncWrapper_96
    utils_constEvalFuncWrapper_96_0 = utils_constEvalFuncWrapper_96[0]
    const_99 = main_const_eval_99
    util_create_list_317 = [input_125]
    utils_constEvalFuncWrapper_97 = utils.constEvalFuncWrapper(
        const_99, util_create_list_317, CACHED_main_const_eval_99
    )
    CACHED_main_const_eval_99 = utils_constEvalFuncWrapper_97
    utils_constEvalFuncWrapper_97_0 = utils_constEvalFuncWrapper_97[0]
    const_100 = main_const_eval_100
    util_create_list_318 = [input_3]
    utils_constEvalFuncWrapper_98 = utils.constEvalFuncWrapper(
        const_100, util_create_list_318, CACHED_main_const_eval_100
    )
    CACHED_main_const_eval_100 = utils_constEvalFuncWrapper_98
    utils_constEvalFuncWrapper_98_0 = utils_constEvalFuncWrapper_98[0]
    const_101 = main_const_eval_101
    util_create_list_319 = [input_76]
    utils_constEvalFuncWrapper_99 = utils.constEvalFuncWrapper(
        const_101, util_create_list_319, CACHED_main_const_eval_101
    )
    CACHED_main_const_eval_101 = utils_constEvalFuncWrapper_99
    utils_constEvalFuncWrapper_99_0 = utils_constEvalFuncWrapper_99[0]
    const_102 = main_const_eval_102
    util_create_list_320 = [input_101]
    utils_constEvalFuncWrapper_100 = utils.constEvalFuncWrapper(
        const_102, util_create_list_320, CACHED_main_const_eval_102
    )
    CACHED_main_const_eval_102 = utils_constEvalFuncWrapper_100
    utils_constEvalFuncWrapper_100_0 = utils_constEvalFuncWrapper_100[0]
    const_103 = main_const_eval_103
    util_create_list_321 = [input_37]
    utils_constEvalFuncWrapper_101 = utils.constEvalFuncWrapper(
        const_103, util_create_list_321, CACHED_main_const_eval_103
    )
    CACHED_main_const_eval_103 = utils_constEvalFuncWrapper_101
    utils_constEvalFuncWrapper_101_0 = utils_constEvalFuncWrapper_101[0]
    const_104 = main_const_eval_104
    util_create_list_322 = [input_132]
    utils_constEvalFuncWrapper_102 = utils.constEvalFuncWrapper(
        const_104, util_create_list_322, CACHED_main_const_eval_104
    )
    CACHED_main_const_eval_104 = utils_constEvalFuncWrapper_102
    utils_constEvalFuncWrapper_102_0 = utils_constEvalFuncWrapper_102[0]
    const_105 = main_const_eval_105
    util_create_list_323 = [input_82]
    utils_constEvalFuncWrapper_103 = utils.constEvalFuncWrapper(
        const_105, util_create_list_323, CACHED_main_const_eval_105
    )
    CACHED_main_const_eval_105 = utils_constEvalFuncWrapper_103
    utils_constEvalFuncWrapper_103_0 = utils_constEvalFuncWrapper_103[0]
    const_106 = main_const_eval_106
    util_create_list_324 = [input_136]
    utils_constEvalFuncWrapper_104 = utils.constEvalFuncWrapper(
        const_106, util_create_list_324, CACHED_main_const_eval_106
    )
    CACHED_main_const_eval_106 = utils_constEvalFuncWrapper_104
    utils_constEvalFuncWrapper_104_0 = utils_constEvalFuncWrapper_104[0]
    const_107 = main_const_eval_107
    util_create_list_325 = [input_88]
    utils_constEvalFuncWrapper_105 = utils.constEvalFuncWrapper(
        const_107, util_create_list_325, CACHED_main_const_eval_107
    )
    CACHED_main_const_eval_107 = utils_constEvalFuncWrapper_105
    utils_constEvalFuncWrapper_105_0 = utils_constEvalFuncWrapper_105[0]
    const_108 = main_const_eval_108
    util_create_list_326 = [input_5]
    utils_constEvalFuncWrapper_106 = utils.constEvalFuncWrapper(
        const_108, util_create_list_326, CACHED_main_const_eval_108
    )
    CACHED_main_const_eval_108 = utils_constEvalFuncWrapper_106
    utils_constEvalFuncWrapper_106_0 = utils_constEvalFuncWrapper_106[0]
    const_109 = main_const_eval_109
    util_create_list_327 = [input_94]
    utils_constEvalFuncWrapper_107 = utils.constEvalFuncWrapper(
        const_109, util_create_list_327, CACHED_main_const_eval_109
    )
    CACHED_main_const_eval_109 = utils_constEvalFuncWrapper_107
    utils_constEvalFuncWrapper_107_0 = utils_constEvalFuncWrapper_107[0]
    const_110 = main_const_eval_110
    util_create_list_328 = [input_124]
    utils_constEvalFuncWrapper_108 = utils.constEvalFuncWrapper(
        const_110, util_create_list_328, CACHED_main_const_eval_110
    )
    CACHED_main_const_eval_110 = utils_constEvalFuncWrapper_108
    utils_constEvalFuncWrapper_108_0 = utils_constEvalFuncWrapper_108[0]
    const_111 = main_const_eval_111
    util_create_list_329 = [input_53]
    utils_constEvalFuncWrapper_109 = utils.constEvalFuncWrapper(
        const_111, util_create_list_329, CACHED_main_const_eval_111
    )
    CACHED_main_const_eval_111 = utils_constEvalFuncWrapper_109
    utils_constEvalFuncWrapper_109_0 = utils_constEvalFuncWrapper_109[0]
    const_112 = main_const_eval_112
    util_create_list_330 = [input_63]
    utils_constEvalFuncWrapper_110 = utils.constEvalFuncWrapper(
        const_112, util_create_list_330, CACHED_main_const_eval_112
    )
    CACHED_main_const_eval_112 = utils_constEvalFuncWrapper_110
    utils_constEvalFuncWrapper_110_0 = utils_constEvalFuncWrapper_110[0]
    const_113 = main_const_eval_113
    util_create_list_331 = [input_25]
    utils_constEvalFuncWrapper_111 = utils.constEvalFuncWrapper(
        const_113, util_create_list_331, CACHED_main_const_eval_113
    )
    CACHED_main_const_eval_113 = utils_constEvalFuncWrapper_111
    utils_constEvalFuncWrapper_111_0 = utils_constEvalFuncWrapper_111[0]
    const_114 = main_const_eval_114
    util_create_list_332 = [input_103]
    utils_constEvalFuncWrapper_112 = utils.constEvalFuncWrapper(
        const_114, util_create_list_332, CACHED_main_const_eval_114
    )
    CACHED_main_const_eval_114 = utils_constEvalFuncWrapper_112
    utils_constEvalFuncWrapper_112_0 = utils_constEvalFuncWrapper_112[0]
    const_115 = main_const_eval_115
    util_create_list_333 = [input_9]
    utils_constEvalFuncWrapper_113 = utils.constEvalFuncWrapper(
        const_115, util_create_list_333, CACHED_main_const_eval_115
    )
    CACHED_main_const_eval_115 = utils_constEvalFuncWrapper_113
    utils_constEvalFuncWrapper_113_0 = utils_constEvalFuncWrapper_113[0]
    const_116 = main_const_eval_116
    util_create_list_334 = [input_49]
    utils_constEvalFuncWrapper_114 = utils.constEvalFuncWrapper(
        const_116, util_create_list_334, CACHED_main_const_eval_116
    )
    CACHED_main_const_eval_116 = utils_constEvalFuncWrapper_114
    utils_constEvalFuncWrapper_114_0 = utils_constEvalFuncWrapper_114[0]
    const_117 = main_const_eval_117
    util_create_list_335 = [input_33]
    utils_constEvalFuncWrapper_115 = utils.constEvalFuncWrapper(
        const_117, util_create_list_335, CACHED_main_const_eval_117
    )
    CACHED_main_const_eval_117 = utils_constEvalFuncWrapper_115
    utils_constEvalFuncWrapper_115_0 = utils_constEvalFuncWrapper_115[0]
    const_118 = main_const_eval_118
    util_create_list_336 = [input_21]
    utils_constEvalFuncWrapper_116 = utils.constEvalFuncWrapper(
        const_118, util_create_list_336, CACHED_main_const_eval_118
    )
    CACHED_main_const_eval_118 = utils_constEvalFuncWrapper_116
    utils_constEvalFuncWrapper_116_0 = utils_constEvalFuncWrapper_116[0]
    const_119 = main_const_eval_119
    util_create_list_337 = [input_110]
    utils_constEvalFuncWrapper_117 = utils.constEvalFuncWrapper(
        const_119, util_create_list_337, CACHED_main_const_eval_119
    )
    CACHED_main_const_eval_119 = utils_constEvalFuncWrapper_117
    utils_constEvalFuncWrapper_117_0 = utils_constEvalFuncWrapper_117[0]
    const_120 = main_const_eval_120
    util_create_list_338 = [input_105]
    utils_constEvalFuncWrapper_118 = utils.constEvalFuncWrapper(
        const_120, util_create_list_338, CACHED_main_const_eval_120
    )
    CACHED_main_const_eval_120 = utils_constEvalFuncWrapper_118
    utils_constEvalFuncWrapper_118_0 = utils_constEvalFuncWrapper_118[0]
    const_121 = main_const_eval_121
    util_create_list_339 = [input_107]
    utils_constEvalFuncWrapper_119 = utils.constEvalFuncWrapper(
        const_121, util_create_list_339, CACHED_main_const_eval_121
    )
    CACHED_main_const_eval_121 = utils_constEvalFuncWrapper_119
    utils_constEvalFuncWrapper_119_0 = utils_constEvalFuncWrapper_119[0]
    const_122 = main_const_eval_122
    util_create_list_340 = [input_24]
    utils_constEvalFuncWrapper_120 = utils.constEvalFuncWrapper(
        const_122, util_create_list_340, CACHED_main_const_eval_122
    )
    CACHED_main_const_eval_122 = utils_constEvalFuncWrapper_120
    utils_constEvalFuncWrapper_120_0 = utils_constEvalFuncWrapper_120[0]
    const_123 = main_const_eval_123
    util_create_list_341 = [input_98]
    utils_constEvalFuncWrapper_121 = utils.constEvalFuncWrapper(
        const_123, util_create_list_341, CACHED_main_const_eval_123
    )
    CACHED_main_const_eval_123 = utils_constEvalFuncWrapper_121
    utils_constEvalFuncWrapper_121_0 = utils_constEvalFuncWrapper_121[0]
    const_124 = main_const_eval_124
    util_create_list_342 = [input_1]
    utils_constEvalFuncWrapper_122 = utils.constEvalFuncWrapper(
        const_124, util_create_list_342, CACHED_main_const_eval_124
    )
    CACHED_main_const_eval_124 = utils_constEvalFuncWrapper_122
    utils_constEvalFuncWrapper_122_0 = utils_constEvalFuncWrapper_122[0]
    const_125 = main_const_eval_125
    util_create_list_343 = [input_84]
    utils_constEvalFuncWrapper_123 = utils.constEvalFuncWrapper(
        const_125, util_create_list_343, CACHED_main_const_eval_125
    )
    CACHED_main_const_eval_125 = utils_constEvalFuncWrapper_123
    utils_constEvalFuncWrapper_123_0 = utils_constEvalFuncWrapper_123[0]
    const_126 = main_const_eval_126
    util_create_list_344 = [input_66]
    utils_constEvalFuncWrapper_124 = utils.constEvalFuncWrapper(
        const_126, util_create_list_344, CACHED_main_const_eval_126
    )
    CACHED_main_const_eval_126 = utils_constEvalFuncWrapper_124
    utils_constEvalFuncWrapper_124_0 = utils_constEvalFuncWrapper_124[0]
    const_127 = main_const_eval_127
    util_create_list_345 = [input_46]
    utils_constEvalFuncWrapper_125 = utils.constEvalFuncWrapper(
        const_127, util_create_list_345, CACHED_main_const_eval_127
    )
    CACHED_main_const_eval_127 = utils_constEvalFuncWrapper_125
    utils_constEvalFuncWrapper_125_0 = utils_constEvalFuncWrapper_125[0]
    const_128 = main_const_eval_128
    util_create_list_346 = [input_56]
    utils_constEvalFuncWrapper_126 = utils.constEvalFuncWrapper(
        const_128, util_create_list_346, CACHED_main_const_eval_128
    )
    CACHED_main_const_eval_128 = utils_constEvalFuncWrapper_126
    utils_constEvalFuncWrapper_126_0 = utils_constEvalFuncWrapper_126[0]
    const_129 = main_const_eval_129
    util_create_list_347 = [input_0]
    utils_constEvalFuncWrapper_127 = utils.constEvalFuncWrapper(
        const_129, util_create_list_347, CACHED_main_const_eval_129
    )
    CACHED_main_const_eval_129 = utils_constEvalFuncWrapper_127
    utils_constEvalFuncWrapper_127_0 = utils_constEvalFuncWrapper_127[0]
    const_130 = main_const_eval_130
    util_create_list_348 = [input_60]
    utils_constEvalFuncWrapper_128 = utils.constEvalFuncWrapper(
        const_130, util_create_list_348, CACHED_main_const_eval_130
    )
    CACHED_main_const_eval_130 = utils_constEvalFuncWrapper_128
    utils_constEvalFuncWrapper_128_0 = utils_constEvalFuncWrapper_128[0]
    const_131 = main_const_eval_131
    util_create_list_349 = [input_57]
    utils_constEvalFuncWrapper_129 = utils.constEvalFuncWrapper(
        const_131, util_create_list_349, CACHED_main_const_eval_131
    )
    CACHED_main_const_eval_131 = utils_constEvalFuncWrapper_129
    utils_constEvalFuncWrapper_129_0 = utils_constEvalFuncWrapper_129[0]
    const_132 = main_const_eval_132
    util_create_list_350 = [input_93]
    utils_constEvalFuncWrapper_130 = utils.constEvalFuncWrapper(
        const_132, util_create_list_350, CACHED_main_const_eval_132
    )
    CACHED_main_const_eval_132 = utils_constEvalFuncWrapper_130
    utils_constEvalFuncWrapper_130_0 = utils_constEvalFuncWrapper_130[0]
    const_133 = main_const_eval_133
    util_create_list_351 = [input_19]
    utils_constEvalFuncWrapper_131 = utils.constEvalFuncWrapper(
        const_133, util_create_list_351, CACHED_main_const_eval_133
    )
    CACHED_main_const_eval_133 = utils_constEvalFuncWrapper_131
    utils_constEvalFuncWrapper_131_0 = utils_constEvalFuncWrapper_131[0]
    const_134 = main_const_eval_134
    util_create_list_352 = [input_77]
    utils_constEvalFuncWrapper_132 = utils.constEvalFuncWrapper(
        const_134, util_create_list_352, CACHED_main_const_eval_134
    )
    CACHED_main_const_eval_134 = utils_constEvalFuncWrapper_132
    utils_constEvalFuncWrapper_132_0 = utils_constEvalFuncWrapper_132[0]
    const_135 = main_const_eval_135
    util_create_list_353 = [input_23]
    utils_constEvalFuncWrapper_133 = utils.constEvalFuncWrapper(
        const_135, util_create_list_353, CACHED_main_const_eval_135
    )
    CACHED_main_const_eval_135 = utils_constEvalFuncWrapper_133
    utils_constEvalFuncWrapper_133_0 = utils_constEvalFuncWrapper_133[0]
    const_136 = main_const_eval_136
    util_create_list_354 = [input_126]
    utils_constEvalFuncWrapper_134 = utils.constEvalFuncWrapper(
        const_136, util_create_list_354, CACHED_main_const_eval_136
    )
    CACHED_main_const_eval_136 = utils_constEvalFuncWrapper_134
    utils_constEvalFuncWrapper_134_0 = utils_constEvalFuncWrapper_134[0]
    const_137 = main_const_eval_137
    util_create_list_355 = [input_13]
    utils_constEvalFuncWrapper_135 = utils.constEvalFuncWrapper(
        const_137, util_create_list_355, CACHED_main_const_eval_137
    )
    CACHED_main_const_eval_137 = utils_constEvalFuncWrapper_135
    utils_constEvalFuncWrapper_135_0 = utils_constEvalFuncWrapper_135[0]
    const_138 = main_const_eval_138
    util_create_list_356 = [input_101]
    utils_constEvalFuncWrapper_136 = utils.constEvalFuncWrapper(
        const_138, util_create_list_356, CACHED_main_const_eval_138
    )
    CACHED_main_const_eval_138 = utils_constEvalFuncWrapper_136
    utils_constEvalFuncWrapper_136_0 = utils_constEvalFuncWrapper_136[0]
    const_139 = main_const_eval_139
    util_create_list_357 = [input_25]
    utils_constEvalFuncWrapper_137 = utils.constEvalFuncWrapper(
        const_139, util_create_list_357, CACHED_main_const_eval_139
    )
    CACHED_main_const_eval_139 = utils_constEvalFuncWrapper_137
    utils_constEvalFuncWrapper_137_0 = utils_constEvalFuncWrapper_137[0]
    const_140 = main_const_eval_140
    util_create_list_358 = [input_40]
    utils_constEvalFuncWrapper_138 = utils.constEvalFuncWrapper(
        const_140, util_create_list_358, CACHED_main_const_eval_140
    )
    CACHED_main_const_eval_140 = utils_constEvalFuncWrapper_138
    utils_constEvalFuncWrapper_138_0 = utils_constEvalFuncWrapper_138[0]
    const_141 = main_const_eval_141
    util_create_list_359 = [input_35]
    utils_constEvalFuncWrapper_139 = utils.constEvalFuncWrapper(
        const_141, util_create_list_359, CACHED_main_const_eval_141
    )
    CACHED_main_const_eval_141 = utils_constEvalFuncWrapper_139
    utils_constEvalFuncWrapper_139_0 = utils_constEvalFuncWrapper_139[0]
    const_142 = main_const_eval_142
    util_create_list_360 = [input_62]
    utils_constEvalFuncWrapper_140 = utils.constEvalFuncWrapper(
        const_142, util_create_list_360, CACHED_main_const_eval_142
    )
    CACHED_main_const_eval_142 = utils_constEvalFuncWrapper_140
    utils_constEvalFuncWrapper_140_0 = utils_constEvalFuncWrapper_140[0]
    const_143 = main_const_eval_143
    util_create_list_361 = [input_89]
    utils_constEvalFuncWrapper_141 = utils.constEvalFuncWrapper(
        const_143, util_create_list_361, CACHED_main_const_eval_143
    )
    CACHED_main_const_eval_143 = utils_constEvalFuncWrapper_141
    utils_constEvalFuncWrapper_141_0 = utils_constEvalFuncWrapper_141[0]
    const_144 = main_const_eval_144
    util_create_list_362 = [input_51]
    utils_constEvalFuncWrapper_142 = utils.constEvalFuncWrapper(
        const_144, util_create_list_362, CACHED_main_const_eval_144
    )
    CACHED_main_const_eval_144 = utils_constEvalFuncWrapper_142
    utils_constEvalFuncWrapper_142_0 = utils_constEvalFuncWrapper_142[0]
    const_145 = main_const_eval_145
    util_create_list_363 = [input_99]
    utils_constEvalFuncWrapper_143 = utils.constEvalFuncWrapper(
        const_145, util_create_list_363, CACHED_main_const_eval_145
    )
    CACHED_main_const_eval_145 = utils_constEvalFuncWrapper_143
    utils_constEvalFuncWrapper_143_0 = utils_constEvalFuncWrapper_143[0]
    const_146 = main_const_eval_146
    util_create_list_364 = [input_47]
    utils_constEvalFuncWrapper_144 = utils.constEvalFuncWrapper(
        const_146, util_create_list_364, CACHED_main_const_eval_146
    )
    CACHED_main_const_eval_146 = utils_constEvalFuncWrapper_144
    utils_constEvalFuncWrapper_144_0 = utils_constEvalFuncWrapper_144[0]
    const_147 = main_const_eval_147
    util_create_list_365 = [input_118]
    utils_constEvalFuncWrapper_145 = utils.constEvalFuncWrapper(
        const_147, util_create_list_365, CACHED_main_const_eval_147
    )
    CACHED_main_const_eval_147 = utils_constEvalFuncWrapper_145
    utils_constEvalFuncWrapper_145_0 = utils_constEvalFuncWrapper_145[0]
    const_148 = main_const_eval_148
    util_create_list_366 = [input_28]
    utils_constEvalFuncWrapper_146 = utils.constEvalFuncWrapper(
        const_148, util_create_list_366, CACHED_main_const_eval_148
    )
    CACHED_main_const_eval_148 = utils_constEvalFuncWrapper_146
    utils_constEvalFuncWrapper_146_0 = utils_constEvalFuncWrapper_146[0]
    const_149 = main_const_eval_149
    util_create_list_367 = [input_62]
    utils_constEvalFuncWrapper_147 = utils.constEvalFuncWrapper(
        const_149, util_create_list_367, CACHED_main_const_eval_149
    )
    CACHED_main_const_eval_149 = utils_constEvalFuncWrapper_147
    utils_constEvalFuncWrapper_147_0 = utils_constEvalFuncWrapper_147[0]
    const_150 = main_const_eval_150
    util_create_list_368 = [input_73]
    utils_constEvalFuncWrapper_148 = utils.constEvalFuncWrapper(
        const_150, util_create_list_368, CACHED_main_const_eval_150
    )
    CACHED_main_const_eval_150 = utils_constEvalFuncWrapper_148
    utils_constEvalFuncWrapper_148_0 = utils_constEvalFuncWrapper_148[0]
    const_151 = main_const_eval_151
    util_create_list_369 = [input_121]
    utils_constEvalFuncWrapper_149 = utils.constEvalFuncWrapper(
        const_151, util_create_list_369, CACHED_main_const_eval_151
    )
    CACHED_main_const_eval_151 = utils_constEvalFuncWrapper_149
    utils_constEvalFuncWrapper_149_0 = utils_constEvalFuncWrapper_149[0]
    const_152 = main_const_eval_152
    util_create_list_370 = [input_58]
    utils_constEvalFuncWrapper_150 = utils.constEvalFuncWrapper(
        const_152, util_create_list_370, CACHED_main_const_eval_152
    )
    CACHED_main_const_eval_152 = utils_constEvalFuncWrapper_150
    utils_constEvalFuncWrapper_150_0 = utils_constEvalFuncWrapper_150[0]
    const_153 = main_const_eval_153
    util_create_list_371 = [input_9]
    utils_constEvalFuncWrapper_151 = utils.constEvalFuncWrapper(
        const_153, util_create_list_371, CACHED_main_const_eval_153
    )
    CACHED_main_const_eval_153 = utils_constEvalFuncWrapper_151
    utils_constEvalFuncWrapper_151_0 = utils_constEvalFuncWrapper_151[0]
    const_154 = main_const_eval_154
    util_create_list_372 = [input_6]
    utils_constEvalFuncWrapper_152 = utils.constEvalFuncWrapper(
        const_154, util_create_list_372, CACHED_main_const_eval_154
    )
    CACHED_main_const_eval_154 = utils_constEvalFuncWrapper_152
    utils_constEvalFuncWrapper_152_0 = utils_constEvalFuncWrapper_152[0]
    const_155 = main_const_eval_155
    util_create_list_373 = [input_82]
    utils_constEvalFuncWrapper_153 = utils.constEvalFuncWrapper(
        const_155, util_create_list_373, CACHED_main_const_eval_155
    )
    CACHED_main_const_eval_155 = utils_constEvalFuncWrapper_153
    utils_constEvalFuncWrapper_153_0 = utils_constEvalFuncWrapper_153[0]
    const_156 = main_const_eval_156
    util_create_list_374 = [input_47]
    utils_constEvalFuncWrapper_154 = utils.constEvalFuncWrapper(
        const_156, util_create_list_374, CACHED_main_const_eval_156
    )
    CACHED_main_const_eval_156 = utils_constEvalFuncWrapper_154
    utils_constEvalFuncWrapper_154_0 = utils_constEvalFuncWrapper_154[0]
    const_157 = main_const_eval_157
    util_create_list_375 = [input_18]
    utils_constEvalFuncWrapper_155 = utils.constEvalFuncWrapper(
        const_157, util_create_list_375, CACHED_main_const_eval_157
    )
    CACHED_main_const_eval_157 = utils_constEvalFuncWrapper_155
    utils_constEvalFuncWrapper_155_0 = utils_constEvalFuncWrapper_155[0]
    const_158 = main_const_eval_158
    util_create_list_376 = [input_19]
    utils_constEvalFuncWrapper_156 = utils.constEvalFuncWrapper(
        const_158, util_create_list_376, CACHED_main_const_eval_158
    )
    CACHED_main_const_eval_158 = utils_constEvalFuncWrapper_156
    utils_constEvalFuncWrapper_156_0 = utils_constEvalFuncWrapper_156[0]
    const_159 = main_const_eval_159
    util_create_list_377 = [input_117]
    utils_constEvalFuncWrapper_157 = utils.constEvalFuncWrapper(
        const_159, util_create_list_377, CACHED_main_const_eval_159
    )
    CACHED_main_const_eval_159 = utils_constEvalFuncWrapper_157
    utils_constEvalFuncWrapper_157_0 = utils_constEvalFuncWrapper_157[0]
    const_160 = main_const_eval_160
    util_create_list_378 = [input_84]
    utils_constEvalFuncWrapper_158 = utils.constEvalFuncWrapper(
        const_160, util_create_list_378, CACHED_main_const_eval_160
    )
    CACHED_main_const_eval_160 = utils_constEvalFuncWrapper_158
    utils_constEvalFuncWrapper_158_0 = utils_constEvalFuncWrapper_158[0]
    const_161 = main_const_eval_161
    util_create_list_379 = [input_43]
    utils_constEvalFuncWrapper_159 = utils.constEvalFuncWrapper(
        const_161, util_create_list_379, CACHED_main_const_eval_161
    )
    CACHED_main_const_eval_161 = utils_constEvalFuncWrapper_159
    utils_constEvalFuncWrapper_159_0 = utils_constEvalFuncWrapper_159[0]
    const_162 = main_const_eval_162
    util_create_list_380 = [input_51]
    utils_constEvalFuncWrapper_160 = utils.constEvalFuncWrapper(
        const_162, util_create_list_380, CACHED_main_const_eval_162
    )
    CACHED_main_const_eval_162 = utils_constEvalFuncWrapper_160
    utils_constEvalFuncWrapper_160_0 = utils_constEvalFuncWrapper_160[0]
    const_163 = main_const_eval_163
    util_create_list_381 = [input_74]
    utils_constEvalFuncWrapper_161 = utils.constEvalFuncWrapper(
        const_163, util_create_list_381, CACHED_main_const_eval_163
    )
    CACHED_main_const_eval_163 = utils_constEvalFuncWrapper_161
    utils_constEvalFuncWrapper_161_0 = utils_constEvalFuncWrapper_161[0]
    const_164 = main_const_eval_164
    util_create_list_382 = [input_80]
    utils_constEvalFuncWrapper_162 = utils.constEvalFuncWrapper(
        const_164, util_create_list_382, CACHED_main_const_eval_164
    )
    CACHED_main_const_eval_164 = utils_constEvalFuncWrapper_162
    utils_constEvalFuncWrapper_162_0 = utils_constEvalFuncWrapper_162[0]
    const_165 = main_const_eval_165
    util_create_list_383 = [input_73]
    utils_constEvalFuncWrapper_163 = utils.constEvalFuncWrapper(
        const_165, util_create_list_383, CACHED_main_const_eval_165
    )
    CACHED_main_const_eval_165 = utils_constEvalFuncWrapper_163
    utils_constEvalFuncWrapper_163_0 = utils_constEvalFuncWrapper_163[0]
    const_166 = main_const_eval_166
    util_create_list_384 = [input_3]
    utils_constEvalFuncWrapper_164 = utils.constEvalFuncWrapper(
        const_166, util_create_list_384, CACHED_main_const_eval_166
    )
    CACHED_main_const_eval_166 = utils_constEvalFuncWrapper_164
    utils_constEvalFuncWrapper_164_0 = utils_constEvalFuncWrapper_164[0]
    const_167 = main_const_eval_167
    util_create_list_385 = [input_81]
    utils_constEvalFuncWrapper_165 = utils.constEvalFuncWrapper(
        const_167, util_create_list_385, CACHED_main_const_eval_167
    )
    CACHED_main_const_eval_167 = utils_constEvalFuncWrapper_165
    utils_constEvalFuncWrapper_165_0 = utils_constEvalFuncWrapper_165[0]
    const_168 = main_const_eval_168
    util_create_list_386 = [input_39]
    utils_constEvalFuncWrapper_166 = utils.constEvalFuncWrapper(
        const_168, util_create_list_386, CACHED_main_const_eval_168
    )
    CACHED_main_const_eval_168 = utils_constEvalFuncWrapper_166
    utils_constEvalFuncWrapper_166_0 = utils_constEvalFuncWrapper_166[0]
    const_169 = main_const_eval_169
    util_create_list_387 = [input_102]
    utils_constEvalFuncWrapper_167 = utils.constEvalFuncWrapper(
        const_169, util_create_list_387, CACHED_main_const_eval_169
    )
    CACHED_main_const_eval_169 = utils_constEvalFuncWrapper_167
    utils_constEvalFuncWrapper_167_0 = utils_constEvalFuncWrapper_167[0]
    const_170 = main_const_eval_170
    util_create_list_388 = [input_23]
    utils_constEvalFuncWrapper_168 = utils.constEvalFuncWrapper(
        const_170, util_create_list_388, CACHED_main_const_eval_170
    )
    CACHED_main_const_eval_170 = utils_constEvalFuncWrapper_168
    utils_constEvalFuncWrapper_168_0 = utils_constEvalFuncWrapper_168[0]
    const_171 = main_const_eval_171
    util_create_list_389 = [input_44]
    utils_constEvalFuncWrapper_169 = utils.constEvalFuncWrapper(
        const_171, util_create_list_389, CACHED_main_const_eval_171
    )
    CACHED_main_const_eval_171 = utils_constEvalFuncWrapper_169
    utils_constEvalFuncWrapper_169_0 = utils_constEvalFuncWrapper_169[0]
    const_172 = main_const_eval_172
    util_create_list_390 = [input_135]
    utils_constEvalFuncWrapper_170 = utils.constEvalFuncWrapper(
        const_172, util_create_list_390, CACHED_main_const_eval_172
    )
    CACHED_main_const_eval_172 = utils_constEvalFuncWrapper_170
    utils_constEvalFuncWrapper_170_0 = utils_constEvalFuncWrapper_170[0]
    const_173 = main_const_eval_173
    util_create_list_391 = [input_27]
    utils_constEvalFuncWrapper_171 = utils.constEvalFuncWrapper(
        const_173, util_create_list_391, CACHED_main_const_eval_173
    )
    CACHED_main_const_eval_173 = utils_constEvalFuncWrapper_171
    utils_constEvalFuncWrapper_171_0 = utils_constEvalFuncWrapper_171[0]
    const_174 = main_const_eval_174
    util_create_list_392 = [input_7]
    utils_constEvalFuncWrapper_172 = utils.constEvalFuncWrapper(
        const_174, util_create_list_392, CACHED_main_const_eval_174
    )
    CACHED_main_const_eval_174 = utils_constEvalFuncWrapper_172
    utils_constEvalFuncWrapper_172_0 = utils_constEvalFuncWrapper_172[0]
    const_175 = main_const_eval_175
    util_create_list_393 = [input_119]
    utils_constEvalFuncWrapper_173 = utils.constEvalFuncWrapper(
        const_175, util_create_list_393, CACHED_main_const_eval_175
    )
    CACHED_main_const_eval_175 = utils_constEvalFuncWrapper_173
    utils_constEvalFuncWrapper_173_0 = utils_constEvalFuncWrapper_173[0]
    const_176 = main_const_eval_176
    util_create_list_394 = [input_55]
    utils_constEvalFuncWrapper_174 = utils.constEvalFuncWrapper(
        const_176, util_create_list_394, CACHED_main_const_eval_176
    )
    CACHED_main_const_eval_176 = utils_constEvalFuncWrapper_174
    utils_constEvalFuncWrapper_174_0 = utils_constEvalFuncWrapper_174[0]
    const_177 = main_const_eval_177
    util_create_list_395 = [input_53]
    utils_constEvalFuncWrapper_175 = utils.constEvalFuncWrapper(
        const_177, util_create_list_395, CACHED_main_const_eval_177
    )
    CACHED_main_const_eval_177 = utils_constEvalFuncWrapper_175
    utils_constEvalFuncWrapper_175_0 = utils_constEvalFuncWrapper_175[0]
    const_178 = main_const_eval_178
    util_create_list_396 = [input_33]
    utils_constEvalFuncWrapper_176 = utils.constEvalFuncWrapper(
        const_178, util_create_list_396, CACHED_main_const_eval_178
    )
    CACHED_main_const_eval_178 = utils_constEvalFuncWrapper_176
    utils_constEvalFuncWrapper_176_0 = utils_constEvalFuncWrapper_176[0]
    const_179 = main_const_eval_179
    util_create_list_397 = [input_2]
    utils_constEvalFuncWrapper_177 = utils.constEvalFuncWrapper(
        const_179, util_create_list_397, CACHED_main_const_eval_179
    )
    CACHED_main_const_eval_179 = utils_constEvalFuncWrapper_177
    utils_constEvalFuncWrapper_177_0 = utils_constEvalFuncWrapper_177[0]
    const_180 = main_const_eval_180
    util_create_list_398 = [input_35]
    utils_constEvalFuncWrapper_178 = utils.constEvalFuncWrapper(
        const_180, util_create_list_398, CACHED_main_const_eval_180
    )
    CACHED_main_const_eval_180 = utils_constEvalFuncWrapper_178
    utils_constEvalFuncWrapper_178_0 = utils_constEvalFuncWrapper_178[0]
    const_181 = main_const_eval_181
    util_create_list_399 = [input_81]
    utils_constEvalFuncWrapper_179 = utils.constEvalFuncWrapper(
        const_181, util_create_list_399, CACHED_main_const_eval_181
    )
    CACHED_main_const_eval_181 = utils_constEvalFuncWrapper_179
    utils_constEvalFuncWrapper_179_0 = utils_constEvalFuncWrapper_179[0]
    const_182 = main_const_eval_182
    util_create_list_400 = [input_95]
    utils_constEvalFuncWrapper_180 = utils.constEvalFuncWrapper(
        const_182, util_create_list_400, CACHED_main_const_eval_182
    )
    CACHED_main_const_eval_182 = utils_constEvalFuncWrapper_180
    utils_constEvalFuncWrapper_180_0 = utils_constEvalFuncWrapper_180[0]
    const_183 = main_const_eval_183
    util_create_list_401 = [input_41]
    utils_constEvalFuncWrapper_181 = utils.constEvalFuncWrapper(
        const_183, util_create_list_401, CACHED_main_const_eval_183
    )
    CACHED_main_const_eval_183 = utils_constEvalFuncWrapper_181
    utils_constEvalFuncWrapper_181_0 = utils_constEvalFuncWrapper_181[0]
    const_184 = main_const_eval_184
    util_create_list_402 = [input_112]
    utils_constEvalFuncWrapper_182 = utils.constEvalFuncWrapper(
        const_184, util_create_list_402, CACHED_main_const_eval_184
    )
    CACHED_main_const_eval_184 = utils_constEvalFuncWrapper_182
    utils_constEvalFuncWrapper_182_0 = utils_constEvalFuncWrapper_182[0]
    const_185 = main_const_eval_185
    util_create_list_403 = [input_1]
    utils_constEvalFuncWrapper_183 = utils.constEvalFuncWrapper(
        const_185, util_create_list_403, CACHED_main_const_eval_185
    )
    CACHED_main_const_eval_185 = utils_constEvalFuncWrapper_183
    utils_constEvalFuncWrapper_183_0 = utils_constEvalFuncWrapper_183[0]
    const_186 = main_const_eval_186
    util_create_list_404 = [input_85]
    utils_constEvalFuncWrapper_184 = utils.constEvalFuncWrapper(
        const_186, util_create_list_404, CACHED_main_const_eval_186
    )
    CACHED_main_const_eval_186 = utils_constEvalFuncWrapper_184
    utils_constEvalFuncWrapper_184_0 = utils_constEvalFuncWrapper_184[0]
    const_187 = main_const_eval_187
    util_create_list_405 = [input_104]
    utils_constEvalFuncWrapper_185 = utils.constEvalFuncWrapper(
        const_187, util_create_list_405, CACHED_main_const_eval_187
    )
    CACHED_main_const_eval_187 = utils_constEvalFuncWrapper_185
    utils_constEvalFuncWrapper_185_0 = utils_constEvalFuncWrapper_185[0]
    const_188 = main_const_eval_188
    util_create_list_406 = [input_66]
    utils_constEvalFuncWrapper_186 = utils.constEvalFuncWrapper(
        const_188, util_create_list_406, CACHED_main_const_eval_188
    )
    CACHED_main_const_eval_188 = utils_constEvalFuncWrapper_186
    utils_constEvalFuncWrapper_186_0 = utils_constEvalFuncWrapper_186[0]
    const_189 = main_const_eval_189
    util_create_list_407 = [input_56]
    utils_constEvalFuncWrapper_187 = utils.constEvalFuncWrapper(
        const_189, util_create_list_407, CACHED_main_const_eval_189
    )
    CACHED_main_const_eval_189 = utils_constEvalFuncWrapper_187
    utils_constEvalFuncWrapper_187_0 = utils_constEvalFuncWrapper_187[0]
    const_190 = main_const_eval_190
    util_create_list_408 = [input_90]
    utils_constEvalFuncWrapper_188 = utils.constEvalFuncWrapper(
        const_190, util_create_list_408, CACHED_main_const_eval_190
    )
    CACHED_main_const_eval_190 = utils_constEvalFuncWrapper_188
    utils_constEvalFuncWrapper_188_0 = utils_constEvalFuncWrapper_188[0]
    const_191 = main_const_eval_191
    util_create_list_409 = [input_60]
    utils_constEvalFuncWrapper_189 = utils.constEvalFuncWrapper(
        const_191, util_create_list_409, CACHED_main_const_eval_191
    )
    CACHED_main_const_eval_191 = utils_constEvalFuncWrapper_189
    utils_constEvalFuncWrapper_189_0 = utils_constEvalFuncWrapper_189[0]
    const_192 = main_const_eval_192
    util_create_list_410 = [input_80]
    utils_constEvalFuncWrapper_190 = utils.constEvalFuncWrapper(
        const_192, util_create_list_410, CACHED_main_const_eval_192
    )
    CACHED_main_const_eval_192 = utils_constEvalFuncWrapper_190
    utils_constEvalFuncWrapper_190_0 = utils_constEvalFuncWrapper_190[0]
    const_193 = main_const_eval_193
    util_create_list_411 = [input_72]
    utils_constEvalFuncWrapper_191 = utils.constEvalFuncWrapper(
        const_193, util_create_list_411, CACHED_main_const_eval_193
    )
    CACHED_main_const_eval_193 = utils_constEvalFuncWrapper_191
    utils_constEvalFuncWrapper_191_0 = utils_constEvalFuncWrapper_191[0]
    const_194 = main_const_eval_194
    util_create_list_412 = [input_76]
    utils_constEvalFuncWrapper_192 = utils.constEvalFuncWrapper(
        const_194, util_create_list_412, CACHED_main_const_eval_194
    )
    CACHED_main_const_eval_194 = utils_constEvalFuncWrapper_192
    utils_constEvalFuncWrapper_192_0 = utils_constEvalFuncWrapper_192[0]
    const_195 = main_const_eval_195
    util_create_list_413 = [input_68]
    utils_constEvalFuncWrapper_193 = utils.constEvalFuncWrapper(
        const_195, util_create_list_413, CACHED_main_const_eval_195
    )
    CACHED_main_const_eval_195 = utils_constEvalFuncWrapper_193
    utils_constEvalFuncWrapper_193_0 = utils_constEvalFuncWrapper_193[0]
    const_196 = main_const_eval_196
    util_create_list_414 = [input_91]
    utils_constEvalFuncWrapper_194 = utils.constEvalFuncWrapper(
        const_196, util_create_list_414, CACHED_main_const_eval_196
    )
    CACHED_main_const_eval_196 = utils_constEvalFuncWrapper_194
    utils_constEvalFuncWrapper_194_0 = utils_constEvalFuncWrapper_194[0]
    const_197 = main_const_eval_197
    util_create_list_415 = [input_68]
    utils_constEvalFuncWrapper_195 = utils.constEvalFuncWrapper(
        const_197, util_create_list_415, CACHED_main_const_eval_197
    )
    CACHED_main_const_eval_197 = utils_constEvalFuncWrapper_195
    utils_constEvalFuncWrapper_195_0 = utils_constEvalFuncWrapper_195[0]
    const_198 = main_const_eval_198
    util_create_list_416 = [input_78]
    utils_constEvalFuncWrapper_196 = utils.constEvalFuncWrapper(
        const_198, util_create_list_416, CACHED_main_const_eval_198
    )
    CACHED_main_const_eval_198 = utils_constEvalFuncWrapper_196
    utils_constEvalFuncWrapper_196_0 = utils_constEvalFuncWrapper_196[0]
    const_199 = main_const_eval_199
    util_create_list_417 = [input_30]
    utils_constEvalFuncWrapper_197 = utils.constEvalFuncWrapper(
        const_199, util_create_list_417, CACHED_main_const_eval_199
    )
    CACHED_main_const_eval_199 = utils_constEvalFuncWrapper_197
    utils_constEvalFuncWrapper_197_0 = utils_constEvalFuncWrapper_197[0]
    const_200 = main_const_eval_200
    util_create_list_418 = [input_93]
    utils_constEvalFuncWrapper_198 = utils.constEvalFuncWrapper(
        const_200, util_create_list_418, CACHED_main_const_eval_200
    )
    CACHED_main_const_eval_200 = utils_constEvalFuncWrapper_198
    utils_constEvalFuncWrapper_198_0 = utils_constEvalFuncWrapper_198[0]
    const_201 = main_const_eval_201
    util_create_list_419 = [input_89]
    utils_constEvalFuncWrapper_199 = utils.constEvalFuncWrapper(
        const_201, util_create_list_419, CACHED_main_const_eval_201
    )
    CACHED_main_const_eval_201 = utils_constEvalFuncWrapper_199
    utils_constEvalFuncWrapper_199_0 = utils_constEvalFuncWrapper_199[0]
    const_202 = main_const_eval_202
    util_create_list_420 = [input_96]
    utils_constEvalFuncWrapper_200 = utils.constEvalFuncWrapper(
        const_202, util_create_list_420, CACHED_main_const_eval_202
    )
    CACHED_main_const_eval_202 = utils_constEvalFuncWrapper_200
    utils_constEvalFuncWrapper_200_0 = utils_constEvalFuncWrapper_200[0]
    const_203 = main_const_eval_203
    util_create_list_421 = [input_4]
    utils_constEvalFuncWrapper_201 = utils.constEvalFuncWrapper(
        const_203, util_create_list_421, CACHED_main_const_eval_203
    )
    CACHED_main_const_eval_203 = utils_constEvalFuncWrapper_201
    utils_constEvalFuncWrapper_201_0 = utils_constEvalFuncWrapper_201[0]
    const_204 = main_const_eval_204
    util_create_list_422 = [input_86]
    utils_constEvalFuncWrapper_202 = utils.constEvalFuncWrapper(
        const_204, util_create_list_422, CACHED_main_const_eval_204
    )
    CACHED_main_const_eval_204 = utils_constEvalFuncWrapper_202
    utils_constEvalFuncWrapper_202_0 = utils_constEvalFuncWrapper_202[0]
    const_205 = main_const_eval_205
    util_create_list_423 = [input_91]
    utils_constEvalFuncWrapper_203 = utils.constEvalFuncWrapper(
        const_205, util_create_list_423, CACHED_main_const_eval_205
    )
    CACHED_main_const_eval_205 = utils_constEvalFuncWrapper_203
    utils_constEvalFuncWrapper_203_0 = utils_constEvalFuncWrapper_203[0]
    const_206 = main_const_eval_206
    util_create_list_424 = [input_43]
    utils_constEvalFuncWrapper_204 = utils.constEvalFuncWrapper(
        const_206, util_create_list_424, CACHED_main_const_eval_206
    )
    CACHED_main_const_eval_206 = utils_constEvalFuncWrapper_204
    utils_constEvalFuncWrapper_204_0 = utils_constEvalFuncWrapper_204[0]
    const_207 = main_const_eval_207
    util_create_list_425 = [input_63]
    utils_constEvalFuncWrapper_205 = utils.constEvalFuncWrapper(
        const_207, util_create_list_425, CACHED_main_const_eval_207
    )
    CACHED_main_const_eval_207 = utils_constEvalFuncWrapper_205
    utils_constEvalFuncWrapper_205_0 = utils_constEvalFuncWrapper_205[0]
    const_208 = main_const_eval_208
    util_create_list_426 = [input_108]
    utils_constEvalFuncWrapper_206 = utils.constEvalFuncWrapper(
        const_208, util_create_list_426, CACHED_main_const_eval_208
    )
    CACHED_main_const_eval_208 = utils_constEvalFuncWrapper_206
    utils_constEvalFuncWrapper_206_0 = utils_constEvalFuncWrapper_206[0]
    const_209 = main_const_eval_209
    util_create_list_427 = [input_58]
    utils_constEvalFuncWrapper_207 = utils.constEvalFuncWrapper(
        const_209, util_create_list_427, CACHED_main_const_eval_209
    )
    CACHED_main_const_eval_209 = utils_constEvalFuncWrapper_207
    utils_constEvalFuncWrapper_207_0 = utils_constEvalFuncWrapper_207[0]
    const_210 = main_const_eval_210
    util_create_list_428 = [input_90]
    utils_constEvalFuncWrapper_208 = utils.constEvalFuncWrapper(
        const_210, util_create_list_428, CACHED_main_const_eval_210
    )
    CACHED_main_const_eval_210 = utils_constEvalFuncWrapper_208
    utils_constEvalFuncWrapper_208_0 = utils_constEvalFuncWrapper_208[0]
    const_211 = main_const_eval_211
    util_create_list_429 = [input_27]
    utils_constEvalFuncWrapper_209 = utils.constEvalFuncWrapper(
        const_211, util_create_list_429, CACHED_main_const_eval_211
    )
    CACHED_main_const_eval_211 = utils_constEvalFuncWrapper_209
    utils_constEvalFuncWrapper_209_0 = utils_constEvalFuncWrapper_209[0]
    const_212 = main_const_eval_212
    util_create_list_430 = [input_114]
    utils_constEvalFuncWrapper_210 = utils.constEvalFuncWrapper(
        const_212, util_create_list_430, CACHED_main_const_eval_212
    )
    CACHED_main_const_eval_212 = utils_constEvalFuncWrapper_210
    utils_constEvalFuncWrapper_210_0 = utils_constEvalFuncWrapper_210[0]
    const_213 = main_const_eval_213
    util_create_list_431 = [input_32]
    utils_constEvalFuncWrapper_211 = utils.constEvalFuncWrapper(
        const_213, util_create_list_431, CACHED_main_const_eval_213
    )
    CACHED_main_const_eval_213 = utils_constEvalFuncWrapper_211
    utils_constEvalFuncWrapper_211_0 = utils_constEvalFuncWrapper_211[0]
    const_214 = main_const_eval_214
    util_create_list_432 = [input_104]
    utils_constEvalFuncWrapper_212 = utils.constEvalFuncWrapper(
        const_214, util_create_list_432, CACHED_main_const_eval_214
    )
    CACHED_main_const_eval_214 = utils_constEvalFuncWrapper_212
    utils_constEvalFuncWrapper_212_0 = utils_constEvalFuncWrapper_212[0]
    const_215 = main_const_eval_215
    util_create_list_433 = [input_42]
    utils_constEvalFuncWrapper_213 = utils.constEvalFuncWrapper(
        const_215, util_create_list_433, CACHED_main_const_eval_215
    )
    CACHED_main_const_eval_215 = utils_constEvalFuncWrapper_213
    utils_constEvalFuncWrapper_213_0 = utils_constEvalFuncWrapper_213[0]
    const_216 = main_const_eval_216
    util_create_list_434 = [input_74]
    utils_constEvalFuncWrapper_214 = utils.constEvalFuncWrapper(
        const_216, util_create_list_434, CACHED_main_const_eval_216
    )
    CACHED_main_const_eval_216 = utils_constEvalFuncWrapper_214
    utils_constEvalFuncWrapper_214_0 = utils_constEvalFuncWrapper_214[0]
    const_217 = main_const_eval_217
    util_create_list_435 = [input_59]
    utils_constEvalFuncWrapper_215 = utils.constEvalFuncWrapper(
        const_217, util_create_list_435, CACHED_main_const_eval_217
    )
    CACHED_main_const_eval_217 = utils_constEvalFuncWrapper_215
    utils_constEvalFuncWrapper_215_0 = utils_constEvalFuncWrapper_215[0]
    const_218 = main_const_eval_218
    util_create_list_436 = [input_8]
    utils_constEvalFuncWrapper_216 = utils.constEvalFuncWrapper(
        const_218, util_create_list_436, CACHED_main_const_eval_218
    )
    CACHED_main_const_eval_218 = utils_constEvalFuncWrapper_216
    utils_constEvalFuncWrapper_216_0 = utils_constEvalFuncWrapper_216[0]
    const_219 = main_const_eval_219
    util_create_list_437 = [input_115]
    utils_constEvalFuncWrapper_217 = utils.constEvalFuncWrapper(
        const_219, util_create_list_437, CACHED_main_const_eval_219
    )
    CACHED_main_const_eval_219 = utils_constEvalFuncWrapper_217
    utils_constEvalFuncWrapper_217_0 = utils_constEvalFuncWrapper_217[0]
    utils_DeviceGetter_get_device_29 = utils.DeviceGetter.get_device((1, 1))
    ttnn_permute_1 = ttnn.permute(
        input_109,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_permute_1,
        [1, 1, 50176, 3],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_from_device_0 = ttnn.from_device(ttnn_reshape_190)
    ttnn.deallocate(ttnn_reshape_190, False)
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_from_device_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_0, False)
    ttnn_to_device_27 = ttnn.to_device(
        ttnn_to_layout_27,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_conv2d_0 = ttnn.conv2d(
        input_tensor=ttnn_to_device_27,
        weight_tensor=input_0,
        device=utils_DeviceGetter_get_device_29,
        in_channels=3,
        out_channels=32,
        batch_size=1,
        input_height=224,
        input_width=224,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_27, False)
    ttnn_reshape_191 = ttnn.reshape(
        ttnn_conv2d_0,
        [1, 112, 112, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_0, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_191,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_191, False)
    ttnn_batch_norm_0 = ttnn.batch_norm(
        ttnn_permute_2,
        running_mean=utils_constEvalFuncWrapper_174_0,
        running_var=utils_constEvalFuncWrapper_187_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_122_0,
        bias=utils_constEvalFuncWrapper_117_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_0 = ttnn.relu(
        ttnn_batch_norm_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_0, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_relu_0,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_192 = ttnn.reshape(
        ttnn_permute_3,
        [1, 1, 12544, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_from_device_1 = ttnn.from_device(ttnn_reshape_192)
    ttnn.deallocate(ttnn_reshape_192, False)
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_from_device_1, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_1, False)
    ttnn_to_device_28 = ttnn.to_device(
        ttnn_to_layout_28,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_28, False)
    ttnn_conv2d_1 = ttnn.conv2d(
        input_tensor=ttnn_to_device_28,
        weight_tensor=input_2,
        device=utils_DeviceGetter_get_device_29,
        in_channels=32,
        out_channels=32,
        batch_size=1,
        input_height=112,
        input_width=112,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=32,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_reshape_193 = ttnn.reshape(
        ttnn_conv2d_1,
        [1, 112, 112, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_1, False)
    ttnn_permute_4 = ttnn.permute(
        ttnn_reshape_193,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_193, False)
    ttnn_batch_norm_1 = ttnn.batch_norm(
        ttnn_permute_4,
        running_mean=utils_constEvalFuncWrapper_87_0,
        running_var=utils_constEvalFuncWrapper_207_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_98_0,
        bias=utils_constEvalFuncWrapper_31_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_1 = ttnn.relu(
        ttnn_batch_norm_1,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_1, False)
    ttnn_permute_5 = ttnn.permute(
        ttnn_relu_1,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_194 = ttnn.reshape(
        ttnn_permute_5,
        [1, 1, 12544, 32],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_5, False)
    ttnn_from_device_2 = ttnn.from_device(ttnn_reshape_194)
    ttnn.deallocate(ttnn_reshape_194, False)
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_from_device_2, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_2, False)
    ttnn_to_device_29 = ttnn.to_device(
        ttnn_to_layout_29,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn_conv2d_2 = ttnn.conv2d(
        input_tensor=ttnn_to_device_29,
        weight_tensor=input_4,
        device=utils_DeviceGetter_get_device_29,
        in_channels=32,
        out_channels=64,
        batch_size=1,
        input_height=112,
        input_width=112,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_29, False)
    ttnn_reshape_195 = ttnn.reshape(
        ttnn_conv2d_2,
        [1, 112, 112, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_2, False)
    ttnn_permute_6 = ttnn.permute(
        ttnn_reshape_195,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_195, False)
    ttnn_batch_norm_2 = ttnn.batch_norm(
        ttnn_permute_6,
        running_mean=utils_constEvalFuncWrapper_215_0,
        running_var=utils_constEvalFuncWrapper_189_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_70_0,
        bias=utils_constEvalFuncWrapper_182_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_2 = ttnn.relu(
        ttnn_batch_norm_2,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_2, False)
    ttnn_permute_7 = ttnn.permute(
        ttnn_relu_2,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_196 = ttnn.reshape(
        ttnn_permute_7,
        [1, 1, 12544, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_7, False)
    ttnn_from_device_3 = ttnn.from_device(ttnn_reshape_196)
    ttnn.deallocate(ttnn_reshape_196, False)
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_from_device_3, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_3, False)
    ttnn_to_device_30 = ttnn.to_device(
        ttnn_to_layout_30,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    ttnn_conv2d_3 = ttnn.conv2d(
        input_tensor=ttnn_to_device_30,
        weight_tensor=input_6,
        device=utils_DeviceGetter_get_device_29,
        in_channels=64,
        out_channels=64,
        batch_size=1,
        input_height=112,
        input_width=112,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=64,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_30, False)
    ttnn_reshape_197 = ttnn.reshape(
        ttnn_conv2d_3,
        [1, 56, 56, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_3, False)
    ttnn_permute_8 = ttnn.permute(
        ttnn_reshape_197,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_197, False)
    ttnn_batch_norm_3 = ttnn.batch_norm(
        ttnn_permute_8,
        running_mean=utils_constEvalFuncWrapper_16_0,
        running_var=utils_constEvalFuncWrapper_140_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_94_0,
        bias=utils_constEvalFuncWrapper_65_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_3 = ttnn.relu(
        ttnn_batch_norm_3,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_3, False)
    ttnn_permute_9 = ttnn.permute(
        ttnn_relu_3,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_198 = ttnn.reshape(
        ttnn_permute_9,
        [1, 1, 3136, 64],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_9, False)
    ttnn_from_device_4 = ttnn.from_device(ttnn_reshape_198)
    ttnn.deallocate(ttnn_reshape_198, False)
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_from_device_4, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_4, False)
    ttnn_to_device_31 = ttnn.to_device(
        ttnn_to_layout_31,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    ttnn_conv2d_4 = ttnn.conv2d(
        input_tensor=ttnn_to_device_31,
        weight_tensor=input_8,
        device=utils_DeviceGetter_get_device_29,
        in_channels=64,
        out_channels=128,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_31, False)
    ttnn_reshape_199 = ttnn.reshape(
        ttnn_conv2d_4,
        [1, 56, 56, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_4, False)
    ttnn_permute_10 = ttnn.permute(
        ttnn_reshape_199,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_199, False)
    ttnn_batch_norm_4 = ttnn.batch_norm(
        ttnn_permute_10,
        running_mean=utils_constEvalFuncWrapper_110_0,
        running_var=utils_constEvalFuncWrapper_96_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_113_0,
        bias=utils_constEvalFuncWrapper_210_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_4 = ttnn.relu(
        ttnn_batch_norm_4,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_4, False)
    ttnn_permute_11 = ttnn.permute(
        ttnn_relu_4,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_200 = ttnn.reshape(
        ttnn_permute_11,
        [1, 1, 3136, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_11, False)
    ttnn_from_device_5 = ttnn.from_device(ttnn_reshape_200)
    ttnn.deallocate(ttnn_reshape_200, False)
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_from_device_5, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_5, False)
    ttnn_to_device_32 = ttnn.to_device(
        ttnn_to_layout_32,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    ttnn_conv2d_5 = ttnn.conv2d(
        input_tensor=ttnn_to_device_32,
        weight_tensor=input_10,
        device=utils_DeviceGetter_get_device_29,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=128,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_32, False)
    ttnn_reshape_201 = ttnn.reshape(
        ttnn_conv2d_5,
        [1, 56, 56, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_5, False)
    ttnn_permute_12 = ttnn.permute(
        ttnn_reshape_201,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_201, False)
    ttnn_batch_norm_5 = ttnn.batch_norm(
        ttnn_permute_12,
        running_mean=utils_constEvalFuncWrapper_19_0,
        running_var=utils_constEvalFuncWrapper_186_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_66_0,
        bias=utils_constEvalFuncWrapper_217_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_5 = ttnn.relu(
        ttnn_batch_norm_5,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_5, False)
    ttnn_permute_13 = ttnn.permute(
        ttnn_relu_5,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_202 = ttnn.reshape(
        ttnn_permute_13,
        [1, 1, 3136, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_13, False)
    ttnn_from_device_6 = ttnn.from_device(ttnn_reshape_202)
    ttnn.deallocate(ttnn_reshape_202, False)
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_from_device_6, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_6, False)
    ttnn_to_device_33 = ttnn.to_device(
        ttnn_to_layout_33,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_33, False)
    ttnn_conv2d_6 = ttnn.conv2d(
        input_tensor=ttnn_to_device_33,
        weight_tensor=input_12,
        device=utils_DeviceGetter_get_device_29,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_reshape_203 = ttnn.reshape(
        ttnn_conv2d_6,
        [1, 56, 56, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_6, False)
    ttnn_permute_14 = ttnn.permute(
        ttnn_reshape_203,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_203, False)
    ttnn_batch_norm_6 = ttnn.batch_norm(
        ttnn_permute_14,
        running_mean=utils_constEvalFuncWrapper_60_0,
        running_var=utils_constEvalFuncWrapper_193_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_135_0,
        bias=utils_constEvalFuncWrapper_67_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_6 = ttnn.relu(
        ttnn_batch_norm_6,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_6, False)
    ttnn_permute_15 = ttnn.permute(
        ttnn_relu_6,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_204 = ttnn.reshape(
        ttnn_permute_15,
        [1, 1, 3136, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_15, False)
    ttnn_from_device_7 = ttnn.from_device(ttnn_reshape_204)
    ttnn.deallocate(ttnn_reshape_204, False)
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_from_device_7, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_7, False)
    ttnn_to_device_34 = ttnn.to_device(
        ttnn_to_layout_34,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn_conv2d_7 = ttnn.conv2d(
        input_tensor=ttnn_to_device_34,
        weight_tensor=input_14,
        device=utils_DeviceGetter_get_device_29,
        in_channels=128,
        out_channels=128,
        batch_size=1,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=128,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_34, False)
    ttnn_reshape_205 = ttnn.reshape(
        ttnn_conv2d_7,
        [1, 28, 28, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_7, False)
    ttnn_permute_16 = ttnn.permute(
        ttnn_reshape_205,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_205, False)
    ttnn_batch_norm_7 = ttnn.batch_norm(
        ttnn_permute_16,
        running_mean=utils_constEvalFuncWrapper_38_0,
        running_var=utils_constEvalFuncWrapper_11_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_88_0,
        bias=utils_constEvalFuncWrapper_157_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_7 = ttnn.relu(
        ttnn_batch_norm_7,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_7, False)
    ttnn_permute_17 = ttnn.permute(
        ttnn_relu_7,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_206 = ttnn.reshape(
        ttnn_permute_17,
        [1, 1, 784, 128],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_17, False)
    ttnn_from_device_8 = ttnn.from_device(ttnn_reshape_206)
    ttnn.deallocate(ttnn_reshape_206, False)
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_from_device_8, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_8, False)
    ttnn_to_device_35 = ttnn.to_device(
        ttnn_to_layout_35,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    ttnn_conv2d_8 = ttnn.conv2d(
        input_tensor=ttnn_to_device_35,
        weight_tensor=input_16,
        device=utils_DeviceGetter_get_device_29,
        in_channels=128,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_35, False)
    ttnn_reshape_207 = ttnn.reshape(
        ttnn_conv2d_8,
        [1, 28, 28, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_8, False)
    ttnn_permute_18 = ttnn.permute(
        ttnn_reshape_207,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_207, False)
    ttnn_batch_norm_8 = ttnn.batch_norm(
        ttnn_permute_18,
        running_mean=utils_constEvalFuncWrapper_51_0,
        running_var=utils_constEvalFuncWrapper_191_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_71_0,
        bias=utils_constEvalFuncWrapper_145_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_8 = ttnn.relu(
        ttnn_batch_norm_8,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_8, False)
    ttnn_permute_19 = ttnn.permute(
        ttnn_relu_8,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_208 = ttnn.reshape(
        ttnn_permute_19,
        [1, 1, 784, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_19, False)
    ttnn_from_device_9 = ttnn.from_device(ttnn_reshape_208)
    ttnn.deallocate(ttnn_reshape_208, False)
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_from_device_9, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_9, False)
    ttnn_to_device_36 = ttnn.to_device(
        ttnn_to_layout_36,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_36, False)
    ttnn_conv2d_9 = ttnn.conv2d(
        input_tensor=ttnn_to_device_36,
        weight_tensor=input_18,
        device=utils_DeviceGetter_get_device_29,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=256,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_36, False)
    ttnn_reshape_209 = ttnn.reshape(
        ttnn_conv2d_9,
        [1, 28, 28, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_9, False)
    ttnn_permute_20 = ttnn.permute(
        ttnn_reshape_209,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_209, False)
    ttnn_batch_norm_9 = ttnn.batch_norm(
        ttnn_permute_20,
        running_mean=utils_constEvalFuncWrapper_148_0,
        running_var=utils_constEvalFuncWrapper_214_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_156_0,
        bias=utils_constEvalFuncWrapper_173_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_9 = ttnn.relu(
        ttnn_batch_norm_9,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_9, False)
    ttnn_permute_21 = ttnn.permute(
        ttnn_relu_9,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_210 = ttnn.reshape(
        ttnn_permute_21,
        [1, 1, 784, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_21, False)
    ttnn_from_device_10 = ttnn.from_device(ttnn_reshape_210)
    ttnn.deallocate(ttnn_reshape_210, False)
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_from_device_10, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_10, False)
    ttnn_to_device_37 = ttnn.to_device(
        ttnn_to_layout_37,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn_conv2d_10 = ttnn.conv2d(
        input_tensor=ttnn_to_device_37,
        weight_tensor=input_20,
        device=utils_DeviceGetter_get_device_29,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_37, False)
    ttnn_reshape_211 = ttnn.reshape(
        ttnn_conv2d_10,
        [1, 28, 28, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_10, False)
    ttnn_permute_22 = ttnn.permute(
        ttnn_reshape_211,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_211, False)
    ttnn_batch_norm_10 = ttnn.batch_norm(
        ttnn_permute_22,
        running_mean=utils_constEvalFuncWrapper_55_0,
        running_var=utils_constEvalFuncWrapper_192_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_30_0,
        bias=utils_constEvalFuncWrapper_34_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_10 = ttnn.relu(
        ttnn_batch_norm_10,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_10, False)
    ttnn_permute_23 = ttnn.permute(
        ttnn_relu_10,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_212 = ttnn.reshape(
        ttnn_permute_23,
        [1, 1, 784, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_23, False)
    ttnn_from_device_11 = ttnn.from_device(ttnn_reshape_212)
    ttnn.deallocate(ttnn_reshape_212, False)
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_from_device_11, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_11, False)
    ttnn_to_device_38 = ttnn.to_device(
        ttnn_to_layout_38,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    ttnn_conv2d_11 = ttnn.conv2d(
        input_tensor=ttnn_to_device_38,
        weight_tensor=input_22,
        device=utils_DeviceGetter_get_device_29,
        in_channels=256,
        out_channels=256,
        batch_size=1,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=256,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_38, False)
    ttnn_reshape_213 = ttnn.reshape(
        ttnn_conv2d_11,
        [1, 14, 14, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_11, False)
    ttnn_permute_24 = ttnn.permute(
        ttnn_reshape_213,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_213, False)
    ttnn_batch_norm_11 = ttnn.batch_norm(
        ttnn_permute_24,
        running_mean=utils_constEvalFuncWrapper_132_0,
        running_var=utils_constEvalFuncWrapper_196_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_133_0,
        bias=utils_constEvalFuncWrapper_149_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_11 = ttnn.relu(
        ttnn_batch_norm_11,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_11, False)
    ttnn_permute_25 = ttnn.permute(
        ttnn_relu_11,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_214 = ttnn.reshape(
        ttnn_permute_25,
        [1, 1, 196, 256],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_25, False)
    ttnn_from_device_12 = ttnn.from_device(ttnn_reshape_214)
    ttnn.deallocate(ttnn_reshape_214, False)
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_from_device_12, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_12, False)
    ttnn_to_device_39 = ttnn.to_device(
        ttnn_to_layout_39,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_conv2d_12 = ttnn.conv2d(
        input_tensor=ttnn_to_device_39,
        weight_tensor=input_24,
        device=utils_DeviceGetter_get_device_29,
        in_channels=256,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_reshape_215 = ttnn.reshape(
        ttnn_conv2d_12,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_12, False)
    ttnn_permute_26 = ttnn.permute(
        ttnn_reshape_215,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_215, False)
    ttnn_batch_norm_12 = ttnn.batch_norm(
        ttnn_permute_26,
        running_mean=utils_constEvalFuncWrapper_57_0,
        running_var=utils_constEvalFuncWrapper_190_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_137_0,
        bias=utils_constEvalFuncWrapper_39_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_12 = ttnn.relu(
        ttnn_batch_norm_12,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_12, False)
    ttnn_permute_27 = ttnn.permute(
        ttnn_relu_12,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_216 = ttnn.reshape(
        ttnn_permute_27,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_27, False)
    ttnn_from_device_13 = ttnn.from_device(ttnn_reshape_216)
    ttnn.deallocate(ttnn_reshape_216, False)
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_from_device_13, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_13, False)
    ttnn_to_device_40 = ttnn.to_device(
        ttnn_to_layout_40,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn_conv2d_13 = ttnn.conv2d(
        input_tensor=ttnn_to_device_40,
        weight_tensor=input_26,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=512,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_reshape_217 = ttnn.reshape(
        ttnn_conv2d_13,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_13, False)
    ttnn_permute_28 = ttnn.permute(
        ttnn_reshape_217,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_217, False)
    ttnn_batch_norm_13 = ttnn.batch_norm(
        ttnn_permute_28,
        running_mean=utils_constEvalFuncWrapper_165_0,
        running_var=utils_constEvalFuncWrapper_153_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_171_0,
        bias=utils_constEvalFuncWrapper_12_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_13 = ttnn.relu(
        ttnn_batch_norm_13,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_13, False)
    ttnn_permute_29 = ttnn.permute(
        ttnn_relu_13,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_218 = ttnn.reshape(
        ttnn_permute_29,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_29, False)
    ttnn_from_device_14 = ttnn.from_device(ttnn_reshape_218)
    ttnn.deallocate(ttnn_reshape_218, False)
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_from_device_14, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_14, False)
    ttnn_to_device_41 = ttnn.to_device(
        ttnn_to_layout_41,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_41, False)
    ttnn_conv2d_14 = ttnn.conv2d(
        input_tensor=ttnn_to_device_41,
        weight_tensor=input_28,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_reshape_219 = ttnn.reshape(
        ttnn_conv2d_14,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_14, False)
    ttnn_permute_30 = ttnn.permute(
        ttnn_reshape_219,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_219, False)
    ttnn_batch_norm_14 = ttnn.batch_norm(
        ttnn_permute_30,
        running_mean=utils_constEvalFuncWrapper_64_0,
        running_var=utils_constEvalFuncWrapper_158_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_89_0,
        bias=utils_constEvalFuncWrapper_108_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_14 = ttnn.relu(
        ttnn_batch_norm_14,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_14, False)
    ttnn_permute_31 = ttnn.permute(
        ttnn_relu_14,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_220 = ttnn.reshape(
        ttnn_permute_31,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_31, False)
    ttnn_from_device_15 = ttnn.from_device(ttnn_reshape_220)
    ttnn.deallocate(ttnn_reshape_220, False)
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_from_device_15, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_15, False)
    ttnn_to_device_42 = ttnn.to_device(
        ttnn_to_layout_42,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_42, False)
    ttnn_conv2d_15 = ttnn.conv2d(
        input_tensor=ttnn_to_device_42,
        weight_tensor=input_30,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=512,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_42, False)
    ttnn_reshape_221 = ttnn.reshape(
        ttnn_conv2d_15,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_15, False)
    ttnn_permute_32 = ttnn.permute(
        ttnn_reshape_221,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_221, False)
    ttnn_batch_norm_15 = ttnn.batch_norm(
        ttnn_permute_32,
        running_mean=utils_constEvalFuncWrapper_184_0,
        running_var=utils_constEvalFuncWrapper_41_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_23_0,
        bias=utils_constEvalFuncWrapper_97_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_15 = ttnn.relu(
        ttnn_batch_norm_15,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_15, False)
    ttnn_permute_33 = ttnn.permute(
        ttnn_relu_15,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_222 = ttnn.reshape(
        ttnn_permute_33,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_33, False)
    ttnn_from_device_16 = ttnn.from_device(ttnn_reshape_222)
    ttnn.deallocate(ttnn_reshape_222, False)
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_from_device_16, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_16, False)
    ttnn_to_device_43 = ttnn.to_device(
        ttnn_to_layout_43,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn_conv2d_16 = ttnn.conv2d(
        input_tensor=ttnn_to_device_43,
        weight_tensor=input_32,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_43, False)
    ttnn_reshape_223 = ttnn.reshape(
        ttnn_conv2d_16,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_16, False)
    ttnn_permute_34 = ttnn.permute(
        ttnn_reshape_223,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_223, False)
    ttnn_batch_norm_16 = ttnn.batch_norm(
        ttnn_permute_34,
        running_mean=utils_constEvalFuncWrapper_74_0,
        running_var=utils_constEvalFuncWrapper_14_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_176_0,
        bias=utils_constEvalFuncWrapper_134_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_16 = ttnn.relu(
        ttnn_batch_norm_16,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_16, False)
    ttnn_permute_35 = ttnn.permute(
        ttnn_relu_16,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_224 = ttnn.reshape(
        ttnn_permute_35,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_35, False)
    ttnn_from_device_17 = ttnn.from_device(ttnn_reshape_224)
    ttnn.deallocate(ttnn_reshape_224, False)
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_from_device_17, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_17, False)
    ttnn_to_device_44 = ttnn.to_device(
        ttnn_to_layout_44,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn_conv2d_17 = ttnn.conv2d(
        input_tensor=ttnn_to_device_44,
        weight_tensor=input_34,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=512,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_44, False)
    ttnn_reshape_225 = ttnn.reshape(
        ttnn_conv2d_17,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_17, False)
    ttnn_permute_36 = ttnn.permute(
        ttnn_reshape_225,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_225, False)
    ttnn_batch_norm_17 = ttnn.batch_norm(
        ttnn_permute_36,
        running_mean=utils_constEvalFuncWrapper_141_0,
        running_var=utils_constEvalFuncWrapper_208_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_139_0,
        bias=utils_constEvalFuncWrapper_18_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_17 = ttnn.relu(
        ttnn_batch_norm_17,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_17, False)
    ttnn_permute_37 = ttnn.permute(
        ttnn_relu_17,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_226 = ttnn.reshape(
        ttnn_permute_37,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_37, False)
    ttnn_from_device_18 = ttnn.from_device(ttnn_reshape_226)
    ttnn.deallocate(ttnn_reshape_226, False)
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_from_device_18, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_18, False)
    ttnn_to_device_45 = ttnn.to_device(
        ttnn_to_layout_45,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_conv2d_18 = ttnn.conv2d(
        input_tensor=ttnn_to_device_45,
        weight_tensor=input_36,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_45, False)
    ttnn_reshape_227 = ttnn.reshape(
        ttnn_conv2d_18,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_18, False)
    ttnn_permute_38 = ttnn.permute(
        ttnn_reshape_227,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_227, False)
    ttnn_batch_norm_18 = ttnn.batch_norm(
        ttnn_permute_38,
        running_mean=utils_constEvalFuncWrapper_203_0,
        running_var=utils_constEvalFuncWrapper_86_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_37_0,
        bias=utils_constEvalFuncWrapper_2_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_18 = ttnn.relu(
        ttnn_batch_norm_18,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_18, False)
    ttnn_permute_39 = ttnn.permute(
        ttnn_relu_18,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_228 = ttnn.reshape(
        ttnn_permute_39,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_39, False)
    ttnn_from_device_19 = ttnn.from_device(ttnn_reshape_228)
    ttnn.deallocate(ttnn_reshape_228, False)
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_from_device_19, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_19, False)
    ttnn_to_device_46 = ttnn.to_device(
        ttnn_to_layout_46,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_conv2d_19 = ttnn.conv2d(
        input_tensor=ttnn_to_device_46,
        weight_tensor=input_38,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=512,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_46, False)
    ttnn_reshape_229 = ttnn.reshape(
        ttnn_conv2d_19,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_19, False)
    ttnn_permute_40 = ttnn.permute(
        ttnn_reshape_229,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_229, False)
    ttnn_batch_norm_19 = ttnn.batch_norm(
        ttnn_permute_40,
        running_mean=utils_constEvalFuncWrapper_198_0,
        running_var=utils_constEvalFuncWrapper_73_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_166_0,
        bias=utils_constEvalFuncWrapper_35_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_19 = ttnn.relu(
        ttnn_batch_norm_19,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_19, False)
    ttnn_permute_41 = ttnn.permute(
        ttnn_relu_19,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_230 = ttnn.reshape(
        ttnn_permute_41,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_41, False)
    ttnn_from_device_20 = ttnn.from_device(ttnn_reshape_230)
    ttnn.deallocate(ttnn_reshape_230, False)
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_from_device_20, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_20, False)
    ttnn_to_device_47 = ttnn.to_device(
        ttnn_to_layout_47,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_conv2d_20 = ttnn.conv2d(
        input_tensor=ttnn_to_device_47,
        weight_tensor=input_40,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_47, False)
    ttnn_reshape_231 = ttnn.reshape(
        ttnn_conv2d_20,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_20, False)
    ttnn_permute_42 = ttnn.permute(
        ttnn_reshape_231,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_231, False)
    ttnn_batch_norm_20 = ttnn.batch_norm(
        ttnn_permute_42,
        running_mean=utils_constEvalFuncWrapper_50_0,
        running_var=utils_constEvalFuncWrapper_8_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_181_0,
        bias=utils_constEvalFuncWrapper_85_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_20 = ttnn.relu(
        ttnn_batch_norm_20,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_20, False)
    ttnn_permute_43 = ttnn.permute(
        ttnn_relu_20,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_232 = ttnn.reshape(
        ttnn_permute_43,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_43, False)
    ttnn_from_device_21 = ttnn.from_device(ttnn_reshape_232)
    ttnn.deallocate(ttnn_reshape_232, False)
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_from_device_21, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_21, False)
    ttnn_to_device_48 = ttnn.to_device(
        ttnn_to_layout_48,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_48, False)
    ttnn_conv2d_21 = ttnn.conv2d(
        input_tensor=ttnn_to_device_48,
        weight_tensor=input_42,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=512,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_48, False)
    ttnn_reshape_233 = ttnn.reshape(
        ttnn_conv2d_21,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_21, False)
    ttnn_permute_44 = ttnn.permute(
        ttnn_reshape_233,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_233, False)
    ttnn_batch_norm_21 = ttnn.batch_norm(
        ttnn_permute_44,
        running_mean=utils_constEvalFuncWrapper_7_0,
        running_var=utils_constEvalFuncWrapper_121_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_159_0,
        bias=utils_constEvalFuncWrapper_46_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_21 = ttnn.relu(
        ttnn_batch_norm_21,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_21, False)
    ttnn_permute_45 = ttnn.permute(
        ttnn_relu_21,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_234 = ttnn.reshape(
        ttnn_permute_45,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_45, False)
    ttnn_from_device_22 = ttnn.from_device(ttnn_reshape_234)
    ttnn.deallocate(ttnn_reshape_234, False)
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_from_device_22, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_22, False)
    ttnn_to_device_49 = ttnn.to_device(
        ttnn_to_layout_49,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_49, False)
    ttnn_conv2d_22 = ttnn.conv2d(
        input_tensor=ttnn_to_device_49,
        weight_tensor=input_44,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_reshape_235 = ttnn.reshape(
        ttnn_conv2d_22,
        [1, 14, 14, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_22, False)
    ttnn_permute_46 = ttnn.permute(
        ttnn_reshape_235,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_235, False)
    ttnn_batch_norm_22 = ttnn.batch_norm(
        ttnn_permute_46,
        running_mean=utils_constEvalFuncWrapper_143_0,
        running_var=utils_constEvalFuncWrapper_24_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_69_0,
        bias=utils_constEvalFuncWrapper_102_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_22 = ttnn.relu(
        ttnn_batch_norm_22,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_22, False)
    ttnn_permute_47 = ttnn.permute(
        ttnn_relu_22,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_236 = ttnn.reshape(
        ttnn_permute_47,
        [1, 1, 196, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_47, False)
    ttnn_from_device_23 = ttnn.from_device(ttnn_reshape_236)
    ttnn.deallocate(ttnn_reshape_236, False)
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_from_device_23, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_23, False)
    ttnn_to_device_50 = ttnn.to_device(
        ttnn_to_layout_50,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_50, False)
    ttnn_conv2d_23 = ttnn.conv2d(
        input_tensor=ttnn_to_device_50,
        weight_tensor=input_46,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=512,
        batch_size=1,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=512,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_50, False)
    ttnn_reshape_237 = ttnn.reshape(
        ttnn_conv2d_23,
        [1, 7, 7, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_23, False)
    ttnn_permute_48 = ttnn.permute(
        ttnn_reshape_237,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_237, False)
    ttnn_batch_norm_23 = ttnn.batch_norm(
        ttnn_permute_48,
        running_mean=utils_constEvalFuncWrapper_100_0,
        running_var=utils_constEvalFuncWrapper_167_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_144_0,
        bias=utils_constEvalFuncWrapper_63_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_23 = ttnn.relu(
        ttnn_batch_norm_23,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_23, False)
    ttnn_permute_49 = ttnn.permute(
        ttnn_relu_23,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_238 = ttnn.reshape(
        ttnn_permute_49,
        [1, 1, 49, 512],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_49, False)
    ttnn_from_device_24 = ttnn.from_device(ttnn_reshape_238)
    ttnn.deallocate(ttnn_reshape_238, False)
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_from_device_24, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_24, False)
    ttnn_to_device_51 = ttnn.to_device(
        ttnn_to_layout_51,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_51, False)
    ttnn_conv2d_24 = ttnn.conv2d(
        input_tensor=ttnn_to_device_51,
        weight_tensor=input_48,
        device=utils_DeviceGetter_get_device_29,
        in_channels=512,
        out_channels=1024,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_51, False)
    ttnn_reshape_239 = ttnn.reshape(
        ttnn_conv2d_24,
        [1, 7, 7, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_24, False)
    ttnn_permute_50 = ttnn.permute(
        ttnn_reshape_239,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_239, False)
    ttnn_batch_norm_24 = ttnn.batch_norm(
        ttnn_permute_50,
        running_mean=utils_constEvalFuncWrapper_28_0,
        running_var=utils_constEvalFuncWrapper_212_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_90_0,
        bias=utils_constEvalFuncWrapper_53_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_24 = ttnn.relu(
        ttnn_batch_norm_24,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_24, False)
    ttnn_permute_51 = ttnn.permute(
        ttnn_relu_24,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_240 = ttnn.reshape(
        ttnn_permute_51,
        [1, 1, 49, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_51, False)
    ttnn_from_device_25 = ttnn.from_device(ttnn_reshape_240)
    ttnn.deallocate(ttnn_reshape_240, False)
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_from_device_25, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_25, False)
    ttnn_to_device_52 = ttnn.to_device(
        ttnn_to_layout_52,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_52, False)
    ttnn_conv2d_25 = ttnn.conv2d(
        input_tensor=ttnn_to_device_52,
        weight_tensor=input_50,
        device=utils_DeviceGetter_get_device_29,
        in_channels=1024,
        out_channels=1024,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1024,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_52, False)
    ttnn_reshape_241 = ttnn.reshape(
        ttnn_conv2d_25,
        [1, 7, 7, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_25, False)
    ttnn_permute_52 = ttnn.permute(
        ttnn_reshape_241,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_241, False)
    ttnn_batch_norm_25 = ttnn.batch_norm(
        ttnn_permute_52,
        running_mean=utils_constEvalFuncWrapper_118_0,
        running_var=utils_constEvalFuncWrapper_77_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_160_0,
        bias=utils_constEvalFuncWrapper_170_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_25 = ttnn.relu(
        ttnn_batch_norm_25,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_25, False)
    ttnn_permute_53 = ttnn.permute(
        ttnn_relu_25,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn_reshape_242 = ttnn.reshape(
        ttnn_permute_53,
        [1, 1, 49, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_permute_53, False)
    ttnn_from_device_26 = ttnn.from_device(ttnn_reshape_242)
    ttnn.deallocate(ttnn_reshape_242, False)
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_from_device_26, ttnn.Layout.ROW_MAJOR, None, memory_config=None
    )
    ttnn.deallocate(ttnn_from_device_26, False)
    ttnn_to_device_53 = ttnn.to_device(
        ttnn_to_layout_53,
        device=utils_DeviceGetter_get_device_29,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_layout_53, False)
    ttnn_conv2d_26 = ttnn.conv2d(
        input_tensor=ttnn_to_device_53,
        weight_tensor=input_52,
        device=utils_DeviceGetter_get_device_29,
        in_channels=1024,
        out_channels=1024,
        batch_size=1,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=None,
        conv_config=ttnn.Conv2dConfig(enable_kernel_stride_folding=False),
        compute_config=None,
        slice_config=None,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_to_device_53, False)
    ttnn_reshape_243 = ttnn.reshape(
        ttnn_conv2d_26,
        [1, 7, 7, 1024],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_conv2d_26, False)
    ttnn_permute_54 = ttnn.permute(
        ttnn_reshape_243,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_243, False)
    ttnn_batch_norm_26 = ttnn.batch_norm(
        ttnn_permute_54,
        running_mean=utils_constEvalFuncWrapper_119_0,
        running_var=utils_constEvalFuncWrapper_68_0,
        training=False,
        eps=9.9999997473787516e-06,
        momentum=0.10000000149011612,
        weight=utils_constEvalFuncWrapper_109_0,
        bias=utils_constEvalFuncWrapper_104_0,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_relu_26 = ttnn.relu(
        ttnn_batch_norm_26,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.deallocate(ttnn_batch_norm_26, False)
    ttnn_mean_0 = ttnn.mean(
        ttnn_relu_26,
        [2, 3],
        False,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn_reshape_244 = ttnn.reshape(
        ttnn_mean_0,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_438 = [
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        utils_constEvalFuncWrapper_127_0,
        utils_constEvalFuncWrapper_183_0,
        input_1,
        utils_constEvalFuncWrapper_177_0,
        utils_constEvalFuncWrapper_164_0,
        input_3,
        utils_constEvalFuncWrapper_201_0,
        utils_constEvalFuncWrapper_106_0,
        input_5,
        utils_constEvalFuncWrapper_152_0,
        utils_constEvalFuncWrapper_172_0,
        input_7,
        utils_constEvalFuncWrapper_216_0,
        utils_constEvalFuncWrapper_151_0,
        input_9,
        utils_constEvalFuncWrapper_76_0,
        utils_constEvalFuncWrapper_29_0,
        input_11,
        utils_constEvalFuncWrapper_52_0,
        utils_constEvalFuncWrapper_3_0,
        input_13,
        utils_constEvalFuncWrapper_80_0,
        utils_constEvalFuncWrapper_44_0,
        input_15,
        utils_constEvalFuncWrapper_20_0,
        utils_constEvalFuncWrapper_1_0,
        input_17,
        utils_constEvalFuncWrapper_155_0,
        utils_constEvalFuncWrapper_131_0,
        input_19,
        utils_constEvalFuncWrapper_15_0,
        utils_constEvalFuncWrapper_116_0,
        input_21,
        utils_constEvalFuncWrapper_62_0,
        utils_constEvalFuncWrapper_168_0,
        input_23,
        utils_constEvalFuncWrapper_120_0,
        utils_constEvalFuncWrapper_111_0,
        input_25,
        utils_constEvalFuncWrapper_10_0,
        utils_constEvalFuncWrapper_209_0,
        input_27,
        utils_constEvalFuncWrapper_146_0,
        utils_constEvalFuncWrapper_92_0,
        input_29,
        utils_constEvalFuncWrapper_197_0,
        utils_constEvalFuncWrapper_26_0,
        input_31,
        utils_constEvalFuncWrapper_211_0,
        utils_constEvalFuncWrapper_115_0,
        input_33,
        utils_constEvalFuncWrapper_25_0,
        utils_constEvalFuncWrapper_178_0,
        input_35,
        utils_constEvalFuncWrapper_83_0,
        utils_constEvalFuncWrapper_101_0,
        input_37,
        utils_constEvalFuncWrapper_61_0,
        utils_constEvalFuncWrapper_43_0,
        input_39,
        utils_constEvalFuncWrapper_138_0,
        utils_constEvalFuncWrapper_79_0,
        input_41,
        utils_constEvalFuncWrapper_213_0,
        utils_constEvalFuncWrapper_204_0,
        input_43,
        utils_constEvalFuncWrapper_169_0,
        utils_constEvalFuncWrapper_59_0,
        input_45,
        utils_constEvalFuncWrapper_125_0,
        utils_constEvalFuncWrapper_154_0,
        input_47,
        utils_constEvalFuncWrapper_40_0,
        utils_constEvalFuncWrapper_114_0,
        input_49,
        utils_constEvalFuncWrapper_22_0,
        utils_constEvalFuncWrapper_142_0,
        input_51,
        utils_constEvalFuncWrapper_81_0,
        utils_constEvalFuncWrapper_175_0,
        input_53,
        utils_constEvalFuncWrapper_75_0,
        input_54,
        utils_constEvalFuncWrapper_4_0,
        input_55,
        utils_constEvalFuncWrapper_126_0,
        input_56,
        utils_constEvalFuncWrapper_129_0,
        input_57,
        utils_constEvalFuncWrapper_150_0,
        input_58,
        utils_constEvalFuncWrapper_13_0,
        input_59,
        utils_constEvalFuncWrapper_128_0,
        input_60,
        utils_constEvalFuncWrapper_78_0,
        input_61,
        utils_constEvalFuncWrapper_147_0,
        input_62,
        utils_constEvalFuncWrapper_205_0,
        input_63,
        utils_constEvalFuncWrapper_91_0,
        input_64,
        utils_constEvalFuncWrapper_17_0,
        input_65,
        utils_constEvalFuncWrapper_124_0,
        input_66,
        utils_constEvalFuncWrapper_0_0,
        input_67,
        utils_constEvalFuncWrapper_195_0,
        input_68,
        utils_constEvalFuncWrapper_58_0,
        input_69,
        utils_constEvalFuncWrapper_32_0,
        input_70,
        utils_constEvalFuncWrapper_49_0,
        input_71,
        utils_constEvalFuncWrapper_56_0,
        input_72,
        utils_constEvalFuncWrapper_163_0,
        input_73,
        utils_constEvalFuncWrapper_161_0,
        input_74,
        utils_constEvalFuncWrapper_33_0,
        input_75,
        utils_constEvalFuncWrapper_99_0,
        input_76,
        utils_constEvalFuncWrapper_6_0,
        input_77,
        utils_constEvalFuncWrapper_45_0,
        input_78,
        utils_constEvalFuncWrapper_93_0,
        input_79,
        utils_constEvalFuncWrapper_162_0,
        input_80,
        utils_constEvalFuncWrapper_179_0,
        input_81,
        utils_constEvalFuncWrapper_103_0,
        input_82,
        utils_constEvalFuncWrapper_9_0,
        input_83,
        utils_constEvalFuncWrapper_123_0,
        input_84,
        utils_constEvalFuncWrapper_5_0,
        input_85,
        utils_constEvalFuncWrapper_202_0,
        input_86,
        utils_constEvalFuncWrapper_36_0,
        input_87,
        utils_constEvalFuncWrapper_105_0,
        input_88,
        utils_constEvalFuncWrapper_199_0,
        input_89,
        utils_constEvalFuncWrapper_188_0,
        input_90,
        utils_constEvalFuncWrapper_194_0,
        input_91,
        utils_constEvalFuncWrapper_42_0,
        input_92,
        utils_constEvalFuncWrapper_130_0,
        input_93,
        utils_constEvalFuncWrapper_107_0,
        input_94,
        utils_constEvalFuncWrapper_180_0,
        input_95,
        utils_constEvalFuncWrapper_200_0,
        input_96,
        utils_constEvalFuncWrapper_21_0,
        input_97,
        utils_constEvalFuncWrapper_82_0,
        input_98,
        utils_constEvalFuncWrapper_84_0,
        input_99,
        utils_constEvalFuncWrapper_54_0,
        input_100,
        utils_constEvalFuncWrapper_136_0,
        input_101,
        utils_constEvalFuncWrapper_72_0,
        input_102,
        utils_constEvalFuncWrapper_112_0,
        input_103,
        utils_constEvalFuncWrapper_185_0,
        input_104,
        utils_constEvalFuncWrapper_47_0,
        input_105,
        utils_constEvalFuncWrapper_48_0,
        input_106,
        utils_constEvalFuncWrapper_95_0,
        input_107,
        utils_constEvalFuncWrapper_206_0,
        input_108,
        input_109,
        ttnn_permute_2,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_0,
        ttnn_permute_4,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_1,
        ttnn_permute_6,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_2,
        ttnn_permute_8,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_3,
        ttnn_permute_10,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_4,
        ttnn_permute_12,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_5,
        ttnn_permute_14,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_6,
        ttnn_permute_16,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_7,
        ttnn_permute_18,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_8,
        ttnn_permute_20,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_9,
        ttnn_permute_22,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_10,
        ttnn_permute_24,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_11,
        ttnn_permute_26,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_12,
        ttnn_permute_28,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_13,
        ttnn_permute_30,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_14,
        ttnn_permute_32,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_15,
        ttnn_permute_34,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_16,
        ttnn_permute_36,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_17,
        ttnn_permute_38,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_18,
        ttnn_permute_40,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_19,
        ttnn_permute_42,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_20,
        ttnn_permute_44,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_21,
        ttnn_permute_46,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_22,
        ttnn_permute_48,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_23,
        ttnn_permute_50,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_24,
        ttnn_permute_52,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_25,
        ttnn_permute_54,
        utils_constEvalFuncWrapperZeroArg_0_0,
        utils_constEvalFuncWrapperZeroArg_0_0,
        ttnn_relu_26,
        ttnn_reshape_244,
        ttnn_mean_0,
        utils_constEvalFuncWrapper_27_0,
    ]
    return util_create_list_438


def load_inputs_for__main():
    utils_DeviceGetter_get_device_30 = utils.DeviceGetter.get_device((1, 1))
    utils_load_tensor_0 = utils.load_tensor(
        "./tensors/arg0.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_1 = utils.load_tensor(
        "./tensors/arg1.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_2 = utils.load_tensor(
        "./tensors/arg2.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_3 = utils.load_tensor(
        "./tensors/arg3.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_4 = utils.load_tensor(
        "./tensors/arg4.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_5 = utils.load_tensor(
        "./tensors/arg5.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_6 = utils.load_tensor(
        "./tensors/arg6.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_7 = utils.load_tensor(
        "./tensors/arg7.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_8 = utils.load_tensor(
        "./tensors/arg8.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_9 = utils.load_tensor(
        "./tensors/arg9.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_10 = utils.load_tensor(
        "./tensors/arg10.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_11 = utils.load_tensor(
        "./tensors/arg11.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_12 = utils.load_tensor(
        "./tensors/arg12.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_13 = utils.load_tensor(
        "./tensors/arg13.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_14 = utils.load_tensor(
        "./tensors/arg14.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_15 = utils.load_tensor(
        "./tensors/arg15.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_16 = utils.load_tensor(
        "./tensors/arg16.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_17 = utils.load_tensor(
        "./tensors/arg17.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_18 = utils.load_tensor(
        "./tensors/arg18.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_19 = utils.load_tensor(
        "./tensors/arg19.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_20 = utils.load_tensor(
        "./tensors/arg20.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_21 = utils.load_tensor(
        "./tensors/arg21.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_22 = utils.load_tensor(
        "./tensors/arg22.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_23 = utils.load_tensor(
        "./tensors/arg23.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_24 = utils.load_tensor(
        "./tensors/arg24.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_25 = utils.load_tensor(
        "./tensors/arg25.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_26 = utils.load_tensor(
        "./tensors/arg26.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_27 = utils.load_tensor(
        "./tensors/arg27.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_28 = utils.load_tensor(
        "./tensors/arg28.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_29 = utils.load_tensor(
        "./tensors/arg29.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_30 = utils.load_tensor(
        "./tensors/arg30.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_31 = utils.load_tensor(
        "./tensors/arg31.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_32 = utils.load_tensor(
        "./tensors/arg32.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_33 = utils.load_tensor(
        "./tensors/arg33.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_34 = utils.load_tensor(
        "./tensors/arg34.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_35 = utils.load_tensor(
        "./tensors/arg35.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_36 = utils.load_tensor(
        "./tensors/arg36.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_37 = utils.load_tensor(
        "./tensors/arg37.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_38 = utils.load_tensor(
        "./tensors/arg38.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_39 = utils.load_tensor(
        "./tensors/arg39.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_40 = utils.load_tensor(
        "./tensors/arg40.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_41 = utils.load_tensor(
        "./tensors/arg41.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_42 = utils.load_tensor(
        "./tensors/arg42.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_43 = utils.load_tensor(
        "./tensors/arg43.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_44 = utils.load_tensor(
        "./tensors/arg44.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_45 = utils.load_tensor(
        "./tensors/arg45.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_46 = utils.load_tensor(
        "./tensors/arg46.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_47 = utils.load_tensor(
        "./tensors/arg47.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_48 = utils.load_tensor(
        "./tensors/arg48.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_49 = utils.load_tensor(
        "./tensors/arg49.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_50 = utils.load_tensor(
        "./tensors/arg50.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_51 = utils.load_tensor(
        "./tensors/arg51.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_52 = utils.load_tensor(
        "./tensors/arg52.tensorbin",
        ttnn.Layout.ROW_MAJOR,
        ttnn.DataType.FLOAT32,
        None,
        None,
    )
    utils_load_tensor_53 = utils.load_tensor(
        "./tensors/arg53.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_54 = utils.load_tensor(
        "./tensors/arg54.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_55 = utils.load_tensor(
        "./tensors/arg55.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_56 = utils.load_tensor(
        "./tensors/arg56.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_57 = utils.load_tensor(
        "./tensors/arg57.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_58 = utils.load_tensor(
        "./tensors/arg58.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_59 = utils.load_tensor(
        "./tensors/arg59.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_60 = utils.load_tensor(
        "./tensors/arg60.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_61 = utils.load_tensor(
        "./tensors/arg61.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_62 = utils.load_tensor(
        "./tensors/arg62.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_63 = utils.load_tensor(
        "./tensors/arg63.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_64 = utils.load_tensor(
        "./tensors/arg64.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_65 = utils.load_tensor(
        "./tensors/arg65.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_66 = utils.load_tensor(
        "./tensors/arg66.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_67 = utils.load_tensor(
        "./tensors/arg67.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_68 = utils.load_tensor(
        "./tensors/arg68.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_69 = utils.load_tensor(
        "./tensors/arg69.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_70 = utils.load_tensor(
        "./tensors/arg70.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_71 = utils.load_tensor(
        "./tensors/arg71.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_72 = utils.load_tensor(
        "./tensors/arg72.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_73 = utils.load_tensor(
        "./tensors/arg73.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_74 = utils.load_tensor(
        "./tensors/arg74.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_75 = utils.load_tensor(
        "./tensors/arg75.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_76 = utils.load_tensor(
        "./tensors/arg76.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_77 = utils.load_tensor(
        "./tensors/arg77.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_78 = utils.load_tensor(
        "./tensors/arg78.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_79 = utils.load_tensor(
        "./tensors/arg79.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_80 = utils.load_tensor(
        "./tensors/arg80.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_81 = utils.load_tensor(
        "./tensors/arg81.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_82 = utils.load_tensor(
        "./tensors/arg82.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_83 = utils.load_tensor(
        "./tensors/arg83.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_84 = utils.load_tensor(
        "./tensors/arg84.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_85 = utils.load_tensor(
        "./tensors/arg85.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_86 = utils.load_tensor(
        "./tensors/arg86.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_87 = utils.load_tensor(
        "./tensors/arg87.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_88 = utils.load_tensor(
        "./tensors/arg88.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_89 = utils.load_tensor(
        "./tensors/arg89.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_90 = utils.load_tensor(
        "./tensors/arg90.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_91 = utils.load_tensor(
        "./tensors/arg91.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_92 = utils.load_tensor(
        "./tensors/arg92.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_93 = utils.load_tensor(
        "./tensors/arg93.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_94 = utils.load_tensor(
        "./tensors/arg94.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_95 = utils.load_tensor(
        "./tensors/arg95.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_96 = utils.load_tensor(
        "./tensors/arg96.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_97 = utils.load_tensor(
        "./tensors/arg97.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_98 = utils.load_tensor(
        "./tensors/arg98.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_99 = utils.load_tensor(
        "./tensors/arg99.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_100 = utils.load_tensor(
        "./tensors/arg100.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_101 = utils.load_tensor(
        "./tensors/arg101.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_102 = utils.load_tensor(
        "./tensors/arg102.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_103 = utils.load_tensor(
        "./tensors/arg103.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_104 = utils.load_tensor(
        "./tensors/arg104.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_105 = utils.load_tensor(
        "./tensors/arg105.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_106 = utils.load_tensor(
        "./tensors/arg106.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_107 = utils.load_tensor(
        "./tensors/arg107.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_108 = utils.load_tensor(
        "./tensors/arg108.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_109 = utils.load_tensor(
        "./tensors/arg109.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_110 = utils.load_tensor(
        "./tensors/arg110.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_111 = utils.load_tensor(
        "./tensors/arg111.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_112 = utils.load_tensor(
        "./tensors/arg112.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_113 = utils.load_tensor(
        "./tensors/arg113.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_114 = utils.load_tensor(
        "./tensors/arg114.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_115 = utils.load_tensor(
        "./tensors/arg115.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_116 = utils.load_tensor(
        "./tensors/arg116.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_117 = utils.load_tensor(
        "./tensors/arg117.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_118 = utils.load_tensor(
        "./tensors/arg118.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_119 = utils.load_tensor(
        "./tensors/arg119.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_120 = utils.load_tensor(
        "./tensors/arg120.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_121 = utils.load_tensor(
        "./tensors/arg121.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_122 = utils.load_tensor(
        "./tensors/arg122.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_123 = utils.load_tensor(
        "./tensors/arg123.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_124 = utils.load_tensor(
        "./tensors/arg124.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_125 = utils.load_tensor(
        "./tensors/arg125.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_126 = utils.load_tensor(
        "./tensors/arg126.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_127 = utils.load_tensor(
        "./tensors/arg127.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_128 = utils.load_tensor(
        "./tensors/arg128.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_129 = utils.load_tensor(
        "./tensors/arg129.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_130 = utils.load_tensor(
        "./tensors/arg130.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_131 = utils.load_tensor(
        "./tensors/arg131.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_132 = utils.load_tensor(
        "./tensors/arg132.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_133 = utils.load_tensor(
        "./tensors/arg133.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_134 = utils.load_tensor(
        "./tensors/arg134.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_135 = utils.load_tensor(
        "./tensors/arg135.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    utils_load_tensor_136 = utils.load_tensor(
        "./tensors/arg136.tensorbin",
        ttnn.Layout.TILE,
        ttnn.DataType.FLOAT32,
        utils_DeviceGetter_get_device_30,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    util_create_list_439 = [
        utils_load_tensor_0,
        utils_load_tensor_1,
        utils_load_tensor_2,
        utils_load_tensor_3,
        utils_load_tensor_4,
        utils_load_tensor_5,
        utils_load_tensor_6,
        utils_load_tensor_7,
        utils_load_tensor_8,
        utils_load_tensor_9,
        utils_load_tensor_10,
        utils_load_tensor_11,
        utils_load_tensor_12,
        utils_load_tensor_13,
        utils_load_tensor_14,
        utils_load_tensor_15,
        utils_load_tensor_16,
        utils_load_tensor_17,
        utils_load_tensor_18,
        utils_load_tensor_19,
        utils_load_tensor_20,
        utils_load_tensor_21,
        utils_load_tensor_22,
        utils_load_tensor_23,
        utils_load_tensor_24,
        utils_load_tensor_25,
        utils_load_tensor_26,
        utils_load_tensor_27,
        utils_load_tensor_28,
        utils_load_tensor_29,
        utils_load_tensor_30,
        utils_load_tensor_31,
        utils_load_tensor_32,
        utils_load_tensor_33,
        utils_load_tensor_34,
        utils_load_tensor_35,
        utils_load_tensor_36,
        utils_load_tensor_37,
        utils_load_tensor_38,
        utils_load_tensor_39,
        utils_load_tensor_40,
        utils_load_tensor_41,
        utils_load_tensor_42,
        utils_load_tensor_43,
        utils_load_tensor_44,
        utils_load_tensor_45,
        utils_load_tensor_46,
        utils_load_tensor_47,
        utils_load_tensor_48,
        utils_load_tensor_49,
        utils_load_tensor_50,
        utils_load_tensor_51,
        utils_load_tensor_52,
        utils_load_tensor_53,
        utils_load_tensor_54,
        utils_load_tensor_55,
        utils_load_tensor_56,
        utils_load_tensor_57,
        utils_load_tensor_58,
        utils_load_tensor_59,
        utils_load_tensor_60,
        utils_load_tensor_61,
        utils_load_tensor_62,
        utils_load_tensor_63,
        utils_load_tensor_64,
        utils_load_tensor_65,
        utils_load_tensor_66,
        utils_load_tensor_67,
        utils_load_tensor_68,
        utils_load_tensor_69,
        utils_load_tensor_70,
        utils_load_tensor_71,
        utils_load_tensor_72,
        utils_load_tensor_73,
        utils_load_tensor_74,
        utils_load_tensor_75,
        utils_load_tensor_76,
        utils_load_tensor_77,
        utils_load_tensor_78,
        utils_load_tensor_79,
        utils_load_tensor_80,
        utils_load_tensor_81,
        utils_load_tensor_82,
        utils_load_tensor_83,
        utils_load_tensor_84,
        utils_load_tensor_85,
        utils_load_tensor_86,
        utils_load_tensor_87,
        utils_load_tensor_88,
        utils_load_tensor_89,
        utils_load_tensor_90,
        utils_load_tensor_91,
        utils_load_tensor_92,
        utils_load_tensor_93,
        utils_load_tensor_94,
        utils_load_tensor_95,
        utils_load_tensor_96,
        utils_load_tensor_97,
        utils_load_tensor_98,
        utils_load_tensor_99,
        utils_load_tensor_100,
        utils_load_tensor_101,
        utils_load_tensor_102,
        utils_load_tensor_103,
        utils_load_tensor_104,
        utils_load_tensor_105,
        utils_load_tensor_106,
        utils_load_tensor_107,
        utils_load_tensor_108,
        utils_load_tensor_109,
        utils_load_tensor_110,
        utils_load_tensor_111,
        utils_load_tensor_112,
        utils_load_tensor_113,
        utils_load_tensor_114,
        utils_load_tensor_115,
        utils_load_tensor_116,
        utils_load_tensor_117,
        utils_load_tensor_118,
        utils_load_tensor_119,
        utils_load_tensor_120,
        utils_load_tensor_121,
        utils_load_tensor_122,
        utils_load_tensor_123,
        utils_load_tensor_124,
        utils_load_tensor_125,
        utils_load_tensor_126,
        utils_load_tensor_127,
        utils_load_tensor_128,
        utils_load_tensor_129,
        utils_load_tensor_130,
        utils_load_tensor_131,
        utils_load_tensor_132,
        utils_load_tensor_133,
        utils_load_tensor_134,
        utils_load_tensor_135,
        utils_load_tensor_136,
    ]
    return util_create_list_439

def load_inputs_for__main():
    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_1.0_224")
    dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
    image = next(iter(dataset.skip(10)))["image"]
    input_dict = preprocessor(images=image, return_tensors="pt")
    inputs = input_dict.pixel_values
    return inputs

class MobileNetV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v1_1.0_224")
    def forward(self, x):
        return self.model(x)


def compute_pcc_list(tt_list, cpu_list):
    pcc_values = []
    for t1, t2 in zip(tt_list, cpu_list):
        a = t1.detach().cpu().numpy().flatten()
        b = t2.detach().cpu().numpy().flatten()
        pcc, _ = pearsonr(a, b)
        pcc_values.append(pcc)
    return pcc_values


def test_mobilenetv1_sanity():
    ## tt run
    v1 = load_inputs_for__main()
    tt_output = _main(v1)

    logger.info("tt_output={}", tt_output)

    ## cpu run - load same weights as ttnn
    x1 = torch.load("/proj_sw/user_dev/mramanathan/bgdlab22_dec7_metal/tt-metal/tests/ttnn/unit_tests/operations/mobilenetv1/mobileentv1_inputs.pt")

    # Load all 137 arg tensor bins from codegen directory (keep weights/bias for CPU model)
    tensor_dir = "/proj_sw/user_dev/mramanathan/bgdlab22_dec7_metal/tt-metal/tests/ttnn/unit_tests/operations/mobilenetv1_codegen/tensors"
    arg_tensors_ttnn = [ttnn.load_tensor(f"{tensor_dir}/arg{i}.tensorbin") for i in range(137)]
    arg_tensors_torch = [ttnn.to_torch(t) for t in arg_tensors_ttnn]
    # Maintain existing variables expected below
    weights_torch = arg_tensors_torch[2]
    bias_torch = arg_tensors_torch[1].squeeze()

    logger.info(f"Loaded weights shape: {weights_torch.shape}, bias shape: {bias_torch.shape}")

    cpu_model = MobileNetV1(weights=weights_torch, bias=bias_torch)

    with torch.no_grad():
        cpu_output = cpu_model(x1)

    for i in range(len(tt_output)):
        tt_output[i] = ttnn.to_torch(tt_output[i])

    logger.info("cpu_output={}", cpu_output)

    # pcc check
    pcc_values = compute_pcc_list(tt_output, cpu_output)

    logger.info(f"PCC values per output tensor: {pcc_values}")
