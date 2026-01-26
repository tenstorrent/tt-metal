# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for CLIP Resampler model."""

import utils

import ttnn


def main_const_eval_0(device):
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_0 = [ttnn_full_0]
    return util_create_list_0


def main_const_eval_1(weights, device):
    ttnn_to_device_0 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_0 = ttnn.to_layout(
        ttnn_to_device_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_0, False)
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_0, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_0, False)
    util_create_list_1 = [ttnn_repeat_0]
    return util_create_list_1


def main_const_eval_2(weights, device):
    ttnn_to_device_1 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_1 = ttnn.to_layout(
        ttnn_to_device_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_1, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_to_layout_1,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_repeat_1 = ttnn.repeat(ttnn_reshape_1, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_1, False)
    util_create_list_2 = [ttnn_repeat_1]
    return util_create_list_2


def main_const_eval_3(weights, device):
    ttnn_to_device_2 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_to_device_2,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_2, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_2,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_repeat_2 = ttnn.repeat(ttnn_reshape_2, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_2, False)
    util_create_list_3 = [ttnn_repeat_2]
    return util_create_list_3


def main_const_eval_4(weights, device):
    ttnn_to_device_3 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_to_device_3,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_3, False)
    ttnn_to_device_4 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_4 = ttnn.to_layout(
        ttnn_to_device_4,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_4, False)
    ttnn_to_device_5 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_to_device_5,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_5, False)
    util_create_list_4 = [ttnn_to_layout_3, ttnn_to_layout_4, ttnn_to_layout_5]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_4,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn.deallocate(ttnn_to_layout_3, False)
    util_create_list_5 = [ttnn_concat_0]
    return util_create_list_5


def main_const_eval_5(weights, device):
    ttnn_to_device_6 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_to_device_6,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_6, False)
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_to_layout_6,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_6, False)
    ttnn_repeat_3 = ttnn.repeat(ttnn_reshape_3, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_3, False)
    util_create_list_6 = [ttnn_repeat_3]
    return util_create_list_6


def main_const_eval_6(weights, device):
    ttnn_to_device_7 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_7 = ttnn.to_layout(
        ttnn_to_device_7,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_7, False)
    ttnn_to_device_8 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_to_device_8,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_8, False)
    ttnn_to_device_9 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_to_device_9,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_9, False)
    util_create_list_7 = [ttnn_to_layout_7, ttnn_to_layout_8, ttnn_to_layout_9]
    ttnn_concat_1 = ttnn.concat(
        util_create_list_7,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_9, False)
    ttnn.deallocate(ttnn_to_layout_8, False)
    ttnn.deallocate(ttnn_to_layout_7, False)
    util_create_list_8 = [ttnn_concat_1]
    return util_create_list_8


def main_const_eval_7(weights, device):
    ttnn_to_device_10 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_10 = ttnn.to_layout(
        ttnn_to_device_10,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_10, False)
    ttnn_to_device_11 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_to_device_11,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_11, False)
    ttnn_to_device_12 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_to_device_12,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_12, False)
    util_create_list_9 = [ttnn_to_layout_10, ttnn_to_layout_11, ttnn_to_layout_12]
    ttnn_concat_2 = ttnn.concat(
        util_create_list_9,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_12, False)
    ttnn.deallocate(ttnn_to_layout_11, False)
    ttnn.deallocate(ttnn_to_layout_10, False)
    util_create_list_10 = [ttnn_concat_2]
    return util_create_list_10


def main_const_eval_8(weights, device):
    ttnn_to_device_13 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_13 = ttnn.to_layout(
        ttnn_to_device_13,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_13, False)
    ttnn_to_device_14 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_14 = ttnn.to_layout(
        ttnn_to_device_14,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_14, False)
    ttnn_to_device_15 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_15 = ttnn.to_layout(
        ttnn_to_device_15,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_15, False)
    util_create_list_11 = [ttnn_to_layout_13, ttnn_to_layout_14, ttnn_to_layout_15]
    ttnn_concat_3 = ttnn.concat(
        util_create_list_11,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_15, False)
    ttnn.deallocate(ttnn_to_layout_14, False)
    ttnn.deallocate(ttnn_to_layout_13, False)
    util_create_list_12 = [ttnn_concat_3]
    return util_create_list_12


def main_const_eval_9(weights, device):
    ttnn_to_device_16 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_16 = ttnn.to_layout(
        ttnn_to_device_16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_16, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_to_layout_16,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_16, False)
    ttnn_repeat_4 = ttnn.repeat(ttnn_reshape_4, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_4, False)
    util_create_list_13 = [ttnn_repeat_4]
    return util_create_list_13


def main_const_eval_10(weights, device):
    ttnn_to_device_17 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_17 = ttnn.to_layout(
        ttnn_to_device_17,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_17, False)
    ttnn_to_device_18 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_18 = ttnn.to_layout(
        ttnn_to_device_18,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_18, False)
    ttnn_to_device_19 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_19 = ttnn.to_layout(
        ttnn_to_device_19,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_19, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_to_layout_17,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_17, False)
    ttnn_repeat_5 = ttnn.repeat(ttnn_reshape_5, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_to_layout_18,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_18, False)
    ttnn_repeat_6 = ttnn.repeat(ttnn_reshape_6, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_6, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_to_layout_19,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_19, False)
    ttnn_repeat_7 = ttnn.repeat(ttnn_reshape_7, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_7, False)
    util_create_list_14 = [ttnn_repeat_5, ttnn_repeat_6, ttnn_repeat_7]
    ttnn_concat_4 = ttnn.concat(
        util_create_list_14,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_7, False)
    ttnn.deallocate(ttnn_repeat_6, False)
    ttnn.deallocate(ttnn_repeat_5, False)
    util_create_list_15 = [ttnn_concat_4]
    return util_create_list_15


def main_const_eval_11(weights, device):
    ttnn_to_device_20 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_20 = ttnn.to_layout(
        ttnn_to_device_20,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_20, False)
    ttnn_to_device_21 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_21 = ttnn.to_layout(
        ttnn_to_device_21,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_21, False)
    ttnn_to_device_22 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_22 = ttnn.to_layout(
        ttnn_to_device_22,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_22, False)
    util_create_list_16 = [ttnn_to_layout_20, ttnn_to_layout_21, ttnn_to_layout_22]
    ttnn_concat_5 = ttnn.concat(
        util_create_list_16,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_22, False)
    ttnn.deallocate(ttnn_to_layout_21, False)
    ttnn.deallocate(ttnn_to_layout_20, False)
    util_create_list_17 = [ttnn_concat_5]
    return util_create_list_17


def main_const_eval_12(weights, device):
    ttnn_to_device_23 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_23 = ttnn.to_layout(
        ttnn_to_device_23,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_23, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_to_layout_23,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_23, False)
    ttnn_repeat_8 = ttnn.repeat(ttnn_reshape_8, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_8, False)
    util_create_list_18 = [ttnn_repeat_8]
    return util_create_list_18


def main_const_eval_13(weights, device):
    ttnn_to_device_24 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_24 = ttnn.to_layout(
        ttnn_to_device_24,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_24, False)
    ttnn_to_device_25 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_25 = ttnn.to_layout(
        ttnn_to_device_25,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_25, False)
    ttnn_to_device_26 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_26 = ttnn.to_layout(
        ttnn_to_device_26,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_26, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_to_layout_24,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_24, False)
    ttnn_repeat_9 = ttnn.repeat(ttnn_reshape_9, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_9, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_to_layout_25,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_25, False)
    ttnn_repeat_10 = ttnn.repeat(ttnn_reshape_10, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_to_layout_26,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_26, False)
    ttnn_repeat_11 = ttnn.repeat(ttnn_reshape_11, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_11, False)
    util_create_list_19 = [ttnn_repeat_9, ttnn_repeat_10, ttnn_repeat_11]
    ttnn_concat_6 = ttnn.concat(
        util_create_list_19,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_11, False)
    ttnn.deallocate(ttnn_repeat_10, False)
    ttnn.deallocate(ttnn_repeat_9, False)
    util_create_list_20 = [ttnn_concat_6]
    return util_create_list_20


def main_const_eval_14(weights, device):
    ttnn_to_device_27 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_27 = ttnn.to_layout(
        ttnn_to_device_27,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_27, False)
    ttnn_to_device_28 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_28 = ttnn.to_layout(
        ttnn_to_device_28,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_28, False)
    ttnn_to_device_29 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_29 = ttnn.to_layout(
        ttnn_to_device_29,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_29, False)
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_to_layout_27,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_27, False)
    ttnn_repeat_12 = ttnn.repeat(ttnn_reshape_12, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_to_layout_28,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_28, False)
    ttnn_repeat_13 = ttnn.repeat(ttnn_reshape_13, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_13, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_to_layout_29,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_29, False)
    ttnn_repeat_14 = ttnn.repeat(ttnn_reshape_14, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_14, False)
    util_create_list_21 = [ttnn_repeat_12, ttnn_repeat_13, ttnn_repeat_14]
    ttnn_concat_7 = ttnn.concat(
        util_create_list_21,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_14, False)
    ttnn.deallocate(ttnn_repeat_13, False)
    ttnn.deallocate(ttnn_repeat_12, False)
    util_create_list_22 = [ttnn_concat_7]
    return util_create_list_22


def main_const_eval_15(weights, device):
    ttnn_to_device_30 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_30 = ttnn.to_layout(
        ttnn_to_device_30,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_30, False)
    ttnn_to_device_31 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_31 = ttnn.to_layout(
        ttnn_to_device_31,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_31, False)
    ttnn_to_device_32 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_32 = ttnn.to_layout(
        ttnn_to_device_32,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_32, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_to_layout_30,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_30, False)
    ttnn_repeat_15 = ttnn.repeat(ttnn_reshape_15, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_to_layout_31,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_31, False)
    ttnn_repeat_16 = ttnn.repeat(ttnn_reshape_16, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_16, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_to_layout_32,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_32, False)
    ttnn_repeat_17 = ttnn.repeat(ttnn_reshape_17, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_17, False)
    util_create_list_23 = [ttnn_repeat_15, ttnn_repeat_16, ttnn_repeat_17]
    ttnn_concat_8 = ttnn.concat(
        util_create_list_23,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_17, False)
    ttnn.deallocate(ttnn_repeat_16, False)
    ttnn.deallocate(ttnn_repeat_15, False)
    util_create_list_24 = [ttnn_concat_8]
    return util_create_list_24


def main_const_eval_16(weights, device):
    ttnn_to_device_33 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_33 = ttnn.to_layout(
        ttnn_to_device_33,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_33, False)
    ttnn_to_device_34 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_34 = ttnn.to_layout(
        ttnn_to_device_34,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_34, False)
    ttnn_to_device_35 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_35 = ttnn.to_layout(
        ttnn_to_device_35,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_35, False)
    util_create_list_25 = [ttnn_to_layout_33, ttnn_to_layout_34, ttnn_to_layout_35]
    ttnn_concat_9 = ttnn.concat(
        util_create_list_25,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_35, False)
    ttnn.deallocate(ttnn_to_layout_34, False)
    ttnn.deallocate(ttnn_to_layout_33, False)
    util_create_list_26 = [ttnn_concat_9]
    return util_create_list_26


def main_const_eval_17(weights, device):
    ttnn_to_device_36 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_36 = ttnn.to_layout(
        ttnn_to_device_36,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_36, False)
    ttnn_to_device_37 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_37 = ttnn.to_layout(
        ttnn_to_device_37,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_37, False)
    ttnn_to_device_38 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_38 = ttnn.to_layout(
        ttnn_to_device_38,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_38, False)
    util_create_list_27 = [ttnn_to_layout_36, ttnn_to_layout_37, ttnn_to_layout_38]
    ttnn_concat_10 = ttnn.concat(
        util_create_list_27,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_38, False)
    ttnn.deallocate(ttnn_to_layout_37, False)
    ttnn.deallocate(ttnn_to_layout_36, False)
    util_create_list_28 = [ttnn_concat_10]
    return util_create_list_28


def main_const_eval_18(weights, device):
    ttnn_to_device_39 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_39 = ttnn.to_layout(
        ttnn_to_device_39,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_39, False)
    ttnn_to_device_40 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_40 = ttnn.to_layout(
        ttnn_to_device_40,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_40, False)
    ttnn_to_device_41 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_41 = ttnn.to_layout(
        ttnn_to_device_41,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_41, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_to_layout_39,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_39, False)
    ttnn_repeat_18 = ttnn.repeat(ttnn_reshape_18, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_18, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_to_layout_40,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_40, False)
    ttnn_repeat_19 = ttnn.repeat(ttnn_reshape_19, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_to_layout_41,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_41, False)
    ttnn_repeat_20 = ttnn.repeat(ttnn_reshape_20, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_20, False)
    util_create_list_29 = [ttnn_repeat_18, ttnn_repeat_19, ttnn_repeat_20]
    ttnn_concat_11 = ttnn.concat(
        util_create_list_29,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_20, False)
    ttnn.deallocate(ttnn_repeat_19, False)
    ttnn.deallocate(ttnn_repeat_18, False)
    util_create_list_30 = [ttnn_concat_11]
    return util_create_list_30


def main_const_eval_19(weights, device):
    ttnn_to_device_42 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_42 = ttnn.to_layout(
        ttnn_to_device_42,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_42, False)
    ttnn_to_device_43 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_43 = ttnn.to_layout(
        ttnn_to_device_43,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_43, False)
    ttnn_to_device_44 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_44 = ttnn.to_layout(
        ttnn_to_device_44,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_44, False)
    util_create_list_31 = [ttnn_to_layout_42, ttnn_to_layout_43, ttnn_to_layout_44]
    ttnn_concat_12 = ttnn.concat(
        util_create_list_31,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_44, False)
    ttnn.deallocate(ttnn_to_layout_43, False)
    ttnn.deallocate(ttnn_to_layout_42, False)
    util_create_list_32 = [ttnn_concat_12]
    return util_create_list_32


def main_const_eval_20(weights, device):
    ttnn_to_device_45 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_45 = ttnn.to_layout(
        ttnn_to_device_45,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_45, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_to_layout_45,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_45, False)
    ttnn_repeat_21 = ttnn.repeat(ttnn_reshape_21, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_21, False)
    util_create_list_33 = [ttnn_repeat_21]
    return util_create_list_33


def main_const_eval_21(weights, device):
    ttnn_to_device_46 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_46 = ttnn.to_layout(
        ttnn_to_device_46,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_46, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_to_layout_46,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_46, False)
    ttnn_repeat_22 = ttnn.repeat(ttnn_reshape_22, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_22, False)
    util_create_list_34 = [ttnn_repeat_22]
    return util_create_list_34


def main_const_eval_22(device):
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 16, 1280]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_35 = [ttnn_full_1]
    return util_create_list_35


def main_const_eval_23(weights, device):
    ttnn_to_device_47 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_47 = ttnn.to_layout(
        ttnn_to_device_47,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_47, False)
    ttnn_to_device_48 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_48 = ttnn.to_layout(
        ttnn_to_device_48,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_48, False)
    ttnn_to_device_49 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_49 = ttnn.to_layout(
        ttnn_to_device_49,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_49, False)
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_to_layout_47,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_47, False)
    ttnn_repeat_23 = ttnn.repeat(ttnn_reshape_23, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_23, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_to_layout_48,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_48, False)
    ttnn_repeat_24 = ttnn.repeat(ttnn_reshape_24, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_reshape_25 = ttnn.reshape(
        ttnn_to_layout_49,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_49, False)
    ttnn_repeat_25 = ttnn.repeat(ttnn_reshape_25, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_25, False)
    util_create_list_36 = [ttnn_repeat_23, ttnn_repeat_24, ttnn_repeat_25]
    ttnn_concat_13 = ttnn.concat(
        util_create_list_36,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_25, False)
    ttnn.deallocate(ttnn_repeat_24, False)
    ttnn.deallocate(ttnn_repeat_23, False)
    util_create_list_37 = [ttnn_concat_13]
    return util_create_list_37


def main_const_eval_24(weights, device):
    ttnn_to_device_50 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_50 = ttnn.to_layout(
        ttnn_to_device_50,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_50, False)
    ttnn_reshape_26 = ttnn.reshape(
        ttnn_to_layout_50,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_50, False)
    ttnn_repeat_26 = ttnn.repeat(ttnn_reshape_26, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_26, False)
    util_create_list_38 = [ttnn_repeat_26]
    return util_create_list_38


def main_const_eval_25(weights, device):
    ttnn_to_device_51 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_51 = ttnn.to_layout(
        ttnn_to_device_51,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_51, False)
    ttnn_reshape_27 = ttnn.reshape(
        ttnn_to_layout_51,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_51, False)
    ttnn_repeat_27 = ttnn.repeat(ttnn_reshape_27, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_27, False)
    util_create_list_39 = [ttnn_repeat_27]
    return util_create_list_39


def main_const_eval_26(weights, device):
    ttnn_to_device_52 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_52 = ttnn.to_layout(
        ttnn_to_device_52,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_52, False)
    ttnn_to_device_53 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_53 = ttnn.to_layout(
        ttnn_to_device_53,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_53, False)
    ttnn_to_device_54 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_54 = ttnn.to_layout(
        ttnn_to_device_54,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_54, False)
    util_create_list_40 = [ttnn_to_layout_52, ttnn_to_layout_53, ttnn_to_layout_54]
    ttnn_concat_14 = ttnn.concat(
        util_create_list_40,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_54, False)
    ttnn.deallocate(ttnn_to_layout_53, False)
    ttnn.deallocate(ttnn_to_layout_52, False)
    util_create_list_41 = [ttnn_concat_14]
    return util_create_list_41


def main_const_eval_27(weights, device):
    ttnn_to_device_55 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_55 = ttnn.to_layout(
        ttnn_to_device_55,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_55, False)
    ttnn_reshape_28 = ttnn.reshape(
        ttnn_to_layout_55,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_55, False)
    ttnn_repeat_28 = ttnn.repeat(ttnn_reshape_28, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_28, False)
    util_create_list_42 = [ttnn_repeat_28]
    return util_create_list_42


def main_const_eval_28(weights, device):
    ttnn_to_device_56 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_56 = ttnn.to_layout(
        ttnn_to_device_56,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_56, False)
    ttnn_to_device_57 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_57 = ttnn.to_layout(
        ttnn_to_device_57,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_57, False)
    ttnn_to_device_58 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_58 = ttnn.to_layout(
        ttnn_to_device_58,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_58, False)
    ttnn_reshape_29 = ttnn.reshape(
        ttnn_to_layout_56,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_56, False)
    ttnn_repeat_29 = ttnn.repeat(ttnn_reshape_29, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_29, False)
    ttnn_reshape_30 = ttnn.reshape(
        ttnn_to_layout_57,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_57, False)
    ttnn_repeat_30 = ttnn.repeat(ttnn_reshape_30, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_30, False)
    ttnn_reshape_31 = ttnn.reshape(
        ttnn_to_layout_58,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_58, False)
    ttnn_repeat_31 = ttnn.repeat(ttnn_reshape_31, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_31, False)
    util_create_list_43 = [ttnn_repeat_29, ttnn_repeat_30, ttnn_repeat_31]
    ttnn_concat_15 = ttnn.concat(
        util_create_list_43,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_31, False)
    ttnn.deallocate(ttnn_repeat_30, False)
    ttnn.deallocate(ttnn_repeat_29, False)
    util_create_list_44 = [ttnn_concat_15]
    return util_create_list_44


def main_const_eval_29(weights, device):
    ttnn_to_device_59 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_59 = ttnn.to_layout(
        ttnn_to_device_59,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_59, False)
    ttnn_reshape_32 = ttnn.reshape(
        ttnn_to_layout_59,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_59, False)
    ttnn_repeat_32 = ttnn.repeat(ttnn_reshape_32, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_32, False)
    util_create_list_45 = [ttnn_repeat_32]
    return util_create_list_45


def main_const_eval_30(weights, device):
    ttnn_to_device_60 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_60 = ttnn.to_layout(
        ttnn_to_device_60,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_60, False)
    ttnn_to_device_61 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_61 = ttnn.to_layout(
        ttnn_to_device_61,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_61, False)
    ttnn_to_device_62 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_62 = ttnn.to_layout(
        ttnn_to_device_62,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_62, False)
    ttnn_reshape_33 = ttnn.reshape(
        ttnn_to_layout_60,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_60, False)
    ttnn_repeat_33 = ttnn.repeat(ttnn_reshape_33, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_33, False)
    ttnn_reshape_34 = ttnn.reshape(
        ttnn_to_layout_61,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_61, False)
    ttnn_repeat_34 = ttnn.repeat(ttnn_reshape_34, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_34, False)
    ttnn_reshape_35 = ttnn.reshape(
        ttnn_to_layout_62,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_62, False)
    ttnn_repeat_35 = ttnn.repeat(ttnn_reshape_35, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_35, False)
    util_create_list_46 = [ttnn_repeat_33, ttnn_repeat_34, ttnn_repeat_35]
    ttnn_concat_16 = ttnn.concat(
        util_create_list_46,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_35, False)
    ttnn.deallocate(ttnn_repeat_34, False)
    ttnn.deallocate(ttnn_repeat_33, False)
    util_create_list_47 = [ttnn_concat_16]
    return util_create_list_47


def main_const_eval_31(weights, device):
    ttnn_to_device_63 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_63 = ttnn.to_layout(
        ttnn_to_device_63,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_63, False)
    ttnn_reshape_36 = ttnn.reshape(
        ttnn_to_layout_63,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_63, False)
    ttnn_repeat_36 = ttnn.repeat(ttnn_reshape_36, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_36, False)
    util_create_list_48 = [ttnn_repeat_36]
    return util_create_list_48


def main_const_eval_32(device):
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 16, 80, 257]),
        fill_value=0.33437016606330872,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_49 = [ttnn_full_2]
    return util_create_list_49


def main_const_eval_33(weights, device):
    ttnn_to_device_64 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_64 = ttnn.to_layout(
        ttnn_to_device_64,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_64, False)
    ttnn_to_device_65 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_65 = ttnn.to_layout(
        ttnn_to_device_65,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_65, False)
    ttnn_to_device_66 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_66 = ttnn.to_layout(
        ttnn_to_device_66,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_66, False)
    ttnn_reshape_37 = ttnn.reshape(
        ttnn_to_layout_64,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_64, False)
    ttnn_repeat_37 = ttnn.repeat(ttnn_reshape_37, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_37, False)
    ttnn_reshape_38 = ttnn.reshape(
        ttnn_to_layout_65,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_65, False)
    ttnn_repeat_38 = ttnn.repeat(ttnn_reshape_38, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_38, False)
    ttnn_reshape_39 = ttnn.reshape(
        ttnn_to_layout_66,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_66, False)
    ttnn_repeat_39 = ttnn.repeat(ttnn_reshape_39, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_39, False)
    util_create_list_50 = [ttnn_repeat_37, ttnn_repeat_38, ttnn_repeat_39]
    ttnn_concat_17 = ttnn.concat(
        util_create_list_50,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_39, False)
    ttnn.deallocate(ttnn_repeat_38, False)
    ttnn.deallocate(ttnn_repeat_37, False)
    util_create_list_51 = [ttnn_concat_17]
    return util_create_list_51


def main_const_eval_34(weights, device):
    ttnn_to_device_67 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_67 = ttnn.to_layout(
        ttnn_to_device_67,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_67, False)
    ttnn_reshape_40 = ttnn.reshape(
        ttnn_to_layout_67,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_67, False)
    ttnn_repeat_40 = ttnn.repeat(ttnn_reshape_40, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_40, False)
    util_create_list_52 = [ttnn_repeat_40]
    return util_create_list_52


def main_const_eval_35(weights, device):
    ttnn_to_device_68 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_68 = ttnn.to_layout(
        ttnn_to_device_68,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_68, False)
    ttnn_to_device_69 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_69 = ttnn.to_layout(
        ttnn_to_device_69,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_69, False)
    ttnn_to_device_70 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_70 = ttnn.to_layout(
        ttnn_to_device_70,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_70, False)
    ttnn_reshape_41 = ttnn.reshape(
        ttnn_to_layout_68,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_68, False)
    ttnn_repeat_41 = ttnn.repeat(ttnn_reshape_41, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_41, False)
    ttnn_reshape_42 = ttnn.reshape(
        ttnn_to_layout_69,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_69, False)
    ttnn_repeat_42 = ttnn.repeat(ttnn_reshape_42, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_42, False)
    ttnn_reshape_43 = ttnn.reshape(
        ttnn_to_layout_70,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_70, False)
    ttnn_repeat_43 = ttnn.repeat(ttnn_reshape_43, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_43, False)
    util_create_list_53 = [ttnn_repeat_41, ttnn_repeat_42, ttnn_repeat_43]
    ttnn_concat_18 = ttnn.concat(
        util_create_list_53,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_43, False)
    ttnn.deallocate(ttnn_repeat_42, False)
    ttnn.deallocate(ttnn_repeat_41, False)
    util_create_list_54 = [ttnn_concat_18]
    return util_create_list_54


def main_const_eval_36(weights, device):
    ttnn_to_device_71 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_71 = ttnn.to_layout(
        ttnn_to_device_71,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_71, False)
    ttnn_to_device_72 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_72 = ttnn.to_layout(
        ttnn_to_device_72,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_72, False)
    ttnn_to_device_73 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_73 = ttnn.to_layout(
        ttnn_to_device_73,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_73, False)
    util_create_list_55 = [ttnn_to_layout_71, ttnn_to_layout_72, ttnn_to_layout_73]
    ttnn_concat_19 = ttnn.concat(
        util_create_list_55,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_73, False)
    ttnn.deallocate(ttnn_to_layout_72, False)
    ttnn.deallocate(ttnn_to_layout_71, False)
    util_create_list_56 = [ttnn_concat_19]
    return util_create_list_56


def main_const_eval_37(weights, device):
    ttnn_to_device_74 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_74 = ttnn.to_layout(
        ttnn_to_device_74,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_74, False)
    ttnn_reshape_44 = ttnn.reshape(
        ttnn_to_layout_74,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_74, False)
    ttnn_repeat_44 = ttnn.repeat(ttnn_reshape_44, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_44, False)
    util_create_list_57 = [ttnn_repeat_44]
    return util_create_list_57


def main_const_eval_38(weights, device):
    ttnn_to_device_75 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_75 = ttnn.to_layout(
        ttnn_to_device_75,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_75, False)
    ttnn_reshape_45 = ttnn.reshape(
        ttnn_to_layout_75,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_75, False)
    ttnn_repeat_45 = ttnn.repeat(ttnn_reshape_45, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_45, False)
    util_create_list_58 = [ttnn_repeat_45]
    return util_create_list_58


def main_const_eval_39(weights, device):
    ttnn_to_device_76 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_76 = ttnn.to_layout(
        ttnn_to_device_76,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_76, False)
    ttnn_reshape_46 = ttnn.reshape(
        ttnn_to_layout_76,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_76, False)
    ttnn_repeat_46 = ttnn.repeat(ttnn_reshape_46, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_46, False)
    util_create_list_59 = [ttnn_repeat_46]
    return util_create_list_59


def main_const_eval_40(weights, device):
    ttnn_to_device_77 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_77 = ttnn.to_layout(
        ttnn_to_device_77,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_77, False)
    ttnn_reshape_47 = ttnn.reshape(
        ttnn_to_layout_77,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_77, False)
    ttnn_repeat_47 = ttnn.repeat(ttnn_reshape_47, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_47, False)
    util_create_list_60 = [ttnn_repeat_47]
    return util_create_list_60


def main_const_eval_41(weights, device):
    ttnn_to_device_78 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_78 = ttnn.to_layout(
        ttnn_to_device_78,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_78, False)
    ttnn_reshape_48 = ttnn.reshape(
        ttnn_to_layout_78,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_78, False)
    ttnn_repeat_48 = ttnn.repeat(ttnn_reshape_48, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_48, False)
    util_create_list_61 = [ttnn_repeat_48]
    return util_create_list_61


def main_const_eval_42(weights, device):
    ttnn_to_device_79 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_79 = ttnn.to_layout(
        ttnn_to_device_79,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_79, False)
    ttnn_reshape_49 = ttnn.reshape(
        ttnn_to_layout_79,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_79, False)
    ttnn_repeat_49 = ttnn.repeat(ttnn_reshape_49, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_49, False)
    util_create_list_62 = [ttnn_repeat_49]
    return util_create_list_62


def main_const_eval_43(weights, device):
    ttnn_to_device_80 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_80 = ttnn.to_layout(
        ttnn_to_device_80,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_80, False)
    ttnn_to_device_81 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_81 = ttnn.to_layout(
        ttnn_to_device_81,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_81, False)
    ttnn_to_device_82 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_82 = ttnn.to_layout(
        ttnn_to_device_82,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_82, False)
    ttnn_reshape_50 = ttnn.reshape(
        ttnn_to_layout_80,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_80, False)
    ttnn_repeat_50 = ttnn.repeat(ttnn_reshape_50, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_50, False)
    ttnn_reshape_51 = ttnn.reshape(
        ttnn_to_layout_81,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_81, False)
    ttnn_repeat_51 = ttnn.repeat(ttnn_reshape_51, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_51, False)
    ttnn_reshape_52 = ttnn.reshape(
        ttnn_to_layout_82,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_82, False)
    ttnn_repeat_52 = ttnn.repeat(ttnn_reshape_52, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_52, False)
    util_create_list_63 = [ttnn_repeat_50, ttnn_repeat_51, ttnn_repeat_52]
    ttnn_concat_20 = ttnn.concat(
        util_create_list_63,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_52, False)
    ttnn.deallocate(ttnn_repeat_51, False)
    ttnn.deallocate(ttnn_repeat_50, False)
    util_create_list_64 = [ttnn_concat_20]
    return util_create_list_64


def main_const_eval_44(weights, device):
    ttnn_to_device_83 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_83 = ttnn.to_layout(
        ttnn_to_device_83,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_83, False)
    ttnn_to_device_84 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_84 = ttnn.to_layout(
        ttnn_to_device_84,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_84, False)
    ttnn_to_device_85 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_85 = ttnn.to_layout(
        ttnn_to_device_85,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_85, False)
    ttnn_reshape_53 = ttnn.reshape(
        ttnn_to_layout_83,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_83, False)
    ttnn_repeat_53 = ttnn.repeat(ttnn_reshape_53, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_53, False)
    ttnn_reshape_54 = ttnn.reshape(
        ttnn_to_layout_84,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_84, False)
    ttnn_repeat_54 = ttnn.repeat(ttnn_reshape_54, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_54, False)
    ttnn_reshape_55 = ttnn.reshape(
        ttnn_to_layout_85,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_85, False)
    ttnn_repeat_55 = ttnn.repeat(ttnn_reshape_55, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_55, False)
    util_create_list_65 = [ttnn_repeat_53, ttnn_repeat_54, ttnn_repeat_55]
    ttnn_concat_21 = ttnn.concat(
        util_create_list_65,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_55, False)
    ttnn.deallocate(ttnn_repeat_54, False)
    ttnn.deallocate(ttnn_repeat_53, False)
    util_create_list_66 = [ttnn_concat_21]
    return util_create_list_66


def main_const_eval_45(weights, device):
    ttnn_to_device_86 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_86 = ttnn.to_layout(
        ttnn_to_device_86,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_86, False)
    ttnn_to_device_87 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_87 = ttnn.to_layout(
        ttnn_to_device_87,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_87, False)
    ttnn_to_device_88 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_88 = ttnn.to_layout(
        ttnn_to_device_88,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_88, False)
    util_create_list_67 = [ttnn_to_layout_86, ttnn_to_layout_87, ttnn_to_layout_88]
    ttnn_concat_22 = ttnn.concat(
        util_create_list_67,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_88, False)
    ttnn.deallocate(ttnn_to_layout_87, False)
    ttnn.deallocate(ttnn_to_layout_86, False)
    util_create_list_68 = [ttnn_concat_22]
    return util_create_list_68


def main_const_eval_46(weights, device):
    ttnn_to_device_89 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_89 = ttnn.to_layout(
        ttnn_to_device_89,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_89, False)
    ttnn_reshape_56 = ttnn.reshape(
        ttnn_to_layout_89,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_89, False)
    ttnn_repeat_56 = ttnn.repeat(ttnn_reshape_56, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_56, False)
    util_create_list_69 = [ttnn_repeat_56]
    return util_create_list_69


def main_const_eval_47(weights, device):
    ttnn_to_device_90 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_90 = ttnn.to_layout(
        ttnn_to_device_90,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_90, False)
    ttnn_reshape_57 = ttnn.reshape(
        ttnn_to_layout_90,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_90, False)
    ttnn_repeat_57 = ttnn.repeat(ttnn_reshape_57, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_57, False)
    util_create_list_70 = [ttnn_repeat_57]
    return util_create_list_70


def main_const_eval_48(weights, device):
    ttnn_to_device_91 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_91 = ttnn.to_layout(
        ttnn_to_device_91,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_91, False)
    ttnn_reshape_58 = ttnn.reshape(
        ttnn_to_layout_91,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_91, False)
    ttnn_repeat_58 = ttnn.repeat(ttnn_reshape_58, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_58, False)
    util_create_list_71 = [ttnn_repeat_58]
    return util_create_list_71


def main_const_eval_49(weights, device):
    ttnn_prepare_conv_weights_0 = ttnn.prepare_conv_weights(
        weight_tensor=weights[0],
        input_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
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


def main_const_eval_50(weights, device):
    ttnn_to_device_92 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_92 = ttnn.to_layout(
        ttnn_to_device_92,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_92, False)
    ttnn_to_device_93 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_93 = ttnn.to_layout(
        ttnn_to_device_93,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_93, False)
    ttnn_to_device_94 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_94 = ttnn.to_layout(
        ttnn_to_device_94,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_94, False)
    util_create_list_73 = [ttnn_to_layout_92, ttnn_to_layout_93, ttnn_to_layout_94]
    ttnn_concat_23 = ttnn.concat(
        util_create_list_73,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_94, False)
    ttnn.deallocate(ttnn_to_layout_93, False)
    ttnn.deallocate(ttnn_to_layout_92, False)
    util_create_list_74 = [ttnn_concat_23]
    return util_create_list_74


def main_const_eval_51(weights, device):
    ttnn_to_device_95 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_95 = ttnn.to_layout(
        ttnn_to_device_95,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_95, False)
    ttnn_reshape_59 = ttnn.reshape(
        ttnn_to_layout_95,
        [1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_95, False)
    ttnn_repeat_59 = ttnn.repeat(ttnn_reshape_59, ttnn.Shape([1, 16, 1]))
    ttnn.deallocate(ttnn_reshape_59, False)
    util_create_list_75 = [ttnn_repeat_59]
    return util_create_list_75


def main_const_eval_52(weights, device):
    ttnn_to_device_96 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_96 = ttnn.to_layout(
        ttnn_to_device_96,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_96, False)
    ttnn_reshape_60 = ttnn.reshape(
        ttnn_to_layout_96,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_96, False)
    ttnn_repeat_60 = ttnn.repeat(ttnn_reshape_60, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_60, False)
    util_create_list_76 = [ttnn_repeat_60]
    return util_create_list_76


def main_const_eval_53(device):
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 273]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_77 = [ttnn_full_3]
    return util_create_list_77


def main_const_eval_54(weights, device):
    ttnn_to_device_97 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_97 = ttnn.to_layout(
        ttnn_to_device_97,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_97, False)
    ttnn_to_device_98 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_98 = ttnn.to_layout(
        ttnn_to_device_98,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_98, False)
    ttnn_to_device_99 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_99 = ttnn.to_layout(
        ttnn_to_device_99,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_99, False)
    ttnn_reshape_61 = ttnn.reshape(
        ttnn_to_layout_97,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_97, False)
    ttnn_repeat_61 = ttnn.repeat(ttnn_reshape_61, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_61, False)
    ttnn_reshape_62 = ttnn.reshape(
        ttnn_to_layout_98,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_98, False)
    ttnn_repeat_62 = ttnn.repeat(ttnn_reshape_62, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_62, False)
    ttnn_reshape_63 = ttnn.reshape(
        ttnn_to_layout_99,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_99, False)
    ttnn_repeat_63 = ttnn.repeat(ttnn_reshape_63, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_63, False)
    util_create_list_78 = [ttnn_repeat_61, ttnn_repeat_62, ttnn_repeat_63]
    ttnn_concat_24 = ttnn.concat(
        util_create_list_78,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_63, False)
    ttnn.deallocate(ttnn_repeat_62, False)
    ttnn.deallocate(ttnn_repeat_61, False)
    util_create_list_79 = [ttnn_concat_24]
    return util_create_list_79


def main_const_eval_55(weights, device):
    ttnn_to_device_100 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_100 = ttnn.to_layout(
        ttnn_to_device_100,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_100, False)
    ttnn_reshape_64 = ttnn.reshape(
        ttnn_to_layout_100,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_100, False)
    ttnn_repeat_64 = ttnn.repeat(ttnn_reshape_64, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_64, False)
    util_create_list_80 = [ttnn_repeat_64]
    return util_create_list_80


def main_const_eval_56(weights, device):
    ttnn_to_device_101 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_101 = ttnn.to_layout(
        ttnn_to_device_101,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_101, False)
    ttnn_reshape_65 = ttnn.reshape(
        ttnn_to_layout_101,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_101, False)
    ttnn_repeat_65 = ttnn.repeat(ttnn_reshape_65, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_65, False)
    util_create_list_81 = [ttnn_repeat_65]
    return util_create_list_81


def main_const_eval_57(weights, device):
    ttnn_to_device_102 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_102 = ttnn.to_layout(
        ttnn_to_device_102,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_102, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_to_layout_102,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_102, False)
    ttnn_to_layout_103 = ttnn.to_layout(
        ttnn_typecast_0,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_to_device_103 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_embedding_0 = ttnn.embedding(ttnn_to_layout_103, ttnn_to_device_103, layout=ttnn.Layout.TILE)
    ttnn.deallocate(ttnn_to_device_103, False)
    ttnn.deallocate(ttnn_to_layout_103, False)
    ttnn_permute_0 = ttnn.permute(
        ttnn_embedding_0,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    util_create_list_82 = [ttnn_permute_0]
    return util_create_list_82


def main_const_eval_58(weights, device):
    ttnn_to_device_104 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_104 = ttnn.to_layout(
        ttnn_to_device_104,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_104, False)
    ttnn_to_device_105 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_105 = ttnn.to_layout(
        ttnn_to_device_105,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_105, False)
    ttnn_to_device_106 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_106 = ttnn.to_layout(
        ttnn_to_device_106,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_106, False)
    ttnn_reshape_66 = ttnn.reshape(
        ttnn_to_layout_104,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_104, False)
    ttnn_repeat_66 = ttnn.repeat(ttnn_reshape_66, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_66, False)
    ttnn_reshape_67 = ttnn.reshape(
        ttnn_to_layout_105,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_105, False)
    ttnn_repeat_67 = ttnn.repeat(ttnn_reshape_67, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_67, False)
    ttnn_reshape_68 = ttnn.reshape(
        ttnn_to_layout_106,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_106, False)
    ttnn_repeat_68 = ttnn.repeat(ttnn_reshape_68, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_68, False)
    util_create_list_83 = [ttnn_repeat_66, ttnn_repeat_67, ttnn_repeat_68]
    ttnn_concat_25 = ttnn.concat(
        util_create_list_83,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_68, False)
    ttnn.deallocate(ttnn_repeat_67, False)
    ttnn.deallocate(ttnn_repeat_66, False)
    util_create_list_84 = [ttnn_concat_25]
    return util_create_list_84


def main_const_eval_59(device):
    ttnn_full_4 = ttnn.full(
        shape=ttnn.Shape([1, 16, 257, 257]),
        fill_value=0.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_85 = [ttnn_full_4]
    return util_create_list_85


def main_const_eval_60(weights, device):
    ttnn_to_device_107 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_107 = ttnn.to_layout(
        ttnn_to_device_107,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_107, False)
    ttnn_to_device_108 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_108 = ttnn.to_layout(
        ttnn_to_device_108,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_108, False)
    ttnn_to_device_109 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_109 = ttnn.to_layout(
        ttnn_to_device_109,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_109, False)
    util_create_list_86 = [ttnn_to_layout_107, ttnn_to_layout_108, ttnn_to_layout_109]
    ttnn_concat_26 = ttnn.concat(
        util_create_list_86,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_109, False)
    ttnn.deallocate(ttnn_to_layout_108, False)
    ttnn.deallocate(ttnn_to_layout_107, False)
    util_create_list_87 = [ttnn_concat_26]
    return util_create_list_87


def main_const_eval_61(weights, device):
    ttnn_to_device_110 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_110 = ttnn.to_layout(
        ttnn_to_device_110,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_110, False)
    ttnn_reshape_69 = ttnn.reshape(
        ttnn_to_layout_110,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_110, False)
    ttnn_repeat_69 = ttnn.repeat(ttnn_reshape_69, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_69, False)
    util_create_list_88 = [ttnn_repeat_69]
    return util_create_list_88


def main_const_eval_62(weights, device):
    ttnn_to_device_111 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_111 = ttnn.to_layout(
        ttnn_to_device_111,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_111, False)
    ttnn_reshape_70 = ttnn.reshape(
        ttnn_to_layout_111,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_111, False)
    ttnn_repeat_70 = ttnn.repeat(ttnn_reshape_70, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_70, False)
    util_create_list_89 = [ttnn_repeat_70]
    return util_create_list_89


def main_const_eval_63(weights, device):
    ttnn_to_device_112 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_112 = ttnn.to_layout(
        ttnn_to_device_112,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_112, False)
    ttnn_reshape_71 = ttnn.reshape(
        ttnn_to_layout_112,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_112, False)
    ttnn_repeat_71 = ttnn.repeat(ttnn_reshape_71, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_71, False)
    util_create_list_90 = [ttnn_repeat_71]
    return util_create_list_90


def main_const_eval_64(weights, device):
    ttnn_to_device_113 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_113 = ttnn.to_layout(
        ttnn_to_device_113,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_113, False)
    ttnn_reshape_72 = ttnn.reshape(
        ttnn_to_layout_113,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_113, False)
    ttnn_repeat_72 = ttnn.repeat(ttnn_reshape_72, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_72, False)
    util_create_list_91 = [ttnn_repeat_72]
    return util_create_list_91


def main_const_eval_65(weights, device):
    ttnn_to_device_114 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_114 = ttnn.to_layout(
        ttnn_to_device_114,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_114, False)
    ttnn_reshape_73 = ttnn.reshape(
        ttnn_to_layout_114,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_114, False)
    ttnn_repeat_73 = ttnn.repeat(ttnn_reshape_73, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_73, False)
    util_create_list_92 = [ttnn_repeat_73]
    return util_create_list_92


def main_const_eval_66(weights, device):
    ttnn_to_device_115 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_115 = ttnn.to_layout(
        ttnn_to_device_115,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_115, False)
    ttnn_reshape_74 = ttnn.reshape(
        ttnn_to_layout_115,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_115, False)
    ttnn_repeat_74 = ttnn.repeat(ttnn_reshape_74, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_74, False)
    util_create_list_93 = [ttnn_repeat_74]
    return util_create_list_93


def main_const_eval_67(weights, device):
    ttnn_to_device_116 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_116 = ttnn.to_layout(
        ttnn_to_device_116,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_116, False)
    ttnn_reshape_75 = ttnn.reshape(
        ttnn_to_layout_116,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_116, False)
    ttnn_repeat_75 = ttnn.repeat(ttnn_reshape_75, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_75, False)
    util_create_list_94 = [ttnn_repeat_75]
    return util_create_list_94


def main_const_eval_68(weights, device):
    ttnn_to_device_117 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_117 = ttnn.to_layout(
        ttnn_to_device_117,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_117, False)
    ttnn_reshape_76 = ttnn.reshape(
        ttnn_to_layout_117,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_117, False)
    ttnn_repeat_76 = ttnn.repeat(ttnn_reshape_76, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_76, False)
    util_create_list_95 = [ttnn_repeat_76]
    return util_create_list_95


def main_const_eval_69(weights, device):
    ttnn_to_device_118 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_118 = ttnn.to_layout(
        ttnn_to_device_118,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_118, False)
    ttnn_reshape_77 = ttnn.reshape(
        ttnn_to_layout_118,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_118, False)
    ttnn_repeat_77 = ttnn.repeat(ttnn_reshape_77, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_77, False)
    util_create_list_96 = [ttnn_repeat_77]
    return util_create_list_96


def main_const_eval_70(weights, device):
    ttnn_to_device_119 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_119 = ttnn.to_layout(
        ttnn_to_device_119,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_119, False)
    ttnn_to_device_120 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_120 = ttnn.to_layout(
        ttnn_to_device_120,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_120, False)
    ttnn_to_device_121 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_121 = ttnn.to_layout(
        ttnn_to_device_121,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_121, False)
    ttnn_reshape_78 = ttnn.reshape(
        ttnn_to_layout_119,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_119, False)
    ttnn_repeat_78 = ttnn.repeat(ttnn_reshape_78, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_78, False)
    ttnn_reshape_79 = ttnn.reshape(
        ttnn_to_layout_120,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_120, False)
    ttnn_repeat_79 = ttnn.repeat(ttnn_reshape_79, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_79, False)
    ttnn_reshape_80 = ttnn.reshape(
        ttnn_to_layout_121,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_121, False)
    ttnn_repeat_80 = ttnn.repeat(ttnn_reshape_80, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_80, False)
    util_create_list_97 = [ttnn_repeat_78, ttnn_repeat_79, ttnn_repeat_80]
    ttnn_concat_27 = ttnn.concat(
        util_create_list_97,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_80, False)
    ttnn.deallocate(ttnn_repeat_79, False)
    ttnn.deallocate(ttnn_repeat_78, False)
    util_create_list_98 = [ttnn_concat_27]
    return util_create_list_98


def main_const_eval_71(weights, device):
    ttnn_to_device_122 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_122 = ttnn.to_layout(
        ttnn_to_device_122,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_122, False)
    ttnn_reshape_81 = ttnn.reshape(
        ttnn_to_layout_122,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_122, False)
    ttnn_repeat_81 = ttnn.repeat(ttnn_reshape_81, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_81, False)
    util_create_list_99 = [ttnn_repeat_81]
    return util_create_list_99


def main_const_eval_72(weights, device):
    ttnn_to_device_123 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_123 = ttnn.to_layout(
        ttnn_to_device_123,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_123, False)
    ttnn_to_device_124 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_124 = ttnn.to_layout(
        ttnn_to_device_124,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_124, False)
    ttnn_to_device_125 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_125 = ttnn.to_layout(
        ttnn_to_device_125,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_125, False)
    ttnn_reshape_82 = ttnn.reshape(
        ttnn_to_layout_123,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_123, False)
    ttnn_repeat_82 = ttnn.repeat(ttnn_reshape_82, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_82, False)
    ttnn_reshape_83 = ttnn.reshape(
        ttnn_to_layout_124,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_124, False)
    ttnn_repeat_83 = ttnn.repeat(ttnn_reshape_83, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_83, False)
    ttnn_reshape_84 = ttnn.reshape(
        ttnn_to_layout_125,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_125, False)
    ttnn_repeat_84 = ttnn.repeat(ttnn_reshape_84, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_84, False)
    util_create_list_100 = [ttnn_repeat_82, ttnn_repeat_83, ttnn_repeat_84]
    ttnn_concat_28 = ttnn.concat(
        util_create_list_100,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_84, False)
    ttnn.deallocate(ttnn_repeat_83, False)
    ttnn.deallocate(ttnn_repeat_82, False)
    util_create_list_101 = [ttnn_concat_28]
    return util_create_list_101


def main_const_eval_73(weights, device):
    ttnn_to_device_126 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_126 = ttnn.to_layout(
        ttnn_to_device_126,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_126, False)
    ttnn_reshape_85 = ttnn.reshape(
        ttnn_to_layout_126,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_126, False)
    ttnn_repeat_85 = ttnn.repeat(ttnn_reshape_85, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_85, False)
    util_create_list_102 = [ttnn_repeat_85]
    return util_create_list_102


def main_const_eval_74(weights, device):
    ttnn_to_device_127 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_127 = ttnn.to_layout(
        ttnn_to_device_127,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_127, False)
    ttnn_to_device_128 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_128 = ttnn.to_layout(
        ttnn_to_device_128,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_128, False)
    ttnn_to_device_129 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_129 = ttnn.to_layout(
        ttnn_to_device_129,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_129, False)
    util_create_list_103 = [ttnn_to_layout_127, ttnn_to_layout_128, ttnn_to_layout_129]
    ttnn_concat_29 = ttnn.concat(
        util_create_list_103,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_129, False)
    ttnn.deallocate(ttnn_to_layout_128, False)
    ttnn.deallocate(ttnn_to_layout_127, False)
    util_create_list_104 = [ttnn_concat_29]
    return util_create_list_104


def main_const_eval_75(weights, device):
    ttnn_to_device_130 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_130 = ttnn.to_layout(
        ttnn_to_device_130,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_130, False)
    ttnn_reshape_86 = ttnn.reshape(
        ttnn_to_layout_130,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_130, False)
    ttnn_repeat_86 = ttnn.repeat(ttnn_reshape_86, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_86, False)
    util_create_list_105 = [ttnn_repeat_86]
    return util_create_list_105


def main_const_eval_76(weights, device):
    ttnn_to_device_131 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_131 = ttnn.to_layout(
        ttnn_to_device_131,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_131, False)
    ttnn_to_device_132 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_132 = ttnn.to_layout(
        ttnn_to_device_132,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_132, False)
    ttnn_to_device_133 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_133 = ttnn.to_layout(
        ttnn_to_device_133,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_133, False)
    ttnn_reshape_87 = ttnn.reshape(
        ttnn_to_layout_131,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_131, False)
    ttnn_repeat_87 = ttnn.repeat(ttnn_reshape_87, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_87, False)
    ttnn_reshape_88 = ttnn.reshape(
        ttnn_to_layout_132,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_132, False)
    ttnn_repeat_88 = ttnn.repeat(ttnn_reshape_88, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_88, False)
    ttnn_reshape_89 = ttnn.reshape(
        ttnn_to_layout_133,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_133, False)
    ttnn_repeat_89 = ttnn.repeat(ttnn_reshape_89, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_89, False)
    util_create_list_106 = [ttnn_repeat_87, ttnn_repeat_88, ttnn_repeat_89]
    ttnn_concat_30 = ttnn.concat(
        util_create_list_106,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_89, False)
    ttnn.deallocate(ttnn_repeat_88, False)
    ttnn.deallocate(ttnn_repeat_87, False)
    util_create_list_107 = [ttnn_concat_30]
    return util_create_list_107


def main_const_eval_77(weights, device):
    ttnn_to_device_134 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_134 = ttnn.to_layout(
        ttnn_to_device_134,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_134, False)
    ttnn_to_device_135 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_135 = ttnn.to_layout(
        ttnn_to_device_135,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_135, False)
    ttnn_to_device_136 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_136 = ttnn.to_layout(
        ttnn_to_device_136,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_136, False)
    util_create_list_108 = [ttnn_to_layout_134, ttnn_to_layout_135, ttnn_to_layout_136]
    ttnn_concat_31 = ttnn.concat(
        util_create_list_108,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_136, False)
    ttnn.deallocate(ttnn_to_layout_135, False)
    ttnn.deallocate(ttnn_to_layout_134, False)
    util_create_list_109 = [ttnn_concat_31]
    return util_create_list_109


def main_const_eval_78(weights, device):
    ttnn_to_device_137 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_137 = ttnn.to_layout(
        ttnn_to_device_137,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_137, False)
    ttnn_to_device_138 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_138 = ttnn.to_layout(
        ttnn_to_device_138,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_138, False)
    ttnn_to_device_139 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_139 = ttnn.to_layout(
        ttnn_to_device_139,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_139, False)
    ttnn_reshape_90 = ttnn.reshape(
        ttnn_to_layout_137,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_137, False)
    ttnn_repeat_90 = ttnn.repeat(ttnn_reshape_90, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_90, False)
    ttnn_reshape_91 = ttnn.reshape(
        ttnn_to_layout_138,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_138, False)
    ttnn_repeat_91 = ttnn.repeat(ttnn_reshape_91, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_91, False)
    ttnn_reshape_92 = ttnn.reshape(
        ttnn_to_layout_139,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_139, False)
    ttnn_repeat_92 = ttnn.repeat(ttnn_reshape_92, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_92, False)
    util_create_list_110 = [ttnn_repeat_90, ttnn_repeat_91, ttnn_repeat_92]
    ttnn_concat_32 = ttnn.concat(
        util_create_list_110,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_92, False)
    ttnn.deallocate(ttnn_repeat_91, False)
    ttnn.deallocate(ttnn_repeat_90, False)
    util_create_list_111 = [ttnn_concat_32]
    return util_create_list_111


def main_const_eval_79(weights, device):
    ttnn_to_device_140 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_140 = ttnn.to_layout(
        ttnn_to_device_140,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_140, False)
    ttnn_reshape_93 = ttnn.reshape(
        ttnn_to_layout_140,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_140, False)
    ttnn_repeat_93 = ttnn.repeat(ttnn_reshape_93, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_93, False)
    util_create_list_112 = [ttnn_repeat_93]
    return util_create_list_112


def main_const_eval_80(weights, device):
    ttnn_to_device_141 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_141 = ttnn.to_layout(
        ttnn_to_device_141,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_141, False)
    ttnn_to_device_142 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_142 = ttnn.to_layout(
        ttnn_to_device_142,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_142, False)
    ttnn_to_device_143 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_143 = ttnn.to_layout(
        ttnn_to_device_143,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_143, False)
    ttnn_reshape_94 = ttnn.reshape(
        ttnn_to_layout_141,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_141, False)
    ttnn_repeat_94 = ttnn.repeat(ttnn_reshape_94, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_94, False)
    ttnn_reshape_95 = ttnn.reshape(
        ttnn_to_layout_142,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_142, False)
    ttnn_repeat_95 = ttnn.repeat(ttnn_reshape_95, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_95, False)
    ttnn_reshape_96 = ttnn.reshape(
        ttnn_to_layout_143,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_143, False)
    ttnn_repeat_96 = ttnn.repeat(ttnn_reshape_96, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_96, False)
    util_create_list_113 = [ttnn_repeat_94, ttnn_repeat_95, ttnn_repeat_96]
    ttnn_concat_33 = ttnn.concat(
        util_create_list_113,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_96, False)
    ttnn.deallocate(ttnn_repeat_95, False)
    ttnn.deallocate(ttnn_repeat_94, False)
    util_create_list_114 = [ttnn_concat_33]
    return util_create_list_114


def main_const_eval_81(weights, device):
    ttnn_to_device_144 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_144 = ttnn.to_layout(
        ttnn_to_device_144,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_144, False)
    ttnn_reshape_97 = ttnn.reshape(
        ttnn_to_layout_144,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_144, False)
    ttnn_repeat_97 = ttnn.repeat(ttnn_reshape_97, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_97, False)
    util_create_list_115 = [ttnn_repeat_97]
    return util_create_list_115


def main_const_eval_82(weights, device):
    ttnn_to_device_145 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_145 = ttnn.to_layout(
        ttnn_to_device_145,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_145, False)
    ttnn_reshape_98 = ttnn.reshape(
        ttnn_to_layout_145,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_145, False)
    ttnn_repeat_98 = ttnn.repeat(ttnn_reshape_98, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_98, False)
    util_create_list_116 = [ttnn_repeat_98]
    return util_create_list_116


def main_const_eval_83(weights, device):
    ttnn_to_device_146 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_146 = ttnn.to_layout(
        ttnn_to_device_146,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_146, False)
    ttnn_reshape_99 = ttnn.reshape(
        ttnn_to_layout_146,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_146, False)
    ttnn_repeat_99 = ttnn.repeat(ttnn_reshape_99, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_99, False)
    util_create_list_117 = [ttnn_repeat_99]
    return util_create_list_117


def main_const_eval_84(weights, device):
    ttnn_to_device_147 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_147 = ttnn.to_layout(
        ttnn_to_device_147,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_147, False)
    ttnn_reshape_100 = ttnn.reshape(
        ttnn_to_layout_147,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_147, False)
    ttnn_repeat_100 = ttnn.repeat(ttnn_reshape_100, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_100, False)
    util_create_list_118 = [ttnn_repeat_100]
    return util_create_list_118


def main_const_eval_85(weights, device):
    ttnn_to_device_148 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_148 = ttnn.to_layout(
        ttnn_to_device_148,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_148, False)
    ttnn_reshape_101 = ttnn.reshape(
        ttnn_to_layout_148,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_148, False)
    ttnn_repeat_101 = ttnn.repeat(ttnn_reshape_101, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_101, False)
    util_create_list_119 = [ttnn_repeat_101]
    return util_create_list_119


def main_const_eval_86(weights, device):
    ttnn_to_device_149 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_149 = ttnn.to_layout(
        ttnn_to_device_149,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_149, False)
    ttnn_reshape_102 = ttnn.reshape(
        ttnn_to_layout_149,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_149, False)
    ttnn_repeat_102 = ttnn.repeat(ttnn_reshape_102, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_102, False)
    util_create_list_120 = [ttnn_repeat_102]
    return util_create_list_120


def main_const_eval_87(weights, device):
    ttnn_to_device_150 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_150 = ttnn.to_layout(
        ttnn_to_device_150,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_150, False)
    ttnn_reshape_103 = ttnn.reshape(
        ttnn_to_layout_150,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_150, False)
    ttnn_repeat_103 = ttnn.repeat(ttnn_reshape_103, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_103, False)
    util_create_list_121 = [ttnn_repeat_103]
    return util_create_list_121


def main_const_eval_88(weights, device):
    ttnn_to_device_151 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_151 = ttnn.to_layout(
        ttnn_to_device_151,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_151, False)
    ttnn_reshape_104 = ttnn.reshape(
        ttnn_to_layout_151,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_151, False)
    ttnn_repeat_104 = ttnn.repeat(ttnn_reshape_104, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_104, False)
    util_create_list_122 = [ttnn_repeat_104]
    return util_create_list_122


def main_const_eval_89(weights, device):
    ttnn_to_device_152 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_152 = ttnn.to_layout(
        ttnn_to_device_152,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_152, False)
    ttnn_to_device_153 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_153 = ttnn.to_layout(
        ttnn_to_device_153,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_153, False)
    ttnn_to_device_154 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_154 = ttnn.to_layout(
        ttnn_to_device_154,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_154, False)
    util_create_list_123 = [ttnn_to_layout_152, ttnn_to_layout_153, ttnn_to_layout_154]
    ttnn_concat_34 = ttnn.concat(
        util_create_list_123,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_154, False)
    ttnn.deallocate(ttnn_to_layout_153, False)
    ttnn.deallocate(ttnn_to_layout_152, False)
    util_create_list_124 = [ttnn_concat_34]
    return util_create_list_124


def main_const_eval_90(weights, device):
    ttnn_to_device_155 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_155 = ttnn.to_layout(
        ttnn_to_device_155,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_155, False)
    ttnn_reshape_105 = ttnn.reshape(
        ttnn_to_layout_155,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_155, False)
    ttnn_repeat_105 = ttnn.repeat(ttnn_reshape_105, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_105, False)
    util_create_list_125 = [ttnn_repeat_105]
    return util_create_list_125


def main_const_eval_91(weights, device):
    ttnn_to_device_156 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_156 = ttnn.to_layout(
        ttnn_to_device_156,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_156, False)
    ttnn_reshape_106 = ttnn.reshape(
        ttnn_to_layout_156,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_156, False)
    ttnn_repeat_106 = ttnn.repeat(ttnn_reshape_106, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_106, False)
    util_create_list_126 = [ttnn_repeat_106]
    return util_create_list_126


def main_const_eval_92(weights, device):
    ttnn_to_device_157 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_157 = ttnn.to_layout(
        ttnn_to_device_157,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_157, False)
    ttnn_reshape_107 = ttnn.reshape(
        ttnn_to_layout_157,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_157, False)
    ttnn_repeat_107 = ttnn.repeat(ttnn_reshape_107, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_107, False)
    util_create_list_127 = [ttnn_repeat_107]
    return util_create_list_127


def main_const_eval_93(weights, device):
    ttnn_to_device_158 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_158 = ttnn.to_layout(
        ttnn_to_device_158,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_158, False)
    ttnn_to_device_159 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_159 = ttnn.to_layout(
        ttnn_to_device_159,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_159, False)
    ttnn_to_device_160 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_160 = ttnn.to_layout(
        ttnn_to_device_160,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_160, False)
    util_create_list_128 = [ttnn_to_layout_158, ttnn_to_layout_159, ttnn_to_layout_160]
    ttnn_concat_35 = ttnn.concat(
        util_create_list_128,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_160, False)
    ttnn.deallocate(ttnn_to_layout_159, False)
    ttnn.deallocate(ttnn_to_layout_158, False)
    util_create_list_129 = [ttnn_concat_35]
    return util_create_list_129


def main_const_eval_94(weights, device):
    ttnn_to_device_161 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_161 = ttnn.to_layout(
        ttnn_to_device_161,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_161, False)
    ttnn_reshape_108 = ttnn.reshape(
        ttnn_to_layout_161,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_161, False)
    ttnn_repeat_108 = ttnn.repeat(ttnn_reshape_108, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_108, False)
    util_create_list_130 = [ttnn_repeat_108]
    return util_create_list_130


def main_const_eval_95(weights, device):
    ttnn_to_device_162 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_162 = ttnn.to_layout(
        ttnn_to_device_162,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_162, False)
    ttnn_reshape_109 = ttnn.reshape(
        ttnn_to_layout_162,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_162, False)
    ttnn_repeat_109 = ttnn.repeat(ttnn_reshape_109, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_109, False)
    util_create_list_131 = [ttnn_repeat_109]
    return util_create_list_131


def main_const_eval_96(weights, device):
    ttnn_to_device_163 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_163 = ttnn.to_layout(
        ttnn_to_device_163,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_163, False)
    ttnn_reshape_110 = ttnn.reshape(
        ttnn_to_layout_163,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_163, False)
    ttnn_repeat_110 = ttnn.repeat(ttnn_reshape_110, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_110, False)
    util_create_list_132 = [ttnn_repeat_110]
    return util_create_list_132


def main_const_eval_97(weights, device):
    ttnn_to_device_164 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_164 = ttnn.to_layout(
        ttnn_to_device_164,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_164, False)
    ttnn_reshape_111 = ttnn.reshape(
        ttnn_to_layout_164,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_164, False)
    ttnn_repeat_111 = ttnn.repeat(ttnn_reshape_111, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_111, False)
    util_create_list_133 = [ttnn_repeat_111]
    return util_create_list_133


def main_const_eval_98(weights, device):
    ttnn_to_device_165 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_165 = ttnn.to_layout(
        ttnn_to_device_165,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_165, False)
    ttnn_to_device_166 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_166 = ttnn.to_layout(
        ttnn_to_device_166,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_166, False)
    ttnn_to_device_167 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_167 = ttnn.to_layout(
        ttnn_to_device_167,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_167, False)
    util_create_list_134 = [ttnn_to_layout_165, ttnn_to_layout_166, ttnn_to_layout_167]
    ttnn_concat_36 = ttnn.concat(
        util_create_list_134,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_167, False)
    ttnn.deallocate(ttnn_to_layout_166, False)
    ttnn.deallocate(ttnn_to_layout_165, False)
    util_create_list_135 = [ttnn_concat_36]
    return util_create_list_135


def main_const_eval_99(weights, device):
    ttnn_to_device_168 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_168 = ttnn.to_layout(
        ttnn_to_device_168,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_168, False)
    ttnn_to_device_169 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_169 = ttnn.to_layout(
        ttnn_to_device_169,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_169, False)
    ttnn_to_device_170 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_170 = ttnn.to_layout(
        ttnn_to_device_170,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_170, False)
    util_create_list_136 = [ttnn_to_layout_168, ttnn_to_layout_169, ttnn_to_layout_170]
    ttnn_concat_37 = ttnn.concat(
        util_create_list_136,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_170, False)
    ttnn.deallocate(ttnn_to_layout_169, False)
    ttnn.deallocate(ttnn_to_layout_168, False)
    util_create_list_137 = [ttnn_concat_37]
    return util_create_list_137


def main_const_eval_100(weights, device):
    ttnn_to_device_171 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_171 = ttnn.to_layout(
        ttnn_to_device_171,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_171, False)
    ttnn_to_device_172 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_172 = ttnn.to_layout(
        ttnn_to_device_172,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_172, False)
    ttnn_to_device_173 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_173 = ttnn.to_layout(
        ttnn_to_device_173,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_173, False)
    util_create_list_138 = [ttnn_to_layout_171, ttnn_to_layout_172, ttnn_to_layout_173]
    ttnn_concat_38 = ttnn.concat(
        util_create_list_138,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_173, False)
    ttnn.deallocate(ttnn_to_layout_172, False)
    ttnn.deallocate(ttnn_to_layout_171, False)
    util_create_list_139 = [ttnn_concat_38]
    return util_create_list_139


def main_const_eval_101(device):
    ttnn_full_5 = ttnn.full(
        shape=ttnn.Shape([1, 16, 257, 257]),
        fill_value=float("-inf"),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_140 = [ttnn_full_5]
    return util_create_list_140


def main_const_eval_102(weights, device):
    ttnn_to_device_174 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_174 = ttnn.to_layout(
        ttnn_to_device_174,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_174, False)
    ttnn_to_device_175 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_175 = ttnn.to_layout(
        ttnn_to_device_175,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_175, False)
    ttnn_to_device_176 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_176 = ttnn.to_layout(
        ttnn_to_device_176,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_176, False)
    ttnn_reshape_112 = ttnn.reshape(
        ttnn_to_layout_174,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_174, False)
    ttnn_repeat_112 = ttnn.repeat(ttnn_reshape_112, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_112, False)
    ttnn_reshape_113 = ttnn.reshape(
        ttnn_to_layout_175,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_175, False)
    ttnn_repeat_113 = ttnn.repeat(ttnn_reshape_113, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_113, False)
    ttnn_reshape_114 = ttnn.reshape(
        ttnn_to_layout_176,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_176, False)
    ttnn_repeat_114 = ttnn.repeat(ttnn_reshape_114, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_114, False)
    util_create_list_141 = [ttnn_repeat_112, ttnn_repeat_113, ttnn_repeat_114]
    ttnn_concat_39 = ttnn.concat(
        util_create_list_141,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_114, False)
    ttnn.deallocate(ttnn_repeat_113, False)
    ttnn.deallocate(ttnn_repeat_112, False)
    util_create_list_142 = [ttnn_concat_39]
    return util_create_list_142


def main_const_eval_103(weights, device):
    ttnn_to_device_177 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_177 = ttnn.to_layout(
        ttnn_to_device_177,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_177, False)
    ttnn_reshape_115 = ttnn.reshape(
        ttnn_to_layout_177,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_177, False)
    ttnn_repeat_115 = ttnn.repeat(ttnn_reshape_115, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_115, False)
    util_create_list_143 = [ttnn_repeat_115]
    return util_create_list_143


def main_const_eval_104(weights, device):
    ttnn_to_device_178 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_178 = ttnn.to_layout(
        ttnn_to_device_178,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_178, False)
    ttnn_reshape_116 = ttnn.reshape(
        ttnn_to_layout_178,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_178, False)
    ttnn_repeat_116 = ttnn.repeat(ttnn_reshape_116, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_116, False)
    util_create_list_144 = [ttnn_repeat_116]
    return util_create_list_144


def main_const_eval_105(weights, device):
    ttnn_to_device_179 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_179 = ttnn.to_layout(
        ttnn_to_device_179,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_179, False)
    ttnn_reshape_117 = ttnn.reshape(
        ttnn_to_layout_179,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_179, False)
    ttnn_repeat_117 = ttnn.repeat(ttnn_reshape_117, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_117, False)
    util_create_list_145 = [ttnn_repeat_117]
    return util_create_list_145


def main_const_eval_106(weights, device):
    ttnn_to_device_180 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_180 = ttnn.to_layout(
        ttnn_to_device_180,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_180, False)
    ttnn_to_device_181 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_181 = ttnn.to_layout(
        ttnn_to_device_181,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_181, False)
    ttnn_to_device_182 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_182 = ttnn.to_layout(
        ttnn_to_device_182,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_182, False)
    util_create_list_146 = [ttnn_to_layout_180, ttnn_to_layout_181, ttnn_to_layout_182]
    ttnn_concat_40 = ttnn.concat(
        util_create_list_146,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_182, False)
    ttnn.deallocate(ttnn_to_layout_181, False)
    ttnn.deallocate(ttnn_to_layout_180, False)
    util_create_list_147 = [ttnn_concat_40]
    return util_create_list_147


def main_const_eval_107(weights, device):
    ttnn_to_device_183 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_183 = ttnn.to_layout(
        ttnn_to_device_183,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_183, False)
    ttnn_to_device_184 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_184 = ttnn.to_layout(
        ttnn_to_device_184,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_184, False)
    ttnn_to_device_185 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_185 = ttnn.to_layout(
        ttnn_to_device_185,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_185, False)
    util_create_list_148 = [ttnn_to_layout_183, ttnn_to_layout_184, ttnn_to_layout_185]
    ttnn_concat_41 = ttnn.concat(
        util_create_list_148,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_185, False)
    ttnn.deallocate(ttnn_to_layout_184, False)
    ttnn.deallocate(ttnn_to_layout_183, False)
    util_create_list_149 = [ttnn_concat_41]
    return util_create_list_149


def main_const_eval_108(weights, device):
    ttnn_to_device_186 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_186 = ttnn.to_layout(
        ttnn_to_device_186,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_186, False)
    ttnn_to_device_187 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_187 = ttnn.to_layout(
        ttnn_to_device_187,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_187, False)
    ttnn_to_device_188 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_188 = ttnn.to_layout(
        ttnn_to_device_188,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_188, False)
    util_create_list_150 = [ttnn_to_layout_186, ttnn_to_layout_187, ttnn_to_layout_188]
    ttnn_concat_42 = ttnn.concat(
        util_create_list_150,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_188, False)
    ttnn.deallocate(ttnn_to_layout_187, False)
    ttnn.deallocate(ttnn_to_layout_186, False)
    util_create_list_151 = [ttnn_concat_42]
    return util_create_list_151


def main_const_eval_109(weights, device):
    ttnn_to_device_189 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_189 = ttnn.to_layout(
        ttnn_to_device_189,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_189, False)
    ttnn_to_device_190 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_190 = ttnn.to_layout(
        ttnn_to_device_190,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_190, False)
    ttnn_to_device_191 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_191 = ttnn.to_layout(
        ttnn_to_device_191,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_191, False)
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_to_layout_189,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_189, False)
    ttnn_repeat_118 = ttnn.repeat(ttnn_reshape_118, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_118, False)
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_to_layout_190,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_190, False)
    ttnn_repeat_119 = ttnn.repeat(ttnn_reshape_119, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_119, False)
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_to_layout_191,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_191, False)
    ttnn_repeat_120 = ttnn.repeat(ttnn_reshape_120, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_120, False)
    util_create_list_152 = [ttnn_repeat_118, ttnn_repeat_119, ttnn_repeat_120]
    ttnn_concat_43 = ttnn.concat(
        util_create_list_152,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_120, False)
    ttnn.deallocate(ttnn_repeat_119, False)
    ttnn.deallocate(ttnn_repeat_118, False)
    util_create_list_153 = [ttnn_concat_43]
    return util_create_list_153


def main_const_eval_110(weights, device):
    ttnn_to_device_192 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_192 = ttnn.to_layout(
        ttnn_to_device_192,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_192, False)
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_to_layout_192,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_192, False)
    ttnn_repeat_121 = ttnn.repeat(ttnn_reshape_121, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_121, False)
    util_create_list_154 = [ttnn_repeat_121]
    return util_create_list_154


def main_const_eval_111(weights, device):
    ttnn_to_device_193 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_193 = ttnn.to_layout(
        ttnn_to_device_193,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_193, False)
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_to_layout_193,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_193, False)
    ttnn_repeat_122 = ttnn.repeat(ttnn_reshape_122, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_122, False)
    util_create_list_155 = [ttnn_repeat_122]
    return util_create_list_155


def main_const_eval_112(weights, device):
    ttnn_to_device_194 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_194 = ttnn.to_layout(
        ttnn_to_device_194,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_194, False)
    ttnn_to_device_195 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_195 = ttnn.to_layout(
        ttnn_to_device_195,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_195, False)
    ttnn_to_device_196 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_196 = ttnn.to_layout(
        ttnn_to_device_196,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_196, False)
    util_create_list_156 = [ttnn_to_layout_194, ttnn_to_layout_195, ttnn_to_layout_196]
    ttnn_concat_44 = ttnn.concat(
        util_create_list_156,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_196, False)
    ttnn.deallocate(ttnn_to_layout_195, False)
    ttnn.deallocate(ttnn_to_layout_194, False)
    util_create_list_157 = [ttnn_concat_44]
    return util_create_list_157


def main_const_eval_113(weights, device):
    ttnn_to_device_197 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_197 = ttnn.to_layout(
        ttnn_to_device_197,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_197, False)
    ttnn_to_device_198 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_198 = ttnn.to_layout(
        ttnn_to_device_198,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_198, False)
    ttnn_to_device_199 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_199 = ttnn.to_layout(
        ttnn_to_device_199,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_199, False)
    ttnn_reshape_123 = ttnn.reshape(
        ttnn_to_layout_197,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_197, False)
    ttnn_repeat_123 = ttnn.repeat(ttnn_reshape_123, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_123, False)
    ttnn_reshape_124 = ttnn.reshape(
        ttnn_to_layout_198,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_198, False)
    ttnn_repeat_124 = ttnn.repeat(ttnn_reshape_124, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_124, False)
    ttnn_reshape_125 = ttnn.reshape(
        ttnn_to_layout_199,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_199, False)
    ttnn_repeat_125 = ttnn.repeat(ttnn_reshape_125, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_125, False)
    util_create_list_158 = [ttnn_repeat_123, ttnn_repeat_124, ttnn_repeat_125]
    ttnn_concat_45 = ttnn.concat(
        util_create_list_158,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_125, False)
    ttnn.deallocate(ttnn_repeat_124, False)
    ttnn.deallocate(ttnn_repeat_123, False)
    util_create_list_159 = [ttnn_concat_45]
    return util_create_list_159


def main_const_eval_114(weights, device):
    ttnn_to_device_200 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_200 = ttnn.to_layout(
        ttnn_to_device_200,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_200, False)
    ttnn_reshape_126 = ttnn.reshape(
        ttnn_to_layout_200,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_200, False)
    ttnn_repeat_126 = ttnn.repeat(ttnn_reshape_126, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_126, False)
    util_create_list_160 = [ttnn_repeat_126]
    return util_create_list_160


def main_const_eval_115(weights, device):
    ttnn_to_device_201 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_201 = ttnn.to_layout(
        ttnn_to_device_201,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_201, False)
    ttnn_reshape_127 = ttnn.reshape(
        ttnn_to_layout_201,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_201, False)
    ttnn_repeat_127 = ttnn.repeat(ttnn_reshape_127, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_127, False)
    util_create_list_161 = [ttnn_repeat_127]
    return util_create_list_161


def main_const_eval_116(weights, device):
    ttnn_to_device_202 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_202 = ttnn.to_layout(
        ttnn_to_device_202,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_202, False)
    ttnn_to_device_203 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_203 = ttnn.to_layout(
        ttnn_to_device_203,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_203, False)
    ttnn_to_device_204 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_204 = ttnn.to_layout(
        ttnn_to_device_204,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_204, False)
    util_create_list_162 = [ttnn_to_layout_202, ttnn_to_layout_203, ttnn_to_layout_204]
    ttnn_concat_46 = ttnn.concat(
        util_create_list_162,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_204, False)
    ttnn.deallocate(ttnn_to_layout_203, False)
    ttnn.deallocate(ttnn_to_layout_202, False)
    util_create_list_163 = [ttnn_concat_46]
    return util_create_list_163


def main_const_eval_117(weights, device):
    ttnn_to_device_205 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_205 = ttnn.to_layout(
        ttnn_to_device_205,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_205, False)
    ttnn_to_device_206 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_206 = ttnn.to_layout(
        ttnn_to_device_206,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_206, False)
    ttnn_to_device_207 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_207 = ttnn.to_layout(
        ttnn_to_device_207,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_207, False)
    ttnn_reshape_128 = ttnn.reshape(
        ttnn_to_layout_205,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_205, False)
    ttnn_repeat_128 = ttnn.repeat(ttnn_reshape_128, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_128, False)
    ttnn_reshape_129 = ttnn.reshape(
        ttnn_to_layout_206,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_206, False)
    ttnn_repeat_129 = ttnn.repeat(ttnn_reshape_129, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_129, False)
    ttnn_reshape_130 = ttnn.reshape(
        ttnn_to_layout_207,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_207, False)
    ttnn_repeat_130 = ttnn.repeat(ttnn_reshape_130, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_130, False)
    util_create_list_164 = [ttnn_repeat_128, ttnn_repeat_129, ttnn_repeat_130]
    ttnn_concat_47 = ttnn.concat(
        util_create_list_164,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_130, False)
    ttnn.deallocate(ttnn_repeat_129, False)
    ttnn.deallocate(ttnn_repeat_128, False)
    util_create_list_165 = [ttnn_concat_47]
    return util_create_list_165


def main_const_eval_118(weights, device):
    ttnn_to_device_208 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_208 = ttnn.to_layout(
        ttnn_to_device_208,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_208, False)
    ttnn_reshape_131 = ttnn.reshape(
        ttnn_to_layout_208,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_208, False)
    ttnn_repeat_131 = ttnn.repeat(ttnn_reshape_131, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_131, False)
    util_create_list_166 = [ttnn_repeat_131]
    return util_create_list_166


def main_const_eval_119(weights, device):
    ttnn_to_device_209 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_209 = ttnn.to_layout(
        ttnn_to_device_209,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_209, False)
    ttnn_reshape_132 = ttnn.reshape(
        ttnn_to_layout_209,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_209, False)
    ttnn_repeat_132 = ttnn.repeat(ttnn_reshape_132, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_132, False)
    util_create_list_167 = [ttnn_repeat_132]
    return util_create_list_167


def main_const_eval_120(weights, device):
    ttnn_to_device_210 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_210 = ttnn.to_layout(
        ttnn_to_device_210,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_210, False)
    ttnn_reshape_133 = ttnn.reshape(
        ttnn_to_layout_210,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_210, False)
    ttnn_repeat_133 = ttnn.repeat(ttnn_reshape_133, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_133, False)
    util_create_list_168 = [ttnn_repeat_133]
    return util_create_list_168


def main_const_eval_121(weights, device):
    ttnn_to_device_211 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_211 = ttnn.to_layout(
        ttnn_to_device_211,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_211, False)
    ttnn_reshape_134 = ttnn.reshape(
        ttnn_to_layout_211,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_211, False)
    ttnn_repeat_134 = ttnn.repeat(ttnn_reshape_134, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_134, False)
    util_create_list_169 = [ttnn_repeat_134]
    return util_create_list_169


def main_const_eval_122(weights, device):
    ttnn_to_device_212 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_212 = ttnn.to_layout(
        ttnn_to_device_212,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_212, False)
    ttnn_reshape_135 = ttnn.reshape(
        ttnn_to_layout_212,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_212, False)
    ttnn_permute_1 = ttnn.permute(
        ttnn_reshape_135,
        [0, 2, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_135, False)
    util_create_list_170 = [ttnn_permute_1]
    return util_create_list_170


def main_const_eval_123(weights, device):
    ttnn_to_device_213 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_213 = ttnn.to_layout(
        ttnn_to_device_213,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_213, False)
    ttnn_reshape_136 = ttnn.reshape(
        ttnn_to_layout_213,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_213, False)
    ttnn_repeat_135 = ttnn.repeat(ttnn_reshape_136, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_136, False)
    util_create_list_171 = [ttnn_repeat_135]
    return util_create_list_171


def main_const_eval_124(device):
    ttnn_full_6 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 273]),
        fill_value=0.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_172 = [ttnn_full_6]
    return util_create_list_172


def main_const_eval_125(weights, device):
    ttnn_to_device_214 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_214 = ttnn.to_layout(
        ttnn_to_device_214,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_214, False)
    ttnn_reshape_137 = ttnn.reshape(
        ttnn_to_layout_214,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_214, False)
    ttnn_repeat_136 = ttnn.repeat(ttnn_reshape_137, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_137, False)
    util_create_list_173 = [ttnn_repeat_136]
    return util_create_list_173


def main_const_eval_126(weights, device):
    ttnn_to_device_215 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_215 = ttnn.to_layout(
        ttnn_to_device_215,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_215, False)
    ttnn_to_device_216 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_216 = ttnn.to_layout(
        ttnn_to_device_216,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_216, False)
    ttnn_to_device_217 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_217 = ttnn.to_layout(
        ttnn_to_device_217,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_217, False)
    util_create_list_174 = [ttnn_to_layout_215, ttnn_to_layout_216, ttnn_to_layout_217]
    ttnn_concat_48 = ttnn.concat(
        util_create_list_174,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_217, False)
    ttnn.deallocate(ttnn_to_layout_216, False)
    ttnn.deallocate(ttnn_to_layout_215, False)
    util_create_list_175 = [ttnn_concat_48]
    return util_create_list_175


def main_const_eval_127(weights, device):
    ttnn_to_device_218 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_218 = ttnn.to_layout(
        ttnn_to_device_218,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_218, False)
    ttnn_to_device_219 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_219 = ttnn.to_layout(
        ttnn_to_device_219,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_219, False)
    ttnn_to_device_220 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_220 = ttnn.to_layout(
        ttnn_to_device_220,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_220, False)
    util_create_list_176 = [ttnn_to_layout_218, ttnn_to_layout_219, ttnn_to_layout_220]
    ttnn_concat_49 = ttnn.concat(
        util_create_list_176,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_220, False)
    ttnn.deallocate(ttnn_to_layout_219, False)
    ttnn.deallocate(ttnn_to_layout_218, False)
    util_create_list_177 = [ttnn_concat_49]
    return util_create_list_177


def main_const_eval_128(weights, device):
    ttnn_to_device_221 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_221 = ttnn.to_layout(
        ttnn_to_device_221,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_221, False)
    ttnn_reshape_138 = ttnn.reshape(
        ttnn_to_layout_221,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_221, False)
    ttnn_repeat_137 = ttnn.repeat(ttnn_reshape_138, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_138, False)
    util_create_list_178 = [ttnn_repeat_137]
    return util_create_list_178


def main_const_eval_129(weights, device):
    ttnn_to_device_222 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_222 = ttnn.to_layout(
        ttnn_to_device_222,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_222, False)
    ttnn_to_device_223 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_223 = ttnn.to_layout(
        ttnn_to_device_223,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_223, False)
    ttnn_to_device_224 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_224 = ttnn.to_layout(
        ttnn_to_device_224,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_224, False)
    ttnn_reshape_139 = ttnn.reshape(
        ttnn_to_layout_222,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_222, False)
    ttnn_repeat_138 = ttnn.repeat(ttnn_reshape_139, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_139, False)
    ttnn_reshape_140 = ttnn.reshape(
        ttnn_to_layout_223,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_223, False)
    ttnn_repeat_139 = ttnn.repeat(ttnn_reshape_140, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_140, False)
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_to_layout_224,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_224, False)
    ttnn_repeat_140 = ttnn.repeat(ttnn_reshape_141, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_141, False)
    util_create_list_179 = [ttnn_repeat_138, ttnn_repeat_139, ttnn_repeat_140]
    ttnn_concat_50 = ttnn.concat(
        util_create_list_179,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_140, False)
    ttnn.deallocate(ttnn_repeat_139, False)
    ttnn.deallocate(ttnn_repeat_138, False)
    util_create_list_180 = [ttnn_concat_50]
    return util_create_list_180


def main_const_eval_130(weights, device):
    ttnn_to_device_225 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_225 = ttnn.to_layout(
        ttnn_to_device_225,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_225, False)
    ttnn_to_device_226 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_226 = ttnn.to_layout(
        ttnn_to_device_226,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_226, False)
    ttnn_to_device_227 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_227 = ttnn.to_layout(
        ttnn_to_device_227,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_227, False)
    util_create_list_181 = [ttnn_to_layout_225, ttnn_to_layout_226, ttnn_to_layout_227]
    ttnn_concat_51 = ttnn.concat(
        util_create_list_181,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_227, False)
    ttnn.deallocate(ttnn_to_layout_226, False)
    ttnn.deallocate(ttnn_to_layout_225, False)
    util_create_list_182 = [ttnn_concat_51]
    return util_create_list_182


def main_const_eval_131(weights, device):
    ttnn_to_device_228 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_228 = ttnn.to_layout(
        ttnn_to_device_228,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_228, False)
    ttnn_to_device_229 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_229 = ttnn.to_layout(
        ttnn_to_device_229,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_229, False)
    ttnn_to_device_230 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_230 = ttnn.to_layout(
        ttnn_to_device_230,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_230, False)
    ttnn_reshape_142 = ttnn.reshape(
        ttnn_to_layout_228,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_228, False)
    ttnn_repeat_141 = ttnn.repeat(ttnn_reshape_142, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_142, False)
    ttnn_reshape_143 = ttnn.reshape(
        ttnn_to_layout_229,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_229, False)
    ttnn_repeat_142 = ttnn.repeat(ttnn_reshape_143, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_143, False)
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_to_layout_230,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_230, False)
    ttnn_repeat_143 = ttnn.repeat(ttnn_reshape_144, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_144, False)
    util_create_list_183 = [ttnn_repeat_141, ttnn_repeat_142, ttnn_repeat_143]
    ttnn_concat_52 = ttnn.concat(
        util_create_list_183,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_143, False)
    ttnn.deallocate(ttnn_repeat_142, False)
    ttnn.deallocate(ttnn_repeat_141, False)
    util_create_list_184 = [ttnn_concat_52]
    return util_create_list_184


def main_const_eval_132(weights, device):
    ttnn_to_device_231 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_231 = ttnn.to_layout(
        ttnn_to_device_231,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_231, False)
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_to_layout_231,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_231, False)
    ttnn_repeat_144 = ttnn.repeat(ttnn_reshape_145, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_145, False)
    util_create_list_185 = [ttnn_repeat_144]
    return util_create_list_185


def main_const_eval_133(weights, device):
    ttnn_to_device_232 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_232 = ttnn.to_layout(
        ttnn_to_device_232,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_232, False)
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_to_layout_232,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_232, False)
    ttnn_repeat_145 = ttnn.repeat(ttnn_reshape_146, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_146, False)
    util_create_list_186 = [ttnn_repeat_145]
    return util_create_list_186


def main_const_eval_134(weights, device):
    ttnn_to_device_233 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_233 = ttnn.to_layout(
        ttnn_to_device_233,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_233, False)
    ttnn_to_device_234 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_234 = ttnn.to_layout(
        ttnn_to_device_234,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_234, False)
    ttnn_to_device_235 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_235 = ttnn.to_layout(
        ttnn_to_device_235,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_235, False)
    ttnn_reshape_147 = ttnn.reshape(
        ttnn_to_layout_233,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_233, False)
    ttnn_repeat_146 = ttnn.repeat(ttnn_reshape_147, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_147, False)
    ttnn_reshape_148 = ttnn.reshape(
        ttnn_to_layout_234,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_234, False)
    ttnn_repeat_147 = ttnn.repeat(ttnn_reshape_148, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_148, False)
    ttnn_reshape_149 = ttnn.reshape(
        ttnn_to_layout_235,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_235, False)
    ttnn_repeat_148 = ttnn.repeat(ttnn_reshape_149, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_149, False)
    util_create_list_187 = [ttnn_repeat_146, ttnn_repeat_147, ttnn_repeat_148]
    ttnn_concat_53 = ttnn.concat(
        util_create_list_187,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_148, False)
    ttnn.deallocate(ttnn_repeat_147, False)
    ttnn.deallocate(ttnn_repeat_146, False)
    util_create_list_188 = [ttnn_concat_53]
    return util_create_list_188


def main_const_eval_135(weights, device):
    ttnn_to_device_236 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_236 = ttnn.to_layout(
        ttnn_to_device_236,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_236, False)
    ttnn_to_device_237 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_237 = ttnn.to_layout(
        ttnn_to_device_237,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_237, False)
    ttnn_to_device_238 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_238 = ttnn.to_layout(
        ttnn_to_device_238,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_238, False)
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_to_layout_236,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_236, False)
    ttnn_repeat_149 = ttnn.repeat(ttnn_reshape_150, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_150, False)
    ttnn_reshape_151 = ttnn.reshape(
        ttnn_to_layout_237,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_237, False)
    ttnn_repeat_150 = ttnn.repeat(ttnn_reshape_151, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_151, False)
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_to_layout_238,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_238, False)
    ttnn_repeat_151 = ttnn.repeat(ttnn_reshape_152, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_152, False)
    util_create_list_189 = [ttnn_repeat_149, ttnn_repeat_150, ttnn_repeat_151]
    ttnn_concat_54 = ttnn.concat(
        util_create_list_189,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_151, False)
    ttnn.deallocate(ttnn_repeat_150, False)
    ttnn.deallocate(ttnn_repeat_149, False)
    util_create_list_190 = [ttnn_concat_54]
    return util_create_list_190


def main_const_eval_136(device):
    ttnn_full_7 = ttnn.full(
        shape=ttnn.Shape([1, 16, 257]),
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_191 = [ttnn_full_7]
    return util_create_list_191


def main_const_eval_137(weights, device):
    ttnn_to_device_239 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_239 = ttnn.to_layout(
        ttnn_to_device_239,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_239, False)
    ttnn_reshape_153 = ttnn.reshape(
        ttnn_to_layout_239,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_239, False)
    ttnn_repeat_152 = ttnn.repeat(ttnn_reshape_153, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_153, False)
    util_create_list_192 = [ttnn_repeat_152]
    return util_create_list_192


def main_const_eval_138(weights, device):
    ttnn_to_device_240 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_240 = ttnn.to_layout(
        ttnn_to_device_240,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_240, False)
    ttnn_reshape_154 = ttnn.reshape(
        ttnn_to_layout_240,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_240, False)
    ttnn_repeat_153 = ttnn.repeat(ttnn_reshape_154, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_154, False)
    util_create_list_193 = [ttnn_repeat_153]
    return util_create_list_193


def main_const_eval_139(weights, device):
    ttnn_to_device_241 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_241 = ttnn.to_layout(
        ttnn_to_device_241,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_241, False)
    ttnn_reshape_155 = ttnn.reshape(
        ttnn_to_layout_241,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_241, False)
    ttnn_repeat_154 = ttnn.repeat(ttnn_reshape_155, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_155, False)
    util_create_list_194 = [ttnn_repeat_154]
    return util_create_list_194


def main_const_eval_140(weights, device):
    ttnn_to_device_242 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_242 = ttnn.to_layout(
        ttnn_to_device_242,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_242, False)
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_to_layout_242,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_242, False)
    ttnn_repeat_155 = ttnn.repeat(ttnn_reshape_156, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_156, False)
    util_create_list_195 = [ttnn_repeat_155]
    return util_create_list_195


def main_const_eval_141(weights, device):
    ttnn_to_device_243 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_243 = ttnn.to_layout(
        ttnn_to_device_243,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_243, False)
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_to_layout_243,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_243, False)
    ttnn_repeat_156 = ttnn.repeat(ttnn_reshape_157, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_157, False)
    util_create_list_196 = [ttnn_repeat_156]
    return util_create_list_196


def main_const_eval_142(weights, device):
    ttnn_to_device_244 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_244 = ttnn.to_layout(
        ttnn_to_device_244,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_244, False)
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_to_layout_244,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_244, False)
    ttnn_repeat_157 = ttnn.repeat(ttnn_reshape_158, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_158, False)
    util_create_list_197 = [ttnn_repeat_157]
    return util_create_list_197


def main_const_eval_143(weights, device):
    ttnn_to_device_245 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_245 = ttnn.to_layout(
        ttnn_to_device_245,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_245, False)
    ttnn_reshape_159 = ttnn.reshape(
        ttnn_to_layout_245,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_245, False)
    ttnn_repeat_158 = ttnn.repeat(ttnn_reshape_159, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_159, False)
    util_create_list_198 = [ttnn_repeat_158]
    return util_create_list_198


def main_const_eval_144(weights, device):
    ttnn_to_device_246 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_246 = ttnn.to_layout(
        ttnn_to_device_246,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_246, False)
    ttnn_reshape_160 = ttnn.reshape(
        ttnn_to_layout_246,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_246, False)
    ttnn_repeat_159 = ttnn.repeat(ttnn_reshape_160, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_160, False)
    util_create_list_199 = [ttnn_repeat_159]
    return util_create_list_199


def main_const_eval_145(weights, device):
    ttnn_to_device_247 = ttnn.to_device(
        weights[3],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_247 = ttnn.to_layout(
        ttnn_to_device_247,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_247, False)
    ttnn_to_device_248 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_248 = ttnn.to_layout(
        ttnn_to_device_248,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_248, False)
    ttnn_to_device_249 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_249 = ttnn.to_layout(
        ttnn_to_device_249,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_249, False)
    ttnn_full_8 = ttnn.full(
        shape=ttnn.Shape([1, 20, 16, 64]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_layer_norm_0 = ttnn.layer_norm(
        weights[0],
        epsilon=9.9999997473787516e-06,
        weight=ttnn_to_layout_248,
        bias=ttnn_to_layout_249,
        residual_input_tensor=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        program_config=None,
    )
    ttnn.deallocate(ttnn_to_layout_249, False)
    ttnn.deallocate(ttnn_to_layout_248, False)
    ttnn_reshape_161 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_reshape_161,
        ttnn_to_layout_247,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_reshape_161, False)
    ttnn.deallocate(ttnn_to_layout_247, False)
    ttnn_reshape_162 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 16, 20, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_permute_2 = ttnn.permute(
        ttnn_reshape_162,
        [0, 2, 1, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_162, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_permute_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_typecast_1,
        ttnn_full_8,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn_reshape_163 = ttnn.reshape(
        ttnn_layer_norm_0,
        [16, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_layer_norm_0, False)
    util_create_list_200 = [ttnn_full_8, ttnn_multiply_0, ttnn_reshape_163]
    return util_create_list_200


def main_const_eval_146(weights, device):
    ttnn_to_device_250 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_250 = ttnn.to_layout(
        ttnn_to_device_250,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_250, False)
    ttnn_to_device_251 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_251 = ttnn.to_layout(
        ttnn_to_device_251,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_251, False)
    ttnn_to_device_252 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_252 = ttnn.to_layout(
        ttnn_to_device_252,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_252, False)
    ttnn_reshape_164 = ttnn.reshape(
        ttnn_to_layout_250,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_250, False)
    ttnn_repeat_160 = ttnn.repeat(ttnn_reshape_164, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_164, False)
    ttnn_reshape_165 = ttnn.reshape(
        ttnn_to_layout_251,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_251, False)
    ttnn_repeat_161 = ttnn.repeat(ttnn_reshape_165, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_165, False)
    ttnn_reshape_166 = ttnn.reshape(
        ttnn_to_layout_252,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_252, False)
    ttnn_repeat_162 = ttnn.repeat(ttnn_reshape_166, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_166, False)
    util_create_list_201 = [ttnn_repeat_160, ttnn_repeat_161, ttnn_repeat_162]
    ttnn_concat_55 = ttnn.concat(
        util_create_list_201,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_162, False)
    ttnn.deallocate(ttnn_repeat_161, False)
    ttnn.deallocate(ttnn_repeat_160, False)
    util_create_list_202 = [ttnn_concat_55]
    return util_create_list_202


def main_const_eval_147(weights, device):
    ttnn_to_device_253 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_253 = ttnn.to_layout(
        ttnn_to_device_253,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_253, False)
    ttnn_reshape_167 = ttnn.reshape(
        ttnn_to_layout_253,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_253, False)
    ttnn_repeat_163 = ttnn.repeat(ttnn_reshape_167, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_167, False)
    util_create_list_203 = [ttnn_repeat_163]
    return util_create_list_203


def main_const_eval_148(weights, device):
    ttnn_to_device_254 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_254 = ttnn.to_layout(
        ttnn_to_device_254,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_254, False)
    ttnn_reshape_168 = ttnn.reshape(
        ttnn_to_layout_254,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_254, False)
    ttnn_repeat_164 = ttnn.repeat(ttnn_reshape_168, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_168, False)
    util_create_list_204 = [ttnn_repeat_164]
    return util_create_list_204


def main_const_eval_149(weights, device):
    ttnn_to_device_255 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_255 = ttnn.to_layout(
        ttnn_to_device_255,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_255, False)
    ttnn_reshape_169 = ttnn.reshape(
        ttnn_to_layout_255,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_255, False)
    ttnn_repeat_165 = ttnn.repeat(ttnn_reshape_169, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_169, False)
    util_create_list_205 = [ttnn_repeat_165]
    return util_create_list_205


def main_const_eval_150(weights, device):
    ttnn_to_device_256 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_256 = ttnn.to_layout(
        ttnn_to_device_256,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_256, False)
    ttnn_to_device_257 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_257 = ttnn.to_layout(
        ttnn_to_device_257,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_257, False)
    ttnn_to_device_258 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_258 = ttnn.to_layout(
        ttnn_to_device_258,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_258, False)
    util_create_list_206 = [ttnn_to_layout_256, ttnn_to_layout_257, ttnn_to_layout_258]
    ttnn_concat_56 = ttnn.concat(
        util_create_list_206,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_258, False)
    ttnn.deallocate(ttnn_to_layout_257, False)
    ttnn.deallocate(ttnn_to_layout_256, False)
    util_create_list_207 = [ttnn_concat_56]
    return util_create_list_207


def main_const_eval_151(device):
    ttnn_full_9 = ttnn.full(
        shape=ttnn.Shape([1, 16, 257, 80]),
        fill_value=0.33437016606330872,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_208 = [ttnn_full_9]
    return util_create_list_208


def main_const_eval_152(weights, device):
    ttnn_to_device_259 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_259 = ttnn.to_layout(
        ttnn_to_device_259,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_259, False)
    ttnn_reshape_170 = ttnn.reshape(
        ttnn_to_layout_259,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_259, False)
    ttnn_repeat_166 = ttnn.repeat(ttnn_reshape_170, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_170, False)
    util_create_list_209 = [ttnn_repeat_166]
    return util_create_list_209


def main_const_eval_153(weights, device):
    ttnn_to_device_260 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_260 = ttnn.to_layout(
        ttnn_to_device_260,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_260, False)
    ttnn_to_device_261 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_261 = ttnn.to_layout(
        ttnn_to_device_261,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_261, False)
    ttnn_to_device_262 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_262 = ttnn.to_layout(
        ttnn_to_device_262,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_262, False)
    ttnn_reshape_171 = ttnn.reshape(
        ttnn_to_layout_260,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_260, False)
    ttnn_repeat_167 = ttnn.repeat(ttnn_reshape_171, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_171, False)
    ttnn_reshape_172 = ttnn.reshape(
        ttnn_to_layout_261,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_261, False)
    ttnn_repeat_168 = ttnn.repeat(ttnn_reshape_172, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_172, False)
    ttnn_reshape_173 = ttnn.reshape(
        ttnn_to_layout_262,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_262, False)
    ttnn_repeat_169 = ttnn.repeat(ttnn_reshape_173, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_173, False)
    util_create_list_210 = [ttnn_repeat_167, ttnn_repeat_168, ttnn_repeat_169]
    ttnn_concat_57 = ttnn.concat(
        util_create_list_210,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_169, False)
    ttnn.deallocate(ttnn_repeat_168, False)
    ttnn.deallocate(ttnn_repeat_167, False)
    util_create_list_211 = [ttnn_concat_57]
    return util_create_list_211


def main_const_eval_154(weights, device):
    ttnn_to_device_263 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_263 = ttnn.to_layout(
        ttnn_to_device_263,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_263, False)
    ttnn_reshape_174 = ttnn.reshape(
        ttnn_to_layout_263,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_263, False)
    ttnn_repeat_170 = ttnn.repeat(ttnn_reshape_174, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_174, False)
    util_create_list_212 = [ttnn_repeat_170]
    return util_create_list_212


def main_const_eval_155(weights, device):
    ttnn_to_device_264 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_264 = ttnn.to_layout(
        ttnn_to_device_264,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_264, False)
    ttnn_reshape_175 = ttnn.reshape(
        ttnn_to_layout_264,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_264, False)
    ttnn_repeat_171 = ttnn.repeat(ttnn_reshape_175, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_175, False)
    util_create_list_213 = [ttnn_repeat_171]
    return util_create_list_213


def main_const_eval_156(weights, device):
    ttnn_to_device_265 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_265 = ttnn.to_layout(
        ttnn_to_device_265,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_265, False)
    ttnn_reshape_176 = ttnn.reshape(
        ttnn_to_layout_265,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_265, False)
    ttnn_repeat_172 = ttnn.repeat(ttnn_reshape_176, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_176, False)
    util_create_list_214 = [ttnn_repeat_172]
    return util_create_list_214


def main_const_eval_157(weights, device):
    ttnn_to_device_266 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_266 = ttnn.to_layout(
        ttnn_to_device_266,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_266, False)
    ttnn_reshape_177 = ttnn.reshape(
        ttnn_to_layout_266,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_266, False)
    ttnn_repeat_173 = ttnn.repeat(ttnn_reshape_177, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_177, False)
    util_create_list_215 = [ttnn_repeat_173]
    return util_create_list_215


def main_const_eval_158(weights, device):
    ttnn_to_device_267 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_267 = ttnn.to_layout(
        ttnn_to_device_267,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_267, False)
    ttnn_reshape_178 = ttnn.reshape(
        ttnn_to_layout_267,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_267, False)
    ttnn_repeat_174 = ttnn.repeat(ttnn_reshape_178, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_178, False)
    util_create_list_216 = [ttnn_repeat_174]
    return util_create_list_216


def main_const_eval_159(weights, device):
    ttnn_to_device_268 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_268 = ttnn.to_layout(
        ttnn_to_device_268,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_268, False)
    ttnn_reshape_179 = ttnn.reshape(
        ttnn_to_layout_268,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_268, False)
    ttnn_repeat_175 = ttnn.repeat(ttnn_reshape_179, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_179, False)
    util_create_list_217 = [ttnn_repeat_175]
    return util_create_list_217


def main_const_eval_160(weights, device):
    ttnn_to_device_269 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_269 = ttnn.to_layout(
        ttnn_to_device_269,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_269, False)
    ttnn_reshape_180 = ttnn.reshape(
        ttnn_to_layout_269,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_269, False)
    ttnn_repeat_176 = ttnn.repeat(ttnn_reshape_180, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_180, False)
    util_create_list_218 = [ttnn_repeat_176]
    return util_create_list_218


def main_const_eval_161(weights, device):
    ttnn_to_device_270 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_270 = ttnn.to_layout(
        ttnn_to_device_270,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_270, False)
    ttnn_reshape_181 = ttnn.reshape(
        ttnn_to_layout_270,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_270, False)
    ttnn_repeat_177 = ttnn.repeat(ttnn_reshape_181, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_181, False)
    util_create_list_219 = [ttnn_repeat_177]
    return util_create_list_219


def main_const_eval_162(weights, device):
    ttnn_to_device_271 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_271 = ttnn.to_layout(
        ttnn_to_device_271,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_271, False)
    ttnn_to_device_272 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_272 = ttnn.to_layout(
        ttnn_to_device_272,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_272, False)
    ttnn_to_device_273 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_273 = ttnn.to_layout(
        ttnn_to_device_273,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_273, False)
    ttnn_reshape_182 = ttnn.reshape(
        ttnn_to_layout_271,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_271, False)
    ttnn_repeat_178 = ttnn.repeat(ttnn_reshape_182, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_182, False)
    ttnn_reshape_183 = ttnn.reshape(
        ttnn_to_layout_272,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_272, False)
    ttnn_repeat_179 = ttnn.repeat(ttnn_reshape_183, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_183, False)
    ttnn_reshape_184 = ttnn.reshape(
        ttnn_to_layout_273,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_273, False)
    ttnn_repeat_180 = ttnn.repeat(ttnn_reshape_184, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_184, False)
    util_create_list_220 = [ttnn_repeat_178, ttnn_repeat_179, ttnn_repeat_180]
    ttnn_concat_58 = ttnn.concat(
        util_create_list_220,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_180, False)
    ttnn.deallocate(ttnn_repeat_179, False)
    ttnn.deallocate(ttnn_repeat_178, False)
    util_create_list_221 = [ttnn_concat_58]
    return util_create_list_221


def main_const_eval_163(weights, device):
    ttnn_to_device_274 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_274 = ttnn.to_layout(
        ttnn_to_device_274,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_274, False)
    ttnn_to_device_275 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_275 = ttnn.to_layout(
        ttnn_to_device_275,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_275, False)
    ttnn_to_device_276 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_276 = ttnn.to_layout(
        ttnn_to_device_276,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_276, False)
    ttnn_reshape_185 = ttnn.reshape(
        ttnn_to_layout_274,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_274, False)
    ttnn_repeat_181 = ttnn.repeat(ttnn_reshape_185, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_185, False)
    ttnn_reshape_186 = ttnn.reshape(
        ttnn_to_layout_275,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_275, False)
    ttnn_repeat_182 = ttnn.repeat(ttnn_reshape_186, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_186, False)
    ttnn_reshape_187 = ttnn.reshape(
        ttnn_to_layout_276,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_276, False)
    ttnn_repeat_183 = ttnn.repeat(ttnn_reshape_187, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_187, False)
    util_create_list_222 = [ttnn_repeat_181, ttnn_repeat_182, ttnn_repeat_183]
    ttnn_concat_59 = ttnn.concat(
        util_create_list_222,
        2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_183, False)
    ttnn.deallocate(ttnn_repeat_182, False)
    ttnn.deallocate(ttnn_repeat_181, False)
    util_create_list_223 = [ttnn_concat_59]
    return util_create_list_223


def main_const_eval_164(weights, device):
    ttnn_to_device_277 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_277 = ttnn.to_layout(
        ttnn_to_device_277,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_277, False)
    ttnn_reshape_188 = ttnn.reshape(
        ttnn_to_layout_277,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_277, False)
    ttnn_repeat_184 = ttnn.repeat(ttnn_reshape_188, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_188, False)
    util_create_list_224 = [ttnn_repeat_184]
    return util_create_list_224


def main_const_eval_165(device):
    ttnn_full_10 = ttnn.full(
        shape=ttnn.Shape([1, 20, 64, 273]),
        fill_value=0.35355338454246521,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_225 = [ttnn_full_10]
    return util_create_list_225


def main_const_eval_166(weights, device):
    ttnn_to_device_278 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_278 = ttnn.to_layout(
        ttnn_to_device_278,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_278, False)
    ttnn_to_device_279 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_279 = ttnn.to_layout(
        ttnn_to_device_279,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_279, False)
    ttnn_to_device_280 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_280 = ttnn.to_layout(
        ttnn_to_device_280,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_280, False)
    util_create_list_226 = [ttnn_to_layout_278, ttnn_to_layout_279, ttnn_to_layout_280]
    ttnn_concat_60 = ttnn.concat(
        util_create_list_226,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_280, False)
    ttnn.deallocate(ttnn_to_layout_279, False)
    ttnn.deallocate(ttnn_to_layout_278, False)
    util_create_list_227 = [ttnn_concat_60]
    return util_create_list_227


def main_const_eval_167(weights, device):
    ttnn_to_device_281 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_281 = ttnn.to_layout(
        ttnn_to_device_281,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_281, False)
    ttnn_reshape_189 = ttnn.reshape(
        ttnn_to_layout_281,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_281, False)
    ttnn_repeat_185 = ttnn.repeat(ttnn_reshape_189, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_189, False)
    util_create_list_228 = [ttnn_repeat_185]
    return util_create_list_228


def main_const_eval_168(weights, device):
    ttnn_to_device_282 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_282 = ttnn.to_layout(
        ttnn_to_device_282,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_282, False)
    ttnn_reshape_190 = ttnn.reshape(
        ttnn_to_layout_282,
        [1, 1, 5120],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_282, False)
    ttnn_repeat_186 = ttnn.repeat(ttnn_reshape_190, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_190, False)
    util_create_list_229 = [ttnn_repeat_186]
    return util_create_list_229


def main_const_eval_169(weights, device):
    ttnn_to_device_283 = ttnn.to_device(
        weights[2],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_283 = ttnn.to_layout(
        ttnn_to_device_283,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_283, False)
    ttnn_to_device_284 = ttnn.to_device(
        weights[1],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_284 = ttnn.to_layout(
        ttnn_to_device_284,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_284, False)
    ttnn_to_device_285 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_285 = ttnn.to_layout(
        ttnn_to_device_285,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_285, False)
    util_create_list_230 = [ttnn_to_layout_283, ttnn_to_layout_284, ttnn_to_layout_285]
    ttnn_concat_61 = ttnn.concat(
        util_create_list_230,
        0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_285, False)
    ttnn.deallocate(ttnn_to_layout_284, False)
    ttnn.deallocate(ttnn_to_layout_283, False)
    util_create_list_231 = [ttnn_concat_61]
    return util_create_list_231


def main_const_eval_170(weights, device):
    ttnn_to_device_286 = ttnn.to_device(
        weights[0],
        device=device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_to_layout_286 = ttnn.to_layout(
        ttnn_to_device_286,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_device_286, False)
    ttnn_reshape_191 = ttnn.reshape(
        ttnn_to_layout_286,
        [1, 1, 1280],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_286, False)
    ttnn_repeat_187 = ttnn.repeat(ttnn_reshape_191, ttnn.Shape([1, 257, 1]))
    ttnn.deallocate(ttnn_reshape_191, False)
    util_create_list_232 = [ttnn_repeat_187]
    return util_create_list_232


def run_const_evals(weights, cache, device):
    """Run all const-eval functions and return their results."""
    const_0 = main_const_eval_0
    const_1 = "main_const_eval_0"
    cez_0 = utils.constEvalFuncWrapperZeroArg(const_0, cache, const_1, device)
    cez_0_0 = cez_0[0]
    const_2 = main_const_eval_1
    util_create_list_233 = [weights[289]]
    const_3 = "main_const_eval_1"
    ce_0 = utils.constEvalFuncWrapper(const_2, util_create_list_233, cache, const_3, device)
    ce_0_0 = ce_0[0]
    const_4 = main_const_eval_2
    util_create_list_234 = [weights[231]]
    const_5 = "main_const_eval_2"
    ce_1 = utils.constEvalFuncWrapper(const_4, util_create_list_234, cache, const_5, device)
    ce_1_0 = ce_1[0]
    const_6 = main_const_eval_3
    util_create_list_235 = [weights[61]]
    const_7 = "main_const_eval_3"
    ce_2 = utils.constEvalFuncWrapper(const_6, util_create_list_235, cache, const_7, device)
    ce_2_0 = ce_2[0]
    const_8 = main_const_eval_4
    util_create_list_236 = [weights[22], weights[513], weights[515]]
    const_9 = "main_const_eval_4"
    ce_3 = utils.constEvalFuncWrapper(const_8, util_create_list_236, cache, const_9, device)
    ce_3_0 = ce_3[0]
    const_10 = main_const_eval_5
    util_create_list_237 = [weights[13]]
    const_11 = "main_const_eval_5"
    ce_4 = utils.constEvalFuncWrapper(const_10, util_create_list_237, cache, const_11, device)
    ce_4_0 = ce_4[0]
    const_12 = main_const_eval_6
    util_create_list_238 = [weights[118], weights[481], weights[483]]
    const_13 = "main_const_eval_6"
    ce_5 = utils.constEvalFuncWrapper(const_12, util_create_list_238, cache, const_13, device)
    ce_5_0 = ce_5[0]
    const_14 = main_const_eval_7
    util_create_list_239 = [weights[334], weights[409], weights[411]]
    const_15 = "main_const_eval_7"
    ce_6 = utils.constEvalFuncWrapper(const_14, util_create_list_239, cache, const_15, device)
    ce_6_0 = ce_6[0]
    const_16 = main_const_eval_8
    util_create_list_240 = [weights[274], weights[429], weights[431]]
    const_17 = "main_const_eval_8"
    ce_7 = utils.constEvalFuncWrapper(const_16, util_create_list_240, cache, const_17, device)
    ce_7_0 = ce_7[0]
    const_18 = main_const_eval_9
    util_create_list_241 = [weights[67]]
    const_19 = "main_const_eval_9"
    ce_8 = utils.constEvalFuncWrapper(const_18, util_create_list_241, cache, const_19, device)
    ce_8_0 = ce_8[0]
    const_20 = main_const_eval_10
    util_create_list_242 = [weights[249], weights[436], weights[438]]
    const_21 = "main_const_eval_10"
    ce_9 = utils.constEvalFuncWrapper(const_20, util_create_list_242, cache, const_21, device)
    ce_9_0 = ce_9[0]
    const_22 = main_const_eval_11
    util_create_list_243 = [weights[202], weights[453], weights[455]]
    const_23 = "main_const_eval_11"
    ce_10 = utils.constEvalFuncWrapper(const_22, util_create_list_243, cache, const_23, device)
    ce_10_0 = ce_10[0]
    const_24 = main_const_eval_12
    util_create_list_244 = [weights[343]]
    const_25 = "main_const_eval_12"
    ce_11 = utils.constEvalFuncWrapper(const_24, util_create_list_244, cache, const_25, device)
    ce_11_0 = ce_11[0]
    const_26 = main_const_eval_13
    util_create_list_245 = [weights[213], weights[448], weights[450]]
    const_27 = "main_const_eval_13"
    ce_12 = utils.constEvalFuncWrapper(const_26, util_create_list_245, cache, const_27, device)
    ce_12_0 = ce_12[0]
    const_28 = main_const_eval_14
    util_create_list_246 = [weights[285], weights[424], weights[426]]
    const_29 = "main_const_eval_14"
    ce_13 = utils.constEvalFuncWrapper(const_28, util_create_list_246, cache, const_29, device)
    ce_13_0 = ce_13[0]
    const_30 = main_const_eval_15
    util_create_list_247 = [weights[33], weights[508], weights[510]]
    const_31 = "main_const_eval_15"
    ce_14 = utils.constEvalFuncWrapper(const_30, util_create_list_247, cache, const_31, device)
    ce_14_0 = ce_14[0]
    const_32 = main_const_eval_16
    util_create_list_248 = [weights[46], weights[505], weights[507]]
    const_33 = "main_const_eval_16"
    ce_15 = utils.constEvalFuncWrapper(const_32, util_create_list_248, cache, const_33, device)
    ce_15_0 = ce_15[0]
    const_34 = main_const_eval_17
    util_create_list_249 = [weights[106], weights[485], weights[487]]
    const_35 = "main_const_eval_17"
    ce_16 = utils.constEvalFuncWrapper(const_34, util_create_list_249, cache, const_35, device)
    ce_16_0 = ce_16[0]
    const_36 = main_const_eval_18
    util_create_list_250 = [weights[165], weights[464], weights[466]]
    const_37 = "main_const_eval_18"
    ce_17 = utils.constEvalFuncWrapper(const_36, util_create_list_250, cache, const_37, device)
    ce_17_0 = ce_17[0]
    const_38 = main_const_eval_19
    util_create_list_251 = [weights[298], weights[421], weights[423]]
    const_39 = "main_const_eval_19"
    ce_18 = utils.constEvalFuncWrapper(const_38, util_create_list_251, cache, const_39, device)
    ce_18_0 = ce_18[0]
    const_40 = main_const_eval_20
    util_create_list_252 = [weights[211]]
    const_41 = "main_const_eval_20"
    ce_19 = utils.constEvalFuncWrapper(const_40, util_create_list_252, cache, const_41, device)
    ce_19_0 = ce_19[0]
    const_42 = main_const_eval_21
    util_create_list_253 = [weights[253]]
    const_43 = "main_const_eval_21"
    ce_20 = utils.constEvalFuncWrapper(const_42, util_create_list_253, cache, const_43, device)
    ce_20_0 = ce_20[0]
    const_44 = main_const_eval_22
    const_45 = "main_const_eval_22"
    cez_1 = utils.constEvalFuncWrapperZeroArg(const_44, cache, const_45, device)
    cez_1_0 = cez_1[0]
    const_46 = main_const_eval_23
    util_create_list_254 = [weights[189], weights[456], weights[458]]
    const_47 = "main_const_eval_23"
    ce_21 = utils.constEvalFuncWrapper(const_46, util_create_list_254, cache, const_47, device)
    ce_21_0 = ce_21[0]
    const_48 = main_const_eval_24
    util_create_list_255 = [weights[103]]
    const_49 = "main_const_eval_24"
    ce_22 = utils.constEvalFuncWrapper(const_48, util_create_list_255, cache, const_49, device)
    ce_22_0 = ce_22[0]
    const_50 = main_const_eval_25
    util_create_list_256 = [weights[267]]
    const_51 = "main_const_eval_25"
    ce_23 = utils.constEvalFuncWrapper(const_50, util_create_list_256, cache, const_51, device)
    ce_23_0 = ce_23[0]
    const_52 = main_const_eval_26
    util_create_list_257 = [weights[286], weights[425], weights[427]]
    const_53 = "main_const_eval_26"
    ce_24 = utils.constEvalFuncWrapper(const_52, util_create_list_257, cache, const_53, device)
    ce_24_0 = ce_24[0]
    const_54 = main_const_eval_27
    util_create_list_258 = [weights[361]]
    const_55 = "main_const_eval_27"
    ce_25 = utils.constEvalFuncWrapper(const_54, util_create_list_258, cache, const_55, device)
    ce_25_0 = ce_25[0]
    const_56 = main_const_eval_28
    util_create_list_259 = [weights[93], weights[488], weights[490]]
    const_57 = "main_const_eval_28"
    ce_26 = utils.constEvalFuncWrapper(const_56, util_create_list_259, cache, const_57, device)
    ce_26_0 = ce_26[0]
    const_58 = main_const_eval_29
    util_create_list_260 = [weights[217]]
    const_59 = "main_const_eval_29"
    ce_27 = utils.constEvalFuncWrapper(const_58, util_create_list_260, cache, const_59, device)
    ce_27_0 = ce_27[0]
    const_60 = main_const_eval_30
    util_create_list_261 = [weights[69], weights[496], weights[498]]
    const_61 = "main_const_eval_30"
    ce_28 = utils.constEvalFuncWrapper(const_60, util_create_list_261, cache, const_61, device)
    ce_28_0 = ce_28[0]
    const_62 = main_const_eval_31
    util_create_list_262 = [weights[55]]
    const_63 = "main_const_eval_31"
    ce_29 = utils.constEvalFuncWrapper(const_62, util_create_list_262, cache, const_63, device)
    ce_29_0 = ce_29[0]
    const_64 = main_const_eval_32
    const_65 = "main_const_eval_32"
    cez_2 = utils.constEvalFuncWrapperZeroArg(const_64, cache, const_65, device)
    cez_2_0 = cez_2[0]
    const_66 = main_const_eval_33
    util_create_list_263 = [weights[81], weights[492], weights[494]]
    const_67 = "main_const_eval_33"
    ce_30 = utils.constEvalFuncWrapper(const_66, util_create_list_263, cache, const_67, device)
    ce_30_0 = ce_30[0]
    const_68 = main_const_eval_34
    util_create_list_264 = [weights[163]]
    const_69 = "main_const_eval_34"
    ce_31 = utils.constEvalFuncWrapper(const_68, util_create_list_264, cache, const_69, device)
    ce_31_0 = ce_31[0]
    const_70 = main_const_eval_35
    util_create_list_265 = [weights[117], weights[480], weights[482]]
    const_71 = "main_const_eval_35"
    ce_32 = utils.constEvalFuncWrapper(const_70, util_create_list_265, cache, const_71, device)
    ce_32_0 = ce_32[0]
    const_72 = main_const_eval_36
    util_create_list_266 = [weights[58], weights[501], weights[503]]
    const_73 = "main_const_eval_36"
    ce_33 = utils.constEvalFuncWrapper(const_72, util_create_list_266, cache, const_73, device)
    ce_33_0 = ce_33[0]
    const_74 = main_const_eval_37
    util_create_list_267 = [weights[151]]
    const_75 = "main_const_eval_37"
    ce_34 = utils.constEvalFuncWrapper(const_74, util_create_list_267, cache, const_75, device)
    ce_34_0 = ce_34[0]
    const_76 = main_const_eval_38
    util_create_list_268 = [weights[79]]
    const_77 = "main_const_eval_38"
    ce_35 = utils.constEvalFuncWrapper(const_76, util_create_list_268, cache, const_77, device)
    ce_35_0 = ce_35[0]
    const_78 = main_const_eval_39
    util_create_list_269 = [weights[133]]
    const_79 = "main_const_eval_39"
    ce_36 = utils.constEvalFuncWrapper(const_78, util_create_list_269, cache, const_79, device)
    ce_36_0 = ce_36[0]
    const_80 = main_const_eval_40
    util_create_list_270 = [weights[207]]
    const_81 = "main_const_eval_40"
    ce_37 = utils.constEvalFuncWrapper(const_80, util_create_list_270, cache, const_81, device)
    ce_37_0 = ce_37[0]
    const_82 = main_const_eval_41
    util_create_list_271 = [weights[205]]
    const_83 = "main_const_eval_41"
    ce_38 = utils.constEvalFuncWrapper(const_82, util_create_list_271, cache, const_83, device)
    ce_38_0 = ce_38[0]
    const_84 = main_const_eval_42
    util_create_list_272 = [weights[15]]
    const_85 = "main_const_eval_42"
    ce_39 = utils.constEvalFuncWrapper(const_84, util_create_list_272, cache, const_85, device)
    ce_39_0 = ce_39[0]
    const_86 = main_const_eval_43
    util_create_list_273 = [weights[201], weights[452], weights[454]]
    const_87 = "main_const_eval_43"
    ce_40 = utils.constEvalFuncWrapper(const_86, util_create_list_273, cache, const_87, device)
    ce_40_0 = ce_40[0]
    const_88 = main_const_eval_44
    util_create_list_274 = [weights[45], weights[504], weights[506]]
    const_89 = "main_const_eval_44"
    ce_41 = utils.constEvalFuncWrapper(const_88, util_create_list_274, cache, const_89, device)
    ce_41_0 = ce_41[0]
    const_90 = main_const_eval_45
    util_create_list_275 = [weights[94], weights[489], weights[491]]
    const_91 = "main_const_eval_45"
    ce_42 = utils.constEvalFuncWrapper(const_90, util_create_list_275, cache, const_91, device)
    ce_42_0 = ce_42[0]
    const_92 = main_const_eval_46
    util_create_list_276 = [weights[63]]
    const_93 = "main_const_eval_46"
    ce_43 = utils.constEvalFuncWrapper(const_92, util_create_list_276, cache, const_93, device)
    ce_43_0 = ce_43[0]
    const_94 = main_const_eval_47
    util_create_list_277 = [weights[171]]
    const_95 = "main_const_eval_47"
    ce_44 = utils.constEvalFuncWrapper(const_94, util_create_list_277, cache, const_95, device)
    ce_44_0 = ce_44[0]
    const_96 = main_const_eval_48
    util_create_list_278 = [weights[339]]
    const_97 = "main_const_eval_48"
    ce_45 = utils.constEvalFuncWrapper(const_96, util_create_list_278, cache, const_97, device)
    ce_45_0 = ce_45[0]
    const_98 = main_const_eval_49
    util_create_list_279 = [weights[389]]
    const_99 = "main_const_eval_49"
    ce_46 = utils.constEvalFuncWrapper(const_98, util_create_list_279, cache, const_99, device)
    ce_46_0 = ce_46[0]
    const_100 = main_const_eval_50
    util_create_list_280 = [weights[154], weights[469], weights[471]]
    const_101 = "main_const_eval_50"
    ce_47 = utils.constEvalFuncWrapper(const_100, util_create_list_280, cache, const_101, device)
    ce_47_0 = ce_47[0]
    const_102 = main_const_eval_51
    util_create_list_281 = [weights[2]]
    const_103 = "main_const_eval_51"
    ce_48 = utils.constEvalFuncWrapper(const_102, util_create_list_281, cache, const_103, device)
    ce_48_0 = ce_48[0]
    const_104 = main_const_eval_52
    util_create_list_282 = [weights[51]]
    const_105 = "main_const_eval_52"
    ce_49 = utils.constEvalFuncWrapper(const_104, util_create_list_282, cache, const_105, device)
    ce_49_0 = ce_49[0]
    const_106 = main_const_eval_53
    const_107 = "main_const_eval_53"
    cez_3 = utils.constEvalFuncWrapperZeroArg(const_106, cache, const_107, device)
    cez_3_0 = cez_3[0]
    const_108 = main_const_eval_54
    util_create_list_283 = [weights[21], weights[512], weights[514]]
    const_109 = "main_const_eval_54"
    ce_50 = utils.constEvalFuncWrapper(const_108, util_create_list_283, cache, const_109, device)
    ce_50_0 = ce_50[0]
    const_110 = main_const_eval_55
    util_create_list_284 = [weights[235]]
    const_111 = "main_const_eval_55"
    ce_51 = utils.constEvalFuncWrapper(const_110, util_create_list_284, cache, const_111, device)
    ce_51_0 = ce_51[0]
    const_112 = main_const_eval_56
    util_create_list_285 = [weights[183]]
    const_113 = "main_const_eval_56"
    ce_52 = utils.constEvalFuncWrapper(const_112, util_create_list_285, cache, const_113, device)
    ce_52_0 = ce_52[0]
    const_114 = main_const_eval_57
    util_create_list_286 = [weights[387], weights[388]]
    const_115 = "main_const_eval_57"
    ce_53 = utils.constEvalFuncWrapper(const_114, util_create_list_286, cache, const_115, device)
    ce_53_0 = ce_53[0]
    const_116 = main_const_eval_58
    util_create_list_287 = [weights[153], weights[468], weights[470]]
    const_117 = "main_const_eval_58"
    ce_54 = utils.constEvalFuncWrapper(const_116, util_create_list_287, cache, const_117, device)
    ce_54_0 = ce_54[0]
    const_118 = main_const_eval_59
    const_119 = "main_const_eval_59"
    cez_4 = utils.constEvalFuncWrapperZeroArg(const_118, cache, const_119, device)
    cez_4_0 = cez_4[0]
    const_120 = main_const_eval_60
    util_create_list_288 = [weights[130], weights[477], weights[479]]
    const_121 = "main_const_eval_60"
    ce_55 = utils.constEvalFuncWrapper(const_120, util_create_list_288, cache, const_121, device)
    ce_55_0 = ce_55[0]
    const_122 = main_const_eval_61
    util_create_list_289 = [weights[123]]
    const_123 = "main_const_eval_61"
    ce_56 = utils.constEvalFuncWrapper(const_122, util_create_list_289, cache, const_123, device)
    ce_56_0 = ce_56[0]
    const_124 = main_const_eval_62
    util_create_list_290 = [weights[43]]
    const_125 = "main_const_eval_62"
    ce_57 = utils.constEvalFuncWrapper(const_124, util_create_list_290, cache, const_125, device)
    ce_57_0 = ce_57[0]
    const_126 = main_const_eval_63
    util_create_list_291 = [weights[315]]
    const_127 = "main_const_eval_63"
    ce_58 = utils.constEvalFuncWrapper(const_126, util_create_list_291, cache, const_127, device)
    ce_58_0 = ce_58[0]
    const_128 = main_const_eval_64
    util_create_list_292 = [weights[37]]
    const_129 = "main_const_eval_64"
    ce_59 = utils.constEvalFuncWrapper(const_128, util_create_list_292, cache, const_129, device)
    ce_59_0 = ce_59[0]
    const_130 = main_const_eval_65
    util_create_list_293 = [weights[99]]
    const_131 = "main_const_eval_65"
    ce_60 = utils.constEvalFuncWrapper(const_130, util_create_list_293, cache, const_131, device)
    ce_60_0 = ce_60[0]
    const_132 = main_const_eval_66
    util_create_list_294 = [weights[291]]
    const_133 = "main_const_eval_66"
    ce_61 = utils.constEvalFuncWrapper(const_132, util_create_list_294, cache, const_133, device)
    ce_61_0 = ce_61[0]
    const_134 = main_const_eval_67
    util_create_list_295 = [weights[295]]
    const_135 = "main_const_eval_67"
    ce_62 = utils.constEvalFuncWrapper(const_134, util_create_list_295, cache, const_135, device)
    ce_62_0 = ce_62[0]
    const_136 = main_const_eval_68
    util_create_list_296 = [weights[375]]
    const_137 = "main_const_eval_68"
    ce_63 = utils.constEvalFuncWrapper(const_136, util_create_list_296, cache, const_137, device)
    ce_63_0 = ce_63[0]
    const_138 = main_const_eval_69
    util_create_list_297 = [weights[27]]
    const_139 = "main_const_eval_69"
    ce_64 = utils.constEvalFuncWrapper(const_138, util_create_list_297, cache, const_139, device)
    ce_64_0 = ce_64[0]
    const_140 = main_const_eval_70
    util_create_list_298 = [weights[261], weights[432], weights[434]]
    const_141 = "main_const_eval_70"
    ce_65 = utils.constEvalFuncWrapper(const_140, util_create_list_298, cache, const_141, device)
    ce_65_0 = ce_65[0]
    const_142 = main_const_eval_71
    util_create_list_299 = [weights[219]]
    const_143 = "main_const_eval_71"
    ce_66 = utils.constEvalFuncWrapper(const_142, util_create_list_299, cache, const_143, device)
    ce_66_0 = ce_66[0]
    const_144 = main_const_eval_72
    util_create_list_300 = [weights[237], weights[440], weights[442]]
    const_145 = "main_const_eval_72"
    ce_67 = utils.constEvalFuncWrapper(const_144, util_create_list_300, cache, const_145, device)
    ce_67_0 = ce_67[0]
    const_146 = main_const_eval_73
    util_create_list_301 = [weights[127]]
    const_147 = "main_const_eval_73"
    ce_68 = utils.constEvalFuncWrapper(const_146, util_create_list_301, cache, const_147, device)
    ce_68_0 = ce_68[0]
    const_148 = main_const_eval_74
    util_create_list_302 = [weights[34], weights[509], weights[511]]
    const_149 = "main_const_eval_74"
    ce_69 = utils.constEvalFuncWrapper(const_148, util_create_list_302, cache, const_149, device)
    ce_69_0 = ce_69[0]
    const_150 = main_const_eval_75
    util_create_list_303 = [weights[319]]
    const_151 = "main_const_eval_75"
    ce_70 = utils.constEvalFuncWrapper(const_150, util_create_list_303, cache, const_151, device)
    ce_70_0 = ce_70[0]
    const_152 = main_const_eval_76
    util_create_list_304 = [weights[321], weights[412], weights[414]]
    const_153 = "main_const_eval_76"
    ce_71 = utils.constEvalFuncWrapper(const_152, util_create_list_304, cache, const_153, device)
    ce_71_0 = ce_71[0]
    const_154 = main_const_eval_77
    util_create_list_305 = [weights[166], weights[465], weights[467]]
    const_155 = "main_const_eval_77"
    ce_72 = utils.constEvalFuncWrapper(const_154, util_create_list_305, cache, const_155, device)
    ce_72_0 = ce_72[0]
    const_156 = main_const_eval_78
    util_create_list_306 = [weights[57], weights[500], weights[502]]
    const_157 = "main_const_eval_78"
    ce_73 = utils.constEvalFuncWrapper(const_156, util_create_list_306, cache, const_157, device)
    ce_73_0 = ce_73[0]
    const_158 = main_const_eval_79
    util_create_list_307 = [weights[337]]
    const_159 = "main_const_eval_79"
    ce_74 = utils.constEvalFuncWrapper(const_158, util_create_list_307, cache, const_159, device)
    ce_74_0 = ce_74[0]
    const_160 = main_const_eval_80
    util_create_list_308 = [weights[333], weights[408], weights[410]]
    const_161 = "main_const_eval_80"
    ce_75 = utils.constEvalFuncWrapper(const_160, util_create_list_308, cache, const_161, device)
    ce_75_0 = ce_75[0]
    const_162 = main_const_eval_81
    util_create_list_309 = [weights[379]]
    const_163 = "main_const_eval_81"
    ce_76 = utils.constEvalFuncWrapper(const_162, util_create_list_309, cache, const_163, device)
    ce_76_0 = ce_76[0]
    const_164 = main_const_eval_82
    util_create_list_310 = [weights[283]]
    const_165 = "main_const_eval_82"
    ce_77 = utils.constEvalFuncWrapper(const_164, util_create_list_310, cache, const_165, device)
    ce_77_0 = ce_77[0]
    const_166 = main_const_eval_83
    util_create_list_311 = [weights[19]]
    const_167 = "main_const_eval_83"
    ce_78 = utils.constEvalFuncWrapper(const_166, util_create_list_311, cache, const_167, device)
    ce_78_0 = ce_78[0]
    const_168 = main_const_eval_84
    util_create_list_312 = [weights[75]]
    const_169 = "main_const_eval_84"
    ce_79 = utils.constEvalFuncWrapper(const_168, util_create_list_312, cache, const_169, device)
    ce_79_0 = ce_79[0]
    const_170 = main_const_eval_85
    util_create_list_313 = [weights[85]]
    const_171 = "main_const_eval_85"
    ce_80 = utils.constEvalFuncWrapper(const_170, util_create_list_313, cache, const_171, device)
    ce_80_0 = ce_80[0]
    const_172 = main_const_eval_86
    util_create_list_314 = [weights[255]]
    const_173 = "main_const_eval_86"
    ce_81 = utils.constEvalFuncWrapper(const_172, util_create_list_314, cache, const_173, device)
    ce_81_0 = ce_81[0]
    const_174 = main_const_eval_87
    util_create_list_315 = [weights[271]]
    const_175 = "main_const_eval_87"
    ce_82 = utils.constEvalFuncWrapper(const_174, util_create_list_315, cache, const_175, device)
    ce_82_0 = ce_82[0]
    const_176 = main_const_eval_88
    util_create_list_316 = [weights[159]]
    const_177 = "main_const_eval_88"
    ce_83 = utils.constEvalFuncWrapper(const_176, util_create_list_316, cache, const_177, device)
    ce_83_0 = ce_83[0]
    const_178 = main_const_eval_89
    util_create_list_317 = [weights[358], weights[401], weights[403]]
    const_179 = "main_const_eval_89"
    ce_84 = utils.constEvalFuncWrapper(const_178, util_create_list_317, cache, const_179, device)
    ce_84_0 = ce_84[0]
    const_180 = main_const_eval_90
    util_create_list_318 = [weights[139]]
    const_181 = "main_const_eval_90"
    ce_85 = utils.constEvalFuncWrapper(const_180, util_create_list_318, cache, const_181, device)
    ce_85_0 = ce_85[0]
    const_182 = main_const_eval_91
    util_create_list_319 = [weights[277]]
    const_183 = "main_const_eval_91"
    ce_86 = utils.constEvalFuncWrapper(const_182, util_create_list_319, cache, const_183, device)
    ce_86_0 = ce_86[0]
    const_184 = main_const_eval_92
    util_create_list_320 = [weights[303]]
    const_185 = "main_const_eval_92"
    ce_87 = utils.constEvalFuncWrapper(const_184, util_create_list_320, cache, const_185, device)
    ce_87_0 = ce_87[0]
    const_186 = main_const_eval_93
    util_create_list_321 = [weights[370], weights[397], weights[399]]
    const_187 = "main_const_eval_93"
    ce_88 = utils.constEvalFuncWrapper(const_186, util_create_list_321, cache, const_187, device)
    ce_88_0 = ce_88[0]
    const_188 = main_const_eval_94
    util_create_list_322 = [weights[331]]
    const_189 = "main_const_eval_94"
    ce_89 = utils.constEvalFuncWrapper(const_188, util_create_list_322, cache, const_189, device)
    ce_89_0 = ce_89[0]
    const_190 = main_const_eval_95
    util_create_list_323 = [weights[157]]
    const_191 = "main_const_eval_95"
    ce_90 = utils.constEvalFuncWrapper(const_190, util_create_list_323, cache, const_191, device)
    ce_90_0 = ce_90[0]
    const_192 = main_const_eval_96
    util_create_list_324 = [weights[327]]
    const_193 = "main_const_eval_96"
    ce_91 = utils.constEvalFuncWrapper(const_192, util_create_list_324, cache, const_193, device)
    ce_91_0 = ce_91[0]
    const_194 = main_const_eval_97
    util_create_list_325 = [weights[193]]
    const_195 = "main_const_eval_97"
    ce_92 = utils.constEvalFuncWrapper(const_194, util_create_list_325, cache, const_195, device)
    ce_92_0 = ce_92[0]
    const_196 = main_const_eval_98
    util_create_list_326 = [weights[70], weights[497], weights[499]]
    const_197 = "main_const_eval_98"
    ce_93 = utils.constEvalFuncWrapper(const_196, util_create_list_326, cache, const_197, device)
    ce_93_0 = ce_93[0]
    const_198 = main_const_eval_99
    util_create_list_327 = [weights[322], weights[413], weights[415]]
    const_199 = "main_const_eval_99"
    ce_94 = utils.constEvalFuncWrapper(const_198, util_create_list_327, cache, const_199, device)
    ce_94_0 = ce_94[0]
    const_200 = main_const_eval_100
    util_create_list_328 = [weights[226], weights[445], weights[447]]
    const_201 = "main_const_eval_100"
    ce_95 = utils.constEvalFuncWrapper(const_200, util_create_list_328, cache, const_201, device)
    ce_95_0 = ce_95[0]
    const_202 = main_const_eval_101
    const_203 = "main_const_eval_101"
    cez_5 = utils.constEvalFuncWrapperZeroArg(const_202, cache, const_203, device)
    cez_5_0 = cez_5[0]
    const_204 = main_const_eval_102
    util_create_list_329 = [weights[297], weights[420], weights[422]]
    const_205 = "main_const_eval_102"
    ce_96 = utils.constEvalFuncWrapper(const_204, util_create_list_329, cache, const_205, device)
    ce_96_0 = ce_96[0]
    const_206 = main_const_eval_103
    util_create_list_330 = [weights[349]]
    const_207 = "main_const_eval_103"
    ce_97 = utils.constEvalFuncWrapper(const_206, util_create_list_330, cache, const_207, device)
    ce_97_0 = ce_97[0]
    const_208 = main_const_eval_104
    util_create_list_331 = [weights[279]]
    const_209 = "main_const_eval_104"
    ce_98 = utils.constEvalFuncWrapper(const_208, util_create_list_331, cache, const_209, device)
    ce_98_0 = ce_98[0]
    const_210 = main_const_eval_105
    util_create_list_332 = [weights[325]]
    const_211 = "main_const_eval_105"
    ce_99 = utils.constEvalFuncWrapper(const_210, util_create_list_332, cache, const_211, device)
    ce_99_0 = ce_99[0]
    const_212 = main_const_eval_106
    util_create_list_333 = [weights[178], weights[461], weights[463]]
    const_213 = "main_const_eval_106"
    ce_100 = utils.constEvalFuncWrapper(const_212, util_create_list_333, cache, const_213, device)
    ce_100_0 = ce_100[0]
    const_214 = main_const_eval_107
    util_create_list_334 = [weights[262], weights[433], weights[435]]
    const_215 = "main_const_eval_107"
    ce_101 = utils.constEvalFuncWrapper(const_214, util_create_list_334, cache, const_215, device)
    ce_101_0 = ce_101[0]
    const_216 = main_const_eval_108
    util_create_list_335 = [weights[142], weights[473], weights[475]]
    const_217 = "main_const_eval_108"
    ce_102 = utils.constEvalFuncWrapper(const_216, util_create_list_335, cache, const_217, device)
    ce_102_0 = ce_102[0]
    const_218 = main_const_eval_109
    util_create_list_336 = [weights[105], weights[484], weights[486]]
    const_219 = "main_const_eval_109"
    ce_103 = utils.constEvalFuncWrapper(const_218, util_create_list_336, cache, const_219, device)
    ce_103_0 = ce_103[0]
    const_220 = main_const_eval_110
    util_create_list_337 = [weights[175]]
    const_221 = "main_const_eval_110"
    ce_104 = utils.constEvalFuncWrapper(const_220, util_create_list_337, cache, const_221, device)
    ce_104_0 = ce_104[0]
    const_222 = main_const_eval_111
    util_create_list_338 = [weights[229]]
    const_223 = "main_const_eval_111"
    ce_105 = utils.constEvalFuncWrapper(const_222, util_create_list_338, cache, const_223, device)
    ce_105_0 = ce_105[0]
    const_224 = main_const_eval_112
    util_create_list_339 = [weights[190], weights[457], weights[459]]
    const_225 = "main_const_eval_112"
    ce_106 = utils.constEvalFuncWrapper(const_224, util_create_list_339, cache, const_225, device)
    ce_106_0 = ce_106[0]
    const_226 = main_const_eval_113
    util_create_list_340 = [weights[369], weights[396], weights[398]]
    const_227 = "main_const_eval_113"
    ce_107 = utils.constEvalFuncWrapper(const_226, util_create_list_340, cache, const_227, device)
    ce_107_0 = ce_107[0]
    const_228 = main_const_eval_114
    util_create_list_341 = [weights[87]]
    const_229 = "main_const_eval_114"
    ce_108 = utils.constEvalFuncWrapper(const_228, util_create_list_341, cache, const_229, device)
    ce_108_0 = ce_108[0]
    const_230 = main_const_eval_115
    util_create_list_342 = [weights[223]]
    const_231 = "main_const_eval_115"
    ce_109 = utils.constEvalFuncWrapper(const_230, util_create_list_342, cache, const_231, device)
    ce_109_0 = ce_109[0]
    const_232 = main_const_eval_116
    util_create_list_343 = [weights[346], weights[405], weights[407]]
    const_233 = "main_const_eval_116"
    ce_110 = utils.constEvalFuncWrapper(const_232, util_create_list_343, cache, const_233, device)
    ce_110_0 = ce_110[0]
    const_234 = main_const_eval_117
    util_create_list_344 = [weights[345], weights[404], weights[406]]
    const_235 = "main_const_eval_117"
    ce_111 = utils.constEvalFuncWrapper(const_234, util_create_list_344, cache, const_235, device)
    ce_111_0 = ce_111[0]
    const_236 = main_const_eval_118
    util_create_list_345 = [weights[307]]
    const_237 = "main_const_eval_118"
    ce_112 = utils.constEvalFuncWrapper(const_236, util_create_list_345, cache, const_237, device)
    ce_112_0 = ce_112[0]
    const_238 = main_const_eval_119
    util_create_list_346 = [weights[145]]
    const_239 = "main_const_eval_119"
    ce_113 = utils.constEvalFuncWrapper(const_238, util_create_list_346, cache, const_239, device)
    ce_113_0 = ce_113[0]
    const_240 = main_const_eval_120
    util_create_list_347 = [weights[135]]
    const_241 = "main_const_eval_120"
    ce_114 = utils.constEvalFuncWrapper(const_240, util_create_list_347, cache, const_241, device)
    ce_114_0 = ce_114[0]
    const_242 = main_const_eval_121
    util_create_list_348 = [weights[169]]
    const_243 = "main_const_eval_121"
    ce_115 = utils.constEvalFuncWrapper(const_242, util_create_list_348, cache, const_243, device)
    ce_115_0 = ce_115[0]
    const_244 = main_const_eval_122
    util_create_list_349 = [weights[391]]
    const_245 = "main_const_eval_122"
    ce_116 = utils.constEvalFuncWrapper(const_244, util_create_list_349, cache, const_245, device)
    ce_116_0 = ce_116[0]
    const_246 = main_const_eval_123
    util_create_list_350 = [weights[73]]
    const_247 = "main_const_eval_123"
    ce_117 = utils.constEvalFuncWrapper(const_246, util_create_list_350, cache, const_247, device)
    ce_117_0 = ce_117[0]
    const_248 = main_const_eval_124
    const_249 = "main_const_eval_124"
    cez_6 = utils.constEvalFuncWrapperZeroArg(const_248, cache, const_249, device)
    cez_6_0 = cez_6[0]
    const_250 = main_const_eval_125
    util_create_list_351 = [weights[351]]
    const_251 = "main_const_eval_125"
    ce_118 = utils.constEvalFuncWrapper(const_250, util_create_list_351, cache, const_251, device)
    ce_118_0 = ce_118[0]
    const_252 = main_const_eval_126
    util_create_list_352 = [weights[82], weights[493], weights[495]]
    const_253 = "main_const_eval_126"
    ce_119 = utils.constEvalFuncWrapper(const_252, util_create_list_352, cache, const_253, device)
    ce_119_0 = ce_119[0]
    const_254 = main_const_eval_127
    util_create_list_353 = [weights[310], weights[417], weights[419]]
    const_255 = "main_const_eval_127"
    ce_120 = utils.constEvalFuncWrapper(const_254, util_create_list_353, cache, const_255, device)
    ce_120_0 = ce_120[0]
    const_256 = main_const_eval_128
    util_create_list_354 = [weights[91]]
    const_257 = "main_const_eval_128"
    ce_121 = utils.constEvalFuncWrapper(const_256, util_create_list_354, cache, const_257, device)
    ce_121_0 = ce_121[0]
    const_258 = main_const_eval_129
    util_create_list_355 = [weights[177], weights[460], weights[462]]
    const_259 = "main_const_eval_129"
    ce_122 = utils.constEvalFuncWrapper(const_258, util_create_list_355, cache, const_259, device)
    ce_122_0 = ce_122[0]
    const_260 = main_const_eval_130
    util_create_list_356 = [weights[382], weights[393], weights[395]]
    const_261 = "main_const_eval_130"
    ce_123 = utils.constEvalFuncWrapper(const_260, util_create_list_356, cache, const_261, device)
    ce_123_0 = ce_123[0]
    const_262 = main_const_eval_131
    util_create_list_357 = [weights[357], weights[400], weights[402]]
    const_263 = "main_const_eval_131"
    ce_124 = utils.constEvalFuncWrapper(const_262, util_create_list_357, cache, const_263, device)
    ce_124_0 = ce_124[0]
    const_264 = main_const_eval_132
    util_create_list_358 = [weights[195]]
    const_265 = "main_const_eval_132"
    ce_125 = utils.constEvalFuncWrapper(const_264, util_create_list_358, cache, const_265, device)
    ce_125_0 = ce_125[0]
    const_266 = main_const_eval_133
    util_create_list_359 = [weights[11]]
    const_267 = "main_const_eval_133"
    ce_126 = utils.constEvalFuncWrapper(const_266, util_create_list_359, cache, const_267, device)
    ce_126_0 = ce_126[0]
    const_268 = main_const_eval_134
    util_create_list_360 = [weights[129], weights[476], weights[478]]
    const_269 = "main_const_eval_134"
    ce_127 = utils.constEvalFuncWrapper(const_268, util_create_list_360, cache, const_269, device)
    ce_127_0 = ce_127[0]
    const_270 = main_const_eval_135
    util_create_list_361 = [weights[225], weights[444], weights[446]]
    const_271 = "main_const_eval_135"
    ce_128 = utils.constEvalFuncWrapper(const_270, util_create_list_361, cache, const_271, device)
    ce_128_0 = ce_128[0]
    const_272 = main_const_eval_136
    const_273 = "main_const_eval_136"
    cez_7 = utils.constEvalFuncWrapperZeroArg(const_272, cache, const_273, device)
    cez_7_0 = cez_7[0]
    const_274 = main_const_eval_137
    util_create_list_362 = [weights[259]]
    const_275 = "main_const_eval_137"
    ce_129 = utils.constEvalFuncWrapper(const_274, util_create_list_362, cache, const_275, device)
    ce_129_0 = ce_129[0]
    const_276 = main_const_eval_138
    util_create_list_363 = [weights[199]]
    const_277 = "main_const_eval_138"
    ce_130 = utils.constEvalFuncWrapper(const_276, util_create_list_363, cache, const_277, device)
    ce_130_0 = ce_130[0]
    const_278 = main_const_eval_139
    util_create_list_364 = [weights[31]]
    const_279 = "main_const_eval_139"
    ce_131 = utils.constEvalFuncWrapper(const_278, util_create_list_364, cache, const_279, device)
    ce_131_0 = ce_131[0]
    const_280 = main_const_eval_140
    util_create_list_365 = [weights[241]]
    const_281 = "main_const_eval_140"
    ce_132 = utils.constEvalFuncWrapper(const_280, util_create_list_365, cache, const_281, device)
    ce_132_0 = ce_132[0]
    const_282 = main_const_eval_141
    util_create_list_366 = [weights[121]]
    const_283 = "main_const_eval_141"
    ce_133 = utils.constEvalFuncWrapper(const_282, util_create_list_366, cache, const_283, device)
    ce_133_0 = ce_133[0]
    const_284 = main_const_eval_142
    util_create_list_367 = [weights[181]]
    const_285 = "main_const_eval_142"
    ce_134 = utils.constEvalFuncWrapper(const_284, util_create_list_367, cache, const_285, device)
    ce_134_0 = ce_134[0]
    const_286 = main_const_eval_143
    util_create_list_368 = [weights[115]]
    const_287 = "main_const_eval_143"
    ce_135 = utils.constEvalFuncWrapper(const_286, util_create_list_368, cache, const_287, device)
    ce_135_0 = ce_135[0]
    const_288 = main_const_eval_144
    util_create_list_369 = [weights[39]]
    const_289 = "main_const_eval_144"
    ce_136 = utils.constEvalFuncWrapper(const_288, util_create_list_369, cache, const_289, device)
    ce_136_0 = ce_136[0]
    const_290 = main_const_eval_145
    util_create_list_370 = [weights[4], weights[7], weights[8], weights[517]]
    const_291 = "main_const_eval_145"
    ce_137 = utils.constEvalFuncWrapper(const_290, util_create_list_370, cache, const_291, device)
    ce_137_0 = ce_137[0]
    ce_137_1 = ce_137[1]
    ce_137_2 = ce_137[2]
    const_292 = main_const_eval_146
    util_create_list_371 = [weights[381], weights[392], weights[394]]
    const_293 = "main_const_eval_146"
    ce_138 = utils.constEvalFuncWrapper(const_292, util_create_list_371, cache, const_293, device)
    ce_138_0 = ce_138[0]
    const_294 = main_const_eval_147
    util_create_list_372 = [weights[25]]
    const_295 = "main_const_eval_147"
    ce_139 = utils.constEvalFuncWrapper(const_294, util_create_list_372, cache, const_295, device)
    ce_139_0 = ce_139[0]
    const_296 = main_const_eval_148
    util_create_list_373 = [weights[97]]
    const_297 = "main_const_eval_148"
    ce_140 = utils.constEvalFuncWrapper(const_296, util_create_list_373, cache, const_297, device)
    ce_140_0 = ce_140[0]
    const_298 = main_const_eval_149
    util_create_list_374 = [weights[243]]
    const_299 = "main_const_eval_149"
    ce_141 = utils.constEvalFuncWrapper(const_298, util_create_list_374, cache, const_299, device)
    ce_141_0 = ce_141[0]
    const_300 = main_const_eval_150
    util_create_list_375 = [weights[214], weights[449], weights[451]]
    const_301 = "main_const_eval_150"
    ce_142 = utils.constEvalFuncWrapper(const_300, util_create_list_375, cache, const_301, device)
    ce_142_0 = ce_142[0]
    const_302 = main_const_eval_151
    const_303 = "main_const_eval_151"
    cez_8 = utils.constEvalFuncWrapperZeroArg(const_302, cache, const_303, device)
    cez_8_0 = cez_8[0]
    const_304 = main_const_eval_152
    util_create_list_376 = [weights[367]]
    const_305 = "main_const_eval_152"
    ce_143 = utils.constEvalFuncWrapper(const_304, util_create_list_376, cache, const_305, device)
    ce_143_0 = ce_143[0]
    const_306 = main_const_eval_153
    util_create_list_377 = [weights[273], weights[428], weights[430]]
    const_307 = "main_const_eval_153"
    ce_144 = utils.constEvalFuncWrapper(const_306, util_create_list_377, cache, const_307, device)
    ce_144_0 = ce_144[0]
    const_308 = main_const_eval_154
    util_create_list_378 = [weights[49]]
    const_309 = "main_const_eval_154"
    ce_145 = utils.constEvalFuncWrapper(const_308, util_create_list_378, cache, const_309, device)
    ce_145_0 = ce_145[0]
    const_310 = main_const_eval_155
    util_create_list_379 = [weights[187]]
    const_311 = "main_const_eval_155"
    ce_146 = utils.constEvalFuncWrapper(const_310, util_create_list_379, cache, const_311, device)
    ce_146_0 = ce_146[0]
    const_312 = main_const_eval_156
    util_create_list_380 = [weights[355]]
    const_313 = "main_const_eval_156"
    ce_147 = utils.constEvalFuncWrapper(const_312, util_create_list_380, cache, const_313, device)
    ce_147_0 = ce_147[0]
    const_314 = main_const_eval_157
    util_create_list_381 = [weights[363]]
    const_315 = "main_const_eval_157"
    ce_148 = utils.constEvalFuncWrapper(const_314, util_create_list_381, cache, const_315, device)
    ce_148_0 = ce_148[0]
    const_316 = main_const_eval_158
    util_create_list_382 = [weights[313]]
    const_317 = "main_const_eval_158"
    ce_149 = utils.constEvalFuncWrapper(const_316, util_create_list_382, cache, const_317, device)
    ce_149_0 = ce_149[0]
    const_318 = main_const_eval_159
    util_create_list_383 = [weights[111]]
    const_319 = "main_const_eval_159"
    ce_150 = utils.constEvalFuncWrapper(const_318, util_create_list_383, cache, const_319, device)
    ce_150_0 = ce_150[0]
    const_320 = main_const_eval_160
    util_create_list_384 = [weights[247]]
    const_321 = "main_const_eval_160"
    ce_151 = utils.constEvalFuncWrapper(const_320, util_create_list_384, cache, const_321, device)
    ce_151_0 = ce_151[0]
    const_322 = main_const_eval_161
    util_create_list_385 = [weights[265]]
    const_323 = "main_const_eval_161"
    ce_152 = utils.constEvalFuncWrapper(const_322, util_create_list_385, cache, const_323, device)
    ce_152_0 = ce_152[0]
    const_324 = main_const_eval_162
    util_create_list_386 = [weights[309], weights[416], weights[418]]
    const_325 = "main_const_eval_162"
    ce_153 = utils.constEvalFuncWrapper(const_324, util_create_list_386, cache, const_325, device)
    ce_153_0 = ce_153[0]
    const_326 = main_const_eval_163
    util_create_list_387 = [weights[141], weights[472], weights[474]]
    const_327 = "main_const_eval_163"
    ce_154 = utils.constEvalFuncWrapper(const_326, util_create_list_387, cache, const_327, device)
    ce_154_0 = ce_154[0]
    const_328 = main_const_eval_164
    util_create_list_388 = [weights[109]]
    const_329 = "main_const_eval_164"
    ce_155 = utils.constEvalFuncWrapper(const_328, util_create_list_388, cache, const_329, device)
    ce_155_0 = ce_155[0]
    const_330 = main_const_eval_165
    const_331 = "main_const_eval_165"
    cez_9 = utils.constEvalFuncWrapperZeroArg(const_330, cache, const_331, device)
    cez_9_0 = cez_9[0]
    const_332 = main_const_eval_166
    util_create_list_389 = [weights[250], weights[437], weights[439]]
    const_333 = "main_const_eval_166"
    ce_156 = utils.constEvalFuncWrapper(const_332, util_create_list_389, cache, const_333, device)
    ce_156_0 = ce_156[0]
    const_334 = main_const_eval_167
    util_create_list_390 = [weights[373]]
    const_335 = "main_const_eval_167"
    ce_157 = utils.constEvalFuncWrapper(const_334, util_create_list_390, cache, const_335, device)
    ce_157_0 = ce_157[0]
    const_336 = main_const_eval_168
    util_create_list_391 = [weights[147]]
    const_337 = "main_const_eval_168"
    ce_158 = utils.constEvalFuncWrapper(const_336, util_create_list_391, cache, const_337, device)
    ce_158_0 = ce_158[0]
    const_338 = main_const_eval_169
    util_create_list_392 = [weights[238], weights[441], weights[443]]
    const_339 = "main_const_eval_169"
    ce_159 = utils.constEvalFuncWrapper(const_338, util_create_list_392, cache, const_339, device)
    ce_159_0 = ce_159[0]
    const_340 = main_const_eval_170
    util_create_list_393 = [weights[301]]
    const_341 = "main_const_eval_170"
    ce_160 = utils.constEvalFuncWrapper(const_340, util_create_list_393, cache, const_341, device)
    ce_160_0 = ce_160[0]

    return {
        "cez_0_0": cez_0_0,
        "ce_0_0": ce_0_0,
        "cez_1_0": cez_1_0,
        "ce_1_0": ce_1_0,
        "cez_2_0": cez_2_0,
        "ce_2_0": ce_2_0,
        "cez_3_0": cez_3_0,
        "ce_3_0": ce_3_0,
        "cez_4_0": cez_4_0,
        "ce_4_0": ce_4_0,
        "cez_5_0": cez_5_0,
        "ce_5_0": ce_5_0,
        "cez_6_0": cez_6_0,
        "ce_6_0": ce_6_0,
        "cez_7_0": cez_7_0,
        "ce_7_0": ce_7_0,
        "cez_8_0": cez_8_0,
        "ce_8_0": ce_8_0,
        "cez_9_0": cez_9_0,
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
    }
