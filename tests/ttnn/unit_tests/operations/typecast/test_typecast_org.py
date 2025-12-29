import ttnn
import tests.ttnn.unit_tests.operations.typecast.utils as utils


def main_const_eval_0():
    utils_DeviceGetter_get_device_0 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_0 = ttnn.Tensor(
        [784.0, 28.0, 1.0, 1.0],
        [4, 1],
        ttnn.DataType.FLOAT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_0,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_0 = [ttnn_Tensor_0]
    return util_create_list_0


def main_const_eval_1():
    utils_DeviceGetter_get_device_1 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_0 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=0.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_1 = [ttnn_full_0]
    return util_create_list_1


def main_const_eval_2():
    utils_DeviceGetter_get_device_2 = utils.DeviceGetter.get_device((1, 1))
    ttnn_Tensor_1 = ttnn.Tensor(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
            176,
            177,
            178,
            179,
            180,
            181,
            182,
            183,
            184,
            185,
            186,
            187,
            188,
            189,
            190,
            191,
            192,
            193,
            194,
            195,
            196,
            197,
            198,
            199,
            200,
            201,
            202,
            203,
            204,
            205,
            206,
            207,
            208,
            209,
            210,
            211,
            212,
            213,
            214,
            215,
            216,
            217,
            218,
            219,
            220,
            221,
            222,
            223,
            224,
            225,
            226,
            227,
            228,
            229,
            230,
            231,
            232,
            233,
            234,
            235,
            236,
            237,
            238,
            239,
            240,
            241,
            242,
            243,
            244,
            245,
            246,
            247,
            248,
            249,
            250,
            251,
            252,
            253,
            254,
            255,
        ],
        [1, 256],
        ttnn.DataType.INT32,
        ttnn.Layout.TILE,
        utils_DeviceGetter_get_device_2,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_full_1 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=256,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_full_2 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_0 = ttnn.reshape(
        ttnn_Tensor_1,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_Tensor_1, False)
    ttnn_lt_0 = ttnn.lt(
        ttnn_reshape_0,
        ttnn_full_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_0 = ttnn.add(
        ttnn_reshape_0,
        ttnn_full_1,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_full_1, False)
    ttnn_repeat_0 = ttnn.repeat(ttnn_reshape_0, ttnn.Shape([1, 1, 7, 25281]))
    ttnn.deallocate(ttnn_reshape_0, False)
    ttnn_typecast_0 = ttnn.typecast(
        ttnn_lt_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_0, False)
    ttnn_typecast_1 = ttnn.typecast(
        ttnn_add_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_0, False)
    ttnn_typecast_2 = ttnn.typecast(
        ttnn_repeat_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_0, False)
    ttnn_where_0 = ttnn.where(
        ttnn_typecast_0,
        ttnn_typecast_1,
        ttnn_typecast_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_2, False)
    ttnn.deallocate(ttnn_typecast_1, False)
    ttnn.deallocate(ttnn_typecast_0, False)
    ttnn_typecast_3 = ttnn.typecast(
        ttnn_where_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_0, False)
    ttnn_reshape_1 = ttnn.reshape(
        ttnn_typecast_3,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_3, False)
    ttnn_typecast_4 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_5 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_6 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_7 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_8 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_9 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_10 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_11 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_12 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_13 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_14 = ttnn.typecast(
        ttnn_full_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_15 = ttnn.typecast(
        ttnn_reshape_1,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_1, False)
    util_create_list_2 = [
        ttnn_full_2,
        ttnn_typecast_4,
        ttnn_typecast_5,
        ttnn_typecast_6,
        ttnn_typecast_7,
        ttnn_typecast_8,
        ttnn_typecast_9,
        ttnn_typecast_10,
        ttnn_typecast_11,
        ttnn_typecast_12,
        ttnn_typecast_13,
        ttnn_typecast_14,
        ttnn_typecast_15,
    ]
    return util_create_list_2


def main_const_eval_3():
    utils_DeviceGetter_get_device_3 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_3 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=14.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_3 = [ttnn_full_3]
    return util_create_list_3


def main_const_eval_4():
    utils_DeviceGetter_get_device_4 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_4 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=13.5,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_4 = [ttnn_full_4]
    return util_create_list_4


def main_const_eval_5():
    utils_DeviceGetter_get_device_5 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_5 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=28.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_5,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_5 = [ttnn_full_5]
    return util_create_list_5


def main_const_eval_6():
    utils_DeviceGetter_get_device_6 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_6 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281, 1]),
        fill_value=0,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_16 = ttnn.typecast(
        ttnn_full_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_17 = ttnn.typecast(
        ttnn_full_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_18 = ttnn.typecast(
        ttnn_full_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_19 = ttnn.typecast(
        ttnn_full_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_full_6, False)
    util_create_list_6 = [
        ttnn_typecast_16,
        ttnn_typecast_17,
        ttnn_typecast_18,
        ttnn_typecast_19,
    ]
    return util_create_list_6


def main_const_eval_7():
    utils_DeviceGetter_get_device_7 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_7 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=28,
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_7 = [ttnn_full_7]
    return util_create_list_7


def main_const_eval_8():
    utils_DeviceGetter_get_device_8 = utils.DeviceGetter.get_device((1, 1))
    ttnn_full_8 = ttnn.full(
        shape=ttnn.Shape([1, 256, 7, 25281]),
        fill_value=1.0,
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.TILE,
        device=utils_DeviceGetter_get_device_8,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    util_create_list_8 = [ttnn_full_8]
    return util_create_list_8


CACHED_main_const_eval_0 = None
CACHED_main_const_eval_1 = None
CACHED_main_const_eval_2 = None
CACHED_main_const_eval_3 = None
CACHED_main_const_eval_4 = None
CACHED_main_const_eval_5 = None
CACHED_main_const_eval_6 = None
CACHED_main_const_eval_7 = None
CACHED_main_const_eval_8 = None


def _main(input):
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
    const_0 = main_const_eval_0
    utils_constEvalFuncWrapperZeroArg_0 = utils.constEvalFuncWrapperZeroArg(const_0, CACHED_main_const_eval_0)
    CACHED_main_const_eval_0 = utils_constEvalFuncWrapperZeroArg_0
    utils_constEvalFuncWrapperZeroArg_0_0 = utils_constEvalFuncWrapperZeroArg_0[0]
    const_1 = main_const_eval_1
    utils_constEvalFuncWrapperZeroArg_1 = utils.constEvalFuncWrapperZeroArg(const_1, CACHED_main_const_eval_1)
    CACHED_main_const_eval_1 = utils_constEvalFuncWrapperZeroArg_1
    utils_constEvalFuncWrapperZeroArg_1_0 = utils_constEvalFuncWrapperZeroArg_1[0]
    const_2 = main_const_eval_2
    utils_constEvalFuncWrapperZeroArg_2 = utils.constEvalFuncWrapperZeroArg(const_2, CACHED_main_const_eval_2)
    CACHED_main_const_eval_2 = utils_constEvalFuncWrapperZeroArg_2
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
    utils_constEvalFuncWrapperZeroArg_2_12 = utils_constEvalFuncWrapperZeroArg_2[12]
    const_3 = main_const_eval_3
    utils_constEvalFuncWrapperZeroArg_3 = utils.constEvalFuncWrapperZeroArg(const_3, CACHED_main_const_eval_3)
    CACHED_main_const_eval_3 = utils_constEvalFuncWrapperZeroArg_3
    utils_constEvalFuncWrapperZeroArg_3_0 = utils_constEvalFuncWrapperZeroArg_3[0]
    const_4 = main_const_eval_4
    utils_constEvalFuncWrapperZeroArg_4 = utils.constEvalFuncWrapperZeroArg(const_4, CACHED_main_const_eval_4)
    CACHED_main_const_eval_4 = utils_constEvalFuncWrapperZeroArg_4
    utils_constEvalFuncWrapperZeroArg_4_0 = utils_constEvalFuncWrapperZeroArg_4[0]
    const_5 = main_const_eval_5
    utils_constEvalFuncWrapperZeroArg_5 = utils.constEvalFuncWrapperZeroArg(const_5, CACHED_main_const_eval_5)
    CACHED_main_const_eval_5 = utils_constEvalFuncWrapperZeroArg_5
    utils_constEvalFuncWrapperZeroArg_5_0 = utils_constEvalFuncWrapperZeroArg_5[0]
    const_6 = main_const_eval_6
    utils_constEvalFuncWrapperZeroArg_6 = utils.constEvalFuncWrapperZeroArg(const_6, CACHED_main_const_eval_6)
    CACHED_main_const_eval_6 = utils_constEvalFuncWrapperZeroArg_6
    utils_constEvalFuncWrapperZeroArg_6_0 = utils_constEvalFuncWrapperZeroArg_6[0]
    utils_constEvalFuncWrapperZeroArg_6_1 = utils_constEvalFuncWrapperZeroArg_6[1]
    utils_constEvalFuncWrapperZeroArg_6_2 = utils_constEvalFuncWrapperZeroArg_6[2]
    utils_constEvalFuncWrapperZeroArg_6_3 = utils_constEvalFuncWrapperZeroArg_6[3]
    const_7 = main_const_eval_7
    utils_constEvalFuncWrapperZeroArg_7 = utils.constEvalFuncWrapperZeroArg(const_7, CACHED_main_const_eval_7)
    CACHED_main_const_eval_7 = utils_constEvalFuncWrapperZeroArg_7
    utils_constEvalFuncWrapperZeroArg_7_0 = utils_constEvalFuncWrapperZeroArg_7[0]
    const_8 = main_const_eval_8
    utils_constEvalFuncWrapperZeroArg_8 = utils.constEvalFuncWrapperZeroArg(const_8, CACHED_main_const_eval_8)
    CACHED_main_const_eval_8 = utils_constEvalFuncWrapperZeroArg_8
    utils_constEvalFuncWrapperZeroArg_8_0 = utils_constEvalFuncWrapperZeroArg_8[0]
    ttnn_to_layout_0 = ttnn.to_layout(
        input_0,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_0, False)
    ttnn_reshape_2 = ttnn.reshape(
        ttnn_to_layout_0,
        [1, 1, 7, 25281, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_0, False)
    ttnn_repeat_1 = ttnn.repeat(ttnn_reshape_2, ttnn.Shape([1, 256, 1, 1, 1]))
    ttnn.deallocate(ttnn_reshape_2, False)
    ttnn_slice_0 = ttnn.slice(
        ttnn_repeat_1,
        [0, 0, 0, 0, 0],
        [1, 256, 7, 25281, 1],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_3 = ttnn.reshape(
        ttnn_slice_0,
        [1, 256, 7, 25281],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_0, False)
    ttnn_multiply_0 = ttnn.multiply(
        ttnn_reshape_3,
        utils_constEvalFuncWrapperZeroArg_3_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_3, False)
    ttnn_add_1 = ttnn.add(
        ttnn_multiply_0,
        utils_constEvalFuncWrapperZeroArg_4_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_0, False)
    ttnn_floor_0 = ttnn.floor(
        ttnn_add_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_ge_0 = ttnn.ge(
        ttnn_floor_0,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_1 = ttnn.lt(
        ttnn_floor_0,
        utils_constEvalFuncWrapperZeroArg_5_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_slice_1 = ttnn.slice(
        ttnn_repeat_1,
        [0, 0, 0, 0, 1],
        [1, 256, 7, 25281, 2],
        [1, 1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_repeat_1, False)
    ttnn_reshape_4 = ttnn.reshape(
        ttnn_slice_1,
        [1, 256, 7, 25281],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_slice_1, False)
    ttnn_multiply_1 = ttnn.multiply(
        ttnn_reshape_4,
        utils_constEvalFuncWrapperZeroArg_3_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_4, False)
    ttnn_add_2 = ttnn.add(
        ttnn_multiply_1,
        utils_constEvalFuncWrapperZeroArg_4_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_1, False)
    ttnn_floor_1 = ttnn.floor(
        ttnn_add_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_ge_1 = ttnn.ge(
        ttnn_floor_1,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_2 = ttnn.lt(
        ttnn_floor_1,
        utils_constEvalFuncWrapperZeroArg_5_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_logical_and_0 = ttnn.logical_and(
        ttnn_ge_1,
        ttnn_lt_2,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_2, False)
    ttnn.deallocate(ttnn_ge_1, False)
    ttnn_logical_and_1 = ttnn.logical_and(
        ttnn_lt_1,
        ttnn_logical_and_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_logical_and_2 = ttnn.logical_and(
        ttnn_ge_0,
        ttnn_logical_and_1,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_1, False)
    ttnn_typecast_20 = ttnn.typecast(
        ttnn_floor_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_21 = ttnn.typecast(
        ttnn_logical_and_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_22 = ttnn.typecast(
        ttnn_typecast_20,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_where_1 = ttnn.where(
        ttnn_typecast_21,
        ttnn_typecast_22,
        utils_constEvalFuncWrapperZeroArg_2_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_22, False)
    ttnn.deallocate(ttnn_typecast_21, False)
    ttnn_typecast_23 = ttnn.typecast(
        ttnn_where_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_3 = ttnn.lt(
        ttnn_typecast_23,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_3 = ttnn.add(
        ttnn_typecast_23,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_23, False)
    ttnn_typecast_24 = ttnn.typecast(
        ttnn_lt_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_3, False)
    ttnn_typecast_25 = ttnn.typecast(
        ttnn_add_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_3, False)
    ttnn_where_2 = ttnn.where(
        ttnn_typecast_24,
        ttnn_typecast_25,
        ttnn_where_1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_25, False)
    ttnn.deallocate(ttnn_typecast_24, False)
    ttnn.deallocate(ttnn_where_1, False)
    ttnn_typecast_26 = ttnn.typecast(
        ttnn_where_2,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_2, False)
    ttnn_reshape_5 = ttnn.reshape(
        ttnn_typecast_26,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_26, False)
    ttnn_typecast_27 = ttnn.typecast(
        ttnn_floor_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_28 = ttnn.typecast(
        ttnn_logical_and_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_29 = ttnn.typecast(
        ttnn_typecast_27,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_where_3 = ttnn.where(
        ttnn_typecast_28,
        ttnn_typecast_29,
        utils_constEvalFuncWrapperZeroArg_2_2,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_29, False)
    ttnn.deallocate(ttnn_typecast_28, False)
    ttnn_typecast_30 = ttnn.typecast(
        ttnn_where_3,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_4 = ttnn.lt(
        ttnn_typecast_30,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_4 = ttnn.add(
        ttnn_typecast_30,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_30, False)
    ttnn_typecast_31 = ttnn.typecast(
        ttnn_lt_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_4, False)
    ttnn_typecast_32 = ttnn.typecast(
        ttnn_add_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_4, False)
    ttnn_where_4 = ttnn.where(
        ttnn_typecast_31,
        ttnn_typecast_32,
        ttnn_where_3,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_32, False)
    ttnn.deallocate(ttnn_typecast_31, False)
    ttnn.deallocate(ttnn_where_3, False)
    ttnn_typecast_33 = ttnn.typecast(
        ttnn_where_4,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_4, False)
    ttnn_reshape_6 = ttnn.reshape(
        ttnn_typecast_33,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_33, False)
    ttnn_typecast_34 = ttnn.typecast(
        ttnn_reshape_5,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_5, False)
    ttnn_typecast_35 = ttnn.typecast(
        ttnn_reshape_6,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_6, False)
    util_create_list_9 = [
        utils_constEvalFuncWrapperZeroArg_6_0,
        utils_constEvalFuncWrapperZeroArg_2_3,
        ttnn_typecast_34,
        ttnn_typecast_35,
    ]
    ttnn_concat_0 = ttnn.concat(
        util_create_list_9,
        4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_35, False)
    ttnn.deallocate(ttnn_typecast_34, False)
    ttnn_typecast_36 = ttnn.typecast(
        ttnn_concat_0,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_0, False)
    ttnn_to_layout_1 = ttnn.to_layout(
        input_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_0 = ttnn.permute(
        ttnn_to_layout_1,
        [1, 2, 3, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_1, False)
    ttnn_reshape_7 = ttnn.reshape(
        ttnn_permute_0,
        [200704, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_0, False)
    ttnn_typecast_37 = ttnn.typecast(
        ttnn_typecast_36,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_36, False)
    ttnn_matmul_0 = ttnn.matmul(
        ttnn_typecast_37,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_typecast_37, False)
    ttnn_reshape_8 = ttnn.reshape(
        ttnn_matmul_0,
        [1, 45303552],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_0, False)
    ttnn_typecast_38 = ttnn.typecast(
        ttnn_reshape_8,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_8, False)
    ttnn_to_layout_2 = ttnn.to_layout(
        ttnn_typecast_38,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_38, False)
    ttnn_typecast_39 = ttnn.typecast(
        ttnn_reshape_7,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_7, False)
    ttnn_to_layout_3 = ttnn.to_layout(
        ttnn_typecast_39,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_39, False)
    ttnn_embedding_0 = ttnn.embedding(ttnn_to_layout_2, ttnn_to_layout_3, layout=ttnn.Layout.TILE)
    ttnn.deallocate(ttnn_to_layout_3, False)
    ttnn.deallocate(ttnn_to_layout_2, False)
    ttnn_typecast_40 = ttnn.typecast(
        ttnn_embedding_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_0, False)
    ttnn_reshape_9 = ttnn.reshape(
        ttnn_typecast_40,
        [1, 256, 7, 25281],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_40, False)
    ttnn_add_5 = ttnn.add(
        ttnn_floor_0,
        utils_constEvalFuncWrapperZeroArg_8_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_subtract_0 = ttnn.subtract(
        ttnn_add_5,
        ttnn_add_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_6 = ttnn.add(
        ttnn_floor_1,
        utils_constEvalFuncWrapperZeroArg_8_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_subtract_1 = ttnn.subtract(
        ttnn_add_6,
        ttnn_add_2,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_multiply_2 = ttnn.multiply(
        ttnn_subtract_0,
        ttnn_subtract_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_41 = ttnn.typecast(
        ttnn_logical_and_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_2, False)
    ttnn_where_5 = ttnn.where(
        ttnn_typecast_41,
        ttnn_multiply_2,
        utils_constEvalFuncWrapperZeroArg_1_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_41, False)
    ttnn.deallocate(ttnn_multiply_2, False)
    ttnn_multiply_3 = ttnn.multiply(
        ttnn_reshape_9,
        ttnn_where_5,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_5, False)
    ttnn.deallocate(ttnn_reshape_9, False)
    ttnn_ge_2 = ttnn.ge(
        ttnn_add_5,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_5 = ttnn.lt(
        ttnn_add_5,
        utils_constEvalFuncWrapperZeroArg_5_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_logical_and_3 = ttnn.logical_and(
        ttnn_lt_5,
        ttnn_logical_and_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_0, False)
    ttnn_logical_and_4 = ttnn.logical_and(
        ttnn_ge_2,
        ttnn_logical_and_3,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_3, False)
    ttnn_typecast_42 = ttnn.typecast(
        ttnn_logical_and_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_43 = ttnn.typecast(
        ttnn_typecast_20,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_20, False)
    ttnn_where_6 = ttnn.where(
        ttnn_typecast_42,
        ttnn_typecast_43,
        utils_constEvalFuncWrapperZeroArg_2_4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_43, False)
    ttnn.deallocate(ttnn_typecast_42, False)
    ttnn_typecast_44 = ttnn.typecast(
        ttnn_where_6,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_6 = ttnn.lt(
        ttnn_typecast_44,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_7 = ttnn.add(
        ttnn_typecast_44,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_44, False)
    ttnn_typecast_45 = ttnn.typecast(
        ttnn_lt_6,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_6, False)
    ttnn_typecast_46 = ttnn.typecast(
        ttnn_add_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_7, False)
    ttnn_where_7 = ttnn.where(
        ttnn_typecast_45,
        ttnn_typecast_46,
        ttnn_where_6,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_46, False)
    ttnn.deallocate(ttnn_typecast_45, False)
    ttnn.deallocate(ttnn_where_6, False)
    ttnn_typecast_47 = ttnn.typecast(
        ttnn_where_7,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_7, False)
    ttnn_reshape_10 = ttnn.reshape(
        ttnn_typecast_47,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_47, False)
    ttnn_typecast_48 = ttnn.typecast(
        ttnn_add_5,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_5, False)
    ttnn_typecast_49 = ttnn.typecast(
        ttnn_logical_and_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_50 = ttnn.typecast(
        ttnn_typecast_48,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_where_8 = ttnn.where(
        ttnn_typecast_49,
        ttnn_typecast_50,
        utils_constEvalFuncWrapperZeroArg_2_5,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_50, False)
    ttnn.deallocate(ttnn_typecast_49, False)
    ttnn_typecast_51 = ttnn.typecast(
        ttnn_where_8,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_7 = ttnn.lt(
        ttnn_typecast_51,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_8 = ttnn.add(
        ttnn_typecast_51,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_51, False)
    ttnn_typecast_52 = ttnn.typecast(
        ttnn_lt_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_7, False)
    ttnn_typecast_53 = ttnn.typecast(
        ttnn_add_8,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_8, False)
    ttnn_where_9 = ttnn.where(
        ttnn_typecast_52,
        ttnn_typecast_53,
        ttnn_where_8,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_53, False)
    ttnn.deallocate(ttnn_typecast_52, False)
    ttnn.deallocate(ttnn_where_8, False)
    ttnn_typecast_54 = ttnn.typecast(
        ttnn_where_9,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_9, False)
    ttnn_reshape_11 = ttnn.reshape(
        ttnn_typecast_54,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_54, False)
    ttnn_typecast_55 = ttnn.typecast(
        ttnn_reshape_10,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_10, False)
    ttnn_typecast_56 = ttnn.typecast(
        ttnn_reshape_11,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_11, False)
    util_create_list_10 = [
        utils_constEvalFuncWrapperZeroArg_6_1,
        utils_constEvalFuncWrapperZeroArg_2_6,
        ttnn_typecast_55,
        ttnn_typecast_56,
    ]
    ttnn_concat_1 = ttnn.concat(
        util_create_list_10,
        4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_56, False)
    ttnn.deallocate(ttnn_typecast_55, False)
    ttnn_typecast_57 = ttnn.typecast(
        ttnn_concat_1,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_1, False)
    ttnn_to_layout_4 = ttnn.to_layout(
        input_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_1 = ttnn.permute(
        ttnn_to_layout_4,
        [1, 2, 3, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_4, False)
    ttnn_reshape_12 = ttnn.reshape(
        ttnn_permute_1,
        [200704, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_1, False)
    ttnn_typecast_58 = ttnn.typecast(
        ttnn_typecast_57,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_57, False)
    ttnn_matmul_1 = ttnn.matmul(
        ttnn_typecast_58,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_typecast_58, False)
    ttnn_reshape_13 = ttnn.reshape(
        ttnn_matmul_1,
        [1, 45303552],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_1, False)
    ttnn_typecast_59 = ttnn.typecast(
        ttnn_reshape_13,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_13, False)
    ttnn_to_layout_5 = ttnn.to_layout(
        ttnn_typecast_59,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_59, False)
    ttnn_typecast_60 = ttnn.typecast(
        ttnn_reshape_12,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_12, False)
    ttnn_to_layout_6 = ttnn.to_layout(
        ttnn_typecast_60,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_60, False)
    ttnn_embedding_1 = ttnn.embedding(ttnn_to_layout_5, ttnn_to_layout_6, layout=ttnn.Layout.TILE)
    ttnn.deallocate(ttnn_to_layout_6, False)
    ttnn.deallocate(ttnn_to_layout_5, False)
    ttnn_typecast_61 = ttnn.typecast(
        ttnn_embedding_1,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_1, False)
    ttnn_reshape_14 = ttnn.reshape(
        ttnn_typecast_61,
        [1, 256, 7, 25281],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_61, False)
    ttnn_subtract_2 = ttnn.subtract(
        ttnn_add_1,
        ttnn_floor_0,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_floor_0, False)
    ttnn.deallocate(ttnn_add_1, False)
    ttnn_multiply_4 = ttnn.multiply(
        ttnn_subtract_2,
        ttnn_subtract_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_subtract_1, False)
    ttnn_typecast_62 = ttnn.typecast(
        ttnn_logical_and_4,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_4, False)
    ttnn_where_10 = ttnn.where(
        ttnn_typecast_62,
        ttnn_multiply_4,
        utils_constEvalFuncWrapperZeroArg_1_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_62, False)
    ttnn.deallocate(ttnn_multiply_4, False)
    ttnn_multiply_5 = ttnn.multiply(
        ttnn_reshape_14,
        ttnn_where_10,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_10, False)
    ttnn.deallocate(ttnn_reshape_14, False)
    ttnn_add_9 = ttnn.add(
        ttnn_multiply_3,
        ttnn_multiply_5,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_5, False)
    ttnn.deallocate(ttnn_multiply_3, False)
    ttnn_ge_3 = ttnn.ge(
        ttnn_add_6,
        utils_constEvalFuncWrapperZeroArg_1_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_8 = ttnn.lt(
        ttnn_add_6,
        utils_constEvalFuncWrapperZeroArg_5_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_logical_and_5 = ttnn.logical_and(
        ttnn_ge_3,
        ttnn_lt_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_8, False)
    ttnn.deallocate(ttnn_ge_3, False)
    ttnn_logical_and_6 = ttnn.logical_and(
        ttnn_lt_1,
        ttnn_logical_and_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_1, False)
    ttnn_logical_and_7 = ttnn.logical_and(
        ttnn_ge_0,
        ttnn_logical_and_6,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_6, False)
    ttnn.deallocate(ttnn_ge_0, False)
    ttnn_typecast_63 = ttnn.typecast(
        ttnn_add_6,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_6, False)
    ttnn_typecast_64 = ttnn.typecast(
        ttnn_logical_and_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_65 = ttnn.typecast(
        ttnn_typecast_63,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_where_11 = ttnn.where(
        ttnn_typecast_64,
        ttnn_typecast_65,
        utils_constEvalFuncWrapperZeroArg_2_7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_65, False)
    ttnn.deallocate(ttnn_typecast_64, False)
    ttnn_typecast_66 = ttnn.typecast(
        ttnn_where_11,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_9 = ttnn.lt(
        ttnn_typecast_66,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_10 = ttnn.add(
        ttnn_typecast_66,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_66, False)
    ttnn_typecast_67 = ttnn.typecast(
        ttnn_lt_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_9, False)
    ttnn_typecast_68 = ttnn.typecast(
        ttnn_add_10,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_10, False)
    ttnn_where_12 = ttnn.where(
        ttnn_typecast_67,
        ttnn_typecast_68,
        ttnn_where_11,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_68, False)
    ttnn.deallocate(ttnn_typecast_67, False)
    ttnn.deallocate(ttnn_where_11, False)
    ttnn_typecast_69 = ttnn.typecast(
        ttnn_where_12,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_12, False)
    ttnn_reshape_15 = ttnn.reshape(
        ttnn_typecast_69,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_69, False)
    ttnn_typecast_70 = ttnn.typecast(
        ttnn_logical_and_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_71 = ttnn.typecast(
        ttnn_typecast_27,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_27, False)
    ttnn_where_13 = ttnn.where(
        ttnn_typecast_70,
        ttnn_typecast_71,
        utils_constEvalFuncWrapperZeroArg_2_8,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_71, False)
    ttnn.deallocate(ttnn_typecast_70, False)
    ttnn_typecast_72 = ttnn.typecast(
        ttnn_where_13,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_10 = ttnn.lt(
        ttnn_typecast_72,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_11 = ttnn.add(
        ttnn_typecast_72,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_72, False)
    ttnn_typecast_73 = ttnn.typecast(
        ttnn_lt_10,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_10, False)
    ttnn_typecast_74 = ttnn.typecast(
        ttnn_add_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_11, False)
    ttnn_where_14 = ttnn.where(
        ttnn_typecast_73,
        ttnn_typecast_74,
        ttnn_where_13,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_74, False)
    ttnn.deallocate(ttnn_typecast_73, False)
    ttnn.deallocate(ttnn_where_13, False)
    ttnn_typecast_75 = ttnn.typecast(
        ttnn_where_14,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_14, False)
    ttnn_reshape_16 = ttnn.reshape(
        ttnn_typecast_75,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_75, False)
    ttnn_typecast_76 = ttnn.typecast(
        ttnn_reshape_15,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_15, False)
    ttnn_typecast_77 = ttnn.typecast(
        ttnn_reshape_16,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_16, False)
    util_create_list_11 = [
        utils_constEvalFuncWrapperZeroArg_6_2,
        utils_constEvalFuncWrapperZeroArg_2_9,
        ttnn_typecast_76,
        ttnn_typecast_77,
    ]
    ttnn_concat_2 = ttnn.concat(
        util_create_list_11,
        4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_77, False)
    ttnn.deallocate(ttnn_typecast_76, False)
    ttnn_typecast_78 = ttnn.typecast(
        ttnn_concat_2,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_2, False)
    ttnn_to_layout_7 = ttnn.to_layout(
        input_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_permute_2 = ttnn.permute(
        ttnn_to_layout_7,
        [1, 2, 3, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_7, False)
    ttnn_reshape_17 = ttnn.reshape(
        ttnn_permute_2,
        [200704, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_2, False)
    ttnn_typecast_79 = ttnn.typecast(
        ttnn_typecast_78,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_78, False)
    ttnn_matmul_2 = ttnn.matmul(
        ttnn_typecast_79,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_typecast_79, False)
    ttnn_reshape_18 = ttnn.reshape(
        ttnn_matmul_2,
        [1, 45303552],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_2, False)
    ttnn_typecast_80 = ttnn.typecast(
        ttnn_reshape_18,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_18, False)
    ttnn_to_layout_8 = ttnn.to_layout(
        ttnn_typecast_80,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_80, False)
    ttnn_typecast_81 = ttnn.typecast(
        ttnn_reshape_17,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_17, False)
    ttnn_to_layout_9 = ttnn.to_layout(
        ttnn_typecast_81,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_81, False)
    ttnn_embedding_2 = ttnn.embedding(ttnn_to_layout_8, ttnn_to_layout_9, layout=ttnn.Layout.TILE)
    ttnn.deallocate(ttnn_to_layout_9, False)
    ttnn.deallocate(ttnn_to_layout_8, False)
    ttnn_typecast_82 = ttnn.typecast(
        ttnn_embedding_2,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_2, False)
    ttnn_reshape_19 = ttnn.reshape(
        ttnn_typecast_82,
        [1, 256, 7, 25281],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_82, False)
    ttnn_subtract_3 = ttnn.subtract(
        ttnn_add_2,
        ttnn_floor_1,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_floor_1, False)
    ttnn.deallocate(ttnn_add_2, False)
    ttnn_multiply_6 = ttnn.multiply(
        ttnn_subtract_0,
        ttnn_subtract_3,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_subtract_0, False)
    ttnn_typecast_83 = ttnn.typecast(
        ttnn_logical_and_7,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_7, False)
    ttnn_where_15 = ttnn.where(
        ttnn_typecast_83,
        ttnn_multiply_6,
        utils_constEvalFuncWrapperZeroArg_1_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_83, False)
    ttnn.deallocate(ttnn_multiply_6, False)
    ttnn_multiply_7 = ttnn.multiply(
        ttnn_reshape_19,
        ttnn_where_15,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_15, False)
    ttnn.deallocate(ttnn_reshape_19, False)
    ttnn_add_12 = ttnn.add(
        ttnn_add_9,
        ttnn_multiply_7,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_7, False)
    ttnn.deallocate(ttnn_add_9, False)
    ttnn_logical_and_8 = ttnn.logical_and(
        ttnn_lt_5,
        ttnn_logical_and_5,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_5, False)
    ttnn.deallocate(ttnn_lt_5, False)
    ttnn_logical_and_9 = ttnn.logical_and(
        ttnn_ge_2,
        ttnn_logical_and_8,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_8, False)
    ttnn.deallocate(ttnn_ge_2, False)
    ttnn_typecast_84 = ttnn.typecast(
        ttnn_logical_and_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_85 = ttnn.typecast(
        ttnn_typecast_63,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_63, False)
    ttnn_where_16 = ttnn.where(
        ttnn_typecast_84,
        ttnn_typecast_85,
        utils_constEvalFuncWrapperZeroArg_2_10,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_85, False)
    ttnn.deallocate(ttnn_typecast_84, False)
    ttnn_typecast_86 = ttnn.typecast(
        ttnn_where_16,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_11 = ttnn.lt(
        ttnn_typecast_86,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_13 = ttnn.add(
        ttnn_typecast_86,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn_typecast_87 = ttnn.typecast(
        ttnn_lt_11,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_11, False)
    ttnn_typecast_88 = ttnn.typecast(
        ttnn_add_13,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_13, False)
    ttnn_where_17 = ttnn.where(
        ttnn_typecast_87,
        ttnn_typecast_88,
        ttnn_where_16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_88, False)
    ttnn.deallocate(ttnn_typecast_87, False)
    ttnn.deallocate(ttnn_where_16, False)
    ttnn_typecast_89 = ttnn.typecast(
        ttnn_where_17,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_17, False)
    ttnn_reshape_20 = ttnn.reshape(
        ttnn_typecast_89,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_89, False)
    ttnn_typecast_90 = ttnn.typecast(
        ttnn_logical_and_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_91 = ttnn.typecast(
        ttnn_typecast_48,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_48, False)
    ttnn_where_18 = ttnn.where(
        ttnn_typecast_90,
        ttnn_typecast_91,
        utils_constEvalFuncWrapperZeroArg_2_11,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_91, False)
    ttnn.deallocate(ttnn_typecast_90, False)
    ttnn_typecast_92 = ttnn.typecast(
        ttnn_where_18,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_lt_12 = ttnn.lt(
        ttnn_typecast_92,
        utils_constEvalFuncWrapperZeroArg_2_0,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_add_14 = ttnn.add(
        ttnn_typecast_92,
        utils_constEvalFuncWrapperZeroArg_7_0,
        dtype=ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn_typecast_93 = ttnn.typecast(
        ttnn_lt_12,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_lt_12, False)
    ttnn_typecast_94 = ttnn.typecast(
        ttnn_add_14,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_14, False)
    ttnn_where_19 = ttnn.where(
        ttnn_typecast_93,
        ttnn_typecast_94,
        ttnn_where_18,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_94, False)
    ttnn.deallocate(ttnn_typecast_93, False)
    ttnn.deallocate(ttnn_where_18, False)
    ttnn_typecast_95 = ttnn.typecast(
        ttnn_where_19,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_19, False)
    ttnn_reshape_21 = ttnn.reshape(
        ttnn_typecast_95,
        [1, 256, 7, 25281, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_95, False)
    ttnn_typecast_96 = ttnn.typecast(
        ttnn_reshape_20,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_20, False)
    ttnn_typecast_97 = ttnn.typecast(
        ttnn_reshape_21,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_21, False)
    util_create_list_12 = [
        utils_constEvalFuncWrapperZeroArg_6_3,
        utils_constEvalFuncWrapperZeroArg_2_12,
        ttnn_typecast_96,
        ttnn_typecast_97,
    ]
    ttnn_concat_3 = ttnn.concat(
        util_create_list_12,
        4,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_97, False)
    ttnn.deallocate(ttnn_typecast_96, False)
    ttnn_typecast_98 = ttnn.typecast(
        ttnn_concat_3,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_concat_3, False)
    ttnn_to_layout_10 = ttnn.to_layout(
        input_1,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(input_1, False)
    ttnn_permute_3 = ttnn.permute(
        ttnn_to_layout_10,
        [1, 2, 3, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_to_layout_10, False)
    ttnn_reshape_22 = ttnn.reshape(
        ttnn_permute_3,
        [200704, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_3, False)
    ttnn_typecast_99 = ttnn.typecast(
        ttnn_typecast_98,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_98, False)
    ttnn_matmul_3 = ttnn.matmul(
        ttnn_typecast_99,
        utils_constEvalFuncWrapperZeroArg_0_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=None,
        program_config=None,
        activation=None,
    )
    ttnn.deallocate(ttnn_typecast_99, False)
    ttnn_reshape_23 = ttnn.reshape(
        ttnn_matmul_3,
        [1, 45303552],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_3, False)
    ttnn_typecast_100 = ttnn.typecast(
        ttnn_reshape_23,
        ttnn.DataType.UINT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_23, False)
    ttnn_to_layout_11 = ttnn.to_layout(
        ttnn_typecast_100,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_100, False)
    ttnn_typecast_101 = ttnn.typecast(
        ttnn_reshape_22,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_22, False)
    ttnn_to_layout_12 = ttnn.to_layout(
        ttnn_typecast_101,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_101, False)
    ttnn_embedding_3 = ttnn.embedding(ttnn_to_layout_11, ttnn_to_layout_12, layout=ttnn.Layout.TILE)
    ttnn.deallocate(ttnn_to_layout_12, False)
    ttnn.deallocate(ttnn_to_layout_11, False)
    ttnn_typecast_102 = ttnn.typecast(
        ttnn_embedding_3,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_embedding_3, False)
    ttnn_reshape_24 = ttnn.reshape(
        ttnn_typecast_102,
        [1, 256, 7, 25281],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_102, False)
    ttnn_multiply_8 = ttnn.multiply(
        ttnn_subtract_2,
        ttnn_subtract_3,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_subtract_3, False)
    ttnn.deallocate(ttnn_subtract_2, False)
    ttnn_typecast_103 = ttnn.typecast(
        ttnn_logical_and_9,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_logical_and_9, False)
    ttnn_where_20 = ttnn.where(
        ttnn_typecast_103,
        ttnn_multiply_8,
        utils_constEvalFuncWrapperZeroArg_1_0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_103, False)
    ttnn.deallocate(ttnn_multiply_8, False)
    ttnn_multiply_9 = ttnn.multiply(
        ttnn_reshape_24,
        ttnn_where_20,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_where_20, False)
    ttnn.deallocate(ttnn_reshape_24, False)
    ttnn_add_15 = ttnn.add(
        ttnn_add_12,
        ttnn_multiply_9,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_9, False)
    ttnn.deallocate(ttnn_add_12, False)
    util_create_list_13 = [ttnn_add_15]
    return util_create_list_13


def create_inputs_for__main():
    utils_DeviceGetter_get_device_9 = utils.DeviceGetter.get_device((1, 1))
    ttnn_ones_0 = ttnn.ones(
        shape=ttnn.Shape([1, 7, 25281, 2]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_9,
    )
    ttnn_ones_1 = ttnn.ones(
        shape=ttnn.Shape([1, 256, 28, 28]),
        dtype=ttnn.DataType.FLOAT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=utils_DeviceGetter_get_device_9,
    )
    util_create_list_14 = [ttnn_ones_0, ttnn_ones_1]
    return util_create_list_14


def test_grid_sample_oft():
    create_inputs_for__main_0 = create_inputs_for__main()
    tt_output = _main(create_inputs_for__main_0)
