import ttnn
import my_get_device
import utils


def main_const_eval_0(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)

    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)

    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)

    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)

    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)

    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)

    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)

    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)

    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)

    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)

    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)

    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)

    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)

    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=3,
        out_channels=64,
        batch_size=16,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)

    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=3,
        out_channels=64,
        batch_size=16,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)

    v30 = [v28, v29]

    return v30


def main_const_eval_1(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)

    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)

    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)

    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)

    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)

    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)

    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)

    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)

    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)

    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)

    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)

    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)

    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)

    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)

    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)

    v30 = [v28, v29]

    return v30


def main_const_eval_2(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)

    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)

    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)

    v30 = [v28, v29]
    return v30


def main_const_eval_3(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_4(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_5(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_6(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=512,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=512,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_7(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=2048,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_8(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_9(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_10(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_11(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_12(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=2048,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=2048,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_13(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_14(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_15(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [6272, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=512,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [6272, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=512,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_16(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_17(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.permute(
        v15,
        [2, 3, 0, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v15, False)
    v17 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v18 = ttnn.to_layout(
        v17,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    v19 = ttnn.permute(
        v18,
        [2, 3, 0, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v18, False)
    v20 = ttnn.multiply(
        v19,
        v16,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v19, False)
    ttnn.deallocate(v16, False)
    v21 = ttnn.permute(
        v20,
        [2, 3, 0, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v20, False)
    v22 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v24 = ttnn.multiply(
        v22,
        v23,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    ttnn.deallocate(v22, False)
    v25 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v26 = ttnn.subtract(
        v25,
        v24,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v25, False)
    ttnn.deallocate(v24, False)
    v27 = ttnn.to_layout(
        v21,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v21, False)
    v28 = ttnn.from_device(v27)
    ttnn.deallocate(v27, False)
    v29 = ttnn.to_layout(
        v26,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v26, False)
    v30 = ttnn.from_device(v29)
    ttnn.deallocate(v29, False)
    v31 = ttnn.prepare_conv_weights(
        weight_tensor=v28,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [448, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=2048,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v28, False)
    v32 = ttnn.prepare_conv_bias(
        bias_tensor=v30,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [448, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=2048,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v30, False)
    v33 = [v31, v32]
    return v33


def main_const_eval_18(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_19(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [448, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [448, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_20(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_21(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=128,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=128,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_22(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_23(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_24(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_25(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_26(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_27(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=1024,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=1024,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_28(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_29(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_30(v1):
    v2 = v1[0]
    v3 = ttnn.reshape(
        v2, [1, 1000], memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v4 = ttnn.repeat(v3, ttnn.Shape([16, 1]))
    ttnn.deallocate(v3, False)
    v5 = [v4]
    return v5


def main_const_eval_31(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_32(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_33(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_34(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_35(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_36(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 64, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [64, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 64],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_37(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_38(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_39(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_40(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_41(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_42(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_43(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_44(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=256,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=256,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_45(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_46(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_47(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_48(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 1024, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [1024, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 1024],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_49(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 512, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [512, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 512],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_50(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_51(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 2048, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [2048, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_52(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 256, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [256, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


def main_const_eval_53(v1):
    v2, v3, v4, v5, v6 = v1

    v7 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v7.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
    )

    v8 = ttnn.reshape(
        v2,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v9 = ttnn.reshape(
        v5,
        [1, 128, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v10 = ttnn.full(
        shape=ttnn.Shape([1]),
        fill_value=9.9999997473787516e-06,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v7,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v11 = ttnn.reshape(
        v10,
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v10, False)
    v12 = ttnn.add(
        v8,
        v11,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v11, False)
    ttnn.deallocate(v8, False)
    v13 = ttnn.sqrt(
        v12, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v12, False)
    v14 = ttnn.divide(
        v9,
        v13,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v13, False)
    ttnn.deallocate(v9, False)
    v15 = ttnn.reshape(
        v14,
        [128, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v16 = ttnn.to_device(
        v6, device=v7, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    v17 = ttnn.to_layout(
        v16,
        ttnn.Layout.TILE,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v16, False)
    v18 = ttnn.multiply(
        v17,
        v15,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v17, False)
    ttnn.deallocate(v15, False)
    v19 = ttnn.reshape(
        v3,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v20 = ttnn.reshape(
        v14,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v14, False)
    v21 = ttnn.multiply(
        v19,
        v20,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v20, False)
    ttnn.deallocate(v19, False)
    v22 = ttnn.reshape(
        v4,
        [1, 1, 1, 128],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    v23 = ttnn.subtract(
        v22,
        v21,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v22, False)
    ttnn.deallocate(v21, False)
    v24 = ttnn.to_layout(
        v18,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v18, False)
    v25 = ttnn.from_device(v24)
    ttnn.deallocate(v24, False)
    v26 = ttnn.to_layout(
        v23,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v23, False)
    v27 = ttnn.from_device(v26)
    ttnn.deallocate(v26, False)
    v28 = ttnn.prepare_conv_weights(
        weight_tensor=v25,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [896, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        weights_format="OIHW",
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        has_bias=True,
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=None,
    )
    ttnn.deallocate(v25, False)
    v29 = ttnn.prepare_conv_bias(
        bias_tensor=v27,
        input_memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [896, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
        input_layout=ttnn.Layout.ROW_MAJOR,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        device=v7,
        input_dtype=ttnn.DataType.BFLOAT8_B,
        output_dtype=ttnn.DataType.BFLOAT8_B,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
    )
    ttnn.deallocate(v27, False)
    v30 = [v28, v29]
    return v30


from collections import OrderedDict

CONST_EVAL_CACHE = OrderedDict()


def main_const_eval(inputs):
    if not CONST_EVAL_CACHE:
        for i in range(54):
            CONST_EVAL_CACHE[f"main_const_eval_{i}"] = eval(f"main_const_eval_{i}")(inputs[i])
    return list(CONST_EVAL_CACHE.values())


def _main(v1):
    v2 = v1[0]
    v3 = v1[1]
    v4 = v1[2]
    v5 = v1[3]
    v6 = v1[4]
    v7 = v1[5]
    v8 = v1[6]
    v9 = v1[7]
    v10 = v1[8]
    v11 = v1[9]
    v12 = v1[10]
    v13 = v1[11]
    v14 = v1[12]
    v15 = v1[13]
    v16 = v1[14]
    v17 = v1[15]
    v18 = v1[16]
    v19 = v1[17]
    v20 = v1[18]
    v21 = v1[19]
    v22 = v1[20]
    v23 = v1[21]
    v24 = v1[22]
    v25 = v1[23]
    v26 = v1[24]
    v27 = v1[25]
    v28 = v1[26]
    v29 = v1[27]
    v30 = v1[28]
    v31 = v1[29]
    v32 = v1[30]
    v33 = v1[31]
    v34 = v1[32]
    v35 = v1[33]
    v36 = v1[34]
    v37 = v1[35]
    v38 = v1[36]
    v39 = v1[37]
    v40 = v1[38]
    v41 = v1[39]
    v42 = v1[40]
    v43 = v1[41]
    v44 = v1[42]
    v45 = v1[43]
    v46 = v1[44]
    v47 = v1[45]
    v48 = v1[46]
    v49 = v1[47]
    v50 = v1[48]
    v51 = v1[49]
    v52 = v1[50]
    v53 = v1[51]
    v54 = v1[52]
    v55 = v1[53]
    v56 = v1[54]
    v57 = v1[55]
    v58 = v1[56]
    v59 = v1[57]
    v60 = v1[58]
    v61 = v1[59]
    v62 = v1[60]
    v63 = v1[61]
    v64 = v1[62]
    v65 = v1[63]
    v66 = v1[64]
    v67 = v1[65]
    v68 = v1[66]
    v69 = v1[67]
    v70 = v1[68]
    v71 = v1[69]
    v72 = v1[70]
    v73 = v1[71]
    v74 = v1[72]
    v75 = v1[73]
    v76 = v1[74]
    v77 = v1[75]
    v78 = v1[76]
    v79 = v1[77]
    v80 = v1[78]
    v81 = v1[79]
    v82 = v1[80]
    v83 = v1[81]
    v84 = v1[82]
    v85 = v1[83]
    v86 = v1[84]
    v87 = v1[85]
    v88 = v1[86]
    v89 = v1[87]
    v90 = v1[88]
    v91 = v1[89]
    v92 = v1[90]
    v93 = v1[91]
    v94 = v1[92]
    v95 = v1[93]
    v96 = v1[94]
    v97 = v1[95]
    v98 = v1[96]
    v99 = v1[97]
    v100 = v1[98]
    v101 = v1[99]
    v102 = v1[100]
    v103 = v1[101]
    v104 = v1[102]
    v105 = v1[103]
    v106 = v1[104]
    v107 = v1[105]
    v108 = v1[106]
    v109 = v1[107]
    v110 = v1[108]
    v111 = v1[109]
    v112 = v1[110]
    v113 = v1[111]
    v114 = v1[112]
    v115 = v1[113]
    v116 = v1[114]
    v117 = v1[115]
    v118 = v1[116]
    v119 = v1[117]
    v120 = v1[118]
    v121 = v1[119]
    v122 = v1[120]
    v123 = v1[121]
    v124 = v1[122]
    v125 = v1[123]
    v126 = v1[124]
    v127 = v1[125]
    v128 = v1[126]
    v129 = v1[127]
    v130 = v1[128]
    v131 = v1[129]
    v132 = v1[130]
    v133 = v1[131]
    v134 = v1[132]
    v135 = v1[133]
    v136 = v1[134]
    v137 = v1[135]
    v138 = v1[136]
    v139 = v1[137]
    v140 = v1[138]
    v141 = v1[139]
    v142 = v1[140]
    v143 = v1[141]
    v144 = v1[142]
    v145 = v1[143]
    v146 = v1[144]
    v147 = v1[145]
    v148 = v1[146]
    v149 = v1[147]
    v150 = v1[148]
    v151 = v1[149]
    v152 = v1[150]
    v153 = v1[151]
    v154 = v1[152]
    v155 = v1[153]
    v156 = v1[154]
    v157 = v1[155]
    v158 = v1[156]
    v159 = v1[157]
    v160 = v1[158]
    v161 = v1[159]
    v162 = v1[160]
    v163 = v1[161]
    v164 = v1[162]
    v165 = v1[163]
    v166 = v1[164]
    v167 = v1[165]
    v168 = v1[166]
    v169 = v1[167]
    v170 = v1[168]
    v171 = v1[169]
    v172 = v1[170]
    v173 = v1[171]
    v174 = v1[172]
    v175 = v1[173]
    v176 = v1[174]
    v177 = v1[175]
    v178 = v1[176]
    v179 = v1[177]
    v180 = v1[178]
    v181 = v1[179]
    v182 = v1[180]
    v183 = v1[181]
    v184 = v1[182]
    v185 = v1[183]
    v186 = v1[184]
    v187 = v1[185]
    v188 = v1[186]
    v189 = v1[187]
    v190 = v1[188]
    v191 = v1[189]
    v192 = v1[190]
    v193 = v1[191]
    v194 = v1[192]
    v195 = v1[193]
    v196 = v1[194]
    v197 = v1[195]
    v198 = v1[196]
    v199 = v1[197]
    v200 = v1[198]
    v201 = v1[199]
    v202 = v1[200]
    v203 = v1[201]
    v204 = v1[202]
    v205 = v1[203]
    v206 = v1[204]
    v207 = v1[205]
    v208 = v1[206]
    v209 = v1[207]
    v210 = v1[208]
    v211 = v1[209]
    v212 = v1[210]
    v213 = v1[211]
    v214 = v1[212]
    v215 = v1[213]
    v216 = v1[214]
    v217 = v1[215]
    v218 = v1[216]
    v219 = v1[217]
    v220 = v1[218]
    v221 = v1[219]
    v222 = v1[220]
    v223 = v1[221]
    v224 = v1[222]
    v225 = v1[223]
    v226 = v1[224]
    v227 = v1[225]
    v228 = v1[226]
    v229 = v1[227]
    v230 = v1[228]
    v231 = v1[229]
    v232 = v1[230]
    v233 = v1[231]
    v234 = v1[232]
    v235 = v1[233]
    v236 = v1[234]
    v237 = v1[235]
    v238 = v1[236]
    v239 = v1[237]
    v240 = v1[238]
    v241 = v1[239]
    v242 = v1[240]
    v243 = v1[241]
    v244 = v1[242]
    v245 = v1[243]
    v246 = v1[244]
    v247 = v1[245]
    v248 = v1[246]
    v249 = v1[247]
    v250 = v1[248]
    v251 = v1[249]
    v252 = v1[250]
    v253 = v1[251]
    v254 = v1[252]
    v255 = v1[253]
    v256 = v1[254]
    v257 = v1[255]
    v258 = v1[256]
    v259 = v1[257]
    v260 = v1[258]
    v261 = v1[259]
    v262 = v1[260]
    v263 = v1[261]
    v264 = v1[262]
    v265 = v1[263]
    v266 = v1[264]
    v267 = v1[265]
    v268 = v1[266]
    v269 = v1[267]

    inputs = [
        # ---------------------------------------------------------- #
        # * * * Const Eval Input Tensors * * *                       #
        # ---------------------------------------------------------- #
        # Const Eval 0
        [v24, v25, v26, v27, v28],
        # Cons Eval 1
        [v150, v151, v152, v153, v154],
        # Cons Eval 2
        [v105, v106, v107, v108, v109],
        # Cons Eval 3
        [v135, v136, v137, v138, v139],
        # Cons Eval 4
        [v70, v71, v72, v73, v74],
        # Cons Eval 5
        [v240, v241, v242, v243, v244],
        # Cons Eval 6
        [v235, v236, v237, v238, v239],
        # Cons Eval 7
        [v265, v266, v267, v268, v269],
        # Cons Eval 8
        [v190, v191, v192, v193, v194],
        # Cons Eval 9
        [v120, v121, v122, v123, v124],
        # Cons Eval 10
        [v55, v56, v57, v58, v59],
        # Cons Eval 11
        [v50, v51, v52, v53, v54],
        # Cons Eval 12
        [v250, v251, v252, v253, v254],
        # Cons Eval 13
        [v255, v256, v257, v258, v259],
        # Cons Eval 14
        [v19, v20, v21, v22, v23],
        # Cons Eval 15
        [v14, v15, v16, v17, v18],
        # Cons Eval 16
        [v45, v46, v47, v48, v49],
        # Cons Eval 17
        [v4, v5, v6, v7, v8],
        # Cons Eval 18
        [v165, v166, v167, v168, v169],
        # Cons Eval 19
        [v230, v231, v232, v233, v234],
        # Cons Eval 20
        [v35, v36, v37, v38, v39],
        # Cons Eval 21
        [v85, v86, v87, v88, v89],
        # Cons Eval 22
        [v125, v126, v127, v128, v129],
        # Cons Eval 23
        [v160, v161, v162, v163, v164],
        # Cons Eval 24,
        [v100, v101, v102, v103, v104],
        # Cons Eval 25
        [v90, v91, v92, v93, v94],
        # Cons Eval 26
        [v195, v196, v197, v198, v199],
        # Cons Eval 27
        [v9, v10, v11, v12, v13],
        # Cons Eval 28
        [v95, v96, v97, v98, v99],
        # Cons Eval 29
        [v185, v186, v187, v188, v189],
        # Cons Eval 30
        [v2],
        # Cons Eval 31
        [v155, v156, v157, v158, v159],
        # Cons Eval 32
        [v215, v216, v217, v218, v219],
        # Cons Eval 33
        [v130, v131, v132, v133, v134],
        # Cons Eval 34
        [v65, v66, v67, v68, v69],
        # Cons Eval 35
        [v115, v116, v117, v118, v119],
        # Cons Eval 36
        [v40, v41, v42, v43, v44],
        # Cons Eval 37
        [v30, v31, v32, v33, v34],
        # Cons Eval 38
        [v140, v141, v142, v143, v144],
        # Cons Eval 39
        [v260, v261, v262, v263, v264],
        # Cons Eval 40
        [v220, v221, v222, v223, v224],
        # Cons Eval 41
        [v110, v111, v112, v113, v114],
        # Cons Eval 42
        [v170, v171, v172, v173, v174],
        # Cons Eval 43
        [v180, v181, v182, v183, v184],
        # Cons Eval 44
        [v145, v146, v147, v148, v149],
        # Cons Eval 45
        [v75, v76, v77, v78, v79],
        # Cons Eval 46
        [v60, v61, v62, v63, v64],
        # Cons Eval 47
        [v205, v206, v207, v208, v209],
        # Cons Eval 48
        [v210, v211, v212, v213, v214],
        # Cons Eval 49
        [v245, v246, v247, v248, v249],
        # Cons Eval 50,
        [v175, v176, v177, v178, v179],
        # Cons Eval 51,
        [v225, v226, v227, v228, v229],
        # Cons Eval 52,
        [v200, v201, v202, v203, v204],
        # Cons Eval 53,
        [v80, v81, v82, v83, v84],
    ]

    outputs = main_const_eval(inputs)

    v272, v273 = outputs[0]
    v276, v277 = outputs[1]
    v280, v281 = outputs[2]
    v284, v285 = outputs[3]
    v288, v289 = outputs[4]
    v292, v293 = outputs[5]
    v296, v297 = outputs[6]
    v300, v301 = outputs[7]
    v304, v305 = outputs[8]
    v308, v309 = outputs[9]
    v312, v313 = outputs[10]
    v316, v317 = outputs[11]
    v320, v321 = outputs[12]
    v324, v325 = outputs[13]
    v328, v329 = outputs[14]
    v332, v333 = outputs[15]
    v336, v337 = outputs[16]
    v340, v341 = outputs[17]
    v344, v345 = outputs[18]
    v348, v349 = outputs[19]
    v352, v353 = outputs[20]
    v356, v357 = outputs[21]
    v360, v361 = outputs[22]
    v364, v365 = outputs[23]
    v368, v369 = outputs[24]
    v372, v373 = outputs[25]
    v376, v377 = outputs[26]
    v380, v381 = outputs[27]
    v384, v385 = outputs[28]
    v388, v389 = outputs[29]
    (v392,) = outputs[30]
    v395, v396 = outputs[31]
    v399, v400 = outputs[32]
    v403, v404 = outputs[33]
    v407, v408 = outputs[34]
    v411, v412 = outputs[35]
    v415, v416 = outputs[36]
    v419, v420 = outputs[37]
    v423, v424 = outputs[38]
    v427, v428 = outputs[39]
    v431, v432 = outputs[40]
    v435, v436 = outputs[41]
    v439, v440 = outputs[42]
    v443, v444 = outputs[43]
    v447, v448 = outputs[44]
    v451, v452 = outputs[45]
    v455, v456 = outputs[46]
    v459, v460 = outputs[47]
    v463, v464 = outputs[48]
    v467, v468 = outputs[49]
    v471, v472 = outputs[50]
    v475, v476 = outputs[51]
    v479, v480 = outputs[52]
    v483, v484 = outputs[53]

    v485 = my_get_device.DeviceGetter.get_device()

    compute_config = ttnn.init_device_compute_kernel_config(
        v485.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        packer_l1_acc=True,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
    )

    # ---------------------------------------------------------- #
    # * * * Full * * *                                           #
    # ---------------------------------------------------------- #

    v486 = ttnn.full(
        shape=ttnn.Shape([16, 2048]),
        fill_value=0.0203857421875,
        dtype=ttnn.DataType.BFLOAT8_B,
        layout=ttnn.Layout.TILE,
        device=v485,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )

    # ---------------------------------------------------------- #
    # * * * Permute * * *                                        #
    # ---------------------------------------------------------- #

    v487 = ttnn.permute(
        v29,
        [0, 2, 3, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    v488 = ttnn.reshape(
        v487,
        [1, 1, 802816, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v487, False)

    # ---------------------------------------------------------- #
    # * * * To Layout * * *                                      #
    # ---------------------------------------------------------- #

    v489 = ttnn.to_layout(
        v488,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v488, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 1 * * *                                  #
    # ---------------------------------------------------------- #

    v490 = ttnn.conv2d(
        input_tensor=v489,
        weight_tensor=v272,
        device=v485,
        in_channels=3,
        out_channels=64,
        batch_size=16,
        input_height=224,
        input_width=224,
        kernel_size=[7, 7],
        stride=[2, 2],
        padding=[3, 3, 3, 3],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v273,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v489, False)
    # ttnn.deallocate(v273, False)
    # ttnn.deallocate(v272, False)

    # ---------------------------------------------------------- #
    # * * * Max Pool * * *                                       #
    # ---------------------------------------------------------- #

    v491 = ttnn.max_pool2d(
        v490,
        16,
        112,
        112,
        64,
        [3, 3],
        [2, 2],
        [1, 1],
        [1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        applied_shard_scheme=None,
        ceil_mode=False,
        in_place_halo=False,
    )
    ttnn.deallocate(v490, False)

    # ---------------------------------------------------------- #
    # * * * To Layout * * *                                      #
    # ---------------------------------------------------------- #

    v492 = ttnn.to_layout(
        v491,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v491, False)

    v493 = ttnn.to_memory_config(
        v492,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 2 * * *                                  #
    # ---------------------------------------------------------- #

    v494 = ttnn.conv2d(
        input_tensor=v493,
        weight_tensor=v328,
        device=v485,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v329,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v493, False)
    # ttnn.deallocate(v329, False)
    # ttnn.deallocate(v328, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 3 * * *                                  #
    # ---------------------------------------------------------- #

    v495 = ttnn.to_memory_config(
        v494, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v494, False)

    v496 = ttnn.to_memory_config(
        v492,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v492, False)

    v497 = ttnn.conv2d(
        input_tensor=v496,
        weight_tensor=v415,
        device=v485,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v416,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v496, False)
    # ttnn.deallocate(v416, False)
    # ttnn.deallocate(v415, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 4 * * *                                  #
    # ---------------------------------------------------------- #

    v498 = ttnn.conv2d(
        input_tensor=v497,
        weight_tensor=v352,
        device=v485,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v353,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v497, False)
    # ttnn.deallocate(v353, False)
    # ttnn.deallocate(v352, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 5 * * *                                  #
    # ---------------------------------------------------------- #

    v499 = ttnn.conv2d(
        input_tensor=v498,
        weight_tensor=v419,
        device=v485,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v420,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v498, False)
    # ttnn.deallocate(v420, False)
    # ttnn.deallocate(v419, False)

    v500 = ttnn.add(
        v499,
        v495,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v499, False)
    ttnn.deallocate(v495, False)

    v501 = ttnn.relu(
        v500,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v500, False)

    v502 = ttnn.to_memory_config(
        v501, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 6 * * *                                  #
    # ---------------------------------------------------------- #

    v503 = ttnn.conv2d(
        input_tensor=v501,
        weight_tensor=v312,
        device=v485,
        in_channels=256,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v313,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v501, False)
    # ttnn.deallocate(v313, False)
    # ttnn.deallocate(v312, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 7 * * *                                  #
    # ---------------------------------------------------------- #

    v504 = ttnn.conv2d(
        input_tensor=v503,
        weight_tensor=v316,
        device=v485,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v317,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v503, False)
    # ttnn.deallocate(v317, False)
    # ttnn.deallocate(v316, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 8 * * *                                  #
    # ---------------------------------------------------------- #

    v505 = ttnn.conv2d(
        input_tensor=v504,
        weight_tensor=v336,
        device=v485,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v337,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v504, False)
    # ttnn.deallocate(v337, False)
    # ttnn.deallocate(v336, False)

    # ---------------------------------------------------------- #
    # * * *                                  #
    # ---------------------------------------------------------- #

    v506 = ttnn.add(
        v505,
        v502,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v505, False)
    ttnn.deallocate(v502, False)

    v507 = ttnn.relu(
        v506,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v506, False)

    v508 = ttnn.to_memory_config(
        v507, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 9 * * *                                  #
    # ---------------------------------------------------------- #

    v509 = ttnn.conv2d(
        input_tensor=v507,
        weight_tensor=v288,
        device=v485,
        in_channels=256,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v289,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v507, False)
    # ttnn.deallocate(v289, False)
    # ttnn.deallocate(v288, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 10 * * *                                 #
    # ---------------------------------------------------------- #

    v510 = ttnn.conv2d(
        input_tensor=v509,
        weight_tensor=v407,
        device=v485,
        in_channels=64,
        out_channels=64,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v408,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v509, False)
    # ttnn.deallocate(v408, False)
    # ttnn.deallocate(v407, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 11 * * *                                 #
    # ---------------------------------------------------------- #

    v511 = ttnn.conv2d(
        input_tensor=v510,
        weight_tensor=v455,
        device=v485,
        in_channels=64,
        out_channels=256,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v456,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v510, False)
    # ttnn.deallocate(v456, False)
    # ttnn.deallocate(v455, False)

    v512 = ttnn.add(
        v511,
        v508,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v511, False)
    ttnn.deallocate(v508, False)

    v513 = ttnn.relu(
        v512,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v512, False)

    v514 = ttnn.to_memory_config(
        v513,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [6272, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 12 * * *                                 #
    # ---------------------------------------------------------- #

    v515 = ttnn.conv2d(
        input_tensor=v514,
        weight_tensor=v332,
        device=v485,
        in_channels=256,
        out_channels=512,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v333,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v514, False)
    # ttnn.deallocate(v333, False)
    # ttnn.deallocate(v332, False)

    v516 = ttnn.to_memory_config(
        v515, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v515, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 13 * * *                                 #
    # ---------------------------------------------------------- #

    v517 = ttnn.conv2d(
        input_tensor=v513,
        weight_tensor=v356,
        device=v485,
        in_channels=256,
        out_channels=128,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v357,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6)),
                        ttnn.CoreRange(ttnn.CoreCoord(0, 7), ttnn.CoreCoord(6, 7)),
                    ]
                ),
                [800, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v513, False)
    # ttnn.deallocate(v357, False)
    # ttnn.deallocate(v356, False)

    v518 = ttnn.to_memory_config(
        v517,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [896, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v517, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 14 * * *                                 #
    # ---------------------------------------------------------- #

    v519 = ttnn.conv2d(
        input_tensor=v518,
        weight_tensor=v483,
        device=v485,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=56,
        input_width=56,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v484,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v518, False)
    # ttnn.deallocate(v484, False)
    # ttnn.deallocate(v483, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 15 * * *                                 #
    # ---------------------------------------------------------- #

    v520 = ttnn.conv2d(
        input_tensor=v519,
        weight_tensor=v451,
        device=v485,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v452,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v519, False)
    # ttnn.deallocate(v452, False)
    # ttnn.deallocate(v451, False)

    v521 = ttnn.add(
        v520,
        v516,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v520, False)
    ttnn.deallocate(v516, False)
    v522 = ttnn.relu(
        v521,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v521, False)
    v523 = ttnn.to_memory_config(
        v522, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 16 * * *                                 #
    # ---------------------------------------------------------- #

    v524 = ttnn.conv2d(
        input_tensor=v522,
        weight_tensor=v368,
        device=v485,
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v369,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v522, False)
    # ttnn.deallocate(v369, False)
    # ttnn.deallocate(v368, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 17 * * *                                 #
    # ---------------------------------------------------------- #

    v525 = ttnn.conv2d(
        input_tensor=v524,
        weight_tensor=v384,
        device=v485,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v385,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v524, False)
    # ttnn.deallocate(v385, False)
    # ttnn.deallocate(v384, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 18 * * *                                 #
    # ---------------------------------------------------------- #

    v526 = ttnn.conv2d(
        input_tensor=v525,
        weight_tensor=v372,
        device=v485,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v373,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v525, False)
    # ttnn.deallocate(v373, False)
    # ttnn.deallocate(v372, False)

    v527 = ttnn.add(
        v526,
        v523,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v526, False)
    ttnn.deallocate(v523, False)
    v528 = ttnn.relu(
        v527,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v527, False)
    v529 = ttnn.to_memory_config(
        v528, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 19 * * *                                 #
    # ---------------------------------------------------------- #

    v530 = ttnn.conv2d(
        input_tensor=v528,
        weight_tensor=v411,
        device=v485,
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v412,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v528, False)
    # ttnn.deallocate(v412, False)
    # ttnn.deallocate(v411, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 20 * * *                                 #
    # ---------------------------------------------------------- #

    v531 = ttnn.conv2d(
        input_tensor=v530,
        weight_tensor=v435,
        device=v485,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v436,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v530, False)
    # ttnn.deallocate(v436, False)
    # ttnn.deallocate(v435, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 21 * * *                                 #
    # ---------------------------------------------------------- #

    v532 = ttnn.conv2d(
        input_tensor=v531,
        weight_tensor=v280,
        device=v485,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v281,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v531, False)
    # ttnn.deallocate(v281, False)
    # ttnn.deallocate(v280, False)

    # ---------------------------------------------------------- #
    # * * *  * * *                                 #
    # ---------------------------------------------------------- #

    v533 = ttnn.add(
        v532,
        v529,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v532, False)
    ttnn.deallocate(v529, False)
    v534 = ttnn.relu(
        v533,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v533, False)
    v535 = ttnn.to_memory_config(
        v534, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 22 * * *                                 #
    # ---------------------------------------------------------- #

    v536 = ttnn.conv2d(
        input_tensor=v534,
        weight_tensor=v403,
        device=v485,
        in_channels=512,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v404,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v534, False)
    # ttnn.deallocate(v404, False)
    # ttnn.deallocate(v403, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 23 * * *                                 #
    # ---------------------------------------------------------- #

    v537 = ttnn.conv2d(
        input_tensor=v536,
        weight_tensor=v360,
        device=v485,
        in_channels=128,
        out_channels=128,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v361,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v536, False)
    # ttnn.deallocate(v361, False)
    # ttnn.deallocate(v360, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 24 * * *                                 #
    # ---------------------------------------------------------- #

    v538 = ttnn.conv2d(
        input_tensor=v537,
        weight_tensor=v308,
        device=v485,
        in_channels=128,
        out_channels=512,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v309,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v537, False)
    # ttnn.deallocate(v309, False)
    # ttnn.deallocate(v308, False)

    # ---------------------------------------------------------- #
    # * * *   * * *                                 #
    # ---------------------------------------------------------- #

    v539 = ttnn.add(
        v538,
        v535,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v538, False)
    ttnn.deallocate(v535, False)
    v540 = ttnn.relu(
        v539,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [224, 512],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v539, False)
    v541 = ttnn.to_memory_config(
        v540,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 25 * * *                                 #
    # ---------------------------------------------------------- #

    v542 = ttnn.conv2d(
        input_tensor=v541,
        weight_tensor=v380,
        device=v485,
        in_channels=512,
        out_channels=1024,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v381,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v541, False)
    # ttnn.deallocate(v381, False)
    # ttnn.deallocate(v380, False)

    v543 = ttnn.to_memory_config(
        v542, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v542, False)
    v544 = ttnn.to_memory_config(
        v540,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v540, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 26 * * *                                 #
    # ---------------------------------------------------------- #

    v545 = ttnn.conv2d(
        input_tensor=v544,
        weight_tensor=v447,
        device=v485,
        in_channels=512,
        out_channels=256,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v448,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [1568, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v544, False)
    # ttnn.deallocate(v448, False)
    # ttnn.deallocate(v447, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 27 * * *                                 #
    # ---------------------------------------------------------- #

    v546 = ttnn.conv2d(
        input_tensor=v545,
        weight_tensor=v423,
        device=v485,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=28,
        input_width=28,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v424,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v545, False)
    # ttnn.deallocate(v424, False)
    # ttnn.deallocate(v423, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 28 * * *                                 #
    # ---------------------------------------------------------- #

    v547 = ttnn.conv2d(
        input_tensor=v546,
        weight_tensor=v284,
        device=v485,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v285,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v546, False)
    # ttnn.deallocate(v285, False)
    # ttnn.deallocate(v284, False)

    v548 = ttnn.add(
        v547,
        v543,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v547, False)
    ttnn.deallocate(v543, False)
    v549 = ttnn.relu(
        v548,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v548, False)
    v550 = ttnn.to_memory_config(
        v549, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 29 * * *                                 #
    # ---------------------------------------------------------- #

    v551 = ttnn.conv2d(
        input_tensor=v549,
        weight_tensor=v364,
        device=v485,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v365,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v549, False)
    # ttnn.deallocate(v365, False)
    # ttnn.deallocate(v364, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 30 * * *                                 #
    # ---------------------------------------------------------- #

    v552 = ttnn.conv2d(
        input_tensor=v551,
        weight_tensor=v395,
        device=v485,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v396,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v551, False)
    # ttnn.deallocate(v396, False)
    # ttnn.deallocate(v395, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 31 * * *                                 #
    # ---------------------------------------------------------- #

    v553 = ttnn.conv2d(
        input_tensor=v552,
        weight_tensor=v276,
        device=v485,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v277,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v552, False)
    # ttnn.deallocate(v277, False)
    # ttnn.deallocate(v276, False)

    v554 = ttnn.add(
        v553,
        v550,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v553, False)
    ttnn.deallocate(v550, False)
    v555 = ttnn.relu(
        v554,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v554, False)
    v556 = ttnn.to_memory_config(
        v555, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 32 * * *                                 #
    # ---------------------------------------------------------- #

    v557 = ttnn.conv2d(
        input_tensor=v555,
        weight_tensor=v471,
        device=v485,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v472,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v555, False)
    # ttnn.deallocate(v472, False)
    # ttnn.deallocate(v471, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 33 * * *                                 #
    # ---------------------------------------------------------- #

    v558 = ttnn.conv2d(
        input_tensor=v557,
        weight_tensor=v439,
        device=v485,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v440,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v557, False)
    # ttnn.deallocate(v440, False)
    # ttnn.deallocate(v439, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 34 * * *                                 #
    # ---------------------------------------------------------- #

    v559 = ttnn.conv2d(
        input_tensor=v558,
        weight_tensor=v344,
        device=v485,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v345,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v558, False)
    # ttnn.deallocate(v345, False)
    # ttnn.deallocate(v344, False)

    v560 = ttnn.add(
        v559,
        v556,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v559, False)
    ttnn.deallocate(v556, False)
    v561 = ttnn.relu(
        v560,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v560, False)
    v562 = ttnn.to_memory_config(
        v561, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 35 * * *                                 #
    # ---------------------------------------------------------- #

    v563 = ttnn.conv2d(
        input_tensor=v561,
        weight_tensor=v304,
        device=v485,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v305,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v561, False)
    # ttnn.deallocate(v305, False)
    # ttnn.deallocate(v304, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 36 * * *                                 #
    # ---------------------------------------------------------- #

    v564 = ttnn.conv2d(
        input_tensor=v563,
        weight_tensor=v388,
        device=v485,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v389,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v563, False)
    # ttnn.deallocate(v389, False)
    # ttnn.deallocate(v388, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 37 * * *                                 #
    # ---------------------------------------------------------- #

    v565 = ttnn.conv2d(
        input_tensor=v564,
        weight_tensor=v443,
        device=v485,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v444,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v564, False)
    # ttnn.deallocate(v444, False)
    # ttnn.deallocate(v443, False)

    v566 = ttnn.add(
        v565,
        v562,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v565, False)
    ttnn.deallocate(v562, False)
    v567 = ttnn.relu(
        v566,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v566, False)
    v568 = ttnn.to_memory_config(
        v567, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 38 * * *                                 #
    # ---------------------------------------------------------- #

    v569 = ttnn.conv2d(
        input_tensor=v567,
        weight_tensor=v459,
        device=v485,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v460,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v567, False)
    # ttnn.deallocate(v460, False)
    # ttnn.deallocate(v459, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 39 * * *                                 #
    # ---------------------------------------------------------- #

    v570 = ttnn.conv2d(
        input_tensor=v569,
        weight_tensor=v479,
        device=v485,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v480,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v569, False)
    # ttnn.deallocate(v480, False)
    # ttnn.deallocate(v479, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 40 * * *                                 #
    # ---------------------------------------------------------- #

    v571 = ttnn.conv2d(
        input_tensor=v570,
        weight_tensor=v376,
        device=v485,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v377,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v570, False)
    # ttnn.deallocate(v377, False)
    # ttnn.deallocate(v376, False)

    v572 = ttnn.add(
        v571,
        v568,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v571, False)
    ttnn.deallocate(v568, False)
    v573 = ttnn.relu(
        v572,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v572, False)
    v574 = ttnn.to_memory_config(
        v573, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 41 * * *                                 #
    # ---------------------------------------------------------- #

    v575 = ttnn.conv2d(
        input_tensor=v573,
        weight_tensor=v431,
        device=v485,
        in_channels=1024,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v432,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v573, False)
    # ttnn.deallocate(v432, False)
    # ttnn.deallocate(v431, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 42 * * *                                 #
    # ---------------------------------------------------------- #

    v576 = ttnn.conv2d(
        input_tensor=v575,
        weight_tensor=v399,
        device=v485,
        in_channels=256,
        out_channels=256,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v400,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v575, False)
    # ttnn.deallocate(v400, False)
    # ttnn.deallocate(v399, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 43 * * *                                 #
    # ---------------------------------------------------------- #

    v577 = ttnn.conv2d(
        input_tensor=v576,
        weight_tensor=v463,
        device=v485,
        in_channels=256,
        out_channels=1024,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v464,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v576, False)
    # ttnn.deallocate(v464, False)
    # ttnn.deallocate(v463, False)

    v578 = ttnn.add(
        v577,
        v574,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v577, False)
    ttnn.deallocate(v574, False)
    v579 = ttnn.relu(
        v578,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v578, False)
    v580 = ttnn.to_memory_config(
        v579,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [448, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 44 * * *                                 #
    # ---------------------------------------------------------- #

    v581 = ttnn.conv2d(
        input_tensor=v580,
        weight_tensor=v340,
        device=v485,
        in_channels=1024,
        out_channels=2048,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[2, 2],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v341,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v580, False)
    # ttnn.deallocate(v341, False)
    # ttnn.deallocate(v340, False)

    v582 = ttnn.to_memory_config(
        v581, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v581, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 45 * * *                                 #
    # ---------------------------------------------------------- #

    v583 = ttnn.conv2d(
        input_tensor=v579,
        weight_tensor=v296,
        device=v485,
        in_channels=1024,
        out_channels=512,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v297,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [416, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v579, False)
    # ttnn.deallocate(v297, False)
    # ttnn.deallocate(v296, False)

    v584 = ttnn.to_memory_config(
        v583,
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [448, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v583, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 46 * * *                                 #
    # ---------------------------------------------------------- #

    v585 = ttnn.conv2d(
        input_tensor=v584,
        weight_tensor=v348,
        device=v485,
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=14,
        input_width=14,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v349,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v584, False)
    # ttnn.deallocate(v349, False)
    # ttnn.deallocate(v348, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 47 * * *                                 #
    # ---------------------------------------------------------- #

    v586 = ttnn.conv2d(
        input_tensor=v585,
        weight_tensor=v475,
        device=v485,
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v476,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v585, False)
    # ttnn.deallocate(v476, False)
    # ttnn.deallocate(v475, False)

    v587 = ttnn.add(
        v586,
        v582,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v586, False)
    ttnn.deallocate(v582, False)
    v588 = ttnn.relu(
        v587,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v587, False)
    v589 = ttnn.to_memory_config(
        v588, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 48 * * *                                 #
    # ---------------------------------------------------------- #

    v590 = ttnn.conv2d(
        input_tensor=v588,
        weight_tensor=v320,
        device=v485,
        in_channels=2048,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v321,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v588, False)
    # ttnn.deallocate(v321, False)
    # ttnn.deallocate(v320, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 49 * * *                                 #
    # ---------------------------------------------------------- #

    v591 = ttnn.conv2d(
        input_tensor=v590,
        weight_tensor=v467,
        device=v485,
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v468,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v590, False)
    # ttnn.deallocate(v468, False)
    # ttnn.deallocate(v467, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 50 * * *                                 #
    # ---------------------------------------------------------- #

    v592 = ttnn.conv2d(
        input_tensor=v591,
        weight_tensor=v292,
        device=v485,
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v293,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v591, False)
    # ttnn.deallocate(v293, False)
    # ttnn.deallocate(v292, False)

    v593 = ttnn.add(
        v592,
        v589,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v592, False)
    ttnn.deallocate(v589, False)
    v594 = ttnn.relu(
        v593,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v593, False)
    v595 = ttnn.to_memory_config(
        v594, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )

    # ---------------------------------------------------------- #
    # * * * Convolution 51 * * *                                 #
    # ---------------------------------------------------------- #

    v596 = ttnn.conv2d(
        input_tensor=v594,
        weight_tensor=v300,
        device=v485,
        in_channels=2048,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v301,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v594, False)
    # ttnn.deallocate(v301, False)
    # ttnn.deallocate(v300, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 52 * * *                                 #
    # ---------------------------------------------------------- #

    v597 = ttnn.conv2d(
        input_tensor=v596,
        weight_tensor=v427,
        device=v485,
        in_channels=512,
        out_channels=512,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[3, 3],
        stride=[1, 1],
        padding=[1, 1, 1, 1],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v428,
        conv_config=ttnn.Conv2dConfig(
            weights_dtype=ttnn.DataType.BFLOAT8_B, activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        ),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 64],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v596, False)
    # ttnn.deallocate(v428, False)
    # ttnn.deallocate(v427, False)

    # ---------------------------------------------------------- #
    # * * * Convolution 53 * * *                                 #
    # ---------------------------------------------------------- #

    v598 = ttnn.conv2d(
        input_tensor=v597,
        weight_tensor=v324,
        device=v485,
        in_channels=512,
        out_channels=2048,
        batch_size=16,
        input_height=7,
        input_width=7,
        kernel_size=[1, 1],
        stride=[1, 1],
        padding=[0, 0, 0, 0],
        dilation=[1, 1],
        groups=1,
        bias_tensor=v325,
        conv_config=ttnn.Conv2dConfig(weights_dtype=ttnn.DataType.BFLOAT8_B),
        compute_config=compute_config,
        slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dL1Full, num_slices=0),
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v597, False)
    # ttnn.deallocate(v325, False)
    # ttnn.deallocate(v324, False)

    v599 = ttnn.add(
        v598,
        v595,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v598, False)
    ttnn.deallocate(v595, False)

    v600 = ttnn.relu(
        v599,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 6))]),
                [128, 256],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v599, False)

    v601 = ttnn.to_memory_config(
        v600, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v600, False)

    v602 = ttnn.reshape(
        v601,
        [16, 7, 7, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v601, False)

    v603 = ttnn.permute(
        v602,
        [0, 3, 1, 2],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(v602, False)

    v604 = ttnn.sum(
        v603,
        [2, 3],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v603, False)

    v605 = ttnn.multiply(
        v604,
        v486,
        dtype=ttnn.DataType.BFLOAT8_B,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                [32, 32],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(v604, False)
    ttnn.deallocate(v486, False)

    v606 = ttnn.to_memory_config(
        v605, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
    )
    ttnn.deallocate(v605, False)

    v607 = ttnn.linear(
        v606,
        v3,
        bias=v392,
        transpose_a=False,
        transpose_b=True,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v606, False)
    # ttnn.deallocate(v392, False)

    v608 = [v607]
    return v608


def create_inputs_for__main():
    v1 = my_get_device.DeviceGetter.get_device()
    v1.enable_program_cache()

    v2 = ttnn.ones(shape=ttnn.Shape([1000]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v3 = ttnn.ones(shape=ttnn.Shape([1000, 2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v4 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v5 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v6 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v7 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v8 = ttnn.ones(
        shape=ttnn.Shape([2048, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v9 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v10 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v11 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v12 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v13 = ttnn.ones(
        shape=ttnn.Shape([1024, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v14 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v15 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v16 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v17 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v18 = ttnn.ones(
        shape=ttnn.Shape([512, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v19 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v20 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v21 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v22 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v23 = ttnn.ones(
        shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v24 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v25 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v26 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v27 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v28 = ttnn.ones(
        shape=ttnn.Shape([64, 3, 7, 7]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v29 = ttnn.ones(
        shape=ttnn.Shape([16, 3, 224, 224]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1
    )
    v30 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v31 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v32 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v33 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v34 = ttnn.ones(
        shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v35 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v36 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v37 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v38 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v39 = ttnn.ones(
        shape=ttnn.Shape([64, 64, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v40 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v41 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v42 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v43 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v44 = ttnn.ones(
        shape=ttnn.Shape([64, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v45 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v46 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v47 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v48 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v49 = ttnn.ones(
        shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v50 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v51 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v52 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v53 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v54 = ttnn.ones(
        shape=ttnn.Shape([64, 64, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v55 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v56 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v57 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v58 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v59 = ttnn.ones(
        shape=ttnn.Shape([64, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v60 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v61 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v62 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v63 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v64 = ttnn.ones(
        shape=ttnn.Shape([256, 64, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v65 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v66 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v67 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v68 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v69 = ttnn.ones(
        shape=ttnn.Shape([64, 64, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v70 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v71 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v72 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v73 = ttnn.ones(shape=ttnn.Shape([64]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v74 = ttnn.ones(
        shape=ttnn.Shape([64, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v75 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v76 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v77 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v78 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v79 = ttnn.ones(
        shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v80 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v81 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v82 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v83 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v84 = ttnn.ones(
        shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v85 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v86 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v87 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v88 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v89 = ttnn.ones(
        shape=ttnn.Shape([128, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v90 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v91 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v92 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v93 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v94 = ttnn.ones(
        shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v95 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v96 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v97 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v98 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v99 = ttnn.ones(
        shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v100 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v101 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v102 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v103 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v104 = ttnn.ones(
        shape=ttnn.Shape([128, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v105 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v106 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v107 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v108 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v109 = ttnn.ones(
        shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v110 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v111 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v112 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v113 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v114 = ttnn.ones(
        shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v115 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v116 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v117 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v118 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v119 = ttnn.ones(
        shape=ttnn.Shape([128, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v120 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v121 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v122 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v123 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v124 = ttnn.ones(
        shape=ttnn.Shape([512, 128, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v125 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v126 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v127 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v128 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v129 = ttnn.ones(
        shape=ttnn.Shape([128, 128, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v130 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v131 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v132 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v133 = ttnn.ones(shape=ttnn.Shape([128]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v134 = ttnn.ones(
        shape=ttnn.Shape([128, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v135 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v136 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v137 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v138 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v139 = ttnn.ones(
        shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v140 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v141 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v142 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v143 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v144 = ttnn.ones(
        shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v145 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v146 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v147 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v148 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v149 = ttnn.ones(
        shape=ttnn.Shape([256, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v150 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v151 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v152 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v153 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v154 = ttnn.ones(
        shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v155 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v156 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v157 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v158 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v159 = ttnn.ones(
        shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v160 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v161 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v162 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v163 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v164 = ttnn.ones(
        shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v165 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v166 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v167 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v168 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v169 = ttnn.ones(
        shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v170 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v171 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v172 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v173 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v174 = ttnn.ones(
        shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v175 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v176 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v177 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v178 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v179 = ttnn.ones(
        shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v180 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v181 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v182 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v183 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v184 = ttnn.ones(
        shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v185 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v186 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v187 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v188 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v189 = ttnn.ones(
        shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v190 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v191 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v192 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v193 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v194 = ttnn.ones(
        shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v195 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v196 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v197 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v198 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v199 = ttnn.ones(
        shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v200 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v201 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v202 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v203 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v204 = ttnn.ones(
        shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v205 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v206 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v207 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v208 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v209 = ttnn.ones(
        shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v210 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v211 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v212 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v213 = ttnn.ones(shape=ttnn.Shape([1024]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v214 = ttnn.ones(
        shape=ttnn.Shape([1024, 256, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v215 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v216 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v217 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v218 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v219 = ttnn.ones(
        shape=ttnn.Shape([256, 256, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v220 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v221 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v222 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v223 = ttnn.ones(shape=ttnn.Shape([256]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v224 = ttnn.ones(
        shape=ttnn.Shape([256, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v225 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v226 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v227 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v228 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v229 = ttnn.ones(
        shape=ttnn.Shape([2048, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v230 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v231 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v232 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v233 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v234 = ttnn.ones(
        shape=ttnn.Shape([512, 512, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v235 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v236 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v237 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v238 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v239 = ttnn.ones(
        shape=ttnn.Shape([512, 1024, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v240 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v241 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v242 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v243 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v244 = ttnn.ones(
        shape=ttnn.Shape([2048, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v245 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v246 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v247 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v248 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v249 = ttnn.ones(
        shape=ttnn.Shape([512, 512, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v250 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v251 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v252 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v253 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v254 = ttnn.ones(
        shape=ttnn.Shape([512, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v255 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v256 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v257 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v258 = ttnn.ones(shape=ttnn.Shape([2048]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v259 = ttnn.ones(
        shape=ttnn.Shape([2048, 512, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v260 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v261 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v262 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v263 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v264 = ttnn.ones(
        shape=ttnn.Shape([512, 512, 3, 3]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v265 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v266 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v267 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v268 = ttnn.ones(shape=ttnn.Shape([512]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=v1)
    v269 = ttnn.ones(
        shape=ttnn.Shape([512, 2048, 1, 1]), dtype=ttnn.DataType.BFLOAT8_B, layout=ttnn.Layout.TILE, device=None
    )
    v270 = [
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
        v16,
        v17,
        v18,
        v19,
        v20,
        v21,
        v22,
        v23,
        v24,
        v25,
        v26,
        v27,
        v28,
        v29,
        v30,
        v31,
        v32,
        v33,
        v34,
        v35,
        v36,
        v37,
        v38,
        v39,
        v40,
        v41,
        v42,
        v43,
        v44,
        v45,
        v46,
        v47,
        v48,
        v49,
        v50,
        v51,
        v52,
        v53,
        v54,
        v55,
        v56,
        v57,
        v58,
        v59,
        v60,
        v61,
        v62,
        v63,
        v64,
        v65,
        v66,
        v67,
        v68,
        v69,
        v70,
        v71,
        v72,
        v73,
        v74,
        v75,
        v76,
        v77,
        v78,
        v79,
        v80,
        v81,
        v82,
        v83,
        v84,
        v85,
        v86,
        v87,
        v88,
        v89,
        v90,
        v91,
        v92,
        v93,
        v94,
        v95,
        v96,
        v97,
        v98,
        v99,
        v100,
        v101,
        v102,
        v103,
        v104,
        v105,
        v106,
        v107,
        v108,
        v109,
        v110,
        v111,
        v112,
        v113,
        v114,
        v115,
        v116,
        v117,
        v118,
        v119,
        v120,
        v121,
        v122,
        v123,
        v124,
        v125,
        v126,
        v127,
        v128,
        v129,
        v130,
        v131,
        v132,
        v133,
        v134,
        v135,
        v136,
        v137,
        v138,
        v139,
        v140,
        v141,
        v142,
        v143,
        v144,
        v145,
        v146,
        v147,
        v148,
        v149,
        v150,
        v151,
        v152,
        v153,
        v154,
        v155,
        v156,
        v157,
        v158,
        v159,
        v160,
        v161,
        v162,
        v163,
        v164,
        v165,
        v166,
        v167,
        v168,
        v169,
        v170,
        v171,
        v172,
        v173,
        v174,
        v175,
        v176,
        v177,
        v178,
        v179,
        v180,
        v181,
        v182,
        v183,
        v184,
        v185,
        v186,
        v187,
        v188,
        v189,
        v190,
        v191,
        v192,
        v193,
        v194,
        v195,
        v196,
        v197,
        v198,
        v199,
        v200,
        v201,
        v202,
        v203,
        v204,
        v205,
        v206,
        v207,
        v208,
        v209,
        v210,
        v211,
        v212,
        v213,
        v214,
        v215,
        v216,
        v217,
        v218,
        v219,
        v220,
        v221,
        v222,
        v223,
        v224,
        v225,
        v226,
        v227,
        v228,
        v229,
        v230,
        v231,
        v232,
        v233,
        v234,
        v235,
        v236,
        v237,
        v238,
        v239,
        v240,
        v241,
        v242,
        v243,
        v244,
        v245,
        v246,
        v247,
        v248,
        v249,
        v250,
        v251,
        v252,
        v253,
        v254,
        v255,
        v256,
        v257,
        v258,
        v259,
        v260,
        v261,
        v262,
        v263,
        v264,
        v265,
        v266,
        v267,
        v268,
        v269,
    ]
    return v270


def main():
    # Generate inputs
    v1 = create_inputs_for__main()
    inputs = v1[27]

    batch_size = v1[27].shape[0]

    import time
    import tqdm

    # First Warmup Run
    warmup_start = time.time()
    v2 = _main(v1)
    for item in v2:
        ttnn.from_device(item, True)
    warmup_end = time.time()
    warmup_duration = warmup_end - warmup_start
    print(100 * "-")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Warmup completed in {warmup_duration} seconds")
    print(f"Total number of iterations: 1")
    print(f"Batch size, number of samples: {batch_size}")
    print(f"Samples per second: {batch_size / warmup_duration}")
    print(100 * "-")
    print("\n")

    # Additional Warmup Runs
    loop_count = 4
    add_warmup_start = time.time()
    for _ in range(loop_count):
        v2 = _main(v1)
        for item in v2:
            ttnn.from_device(item, True)
    add_warmup_end = time.time()
    add_warmup_duration = add_warmup_end - add_warmup_start
    print(100 * "-")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Warmup completed in {add_warmup_duration} seconds")
    print(f"Total number of iterations: 1")
    print(f"Batch size, number of samples: {batch_size}")
    print(f"Total number of samples: {loop_count * batch_size}")
    print(f"Samples per second: {loop_count * batch_size / add_warmup_duration}")
    print(100 * "-")
    print("\n")

    # Measured Run
    loop_count = 15
    start = time.time()
    for _ in range(loop_count):
        v2 = _main(v1)
        for item in v2:
            ttnn.from_device(item, True)
    end = time.time()
    duration = end - start
    print(100 * "-")
    print(f"Inputs shape: {inputs.shape}")
    print(f"Measure completed in {duration} seconds")
    print(f"Total number of iterations: 1")
    print(f"Batch size, number of samples: {batch_size}")
    print(f"Total number of samples: {loop_count * batch_size}")
    print(f"Samples per second: {loop_count * batch_size / duration}")
    print(100 * "-")
    print("\n")


if __name__ == "__main__":
    main()
