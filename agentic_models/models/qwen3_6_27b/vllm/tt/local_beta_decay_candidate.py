# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in compact TP4 beta/decay; frozen gate178 compute and writer unchanged."""


def local_reader_source(reader):
    """Adapt only head/page ownership; fail closed if frozen reader drifts."""
    for old, new in (
        ("task=get_arg_val<uint32_t>(3),user=task/2,col=task%2", "task=get_arg_val<uint32_t>(3),user=task,col=0"),
        (".page_id=514+col", ".page_id=129+col"),
        (".page_id=512+col", ".page_id=128+col"),
    ):
        assert reader.count(old) == 1, (old, reader.count(old))
        reader = reader.replace(old, new, 1)
    return reader


def local_beta_decay(packed, dt_bias, neg_a, *, outputs=None, diagnostic_stage=0):
    """BF16 packed[1,B,>=4160] -> betaBF16/gFP32[B,1,12].

    Stage0 is the full epilogue; stage1 exposes native mixed-format add and
    stage2 exposes native softplus. These controls do not change beta.
    """
    import ttnn
    from .optimized_decoder import _gate178_READER, _gate178_COMPUTE, _gate178_WRITER

    assert diagnostic_stage in (0, 1, 2)
    assert len(packed.shape) == 3 and packed.shape[0] == 1 and packed.shape[2] >= 4140
    batch = packed.shape[1]
    assert 1 <= batch <= 32 and packed.padded_shape[-2] == 32 and packed.padded_shape[-1] % 32 == 0
    assert packed.dtype == ttnn.bfloat16
    device = packed.device()
    for t in (dt_bias, neg_a):
        assert tuple(t.shape) == (1, 1, 12) and tuple(t.padded_shape) == (1, 32, 32)
        assert t.dtype == ttnn.float32
    for t in (packed, dt_bias, neg_a):
        assert t.layout == ttnn.TILE_LAYOUT and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG and t.device() == device
    if outputs is None:
        outputs = tuple(
            ttnn.empty(
                (batch, 1, 12),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for dtype in (ttnn.bfloat16, ttnn.float32)
        )
    assert len(outputs) == 2
    beta, g = outputs
    for t, dtype in zip(outputs, (ttnn.bfloat16, ttnn.float32)):
        assert tuple(t.shape) == (batch, 1, 12) and tuple(t.padded_shape) == (batch, 32, 32)
        assert t.dtype == dtype and t.layout == ttnn.TILE_LAYOUT
        assert t.device() == device and t.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert len({t.buffer_address() for t in (packed, dt_bias, neg_a, beta, g)}) == 5
    grid = device.compute_with_storage_grid_size()
    assert grid.x * grid.y >= batch
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(batch)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    for task, c in enumerate(coords):
        rr[c.x][c.y] = [packed.buffer_address(), dt_bias.buffer_address(), neg_a.buffer_address(), task]
        wr[c.x][c.y] = [beta.buffer_address(), g.buffer_address(), task]
        cr[c.x][c.y] = []

    def accessors(ts):
        result = []
        for t in ts:
            result.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
        return result

    def kernel(source, config, rt, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=rt,
            config=config,
        )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    modes = [ttnn.UnpackToDestMode.Default] * 64
    for i in (1, 3):
        modes[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    config.unpack_to_dest_mode = modes
    kernels = [
        kernel(
            local_reader_source(_gate178_READER), ttnn.ReaderConfigDescriptor(), rr, accessors((packed, dt_bias, neg_a))
        ),
        kernel(_gate178_WRITER, ttnn.WriterConfigDescriptor(), wr, accessors(outputs)),
        kernel(_gate178_COMPUTE, config, cr, [diagnostic_stage]),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=2048 if i in (0, 4) else 4096,
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i,
                    data_format=ttnn.bfloat16 if i in (0, 4) else ttnn.float32,
                    page_size=2048 if i in (0, 4) else 4096,
                )
            ],
        )
        for i in range(7)
    ]
    ttnn.generic_op([packed, dt_bias, neg_a, beta, g], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
    return beta, g
