# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4 local affine/gate/repack; no TTNN import at module load.

Frozen affine compute/writer arithmetic is reused unchanged. Inputs/output must
be compact tile-interleaved DRAM. Caller-owned output is optional; no owner hook
or global tensor cache is installed, and model defaults are untouched.
"""


def local_affine_sources(reader):
    """Change only local head stride and packed gate tile offset."""
    for before, after in (
        ("source_head=user*48+head", "source_head=user*12+head"),
        ("row=(user*48+head)%32", "row=(user*12+head)%32"),
        (".page_id=320+tile", ".page_id=80+tile"),
    ):
        assert reader.count(before) == 1, (before, reader.count(before))
        reader = reader.replace(before, after, 1)
    return reader


def local_affine_gate(normalized, weight, packed, *, output=None, cores=None):
    """[1,B*12,128] BF16 normalized heads -> [1,B,1536] BF16 gated output."""
    import ttnn

    from .optimized_decoder import (
        _affinev3__GDN_AFFINE_GATE_READER,
        _affinev3__GDN_AFFINE_GATE_COMPUTE,
        _affinev3__GDN_AFFINE_GATE_WRITER,
    )

    assert len(normalized.shape) == 3 and normalized.shape[0] == 1 and normalized.shape[2] == 128
    assert normalized.shape[1] % 12 == 0
    batch = normalized.shape[1] // 12
    assert 1 <= batch <= 32 and tuple(weight.shape) == (1, 1, 128)
    assert len(packed.shape) == 3 and tuple(packed.shape)[:2] == (1, batch) and packed.shape[2] >= 4096
    for tensor in (normalized, weight, packed):
        assert tensor.dtype == ttnn.bfloat16 and tensor.layout == ttnn.TILE_LAYOUT
        assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(normalized.padded_shape) == (1, ((batch * 12 + 31) // 32) * 32, 128)
    assert tuple(weight.padded_shape) == (1, 32, 128)
    assert packed.padded_shape[-2] == 32 and packed.padded_shape[-1] % 32 == 0
    if output is None:
        output = ttnn.empty(
            (1, batch, 1536),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=normalized.device(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    assert tuple(output.shape) == (1, batch, 1536) and tuple(output.padded_shape) == (1, 32, 1536)
    assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert output.buffer_address() not in [t.buffer_address() for t in (normalized, weight, packed)]
    assert all(
        t.device() == normalized.device() for t in (weight, packed, output)
    ), "All tensors must share a device/mesh"
    assert cores is None or cores >= 1
    reader = local_affine_sources(_affinev3__GDN_AFFINE_GATE_READER)
    grid = normalized.device().compute_with_storage_grid_size()
    workers = min(48, cores or grid.x * grid.y)
    assert 1 <= workers <= grid.x * grid.y
    coords = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(workers)]
    ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
    rr, wr, cr = (ttnn.RuntimeArgs() for _ in range(3))
    start = 0
    for i, c in enumerate(coords):
        count = 48 // workers + (i < 48 % workers)
        rr[c.x][c.y] = [
            normalized.buffer_address(),
            weight.buffer_address(),
            packed.buffer_address(),
            batch,
            start,
            count,
        ]
        wr[c.x][c.y] = [output.buffer_address(), start, count]
        cr[c.x][c.y] = [count]
        start += count
    rct = []
    for t in (normalized, weight, packed):
        rct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())

    def kernel(source, config, args, ct):
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=ranges,
            compile_time_args=ct,
            runtime_args=args,
            config=config,
        )

    config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )
    kernels = [
        kernel(reader, ttnn.ReaderConfigDescriptor(), rr, rct),
        kernel(
            _affinev3__GDN_AFFINE_GATE_WRITER,
            ttnn.WriterConfigDescriptor(),
            wr,
            ttnn.TensorAccessorArgs(output).get_compile_time_args(),
        ),
        kernel(_affinev3__GDN_AFFINE_GATE_COMPUTE, config, cr, []),
    ]
    cbs = [
        ttnn.CBDescriptor(
            total_size=(8192 if i == 3 else 2048),
            core_ranges=ranges,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=i,
                    data_format=ttnn.float32 if i == 3 else ttnn.bfloat16,
                    page_size=4096 if i == 3 else 2048,
                )
            ],
        )
        for i in range(5)
    ]
    ttnn.generic_op(
        [normalized, weight, packed, output], ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
    )
    return output
