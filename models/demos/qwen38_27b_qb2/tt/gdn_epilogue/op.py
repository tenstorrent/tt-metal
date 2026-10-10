# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental single-token GDN output layout, gated RMSNorm and z multiply.

Consumes the recurrence's FP32 row-major output directly. Retains the native
BF16 rounding boundary before multiplying by z. Model integration is opt-in
at B16/B32; a standalone kernel pass does not qualify model accuracy.
"""

import math
import struct
from pathlib import Path

from models.demos.qwen38_27b_qb2.tt.gdn_step.op import work_items

HERE = Path(__file__).parent
BUFFERS = dict(x=0, gate=1, weight=2, tmp=3, stats=4, inv=5, norm=6, out=7, scaler=8, epsilon=9, rounded=10)


def source(name):
    bindings = "namespace dfb {" + ";".join(f"constexpr uint32_t {k}={v}" for k, v in BUFFERS.items()) + ";}\n"
    return (HERE / name).read_text().replace("// DFB_BINDINGS", bindings)


def epilogue(
    raw,
    gate,
    weight,
    output,
    *,
    heads=12,
    epsilon=1e-6,
    multiply_z=True,
    compact_gate=False,
    compact_output=False,
    gate_offset=0,
):
    """Normalize one token/user into caller-owned BF16 tiles.

    Public operands use [B,1,H*128]; compact operands use [1,B,H*128].
    A compact gate may be the whole packed projection, with a tile-aligned
    gate_offset selecting z without an intermediate slice. Compact users
    occupy disjoint face rows, including explicitly zeroed output padding.
    Both layout options are experimental and preserve the arithmetic kernel.
    """
    import ttnn

    if any(type(flag) is not bool for flag in (multiply_z, compact_gate, compact_output)):
        raise ValueError("Layout and multiply_z flags must be Boolean")
    if type(heads) is not int or heads < 1 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Require positive head count and finite positive epsilon")
    if type(gate_offset) is not int or gate_offset < 0 or gate_offset % 32 or (gate_offset and not compact_gate):
        raise ValueError("gate_offset must be nonnegative, tile aligned, and used only with compact gates")
    if len(gate.shape) != 3:
        raise ValueError("Gate must be three dimensional")
    batch = gate.shape[1 if compact_gate else 0]
    width = gate.shape[-1]
    if compact_gate:
        gate_shape = (1, batch, width)
        if width % 32 or width < gate_offset + heads * 128:
            raise ValueError("Packed compact gate does not cover the selected heads")
    else:
        gate_shape = (batch, 1, heads * 128)
    if batch < 1 or ((compact_gate or compact_output) and batch > 32):
        raise ValueError("Compact epilogue requires batch 1..32")
    output_shape = (1, batch, heads * 128) if compact_output else (batch, 1, heads * 128)
    tensors = [raw, gate, weight, output]
    mesh = raw.device()
    if "BLACKHOLE" not in str(mesh.arch()).upper():
        raise ValueError("Experimental epilogue currently targets Blackhole only")
    contracts = [
        ((batch * heads, 128), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        (gate_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((128,), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        (output_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT),
    ]
    for tensor, (shape, dtype, layout) in zip(tensors, contracts):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.layout != layout
            or tensor.memory_config() not in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG)
            or tensor.device() != mesh
        ):
            raise ValueError("Epilogue requires matching interleaved tensors with the declared shape/dtype/layout")
    gate_padded = (1, 32, width) if compact_gate else (batch, 32, heads * 128)
    output_padded = (1, 32, heads * 128) if compact_output else (batch, 32, heads * 128)
    if tuple(gate.padded_shape) != gate_padded or tuple(output.padded_shape) != output_padded:
        raise ValueError("Epilogue requires exact physical gate/output padding")
    if len({(str(t.memory_config().buffer_type), t.buffer_address()) for t in tensors}) != 4:
        raise ValueError("Epilogue inputs and output must not alias")
    grid = mesh.compute_with_storage_grid_size()
    assignments = work_items(batch * heads, grid.x, grid.y)
    cores = ttnn.num_cores_to_corerangeset(len(assignments), grid, row_wise=True)
    read, write, compute = (ttnn.RuntimeArgs() for _ in range(3))
    for x, y, first, stride, count in assignments:
        read[x][y] = [raw.buffer_address(), gate.buffer_address(), weight.buffer_address(), first, stride, count]
        write[x][y] = [output.buffer_address(), first, stride, count]
        compute[x][y] = [count]

    def accessors(values):
        return [a for t in values for a in ttnn.TensorAccessorArgs(t).get_compile_time_args()]

    eps_bits = struct.unpack("I", struct.pack("f", epsilon))[0]
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=source(name),
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=cores,
            compile_time_args=ctargs,
            runtime_args=args,
            config=config,
        )
        for name, args, ctargs, config in (
            (
                "reader.cpp",
                read,
                [eps_bits, int(compact_gate), heads, gate_offset // 32, *accessors(tensors[:3])],
                ttnn.ReaderConfigDescriptor(),
            ),
            (
                "writer.cpp",
                write,
                [int(compact_output), heads, batch, *accessors([output])],
                ttnn.WriterConfigDescriptor(),
            ),
            (
                "compute.cpp",
                compute,
                [int(multiply_z)],
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=True, fp32_dest_acc_en=True
                ),
            ),
        )
    ]
    # Match native norm intermediates and preserve the BF16 output before z.
    formats = [
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.float32,
        ttnn.float32,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.float32,
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.float32,
    ]
    pages = [8, 8, 4, 4, 1, 1, 4, 8, 1, 1, 4, 1]
    if compact_output:
        # Writer-private zero row: never shares the reader's CB11 staging.
        formats.append(ttnn.bfloat16)
        pages.append(1)
    cbs = []
    for index, (dtype, count) in enumerate(zip(formats, pages)):
        size = 4096 if dtype == ttnn.float32 else 2048
        cbs.append(
            ttnn.CBDescriptor(
                total_size=count * size,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=size)],
            )
        )
    return ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
