# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Head layout of the V4.1 attention around ``sparse_sdpa`` (bead 8y7.9.9), one fused op per side.

``sparse_sdpa`` reads its queries and writes its output row-major and head-major, ``[1, H, S, D]``; the projections
around it are seq-major tiled matmuls. Instead of ``nlp_create_qkv_heads`` / ``nlp_concat_heads`` plus the RoPE-tail
slice / rotate / untilize / slice_write / tilize glue, each side is one pass over the heads:

* ``q_heads``: the tiled ``wq_b`` projection ``[1, 1, S, H * D]`` (heads side by side) -> row-major ``[1, H, S, D]``
  with RoPE on each head's trailing ``rope_dim`` channels;
* ``o_heads``: the row-major attention ``[1, H, S, D]`` -> tiled ``[1, G, S, (H / G) * D]`` (group g's heads side by
  side: the grouped ``wo_a`` input) with the inverse RoPE of each tail.

The RoPE is ``rotary_embedding_llama``'s op sequence with its bf16 intermediates (``kernels/heads_rope.hpp``), so
both ops reproduce the composite path bit for bit (``tests/v41/test_v41_head_layout.py``).
"""

import ttnn

_KERNEL_DIR = "models/demos/deepseek_v3_d_p/tt/v41/kernels"
_TILE = 32
_BF16_TILE_BYTES = _TILE * _TILE * 2
# rotary_embedding_llama's default compute config (HiFi4, approx, bf16 DST)
_COMPUTE = dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=True)


def _cores(device, units: int):
    """The first ``min(units, grid)`` compute cores, row-major, and their contiguous unit ranges."""
    grid = device.compute_with_storage_grid_size()
    num_cores = min(grid.x * grid.y, units)
    base, extra = divmod(units, num_cores)
    last = ttnn.CoreCoord((num_cores - 1) % grid.x, (num_cores - 1) // grid.x)
    ranges = []
    if last.y > 0:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, last.y - 1)))
    ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, last.y), last))
    spans, start = [], 0
    for i in range(num_cores):
        count = base + (i < extra)
        spans.append((i % grid.x, i // grid.x, start, count))
        start += count
    return ttnn.CoreRangeSet(ranges), spans


def _cbs(cores, tiles: dict):
    def cb(index: int, n: int) -> ttnn.CBDescriptor:
        page = ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.bfloat16, page_size=_BF16_TILE_BYTES)
        return ttnn.CBDescriptor(total_size=n * _BF16_TILE_BYTES, core_ranges=cores, format_descriptors=[page])

    return [cb(i, n) for i, n in tiles.items()]


def _check_rope(cos, sin, trans, tokens: int, rope_dim: int):
    for t in (cos, sin):
        assert t.dtype == ttnn.bfloat16 and t.layout == ttnn.TILE_LAYOUT, (t.dtype, t.layout)
        assert tuple(t.shape)[-2:] == (tokens, rope_dim), (tuple(t.shape), tokens, rope_dim)
    assert trans.dtype == ttnn.bfloat16 and tuple(trans.shape)[-2:] == (_TILE, _TILE), (trans.dtype, trans.shape)


def _geometry(head_dim: int, rope_dim: int, tokens: int):
    assert head_dim % _TILE == 0 and rope_dim % _TILE == 0 and 0 < rope_dim < head_dim, (head_dim, rope_dim)
    assert tokens % _TILE == 0, tokens
    return (head_dim - rope_dim) // _TILE, rope_dim // _TILE


def _accessor(t):
    return ttnn.TensorAccessorArgs(t).get_compile_time_args()


def _program(device, units, kernels, cbs, tensors):
    """One generic op: ``kernels`` = [(file, compile args, per-core runtime args(start, count), config)]."""
    cores, spans = _cores(device, units)
    descriptors = []
    for source, ct_args, rt_args, config in kernels:
        args = ttnn.RuntimeArgs()
        for cx, cy, start, count in spans:
            args[cx][cy] = rt_args(start, count)
        descriptors.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{_KERNEL_DIR}/{source}",
                core_ranges=cores,
                compile_time_args=ct_args,
                runtime_args=args,
                config=config,
            )
        )
    return ttnn.generic_op(tensors, ttnn.ProgramDescriptor(kernels=descriptors, semaphores=[], cbs=_cbs(cores, cbs)))


def q_heads(q, cos, sin, trans, heads: int, rope_dim: int) -> ttnn.Tensor:
    """Tiled bf16 ``q`` [1, 1, S, heads * D] -> row-major bf16 [1, heads, S, D], RoPE (``cos`` / ``sin`` [1, 1, S,
    rope_dim] tiled, ``trans`` the 32x32 rotation) on each head's last ``rope_dim`` channels."""
    assert q.dtype == ttnn.bfloat16 and q.layout == ttnn.TILE_LAYOUT, (q.dtype, q.layout)
    assert not q.memory_config().is_sharded(), "q_heads takes an interleaved q"
    tokens, width = q.shape[-2], q.shape[-1]
    assert q.shape[0] == 1 and q.shape[1] == 1 and width % heads == 0, (tuple(q.shape), heads)
    head_dim = width // heads
    nt, rt = _geometry(head_dim, rope_dim, tokens)
    _check_rope(cos, sin, trans, tokens, rope_dim)
    device = q.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, heads, tokens, head_dim]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    units = tokens // _TILE * heads  # (tile row, head), row-major: a core's heads of one row share its cos / sin
    addrs = [q.buffer_address(), cos.buffer_address(), sin.buffer_address(), trans.buffer_address()]
    compute = ttnn.ComputeConfigDescriptor(**_COMPUTE)
    kernels = [
        (
            "heads_q_reader.cpp",
            [nt, rt, heads] + _accessor(q) + _accessor(cos) + _accessor(sin) + _accessor(trans),
            lambda s, c: addrs + [s, c],
            ttnn.ReaderConfigDescriptor(),
        ),
        (
            "heads_q_writer.cpp",
            [nt, rt, heads, tokens] + _accessor(out),
            lambda s, c: [out.buffer_address(), s, c],
            ttnn.WriterConfigDescriptor(),
        ),
        ("heads_q_compute.cpp", [nt, rt, heads], lambda s, c: [s, c], compute),
    ]
    # cb 0 no-rope / 1 tail tiles, 2 / 3 cos / sin, 4 rotation, 5-7 RoPE intermediates, 8 rotated tail,
    # 16 / 17 the untilized no-rope / tail rows
    cbs = {0: 2 * nt, 1: 2 * rt, 2: 2 * rt, 3: 2 * rt, 4: 1, 5: rt, 6: rt, 7: rt, 8: 2 * rt, 16: 2 * nt, 17: 2 * rt}
    return _program(device, units, kernels, cbs, [q, cos, sin, trans, out])


def o_heads(attn, cos, sin, trans, groups: int, rope_dim: int) -> ttnn.Tensor:
    """Row-major bf16 ``attn`` [1, H, S, D] -> tiled bf16 [1, groups, S, (H / groups) * D], RoPE with ``cos`` /
    ``sin`` (pass -sin for the inverse) on each head's last ``rope_dim`` channels."""
    assert attn.dtype == ttnn.bfloat16 and attn.layout == ttnn.ROW_MAJOR_LAYOUT, (attn.dtype, attn.layout)
    assert not attn.memory_config().is_sharded(), "o_heads takes an interleaved attn"
    _, heads, tokens, head_dim = attn.shape
    assert attn.shape[0] == 1 and heads % groups == 0, (tuple(attn.shape), groups)
    nt, rt = _geometry(head_dim, rope_dim, tokens)
    _check_rope(cos, sin, trans, tokens, rope_dim)
    hpg = heads // groups
    device = attn.device()
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, groups, tokens, hpg * head_dim]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    units = tokens // _TILE * heads
    addrs = [attn.buffer_address(), cos.buffer_address(), sin.buffer_address(), trans.buffer_address()]
    compute = ttnn.ComputeConfigDescriptor(**_COMPUTE)
    kernels = [
        (
            "heads_o_reader.cpp",
            [nt, rt, heads, tokens] + _accessor(attn) + _accessor(cos) + _accessor(sin) + _accessor(trans),
            lambda s, c: addrs + [s, c],
            ttnn.ReaderConfigDescriptor(),
        ),
        (
            "heads_o_writer.cpp",
            [nt, rt, heads, hpg, tokens // _TILE] + _accessor(out),
            lambda s, c: [out.buffer_address(), s, c],
            ttnn.WriterConfigDescriptor(),
        ),
        ("heads_o_compute.cpp", [nt, rt, heads], lambda s, c: [s, c], compute),
    ]
    # cb 0 / 1 no-rope / tail rows, 2 / 3 cos / sin, 4 rotation, 5-7 RoPE intermediates, 8 tail tiles,
    # 16 / 17 the no-rope / rotated tail tiles
    cbs = {0: 2 * nt, 1: 2 * rt, 2: 2 * rt, 3: 2 * rt, 4: 1, 5: rt, 6: rt, 7: rt, 8: rt, 16: 2 * nt, 17: 2 * rt}
    return _program(device, units, kernels, cbs, [attn, cos, sin, trans, out])
