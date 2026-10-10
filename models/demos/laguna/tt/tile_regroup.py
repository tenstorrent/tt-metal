# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Head split / join of 32-row tensors as one multi-core tile copy (Laguna DFlash draft; kernels/tile_regroup.cpp).

For a 32-row TILE tensor, [1, 1, 32, n * hd] (heads side by side) and [1, n, 32, hd] (one head per batch row) hold a
head's tiles in the same order, so nlp_create_qkv_heads / nlp_concat_heads (one core each at 32 rows: ~36 / ~29 us
for the draft's 18 + 2 + 2 heads) reduce to copying runs of tiles."""

import math
from pathlib import Path

import ttnn

_KDIR = Path(__file__).resolve().parent / "kernels"
_PER = 2  # tiles per core


def regroup(src, pieces, memory_config=None):
    """src: a 32-row bf16 TILE tensor; pieces: [(shape, first source tile)] with shape [1, n, 32, hd] or
    [1, 1, 32, w] (any 32-row shape whose tiles are a contiguous run of src's). Returns the pieces (up to 3)."""
    device = src.device()
    assert src.dtype == ttnn.bfloat16 and src.layout == ttnn.TILE_LAYOUT and src.padded_shape[-2] == 32, src.shape
    assert 1 <= len(pieces) <= 3, len(pieces)
    mem = memory_config or src.memory_config()
    outs, args, core = [], [src.buffer_address()], 0
    for shape, off in pieces:
        assert shape[-2] == 32 and shape[-1] % 32 == 0, shape
        out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mem)
        n = math.prod(list(out.padded_shape)) // 1024
        args += [out.buffer_address(), int(off), n, core, _PER]
        core += -(-n // _PER)
        outs.append(out)
    gs = device.compute_with_storage_grid_size()
    assert core <= gs.x * gs.y, core
    grid = ttnn.num_cores_to_corerangeset(core, gs, True)
    k = ttnn.KernelDescriptor(
        kernel_source=str(_KDIR / "tile_regroup.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[len(pieces), gs.x, 2048]
        + list(ttnn.TensorAccessorArgs(src).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(outs[0]).get_compile_time_args()),
        common_runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=2048 * _PER,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
    )
    ttnn.generic_op([src] + outs, ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cb]))
    return outs


def split_heads(qkv, num_heads, num_kv_heads, head_dim, memory_config=None):
    """[1, 1, 32, (nh + 2 nkv) hd] -> q [1, nh, 32, hd], k, v [1, nkv, 32, hd] (nlp_create_qkv_heads at 32 rows)."""
    t = head_dim // 32
    return regroup(qkv, [((1, num_heads, 32, head_dim), 0), ((1, num_kv_heads, 32, head_dim), num_heads * t),
                         ((1, num_kv_heads, 32, head_dim), (num_heads + num_kv_heads) * t)], memory_config)  # fmt: skip


def join_heads(attn, memory_config=None):
    """[1, n, 32, hd] -> [1, 1, 32, n hd] (nlp_concat_heads at 32 rows)."""
    return regroup(attn, [((1, 1, 32, attn.shape[1] * attn.shape[-1]), 0)], memory_config)[0]
