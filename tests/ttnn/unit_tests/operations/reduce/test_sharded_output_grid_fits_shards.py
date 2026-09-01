# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""A sharded tensor's grid must not hold more cores than it has shards.

Reduce ops build their output spec by borrowing the *input's* core grid whenever the
output memory config supplies a sharded layout but no shard spec
(build_reduce_output_tensor_spec -> get_grid_and_orientation,
ttnn/cpp/ttnn/operations/reduction/generic/device/common.cpp:207). The three
TensorSpec sharding builders then honour that grid verbatim while deriving a shard
shape that may need far fewer cores, so a 16x1-tile reduction result can claim the
88- or 109-core grid of its 16x64-tile input.

Nothing rejects it: the shard-count checks at tt_metal/impl/tensor/spec/tensor_spec.cpp
only fire when there are too *few* cores. The numerics stay correct, which is why no
functional test catches this -- the defect is in the spec, so it has to be asserted
on the spec.

Reachable from plain ttnn; a compiler that emits bare sharded layouts hits it on every
reduction.
"""

import pytest
import torch
import ttnn

TILE = 32


def _shard_spec(memory_config):
    spec = memory_config.shard_spec
    return spec() if callable(spec) else spec


def _grid(x_start, y_start, x_end, y_end):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x_start, y_start), ttnn.CoreCoord(x_end, y_end))})


CASES = [
    # label, input layout, input grid, input shard in tiles, tensor shape, dtype
    (
        "block_sharded 2x6 over 88 cores",
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        _grid(0, 0, 10, 7),
        (2, 6),
        (1, 512, 2048),
        ttnn.float32,
    ),
    (
        "width_sharded 37 tiles over 88 cores",
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        _grid(0, 0, 10, 7),
        (16, 37),
        (1, 512, 37 * TILE * 88),
        ttnn.bfloat16,
    ),
]

OUT_LAYOUTS = [
    ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
]


@pytest.mark.parametrize("keepdim", [True, False], ids=["keepdim", "no_keepdim"])
@pytest.mark.parametrize("out_layout", OUT_LAYOUTS, ids=lambda l: str(l).split(".")[-1])
@pytest.mark.parametrize("label, in_layout, in_grid, in_shard_tiles, shape, dtype", CASES, ids=[c[0] for c in CASES])
def test_reduce_output_grid_fits_its_shards(
    device, label, in_layout, in_grid, in_shard_tiles, shape, dtype, out_layout, keepdim
):
    in_mem = ttnn.MemoryConfig(
        in_layout,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(in_grid, (in_shard_tiles[0] * TILE, in_shard_tiles[1] * TILE), ttnn.ShardOrientation.ROW_MAJOR),
    )
    # A layout and buffer type with no shard spec: the shape is left to tt-metal.
    out_mem = ttnn.MemoryConfig(out_layout, ttnn.BufferType.L1)

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    x = ttnn.from_torch(
        torch.randn(*shape, dtype=torch_dtype),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem,
    )
    out = ttnn.mean(x, dim=-1, keepdim=keepdim, memory_config=out_mem)

    spec = _shard_spec(out.memory_config())
    if spec is None:
        ttnn.deallocate(x)
        ttnn.deallocate(out)
        pytest.skip(
            f"{label} -> bare {str(out_layout).split('.')[-1]} (keepdim={keepdim}): "
            "op returned an interleaved output, so there is no grid to check"
        )

    cores = spec.grid.num_cores()
    shard_h, shard_w = spec.shape[0], spec.shape[1]
    # ttnn Shape does not support slicing; materialise it first
    padded = tuple(out.padded_shape)
    height = 1
    for d in padded[:-1]:
        height *= d
    shards = ((height + shard_h - 1) // shard_h) * ((padded[-1] + shard_w - 1) // shard_w)

    ttnn.deallocate(x)
    ttnn.deallocate(out)

    assert cores <= shards, (
        f"{label} -> bare {str(out_layout).split('.')[-1]}: output grid holds {cores} cores "
        f"for {shards} shard(s) of {shard_h}x{shard_w} px. "
        f"{cores - shards} core(s) have L1 reserved for data that does not exist."
    )
