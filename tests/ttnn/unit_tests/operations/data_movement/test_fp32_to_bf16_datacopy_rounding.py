# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import ttnn

SHAPE = (1, 1, 256, 256)
NUM_CORES = 8


def _fp32_input():
    g = torch.Generator().manual_seed(0)
    n = SHAPE[-2] * SHAPE[-1]
    mantissa = torch.randint(0, 1 << 23, (n,), generator=g, dtype=torch.int64)
    exponent = torch.randint(127 - 20, 127 + 21, (n,), generator=g, dtype=torch.int64)
    sign = torch.randint(0, 2, (n,), generator=g, dtype=torch.int64)
    x = ((sign << 31) | (exponent << 23) | mantissa).to(torch.int32).view(torch.float32).reshape(SHAPE)
    x[0, 0, 0, :12] = torch.tensor(
        [257.0, 369.0, 1 + 2**-8, 4.140625, -257.0, -4.140625, 0.1, 1 / 3, 519.0, 65535.0, 2.7182817, -0.1]
    )
    return x


def _height_sharded(device):
    return ttnn.create_sharded_memory_config(
        SHAPE, core_grid=ttnn.CoreGrid(y=1, x=NUM_CORES), strategy=ttnn.ShardStrategy.HEIGHT
    )


def _convert(op, x, device):
    t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    if op == "clone":
        return ttnn.clone(t, dtype=ttnn.bfloat16)
    if op == "clone_l1":
        return ttnn.clone(t, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
    if op in ("copy_dram", "copy_l1"):
        mem = ttnn.DRAM_MEMORY_CONFIG if op == "copy_dram" else ttnn.L1_MEMORY_CONFIG
        out = ttnn.from_torch(
            torch.zeros(SHAPE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        ttnn.copy(t, out)
        return out
    if op == "to_memory_config":
        return ttnn.to_memory_config(t, _height_sharded(device), dtype=ttnn.bfloat16)
    if op == "interleaved_to_sharded":
        return ttnn.interleaved_to_sharded(t, _height_sharded(device), ttnn.bfloat16)
    if op == "sharded_to_interleaved":
        sharded = ttnn.to_memory_config(t, _height_sharded(device))
        return ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG, ttnn.bfloat16)
    if op in ("interleaved_to_sharded_partial", "sharded_to_interleaved_partial"):
        num_slices = 2
        slice_h = SHAPE[-2] // num_slices
        shard_shape = [slice_h // 4, SHAPE[-1]]
        out = ttnn.from_torch(torch.zeros(SHAPE), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        for i in range(num_slices):
            to_bf16_here = op == "interleaved_to_sharded_partial"
            sliced = ttnn.interleaved_to_sharded_partial(
                t,
                (4, 1),
                shard_shape,
                num_slices,
                i,
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.ShardOrientation.ROW_MAJOR,
                output_dtype=ttnn.bfloat16 if to_bf16_here else None,
            )
            ttnn.sharded_to_interleaved_partial(
                sliced,
                out,
                num_slices,
                i,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                output_dtype=ttnn.bfloat16,
            )
        return out
    raise ValueError(op)


def _to_bf16_ties_away(x):
    bits = (x.view(torch.int32).to(torch.int64) & 0xFFFFFFFF) + 0x8000
    bits = (bits & 0xFFFF0000) - ((bits & 0x80000000) << 1)
    return bits.to(torch.int32).view(torch.float32).to(torch.bfloat16)


@pytest.mark.parametrize(
    "op",
    [
        "clone",
        "clone_l1",
        "copy_dram",
        "copy_l1",
        "to_memory_config",
        "interleaved_to_sharded",
        "sharded_to_interleaved",
        "interleaved_to_sharded_partial",
        "sharded_to_interleaved_partial",
    ],
)
def test_fp32_to_bf16_datacopy_rounds_to_nearest(device, op):
    x = _fp32_input()
    expected = _to_bf16_ties_away(x)
    got = ttnn.to_torch(_convert(op, x, device))
    assert got.dtype == torch.bfloat16
    mismatches = int((got != expected).sum())
    assert mismatches == 0, f"{op}: {mismatches} of {x.numel()} values differ from round-to-nearest (ties away)"
    ties = (x.view(torch.int32) & 0xFFFF) == 0x8000
    assert torch.equal(got[~ties], x.to(torch.bfloat16)[~ties])


@pytest.mark.parametrize("start, end, step", [(0, 2000, 3), (0.1, 0.5, 0.1), (0, 1, 1 / 3), (257, 385, 2)])
@pytest.mark.parametrize("untilize_out", [False, True])
def test_moreh_arange_bf16_rounds_to_nearest_even(device, start, end, step, untilize_out):
    ref = torch.arange(start, end, step, dtype=torch.float32)
    got = ttnn.to_torch(ttnn.moreh_arange(start, end, step, device, dtype=ttnn.bfloat16, untilize_out=untilize_out))
    got = got.flatten()[: ref.numel()]
    assert torch.equal(got, ref.to(torch.bfloat16))
