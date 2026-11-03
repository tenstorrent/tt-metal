# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Row-major tensors whose rows span several shards, with a shard row that is not a multiple of the buffer
alignment (L1: 16 B; DRAM: 32 B on Wormhole, 64 B on Blackhole). noc_async_{read,write}_sharded move a full
aligned page per shard row, so the ops whose kernels use them must not hand such a tensor to those kernels
(#58402 item 17). Each case checks that the output keeps the sharded config the op derives and that every value
is right."""

import pytest
import torch

import ttnn

L1, DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
RM = ttnn.ROW_MAJOR_LAYOUT
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
CONTIGUOUS, ROUND_ROBIN = ttnn.ShardDistributionStrategy.CONTIGUOUS_1D, ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D


def grid(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*a), ttnn.CoreCoord(*b)) for a, b in ranges})


def nd_config(shard_shape, cores, strategy):
    spec = ttnn.NdShardSpec(ttnn.Shape(shard_shape), cores, ttnn.ShardOrientation.ROW_MAJOR, strategy)
    return ttnn.MemoryConfig(L1, spec)


def width_config(buffer_type, num_shards, shard_shape):
    # Shards in rows of 8 cores, or on DRAM banks 0 .. num_shards - 1.
    if buffer_type == L1 and num_shards > 8:
        assert num_shards % 8 == 0
        cores = grid(((0, 0), (7, num_shards // 8 - 1)))
    else:
        cores = grid(((0, 0), (num_shards - 1, 0)))
    spec = ttnn.ShardSpec(cores, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(WIDTH, buffer_type, spec)


def ramp(shape, period=97):
    # Multiples of 1/8 in [-6, 6], exact in bfloat16, with a period that does not divide the row lengths used here.
    n = torch.Size(shape).numel()
    return ((torch.arange(n) % period - period // 2) / 8).reshape(shape).to(torch.bfloat16)


def dram_shard_width(device):
    # bfloat16 values per DRAM shard row: a multiple of 16 B, but not of the DRAM alignment.
    return 8 if device.arch() == ttnn.device.Arch.WORMHOLE_B0 else 16


def shard_row_bytes(memory_config):
    if memory_config.shard_spec is not None:
        return memory_config.shard_spec.shape[1] * 2
    return memory_config.nd_shard_spec.shard_shape[-1] * 2


def alignment(memory_config):
    if memory_config.buffer_type == DRAM:
        return 32 if ttnn.get_arch_name() == "wormhole_b0" else 64
    return 16


def run_repeat(device):
    t = ttnn.from_torch(ramp([1, 1, 4, 24]), layout=RM, device=device, memory_config=width_config(L1, 2, [4, 12]))
    return ttnn.repeat(t, ttnn.Shape([1, 1, 2, 1])), ttnn.to_torch(t).repeat(1, 1, 2, 1)


def run_slice_nd(device):
    cores = grid(((0, 0), (7, 2)), ((0, 3), (3, 3)))
    config = nd_config([1, 1, 5, 64], cores, CONTIGUOUS)
    t = ttnn.from_torch(ramp([1, 1, 32, 224]), layout=RM, device=device, memory_config=config)
    return ttnn.slice(t, [0, 0, 16, 0], [1, 1, 32, 32], [1, 1, 2, 2]), ttnn.to_torch(t)[:, :, 16:32:2, 0:32:2]


def run_slice_into_preallocated_nd(device):
    # The staged output is copied into the caller's preallocated tensor.
    t = ttnn.from_torch(ramp([1, 1, 32, 64]), layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    config = nd_config([1, 1, 16, 4], grid(((0, 0), (7, 0))), CONTIGUOUS)
    out = ttnn.from_torch(
        torch.zeros(1, 1, 16, 32, dtype=torch.bfloat16), layout=RM, device=device, memory_config=config
    )
    ttnn.slice(t, [0, 0, 8, 16], [1, 1, 24, 48], [1, 1, 1, 1], output_tensor=out)
    return out, ttnn.to_torch(t)[:, :, 8:24, 16:48]


def run_permute_nd(device):
    cores = grid(((0, 0), (6, 5)), ((0, 6), (5, 6)))
    config = nd_config([1, 1, 2, 80], cores, ROUND_ROBIN)
    t = ttnn.from_torch(ramp([1, 1, 24, 160]), layout=RM, device=device, memory_config=config)
    return ttnn.permute(t, [0, 1, 3, 2]), ttnn.to_torch(t).permute(0, 1, 3, 2)


def run_permute_width_l1(device):
    t = ttnn.from_torch(ramp([2, 2, 32, 64]), layout=RM, device=device, memory_config=width_config(L1, 16, [128, 4]))
    return ttnn.permute(t, (2, 0, 1, 3)), ttnn.to_torch(t).permute(2, 0, 1, 3)


def run_transpose_width_l1(device):
    t = ttnn.from_torch(ramp([1, 1, 32, 64]), layout=RM, device=device, memory_config=width_config(L1, 16, [32, 4]))
    return ttnn.transpose(t, 2, 3), ttnn.to_torch(t).transpose(2, 3)


def run_transpose_width_dram(device):
    e = dram_shard_width(device)
    config = width_config(DRAM, 32 // e, [32, e])
    t = ttnn.from_torch(ramp([1, 1, 32, 32]), layout=RM, device=device, memory_config=config)
    return ttnn.transpose(t, 2, 3), ttnn.to_torch(t).transpose(2, 3)


def run_permute_into_width_dram(device):
    e = dram_shard_width(device)
    want = ramp([1, 1, 64, 33])
    src = ttnn.from_torch(
        want.permute(3, 0, 1, 2).contiguous(), layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    config = width_config(DRAM, -(-33 // e), [64, e])
    return ttnn.permute(src, (1, 2, 3, 0), memory_config=config), want


def run_indexed_fill_width_dram(device):
    # indexed_fill along W ends with a permute into input_a's memory config.
    e = dram_shard_width(device)
    x = ramp([1, 1, 64, 33])
    b_host = -(torch.arange(64) + 1).reshape(1, 1, 64, 1).to(torch.bfloat16)
    a = ttnn.from_torch(x, layout=RM, device=device, memory_config=width_config(DRAM, -(-33 // e), [64, e]))
    b = ttnn.from_torch(b_host, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    index = ttnn.from_torch(
        torch.tensor([3], dtype=torch.int32),
        dtype=ttnn.int32,
        layout=RM,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    want = ttnn.to_torch(a).clone()
    want[..., 3:4] = b_host
    return ttnn.indexed_fill(index, a, b, dim=3), want


def run_pad_into_width_dram(device):
    e = dram_shard_width(device)
    x = ramp([1, 1, 2, 2 * e - 4])
    t = ttnn.from_torch(x, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    y = ttnn.pad(t, [1, 1, 2, 2 * e], [0, 0, 0, 0], 1.0, memory_config=width_config(DRAM, 2, [2, e]))
    return y, torch.nn.functional.pad(x, (0, 4), value=1.0)


def run_gather_nd(device):
    config = nd_config([1, 1, 32, 4], grid(((0, 0), (7, 1))), CONTIGUOUS)
    t = ttnn.from_torch(ramp([1, 1, 32, 64]), layout=RM, device=device, memory_config=config)
    index = torch.randint(0, 64, (1, 1, 32, 64), generator=torch.Generator().manual_seed(0), dtype=torch.int32)
    i = ttnn.from_torch(index, dtype=ttnn.uint32, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # The output takes the index's interleaved config, so check the input's config instead.
    return ttnn.gather(t, -1, index=i), torch.gather(ttnn.to_torch(t), -1, index.long()), t


def run_fold_nd(device):
    # Fold synthesizes an output shard of ceil(4 * C / grid width) values per row; pick C so that row is sub-aligned.
    grid_width = device.compute_with_storage_grid_size().x
    c = next(c for c in range(3, 64) if (-(-4 * c // grid_width) * 2) % 16 != 0 and 4 * c > grid_width)
    shape = (1, 4, 12, c)
    t = ttnn.from_torch(
        ramp(shape, 29),
        layout=RM,
        device=device,
        memory_config=nd_config(list(shape), grid(((0, 0), (0, 0))), CONTIGUOUS),
    )
    n, h, w, _ = shape
    want = (
        ttnn.to_torch(t).reshape(n, h // 2, 2, w // 2, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(n, h // 2, w // 2, 4 * c)
    )
    return ttnn.fold(t, 2, 2), want


@pytest.mark.parametrize(
    "run",
    [
        run_repeat,
        run_slice_nd,
        run_slice_into_preallocated_nd,
        run_permute_nd,
        run_permute_width_l1,
        run_transpose_width_l1,
        run_transpose_width_dram,
        run_permute_into_width_dram,
        run_indexed_fill_width_dram,
        run_pad_into_width_dram,
        run_fold_nd,
        run_gather_nd,
    ],
    ids=lambda run: run.__name__[len("run_") :],
)
def test_subaligned_shard_rows(device, run):
    output, want, *sharded = run(device)
    config = (sharded[0] if sharded else output).memory_config()
    # The case only covers item 17 if the sharded tensor really has rows split across sub-aligned shard rows.
    assert config.is_sharded() and config.memory_layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED, config
    assert shard_row_bytes(config) % alignment(config) != 0, config
    got = ttnn.to_torch(output)
    assert got.shape == want.shape
    wrong = int((got != want).sum())
    assert wrong == 0, f"{wrong} of {want.numel()} values wrong ({config})"
