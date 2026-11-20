# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""API probes of the KV store piece (what the mock can answer)."""

import pytest
import torch
from loguru import logger

import ttnn

pytestmark = pytest.mark.use_module_device


def _nd(device, width, rows=32):
    banks = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, rows, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def test_probe_tensors(device):
    raw = torch.randint(0, 256, (6, 1, 256, 656), dtype=torch.uint8)
    for name, make in {
        "uint8 from_torch": lambda: ttnn.from_torch(
            raw, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=_nd(device, 656)
        ),
        "fp8 allocate": lambda: ttnn.allocate_tensor_on_device(
            ttnn.Shape([6, 1, 256, 656]), ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT, device, _nd(device, 656)
        ),
        "fp8 from_torch": lambda: ttnn.from_torch(
            raw.view(torch.float8_e4m3fn),
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_nd(device, 656),
        ),
        "bfp8 allocate": lambda: ttnn.allocate_tensor_on_device(
            ttnn.Shape([6, 1, 256, 128]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, _nd(device, 128)
        ),
        "uint8 key twin": lambda: ttnn.from_torch(
            torch.zeros(6, 1, 32, 1088, dtype=torch.uint8),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_nd(device, 1088, rows=4),
        ),
        "uint32 l1": lambda: ttnn.from_torch(
            torch.full((1, 1, 16, 128), -1, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    (16, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        ),
    }.items():
        try:
            t = make()
            logger.info(
                f"HY4_MK10 probe {name}: shape {tuple(t.shape)} dtype {t.dtype} layout {t.layout} page {t.buffer_aligned_page_size()} pages {t.buffer_num_pages()} addr {t.buffer_address():#x}"
            )
        except Exception as e:
            logger.info(f"HY4_MK10 probe {name}: FAILED {type(e).__name__}: {str(e)[:300]}")
    try:
        t = ttnn.from_torch(
            torch.arange(16 * 128, dtype=torch.int32).reshape(1, 1, 16, 128),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))]),
                    (16, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )
        cb = ttnn.cb_descriptor_from_sharded_tensor(3, t)
        logger.info(
            f"HY4_MK10 probe cb from sharded: total {cb.total_size} formats {[(f.buffer_index, f.page_size) for f in cb.format_descriptors]}"
        )
    except Exception as e:
        logger.info(f"HY4_MK10 probe cb from sharded: FAILED {type(e).__name__}: {str(e)[:300]}")


@pytest.mark.parametrize("frames", [757, 1353])
def test_probe_pool_allocation(device, frames):
    """KV_STORE.md 9, question 7, as far as the mock can answer it: does the allocator take a pool of 757 frames
    (6.2 M tokens: 59,046 latent slots) or 1,353 frames (11.08 M) as ONE tensor per kind? (The mock keeps the real
    allocator; nothing is written.)"""
    from models.experimental.hy4_preview.fused.kv_store import device as dv
    from models.experimental.hy4_preview.fused.kv_store import frames as fr

    for kind in fr.KINDS:
        g = fr.Geometry.pool(kind)
        try:
            t = dv.allocate(device, g, frames)
            logger.info(
                f"HY4_MK10 probe pool {kind.name} {frames} frames: shape {tuple(t.shape)}, {t.buffer_num_pages()} pages, "
                f"{g.bytes_per_bank(frames) / 1e9:.3f} GB per bank, address {t.buffer_address():#x}"
            )
        except Exception as e:
            logger.info(f"HY4_MK10 probe pool {kind.name} {frames} frames: FAILED {type(e).__name__}: {str(e)[:300]}")
