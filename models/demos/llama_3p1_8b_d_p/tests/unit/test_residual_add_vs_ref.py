# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The exact BF16 residual operation is a prerequisite for decoder hardware tests."""

import pytest
import torch
from loguru import logger

import ttnn


def _addresses(tensor):
    return tuple(int(shard.buffer_address()) for shard in ttnn.get_device_tensors(tensor))


def _operands(variant):
    # Adjacent BF16 mantissas plus half-ULP increments exercise both nearest-even tie directions.
    a = torch.tensor([1, 1 + 1 / 128, -1, -1 - 1 / 128, 2, -2, 0, 0], dtype=torch.bfloat16)
    b = torch.tensor([1 / 256, 1 / 256, -1 / 256, -1 / 256, -2, 2, 0, 1], dtype=torch.bfloat16)
    rows = torch.arange(1024).reshape(1024, 1)
    cols = torch.arange(4096).reshape(1, 4096)
    indices = (rows * 3 + cols + variant) % len(a)
    # Powers of two retain exact ties while making SP rows and changed calls distinguishable.
    scale = 2.0 ** ((rows // 256 + cols // 512 + variant) % 5 - 2)
    return [(values[indices] * scale).bfloat16().reshape(1, 1, 1024, 4096) for values in (a, b)]


def _upload(mesh_device, host):
    return ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(4, 8), dims=(2, None)),
    )


# Exact sums on every chip catch approximate addition, wrong BF16 tie rounding, input aliasing,
# stale cached addresses and accidental SP movement before either decoder residual can hide them.
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_residual_add_exact_bf16_and_cached_replay(mesh_device):
    assert tuple(mesh_device.shape) == (4, 8)
    assert mesh_device.get_num_devices() == 32
    mesh_device.enable_program_cache()
    retained = []
    entries = None
    first_addresses = None
    try:
        for variant in (0, 1, 2, 0):
            host_a, host_b = _operands(variant)
            expected = (host_a.float() + host_b.float()).bfloat16()
            a, b = _upload(mesh_device, host_a), _upload(mesh_device, host_b)
            result = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, fast_and_approximate_mode=False)
            ttnn.synchronize_device(mesh_device)
            assert tuple(result.shape) == (1, 1, 256, 4096)
            assert result.dtype == ttnn.bfloat16
            assert result.layout == ttnn.TILE_LAYOUT
            assert result.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            addresses = [_addresses(tensor) for tensor in (a, b, result)]
            for chip in range(32):
                assert len({group[chip] for group in addresses}) == 3
                if first_addresses is not None:
                    assert set(group[chip] for group in addresses).isdisjoint(first_addresses[chip])
                sp = chip // 8
                rows = slice(sp * 256, (sp + 1) * 256)
                for tensor, host in ((a, host_a), (b, host_b), (result, expected)):
                    actual = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip])
                    assert torch.isfinite(actual).all()
                    assert torch.equal(actual, host[:, :, rows]), f"variant={variant}, chip={chip}"
            count = mesh_device.num_program_cache_entries()
            if entries is None:
                entries = count
                assert entries > 0
                first_addresses = [set(group[chip] for group in addresses) for chip in range(32)]
                # Retain the actual first inputs/output, making address separation deterministic.
                retained.extend((a, b, result))
            else:
                assert count == entries
                for tensor in (a, b, result):
                    tensor.deallocate(True)
            logger.info(f"residual variant={variant}: exact on 32 chips; programs={count}; addresses={addresses}")
    finally:
        for tensor in retained:
            tensor.deallocate(True)
