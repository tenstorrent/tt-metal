# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the self-contained Python all_gather CCL op.

all_gather is PURE DATA MOVEMENT (identity gather, no arithmetic): every device's
output is the host-side concatenation of all N shards along ``gather_dim``. The
transfer is a bit-for-bit byte copy, so the only error present is the dtype
quantization that already happened at ``from_torch`` (before the op ran) — the
gather itself adds nothing. The expectation is therefore PCC ≈ 1.0 and ~0 error
for every dtype.

This file measures PCC, max abs error, mean abs error, and relative RMS error
across several shard shapes × {bf16, f32} on the proven primary case
(gather_dim=0, TILE_LAYOUT, Linear).

Run it the same way as the acceptance suite — on the deterministic WH sim:

    scripts/run_multidevice_sim_pytest.py --op all_gather -- \
        tests/ttnn/unit_tests/operations/all_gather/test_all_gather_precision_baseline.py -v

The mesh shape (1, 8) + FABRIC_1D MUST match the sim's mesh-graph descriptor
(else fabric init hangs: "Fabric Router Sync: Timeout").
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose

from ttnn.operations.all_gather import all_gather


PCC = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.99,  # block-float: shared-exponent quantization at from_torch
}

# small / multi-tile / non-square / one larger — per-device SHARD shapes.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile (small)
    (1, 1, 64, 128),  # multi-tile
    (1, 1, 96, 64),  # non-square, tile-aligned
    (1, 1, 256, 256),  # larger
]

DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _torch_dtype(dtype):
    # bf8b has no native torch dtype; reference it in bf16 (from_torch quantizes to
    # block-float). f32 references in f32; bf16/bf8b in bf16.
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _make_sharded_input(mesh_device, shard_shape, dtype):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32).to(_torch_dtype(dtype))

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, torch_full


def _metrics(golden, calculated):
    g = golden.to(torch.float32)
    c = calculated.to(torch.float32)
    abs_err = torch.abs(g - c)
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    denom = torch.sqrt(torch.mean(g * g)).item()
    rms = torch.sqrt(torch.mean((g - c) ** 2)).item()
    rel_rms = rms / denom if denom > 0 else rms
    return max_abs, mean_abs, rel_rms


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_gather_precision_baseline(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = all_gather(input_tensor, 0, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    # Every device holds the full concat; report metrics from device 0 (all equal).
    dev0 = output_shards[0]
    assert tuple(dev0.shape) == tuple(torch_full.shape)

    max_abs, mean_abs, rel_rms = _metrics(torch_full, dev0)
    _, allclose_msg = comp_allclose(torch_full, dev0)

    logger.info(
        f"[precision] all_gather {dtype} shard={shard_shape} full={tuple(torch_full.shape)}: "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_rms={rel_rms:.6g} | {allclose_msg}"
    )

    # All N devices must agree bit-for-bit (replicated output) and match the oracle.
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(torch_full.shape), f"device {dev_idx} shape mismatch"
        assert_with_pcc(torch_full, dev_out, pcc)
