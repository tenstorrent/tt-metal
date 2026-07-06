# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the all_reduce CCL op.

Measures PCC, max/mean absolute error, and relative RMS error of the element-wise
SUM across the (1, 8) line mesh, over a small shape sweep and both supported
dtypes. This is a MULTI-DEVICE op — drive it via the deterministic craq-sim
runner (mesh MUST be (1, 8) with FABRIC_1D or fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op all_reduce -- \
        tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_precision_baseline.py -v

The oracle accumulates in fp32 then casts to the tensor dtype, so the reference
is not itself limited by bf16 rounding — the measured error reflects the device
reduction's own accumulation budget. Metrics are logged per case for the
verification report; PCC + a loose allclose assert keep the test a real gate.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_reduce import all_reduce


# PCC floor per dtype: a bf16 sum of N terms accumulates rounding (0.99), float32
# is tight (0.999). These are the acceptance/golden thresholds.
PCC = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

# small, medium, larger (multi-tile wide), multi-batch — all tile-aligned.
SHARD_SHAPES = [
    (1, 1, 32, 32),
    (1, 1, 64, 128),
    (1, 1, 256, 256),
    (2, 1, 32, 64),
]

DTYPES = [ttnn.bfloat16, ttnn.float32]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    # Accumulate the oracle in fp32 then cast, so the reference is not itself
    # limited by bf16 rounding.
    oracle = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        oracle = oracle.to(torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle


def _error_metrics(oracle_f32, actual_f32):
    """PCC-independent scalar error metrics (all computed in fp32)."""
    diff = (actual_f32 - oracle_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = torch.sqrt(torch.mean(oracle_f32**2)).item()
    rel_rms = (torch.sqrt(torch.mean((actual_f32 - oracle_f32) ** 2)).item() / denom) if denom > 0 else 0.0
    return max_abs, mean_abs, rel_rms


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_reduce_precision_baseline(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_reduce requires at least 2 mesh devices")

    input_tensor, oracle = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = all_reduce(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    # Every device holds the identical sum — measure on device 0, sanity-check the rest.
    oracle_f32 = oracle.to(torch.float32)
    dev0 = output_shards[0].to(torch.float32)
    max_abs, mean_abs, rel_rms = _error_metrics(oracle_f32, dev0)
    _, allclose_str = comp_allclose(oracle, output_shards[0])
    _, pcc_val = comp_pcc(oracle, output_shards[0], PCC[dtype])

    logger.info(
        f"[precision] all_reduce dtype={dtype} shard={shard_shape} N={num_devices}: "
        f"pcc={pcc_val} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rel_rms={rel_rms:.6f} | {allclose_str}"
    )

    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(shard_shape)
        assert_with_pcc(oracle, dev_out, PCC[dtype])
