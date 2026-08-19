# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the reduce_scatter CCL op.

Measures PCC, max/mean absolute error, and relative RMS error of the
sum-then-slice output across the line mesh, over a small shape sweep and both
supported dtypes. reduce_scatter's output is PER-DEVICE DISTINCT (device i holds
slice i of the sum), so the metrics are computed per device and the WORST device
is reported — a schedule bug that mis-slices a single device must not hide
behind a healthy mean.

This is a MULTI-DEVICE op — drive it via the deterministic multi-device runner
(the mesh shape MUST match the active topology or fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_precision_baseline.py -v

The oracle accumulates in fp32 then casts to the tensor dtype, so the reference
is not itself limited by bf16 rounding — the measured error reflects the device
reduction's own accumulation budget (HiFi4 + fp32_dest_acc_en). Metrics are
logged per case for the verification report; PCC + a loose allclose assert keep
the test a real gate.
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter

# The verification topology's mesh shape — (1, 4) is the bh_quietbox_1x4_hw
# hardware contract; the multidevice runner overrides via MULTIDEV_SIM_MESH_SHAPE.
MESH_SHAPE = tuple(int(x) for x in os.environ.get("MULTIDEV_SIM_MESH_SHAPE", "1,4").split(","))

# PCC floor per dtype: a bf16 sum of N terms accumulates rounding (0.99), float32
# is tight (0.999). These are the acceptance/golden thresholds.
PCC = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

# small, medium (multi-tile rows+cols), larger, multi-batch — all tile-aligned,
# widths multiples of 256 so the per-device slice is tile-aligned for N in {4, 8}.
SHARD_SHAPES = [
    (1, 1, 32, 256),
    (1, 1, 64, 512),
    (1, 1, 256, 256),
    (2, 1, 64, 256),
]

DTYPES = [ttnn.bfloat16, ttnn.float32]

SCATTER_DIM = 3

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    # Accumulate the oracle in fp32 then cast, so the reference is not itself
    # limited by bf16 rounding; then slice: device i's expected output.
    summed = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        summed = summed.to(torch.bfloat16)
    oracle_slices = torch.chunk(summed, num_devices, dim=SCATTER_DIM)

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


def _error_metrics(oracle_f32, actual_f32):
    """PCC-independent scalar error metrics (all computed in fp32)."""
    diff = (actual_f32 - oracle_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = torch.sqrt(torch.mean(oracle_f32**2)).item()
    rel_rms = (torch.sqrt(torch.mean((actual_f32 - oracle_f32) ** 2)).item() / denom) if denom > 0 else 0.0
    return max_abs, mean_abs, rel_rms


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter_precision_baseline(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter(input_tensor, dim=SCATTER_DIM, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices

    # Per-device metrics — the output is DISTINCT per device; report the worst.
    worst = {"pcc": 1.0, "max_abs": 0.0, "mean_abs": 0.0, "rel_rms": 0.0}
    for dev_idx, dev_out in enumerate(output_shards):
        oracle_f32 = oracle_slices[dev_idx].to(torch.float32)
        actual_f32 = dev_out.to(torch.float32)
        max_abs, mean_abs, rel_rms = _error_metrics(oracle_f32, actual_f32)
        _, pcc_val = comp_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])
        worst["pcc"] = min(worst["pcc"], float(pcc_val))
        worst["max_abs"] = max(worst["max_abs"], max_abs)
        worst["mean_abs"] = max(worst["mean_abs"], mean_abs)
        worst["rel_rms"] = max(worst["rel_rms"], rel_rms)

    _, allclose_str = comp_allclose(oracle_slices[0], output_shards[0])
    logger.info(
        f"[precision] reduce_scatter dtype={dtype} shard={shard_shape} N={num_devices} dim={SCATTER_DIM}: "
        f"worst-device pcc={worst['pcc']:.7f} max_abs={worst['max_abs']:.6f} "
        f"mean_abs={worst['mean_abs']:.6f} rel_rms={worst['rel_rms']:.6f} | dev0 {allclose_str}"
    )

    for dev_idx, dev_out in enumerate(output_shards):
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])
