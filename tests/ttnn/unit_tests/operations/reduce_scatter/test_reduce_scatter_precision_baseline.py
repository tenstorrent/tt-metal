# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the reduce_scatter CCL op (CCL + compute).

Measures PCC, max/mean absolute error, relative RMS error, and max ULP error of
the element-wise SUM-then-SLICE across the (1, 4) Blackhole line mesh, over a
small shape sweep and both supported dtypes. Unlike all_reduce the output is
PER-DEVICE DISTINCT (device i holds slice i of the sum), so the metrics are
computed per device against that device's own oracle slice and the WORST device
is reported — a slice-addressing bug shows up as one catastrophically-bad device,
not a uniform precision shift.

This is a MULTI-DEVICE op — drive it via the deterministic multi-device runner
(mesh MUST be (1, 4) with FABRIC_1D or fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_precision_baseline.py -v

The oracle accumulates in fp32 then casts to the tensor dtype, so the reference
is not itself limited by bf16 rounding — the measured error reflects the device
reduction's own accumulation budget (HiFi4 + fp32_dest_acc_en in Phase B).
Metrics are logged per case for the verification report; PCC asserts keep the
test a real gate.
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter


# PCC floor per dtype: a bf16 sum of N terms accumulates rounding (0.99), float32
# is tight (0.999). These are the acceptance/golden thresholds.
PCC = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

# Explicit mantissa bits for the ULP metric.
_MANTISSA_BITS = {
    ttnn.bfloat16: 7,
    ttnn.float32: 23,
}

_SCATTER_DIM = 3

# small, medium, larger (multi-tile rows AND columns), multi-batch — all with
# shape[3] a multiple of num_devices * 32 = 128 on the (1, 4) contract mesh.
SHARD_SHAPES = [
    (1, 1, 32, 128),
    (1, 1, 64, 256),
    (1, 1, 256, 512),
    (2, 1, 32, 256),
]

DTYPES = [ttnn.bfloat16, ttnn.float32]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

# The verification mesh shape — the bh_quietbox_1x4_hw contract. Do NOT change.
MESH_SHAPE = (1, 4)


def _make_sharded_input(mesh_device, shard_shape, dtype):
    """Shard a fresh full tensor along dim 0; return input + per-device oracle slices."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32)

    # Accumulate the oracle in fp32 then cast, so the reference is not itself
    # limited by bf16 rounding.
    summed = torch_full.reshape(num_devices, *shard_shape).sum(dim=0)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)
        summed = summed.to(torch.bfloat16)
    oracle_slices = torch.chunk(summed, num_devices, dim=_SCATTER_DIM)

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


def _error_metrics(oracle_f32, actual_f32, mantissa_bits):
    """PCC-independent scalar error metrics (all computed in fp32)."""
    diff = (actual_f32 - oracle_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = torch.sqrt(torch.mean(oracle_f32**2)).item()
    rel_rms = (torch.sqrt(torch.mean(diff**2)).item() / denom) if denom > 0 else 0.0
    # ULP at TENSOR SCALE: spacing from the element magnitude clamped below at the
    # oracle RMS, so near-zero elements (whose absolute error comes from the
    # quantization of the O(1) addends, not of the tiny result) don't blow the
    # metric up into the millions. This measures "how many representable steps at
    # the tensor's working magnitude" the worst element is off by.
    scale = oracle_f32.abs().clamp(min=max(denom, 2.0**-20))
    ulp = torch.pow(torch.tensor(2.0), torch.floor(torch.log2(scale)) - mantissa_bits)
    max_ulp = (diff / ulp).max().item()
    return max_abs, mean_abs, rel_rms, max_ulp


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter_precision_baseline(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter(input_tensor, dim=_SCATTER_DIM, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    # Per-device DISTINCT outputs: measure every device against ITS slice, report the worst.
    worst = {"pcc": 1.0, "max_abs": 0.0, "mean_abs": 0.0, "rel_rms": 0.0, "max_ulp": 0.0}
    for dev_idx, dev_out in enumerate(output_shards):
        oracle_f32 = oracle_slices[dev_idx].to(torch.float32)
        actual_f32 = dev_out.to(torch.float32)
        max_abs, mean_abs, rel_rms, max_ulp = _error_metrics(oracle_f32, actual_f32, _MANTISSA_BITS[dtype])
        _, pcc_val = comp_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])
        worst["pcc"] = min(worst["pcc"], float(pcc_val))
        worst["max_abs"] = max(worst["max_abs"], max_abs)
        worst["mean_abs"] = max(worst["mean_abs"], mean_abs)
        worst["rel_rms"] = max(worst["rel_rms"], rel_rms)
        worst["max_ulp"] = max(worst["max_ulp"], max_ulp)

    _, allclose_str = comp_allclose(oracle_slices[0], output_shards[0])
    logger.info(
        f"[precision] reduce_scatter dtype={dtype} shard={shard_shape} N={num_devices} "
        f"(worst device): pcc={worst['pcc']:.7f} max_abs={worst['max_abs']:.6f} "
        f"mean_abs={worst['mean_abs']:.6f} rel_rms={worst['rel_rms']:.6f} "
        f"max_ulp={worst['max_ulp']:.1f} | dev0 {allclose_str}"
    )

    expected_shape = list(shard_shape)
    expected_shape[_SCATTER_DIM] //= num_devices
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(expected_shape)
        assert_with_pcc(oracle_slices[dev_idx], dev_out, PCC[dtype])
