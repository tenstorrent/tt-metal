# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the reduce_scatter CCL op.

Measures PCC, max/mean absolute error, relative RMS error, and max error in
output-dtype ULPs of the element-wise SUM-then-slice across the mesh line, over
a small shape sweep and both supported dtypes. Unlike all_reduce the output is
per-device DISTINCT (device i holds slice i of the sum), so metrics are
aggregated worst-case across all devices' slices.

This is a MULTI-DEVICE op — drive it via the deterministic multi-device runner
(the mesh shape MUST match the selected topology or fabric init hangs):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_precision_baseline.py -v

The committed default mesh is the (1, 8) grade line; CCL_HW_MESH_SHAPE (set by
the runner from the topology, e.g. "1,4" on a 4-chip Blackhole QuietBox)
overrides it — same pattern as the golden suite. Shard widths are multiples of
256 so the per-device slice stays tile-aligned for N in {4, 8}.

The oracle accumulates in fp32 then casts to the tensor dtype, so the reference
is not itself limited by bf16 rounding — the measured error reflects the device
reduction's own accumulation budget (HiFi4 + fp32_dest_acc_en, with a bf16 pack
rounding per accumulate pass under bf16, see op_design.md R16).
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

# PCC floor per dtype: a bf16 sum of N terms accumulates rounding (0.99),
# float32 is tight (0.999). These are the acceptance/golden thresholds.
PCC = {
    ttnn.bfloat16: 0.99,
    ttnn.float32: 0.999,
}

# Machine epsilon of the OUTPUT dtype, for the ULP metric (bf16: 7 explicit
# mantissa bits; fp32: 23).
EPS = {
    ttnn.bfloat16: 2.0**-7,
    ttnn.float32: 2.0**-23,
}

TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# small (single-tile slice on N=8), medium, larger (the worst Phase-0 L1 case:
# S=32 resident accumulator tiles on N=4), multi-batch. All widths are
# multiples of 256 = lcm(tile, 8) so the slice is tile-aligned for N in {4, 8}.
SHARD_SHAPES = [
    (1, 1, 32, 256),
    (1, 1, 64, 512),
    (1, 1, 256, 512),
    (2, 1, 64, 256),
]

DTYPES = [ttnn.bfloat16, ttnn.float32]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _hw_mesh_shape(default):
    """Mesh shape, overridable for smaller boxes (same pattern as test_golden.py)."""
    raw = os.environ.get("CCL_HW_MESH_SHAPE")
    return tuple(int(x) for x in raw.split(",")) if raw else default


def _make_sharded_input(mesh_device, shard_shape, dtype, dim=3):
    """N distinct same-shape shards sharded across the line; oracle_slices[i] is
    slice i of the fp32-accumulated element-wise sum of the quantized shards."""
    num_devices = prod(tuple(mesh_device.shape))
    torch_dtype = TORCH_DTYPE[dtype]
    torch.manual_seed(42)
    full = torch.randn((num_devices * shard_shape[0], *shard_shape[1:]), dtype=torch.float32)
    quantized = full.to(torch_dtype)

    summed = quantized.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(torch_dtype)
    oracle_slices = list(torch.chunk(summed, num_devices, dim=dim))

    input_tensor = ttnn.from_torch(
        quantized,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, oracle_slices


def _error_metrics(oracle_f32, actual_f32, eps):
    """Scalar error metrics, all computed in fp32.

    max_ulp measures |err| in units of the output dtype's spacing AT the
    oracle's magnitude: ulp(x) = eps * 2^floor(log2(|x|)), with |x| floored at
    1.0 so near-zero oracle values measure error in plain eps units instead of
    dividing by a vanishing spacing.
    """
    diff = (actual_f32 - oracle_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    denom = torch.sqrt(torch.mean(oracle_f32**2)).item()
    rel_rms = (torch.sqrt(torch.mean(diff**2)).item() / denom) if denom > 0 else 0.0
    ulp_size = eps * torch.exp2(torch.floor(torch.log2(oracle_f32.abs().clamp(min=1.0))))
    max_ulp = (diff / ulp_size).max().item()
    return max_abs, mean_abs, rel_rms, max_ulp


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape((1, 8))], indirect=True)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_reduce_scatter_precision_baseline(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("reduce_scatter requires at least 2 mesh devices")

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)

    output_tensor = reduce_scatter(input_tensor, dim=3, topology=topology)
    ttnn.synchronize_device(mesh_device)

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]
    assert len(output_shards) == num_devices

    expected_shape = list(shard_shape)
    expected_shape[3] //= num_devices

    # Per-device DISTINCT slices: aggregate the worst case across devices.
    worst = {"max_abs": 0.0, "mean_abs": 0.0, "rel_rms": 0.0, "max_ulp": 0.0, "pcc": 1.0}
    for dev, (oracle, dev_out) in enumerate(zip(oracle_slices, output_shards)):
        assert tuple(dev_out.shape) == tuple(expected_shape)
        max_abs, mean_abs, rel_rms, max_ulp = _error_metrics(
            oracle.to(torch.float32), dev_out.to(torch.float32), EPS[dtype]
        )
        _, pcc_val = comp_pcc(oracle, dev_out, PCC[dtype])
        worst["max_abs"] = max(worst["max_abs"], max_abs)
        worst["mean_abs"] = max(worst["mean_abs"], mean_abs)
        worst["rel_rms"] = max(worst["rel_rms"], rel_rms)
        worst["max_ulp"] = max(worst["max_ulp"], max_ulp)
        worst["pcc"] = min(worst["pcc"], float(pcc_val))
        assert_with_pcc(oracle, dev_out, PCC[dtype])

    _, allclose_str = comp_allclose(oracle_slices[0], output_shards[0])
    logger.info(
        f"[precision] reduce_scatter dtype={dtype} shard={shard_shape} N={num_devices}: "
        f"worst pcc={worst['pcc']:.7f} max_abs={worst['max_abs']:.6f} "
        f"mean_abs={worst['mean_abs']:.6f} rel_rms={worst['rel_rms']:.6f} "
        f"max_ulp={worst['max_ulp']:.1f} | dev0 {allclose_str}"
    )
