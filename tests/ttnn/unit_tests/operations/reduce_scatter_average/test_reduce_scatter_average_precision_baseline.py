# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for reduce_scatter_average (verifier-authored).

Measures worst-device PCC, max/mean absolute error, relative RMS error, and max
ULP error against an fp32-accumulated torch mean-then-slice oracle, across four
shard shapes x two dtypes. The oracle quantizes the shards to the device dtype
FIRST (what the device actually sees), then accumulates the mean in fp32, so
the reference is not itself limited by bf16 rounding — the measured error is
the op's own (N-term accumulate + 1/N scale + output quantization).

Mesh-shape-adaptive like the debug mirror: sizes the mesh from CCL_HW_MESH_SHAPE
(default (1, 4) — the bh_quietbox_1x4_hw topology). Drive via:

    scripts/run_multidevice_sim_pytest.py --runtime hardware \
        --op reduce_scatter_average -- <this file>
"""

import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter_average import reduce_scatter_average

MESH_SHAPE = tuple(int(x) for x in os.environ.get("CCL_HW_MESH_SHAPE", "1,4").split(","))

# Golden-suite thresholds (bf16 accumulates N-term rounding + the 1/N scale).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}
ALLCLOSE_ATOL = {
    ttnn.float32: 0.02,
    ttnn.bfloat16: 0.1,
}
TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}

# Small (single output tile), medium square, non-square wide, multi-batch.
SHARD_SHAPES = [
    (1, 1, 32, 256),
    (1, 1, 256, 256),
    (1, 1, 64, 512),
    (2, 1, 32, 256),
]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)


def _max_ulp_at_scale(out_f32: torch.Tensor, ref_f32: torch.Tensor, torch_dtype) -> float:
    """Max |out - ref| in units of the ULP spacing at the TENSOR's max magnitude.

    Per-element ULP error explodes meaninglessly where the reference passes
    near zero (~1e9 representable floats between +/-1e-4), so measure against
    the spacing at the tensor's scale instead: 1.0 here means the worst error
    equals one representable step at the output's largest value."""
    fi = torch.finfo(torch_dtype)
    scale = ref_f32.abs().max().clamp_min(fi.smallest_normal)
    ulp_at_scale = torch.exp2(torch.floor(torch.log2(scale))) * fi.eps
    return ((out_f32 - ref_f32).abs().max() / ulp_at_scale).item()


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_precision_baseline(mesh_device, topology, dtype, shard_shape):
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("requires at least 2 mesh devices")

    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])
    torch.manual_seed(1234)
    torch_full = torch.randn(full_shape, dtype=torch.float32)
    if dtype == ttnn.bfloat16:
        torch_full = torch_full.to(torch.bfloat16)  # quantize BEFORE the oracle

    shards = torch_full.reshape(num_devices, *shard_shape).to(torch.float32)
    mean = shards.mean(dim=0)  # fp32-accumulated
    oracle_slices = list(mean.chunk(num_devices, dim=3))

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)

    output_tensor = reduce_scatter_average(input_tensor, topology=topology)
    ttnn.synchronize_device(mesh_device)

    torch_dtype = TORCH_DTYPE[dtype]
    worst = {"pcc": 1.0, "max_abs": 0.0, "mean_abs": 0.0, "rel_rms": 0.0, "max_ulp": 0.0}
    for dev_idx, t in enumerate(ttnn.get_device_tensors(output_tensor)):
        out = ttnn.to_torch(t).to(torch.float32)
        ref = oracle_slices[dev_idx].to(torch_dtype).to(torch.float32)

        # Mandated helpers: PCC assertion + allclose deltas.
        _, pcc_val = assert_with_pcc(ref, out, PCC[dtype])
        passed_allclose, allclose_msg = comp_allclose(ref, out, rtol=0.05, atol=ALLCLOSE_ATOL[dtype])
        assert passed_allclose, f"device {dev_idx}: {allclose_msg}"

        diff = (out - ref).abs()
        rel_rms = (torch.linalg.vector_norm(out - ref) / torch.linalg.vector_norm(ref)).item()
        worst["pcc"] = min(worst["pcc"], float(pcc_val))
        worst["max_abs"] = max(worst["max_abs"], diff.max().item())
        worst["mean_abs"] = max(worst["mean_abs"], diff.mean().item())
        worst["rel_rms"] = max(worst["rel_rms"], rel_rms)
        worst["max_ulp"] = max(worst["max_ulp"], _max_ulp_at_scale(out, ref, torch_dtype))

    logger.info(
        f"PRECISION_BASELINE shard={shard_shape} dtype={dtype} N={num_devices}: "
        f"worst-device PCC={worst['pcc']:.7f} max_abs={worst['max_abs']:.6f} "
        f"mean_abs={worst['mean_abs']:.6f} rel_rms={worst['rel_rms']:.6f} max_ulp_at_scale={worst['max_ulp']:.2f}"
    )
