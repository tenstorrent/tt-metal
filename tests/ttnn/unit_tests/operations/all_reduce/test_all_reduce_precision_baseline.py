# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for the self-contained Python all_reduce CCL+compute op.

all_reduce carries ARITHMETIC (an N-way element-wise SUM accumulated in fp32
DEST on the reduce core), so unlike the pure-movement CCLs the output error is
not zero: a bf16 sum of N terms accumulates pack/unpack rounding, and fp32
operands are truncated in the FPU srcA/srcB registers (~10-bit mantissa) even
with fp32_dest_acc_en. This file measures, per shard-shape × dtype cell:

  * PCC (worst device of the N replicated outputs),
  * max / mean abs error,
  * relative RMS error,
  * max ULP@scale — max |err| in units of the ULP spacing at the oracle's max
    magnitude (how many representable steps the output is off at tensor scale).

Oracle: host element-wise sum of the N shards, accumulated in fp32, cast once
to the device dtype — so the reference is not itself limited by bf16 rounding.

Run on the same runner as the acceptance suite (mesh shape must match the box):

    scripts/run_multidevice_sim_pytest.py --runtime hardware --op all_reduce -- \
        tests/ttnn/unit_tests/operations/all_reduce/test_all_reduce_precision_baseline.py -v
"""

import math
import os
from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose

from ttnn.operations.all_reduce import all_reduce


def _hw_mesh_shape(default=(1, 4)):
    raw = os.environ.get("CCL_HW_MESH_SHAPE")
    return tuple(int(x) for x in raw.split(",")) if raw else default


# Match the acceptance/golden thresholds (a bf16 sum of N terms accumulates
# rounding; observed PCC is far above these).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.99,
}

# small / multi-tile / multi-batch / one larger (P=64 — the largest resident
# accumulator any suite exercises; acceptance and golden stop at P=8).
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile: P=1, g=1
    (1, 1, 64, 128),  # multi-tile: P=8, g=4
    (2, 1, 32, 64),  # multi-batch: P=4, g=4
    (1, 1, 256, 256),  # larger: P=64, g=4 (128 KB bf16 / 256 KB f32 accumulator)
]

DTYPES = [ttnn.bfloat16, ttnn.float32]

LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
_MANTISSA_BITS = {ttnn.bfloat16: 7, ttnn.float32: 23}


def _make_sharded_input(mesh_device, shard_shape, dtype):
    """Seeded full tensor sharded along dim 0; returns (ttnn input, fp32-accumulated
    SUM oracle cast once to the device dtype)."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32).to(_TORCH_DTYPE[dtype])

    oracle = torch_full.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(_TORCH_DTYPE[dtype])

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


def _metrics(golden, calculated, dtype):
    g = golden.to(torch.float32)
    c = calculated.to(torch.float32)
    abs_err = torch.abs(g - c)
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    rms = torch.sqrt(torch.mean((g - c) ** 2)).item()
    denom = torch.sqrt(torch.mean(g * g)).item()
    rel_rms = rms / denom if denom > 0 else rms
    # ULP spacing at the oracle's max magnitude: 2^(floor(log2(max)) - mantissa_bits).
    g_max = torch.abs(g).max().item()
    ulp_at_scale = 2.0 ** (math.floor(math.log2(g_max)) - _MANTISSA_BITS[dtype]) if g_max > 0 else 1.0
    max_ulp = max_abs / ulp_at_scale
    pcc = torch.corrcoef(torch.stack([g.flatten(), c.flatten()]))[0, 1].item()
    return pcc, max_abs, mean_abs, rel_rms, max_ulp


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [_hw_mesh_shape()], indirect=True)
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

    # Worst-device metrics: every device holds its own independently-computed sum
    # (own reduce core, own arrival order), so report the weakest of the N.
    worst = None
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(shard_shape), f"device {dev_idx} shape mismatch"
        m = _metrics(oracle, dev_out, dtype)
        if worst is None or m[3] > worst[1][3]:
            worst = (dev_idx, m)
        assert_with_pcc(oracle, dev_out, PCC[dtype])

    dev_idx, (pcc, max_abs, mean_abs, rel_rms, max_ulp) = worst
    _, allclose_msg = comp_allclose(oracle, output_shards[dev_idx])
    logger.info(
        f"[precision] all_reduce {dtype} shard={shard_shape} N={num_devices} "
        f"(worst dev {dev_idx}): pcc={pcc:.7f} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
        f"rel_rms={rel_rms:.6g} max_ulp@scale={max_ulp:.3g} | {allclose_msg}"
    )
