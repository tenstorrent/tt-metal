# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 (dim=2 scatter) — refinement-specific tests.

For dim=2 the per-device slice is rows [i*slice_H, (i+1)*slice_H) of EVERY
(batch, channel) plane, so the reduce reader's walk becomes per-plane dense
row-blocks: walk width = Wt (full rows), base = my_chip_id*slice_Ht*Wt, then
bump_base(Ht*Wt) per plane, B*C planes per contribution. The walk order still
equals the output's row-major tile order plane-by-plane, so the dense writer
and the compute kernel are UNCHANGED (op_requirements.md Refinement 2).

Coverage added here (device tests mirror the acceptance-test oracle):
  * dim=2 functional grid over BOTH topologies (Linear + Ring — Refinement 1
    landed first, so this refinement collects the Ring x dim=2 cells too),
    bf16 + fp32, on shapes that pin every plane-walk seam:
      - (1,1,128,32):  minimal — single-tile slice (slice_Ht=1, Wt=1, S=1, g=1)
      - (1,1,256,64):  one plane, whole run inside one granule (run=4=g)
      - (2,1,128,96):  B=2 planes, run=3 straddles the g=2 granule boundary —
                       the per-tile boundary tracking, not per-chunk
      - (1,2,128,64):  C=2 planes, run=2 INSIDE one g=4 granule — two plane
                       hops within a single reserve/push window
      - (2,1,256,256): the golden multibatch shape on the (1,4) box — the
                       per-batch walk restart the verifier notes call out (a
                       cursor hoisted out of the plane loop reads the WRONG
                       slice on every plane after the first, silently)
      - (1,1,384,96):  odd S=9 forces the g=1 granule path
  * Negative alias: dim=-2 canonicalizes to 2 and reduces correctly.
  * Program-cache HIT with dim=2 (both topologies, 2 iterations) — the cached
    dim=2 program's semaphore re-arm.
  * dim=3 and dim=2 in ONE mesh session — two cached programs with different
    reduce-reader walks coexist.
  * Loud rejection: H that splits into non-tile-aligned dim=2 slices.
  * Precision metrics (changelog numbers): worst-device PCC / max-abs /
    rel-RMS vs the fp32-accumulated oracle — expected IDENTICAL error budget
    to dim=3 (the arithmetic is untouched; only the reduce reader's tile-id
    walk changed).

Multi-device — drive via the multi-device runner (mesh MUST match the topology):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_dim2.py -v
"""

from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter

PCC = {ttnn.bfloat16: 0.99, ttnn.float32: 0.999}
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# Both topologies run under the SAME FABRIC_1D config (Refinement 1 contract).
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)
RING = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Ring)
MESH_SHAPE = (1, 4)  # bh_quietbox_1x4_hw contract (see test_reduce_scatter.py)


def _num_devices(mesh_device):
    return prod(tuple(mesh_device.shape))


def _make_sharded_input(mesh_device, shard_shape, dtype, dim=2, seed=13):
    """N distinct same-shape shards on the line mesh + the sum-then-slice oracle
    along `dim` (fp32-accumulated so the reference is not bf16-rounding-bound)."""
    num_devices = _num_devices(mesh_device)
    torch.manual_seed(seed)
    full = torch.randn((num_devices * shard_shape[0], *shard_shape[1:]), dtype=torch.float32)
    quantized = full.to(TORCH_DTYPE[dtype])
    summed = quantized.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(TORCH_DTYPE[dtype])
    oracle_slices = list(torch.chunk(summed, num_devices, dim=dim))
    input_tensor = ttnn.from_torch(
        quantized,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return input_tensor, oracle_slices


def _check(output_tensor, oracle_slices, dtype):
    for i, device_out in enumerate(ttnn.get_device_tensors(output_tensor)):
        actual = ttnn.to_torch(device_out)
        assert tuple(actual.shape) == tuple(
            oracle_slices[i].shape
        ), f"device {i}: output shape {tuple(actual.shape)} != expected {tuple(oracle_slices[i].shape)}"
        assert_with_pcc(oracle_slices[i], actual, PCC[dtype])


# Shapes pinning every plane-walk seam (see module docstring). All heights are
# multiples of N*32 = 128 on the (1, 4) box so the dim=2 slice is tile-aligned.
DIM2_SHAPES = [
    (1, 1, 128, 32),  # minimal: S = 1
    (1, 1, 256, 64),  # one plane, run = granule (4)
    (2, 1, 128, 96),  # B = 2: run = 3 straddles the g = 2 granule
    (1, 2, 128, 64),  # C = 2: run = 2 inside one g = 4 granule
    (2, 1, 256, 256),  # golden multibatch shape: per-batch restart
    (1, 1, 384, 96),  # odd S = 9 -> g = 1
]


@pytest.mark.parametrize("device_params, topology", [LINEAR, RING], ids=["Linear", "Ring"], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("shard_shape", DIM2_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_dim2_functional(mesh_device, device_params, topology, shard_shape, dtype):
    """Device i's output equals rows [i*slice_H, (i+1)*slice_H) of every plane of
    the element-wise sum of all shards."""
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)
    output = reduce_scatter(input_tensor, dim=2, topology=topology)
    _check(output, oracle_slices, dtype)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_negative_alias(mesh_device, device_params, topology):
    """dim=-2 canonicalizes to dim=2 (positive convention) and reduces correctly."""
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, (1, 1, 256, 64), ttnn.bfloat16)
    output = reduce_scatter(input_tensor, dim=-2, topology=topology)
    _check(output, oracle_slices, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR, RING], ids=["Linear", "Ring"], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_program_cache_hit(mesh_device, device_params, topology):
    """Second call re-runs the SAME cached dim=2 program — a missing semaphore
    re-arm hangs or corrupts here, on both topologies."""
    for it in range(2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, (2, 1, 128, 96), ttnn.bfloat16, seed=100 + it)
        output = reduce_scatter(input_tensor, dim=2, topology=topology)
        _check(output, oracle_slices, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim3_and_dim2_in_one_session(mesh_device, device_params, topology):
    """dim=3 then dim=2 (then again) in one mesh session: two cached programs
    with DIFFERENT compile-time reduce-reader walks coexist and stay correct."""
    for dim in (3, 2, 3, 2):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, (1, 1, 128, 128), ttnn.bfloat16, dim=dim)
        output = reduce_scatter(input_tensor, dim=dim, topology=topology)
        _check(output, oracle_slices, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_dim2_rejects_non_tile_aligned_slice(mesh_device, device_params, topology):
    """H that splits into non-tile-aligned per-device slices is rejected loudly
    (ValueError), never silently padded: H=64 on N=4 gives 16-row slices."""
    input_tensor, _ = _make_sharded_input(mesh_device, (1, 1, 64, 128), ttnn.bfloat16, dim=3)
    with pytest.raises(ValueError, match="divisible"):
        reduce_scatter(input_tensor, dim=2, topology=topology)


@pytest.mark.parametrize("device_params, topology", [LINEAR], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shard_shape", [(1, 1, 256, 64), (2, 1, 256, 256)], ids=["1x1x256x64", "2x1x256x256"])
def test_dim2_precision_metrics(mesh_device, device_params, topology, dtype, shard_shape):
    """dim=2 accuracy measurement (changelog numbers): worst-device PCC / max-abs /
    rel-RMS vs the fp32-accumulated oracle. Expect the SAME error budget as the
    dim=3 baseline — the arithmetic (arrival-ordered incremental sum, fp32 DEST,
    accumulator packs) is identical; only the reduce reader's tile-id walk moved."""
    from loguru import logger
    from models.common.utility_functions import comp_pcc

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype, seed=42)
    output = reduce_scatter(input_tensor, dim=2, topology=topology)
    worst = {"pcc": 1.0, "max_abs": 0.0, "rel_rms": 0.0}
    for i, device_out in enumerate(ttnn.get_device_tensors(output)):
        oracle = oracle_slices[i].to(torch.float32)
        actual = ttnn.to_torch(device_out).to(torch.float32)
        _, pcc_val = comp_pcc(oracle, actual, PCC[dtype])
        diff = (actual - oracle).abs()
        worst["pcc"] = min(worst["pcc"], float(pcc_val))
        worst["max_abs"] = max(worst["max_abs"], float(diff.max()))
        worst["rel_rms"] = max(worst["rel_rms"], float((diff.pow(2).mean().sqrt() / oracle.abs().mean())))
        assert_with_pcc(oracle, actual, PCC[dtype])
    logger.info(
        f"DIM2_PRECISION shape={shard_shape} dtype={dtype} "
        f"worst pcc={worst['pcc']:.7f} max_abs={worst['max_abs']:.6f} rel_rms={worst['rel_rms']:.6f}"
    )
