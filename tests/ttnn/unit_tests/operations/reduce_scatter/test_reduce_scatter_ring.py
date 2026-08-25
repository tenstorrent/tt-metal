# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 (Ring topology) — refinement-specific tests.

Ring closes the wrap link (device N-1 <-> device 0) so every block travels the
SHORT way round: per direction the send/arrival depths become uniform across
devices (fwd sends N//2 = own + N//2 - 1 relays, bwd sends (N-1)//2; arrivals
mirror). The kernels' block indices were already ring-modular (op_design.md T3);
Refinement 1 is host-side — the Ring depth table in `_block_flow` plus the
wrap-link neighbour wiring through `ccl_dm_route(.., Ring)` — selected by the
`topology` kwarg alone under the SAME FABRIC_1D fabric config.

Coverage added here (device tests mirror the acceptance-test oracle):
  * Host-only depth-table invariants for N in 2..9: exactly N-1 blocks out and
    N-1 in per device, the relay prefix invariant, the send->arrival handshake
    between neighbours, disjoint fwd/bwd source coverage of all N-1 remote chips
    (the even-N N/2-distance tie carried by FORWARD only), and the kernel-side
    static_assert (fwd_arrivals + bwd_arrivals + 1 == N) holding by construction.
  * Ring functional grid: S = 1 minimal, multi-tile, multi-batch, and the S = 9
    odd tile count that forces the g = 1 granule path; bf16 + fp32.
  * Ring program-cache HIT (3 iterations) — the cache-reuse semaphore re-arm now
    exercises the wrap link too (the first-run-green / second-run-hangs footgun).
  * Ring output_tensor path.
  * Linear then Ring in ONE mesh session — behaviour must switch on the topology
    kwarg alone (no fabric-config change between the calls).

Multi-device — drive via the multi-device runner (mesh MUST match the topology):

    scripts/run_multidevice_sim_pytest.py --op reduce_scatter -- \
        tests/ttnn/unit_tests/operations/reduce_scatter/test_reduce_scatter_ring.py -v
"""

from math import prod

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.reduce_scatter import reduce_scatter
from ttnn.operations.reduce_scatter.reduce_scatter_program_descriptor import _block_flow
from ttnn._ttnn.operations.ccl import Topology as _Topology

PCC = {ttnn.bfloat16: 0.99, ttnn.float32: 0.999}
TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

# Topology <-> fabric_config pairing: Ring runs under the SAME FABRIC_1D config
# as Linear (the golden suite's contract) — the op kwarg alone selects behaviour.
RING = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Ring)
LINEAR = ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}, ttnn.Topology.Linear)
MESH_SHAPE = (1, 4)  # bh_quietbox_1x4_hw contract (see test_reduce_scatter.py)


# ---------------------------------------------------------------------------
# Host-only: Ring depth-table invariants (no device).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_devices", [2, 3, 4, 5, 6, 7, 8, 9])
def test_ring_block_flow_invariants(num_devices):
    """The Ring depth table satisfies every seam the verifier notes name, and its
    source-chip sets (via the kernels' ring-modular formulas) tile all N chips."""
    n = num_devices
    for i in range(n):
        fwd_sends, fwd_arrivals, bwd_sends, bwd_arrivals, fwd_nb, bwd_nb = _block_flow(i, n, _Topology.Ring)

        # Uniform depths (no line ends on a ring).
        assert fwd_sends == n // 2 and fwd_arrivals == n // 2
        assert bwd_sends == (n - 1) // 2 and bwd_arrivals == (n - 1) // 2

        # Host-assert invariants (verifier notes): N-1 blocks out, N-1 in.
        assert fwd_sends + bwd_sends == n - 1
        assert fwd_arrivals + bwd_arrivals == n - 1
        # Kernel-side static_assert holds by construction.
        assert fwd_arrivals + bwd_arrivals + 1 == n

        # Wrap-link neighbour wiring (modular, unlike Linear's clamped i +/- 1).
        assert fwd_nb == (i + 1) % n  # fwd_sends = n//2 >= 1 for n >= 2
        assert bwd_nb == ((i - 1) % n if bwd_sends > 0 else None)

        # Relay prefix invariant (relay reader static_assert): relayed blocks
        # are a prefix of arrivals.
        for sends, arrivals in ((fwd_sends, fwd_arrivals), (bwd_sends, bwd_arrivals)):
            num_relays = sends - 1 if sends > 0 else 0
            assert num_relays <= arrivals

        # Reduce-reader source chips (the kernels' ring-modular formulas): fwd
        # arrival a carries chip (i - 1 - a) mod N, bwd arrival b chip
        # (i + 1 + b) mod N. Disjoint, exclude self, and with own cover all N —
        # the even-N N/2-distance tie carried by exactly ONE direction (fwd).
        fwd_srcs = [(i - 1 - a) % n for a in range(fwd_arrivals)]
        bwd_srcs = [(i + 1 + b) % n for b in range(bwd_arrivals)]
        assert i not in fwd_srcs + bwd_srcs
        assert len(set(fwd_srcs) & set(bwd_srcs)) == 0, "a block carried by both directions double-counts"
        assert set(fwd_srcs) | set(bwd_srcs) | {i} == set(range(n))

        # Send->arrival handshake: what device i sends at step k (writer formula:
        # k=0 own, k>=1 fwd (i-k) mod N / bwd (i+k) mod N) must be exactly what
        # its neighbour receives as arrival k (reader formula above).
        for k in range(fwd_sends):
            sent = i if k == 0 else (i - k) % n
            receiver = (i + 1) % n
            assert sent == (receiver - 1 - k) % n
        for k in range(bwd_sends):
            sent = i if k == 0 else (i + k) % n
            receiver = (i - 1) % n
            assert sent == (receiver + 1 + k) % n


def test_linear_block_flow_unchanged():
    """Non-regression: the Linear table is untouched by the Ring branch."""
    n = 4
    expected = {
        0: (1, 0, 0, 3, 1, None),
        1: (2, 1, 3, 2, 2, 0),
        2: (3, 2, 2, 1, 3, 1),
        3: (0, 3, 1, 0, None, 2),
    }
    for i in range(n):
        assert _block_flow(i, n, _Topology.Linear) == expected[i]


# ---------------------------------------------------------------------------
# Device tests — the acceptance-test oracle (sum-then-slice, fp32-accumulated).
# ---------------------------------------------------------------------------
def _num_devices(mesh_device):
    return prod(tuple(mesh_device.shape))


def _make_sharded_input(mesh_device, shard_shape, dtype, seed=13):
    num_devices = _num_devices(mesh_device)
    torch.manual_seed(seed)
    full = torch.randn((num_devices * shard_shape[0], *shard_shape[1:]), dtype=torch.float32)
    quantized = full.to(TORCH_DTYPE[dtype])
    summed = quantized.reshape(num_devices, *shard_shape).to(torch.float32).sum(dim=0).to(TORCH_DTYPE[dtype])
    oracle_slices = list(torch.chunk(summed, num_devices, dim=3))
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
        assert_with_pcc(oracle_slices[i], ttnn.to_torch(device_out), PCC[dtype])


# Shapes: S = 1 minimal (single-tile slice), multi-tile, multi-batch, and the
# odd tile count S = 9 that forces the g = 1 granule path.
RING_SHAPES = [
    (1, 1, 32, 128),
    (1, 1, 64, 256),
    (2, 1, 64, 256),
    (1, 1, 96, 384),
]


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("shard_shape", RING_SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_ring_functional(mesh_device, device_params, topology, shard_shape, dtype):
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)
    output = reduce_scatter(input_tensor, dim=3, topology=topology)
    _check(output, oracle_slices, dtype)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_program_cache_hit(mesh_device, device_params, topology):
    """Cache-hit re-arm across the wrap link: iteration 0 primes the cache;
    iterations 1-2 re-run the SAME cached program, so a missing semaphore re-arm
    on any core (both wrap-link relay writers now send) hangs or corrupts here."""
    for it in range(3):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, (1, 1, 64, 256), ttnn.bfloat16, seed=100 + it)
        output = reduce_scatter(input_tensor, dim=3, topology=topology)
        _check(output, oracle_slices, ttnn.bfloat16)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_ring_output_tensor_path(mesh_device, device_params, topology):
    """The supplied output_tensor is written into and the SAME handle returned."""
    shard_shape = (1, 1, 64, 256)
    dtype = ttnn.float32
    num_devices = _num_devices(mesh_device)
    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype)
    out_shape = list(shard_shape)
    out_shape[3] //= num_devices
    preallocated = ttnn.from_torch(
        torch.zeros((num_devices * out_shape[0], *out_shape[1:]), dtype=TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    returned = reduce_scatter(input_tensor, dim=3, topology=topology, output_tensor=preallocated)
    assert returned is preallocated
    _check(returned, oracle_slices, dtype)


@pytest.mark.parametrize("device_params, topology", [RING], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shard_shape", [(1, 1, 64, 256), (2, 1, 64, 256)], ids=["1x1x64x256", "2x1x64x256"])
def test_ring_precision_metrics(mesh_device, device_params, topology, dtype, shard_shape):
    """Ring accuracy measurement (changelog numbers): worst-device PCC / max-abs /
    rel-RMS vs the fp32-accumulated oracle. Expect the SAME error budget as
    Linear — the arithmetic (arrival-ordered incremental sum, fp32 DEST, bf16
    accumulator packs) is identical; only who-relays-what changed."""
    from loguru import logger
    from models.common.utility_functions import comp_pcc

    input_tensor, oracle_slices = _make_sharded_input(mesh_device, shard_shape, dtype, seed=42)
    output = reduce_scatter(input_tensor, dim=3, topology=topology)
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
        f"RING_PRECISION shape={shard_shape} dtype={dtype} "
        f"worst pcc={worst['pcc']:.7f} max_abs={worst['max_abs']:.6f} rel_rms={worst['rel_rms']:.6f}"
    )


@pytest.mark.parametrize("device_params, topology_a, topology_b", [LINEAR + (RING[1],)], indirect=["device_params"])
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_topology_kwarg_switches_in_one_session(mesh_device, device_params, topology_a, topology_b):
    """Linear then Ring in one mesh session under ONE fabric config — behaviour
    is selected by the topology kwarg alone, and the two cached programs coexist."""
    for topology in (topology_a, topology_b, topology_a, topology_b):
        input_tensor, oracle_slices = _make_sharded_input(mesh_device, (1, 1, 32, 128), ttnn.bfloat16)
        output = reduce_scatter(input_tensor, dim=3, topology=topology)
        _check(output, oracle_slices, ttnn.bfloat16)
