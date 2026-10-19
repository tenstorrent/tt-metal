# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Output chunks (matmul_reduce_scatter_program_descriptor: ZONE_SETS / plan_chunk_geometry): correctness of forced and
automatic chunk splits on the QB2 perf placement (1x4 ring under FABRIC_2D_TORUS_XY, 2 links).

- The Kimi K2.7 o_proj focus case (fits unchunked) under forced splits along m, n and both, ragged ones, one zone set
  and the Linear schedule; each must match the fp32 torch reference like the unchunked op.
- MiMo-V2 o_proj (scatter -2: W resident, n-outer chunk walk).
- Wan2.2 FF2 at 720p (9472x3456x5120 per device, bf16 x bf16, HiFi2, fp32 DEST: the tt_dit defaults), which does not
  fit L1 unchunked: the automatic split and several forced ones.

Run: scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/matmul_reduce_scatter/test_matmul_reduce_scatter_chunked.py
"""

from __future__ import annotations

import pytest
import torch
import ttnn

import ttnn.operations.matmul_reduce_scatter as pkg
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as desc
from eval.golden_tests.matmul_reduce_scatter.helpers import (
    create_ttnn_input_tensor,
    pytorch_matmul_reduce_scatter,
    quantize_like_device,
    random_stacked,
)
from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter

KIMI = ((1, 1, 640, 2048), (2048, 7168))
MIMO = ((1, 1, 2048, 2048), (2048, 4096))
WAN720 = ((1, 1, 9472, 3456), (3456, 5120))

BF8, BF16 = ttnn.bfloat8_b, ttnn.bfloat16
RING, LINE = ttnn.Topology.Ring, ttnn.Topology.Linear

# (id, inputs, scatter_dim, weight dtype, fp32 DEST acc, forced chunks (None: auto), zone sets, topology)
CASES = [
    ("kimi_1x1", KIMI, -1, BF8, False, (1, 1), 2, RING),
    ("kimi_1x2", KIMI, -1, BF8, False, (1, 2), 2, RING),
    ("kimi_2x1", KIMI, -1, BF8, False, (2, 1), 2, RING),
    ("kimi_2x2", KIMI, -1, BF8, False, (2, 2), 2, RING),
    ("kimi_3x3_ragged", KIMI, -1, BF8, False, (3, 3), 2, RING),
    ("kimi_1x8", KIMI, -1, BF8, False, (1, 8), 2, RING),
    ("kimi_1x4_zone1", KIMI, -1, BF8, False, (1, 4), 1, RING),
    ("kimi_2x2_linear", KIMI, -1, BF8, False, (2, 2), 2, LINE),
    ("mimo_2x1", MIMO, -2, BF8, True, (2, 1), 2, RING),
    ("mimo_1x2", MIMO, -2, BF8, True, (1, 2), 2, RING),
    ("mimo_3x3_ragged", MIMO, -2, BF8, True, (3, 3), 2, RING),
    ("wan720_auto", WAN720, -1, BF16, True, None, 2, RING),
    ("wan720_4x1", WAN720, -1, BF16, True, (4, 1), 2, RING),
    ("wan720_2x4", WAN720, -1, BF16, True, (2, 4), 2, RING),
    ("wan720_8x4_resident", WAN720, -1, BF16, True, (8, 4), 2, RING),
    ("wan720_16x2_resident", WAN720, -1, BF16, True, (16, 2), 2, RING),
]
# rel-RMS gates: the unchunked op's per-weight-dtype tolerances (golden helpers.TOLERANCES), bf16 weights tighter
REL_RMS = {BF8: 0.035, BF16: 0.01}


def _open():
    from tests.scripts.common import get_updated_device_params

    cfg = ttnn._ttnn.fabric.FabricRouterConfig()
    cfg.max_packet_payload_size_bytes = 14 * 1024 + 64
    fc = ttnn.FabricConfig.FABRIC_2D_TORUS_XY
    ttnn.set_fabric_config(
        fc,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        cfg,
    )
    params = get_updated_device_params({"fabric_config": fc})
    params.pop("fabric_config")
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4), **params)


@pytest.fixture(scope="module")
def mesh():
    m = _open()
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def run_case(mesh, inputs, scatter_dim, weight_dtype, fp32_acc, chunks, zone_sets, topology, seed=0):
    """(output tensor, torch reference, plan) of one call with the given forced chunks (None: the planner's choice)."""
    a_shape, w_shape = inputs
    a = random_stacked(mesh, a_shape, torch.bfloat16, seed=seed)
    w = quantize_like_device(
        random_stacked(mesh, w_shape, torch.bfloat16, seed=seed + 1, scale=w_shape[0] ** -0.5), weight_dtype
    )
    ta = create_ttnn_input_tensor(a, mesh, dtype=BF16, layout=ttnn.TILE_LAYOUT)
    tw = create_ttnn_input_tensor(w, mesh, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT)
    ckc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_acc)
    old_sets = desc.ZONE_SETS
    pkg._FORCE_CHUNKS = chunks
    desc.ZONE_SETS = zone_sets
    try:
        out = matmul_reduce_scatter(
            ta, tw, cluster_axis=1, scatter_dim=scatter_dim, topology=topology, num_links=2, compute_kernel_config=ckc
        )
        ttnn.synchronize_device(mesh)
    finally:
        pkg._FORCE_CHUNKS = None
        desc.ZONE_SETS = old_sets
    expected = pytorch_matmul_reduce_scatter(a, w, 1, scatter_dim)
    return out, expected, dict(pkg._LAST_PLAN)


def rel_rms_per_device(mesh, out, expected):
    cols = tuple(mesh.shape)[1]
    res = []
    for idx, t in enumerate(ttnn.get_device_tensors(out)):
        got = ttnn.to_torch(t).double()
        ref = expected[idx // cols, idx % cols].double()
        assert got.shape == ref.shape, (got.shape, ref.shape)
        assert torch.isfinite(got).all(), f"device {idx}: non-finite output"
        res.append(((got - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item())
    return res


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_chunked(mesh, case):
    name, inputs, sd, wdt, fp32_acc, chunks, sets, topo = case
    out, expected, plan = run_case(mesh, inputs, sd, wdt, fp32_acc, chunks, sets, topo)
    rms = rel_rms_per_device(mesh, out, expected)
    print(f"PLAN {name}: {plan} rel_rms {' '.join(f'{r:.5f}' for r in rms)}")
    if chunks is not None:
        assert plan["chunks"] == tuple(chunks) or name.endswith("ragged"), plan["chunks"]
    else:
        assert plan["chunks"] != (1, 1), "the auto plan should have needed chunks"
    assert max(rms) <= REL_RMS[wdt], f"rel_rms {rms} > {REL_RMS[wdt]}"


def test_chunked_repeat_bits(mesh):
    """Back-to-back calls re-arm every counter (arrivals, credits, ready) and are deterministic: chunked and unchunked
    calls interleaved, each chunking giving the same bits every time. (Across chunkings the bits may differ: an
    entry that reloads an operand packs per K-window, the others per super-block, so the fp32 sums round apart.)"""
    seq = ((2, 2), (1, 1), (1, 4), (2, 2), (1, 1), (1, 4))
    outs = {}
    for chunks in seq:
        out, _, _ = run_case(mesh, KIMI, -1, BF8, False, chunks, 2, RING)
        got = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)]
        if chunks in outs:
            for d, (x, y) in enumerate(zip(outs[chunks], got)):
                assert torch.equal(x, y), f"chunks {chunks} device {d}: a repeated call gave different bits"
        else:
            outs[chunks] = got
