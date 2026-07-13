# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Absolute (vs-torch) correctness for every matmul blocking the gate-merge put into play.

The merge widened the Q/QKV projections, which moved them onto different `grid_12_9_configs`
entries and added three new ones. A blocking is a silent corruptor — a bad one returns a
well-shaped tensor of wrong numbers, and a merged-vs-standalone A/B cannot see it whenever BOTH
sides read the same table entry. `(32, 2048, 1024)` is exactly that case: it is the merged audio
cross-attn `to_q`, but it is ALSO the pre-existing audio `to_kv`, which used the untuned fallback
before this entry existed. So check each shape against torch, not against another TT path."""

import pytest
import torch

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import from_torch, to_torch
from models.tt_dit.utils.test import ring_params

# (id, M, K, N_per_device, chunks) — N_per_device is what ColParallelLinear hands get_matmul_config.
CASES = [
    # NEW table entries the merge introduced.
    ("s1_cross_q_1216_4096_2048", 1216, 4096, 2048, 2),
    ("s2_cross_q_4864_4096_2048", 4864, 4096, 2048, 2),
    # Also the pre-existing audio to_kv shape: same key, and it did NOT have a tuned entry before.
    ("audio_q_kv_32_2048_1024", 32, 2048, 1024, 2),
    # Pre-existing entries the merged projections newly land on (shared with the FFNs).
    ("s2_self_qkv_4864_4096_4096", 4864, 4096, 4096, 4),
    ("s1_self_qkv_1216_4096_4096", 1216, 4096, 4096, 4),
    ("audio_self_qkv_32_2048_2048", 32, 2048, 2048, 4),
    # Merged a2v Q lands on the pre-existing to_out entries.
    ("a2v_q_1216_4096_1024", 1216, 4096, 1024, 2),
    ("a2v_q_4864_4096_1024", 4864, 4096, 1024, 2),
]


@pytest.mark.parametrize(
    ("mesh_device", "sp_axis", "tp_axis", "num_links", "device_params", "topology"),
    [pytest.param((4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, id="ring_bh_4x8sp1tp0")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("case", "M", "K", "N_dev", "chunks"), CASES, ids=[c[0] for c in CASES])
def test_merged_matmul_blocking_matches_torch(
    mesh_device, sp_axis, tp_axis, num_links, topology, case, M, K, N_dev, chunks
):
    torch.manual_seed(0)
    tp = tuple(mesh_device.shape)[tp_axis]
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=tuple(mesh_device.shape)[sp_axis], mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tp, mesh_axis=tp_axis),
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

    out_features = N_dev * tp  # ColParallelLinear shards out_features evenly over TP
    linear = ColParallelLinear(
        K,
        out_features,
        bias=True,
        chunks=chunks,
        mesh_device=mesh_device,
        mesh_axis=tp_axis,
        ccl_manager=ccl_manager,
    )
    weight = torch.randn(out_features, K) * 0.05
    bias = torch.randn(out_features) * 0.05
    linear.load_torch_state_dict({"weight": weight.clone(), "bias": bias.clone()})

    x = torch.randn(1, 1, M, K)
    x_tt = from_torch(
        x, device=mesh_device, layout=ttnn.Layout.TILE, dtype=ttnn.bfloat16, mesh_axes=[None, None, None, tp_axis]
    )

    outs = linear(x_tt, parallel_config=parallel_config)
    assert len(outs) == chunks

    ref = x @ weight.T + bias  # (1, 1, M, out_features)
    for c, o in enumerate(outs):
        got = to_torch(o, mesh_axes=[None, None, None, tp_axis])  # (1,1,M, N_dev*tp) reassembled device-major
        # Device d's chunk c occupies global columns [d*N_dev + c*N_dev/chunks, ... + N_dev/chunks).
        w = N_dev // chunks
        exp = torch.cat([ref[..., d * N_dev + c * w : d * N_dev + (c + 1) * w] for d in range(tp)], dim=-1)
        assert_quality(exp, got, pcc=0.999, relative_rmse=0.03)
