# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Regression test for local-only prefill dispatch→combine.

A dispatch group of one chip has no fabric peers: on a single-device mesh, or on a 1xN mesh with
cluster_axis=0 (Blackhole QB2 1x4 TP serving), every chip dispatches its own tokens to experts that
all live on that chip. dispatch/combine then run with zero fabric links. The round trip must return
each token unchanged in every top-k slot, and a second call must reuse the cached programs.
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric_1d_device_params, torus_x_device_params

NUM_ROUTED_EXPERTS = 64
NUM_EXPERTS_PER_TOK = 6
SEQ_LEN = 128
EMB_DIM = 2560
METADATA_LEN = 3


def _routing(indices):
    """Tile-aligned expert regions in expert order, as a local-only caller builds them."""
    counts = torch.bincount(indices.flatten().long(), minlength=NUM_ROUTED_EXPERTS).to(torch.int32)
    aligned = (counts + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    offsets = torch.cumsum(aligned, 0, dtype=torch.int32) - aligned
    return counts, offsets


def _upload(mesh_device, tensor, dtype, layout):
    # Dim 0 indexes the chip: each chip gets its own independent local problem.
    return ttnn.from_torch(
        tensor,
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )


def _round_trip(mesh_device, x, indices, capacity):
    num_chips = x.shape[0]
    routing = [_routing(indices[c]) for c in range(num_chips)]
    counts = torch.stack([r[0] for r in routing])
    offsets = torch.stack([r[1] for r in routing])

    tt_x = _upload(mesh_device, x.unsqueeze(1), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    tt_indices = _upload(mesh_device, indices.unsqueeze(1).to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
    tt_offsets = _upload(mesh_device, offsets.unsqueeze(1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    # Every expert maps to chip 0 of its one-chip dispatch group.
    tt_table = _upload(
        mesh_device,
        torch.zeros(num_chips, 1, NUM_ROUTED_EXPERTS, dtype=torch.int32),
        ttnn.int32,
        ttnn.ROW_MAJOR_LAYOUT,
    )

    routed, metadata = ttnn.experimental.deepseek_prefill.dispatch(
        ttnn.reshape(tt_x, (1, SEQ_LEN, EMB_DIM)),
        ttnn.reshape(tt_indices, (1, SEQ_LEN, NUM_EXPERTS_PER_TOK)),
        ttnn.reshape(tt_offsets, (1, NUM_ROUTED_EXPERTS)),
        ttnn.reshape(tt_table, (1, NUM_ROUTED_EXPERTS)),
        dispatch_group_size=1,
        experts_per_chip=NUM_ROUTED_EXPERTS,
        num_routed_experts=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        metadata_len=METADATA_LEN,
        max_dispatch_buffer_token_size=capacity,
        cluster_axis=0,
    )
    routed = ttnn.to_layout(ttnn.reshape(routed, (1, 1, capacity, EMB_DIM)), ttnn.TILE_LAYOUT)
    combined = ttnn.experimental.deepseek_prefill.combine(
        routed,
        ttnn.reshape(metadata, (1, 1, capacity, METADATA_LEN)),
        ttnn.reshape(_upload(mesh_device, counts.unsqueeze(1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), (1, 1, -1)),
        ttnn.reshape(tt_offsets, (1, 1, NUM_ROUTED_EXPERTS)),
        dispatch_group_size=1,
        experts_per_chip=NUM_ROUTED_EXPERTS,
        num_experts_per_tok=NUM_EXPERTS_PER_TOK,
        seq_len_per_chip=SEQ_LEN,
        cluster_axis=0,
    )
    ttnn.synchronize_device(mesh_device)
    return ttnn.to_torch(combined, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).reshape(
        num_chips, SEQ_LEN, NUM_EXPERTS_PER_TOK, EMB_DIM
    )


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (1, 1),
            fabric_1d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 1), topology="linear"),
            id="local-1x1",
        ),
        pytest.param(
            (1, 4),
            # QB2 only opens a 1x4 mesh on a wrapped fabric; the local path sends nothing over it.
            torus_x_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(1, 4), topology="ring"),
            id="local-1x4",
        ),
    ],
    indirect=True,
)
def test_dispatch_combine_local(mesh_device, device_params):
    num_chips = mesh_device.get_num_devices()
    raw = NUM_EXPERTS_PER_TOK * SEQ_LEN
    capacity = (raw + (ttnn.TILE_SIZE - 1) * NUM_ROUTED_EXPERTS + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    num_programs = None
    for seed in (0, 1):  # the second call replays the cached programs with new routing and data
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(num_chips, SEQ_LEN, EMB_DIM, generator=generator).to(torch.bfloat16)
        scores = torch.rand(num_chips, SEQ_LEN, NUM_ROUTED_EXPERTS, generator=generator)
        indices = scores.topk(NUM_EXPERTS_PER_TOK, dim=-1).indices.to(torch.int32)

        y = _round_trip(mesh_device, x, indices, capacity)
        expected = x.unsqueeze(2).expand(-1, -1, NUM_EXPERTS_PER_TOK, -1)
        assert torch.equal(y, expected), f"local round trip mismatch on call {seed}"

        if num_programs is None:
            num_programs = mesh_device.num_program_cache_entries()
        else:
            assert mesh_device.num_program_cache_entries() == num_programs, "second call recompiled"
