# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device validation for the moe_padding_config experimental op.

The op is the trace-safe producer of the MoE padding config: it derives each chip's
``[local_real_tokens, pad_side]`` row on-device from two 1-element uint32 tensors (this chunk's
``actual_start`` / ``actual_end``), replacing the host builder's ``ttnn.from_torch`` — which cannot
run inside a trace capture.

The host builder (``TtMoEGatePrefill.build_padding_config``, via ``rotated_chip_real_token_counts``)
is the reference: these tests assert the device op reproduces it EXACTLY (integer counts, so exact
equality, not PCC) across the rotated and sequential layouts.

Also covers the two properties the trace path depends on:
  * one cached program serves every chunk (the per-chunk values are read on-device, so they must not
    enter the program hash) — otherwise a capture could not replay across chunks;
  * refreshing the metadata tensors in place and re-running yields the new chunk's config.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params, torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_real_token_counts

# Galaxy-shaped SP=8 config (chunk_size_global 5120 -> tokens_per_chip 640) plus the existing small
# 2x4 case for boxes that cannot host 32 chips. Keep this one-for-one: the production row owns
# TorusXY, while the local row owns plain Fabric2D.
_MESHES = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
    pytest.param(
        (2, 4),
        fabric2d_device_params(),
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
        id="fabric2d-2x4",
    ),
]

# (actual_start, actual_isl) pairs. Chosen to cover every branch of the rotation:
#   * start == 0                -> sequential layout (the degenerate case)
#   * start slab-aligned        -> rotation is the identity
#   * start mid-slab            -> boundary chip splits across two slabs (the 2-segment case)
#   * isl == full chunk         -> no padding at all (every chip full)
#   * isl == 0                  -> everything is pad (every chip zero)
# The 15k-style starts (2592, 4160, 9280, 10080, 13440) are the ones the rotated-padded CI test uses.
_CASES = [
    (0, 0),
    (0, 640),
    (0, 2592),
    (0, 5120),
    (2592, 1568),
    (2592, 5120),
    (4160, 5120),
    (4160, 800),
    (5120, 3360),
    (9280, 800),
    (10080, 3360),
    (13440, 1920),
]
_IDS = [f"start{s}_isl{i}" for (s, i) in _CASES]


def _meta1(mesh_device, val: int) -> ttnn.Tensor:
    """1-element uint32 DRAM tensor ([1,1,1,1], ROW_MAJOR, replicated) — the metadata form the op reads."""
    return ttnn.from_torch(
        torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _alloc_config(mesh_device, sp_factor: int) -> ttnn.Tensor:
    """Persistent [sp_factor, 2] uint32 config row, sharded along the SP axis (one row per chip)."""
    return ttnn.from_torch(
        torch.zeros((sp_factor, 2), dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
    )


def _read_counts(config: ttnn.Tensor, mesh_device, sp_factor: int) -> tuple[list[int], list[int]]:
    """Gather the per-chip rows back to host -> (local_real_tokens per chip, pad_side per chip)."""
    host = ttnn.to_torch(
        config,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=mesh_device.shape),
    )
    # dim 1 replicates across the TP axis; take one replica. Rows are SP-major.
    rows = host[:sp_factor, :2].to(torch.int64)
    return rows[:, 0].tolist(), rows[:, 1].tolist()


def _run_case(mesh_device, actual_start: int, actual_isl: int, padding_side: str = "right"):
    sp_factor = int(mesh_device.shape[0])
    tokens_per_chip = 640
    chunk_global = sp_factor * tokens_per_chip

    config = _alloc_config(mesh_device, sp_factor)
    start_t = _meta1(mesh_device, actual_start)
    end_t = _meta1(mesh_device, actual_start + actual_isl)

    ttnn.experimental.deepseek_prefill.moe_padding_config(
        config,
        start_t,
        end_t,
        tokens_per_chip=tokens_per_chip,
        pad_side=0 if padding_side == "right" else 1,
        cluster_axis=0,
    )
    ttnn.synchronize_device(mesh_device)

    got_counts, got_sides = _read_counts(config, mesh_device, sp_factor)
    expected = rotated_chip_real_token_counts(actual_start, actual_isl, sp_factor, tokens_per_chip)

    logger.info(
        f"start={actual_start} isl={actual_isl} sp={sp_factor} chunk_global={chunk_global}\n"
        f"  expected={expected}\n  device  ={got_counts}"
    )
    assert got_counts == expected, f"per-chip real-token counts differ: device={got_counts} host={expected}"
    assert got_sides == [0 if padding_side == "right" else 1] * sp_factor, f"pad_side mismatch: {got_sides}"
    return config, start_t, end_t


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
@pytest.mark.parametrize("actual_start,actual_isl", _CASES, ids=_IDS)
def test_moe_padding_config_matches_host(mesh_device, actual_start, actual_isl):
    """The device op reproduces the host builder's per-chip counts exactly."""
    sp_factor = int(mesh_device.shape[0])
    if actual_start + actual_isl > sp_factor * 640 * 8:
        pytest.skip("case exceeds this mesh's addressable range")
    _run_case(mesh_device, actual_start, actual_isl)


@pytest.mark.parametrize("mesh_device,device_params", _MESHES, indirect=True)
def test_moe_padding_config_left_padding(mesh_device):
    """pad_side is written through for left padding (start 0 only: rotation implies right padding)."""
    _run_case(mesh_device, 0, 2592, padding_side="left")


@pytest.mark.parametrize(
    "mesh_device,device_params",
    [
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        )
    ],
    indirect=True,
)
def test_moe_padding_config_one_program_across_chunks(mesh_device):
    """The per-chunk values must stay OUT of the program hash, and refreshing the metadata tensors in
    place must change the result — together these are exactly what lets one capture replay across
    chunks. Asserts a single cached program serves a sequence of different chunks, each still correct."""
    sp_factor = int(mesh_device.shape[0])
    tokens_per_chip = 640

    config = _alloc_config(mesh_device, sp_factor)
    start_t = _meta1(mesh_device, 0)
    end_t = _meta1(mesh_device, 0)

    mesh_device.enable_program_cache()
    entries_before = mesh_device.num_program_cache_entries()

    chunks = [(0, 640), (640, 1280), (1280, 300), (2592, 1568)]
    for actual_start, actual_isl in chunks:
        # Refresh the SAME metadata buffers in place — the trace path's mechanism.
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([actual_start], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            start_t,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([actual_start + actual_isl], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            ),
            end_t,
        )
        ttnn.experimental.deepseek_prefill.moe_padding_config(
            config,
            start_t,
            end_t,
            tokens_per_chip=tokens_per_chip,
            pad_side=0,
            cluster_axis=0,
        )
        ttnn.synchronize_device(mesh_device)

        got_counts, _ = _read_counts(config, mesh_device, sp_factor)
        expected = rotated_chip_real_token_counts(actual_start, actual_isl, sp_factor, tokens_per_chip)
        assert got_counts == expected, (
            f"chunk (start={actual_start}, isl={actual_isl}): device={got_counts} host={expected} "
            "-- the op read stale metadata (missing L1-cache invalidate?) or the config went unrefreshed"
        )

    # Exactly ONE program for all four chunks: the per-chunk values are read on-device, so they must
    # not be hashed. More than one entry means a capture could not replay across chunks.
    assert mesh_device.num_program_cache_entries() == entries_before + 1, (
        f"expected 1 cached program for {len(chunks)} chunks, got "
        f"{mesh_device.num_program_cache_entries() - entries_before}"
    )
