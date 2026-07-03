# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 — format-axis coverage for the all_gather CCL op.

Adds the two format axes promoted to SUPPORTED in Refinement 1:

  * layout = ttnn.ROW_MAJOR_LAYOUT  (native in-kernel raw-row byte movement — NOT a
    to_layout/tilize wrapper). Exercised with bfloat16 and float32.
  * dtype  = ttnn.bfloat8_b         (whole-tile block-float page relayed intact).
    TILE_LAYOUT only — bf8b x ROW_MAJOR is structurally impossible (INVALID).

all_gather is PURE BYTE MOVEMENT (identity gather, no arithmetic), so every format is
correct by construction: the output on every device is the bit-for-bit concatenation of
the N per-device shards. This is the CCL analog of the single-device layout-matrix /
precision-matrix contract — it locks in the newly-supported (dtype, layout) rectangle
across tile-aligned AND non-tile-aligned shard shapes at the proven gather_dim=0, Linear.

The non-tile-aligned shapes matter for both new axes:
  * RM + non-aligned H: raw-row byte movement must handle a shard whose H is not %32.
  * bf8b + non-aligned:  the TILE-padded shard's packed exponents must survive the relay
    (tiles arrive already-packed and are NEVER re-tilized, so no exponent corruption —
    this is why bf8b x non_tile_aligned needs no EXCLUSIONS).

Drive on the deterministic WH multi-device sim (mesh (1, 8) + FABRIC_1D):

    scripts/run_multidevice_sim_pytest.py --op all_gather -- \
        tests/ttnn/unit_tests/operations/all_gather/test_all_gather_formats.py -v

The mesh shape (1, 8) + FABRIC_1D MUST match the sim's mesh-graph descriptor (else
fabric init hangs: "Fabric Router Sync: Timeout").
"""

from math import prod

import pytest
import torch
from loguru import logger

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.all_gather import all_gather


# PCC keyed by dtype. all_gather adds NO error (bit-exact byte movement); the only
# error present is the from_torch quantization that happened before the op ran.
PCC = {
    ttnn.float32: 0.9999,
    ttnn.bfloat16: 0.999,
    ttnn.bfloat8_b: 0.999,  # measured bf8b round-trip PCC ~0.99997 (host probe)
}

# The two newly-supported format pairs (Refinement 1), plus the two Phase-0 TILE pairs
# for a full rectangle. bf8b x ROW_MAJOR is INVALID (block-float has no RM form) and is
# NOT listed — it never reaches the op.
DTYPE_LAYOUTS = [
    (ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),  # Refinement 1
    (ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),  # Refinement 1
    (ttnn.bfloat8_b, ttnn.TILE_LAYOUT),  # Refinement 1
    (ttnn.bfloat16, ttnn.TILE_LAYOUT),  # Phase 0 (regression guard)
    (ttnn.float32, ttnn.TILE_LAYOUT),  # Phase 0 (regression guard)
]

# Per-device SHARD shapes: single-tile, multi-tile, non-square, multi-batch, and two
# non-tile-aligned (H not %32) — the 16-B page constraint stays satisfied for all.
SHARD_SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile
    (2, 1, 32, 64),  # multi-batch (dim 0 = 2)
    (1, 1, 48, 64),  # non-tile-aligned (H=48 not %32)
    (1, 1, 96, 64),  # non-square, tile-aligned
]

LINEAR = {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _torch_dtype(dtype):
    # bf8b has no native torch dtype; reference in bf16 (matches the golden helper).
    return torch.float32 if dtype == ttnn.float32 else torch.bfloat16


def _make_sharded_input(mesh_device, shard_shape, dtype, layout):
    """Shard a freshly-seeded full tensor along dim 0 across the whole line.

    Full tensor shape = (N * shard_shape[0], *shard_shape[1:]); each device gets one
    shard. Returns the ttnn input and the torch full tensor (the gather_dim=0 oracle)."""
    num_devices = prod(tuple(mesh_device.shape))
    full_shape = (shard_shape[0] * num_devices, *shard_shape[1:])

    torch.manual_seed(42)
    torch_full = torch.randn(full_shape, dtype=torch.float32).to(_torch_dtype(dtype))

    input_tensor = ttnn.from_torch(
        torch_full,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    ttnn.synchronize_device(mesh_device)
    return input_tensor, torch_full


@pytest.mark.parametrize("device_params", [LINEAR], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize("dtype, layout", DTYPE_LAYOUTS)
@pytest.mark.parametrize("shard_shape", SHARD_SHAPES)
def test_all_gather_format_matrix(mesh_device, dtype, layout, shard_shape):
    """Every device's output equals the concat of all shards along gather_dim=0,
    across the newly-supported (dtype, layout) rectangle and aligned/non-aligned shapes."""
    num_devices = prod(tuple(mesh_device.shape))
    if num_devices < 2:
        pytest.skip("all_gather requires at least 2 mesh devices")

    input_tensor, torch_full = _make_sharded_input(mesh_device, shard_shape, dtype, layout)

    output_tensor = all_gather(input_tensor, 0, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(mesh_device)

    # Output must preserve the input format (pure byte movement).
    assert output_tensor.dtype == dtype, f"dtype {output_tensor.dtype} != {dtype}"
    assert output_tensor.layout == layout, f"layout {output_tensor.layout} != {layout}"

    output_shards = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(output_tensor)]

    pcc = PCC[dtype]
    torch_ref = torch_full.to(torch.float32)
    for dev_idx, dev_out in enumerate(output_shards):
        assert tuple(dev_out.shape) == tuple(
            torch_full.shape
        ), f"device {dev_idx} output shape {tuple(dev_out.shape)} != full {tuple(torch_full.shape)}"
        assert_with_pcc(torch_ref, dev_out.to(torch.float32), pcc)
    logger.info(
        f"all_gather format-matrix {dtype} {layout} shard={shard_shape} on "
        f"{num_devices} devices: every device holds the full tensor (PCC>={pcc})"
    )
