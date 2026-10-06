# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device coverage for the SP-sequence/TP-width DFlash accumulator."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gpt_oss_d_p.tt.dflash import (
    DFlashPrefillConfig,
    TtDFlashFeatureAccumulator,
    reference_accumulate_reduced_hidden,
)


@pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2")
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_dflash_accumulator_sp_sequence_tp_width_sharded(mesh_device, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    hidden = 64
    seq = 64
    targets = (1, 3, 5, 7, 9)
    generator = torch.Generator().manual_seed(31)
    fc = torch.randn(hidden, len(targets) * hidden, generator=generator, dtype=torch.bfloat16) * 0.02
    activations = {
        layer_id: torch.randn(1, 1, seq, hidden, generator=generator, dtype=torch.bfloat16) for layer_id in targets
    }
    cfg = DFlashPrefillConfig(
        checkpoint_path=Path("/synthetic/not-read"),
        hidden_size=hidden,
        num_target_layers=12,
        target_layer_ids=targets,
    )
    accumulator = TtDFlashFeatureAccumulator(
        mesh_device,
        cfg,
        fc,
        sp_axis=0,
        tp_axis=1,
        dtype=ttnn.bfloat16,
    )

    def to_device(x):
        return ttnn.from_torch(
            x,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                mesh_shape=tuple(mesh_device.shape),
                dims=(2, None),
            ),
        )

    tt_activations = {layer_id: to_device(value) for layer_id, value in activations.items()}

    # First prove one synthetic FC target projection in isolation.
    one_cfg = DFlashPrefillConfig(
        checkpoint_path=Path("/synthetic/not-read"),
        hidden_size=hidden,
        num_target_layers=12,
        target_layer_ids=(targets[0],),
    )
    one = TtDFlashFeatureAccumulator(
        mesh_device,
        one_cfg,
        fc[:, :hidden],
        sp_axis=0,
        tp_axis=1,
        dtype=ttnn.bfloat16,
    )
    one.tap(tt_activations[targets[0]], targets[0])
    tt_one = one.export()
    one_host = ttnn.to_torch(
        tt_one,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, 3),
        ),
    ).float()
    one_reference = activations[targets[0]].float() @ fc[:, :hidden].T.float()
    passing, pcc = comp_pcc(one_reference, one_host, 0.99)
    assert passing, f"DFlash one-target projection PCC={pcc}"
    ttnn.deallocate(tt_one)

    for layer_id in targets:
        accumulator.tap(tt_activations[layer_id], layer_id)
    tt_result = accumulator.export()
    ttnn.synchronize_device(mesh_device)

    # The result remains seq/SP x feature/TP sharded.  Reconstruct both axes
    # solely for the test's torch comparison.
    assert tuple(ttnn.get_device_tensors(tt_result)[0].shape)[-2:] == (seq // rows, hidden // cols)
    result = ttnn.to_torch(
        tt_result,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            mesh_device,
            mesh_shape=tuple(mesh_device.shape),
            dims=(2, 3),
        ),
    ).float()
    reference = reference_accumulate_reduced_hidden(
        activations,
        fc,
        target_layer_ids=targets,
    ).float()
    passing, pcc = comp_pcc(reference, result, 0.99)
    assert passing, f"DFlash five-target accumulator PCC={pcc}"

    for tensor in tt_activations.values():
        ttnn.deallocate(tensor)
    ttnn.deallocate(tt_result)
