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

TARGETS = (1, 3, 5, 7, 9)


def _config(hidden, targets=TARGETS):
    return DFlashPrefillConfig(
        checkpoint_path=Path("/synthetic/not-read"),
        hidden_size=hidden,
        num_target_layers=12,
        target_layer_ids=targets,
    )


def _inputs(hidden, seq, seed):
    generator = torch.Generator().manual_seed(seed)
    fc = torch.randn(hidden, len(TARGETS) * hidden, generator=generator, dtype=torch.bfloat16) * 0.02
    activations = {
        layer_id: torch.randn(1, 1, seq, hidden, generator=generator, dtype=torch.bfloat16) for layer_id in TARGETS
    }
    return fc, activations


def _to_device(mesh_device, x):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None)),
    )


def _to_host(mesh_device, tensor):
    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 3)),
    ).float()


def _accumulate(mesh_device, accumulator, activations):
    tt_activations = {layer_id: _to_device(mesh_device, value) for layer_id, value in activations.items()}
    accumulator.reset()
    for layer_id in TARGETS:
        accumulator.tap(tt_activations[layer_id], layer_id)
    tt_result = accumulator.export()
    host = _to_host(mesh_device, tt_result)
    for tensor in tt_activations.values():
        ttnn.deallocate(tensor)
    ttnn.deallocate(tt_result)
    return host


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_dflash_accumulator_sp_sequence_tp_width_sharded(mesh_device, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    hidden = 64
    seq = 64
    fc, activations = _inputs(hidden, seq, seed=31)
    accumulator = TtDFlashFeatureAccumulator(mesh_device, _config(hidden), fc, sp_axis=0, tp_axis=1)

    # First prove one synthetic FC target projection in isolation.
    one = TtDFlashFeatureAccumulator(mesh_device, _config(hidden, (TARGETS[0],)), fc[:, :hidden], sp_axis=0, tp_axis=1)
    tt_activation = _to_device(mesh_device, activations[TARGETS[0]])
    one.tap(tt_activation, TARGETS[0])
    tt_one = one.export()
    one_reference = activations[TARGETS[0]].float() @ fc[:, :hidden].T.float()
    passing, pcc = comp_pcc(one_reference, _to_host(mesh_device, tt_one), 0.99)
    assert passing, f"DFlash one-target projection PCC={pcc}"
    ttnn.deallocate(tt_one)
    ttnn.deallocate(tt_activation)

    tt_activations = {layer_id: _to_device(mesh_device, value) for layer_id, value in activations.items()}
    for layer_id in TARGETS:
        accumulator.tap(tt_activations[layer_id], layer_id)
    tt_result = accumulator.export()
    ttnn.synchronize_device(mesh_device)

    # The result remains seq/SP x feature/TP sharded.  Reconstruct both axes
    # solely for the test's torch comparison.
    assert tuple(ttnn.get_device_tensors(tt_result)[0].shape)[-2:] == (seq // rows, hidden // cols)
    reference = reference_accumulate_reduced_hidden(activations, fc, target_layer_ids=TARGETS).float()
    passing, pcc = comp_pcc(reference, _to_host(mesh_device, tt_result), 0.99)
    assert passing, f"DFlash five-target accumulator PCC={pcc}"

    for tensor in tt_activations.values():
        ttnn.deallocate(tensor)
    ttnn.deallocate(tt_result)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dflash_accumulator_non_tile_aligned_width(mesh_device, reset_seeds):
    # 2880 / TP=8 = 360 output columns per device, which is not a multiple of the 32-wide tile.
    hidden = 360
    seq = 40
    fc, activations = _inputs(hidden, seq, seed=7)
    accumulator = TtDFlashFeatureAccumulator(mesh_device, _config(hidden), fc, sp_axis=0, tp_axis=1)

    result = _accumulate(mesh_device, accumulator, activations)

    assert tuple(result.shape) == (1, 1, seq, hidden)
    reference = reference_accumulate_reduced_hidden(activations, fc, target_layer_ids=TARGETS).float()
    passing, pcc = comp_pcc(reference, result, 0.99)
    assert passing, f"DFlash accumulator PCC at width {hidden}: {pcc}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_dflash_two_chunks_match_one_shot(mesh_device, reset_seeds):
    hidden = 64
    seq = 128
    fc, activations = _inputs(hidden, seq, seed=11)
    accumulator = TtDFlashFeatureAccumulator(mesh_device, _config(hidden), fc, sp_axis=0, tp_axis=1)

    one_shot = _accumulate(mesh_device, accumulator, activations)
    half = seq // 2
    chunks = [
        _accumulate(mesh_device, accumulator, {k: v[..., start : start + half, :] for k, v in activations.items()})
        for start in (0, half)
    ]
    chunked = torch.cat(chunks, dim=2)

    passing, pcc = comp_pcc(one_shot, chunked, 0.999)
    assert passing, f"two-chunk DFlash features differ from one-shot: PCC={pcc}"
    reference = reference_accumulate_reduced_hidden(activations, fc, target_layer_ids=TARGETS).float()
    passing, pcc = comp_pcc(reference, chunked, 0.99)
    assert passing, f"two-chunk DFlash features PCC={pcc}"
