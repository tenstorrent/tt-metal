# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for GPT-OSS true-prefill DFlash feature generation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from models.demos.gpt_oss_d_p.tt.dflash import (
    DFlashFeatureLayout,
    DFlashPrefillConfig,
    DFlashPrefillResult,
    FinalTokenLocation,
    final_token_location,
    mesh_device_index,
    reference_accumulate_reduced_hidden,
    reference_linear_reduced_hidden,
    slice_dflash_fc_weight,
)
from models.demos.gpt_oss_d_p.tt.runners.adapters.gpt_oss import GptOssPrefillAdapter

TARGETS = (1, 3, 5, 7, 9)
HIDDEN = 4


def test_fc_slices_transpose_nn_linear_blocks():
    fc = torch.arange(HIDDEN * len(TARGETS) * HIDDEN, dtype=torch.float32).reshape(HIDDEN, len(TARGETS) * HIDDEN)
    slices = slice_dflash_fc_weight(fc, hidden_size=HIDDEN, target_layer_ids=TARGETS)

    assert tuple(slices) == TARGETS
    for i, layer_id in enumerate(TARGETS):
        expected = fc[:, i * HIDDEN : (i + 1) * HIDDEN].T
        torch.testing.assert_close(slices[layer_id], expected)


def test_streamed_accumulation_equals_single_linear():
    generator = torch.Generator().manual_seed(17)
    activations = {layer_id: torch.randn(2, 11, HIDDEN, generator=generator) for layer_id in TARGETS}
    fc = torch.randn(HIDDEN, len(TARGETS) * HIDDEN, generator=generator)

    streamed = reference_accumulate_reduced_hidden(activations, fc, target_layer_ids=TARGETS)
    unsliced = reference_linear_reduced_hidden(activations, fc, target_layer_ids=TARGETS)
    torch.testing.assert_close(streamed, unsliced, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "target_ids, message",
    [
        ((1, 5, 3), "unique and increasing"),
        ((1, 3, 3), "unique and increasing"),
        ((), "must not be empty"),
    ],
)
def test_invalid_target_order_fails(target_ids, message, expect_error):
    with expect_error(ValueError, message):
        slice_dflash_fc_weight(
            torch.empty(HIDDEN, max(1, len(target_ids)) * HIDDEN),
            hidden_size=HIDDEN,
            target_layer_ids=target_ids,
        )


def test_invalid_fc_and_activation_widths_fail_loudly(expect_error):
    with expect_error(ValueError, "does not match expected"):
        slice_dflash_fc_weight(
            torch.empty(HIDDEN, len(TARGETS) * HIDDEN + 1),
            hidden_size=HIDDEN,
            target_layer_ids=TARGETS,
        )

    activations = {layer_id: torch.zeros(3, HIDDEN) for layer_id in TARGETS}
    activations[TARGETS[-1]] = torch.zeros(3, HIDDEN + 1)
    with expect_error(ValueError, f"layer {TARGETS[-1]} has shape"):
        reference_accumulate_reduced_hidden(
            activations,
            torch.zeros(HIDDEN, len(TARGETS) * HIDDEN),
            target_layer_ids=TARGETS,
        )


def _write_checkpoint(path, *, target_ids=TARGETS, hidden=HIDDEN, num_target_layers=12, weights=None):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": hidden,
                "num_target_layers": num_target_layers,
                "dflash_config": {"target_layer_ids": list(target_ids)},
            }
        )
    )
    save_file(
        weights if weights is not None else {"fc.weight": torch.zeros(hidden, len(target_ids) * hidden)},
        path / "model.safetensors",
    )


def test_checkpoint_contract_validates_metadata_and_fc(tmp_path):
    checkpoint = tmp_path / "drafter"
    _write_checkpoint(checkpoint)
    cfg = DFlashPrefillConfig.from_checkpoint(
        checkpoint,
        expected_hidden_size=HIDDEN,
        expected_num_target_layers=12,
        expected_target_layer_ids=TARGETS,
    )
    assert cfg.checkpoint_path == checkpoint
    assert cfg.target_layer_ids == TARGETS


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"hidden": HIDDEN + 1}, "hidden_size"),
        ({"num_target_layers": 13}, "num_target_layers"),
        ({"target_ids": (1, 3, 5, 9, 7)}, "unique and increasing"),
    ],
)
def test_checkpoint_metadata_mismatch_fails(tmp_path, kwargs, expected, expect_error):
    checkpoint = tmp_path / "drafter"
    _write_checkpoint(checkpoint, **kwargs)
    with expect_error(ValueError, expected):
        DFlashPrefillConfig.from_checkpoint(
            checkpoint,
            expected_hidden_size=HIDDEN,
            expected_num_target_layers=12,
            expected_target_layer_ids=TARGETS,
        )


def test_checkpoint_missing_key_and_bad_fc_shape_fail(tmp_path, expect_error):
    missing = tmp_path / "missing"
    _write_checkpoint(missing, weights={"other.weight": torch.zeros(1)})
    with expect_error(KeyError, "fc.weight"):
        DFlashPrefillConfig.from_checkpoint(
            missing,
            expected_hidden_size=HIDDEN,
            expected_num_target_layers=12,
            expected_target_layer_ids=TARGETS,
        )

    malformed = tmp_path / "malformed"
    _write_checkpoint(
        malformed,
        weights={"fc.weight": torch.zeros(HIDDEN, len(TARGETS) * HIDDEN + 1)},
    )
    with expect_error(ValueError, "expected"):
        DFlashPrefillConfig.from_checkpoint(
            malformed,
            expected_hidden_size=HIDDEN,
            expected_num_target_layers=12,
            expected_target_layer_ids=TARGETS,
        )


def test_handoff_real_range_excludes_padded_tail(expect_error):
    result = DFlashPrefillResult(
        slot_id=3,
        actual_start=1024,
        actual_end=1111,
        chunk_size=128,
        reduced_hidden=object(),
        layout=DFlashFeatureLayout(mesh_shape=(4, 8), sp_axis=0, tp_axis=1),
        logits=torch.zeros(10),
        y0=7,
    )
    assert result.num_real_tokens == 87

    with expect_error(ValueError, "outside chunk"):
        DFlashPrefillResult(
            slot_id=0,
            actual_start=1024,
            actual_end=1153,
            chunk_size=128,
            reduced_hidden=object(),
            layout=result.layout,
        )


def test_adapter_refuses_runner_dflash(expect_error):
    assert GptOssPrefillAdapter.supports_dflash is False
    params = SimpleNamespace(dflash_enabled=True)
    with expect_error(NotImplementedError, "not wired through the common runner"):
        GptOssPrefillAdapter().build_runtime(mesh_device=None, hf_config=None, params=params)


@pytest.mark.parametrize(
    "actual_start, actual_end, chunk_size, expected",
    [
        # 1000 real tokens in a 1024 chunk at SP=4: row 3 holds local 231 = tile 224 + 7.
        (0, 1000, 1024, FinalTokenLocation(owner_row=3, tile_start=224, row_in_tile=7)),
        # A full chunk ends on the last row of the last tile on the last SP row.
        (0, 1024, 1024, FinalTokenLocation(owner_row=3, tile_start=224, row_in_tile=31)),
        # The range is relative to the chunk, not absolute: the second 512 chunk's 300th token.
        (512, 812, 512, FinalTokenLocation(owner_row=2, tile_start=32, row_in_tile=11)),
        # One real token lands on SP row 0, tile 0, row 0.
        (2048, 2049, 1024, FinalTokenLocation(owner_row=0, tile_start=0, row_in_tile=0)),
        # Exactly on an SP boundary: local 255 of row 0.
        (0, 256, 1024, FinalTokenLocation(owner_row=0, tile_start=224, row_in_tile=31)),
        (0, 257, 1024, FinalTokenLocation(owner_row=1, tile_start=0, row_in_tile=0)),
    ],
)
def test_final_token_location(actual_start, actual_end, chunk_size, expected):
    assert final_token_location(actual_start, actual_end, chunk_size, sp_factor=4) == expected


def test_final_token_location_rejects_bad_ranges(expect_error):
    with expect_error(ValueError, "multiple of"):
        final_token_location(0, 10, 1000, sp_factor=4)
    with expect_error(ValueError, "within one"):
        final_token_location(0, 1025, 1024, sp_factor=4)
    with expect_error(ValueError, "within one"):
        final_token_location(5, 5, 1024, sp_factor=4)


@pytest.mark.parametrize("sp_axis, tp_axis", [(0, 1), (1, 0)])
def test_mesh_device_index_matches_row_major_device_order(sp_axis, tp_axis):
    mesh_shape = (4, 8) if sp_axis == 0 else (8, 4)
    seen = set()
    for sp in range(mesh_shape[sp_axis]):
        for tp in range(mesh_shape[tp_axis]):
            coords = [0, 0]
            coords[sp_axis], coords[tp_axis] = sp, tp
            index = mesh_device_index(mesh_shape, sp_axis, tp_axis, sp, tp)
            assert index == coords[0] * mesh_shape[1] + coords[1]
            seen.add(index)
    assert seen == set(range(mesh_shape[0] * mesh_shape[1]))


def test_mesh_device_index_rejects_out_of_mesh(expect_error):
    with expect_error(ValueError, "outside mesh"):
        mesh_device_index((4, 8), 0, 1, 4, 0)
