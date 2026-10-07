# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for GPT-OSS true-prefill DFlash feature generation."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from models.demos.gpt_oss_d_p.tt.dflash import (
    DFlashFeatureLayout,
    DFlashPrefillConfig,
    DFlashPrefillResult,
    reference_accumulate_reduced_hidden,
    reference_linear_reduced_hidden,
    slice_dflash_fc_weight,
)
from models.demos.gpt_oss_d_p.tt.runners.adapters.gpt_oss import _resolve_dflash_checkpoint_path

TARGETS = (1, 3, 5, 7, 9)
HIDDEN = 4


def test_fc_slices_transpose_nn_linear_blocks():
    fc = torch.arange(HIDDEN * len(TARGETS) * HIDDEN, dtype=torch.float32).reshape(HIDDEN, len(TARGETS) * HIDDEN)
    slices = slice_dflash_fc_weight(fc, hidden_size=HIDDEN, target_layer_ids=TARGETS)

    assert tuple(slices) == TARGETS
    for i, layer_id in enumerate(TARGETS):
        expected = fc[:, i * HIDDEN : (i + 1) * HIDDEN].T
        torch.testing.assert_close(slices[layer_id], expected)

    # The square slice makes an accidental missing transpose shape-correct; use
    # asymmetric values to prove the multiplication orientation.
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert not torch.equal(x @ slices[TARGETS[0]], x @ slices[TARGETS[0]].T)


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
    assert result.actual_end < result.actual_start + result.chunk_size
    assert result.layout.sequence == "sp_block_cyclic"
    assert result.layout.feature == "tp_width_sharded"

    with expect_error(ValueError, "outside chunk"):
        DFlashPrefillResult(
            slot_id=0,
            actual_start=1024,
            actual_end=1153,
            chunk_size=128,
            reduced_hidden=object(),
            layout=result.layout,
        )


def test_adapter_resolves_drafter_path_only_when_opted_in(monkeypatch, tmp_path, expect_error):
    monkeypatch.setenv("TT_HF_DRAFT_MODEL", str(tmp_path / "draft"))
    monkeypatch.delenv("PREFILL_DFLASH", raising=False)
    assert _resolve_dflash_checkpoint_path() is None

    monkeypatch.setenv("PREFILL_DFLASH", "1")
    assert _resolve_dflash_checkpoint_path() == tmp_path / "draft"

    monkeypatch.delenv("TT_HF_DRAFT_MODEL")
    with expect_error(ValueError, "TT_HF_DRAFT_MODEL"):
        _resolve_dflash_checkpoint_path()
