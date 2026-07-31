# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only invariants for masked-camera prefix compaction."""

import torch

from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_16_decode import (
    _NUM_PATCHES,
    _prefix_compaction_plan,
)
from models.experimental.pi0_5.tt.tt_pipeline import denoise_pipeline


def _original_to_compact_rows(img_present, lang_len):
    rows = []
    for camera, present in enumerate(img_present):
        if present:
            rows.extend(range(camera * _NUM_PATCHES, (camera + 1) * _NUM_PATCHES))
    language_start = len(img_present) * _NUM_PATCHES
    rows.extend(range(language_start, language_start + lang_len))
    return torch.tensor(rows)


def test_compaction_preserves_masks_positions_and_attention():
    img_masks = [torch.tensor(True), torch.tensor(False), torch.tensor(True)]
    lang_mask = torch.tensor([[True, True, False, True, False, True]])

    img_present, camera_indices, compact_pad = _prefix_compaction_plan(img_masks, lang_mask, True)
    _, full_camera_indices, full_pad = _prefix_compaction_plan(img_masks, lang_mask, False)

    assert camera_indices == (0, 2)
    assert full_camera_indices == (0, 1, 2)
    row_map = _original_to_compact_rows(img_present, lang_mask.shape[1])
    assert torch.equal(full_pad[row_map], compact_pad)

    full_positions = (full_pad.long().cumsum(0) - 1).clamp_min(0)
    compact_positions = (compact_pad.long().cumsum(0) - 1).clamp_min(0)
    assert torch.equal(full_positions[row_map], compact_positions)

    # Invalid rows/columns are excluded at every VLM attention layer. Selecting the
    # corresponding compact rows must therefore preserve all valid query outputs.
    torch.manual_seed(0)
    head_dim = 16
    q = torch.randn(full_pad.numel(), head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    full_valid = full_pad.nonzero(as_tuple=True)[0]
    compact_valid = compact_pad.nonzero(as_tuple=True)[0]
    full_scores = q[full_valid] @ k[full_valid].T
    compact_scores = q[row_map][compact_valid] @ k[row_map][compact_valid].T
    full_out = torch.softmax(full_scores, dim=-1) @ v[full_valid]
    compact_out = torch.softmax(compact_scores, dim=-1) @ v[row_map][compact_valid]
    torch.testing.assert_close(compact_out, full_out, rtol=0, atol=0)


def test_compaction_rejects_all_absent_cameras():
    img_masks = [torch.tensor(False), torch.tensor(False)]
    lang_mask = torch.ones((1, 4), dtype=torch.bool)
    try:
        _prefix_compaction_plan(img_masks, lang_mask, True)
    except ValueError as error:
        assert "at least one present camera" in str(error)
    else:
        raise AssertionError("expected all-absent camera compaction to fail")


def test_compaction_is_noop_for_all_real_cameras():
    img_masks = [torch.tensor(True), torch.tensor(True), torch.tensor(True)]
    lang_mask = torch.ones((1, 256), dtype=torch.bool)
    _, compact_indices, compact_pad = _prefix_compaction_plan(img_masks, lang_mask, True)
    _, full_indices, full_pad = _prefix_compaction_plan(img_masks, lang_mask, False)
    assert compact_indices == full_indices == (0, 1, 2)
    assert torch.equal(compact_pad, full_pad)


def test_d2h_drain_only_routes_readback_as_the_barrier(monkeypatch):
    calls = []

    class FakePipeline:
        @staticmethod
        def replay_loop(loop_tids, *, drain, drain_mesh):
            calls.append(("replay", loop_tids, drain, drain_mesh))

    driver = denoise_pipeline.TTNNPi05DenoiseStreamedPipeline.__new__(
        denoise_pipeline.TTNNPi05DenoiseStreamedPipeline
    )
    driver._loop_tids = ["trace"]
    driver._pipe = FakePipeline()
    driver._stage0_mesh = "stage0"
    driver._x_t = "x_t"
    driver._ah = 2
    monkeypatch.setenv("PI05_D2H_DRAIN_ONLY", "1")

    def fake_to_torch(tensor):
        calls.append(("readback", tensor))
        return torch.zeros((1, 3, 4))

    monkeypatch.setattr(denoise_pipeline.ttnn, "to_torch", fake_to_torch)
    result = driver.replay()
    assert calls == [
        ("replay", ["trace"], "none", "stage0"),
        ("readback", "x_t"),
    ]
    assert result.shape == (1, 2, 4)
