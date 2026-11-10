# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only gate for the PDD sampling contract: node sigmas, head fusion and the Euler step that applies it.

The defining property is that one fused forward plus one Euler step between the block's node sigmas advances the
state by exactly the sum of the block's heads, each weighted by its own interval's sigma increment.
"""

from __future__ import annotations

import json

import pytest
import torch

from ....pipelines.minimax_h3.pdd_minimax_h3 import MiniMaxH3PDDContract
from ....pipelines.minimax_h3.scheduler import MiniMaxH3Scheduler

# FastH3 OmniRef PDD-8's fastvideo_inference.json.
OMNIREF_PDD8 = {
    "attention_backend": "VIDEO_SPARSE_ATTN_H3",
    "audio_scheduler_shift": 3.0,
    "base_model_revision": "hf://MiniMaxAI/MiniMax-H3@9bfb6693f2cf6de171db46d1aa586f67d773a1da",
    "conditioning": "fixed_ordered_references_target_only_flow",
    "grid_max_t": 0.999,
    "guidance_scale": 1.0,
    "model_type": "ref2va",
    "num_inference_steps": 8,
    "pdd_step_indices": [0, 4, 8, 12, 16, 20, 24, 28, 32],
    "pdd_steps": 32,
    "schema": "fasth3-inference-contract-v1",
    "schema_version": "fasth3-inference-contract-v1",
    "transformer_component": "transformer_ref",
    "transformer_forwards": 8,
    "video_scheduler_shift": 12.0,
    "vsa_ref_keep_rate": 0.1,
    "vsa_ref_policy": "p2_multi_region",
    "vsa_sparsity": 0.9,
    "vsa_tile_size": 128,
}


def _export(tmp_path, contract=OMNIREF_PDD8, pdd_steps=32):
    (tmp_path / "transformer_ref").mkdir()
    (tmp_path / "transformer_ref" / "config.json").write_text(json.dumps({"pdd_steps": pdd_steps}))
    (tmp_path / "fastvideo_inference.json").write_text(json.dumps(contract))
    return MiniMaxH3PDDContract.from_export(tmp_path)


def test_contract_reads_omniref_pdd8(tmp_path):
    pdd = _export(tmp_path)
    assert pdd.num_forwards == 8
    assert [pdd.block(i) for i in range(8)] == [(4 * i, 4 * i + 4) for i in range(8)]
    assert (pdd.vsa_tile_size, pdd.vsa_sparsity, pdd.vsa_ref_keep_rate) == (128, 0.9, 0.1)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        ({"pdd_step_indices": [0, 4, 4, 32]}, "strictly increasing"),
        ({"pdd_step_indices": [1, 32]}, "strictly increasing"),
        ({"num_inference_steps": 4}, "step counts disagree"),
        ({"vsa_ref_policy": "dense"}, "vsa_ref_policy"),
        ({"model_type": "t2va"}, "only ref2va"),
    ],
)
def test_contract_rejects_inconsistent_exports(tmp_path, expect_error, edit, message):
    with expect_error(ValueError, message):
        _export(tmp_path, {**OMNIREF_PDD8, **edit})


def test_contract_checks_the_transformer_config(tmp_path, expect_error):
    with expect_error(ValueError, "pdd_steps"):
        _export(tmp_path, pdd_steps=16)


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_node_sigmas_start_at_the_fixed_point_and_end_at_zero(tmp_path, modality):
    pdd = _export(tmp_path)
    sigmas = pdd.node_sigmas(modality)
    assert sigmas.numel() == 9
    assert sigmas[0].item() == pytest.approx(0.999, abs=1e-7)
    assert sigmas[-1].item() == 0.0
    assert bool((sigmas[1:] < sigmas[:-1]).all())
    # The integration weights telescope to the node sigmas, up to the float32 rounding of two sigmas near 1.
    weights = pdd.integration_weights(modality)
    for step in range(pdd.num_forwards):
        start, end = pdd.block(step)
        assert weights[start:end].sum().item() == pytest.approx((sigmas[step + 1] - sigmas[step]).item(), abs=2e-7)


def test_fusion_of_identical_heads_is_that_head(tmp_path):
    pdd = _export(tmp_path)
    head = torch.randn(96, 16)
    widened = head.repeat(32, 1)
    for step in range(pdd.num_forwards):
        torch.testing.assert_close(pdd.fuse_head("video", step, widened), head)


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_fused_forward_plus_euler_step_is_the_weighted_head_sum(tmp_path, modality):
    pdd = _export(tmp_path)
    channels, features = 8, 16
    widened = torch.randn(pdd.pdd_steps * channels, features)
    hidden = torch.randn(5, features)
    heads = (hidden @ widened.T).unflatten(1, (pdd.pdd_steps, channels))  # [rows, grid, C]
    weights = pdd.integration_weights(modality).to(torch.float32)

    scheduler = MiniMaxH3Scheduler(shift=pdd.shift(modality))
    scheduler.set_timesteps(sigmas=pdd.node_sigmas(modality))
    for step in range(pdd.num_forwards):
        start, end = pdd.block(step)
        fused = hidden @ pdd.fuse_head(modality, step, widened).T
        applied = scheduler.step_coefficient(step) * fused
        # The device adds c * v with c = sigma - sigma_next; FastVideo's increment is sum_j w_j q_j, w_j < 0.
        expected = -torch.einsum("n,rnc->rc", weights[start:end], heads[:, start:end])
        torch.testing.assert_close(applied, expected, rtol=1e-4, atol=1e-5)
