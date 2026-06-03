# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Option C end-to-end PCC vs torch reference, real pi05 weights.

Drives `Pi0_5PipelineC.run_inference()` on a shrunk layout
(vlm_depth=2 expert_depth=1, the same depth the smoke + e2e perf tests
use) and compares its final clean-action tensor to a torch reference
built from the same checkpoint, restricted to the same depth via the
existing `Pi0_5Model` reference.

The reference is `models.experimental.pi0_5.reference.torch_pi0_5_model.Pi0_5Model`
(same one used by `test_pcc_pi05_model_vs_torch.py`). Both models share the
initial noise so flow-matching trajectories converge from the same x_0 —
without that, single-seed e2e PCC is noisy (the existing test does the
same trick at line 166).

Skipped if the checkpoint or `model.safetensors` isn't present locally.

Run:
    PI0_OC_PCC=1 pytest -xvs \
      models/experimental/pi0_5/tests/pcc/test_pcc_option_c_vs_torch.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import PaliGemmaConfig, Pi0_5ModelConfig, SigLIPConfig
from models.experimental.pi0_5.tt.option_c.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.option_c.pipeline import Pi0_5PipelineC
from models.experimental.pi0_5.tt.option_c.stages import build_shrunk_layout

# Skip cleanly when not opted-in — the test takes minutes on a Galaxy.
PCC_ENABLED = os.environ.get("PI0_OC_PCC") == "1"
pytestmark = pytest.mark.skipif(not PCC_ENABLED, reason="set PI0_OC_PCC=1 to run Option C end-to-end PCC test")

# Same params as test_option_b_vs_c_e2e.py so the runs compare apples-to-apples.
LANG_SEQ_LEN = 256
ACTION_HORIZON = 10
ACTION_HORIZON_PADDED = 32
NUM_DENOISE_STEPS = int(os.environ.get("PI0_OC_PCC_STEPS", "10"))
VLM_DEPTH = int(os.environ.get("PI0_OC_PCC_VLM_DEPTH", "2"))
EXPERT_DEPTH = int(os.environ.get("PI0_OC_PCC_EXPERT_DEPTH", "1"))
PCC_THRESHOLD = float(os.environ.get("PI0_OC_PCC_THRESHOLD", "0.90"))
SEED = int(os.environ.get("PI0_OC_PCC_SEED", "42"))

_REAL_CKPT = os.environ.get("PI0_OC_PCC_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


def _make_shrunk_torch_config() -> Pi0_5ModelConfig:
    """Pi0_5ModelConfig with vlm/expert depth shrunk to match the TT layout.

    The torch reference (`Pi0_5Model`) reads `paligemma_variant` /
    `action_expert_variant` to pick block depths. To get the same depth
    as the TT shrunk layout (`build_shrunk_layout(vlm_depth=VLM_DEPTH,
    expert_depth=EXPERT_DEPTH)`), we override the per-sub-config depths
    after the variant has filled in widths.
    """
    cfg = Pi0_5ModelConfig(
        action_dim=32,
        action_horizon=ACTION_HORIZON,
        state_dim=32,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        num_denoising_steps=NUM_DENOISE_STEPS,
    )
    cfg.siglip_config = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    # Shrink the depths so the torch reference only walks the same number
    # of layers as the TT pipeline. The widths are intact, so per-layer
    # math is identical.
    cfg.vlm_config.depth = VLM_DEPTH
    cfg.expert_config.depth = EXPERT_DEPTH
    return cfg


@pytest.mark.timeout(2400)
def test_option_c_e2e_pcc_vs_torch():
    """Same x_0, same prefix → final clean_actions PCC ≥ PCC_THRESHOLD."""
    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    # ---- torch reference --------------------------------------------------
    from models.experimental.pi0_5.common import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    weight_loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = weight_loader.categorized_weights
    torch_cfg = _make_shrunk_torch_config()

    # The torch reference does its own categorized-weight slicing in
    # __init__; we hand it the SAME loader (its constructor walks the
    # config.expert_config.depth to pick which expert block weights to
    # read, so the shrunk depth is honoured automatically).
    torch_model = Pi0_5Model(torch_cfg, weight_loader)

    # ---- shared inputs ----------------------------------------------------
    gen = torch.Generator().manual_seed(SEED)
    pixel_values = torch.randn(1, 3, 224, 224, generator=gen, dtype=torch.float32)
    # Match the e2e benchmark: 256 lang slots, first 32 carry real tokens.
    lang_tokens = torch.zeros(1, LANG_SEQ_LEN, dtype=torch.int64)
    lang_tokens[:, :32] = torch.randint(0, 256000, (1, 32), generator=gen)
    lang_masks = torch.zeros(1, LANG_SEQ_LEN, dtype=torch.bool)
    lang_masks[:, :32] = True
    img_masks = [torch.ones(1, dtype=torch.bool)]
    state = torch.randn(1, torch_cfg.state_dim, generator=gen, dtype=torch.float32)
    # Shared initial noise — Option C reads it as `noisy_actions`; the
    # torch reference samples in `denoising.sample_noise` (we monkeypatch
    # below).
    x_0 = torch.randn(1, ACTION_HORIZON, torch_cfg.action_dim, generator=gen, dtype=torch.float32)

    # Pad x_0 to the tile-aligned horizon for the TT side.
    x_0_padded = torch.zeros(1, ACTION_HORIZON_PADDED, torch_cfg.action_dim, dtype=torch.float32)
    x_0_padded[:, :ACTION_HORIZON, :] = x_0

    # ---- run torch reference ---------------------------------------------
    print("\n== Running torch reference (Pi0_5Model.forward_inference, shrunk depth) ==")
    with torch.no_grad():
        saved_sample_noise = torch_model.denoising.sample_noise
        torch_model.denoising.sample_noise = lambda bs, device=None, dtype=torch.float32: x_0.clone()
        try:
            torch_actions = torch_model.forward_inference(
                images=[pixel_values],
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state,
            )
        finally:
            torch_model.denoising.sample_noise = saved_sample_noise
    print(f"   torch_actions shape={tuple(torch_actions.shape)}")

    # ---- run Option C TT pipeline ----------------------------------------
    print("\n== Running Pi0_5PipelineC.run_inference (real ckpt, shrunk layout) ==")
    pali_cfg = PaliGemmaConfig()
    # Same depth override on the TT side via build_shrunk_layout.
    shrunk_layout = build_shrunk_layout(vlm_depth=VLM_DEPTH, expert_depth=EXPERT_DEPTH)

    # language_token_ids for Option C: int32 [B, S_lang].
    lang_token_ids_int32 = lang_tokens.to(torch.int32)

    with open_galaxy_mesh(shrunk_layout) as (_parent, submeshes):
        pipe = Pi0_5PipelineC(
            layout=shrunk_layout,
            submeshes=submeshes,
            config=pali_cfg,
            weights=weights,
            denoise_steps=NUM_DENOISE_STEPS,
            action_horizon=ACTION_HORIZON,
            action_dim=32,
        )
        pipe.initialize()
        actions_tt, timings = pipe.run_inference(
            pixel_values=pixel_values,
            language_token_ids=lang_token_ids_int32,
            noisy_actions=x_0_padded,
        )

        tt_actions = ttnn.to_torch(ttnn.get_device_tensors(actions_tt)[0])
        ttnn.deallocate(actions_tt)
        tt_actions = tt_actions[:, :ACTION_HORIZON, : torch_cfg.action_dim]

    print(f"   tt_actions  shape={tuple(tt_actions.shape)}")
    print(
        "   stage timings (ms): "
        f"vision={timings.stage_0_vision_ms:.1f} "
        f"prefill={timings.stage_1_prefill_ms:.1f} "
        f"denoise={timings.stage_2_denoise_ms:.1f} "
        f"total={timings.total_ms:.1f}"
    )

    pcc = _compute_pcc(torch_actions, tt_actions)
    print(f"\n   PCC(option_c vs torch) = {pcc:.6f}  (threshold {PCC_THRESHOLD})")

    # First few rows side-by-side helps diagnose magnitude/sign issues.
    print("\n   torch_actions[0, :3, :6] =")
    print(torch_actions[0, :3, :6])
    print("\n   tt_actions[0, :3, :6] =")
    print(tt_actions[0, :3, :6])

    assert pcc >= PCC_THRESHOLD, f"Option C end-to-end PCC = {pcc:.4f} below threshold {PCC_THRESHOLD}"
