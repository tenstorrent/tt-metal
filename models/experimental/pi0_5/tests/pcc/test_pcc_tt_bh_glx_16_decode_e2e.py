# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end PCC for the 16-chip socket-KV decode pipeline (the SHIPPING config).

Why this file exists. The repo had full-model PCC gates for the 28-chip host-bounce pipeline
(test_pcc_tt_bh_glx_e2e.py) and for 1x8 (test_pcc_1x8_all_stages), but NONE for the 16-chip
decode path -- which is the one we actually ship and the one
tests/perf/test_perf_tt_bh_glx_16_e2e_trace_2cq.py profiles. The only 16-chip correctness gates were
single-layer unit checks, and a single-layer gate cannot see SigLIP, prefill, the device-to-device KV
sockets, or the multi-step denoise loop. That gap is not hypothetical for this model: the 16-chip perf
gate feeds every camera present, so it stayed green straight through a total loss of kv_sdpa mask
support, and test_l1_single_layer_pcc read 0.999894 both before and after a fix that moved attention
from PCC 0.389 to 0.9996. A whole-pipeline artifact needs a whole-pipeline gate.

Mirrors test_pcc_tt_bh_glx_e2e.py's structure (same torch reference, same seeding discipline) but
builds the 16-chip pipeline exactly as the perf test does, so the correctness gate and the perf
artifact exercise the same code path.

Run:
    export PI05_CHECKPOINT_DIR=/home/tt-admin/pi05_cache/pi05_libero_upstream
    export PYTHONPATH=$PWD TT_METAL_HOME=$PWD
    python_env/bin/pytest -sq \
      models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_16_decode_e2e.py
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
import torch

# 16-chip decode flags, set BEFORE the production env is applied and before any pi0_5/ttnn import --
# identical to the perf test and eval/libero_rollout.py's ttnn_16_decode path, so this gate and the
# profiled artifact resolve the same pipeline.
for _k, _v in {
    "PI0_TP": "8",
    "PI0_TP4_ATTN_HEADPAR": "1",
    "PI0_MLP_BS": "1",
    "PI0_MLP_FUSED_RS": "0",
    "PI0_KV_SOCKET": "1",
}.items():
    os.environ.setdefault(_k, _v)


def _apply_production_env_defaults():
    """setdefault the validated production flags, as the perf test and libero_rollout do."""
    if os.environ.get("PI05_NO_PROD_ENV", "").lower() in ("1", "true", "yes", "on"):
        return
    root = os.environ.get("TT_METAL_HOME") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    envf = os.path.join(root, "_bench_runs", "pi05_production.env")
    if not os.path.exists(envf):
        return
    with open(envf) as f:
        for line in f:
            m = re.match(r"\s*export\s+([A-Z0-9_]+)=(\S+)", line)
            if m and m.group(1) != "PI05_CHECKPOINT_DIR":
                os.environ.setdefault(m.group(1), m.group(2))


_apply_production_env_defaults()

ttnn = pytest.importorskip("ttnn")

SEED = 42
CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
N_CAMS = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
LANG_LEN = 256
# 0.95 matches the other full-model gates (test_pcc_tt_bh_glx_e2e, test_pcc_1x8_vs_torch). The bar is
# looser than a per-layer 0.99 on purpose: this composes SigLIP + prefill + sockets + a multi-step
# flow-matching denoise loop, so bf8/bf4 error accumulates across the whole chain.
PCC_MIN = float(os.environ.get("PI05_E2E_PCC_MIN", "0.95"))


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1, t2 = a.flatten().float(), b.flatten().float()
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - torch.mean(t1)) * (t2 - torch.mean(t2)))
    return (cov / (s1 * s2)).item()


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"pi0.5 checkpoint not found at {CHECKPOINT_DIR}",
)
def test_pi0_5_16_decode_e2e_pcc():
    """Full sample_actions on the 16-chip socket-KV pipeline vs the torch reference."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model as TorchPi0_5Model
    from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_decode_16_mesh
    from models.experimental.pi0_5.tt.tt_bh_glx.pipeline_16_decode import Pi0_5GLX16DecodePipeline

    num_steps = int(os.environ.get("PI05_NUM_DENOISE_STEPS", "5"))
    cfg = Pi0_5ModelConfig(
        action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR),
        num_denoising_steps=num_steps,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))

    img_h = img_w = cfg.siglip_config.image_size
    torch.manual_seed(SEED)
    images = [torch.randn(1, 3, img_h, img_w) for _ in range(N_CAMS)]
    lang_tokens = torch.randint(0, 256000, (1, LANG_LEN), dtype=torch.int64)
    # LAST CAMERA ABSENT: this is what LIBERO feeds (slot 3 padded), and it is the case that engages
    # the mask / prefix-compaction paths. An all-cameras-present gate would leave them unexercised --
    # exactly how the mask regression stayed hidden behind a green perf gate.
    img_masks = [torch.ones(1, dtype=torch.bool) for _ in range(N_CAMS)]
    if N_CAMS >= 2 and os.environ.get("PI05_E2E_PCC_PAD_LAST", "1") == "1":
        img_masks[-1] = torch.zeros(1, dtype=torch.bool)
    lang_masks = torch.ones(1, LANG_LEN, dtype=torch.bool)

    print(f"\n[16-decode e2e PCC] cams={N_CAMS} (last absent) steps={num_steps} horizon={cfg.action_horizon}")

    # INJECT IDENTICAL INITIAL NOISE into both sides. Seeding alone is not enough: this is a
    # flow-matching sampler, so a different starting x_t integrates to a different-but-equally-valid
    # trajectory (measured PCC ~0.46, i.e. "uncorrelated but same statistics", not a real regression).
    # Both sides draw torch.randn(1, action_horizon, action_dim), but the TT pipeline builds its denoise
    # driver inside sample_actions BEFORE drawing, and that construction consumes RNG -- so a seed set
    # outside desynchronises the two streams. Pin the tensor instead of the seed.
    from models.experimental.pi0_5.reference.torch_denoise import DenoisingModule

    torch.manual_seed(SEED + 1)
    fixed_noise = torch.randn(1, cfg.action_horizon, cfg.action_dim, dtype=torch.float32)

    torch.manual_seed(SEED)
    ref = TorchPi0_5Model(cfg, loader)
    _orig_sample_noise = DenoisingModule.sample_noise
    DenoisingModule.sample_noise = lambda self, batch_size, device=None, dtype=torch.float32: (
        fixed_noise.to(device=device, dtype=dtype) if device is not None else fixed_noise.to(dtype)
    )
    try:
        with torch.no_grad():
            ref_actions = ref.sample_actions(
                images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks, state=None
            )
    finally:
        DenoisingModule.sample_noise = _orig_sample_noise

    with open_decode_16_mesh(l1_small_size=24576, trace_region_size=134_217_728) as mesh_handles:
        pipe = Pi0_5GLX16DecodePipeline(cfg, loader.categorized_weights, mesh_handles)

        def _fixed_noise_padded(_pipe=pipe):
            pad = torch.zeros(1, _pipe._action_horizon_padded, _pipe.action_dim, dtype=torch.float32)
            pad[:, : _pipe.action_horizon, :] = fixed_noise
            return pad

        pipe._build_noise_torch = _fixed_noise_padded
        actions = pipe.sample_actions(
            images=images, img_masks=img_masks, lang_tokens=lang_tokens, lang_masks=lang_masks
        )

    if isinstance(actions, tuple):
        actions = actions[0]
    assert tuple(actions.shape) == tuple(ref_actions.shape), (
        f"shape mismatch: {tuple(actions.shape)} vs {tuple(ref_actions.shape)}"
    )
    pcc = _compute_pcc(ref_actions, actions)
    print(f"[16-decode e2e PCC] PCC vs torch = {pcc:.6f}  (bar {PCC_MIN})")
    assert pcc >= PCC_MIN, f"16-chip decode e2e PCC {pcc:.6f} < {PCC_MIN}"
