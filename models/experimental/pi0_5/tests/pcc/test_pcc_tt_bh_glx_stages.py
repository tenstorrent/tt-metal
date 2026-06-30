# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage PCC tests for the in-scope pipeline targets.

Covers the prefill stage in two parallelism modes and the vision stage on a
single chip — all vs the PyTorch reference:
    test_prefill_tp4_pcc       — TP=N VLM prefill (PI0_TP chips) vs torch. (1×8 path with PI0_TP=8.)
    test_prefill_tp4_perf_dummy — TP=N prefill timing smoke (no torch ref).
    test_prefill_tp1_pcc       — single-chip VLM prefill vs torch.
    test_vision_tp1_pcc        — single-chip SigLIP vision tower + projector vs torch.

Run a single test:
    pytest -xvs models/experimental/pi0_5/tests/pcc/test_pcc_tt_bh_glx_stages.py::test_prefill_tp1_pcc
"""

from __future__ import annotations

import torch
import ttnn

import os
from pathlib import Path

import pytest

from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh


CHECKPOINT_DIR = Path(os.environ.get("PI05_CHECKPOINT_DIR", "/home/tt-admin/pi05_cache/pi05_libero_upstream"))
SEED = 42


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-6 or s2 < 1e-6:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    cov = torch.mean((t1 - m1) * (t2 - m2))
    return (cov / (s1 * s2)).item()


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_prefill_tp4_pcc():
    """TP=4 VLM prefill (all 18 blocks on a 4-chip mesh) vs torch reference. Target PCC ≥ 0.99."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill_tp4 import StagePrefillTP4

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=5)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    weights = loader.categorized_weights

    B = 1
    seq_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "1024"))
    torch.manual_seed(SEED)
    prefix_embs = torch.randn(B, seq_len, cfg.vlm_config.width) * 0.5

    # Torch reference (full 18-block chain + final RMS norm). Skipped under
    # PI0_SKIP_TORCH_REF=1 — for profiling runs where PCC is already confirmed,
    # the CPU reference just adds runtime and pollutes the host-side report.
    ref_out = None
    if not os.environ.get("PI0_SKIP_TORCH_REF"):
        ref = TorchBackbone(cfg, weights)
        with torch.no_grad():
            ref_out, _ = ref.forward_vlm(prefix_embs, attention_mask=None, position_ids=None, use_cache=False)

    with open_prefill_tp4_mesh(tp=int(os.environ.get("PI0_TP", "4")), l1_small_size=24576) as mesh:
        stage = StagePrefillTP4(cfg, weights, mesh)
        prefix_ttnn = ttnn.from_torch(
            prefix_embs,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_ttnn, _ = stage.run(prefix_ttnn, attention_mask=None, position_ids=None)
        # Output is replicated on all 4 chips — take the first chip's copy.
        out = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0])

    if ref_out is None:
        print(f"\n✅ Prefill TP=4 stage ran (torch ref skipped)  (shape {tuple(out.shape)})")
        return
    assert out.shape == ref_out.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref_out.shape)}"
    pcc = _compute_pcc(ref_out, out)
    print(f"\n✅ Prefill TP=4 stage PCC vs torch: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


def test_prefill_tp4_perf_dummy():
    """TP=4/8 VLM prefill with DUMMY weights — no checkpoint required.

    Profiling: op shapes/dtypes (hence device-kernel timing) are identical to the
    real-weight run; weight *values* don't affect the profile. PCC is also valid —
    it measures TTNN-vs-torch fidelity on the SAME weights, independent of whether
    they're real or random — so we run the torch reference on the same dummy weights
    and check PCC too (skip via PI0_SKIP_TORCH_REF=1 for pure profiling).

    Mesh tp from PI0_TP (default 4); seq from PI0_VLM_CHUNK_SIZE. Run under tracy
    exactly like test_prefill_tp4_pcc.
    """
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.tt.tt_bh_glx.stage_prefill_tp4 import StagePrefillTP4

    cfg = Pi0_5ModelConfig(num_denoising_steps=5)
    v = cfg.vlm_config
    torch.manual_seed(SEED)

    def _w(*shape):
        return torch.randn(*shape) * 0.02

    vw = {}
    for i in range(v.depth):
        p = f"model.layers.{i}."
        vw[p + "self_attn.q_proj.weight"] = _w(v.num_heads * v.head_dim, v.width)
        vw[p + "self_attn.k_proj.weight"] = _w(v.num_kv_heads * v.head_dim, v.width)
        vw[p + "self_attn.v_proj.weight"] = _w(v.num_kv_heads * v.head_dim, v.width)
        vw[p + "self_attn.o_proj.weight"] = _w(v.width, v.num_heads * v.head_dim)
        vw[p + "mlp.gate_proj.weight"] = _w(v.mlp_dim, v.width)
        vw[p + "mlp.up_proj.weight"] = _w(v.mlp_dim, v.width)
        vw[p + "mlp.down_proj.weight"] = _w(v.width, v.mlp_dim)
        vw[p + "input_layernorm.weight"] = _w(v.width)
        vw[p + "post_attention_layernorm.weight"] = _w(v.width)
    vw["model.norm.weight"] = _w(v.width)
    weights = {"vlm_language": vw}

    seq_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "1024"))
    prefix_embs = torch.randn(1, seq_len, v.width) * 0.5

    # Torch reference on the SAME dummy weights → valid PCC. Replicate forward_vlm
    # directly from GemmaBlock (the 18 VLM blocks + RoPE + final norm) — avoids
    # instantiating the full backbone, which would also need vision/expert weights
    # that forward_vlm never uses. Skip via PI0_SKIP_TORCH_REF=1 for pure profiling.
    ref_out = None
    if not os.environ.get("PI0_SKIP_TORCH_REF"):
        from models.experimental.pi0_5.reference.torch_gemma import GemmaBlock, precompute_freqs_cis, rms_norm

        cos, sin = precompute_freqs_cis(v.head_dim, cfg.max_seq_len, v.rope_base)
        with torch.no_grad():
            h = prefix_embs
            for i in range(v.depth):
                pfx = f"model.layers.{i}."
                bw = {k[len(pfx) :]: val for k, val in vw.items() if k.startswith(pfx)}
                h, _ = GemmaBlock(v, bw, i).forward(h, cos, sin, None, None, None, False)
            ref_out = rms_norm(h, vw["model.norm.weight"], v.rms_norm_eps)

    with open_prefill_tp4_mesh(tp=int(os.environ.get("PI0_TP", "4")), l1_small_size=24576) as mesh:
        stage = StagePrefillTP4(cfg, weights, mesh)
        prefix_ttnn = ttnn.from_torch(
            prefix_embs,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out_ttnn, _ = stage.run(prefix_ttnn, attention_mask=None, position_ids=None)
        ttnn.synchronize_device(mesh)
        out = ttnn.to_torch(ttnn.get_device_tensors(out_ttnn)[0])

    assert tuple(out.shape) == (1, seq_len, v.width), f"unexpected shape {tuple(out.shape)}"
    if ref_out is None:
        print(f"\n✅ Prefill TP4 (dummy weights) ran  (shape {tuple(out.shape)})")
        return
    pcc = _compute_pcc(ref_out, out)
    print(f"\n✅ Prefill TP4 (dummy weights) PCC vs torch: {pcc:.6f}  (shape {tuple(out.shape)})")
    # Looser bar than the real-weight test (0.99): RANDOM weights + bf8 MLP outputs
    # compound worse over 18 layers than trained weights (no structure for bf8 to
    # exploit) — measured ~0.981 dummy vs 0.9939 real. This is a structural-fidelity
    # check (TTNN matches torch); the real-weight test_prefill_tp4_pcc is the bar.
    assert pcc >= 0.97, f"PCC {pcc:.6f} < 0.97 (dummy-weight fidelity bar)"


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_prefill_tp1_pcc():
    """TP=1 single-device VLM prefill (PaliGemmaBackboneTTNN.forward_vlm — all 18
    blocks on one chip) vs torch reference. Mirrors test_prefill_tp4_pcc (same
    input, same torch reference) for an apples-to-apples TP=1-vs-TP=4 profile.
    Device id from PI0_DEVICE_ID (default 0 → first TT_VISIBLE_DEVICES chip).
    Target PCC ≥ 0.99."""
    from models.experimental.pi0_5.common.checkpoint_meta import action_horizon_from_checkpoint
    from models.experimental.pi0_5.common.configs import Pi0_5ModelConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone as TorchBackbone
    from models.experimental.pi0_5.tt.ttnn_paligemma import Pi0_5PaliGemmaBackboneTTNN

    cfg = Pi0_5ModelConfig(action_horizon=action_horizon_from_checkpoint(CHECKPOINT_DIR), num_denoising_steps=5)
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    weights = loader.categorized_weights

    B = 1
    seq_len = int(os.environ.get("PI0_VLM_CHUNK_SIZE", "1024"))
    torch.manual_seed(SEED)
    prefix_embs = torch.randn(B, seq_len, cfg.vlm_config.width) * 0.5

    # Torch reference (full 18-block chain + final RMS norm) — identical to tp4.
    # Skipped under PI0_SKIP_TORCH_REF=1 for profiling runs (PCC already confirmed).
    ref_out = None
    if not os.environ.get("PI0_SKIP_TORCH_REF"):
        ref = TorchBackbone(cfg, weights)
        with torch.no_grad():
            ref_out, _ = ref.forward_vlm(prefix_embs, attention_mask=None, position_ids=None, use_cache=False)

    device = ttnn.open_device(device_id=int(os.environ.get("PI0_DEVICE_ID", "0")), l1_small_size=24576)
    try:
        backbone = Pi0_5PaliGemmaBackboneTTNN(cfg, weights, device)
        prefix_ttnn = ttnn.from_torch(
            prefix_embs,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # PI0_PERF_ITERS>1: run forward_vlm repeatedly so the ops CSV holds
        # multiple identical op blocks (compile run + steady-state runs). Device
        # perf is read from the LAST run — the compile pass has corrupt multi-core
        # kernel durations (cf. models/tt_transformers split_compile_and_trace).
        iters = int(os.environ.get("PI0_PERF_ITERS", "1"))
        for _ in range(iters):
            out_ttnn, _ = backbone.forward_vlm(prefix_ttnn, attention_mask=None, position_ids=None, use_cache=False)
            ttnn.synchronize_device(device)
        out = ttnn.to_torch(out_ttnn)
    finally:
        ttnn.close_device(device)

    if ref_out is None:
        print(f"\n✅ Prefill TP=1 stage ran ×{iters} (torch ref skipped)  (shape {tuple(out.shape)})")
        return
    assert out.shape == ref_out.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref_out.shape)}"
    pcc = _compute_pcc(ref_out, out)
    print(f"\n✅ Prefill TP=1 stage PCC vs torch: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= 0.99, f"PCC {pcc:.6f} < 0.99"


@pytest.mark.skipif(
    not (CHECKPOINT_DIR / "model.safetensors").exists(),
    reason=f"checkpoint not found at {CHECKPOINT_DIR}",
)
def test_vision_tp1_pcc():
    """TP=1 single-device SigLIP vision tower + mm_projector vs torch reference.

    This is the SigLIP path the production e2e actually runs (ttnn_siglip /
    backbone.embed_image, single chip, bs=PI0_NUM_CAMERAS). Mirrors
    test_prefill_tp1_pcc: device id from PI0_DEVICE_ID, honors PI0_PERF_ITERS
    (repeat for profiling) and PI0_SKIP_TORCH_REF. Target PCC ≥ 0.997.
    """
    from models.experimental.pi0_5.common.configs import SigLIPConfig
    from models.experimental.pi0_5.common.weight_loader import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_siglip import SigLIPVisionTower as TorchSigLIPVisionTower
    from models.experimental.pi0_5.reference.torch_siglip import MultiModalProjector as TorchMMProjector
    from models.experimental.pi0_5.tt.ttnn_siglip import MultiModalProjectorTTNN, SigLIPVisionTowerTTNN

    cfg = SigLIPConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        image_size=224,
        patch_size=14,
    )
    loader = Pi0_5WeightLoader(str(CHECKPOINT_DIR))
    vision_w = loader.categorized_weights["vlm_vision"]
    projector_w = loader.categorized_weights["vlm_projector"]

    bs = int(os.environ.get("PI0_NUM_CAMERAS", "3"))
    torch.manual_seed(SEED)
    pixel_values = torch.randn(bs, 3, cfg.image_size, cfg.image_size)

    # Torch reference: SigLIP vision tower + mm_projector → (bs, 256, 2048).
    # Skipped under PI0_SKIP_TORCH_REF=1 for profiling runs.
    ref_out = None
    if not os.environ.get("PI0_SKIP_TORCH_REF"):
        ref_tower = TorchSigLIPVisionTower(cfg, vision_w)
        ref_proj = TorchMMProjector(projector_w)
        with torch.no_grad():
            ref_out = ref_proj.forward(ref_tower.forward(pixel_values))

    device = ttnn.open_device(device_id=int(os.environ.get("PI0_DEVICE_ID", "0")), l1_small_size=24576)
    try:
        tower = SigLIPVisionTowerTTNN(cfg, vision_w, device)
        proj = MultiModalProjectorTTNN(projector_w, device)
        # The device-side fold patch-embed (PI0_SIGLIP_USE_FOLD, prod default)
        # expects a ttnn input — the e2e host-permutes BCHW→NHWC and uploads
        # before calling. Mirror that so the prod fold path runs self-contained.
        # (last_dim == in_channels=3 → the device-reshape branch of _forward_fold.)
        pix_in = ttnn.from_torch(
            pixel_values.permute(0, 2, 3, 1).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # PI0_PERF_ITERS>1: repeat so the ops CSV holds steady-state op blocks
        # (read the LAST run; the compile pass has corrupt multi-core durations).
        iters = int(os.environ.get("PI0_PERF_ITERS", "1"))
        for _ in range(iters):
            out_ttnn = proj.forward(tower.forward(pix_in))
            ttnn.synchronize_device(device)
        out = ttnn.to_torch(out_ttnn)
    finally:
        ttnn.close_device(device)

    if ref_out is None:
        print(f"\n✅ SigLIP TP=1 stage ran ×{iters} (torch ref skipped)  (shape {tuple(out.shape)})")
        return
    assert out.shape == ref_out.shape, f"shape mismatch: {tuple(out.shape)} vs {tuple(ref_out.shape)}"
    pcc = _compute_pcc(ref_out, out)
    print(f"\n✅ SigLIP TP=1 stage PCC vs torch: {pcc:.6f}  (shape {tuple(out.shape)})")
    assert pcc >= 0.997, f"PCC {pcc:.6f} < 0.997"
