# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-stage PCC bisect for Option C full-L1 — isolates where PCC drifts.

The end-to-end PCC test (`test_pcc_option_c_vs_torch.py`) shows a cliff
between vlm=expert=9 (PCC ≈ 0.86) and vlm=expert=18 (PCC ≈ 0.40) with all
full-L1 flags on. This test drives the Option C pipeline stage by stage,
reading intermediates back to host and comparing to a torch reference at
matched depth so we can see WHICH stage first loses signal.

Captures, in order:

  [A] After Stage 0 vision:    prefix_hidden vs torch.embed_prefix
  [B] After Stage 1 prefill:   per-layer (K, V) vs torch.forward_vlm cache
  [C] After KV migration:      migrated_kv vs torch.forward_vlm cache
                               (host-bounce sanity check)
  [D] After Stage 2 step 0:    velocity vs torch._denoise_forward(t=1.0)

The torch reference uses the SAME mask/position contract as the TT pipeline
currently builds (all-zeros prefix and joint masks → all-unmasked); that's
the production path of the existing e2e test. Mask divergence between TT
all-zero and the openpi proper mask is its own investigation — gated below
behind PI0_OC_STAGE_PCC_PROPER_MASK=1.

Run:
    PI0_OC_STAGE_PCC=1 pytest -xvs \\
      models/experimental/pi0_5/tests/pcc/test_pcc_option_c_stage_bisect.py

Env vars (all optional; defaults match the full-L1 working flag set):

    PI0_OC_STAGE_VLM_DEPTH=18          # VLM transformer depth
    PI0_OC_STAGE_EXPERT_DEPTH=18       # Expert transformer depth
    PI0_OC_STAGE_PCC_PROPER_MASK=1     # Use openpi-style padding-aware masks
    PI0_OC_STAGE_DEVICE_SIGLIP=1       # SigLIP on 8 vision chips
    PI0_OC_STAGE_VISION_WEIGHTS_L1=1   # Vision weights → L1
    PI0_OC_STAGE_PREFILL_TP=2          # Prefill TP=2
    PI0_OC_STAGE_WEIGHTS_L1=1          # Prefill weights → L1
    PI0_OC_STAGE_WEIGHTS_L1_MLP_ONLY=1 # Q/K/V/O stay DRAM
    PI0_OC_STAGE_DENOISE_LAYER_PAIRED=1
    PI0_OC_STAGE_DENOISE_WEIGHTS_L1=1
    PI0_OC_STAGE_CHECKPOINT=<path>     # Override checkpoint path
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import (
    PaliGemmaConfig,
    Pi0_5ModelConfig,
    SigLIPConfig,
)
from models.experimental.pi0_5.tt.option_c.mesh_setup import open_galaxy_mesh
from models.experimental.pi0_5.tt.option_c.pipeline import Pi0_5PipelineC
from models.experimental.pi0_5.tt.option_c.stages import build_shrunk_layout
from models.experimental.pi0_5.tt.option_c.transport import send_activation_via_host


# ---------- gating + env knobs ------------------------------------------------

PCC_ENABLED = os.environ.get("PI0_OC_STAGE_PCC") == "1"
pytestmark = pytest.mark.skipif(not PCC_ENABLED, reason="set PI0_OC_STAGE_PCC=1 to run the per-stage PCC bisect")

LANG_SEQ_LEN = 256
ACTION_HORIZON = 10
ACTION_HORIZON_PADDED = 32
NUM_DENOISE_STEPS = int(os.environ.get("PI0_OC_STAGE_STEPS", "10"))
VLM_DEPTH = int(os.environ.get("PI0_OC_STAGE_VLM_DEPTH", "18"))
EXPERT_DEPTH = int(os.environ.get("PI0_OC_STAGE_EXPERT_DEPTH", "18"))
SEED = int(os.environ.get("PI0_OC_STAGE_SEED", "42"))

# Full-L1 flags — default to the working flag combo so this test exercises
# the broken config out of the box.
DEVICE_SIGLIP = os.environ.get("PI0_OC_STAGE_DEVICE_SIGLIP", "1") == "1"
VISION_WEIGHTS_L1 = os.environ.get("PI0_OC_STAGE_VISION_WEIGHTS_L1", "1") == "1"
PREFILL_TP_SIZE = int(os.environ.get("PI0_OC_STAGE_PREFILL_TP", "2"))
PREFILL_WEIGHTS_L1 = os.environ.get("PI0_OC_STAGE_WEIGHTS_L1", "1") == "1"
PREFILL_WEIGHTS_L1_MLP_ONLY = os.environ.get("PI0_OC_STAGE_WEIGHTS_L1_MLP_ONLY", "1") == "1"
PREFILL_WEIGHTS_L1_SKIP_KV = os.environ.get("PI0_OC_STAGE_WEIGHTS_L1_SKIP_KV") == "1"
_dlp = os.environ.get("PI0_OC_STAGE_DENOISE_LAYER_PAIRED", "1")
DENOISE_LAYER_PAIRED_L1 = _dlp == "1"
DENOISE_WEIGHTS_L1 = os.environ.get("PI0_OC_STAGE_DENOISE_WEIGHTS_L1", "1") == "1"
LAYER_PAIRED_L1 = os.environ.get("PI0_OC_STAGE_LAYER_PAIRED") == "1"
PROPER_MASK = os.environ.get("PI0_OC_STAGE_PCC_PROPER_MASK") == "1"

# Per-stage PCC thresholds — informational. Test passes as long as it
# completes; the report tells us where the drift starts.
PCC_VISION_TARGET = 0.99
PCC_PREFILL_PER_LAYER_TARGET = 0.95
PCC_VELOCITY_TARGET = 0.95

_REAL_CKPT = os.environ.get("PI0_OC_STAGE_CHECKPOINT", "/home/tt-admin/pi05_cache/pi05_libero_upstream")


# ---------- helpers -----------------------------------------------------------


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    t1 = a.flatten().float()
    t2 = b.flatten().float()
    if t1.numel() != t2.numel():
        return -1.0
    m1, m2 = torch.mean(t1), torch.mean(t2)
    s1, s2 = torch.std(t1), torch.std(t2)
    if s1 < 1e-9 or s2 < 1e-9:
        return 1.0 if torch.allclose(t1, t2, atol=1e-5) else 0.0
    return (torch.mean((t1 - m1) * (t2 - m2)) / (s1 * s2)).item()


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() != b.numel():
        return -1.0
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().unsqueeze(0), b.flatten().float().unsqueeze(0)
    ).item()


def _to_host(t: "ttnn.Tensor") -> torch.Tensor:
    """First-shard read-back; bf16/bf8 → torch keeps dtype."""
    shards = ttnn.get_device_tensors(t)
    return ttnn.to_torch(shards[0])


def _crop_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Crop both tensors to the per-axis min size. Handles tile-padding mismatch."""
    if a.shape == b.shape:
        return a, b
    if a.ndim != b.ndim:
        return a, b  # shape mismatch we can't fix here — caller will see -1 PCC
    slc = tuple(slice(0, min(a.shape[i], b.shape[i])) for i in range(a.ndim))
    return a[slc], b[slc]


def _pcc_safe(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    """Crop-to-min-shape, then return (pcc, cos, l2). -1/-1/inf on rank mismatch."""
    if a.ndim != b.ndim or a.numel() == 0 or b.numel() == 0:
        return -1.0, -1.0, float("inf")
    a, b = _crop_to_match(a, b)
    p = _pcc(a, b)
    c = _cos(a, b)
    l = (a.float() - b.float()).norm().item()
    return p, c, l


def _make_shrunk_torch_config() -> Pi0_5ModelConfig:
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
    cfg.vlm_config.depth = VLM_DEPTH
    cfg.expert_config.depth = EXPERT_DEPTH
    return cfg


# ---------- the test ----------------------------------------------------------


@pytest.mark.timeout(2400)
def test_option_c_stage_bisect_vs_torch():
    if not Path(_REAL_CKPT, "model.safetensors").exists():
        pytest.skip(f"Real checkpoint not found at {_REAL_CKPT}")

    from models.experimental.pi0_5.common import Pi0_5WeightLoader
    from models.experimental.pi0_5.reference.torch_pi0_5_model import Pi0_5Model

    print("\n" + "=" * 80)
    print(f"  Option C PER-STAGE PCC BISECT  vlm={VLM_DEPTH}  expert={EXPERT_DEPTH}")
    print("=" * 80)
    print(
        f"  flags: device_siglip={DEVICE_SIGLIP}  vision_l1={VISION_WEIGHTS_L1}  "
        f"prefill_tp={PREFILL_TP_SIZE}  prefill_l1={PREFILL_WEIGHTS_L1}  "
        f"mlp_only={PREFILL_WEIGHTS_L1_MLP_ONLY}  denoise_paired={DENOISE_LAYER_PAIRED_L1}  "
        f"denoise_l1={DENOISE_WEIGHTS_L1}  proper_mask={PROPER_MASK}"
    )
    print("=" * 80)

    weight_loader = Pi0_5WeightLoader(_REAL_CKPT)
    weights = weight_loader.categorized_weights
    torch_cfg = _make_shrunk_torch_config()

    # ---- torch reference --------------------------------------------------
    torch_model = Pi0_5Model(torch_cfg, weight_loader)

    # ---- shared inputs ----------------------------------------------------
    gen = torch.Generator().manual_seed(SEED)
    pixel_values = torch.randn(1, 3, 224, 224, generator=gen, dtype=torch.float32)
    lang_tokens = torch.zeros(1, LANG_SEQ_LEN, dtype=torch.int64)
    lang_tokens[:, :32] = torch.randint(0, 256000, (1, 32), generator=gen)
    lang_masks = torch.zeros(1, LANG_SEQ_LEN, dtype=torch.bool)
    lang_masks[:, :32] = True
    img_masks_torch = [torch.ones(1, dtype=torch.bool)]
    state = torch.randn(1, torch_cfg.state_dim, generator=gen, dtype=torch.float32)
    x_0 = torch.randn(1, ACTION_HORIZON, torch_cfg.action_dim, generator=gen, dtype=torch.float32)
    x_0_padded = torch.zeros(1, ACTION_HORIZON_PADDED, torch_cfg.action_dim, dtype=torch.float32)
    x_0_padded[:, :ACTION_HORIZON, :] = x_0

    # ---- run torch reference, capturing intermediates ---------------------
    print("\n>> Running torch reference (depth-matched), capturing intermediates ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        prefix_embs_t, prefix_pad_masks_t, prefix_att_masks_t = torch_model.embed_prefix(
            [pixel_values], img_masks_torch, lang_tokens, lang_masks
        )

        # Build masks + positions that mirror the TT pipeline contract.
        # By default TT passes all-zeros (all-unmasked) — match that here so
        # any KV PCC difference can be attributed to math, not to masks.
        # `PROPER_MASK=1` enables openpi-style padding-aware masks on BOTH
        # sides for the secondary investigation.
        S_prefix = prefix_embs_t.shape[1]
        if PROPER_MASK:
            from models.experimental.pi0_5.reference.torch_pi0_5_model import (
                _build_prefix_mask_and_pos,
            )

            position_ids_t, attention_mask_4d_t = _build_prefix_mask_and_pos(
                prefix_pad_masks_t, prefix_att_masks_t, prefix_embs_t.dtype
            )
        else:
            position_ids_t = None
            attention_mask_4d_t = torch.zeros(1, 1, S_prefix, S_prefix, dtype=prefix_embs_t.dtype)

        _, vlm_cache_t = torch_model.backbone.forward_vlm(
            prefix_embs_t,
            attention_mask=attention_mask_4d_t,
            position_ids=position_ids_t,
            use_cache=True,
        )

        # Step-0 velocity: t=1.0 (the first integration step).
        t_scalar = torch.tensor([1.0], dtype=torch.float32)
        velocity_t = torch_model._denoise_forward(
            noisy_actions=x_0,
            timestep=t_scalar,
            kv_cache=vlm_cache_t,
            state=state,
            prefix_pad_masks=prefix_pad_masks_t if PROPER_MASK else None,
        )
    print(f"   torch ran in {(time.perf_counter() - t0):.1f}s, S_prefix={S_prefix}")

    # ---- run Option C TT pipeline stage by stage --------------------------
    print("\n>> Running Option C TT pipeline stage by stage ...")
    pali_cfg = PaliGemmaConfig()
    shrunk_layout = build_shrunk_layout(vlm_depth=VLM_DEPTH, expert_depth=EXPERT_DEPTH)
    lang_token_ids_int32 = lang_tokens.to(torch.int32)

    enable_fabric = PREFILL_TP_SIZE > 1
    l1_small_size = (
        24576
        if (
            PREFILL_TP_SIZE > 1 or DENOISE_WEIGHTS_L1 or VISION_WEIGHTS_L1 or DENOISE_LAYER_PAIRED_L1 or LAYER_PAIRED_L1
        )
        else None
    )

    pcc_report: List[Tuple[str, float, float, float]] = []  # (label, pcc, cos, l2)

    with open_galaxy_mesh(shrunk_layout, l1_small_size=l1_small_size, enable_fabric=enable_fabric) as (
        _parent,
        submeshes,
    ):
        pipe = Pi0_5PipelineC(
            layout=shrunk_layout,
            submeshes=submeshes,
            config=pali_cfg,
            weights=weights,
            denoise_steps=NUM_DENOISE_STEPS,
            action_horizon=ACTION_HORIZON,
            action_dim=32,
            layer_paired_l1=LAYER_PAIRED_L1,
            device_siglip=DEVICE_SIGLIP,
            vision_weights_l1=VISION_WEIGHTS_L1,
            prefill_tp_size=PREFILL_TP_SIZE,
            prefill_weights_l1=PREFILL_WEIGHTS_L1,
            prefill_weights_l1_mlp_only=PREFILL_WEIGHTS_L1_MLP_ONLY,
            prefill_weights_l1_skip_kv=PREFILL_WEIGHTS_L1_SKIP_KV,
            denoise_layer_paired_l1=DENOISE_LAYER_PAIRED_L1,
            denoise_weights_l1=DENOISE_WEIGHTS_L1,
        )
        pipe.initialize()

        # ---- [A] Stage 0 vision ----------------------------------------------
        print("\n[A] Stage 0 vision ...")
        t_s0 = time.perf_counter()
        prefix_hidden_s0 = pipe.stage_0.forward(pixel_values, lang_token_ids_int32)
        prefix_hidden_host = _to_host(prefix_hidden_s0)
        print(f"    shapes: torch={tuple(prefix_embs_t.shape)} tt={tuple(prefix_hidden_host.shape)}")
        pcc_A, cos_A, l2_A = _pcc_safe(prefix_embs_t, prefix_hidden_host)
        pcc_report.append(("[A] Stage 0 vision (prefix_hidden)", pcc_A, cos_A, l2_A))
        print(f"    {pcc_report[-1]}  ({(time.perf_counter() - t_s0):.1f}s)")

        # Ship prefix_hidden to prefill stage's first chip.
        prefill_in_submesh = pipe.stage_1.first_chip_submesh
        prefix_hidden_s1 = send_activation_via_host(prefix_hidden_s0, prefill_in_submesh)
        ttnn.deallocate(prefix_hidden_s0)

        # ---- [B] Stage 1 prefill (per-layer KV) -------------------------------
        print("\n[B] Stage 1 prefill (per-layer KV) ...")
        t_s1 = time.perf_counter()
        S_prefix_tt = prefix_hidden_s1.shape[1]
        if PROPER_MASK:
            # When the proper-mask path is on, also pass the proper mask + pos
            # ids to the TT pipeline so the two paths match. We re-use the
            # torch mask values, uploaded to the TT submesh.
            mask_prefix_s1 = pipe._upload_replicated(
                attention_mask_4d_t.float(),
                prefill_in_submesh,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            position_ids_s1 = pipe._upload_replicated(
                position_ids_t.float(),
                prefill_in_submesh,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            mask_prefix_s1 = pipe._build_or_upload_prefix_mask(None, S_prefix_tt, prefill_in_submesh)
            position_ids_s1 = None

        h_after_1 = pipe.stage_1.forward(
            prefix_hidden_s1,
            attention_mask=mask_prefix_s1,
            position_ids=position_ids_s1,
            use_cache=True,
        )
        # Read per-layer KV from the prefill slice's get_kv_cache() emitter,
        # before migration. The KV lives on the prefill submesh (replicated
        # across (2,1) sub-meshes for TP=2; one shard per chip in paired).
        per_layer_kv_pre: List[Optional[Tuple["ttnn.Tensor", "ttnn.Tensor"]]] = [None] * pipe.config.vlm_config.depth
        for global_idx, kv in pipe.stage_1.get_kv_cache():
            per_layer_kv_pre[global_idx] = kv

        # Per-layer KV PCC. Layer i's K, V are in torch's vlm_cache_t[i].
        pcc_K_layers: List[float] = []
        pcc_V_layers: List[float] = []
        for i in range(VLM_DEPTH):
            kv_tt = per_layer_kv_pre[i]
            if kv_tt is None:
                pcc_K_layers.append(-1.0)
                pcc_V_layers.append(-1.0)
                continue
            k_tt, v_tt = kv_tt
            k_host = _to_host(k_tt)
            v_host = _to_host(v_tt)
            if i == 0:
                print(
                    f"    layer 0 KV shapes: torch K={tuple(vlm_cache_t[i][0].shape)} "
                    f"tt K={tuple(k_host.shape)}; torch V={tuple(vlm_cache_t[i][1].shape)} "
                    f"tt V={tuple(v_host.shape)}"
                )
            pK, _, _ = _pcc_safe(vlm_cache_t[i][0], k_host)
            pV, _, _ = _pcc_safe(vlm_cache_t[i][1], v_host)
            pcc_K_layers.append(pK)
            pcc_V_layers.append(pV)
        # Report the worst (informative) layer.
        worst_K_idx = int(torch.tensor(pcc_K_layers).argmin())
        worst_V_idx = int(torch.tensor(pcc_V_layers).argmin())
        print(f"    per-layer K PCC: min={pcc_K_layers[worst_K_idx]:.4f} @ layer {worst_K_idx}")
        print(f"    per-layer V PCC: min={pcc_V_layers[worst_V_idx]:.4f} @ layer {worst_V_idx}")
        print(f"    per-layer K PCC: " + " ".join(f"{p:.3f}" for p in pcc_K_layers))
        print(f"    per-layer V PCC: " + " ".join(f"{p:.3f}" for p in pcc_V_layers))
        pcc_report.append(
            (
                "[B] Stage 1 K worst",
                pcc_K_layers[worst_K_idx],
                0.0,
                0.0,
            )
        )
        pcc_report.append(
            (
                "[B] Stage 1 V worst",
                pcc_V_layers[worst_V_idx],
                0.0,
                0.0,
            )
        )
        print(f"    ({(time.perf_counter() - t_s1):.1f}s)")
        ttnn.deallocate(h_after_1)

        # ---- [C] After KV migration (host bounce sanity) ---------------------
        print("\n[C] KV migration (host bounce) ...")
        t_kv = time.perf_counter()
        denoise_micro = pipe.stage_2.micro_submeshes if pipe.stage_2.layer_paired_l1 else None
        pipe.kv_migrator.migrate_layer_paired(per_layer_kv_pre, denoise_micro_submeshes=denoise_micro)
        migrated = pipe.kv_migrator.as_list(VLM_DEPTH)
        pcc_K_mig: List[float] = []
        pcc_V_mig: List[float] = []
        for i in range(VLM_DEPTH):
            if migrated[i] is None:
                pcc_K_mig.append(-1.0)
                pcc_V_mig.append(-1.0)
                continue
            k_tt, v_tt = migrated[i]
            k_host = _to_host(k_tt)
            v_host = _to_host(v_tt)
            pK, _, _ = _pcc_safe(vlm_cache_t[i][0], k_host)
            pV, _, _ = _pcc_safe(vlm_cache_t[i][1], v_host)
            pcc_K_mig.append(pK)
            pcc_V_mig.append(pV)
        worst_K_idx = int(torch.tensor(pcc_K_mig).argmin())
        worst_V_idx = int(torch.tensor(pcc_V_mig).argmin())
        print(f"    migrated K PCC: min={pcc_K_mig[worst_K_idx]:.4f} @ layer {worst_K_idx}")
        print(f"    migrated V PCC: min={pcc_V_mig[worst_V_idx]:.4f} @ layer {worst_V_idx}")
        pcc_report.append(("[C] KV migration K worst", pcc_K_mig[worst_K_idx], 0.0, 0.0))
        pcc_report.append(("[C] KV migration V worst", pcc_V_mig[worst_V_idx], 0.0, 0.0))
        print(f"    ({(time.perf_counter() - t_kv):.1f}s)")

        # ---- [D] Stage 2 single-step velocity (t=1.0) ------------------------
        print("\n[D] Stage 2 single-step velocity ...")
        t_s2 = time.perf_counter()
        denoise_in_submesh = pipe.stage_2.first_chip_submesh
        # Upload the SHARED x_0 (not random; matches torch step-0 input).
        noisy_on_denoise = pipe._upload_replicated(x_0_padded, denoise_in_submesh, dtype=ttnn.bfloat16)
        joint_mask_on_denoise = pipe._build_or_upload_joint_mask(
            None,
            S_prefix=S_prefix_tt,
            S_suffix_padded=noisy_on_denoise.shape[1],
            submesh=denoise_in_submesh,
        )

        # Mimic _denoise_with_per_step_timing's first iteration (i=0, t=1.0).
        B = noisy_on_denoise.shape[0]
        adarms_cond = pipe.stage_2.suffix.embed_adarms_cond(1.0, batch_size=B)
        suffix_h = pipe.stage_2.suffix.embed_actions(noisy_on_denoise)
        velocity_hidden = pipe.stage_2.slice.forward(
            suffix_h,
            adarms_cond,
            prefix_kv_cache=migrated,
            attention_mask=joint_mask_on_denoise,
        )
        ttnn.deallocate(suffix_h)
        ttnn.deallocate(adarms_cond)
        if (
            pipe.stage_2.layer_paired_l1
            and pipe.stage_2.micro_submeshes is not None
            and len(pipe.stage_2.micro_submeshes) > 1
        ):
            velocity_hidden_first = send_activation_via_host(velocity_hidden, pipe.stage_2.micro_submeshes[0])
            ttnn.deallocate(velocity_hidden)
            velocity_hidden = velocity_hidden_first
        v_t_tt = pipe.stage_2.suffix.project_output(velocity_hidden)
        ttnn.deallocate(velocity_hidden)

        v_tt_host_full = _to_host(v_t_tt)
        ttnn.deallocate(v_t_tt)
        v_tt_host = v_tt_host_full[:, :ACTION_HORIZON, :32]
        print(f"    velocity shapes: torch={tuple(velocity_t.shape)} " f"tt(cropped)={tuple(v_tt_host.shape)}")
        pcc_D, cos_D, l2_D = _pcc_safe(velocity_t, v_tt_host)
        pcc_report.append(("[D] Stage 2 velocity step 0", pcc_D, cos_D, l2_D))
        print(f"    {pcc_report[-1]}  ({(time.perf_counter() - t_s2):.1f}s)")

    # ---- summary ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    print(f"  {'stage':<40}  {'pcc':>8}  {'cos':>8}  {'L2':>10}")
    for label, p, c, l in pcc_report:
        print(f"  {label:<40}  {p:>8.4f}  {c:>8.4f}  {l:>10.4f}")
    print("=" * 80)

    # The test passes as long as it completes — the report itself is the
    # signal. Optionally tighten with explicit targets for CI later.
    assert pcc_report[0][1] >= -1.0  # sanity: nothing crashed
