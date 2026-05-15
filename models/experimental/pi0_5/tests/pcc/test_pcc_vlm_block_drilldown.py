# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sub-block PCC drilldown for the PaliGemma VLM transformer block.

We previously found block-level PCC = 0.734 while every primitive
component (RMSNorm weights load, MLP, embed_image, etc) measures
0.999+ in isolation. This test localizes the drop INSIDE one block
by probing each sub-step:

    input ──► RMSNorm1 ──► Attention ──► (+ residual) ──► RMSNorm2 ──► MLP ──► (+ residual) ──► output

For each labeled point we compare TTNN intermediate vs torch
intermediate and report cosine similarity / PCC.

Run:
    pytest models/experimental/pi0_5/tests/pcc/test_pcc_vlm_block_drilldown.py -x -s
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone
from models.experimental.pi0_5.reference.torch_gemma import rms_norm as torch_rms_norm
from models.experimental.pi0_5.tt.ttnn_paligemma import Pi0_5PaliGemmaBackboneTTNN
from models.experimental.pi0_5.tt.ttnn_gemma import rms_norm_ttnn
from models.experimental.pi0_5.common.configs import SigLIPConfig, GemmaConfig, PaliGemmaConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


CHECKPOINT_PATH = str(Path(__file__).resolve().parents[2] / "weights" / "pi05_base")
SEED = 42


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() != b.numel():
        return -1.0
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return 1.0 if torch.allclose(a, b, atol=1e-5) else 0.0
    cov = ((a - ma) * (b - mb)).mean()
    return (cov / (sa * sb)).item()


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def to_t(x):
    if isinstance(x, ttnn.Tensor):
        return ttnn.to_torch(x)
    return x


def create_config() -> PaliGemmaConfig:
    return PaliGemmaConfig(
        siglip_config=SigLIPConfig(
            hidden_size=1152,
            intermediate_size=4304,
            num_hidden_layers=27,
            num_attention_heads=16,
            image_size=224,
            patch_size=14,
        ),
        vlm_config=GemmaConfig(
            width=2048,
            depth=18,
            mlp_dim=16384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        ),
        expert_config=GemmaConfig(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        ),
        max_seq_len=512,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_vlm_block_drilldown(device):
    torch.manual_seed(SEED)
    cfg = create_config()

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        pytest.skip(f"Checkpoint not found: {ckpt}")

    print("\n📋 Loading PaliGemma weights...")
    loader = PI0WeightLoader(str(ckpt))
    weights = {
        "vlm_vision": loader.get_vlm_vision_weights(),
        "vlm_language": loader.get_vlm_language_weights(),
        "vlm_projector": {},
        "action_expert": loader.get_action_expert_weights(),
    }

    # Patch in the multimodal projector keys the constructor expects.
    state = loader._state_dict
    for k in state:
        if "multi_modal_projector" in k:
            new_k = k.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
            new_k = new_k.replace("paligemma_with_expert.paligemma.multi_modal_projector.", "")
            weights["vlm_projector"][new_k] = state[k]

    model_torch = Pi0_5PaliGemmaBackbone(cfg, weights)
    model_ttnn = Pi0_5PaliGemmaBackboneTTNN(cfg, weights, device)

    # Single block, seq_len=64 (matches upstream test), no mask / no cache.
    B, S = 1, 64
    W = cfg.vlm_config.width
    hidden = torch.randn(B, S, W)

    block_torch = model_torch.vlm_blocks[0]
    block_ttnn = model_ttnn.vlm_blocks[0]

    # Pre-load RoPE
    cos_t = model_torch.cos[:S]
    sin_t = model_torch.sin[:S]

    # ------------------------------------------------------------------
    # Step 1 — RMSNorm1 (pre-attn)
    # ------------------------------------------------------------------
    normed_t = torch_rms_norm(hidden, block_torch.input_layernorm_weight, cfg.vlm_config.rms_norm_eps)

    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sh_pc, sh_mc = block_ttnn._get_sharded_norm(S)
    normed_ttnn = rms_norm_ttnn(
        hidden_ttnn,
        block_ttnn.input_layernorm_weight,
        cfg.vlm_config.rms_norm_eps,
        sharded_pcfg=sh_pc,
        sharded_memcfg=sh_mc,
    )
    if sh_pc is not None:
        normed_ttnn = ttnn.sharded_to_interleaved(normed_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)

    pcc_ln1 = compute_pcc(normed_t, to_t(normed_ttnn))
    cos_ln1 = cos_sim(normed_t, to_t(normed_ttnn))
    print(f"\n  [1] After RMSNorm1 (pre-attn):  PCC={pcc_ln1:.6f}  cos={cos_ln1:.6f}")

    # ------------------------------------------------------------------
    # Step 2 — Attention output (PRE-residual)
    # ------------------------------------------------------------------
    attn_out_t, _ = block_torch.attention.forward(normed_t, cos_t, sin_t, None, None, None, False)
    cos_ttnn = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_ttnn = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    attn_out_ttnn, _ = block_ttnn.attention.forward(normed_ttnn, cos_ttnn, sin_ttnn, None, None, None, False)

    pcc_attn = compute_pcc(attn_out_t, to_t(attn_out_ttnn))
    cos_attn = cos_sim(attn_out_t, to_t(attn_out_ttnn))
    print(f"  [2] After Attention (pre-resid): PCC={pcc_attn:.6f}  cos={cos_attn:.6f}")

    # ------------------------------------------------------------------
    # Step 3 — Residual add (attention)
    # ------------------------------------------------------------------
    resid1_t = hidden + attn_out_t
    resid1_ttnn = ttnn.add(hidden_ttnn, attn_out_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    pcc_r1 = compute_pcc(resid1_t, to_t(resid1_ttnn))
    cos_r1 = cos_sim(resid1_t, to_t(resid1_ttnn))
    print(f"  [3] After attn + residual:       PCC={pcc_r1:.6f}  cos={cos_r1:.6f}")

    # ------------------------------------------------------------------
    # Step 4 — RMSNorm2 (pre-MLP)
    # ------------------------------------------------------------------
    normed2_t = torch_rms_norm(resid1_t, block_torch.post_attention_layernorm_weight, cfg.vlm_config.rms_norm_eps)
    normed2_ttnn = rms_norm_ttnn(
        resid1_ttnn,
        block_ttnn.post_attention_layernorm_weight,
        cfg.vlm_config.rms_norm_eps,
        sharded_pcfg=sh_pc,
        sharded_memcfg=sh_mc,
    )
    if sh_pc is not None:
        normed2_ttnn = ttnn.sharded_to_interleaved(normed2_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    pcc_ln2 = compute_pcc(normed2_t, to_t(normed2_ttnn))
    cos_ln2 = cos_sim(normed2_t, to_t(normed2_ttnn))
    print(f"  [4] After RMSNorm2 (pre-MLP):    PCC={pcc_ln2:.6f}  cos={cos_ln2:.6f}")

    # ------------------------------------------------------------------
    # Step 5 — MLP output (pre-residual)
    # ------------------------------------------------------------------
    mlp_out_t = block_torch.mlp.forward(normed2_t)
    mlp_out_ttnn = block_ttnn.mlp.forward(normed2_ttnn)
    pcc_mlp = compute_pcc(mlp_out_t, to_t(mlp_out_ttnn))
    cos_mlp = cos_sim(mlp_out_t, to_t(mlp_out_ttnn))
    print(f"  [5] After MLP (pre-resid):       PCC={pcc_mlp:.6f}  cos={cos_mlp:.6f}")

    # ------------------------------------------------------------------
    # Step 6 — Final residual (block output)
    # ------------------------------------------------------------------
    out_t = resid1_t + mlp_out_t
    out_ttnn = ttnn.add(resid1_ttnn, mlp_out_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    pcc_full = compute_pcc(out_t, to_t(out_ttnn))
    cos_full = cos_sim(out_t, to_t(out_ttnn))
    print(f"  [6] Block output (full):         PCC={pcc_full:.6f}  cos={cos_full:.6f}")

    # ------------------------------------------------------------------
    # Step 7 — Integrated block.forward() on the SAME input
    # ------------------------------------------------------------------
    # Reproduce inputs to avoid seed-drift between the two paths.
    hidden_ttnn_v2 = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_ttnn_v2 = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_ttnn_v2 = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_torch_full, _ = block_torch.forward(hidden, cos_t, sin_t)
    out_ttnn_full, _ = block_ttnn.forward(hidden_ttnn_v2, cos_ttnn_v2, sin_ttnn_v2)
    pcc_integrated = compute_pcc(out_torch_full, to_t(out_ttnn_full))
    cos_integrated = cos_sim(out_torch_full, to_t(out_ttnn_full))
    print(f"  [7] Integrated block.forward():  PCC={pcc_integrated:.6f}  cos={cos_integrated:.6f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("  VLM Block[0] sub-step PCC drilldown")
    print("=" * 70)
    print(f"  [1] After RMSNorm1 .................. {pcc_ln1:.6f}")
    print(f"  [2] After Attention ................. {pcc_attn:.6f}  <-- attention path")
    print(f"  [3] After attn residual ............. {pcc_r1:.6f}")
    print(f"  [4] After RMSNorm2 .................. {pcc_ln2:.6f}")
    print(f"  [5] After MLP ....................... {pcc_mlp:.6f}")
    print(f"  [6] Piecewise full block ............ {pcc_full:.6f}")
    print(f"  [7] Integrated block.forward() ...... {pcc_integrated:.6f}  <-- the production path")
    print("=" * 70)
    if pcc_integrated < pcc_full - 0.05:
        print("  ⚠ Integrated path differs from piecewise — bug in block.forward!")
    else:
        print("  ✅ Integrated path matches piecewise — model ops are clean")
