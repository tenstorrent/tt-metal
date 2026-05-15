# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Sub-step PCC drilldown for ONE pi0.5 action-expert (AdaRMSGemmaBlock) layer.

Mirrors test_pcc_vlm_block_drilldown.py but for the expert block, which adds:
  - modulated RMSNorm (scale + shift from adarms_cond mod-Dense)
  - gated residual (gate * y + x, where gate is also from mod-Dense)

Sub-steps probed:
  [1] After modulated RMSNorm 1 (pre-attn)
  [2] After expert attention (pre-gate)
  [3] After attn gate × output  (gate multiply only)
  [4] After attn residual  (hidden + gated)
  [5] After modulated RMSNorm 2 (pre-FFW)
  [6] After MLP (pre-gate)
  [7] After MLP gate × output
  [8] After MLP residual  (full block output, piecewise)
  [9] Integrated block.forward()  (production path)

Where exactly the per-step velocity err⊥ originates inside one expert layer.
"""

import sys
from pathlib import Path

import pytest
import torch
import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.pi0_5.reference.torch_paligemma import Pi0_5PaliGemmaBackbone
from models.experimental.pi0_5.reference.torch_gemma import ada_rms_norm as torch_ada_rms_norm
from models.experimental.pi0_5.tt.ttnn_paligemma import Pi0_5PaliGemmaBackboneTTNN
from models.experimental.pi0_5.common.configs import SigLIPConfig, GemmaConfig, PaliGemmaConfig
from models.experimental.pi0_5.common.weight_loader import PI0WeightLoader


CHECKPOINT_PATH = str(Path(__file__).resolve().parents[2] / "weights" / "pi05_base")
SEED = 42


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() != b.numel():
        return -1.0
    ma, mb = a.mean(), b.mean()
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return 1.0 if torch.allclose(a, b, atol=1e-5) else 0.0
    return (((a - ma) * (b - mb)).mean() / (sa * sb)).item()


def cos_sim(a, b):
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
def test_expert_block_drilldown(device):
    torch.manual_seed(SEED)
    cfg = create_config()

    ckpt = Path(CHECKPOINT_PATH)
    if not ckpt.exists():
        pytest.skip(f"Checkpoint not found: {ckpt}")

    print("\n📋 Loading weights...")
    loader = PI0WeightLoader(str(ckpt))
    weights = {
        "vlm_vision": loader.get_vlm_vision_weights(),
        "vlm_language": loader.get_vlm_language_weights(),
        "vlm_projector": {},
        "action_expert": loader.get_action_expert_weights(),
    }
    state = loader._state_dict
    for k in state:
        if "multi_modal_projector" in k:
            new_k = k.replace("paligemma_with_expert.paligemma.model.multi_modal_projector.", "")
            new_k = new_k.replace("paligemma_with_expert.paligemma.multi_modal_projector.", "")
            weights["vlm_projector"][new_k] = state[k]

    model_torch = Pi0_5PaliGemmaBackbone(cfg, weights)
    model_ttnn = Pi0_5PaliGemmaBackboneTTNN(cfg, weights, device)

    # Use action-horizon-padded shape and Gemma expert width
    B, S = 1, 64
    W = cfg.expert_config.width  # 1024

    hidden = torch.randn(B, S, W) * 0.5  # match approximate post-prefix scale

    # adarms_cond: dim = expert_width (after time-MLP) = 1024
    adarms_cond = torch.randn(B, W) * 0.1

    # Pick expert block 0
    block_torch = model_torch.expert_blocks[0]
    block_ttnn = model_ttnn.expert_blocks[0]

    # RoPE for expert (max_seq_len = 64 for action_horizon)
    cos_t = model_torch.cos[:S]
    sin_t = model_torch.sin[:S]

    # No KV cache for this isolated test
    cos_ttnn = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_ttnn = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    adarms_cond_ttnn = ttnn.from_torch(
        adarms_cond.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )

    print(f"\nExpert block 0 drilldown — hidden={tuple(hidden.shape)} W={W}, adarms_cond={tuple(adarms_cond.shape)}\n")

    eps = cfg.expert_config.rms_norm_eps

    # ------- [1] After modulated RMSNorm 1 (pre-attn) -------
    normed_t, gate_attn_t = torch_ada_rms_norm(
        hidden, adarms_cond, block_torch.pre_attn_mod_weight, block_torch.pre_attn_mod_bias, eps
    )

    # TTNN side: compute mod on device via the block's mod_weight/bias,
    # then split into (sa1, ta, ga, sf1, tf, gf), and apply modulated rms norm
    # using the same path as production (_modulated_rms_norm).
    from models.experimental.pi0_5.tt.ttnn_gemma import _modulated_rms_norm, _split_modulation_6

    mod = ttnn.linear(
        adarms_cond_ttnn,
        block_ttnn.mod_weight,
        bias=block_ttnn.mod_bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=block_ttnn.core_grid,
        compute_kernel_config=block_ttnn.mod_compute_kernel_config,
    )
    sa1, ta, ga, sf1, tf, gf = _split_modulation_6(mod)
    ttnn.deallocate(mod)

    # Build sharded LN config the same way block.forward() does inline
    # (the expert block doesn't expose a _get_sharded_norm helper)
    from models.experimental.pi0_5.tt.ttnn_gemma import build_sharded_norm_pcfg

    m_tiles = S // 32
    hidden_tiles = cfg.expert_config.width // 32
    norm_cfg = build_sharded_norm_pcfg(m_tiles, hidden_tiles, max_grid_x=8, max_grid_y=max(1, m_tiles))
    if norm_cfg is not None:
        pc, memcfg_factory, _grid = norm_cfg
        sh_pc = pc
        sh_mc = memcfg_factory(B, S, S, cfg.expert_config.width)
    else:
        sh_pc, sh_mc = None, None
    normed_ttnn = _modulated_rms_norm(
        hidden_ttnn,
        sa1,
        ta,
        eps,
        pre_added=False,
        sharded_pcfg=sh_pc,
        sharded_memcfg=sh_mc,
    )
    if sh_pc is not None:
        normed_ttnn = ttnn.sharded_to_interleaved(normed_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    p = pcc(normed_t, to_t(normed_ttnn))
    print(
        f"  [1] After modulated RMSNorm 1 (pre-attn) ......  PCC={p:.6f}  cos={cos_sim(normed_t, to_t(normed_ttnn)):.6f}"
    )

    # ------- [2] After expert attention (pre-gate) -------
    attn_out_t, _ = block_torch.attention.forward(normed_t, cos_t, sin_t, None, None, None, False)
    attn_out_ttnn, _ = block_ttnn.attention.forward(normed_ttnn, cos_ttnn, sin_ttnn, None, None, None, False)
    p = pcc(attn_out_t, to_t(attn_out_ttnn))
    print(
        f"  [2] After expert attention (pre-gate) .........  PCC={p:.6f}  cos={cos_sim(attn_out_t, to_t(attn_out_ttnn)):.6f}"
    )

    # ------- [3] After attn gate × output -------
    gated_attn_t = gate_attn_t * attn_out_t
    gated_attn_ttnn = ttnn.mul(attn_out_ttnn, ga, memory_config=ttnn.L1_MEMORY_CONFIG)
    p = pcc(gated_attn_t, to_t(gated_attn_ttnn))
    print(
        f"  [3] After attn gate × output (gate*y) .........  PCC={p:.6f}  cos={cos_sim(gated_attn_t, to_t(gated_attn_ttnn)):.6f}"
    )

    # ------- [4] After attn residual -------
    resid1_t = hidden + gated_attn_t
    resid1_ttnn = ttnn.add(hidden_ttnn, gated_attn_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    p = pcc(resid1_t, to_t(resid1_ttnn))
    print(
        f"  [4] After attn residual (x + gate*y) ..........  PCC={p:.6f}  cos={cos_sim(resid1_t, to_t(resid1_ttnn)):.6f}"
    )

    # ------- [5] After modulated RMSNorm 2 (pre-FFW) -------
    normed2_t, gate_ffw_t = torch_ada_rms_norm(
        resid1_t, adarms_cond, block_torch.pre_ffw_mod_weight, block_torch.pre_ffw_mod_bias, eps
    )
    normed2_ttnn = _modulated_rms_norm(
        resid1_ttnn,
        sf1,
        tf,
        eps,
        pre_added=False,
        sharded_pcfg=sh_pc,
        sharded_memcfg=sh_mc,
    )
    if sh_pc is not None:
        normed2_ttnn = ttnn.sharded_to_interleaved(normed2_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    p = pcc(normed2_t, to_t(normed2_ttnn))
    print(
        f"  [5] After modulated RMSNorm 2 (pre-FFW) .......  PCC={p:.6f}  cos={cos_sim(normed2_t, to_t(normed2_ttnn)):.6f}"
    )

    # ------- [6] After MLP (pre-gate) -------
    mlp_out_t = block_torch.mlp.forward(normed2_t)
    mlp_out_ttnn = block_ttnn.mlp.forward(normed2_ttnn)
    p = pcc(mlp_out_t, to_t(mlp_out_ttnn))
    print(
        f"  [6] After MLP (pre-gate) ......................  PCC={p:.6f}  cos={cos_sim(mlp_out_t, to_t(mlp_out_ttnn)):.6f}"
    )

    # ------- [7] After MLP gate × output -------
    gated_mlp_t = gate_ffw_t * mlp_out_t
    gated_mlp_ttnn = ttnn.mul(mlp_out_ttnn, gf, memory_config=ttnn.L1_MEMORY_CONFIG)
    p = pcc(gated_mlp_t, to_t(gated_mlp_ttnn))
    print(
        f"  [7] After MLP gate × output (gate*y) ..........  PCC={p:.6f}  cos={cos_sim(gated_mlp_t, to_t(gated_mlp_ttnn)):.6f}"
    )

    # ------- [8] After MLP residual (piecewise full block output) -------
    out_t = resid1_t + gated_mlp_t
    out_ttnn = ttnn.add(resid1_ttnn, gated_mlp_ttnn, memory_config=ttnn.L1_MEMORY_CONFIG)
    p_full = pcc(out_t, to_t(out_ttnn))
    print(
        f"  [8] Piecewise full block output ...............  PCC={p_full:.6f}  cos={cos_sim(out_t, to_t(out_ttnn)):.6f}"
    )

    # ------- [9] Integrated block.forward() -------
    # Use a clone — same inputs again
    hidden_ttnn_v2 = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_ttnn_v2 = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    sin_ttnn_v2 = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    adarms_cond_ttnn_v2 = ttnn.from_torch(
        adarms_cond.unsqueeze(1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    out_torch_full, _ = block_torch.forward(hidden, cos_t, sin_t, adarms_cond, None, None, None, False)
    out_ttnn_full, _ = block_ttnn.forward(
        hidden_ttnn_v2, cos_ttnn_v2, sin_ttnn_v2, adarms_cond_ttnn_v2, None, None, None, False
    )
    p_int = pcc(out_torch_full, to_t(out_ttnn_full))
    print(
        f"  [9] Integrated block.forward() ................  PCC={p_int:.6f}  cos={cos_sim(out_torch_full, to_t(out_ttnn_full)):.6f}"
    )

    print("\n" + "=" * 70)
    print("  Expert block sub-step PCC drilldown")
    print("=" * 70)
    if p_full < 0.99:
        print(f"  ⚠ block output PCC = {p_full:.6f} — find the biggest drop in the table above.")
    else:
        print(f"  ✅ block output PCC = {p_full:.6f}")
