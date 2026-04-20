# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Per-op PCC for ``TextAttention.forward_decode`` vs PyTorch reference.

Uses ``capture_decode_stages`` to pull TTNN tensors to host and compares to a staged
reference built with the same weights (fp32) as the golden path in ``test_decode_pcc``.
After ``forward_decode``, reads the device KV cache and reports PCC for the filled
prefix and for the newly written token slot (isolates cache write vs historical drift).
"""

from typing import Dict

import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.demos.molmo2.reference.functional import _apply_rope


def compute_pcc(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute Pearson Correlation Coefficient between two tensors."""
    ref_flat = ref.flatten().float()
    test_flat = test.flatten().float()

    ref_mean = ref_flat.mean()
    test_mean = test_flat.mean()

    ref_centered = ref_flat - ref_mean
    test_centered = test_flat - test_mean

    numerator = (ref_centered * test_centered).sum()
    denominator = torch.sqrt((ref_centered**2).sum() * (test_centered**2).sum())

    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0

    return (numerator / denominator).item()


def _reference_decode_stages(
    hidden_states: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    kv_cache_k: torch.Tensor,
    kv_cache_v: torch.Tensor,
    current_pos: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rope_theta: float = 1_000_000.0,
) -> Dict[str, torch.Tensor]:
    """
    Stages aligned to ``TextAttention.forward_decode`` / ``capture_decode_stages`` keys.
    Layout convention: ``[1, batch, heads, dim]`` where TTNN uses that for Q/K/V decode.
    """
    batch_size = hidden_states.shape[0]
    num_kv_groups = num_heads // num_kv_heads

    q = F.linear(hidden_states, wq)
    k = F.linear(hidden_states, wk)
    v = F.linear(hidden_states, wv)

    q = q.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

    def rms_norm(x, weight, eps: float = 1e-5):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + eps)
        return x * weight

    # [B, nh, 1, d] -> TT layout [1, B, nh, d]
    def to_tt(x_bhn1d: torch.Tensor) -> torch.Tensor:
        return x_bhn1d.transpose(1, 2).contiguous()

    out: Dict[str, torch.Tensor] = {}
    out["q_after_heads"] = to_tt(q)
    out["k_after_heads"] = to_tt(k)
    out["v_after_heads"] = to_tt(v)

    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)
    out["q_after_rms"] = to_tt(q)
    out["k_after_rms"] = to_tt(k)

    q_for_rope = q.transpose(1, 2)
    k_for_rope = k.transpose(1, 2)
    position_ids = torch.full(
        (batch_size, 1),
        float(current_pos),
        dtype=torch.float32,
        device=hidden_states.device,
    )
    q_rot, k_rot = _apply_rope(q_for_rope, k_for_rope, position_ids, head_dim, rope_theta)
    q_rot = q_rot.transpose(1, 2)
    k_rot = k_rot.transpose(1, 2)

    out["q_after_rope"] = to_tt(q_rot)
    out["k_after_rope"] = to_tt(k_rot)
    out["v_sliced"] = to_tt(v)

    kv_k = kv_cache_k.clone()
    kv_v = kv_cache_v.clone()
    kv_k[:, :, current_pos : current_pos + 1, :] = k_rot
    kv_v[:, :, current_pos : current_pos + 1, :] = v

    # KV cache after this decode step (for PCC vs device cache after ``paged_update_cache``)
    out["kv_k_prefix_after_update"] = kv_k[:, :, : current_pos + 1, :].contiguous()
    out["kv_v_prefix_after_update"] = kv_v[:, :, : current_pos + 1, :].contiguous()
    out["kv_k_new_token"] = kv_k[:, :, current_pos : current_pos + 1, :].contiguous()
    out["kv_v_new_token"] = kv_v[:, :, current_pos : current_pos + 1, :].contiguous()

    k_full = kv_k[:, :, : current_pos + 1, :]
    v_full = kv_v[:, :, : current_pos + 1, :]
    k_expanded = k_full.repeat_interleave(num_kv_groups, dim=1)
    v_expanded = v_full.repeat_interleave(num_kv_groups, dim=1)

    scale = head_dim**-0.5
    attn_weights = torch.matmul(q_rot, k_expanded.transpose(-2, -1)) * scale
    attn_weights = F.softmax(attn_weights, dim=-1)
    attn_heads = torch.matmul(attn_weights, v_expanded)
    out["attn_after_sdpa"] = to_tt(attn_heads)

    attn_merged = attn_heads.transpose(1, 2).contiguous().view(batch_size, 1, num_heads * head_dim)
    out["attn_after_concat"] = attn_merged.unsqueeze(0)
    out["output"] = F.linear(attn_merged, wo).unsqueeze(0)

    out["q_before_sdpa"] = out["q_after_rope"].clone()
    return out


def test_text_attention_decode_stage_pcc():
    """Log PCC at each captured stage vs fp32 reference (single decode step)."""
    from models.demos.molmo2.demo.demo import load_model_weights
    from models.demos.molmo2.tt.text_attention import TextAttention
    from models.demos.molmo2.tt.text_model import init_decode_position, init_kv_cache
    from models.demos.molmo2.tt.text_rotary_setup import TextRotarySetup

    hidden_dim = 4096
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    seq_len = 128
    max_seq_len = 256
    layer_idx = 0
    rope_theta = 1_000_000.0

    state_dict = load_model_weights()
    prefix = f"model.transformer.blocks.{layer_idx}"
    att_proj = state_dict[f"{prefix}.self_attn.att_proj.weight"].float()
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    wq = att_proj[:q_dim, :]
    wk = att_proj[q_dim : q_dim + kv_dim, :]
    wv = att_proj[q_dim + kv_dim :, :]
    wo = state_dict[f"{prefix}.self_attn.attn_out.weight"].float()
    q_norm = state_dict[f"{prefix}.self_attn.q_norm.weight"].float()
    k_norm = state_dict[f"{prefix}.self_attn.k_norm.weight"].float()

    torch.manual_seed(42)
    prefill_hidden = torch.randn(1, seq_len, hidden_dim)
    ref_k_cache = torch.zeros(1, num_kv_heads, max_seq_len, head_dim)
    ref_v_cache = torch.zeros(1, num_kv_heads, max_seq_len, head_dim)

    from models.demos.molmo2.tests.test_decode_pcc import reference_attention_with_kv_cache

    for pos in range(seq_len):
        token_hidden = prefill_hidden[:, pos : pos + 1, :]
        _ = reference_attention_with_kv_cache(
            token_hidden,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            ref_k_cache,
            ref_v_cache,
            pos,
            num_heads,
            num_kv_heads,
            head_dim,
        )

    device = ttnn.open_device(device_id=0)
    try:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        attn_weight_dtype = ttnn.bfloat16
        kv_cache_dtype = ttnn.bfloat16

        ttnn_attn = TextAttention(
            mesh_device=device,
            state_dict=state_dict,
            layer_num=layer_idx,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=attn_weight_dtype,
        )
        rotary_setup = TextRotarySetup(
            mesh_device=device,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        transformation_mats = rotary_setup.get_transformation_mats()
        ttnn_kv_cache = init_kv_cache(
            device,
            num_layers=1,
            batch_size=1,
            num_kv_heads=num_kv_heads,
            max_seq_len=max_seq_len,
            head_dim=head_dim,
            dtype=kv_cache_dtype,
        )[0]

        ref_k_to_copy = ref_k_cache[:, :, :seq_len, :].to(torch.bfloat16)
        ref_v_to_copy = ref_v_cache[:, :, :seq_len, :].to(torch.bfloat16)
        k_ttnn = ttnn.from_torch(
            ref_k_to_copy,
            device=device,
            dtype=kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        v_ttnn = ttnn.from_torch(
            ref_v_to_copy,
            device=device,
            dtype=kv_cache_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        ttnn.fill_cache(ttnn_kv_cache[0], k_ttnn, batch_idx=0)
        ttnn.fill_cache(ttnn_kv_cache[1], v_ttnn, batch_idx=0)
        ttnn.deallocate(k_ttnn)
        ttnn.deallocate(v_ttnn)

        current_pos = init_decode_position(device, batch_size=1, initial_pos=seq_len)
        pos = seq_len
        torch.manual_seed(123)
        decode_hidden = torch.randn(1, 1, hidden_dim)

        kv_k_snap = ref_k_cache.clone()
        kv_v_snap = ref_v_cache.clone()
        ref_stages = _reference_decode_stages(
            decode_hidden,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            kv_k_snap,
            kv_v_snap,
            pos,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_theta,
        )

        decode_ttnn = ttnn.from_torch(
            decode_hidden.unsqueeze(0).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mesh_mapper,
        )
        rot_mats = rotary_setup.get_rot_mats_decode(torch.tensor([pos], dtype=torch.int32))
        tt_stages: Dict[str, torch.Tensor] = {}
        _ = ttnn_attn.forward_decode(
            decode_ttnn,
            rot_mats,
            transformation_mats["decode"],
            ttnn_kv_cache,
            current_pos,
            capture_decode_stages=tt_stages,
        )
        ttnn.synchronize_device(device)

        # --- KV cache on device after decode (same layout as init_kv_cache: [B, n_kv, S, D]) ---
        k_cache_ttnn = ttnn.to_torch(ttnn_kv_cache[0]).float()
        v_cache_ttnn = ttnn.to_torch(ttnn_kv_cache[1]).float()
        pos_end = pos + 1
        tt_k_prefix = k_cache_ttnn[:, :, :pos_end, :]
        tt_v_prefix = v_cache_ttnn[:, :, :pos_end, :]
        tt_k_new = k_cache_ttnn[:, :, pos : pos + 1, :]
        tt_v_new = v_cache_ttnn[:, :, pos : pos + 1, :]

        order = [
            "q_after_heads",
            "k_after_heads",
            "v_after_heads",
            "q_after_rms",
            "k_after_rms",
            "q_after_rope",
            "k_after_rope",
            "v_sliced",
            "q_before_sdpa",
            "attn_after_sdpa",
            "attn_after_concat",
            "output",
        ]
        logger.info("TextAttention.forward_decode vs reference — PCC by stage")
        logger.info("-" * 72)
        for key in order:
            if key not in tt_stages or key not in ref_stages:
                logger.warning(f"  {key}: missing (tt={key in tt_stages}, ref={key in ref_stages})")
                continue
            p = compute_pcc(ref_stages[key], tt_stages[key])
            logger.info(f"  {key:22s} PCC = {p:.6f}")

        logger.info("KV cache vs reference (after decode write; prefix = positions 0..pos inclusive)")
        logger.info("-" * 72)
        kv_pairs = [
            ("kv_k_prefix_after_update", tt_k_prefix),
            ("kv_v_prefix_after_update", tt_v_prefix),
            ("kv_k_new_token", tt_k_new),
            ("kv_v_new_token", tt_v_new),
        ]
        for key, tt_slice in kv_pairs:
            if key not in ref_stages:
                logger.warning(f"  {key}: missing reference")
                continue
            ref_slice = ref_stages[key].float()
            if ref_slice.shape != tt_slice.shape:
                logger.warning(f"  {key}: shape mismatch ref={tuple(ref_slice.shape)} tt={tuple(tt_slice.shape)}")
                continue
            p = compute_pcc(ref_slice, tt_slice)
            logger.info(f"  {key:22s} PCC = {p:.6f}")

        # Weakest stage helps pinpoint where to optimize; assert only final output loosely
        out_pcc = compute_pcc(ref_stages["output"], tt_stages["output"])
        assert out_pcc >= 0.75, f"Final output PCC {out_pcc} unexpectedly low (regression)"

    finally:
        ttnn.close_device(device)
