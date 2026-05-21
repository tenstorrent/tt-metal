# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V3-DEBUG-ATTN: per-stage PCC diagnostic for vision attention.

Localizes where the 0.987 PCC drop happens. Runs each step of the attention
forward independently on the TT side AND on a CPU "instrumented HF" path,
and PCC-compares at each stage:

  Stage A: x → qkv_proj output
  Stage B: qkv output → q, k, v (after split)
  Stage C: q, k after RoPE rotation
  Stage D: SDPA output
  Stage E: after concat_heads + all_gather
  Stage F: after o_proj + all_gather (the final attention output)
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen3_6_galaxy_v2.tt.vision_attention_tp import Qwen36VisionAttentionTP, build_vision_rope_tensors
from models.demos.qwen3_6_galaxy_v2.tt.vision_model_args import Qwen36VisionModelArgs
from models.demos.qwen3_vl.reference.functional import apply_rotary_pos_emb_vision, qwen3_vision_transformer_preprocess
from models.tt_dit.parallel.manager import CCLManager


def _to_torch_chip0(tt_tensor, mesh_device):
    """Pull chip 0's view to torch."""
    t = ttnn.to_torch(tt_tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    return t[:1] if t.dim() == 4 else t[:1].unsqueeze(0)


def _hf_attention_per_stage(reference_attn, x_2d, cu_seqlens, cos_ref, sin_ref):
    """Mirror of Qwen3VLVisionAttention.forward but expose every intermediate.

    Args:
        x_2d: torch input [seq, hidden]
        cos_ref, sin_ref: shape [seq, head_dim] from qwen3_vision_transformer_preprocess
    Returns:
        dict with keys 'qkv', 'q', 'k', 'v', 'q_rot', 'k_rot', 'attn_out',
                       'attn_concat', 'final'.
    """
    # Mirrors HF Qwen3VLVisionAttention.forward
    seq_length = x_2d.shape[0]
    qkv = reference_attn.qkv(x_2d)  # [seq, 3*hidden]
    h = reference_attn.qkv.in_features
    num_heads = reference_attn.num_heads
    head_dim = h // num_heads
    # reshape to [seq, 3, num_heads, head_dim] → permute → [3, seq, num_heads, head_dim]
    qkv_reshaped = qkv.reshape(seq_length, 3, num_heads, head_dim).permute(1, 0, 2, 3)
    q, k, v = qkv_reshaped.unbind(0)  # each [seq, num_heads, head_dim]

    # RoPE
    q_rot, k_rot = apply_rotary_pos_emb_vision(q, k, cos_ref, sin_ref)

    # SDPA (non-causal). HF does it via scaled_dot_product_attention with
    # transpose to [num_heads, seq, head_dim].
    q_for_sdpa = q_rot.transpose(0, 1)  # [num_heads, seq, head_dim]
    k_for_sdpa = k_rot.transpose(0, 1)
    v_for_sdpa = v.transpose(0, 1)
    # Non-causal SDPA across the single packed sequence
    attn_out = torch.nn.functional.scaled_dot_product_attention(
        q_for_sdpa.unsqueeze(0),
        k_for_sdpa.unsqueeze(0),
        v_for_sdpa.unsqueeze(0),
        is_causal=False,
    ).squeeze(
        0
    )  # [num_heads, seq, head_dim]
    attn_out = attn_out.transpose(0, 1).contiguous()  # [seq, num_heads, head_dim]
    attn_concat = attn_out.reshape(seq_length, num_heads * head_dim)  # [seq, hidden]
    final = reference_attn.proj(attn_concat)

    return {
        "qkv": qkv,
        "q": q,
        "k": k,
        "v": v,
        "q_rot": q_rot,
        "k_rot": k_rot,
        "attn_out": attn_out,
        "attn_concat": attn_concat,
        "final": final,
    }


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4), "BH_GLX": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": True}], indirect=True)
@pytest.mark.parametrize("grid_h,grid_w", [(14, 14)])
def test_vision_attention_per_stage_pcc(grid_h, grid_w, mesh_device, reset_seeds, ensure_gc):
    if not os.environ.get("HF_MODEL"):
        os.environ["HF_MODEL"] = "Qwen/Qwen3.6-27B"

    seq_len = grid_h * grid_w
    model_args = Qwen36VisionModelArgs(mesh_device, dummy_weights=False, max_batch_size=1, max_seq_len=seq_len)
    vc = model_args.hf_config.vision_config
    H, NH, HD = vc.hidden_size, vc.num_heads, vc.hidden_size // vc.num_heads
    PHD = ((HD + 31) // 32) * 32  # padded_head_dim = 96 for qwen3.6

    # Reference attention (layer 0) + HF cos/sin
    reference_full = model_args.reference_vision_model()
    reference_attn = reference_full.blocks[0].attn
    hf_state = reference_attn.state_dict()
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]])
    cu_seqlens, (cos_ref, sin_ref) = qwen3_vision_transformer_preprocess(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        spatial_merge_size=vc.spatial_merge_size,
    )

    # Random input
    torch.manual_seed(0)
    x_2d = torch.randn(seq_len, H, dtype=torch.float32)

    # --- HF instrumented forward ---
    hf = _hf_attention_per_stage(reference_attn, x_2d, cu_seqlens, cos_ref, sin_ref)
    logger.info(
        f"HF reference shapes: qkv={tuple(hf['qkv'].shape)}, q={tuple(hf['q'].shape)}, "
        f"q_rot={tuple(hf['q_rot'].shape)}, attn_out={tuple(hf['attn_out'].shape)}, "
        f"final={tuple(hf['final'].shape)}"
    )

    # --- TT attention: build it ---
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    tt_attn = Qwen36VisionAttentionTP(
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        state_dict=hf_state,
        hidden_size=H,
        num_heads=NH,
        head_dim=HD,
        tp_mesh_axis=0,
        dtype=ttnn.bfloat16,
    )
    cos_tt, sin_tt = build_vision_rope_tensors(
        seq_len=seq_len,
        grid_thw=image_grid_thw,
        head_dim=HD,
        padded_head_dim=PHD,
        spatial_merge_size=vc.spatial_merge_size,
        mesh_device=mesh_device,
    )

    # --- TT per-stage forward (manually unrolling the attention forward) ---
    x_tt = ttnn.from_torch(
        x_2d.view(1, 1, seq_len, H),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Stage A: qkv_proj
    xqkv_tt = tt_attn.qkv_proj.forward(x_tt)
    if len(xqkv_tt.shape) == 3:
        xqkv_tt = ttnn.unsqueeze(xqkv_tt, 1)

    # Gather the FULL fused-qkv output (all 32 chips, each chip has 576 cols → all_gather makes 4608)
    # Pull per-chip and stack to compare with HF's qkv.
    # Per-chip xqkv shape: [1, 1, seq, 576] = [1, 1, seq, num_local_heads * 3 * PHD]
    # We can't easily compare this to HF qkv (shape [seq, 3*1152]) without reorganizing.
    # Skip Stage A direct PCC; verify stages B onward via the head-split output.

    # Stage B: split into q, k, v via nlp_create_qkv_heads
    q_tt, k_tt, v_tt = ttnn.experimental.nlp_create_qkv_heads(
        xqkv_tt,
        num_heads=tt_attn.num_local_heads,
        num_kv_heads=tt_attn.num_local_heads,
        transpose_k_heads=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.deallocate(xqkv_tt)

    # Compare CHIP-0 directly (has heads 0..1) WITHOUT all_gather, eliminating
    # any all_gather-related ambiguity. Per-chip q,k,v shape: [1, num_local=2, seq, PHD=96].
    q_chip0 = _to_torch_chip0(q_tt, mesh_device).squeeze(0).transpose(0, 1)[..., :HD]
    k_chip0 = _to_torch_chip0(k_tt, mesh_device).squeeze(0).transpose(0, 1)[..., :HD]
    v_chip0 = _to_torch_chip0(v_tt, mesh_device).squeeze(0).transpose(0, 1)[..., :HD]
    # shape: [seq, num_local=2, head_dim=72]; compare to HF first 2 heads
    p_q0, m_q0 = comp_pcc(hf["q"][:, 0:2, :], q_chip0, 0.99)
    p_k0, m_k0 = comp_pcc(hf["k"][:, 0:2, :], k_chip0, 0.99)
    p_v0, m_v0 = comp_pcc(hf["v"][:, 0:2, :], v_chip0, 0.99)
    logger.info(f"Stage B chip-0 (heads 0..1): q PCC={m_q0}  k PCC={m_k0}  v PCC={m_v0}")

    # Sample values for first token, head 0
    logger.info(f"  HF q[0,0,:5]   = {hf['q'][0, 0, :5].tolist()}")
    logger.info(f"  TT q[0,0,:5]   = {q_chip0[0, 0, :5].tolist()}")
    logger.info(f"  HF q[0,1,:5]   = {hf['q'][0, 1, :5].tolist()}")
    logger.info(f"  TT q[0,1,:5]   = {q_chip0[0, 1, :5].tolist()}")
    logger.info(f"  HF k[0,0,:5]   = {hf['k'][0, 0, :5].tolist()}")
    logger.info(f"  TT k[0,0,:5]   = {k_chip0[0, 0, :5].tolist()}")

    # Stage C: RoPE on q, k via rotary_embedding_llama — compare chip-0 directly
    q_rot_tt = ttnn.experimental.rotary_embedding_llama(
        q_tt,
        cos_tt,
        sin_tt,
        tt_attn._rope_transformation_mat,
        is_decode_mode=False,
    )
    k_rot_tt = ttnn.experimental.rotary_embedding_llama(
        k_tt,
        cos_tt,
        sin_tt,
        tt_attn._rope_transformation_mat,
        is_decode_mode=False,
    )

    q_rot_chip0 = _to_torch_chip0(q_rot_tt, mesh_device).squeeze(0).transpose(0, 1)[..., :HD]
    k_rot_chip0 = _to_torch_chip0(k_rot_tt, mesh_device).squeeze(0).transpose(0, 1)[..., :HD]
    p_qr0, m_qr0 = comp_pcc(hf["q_rot"][:, 0:2, :], q_rot_chip0, 0.99)
    p_kr0, m_kr0 = comp_pcc(hf["k_rot"][:, 0:2, :], k_rot_chip0, 0.99)
    logger.info(f"Stage C chip-0 (post RoPE):   q_rot PCC={m_qr0}  k_rot PCC={m_kr0}")
    logger.info(f"  HF q_rot[0,0,:5]  = {hf['q_rot'][0, 0, :5].tolist()}")
    logger.info(f"  TT q_rot[0,0,:5]  = {q_rot_chip0[0, 0, :5].tolist()}")

    # Stage D: SDPA — compare chip-0 directly
    attn_tt = ttnn.transformer.scaled_dot_product_attention(
        q_rot_tt,
        k_rot_tt,
        v_tt,
        attn_mask=None,
        is_causal=False,
        scale=HD**-0.5,
        program_config=tt_attn._sdpa_program_config(q_rot_tt.shape[2]),
        compute_kernel_config=tt_attn._sdpa_compute_kernel_config,
    )
    attn_chip0 = _to_torch_chip0(attn_tt, mesh_device).squeeze(0).transpose(0, 1)[..., :HD]
    p_a0, m_a0 = comp_pcc(hf["attn_out"][:, 0:2, :], attn_chip0, 0.99)
    logger.info(f"Stage D chip-0 (post SDPA):    attn_out PCC={m_a0}")
    logger.info(f"  HF attn_out[0,0,:5] = {hf['attn_out'][0, 0, :5].tolist()}")
    logger.info(f"  TT attn_out[0,0,:5] = {attn_chip0[0, 0, :5].tolist()}")

    # Stage E + F: concat_heads → all_gather → o_proj → all_gather (skip — we have final test elsewhere)
