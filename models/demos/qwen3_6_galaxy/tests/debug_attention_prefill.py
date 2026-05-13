"""Per-op debug for prefill forward pass - find where PCC drops."""
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")
import ttnn

_SNAPSHOT = (
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)
_LAYER = 3
_B, _T, _H = 1, 32, 5120
_N_Q, _N_KV, _HD = 24, 4, 256

TILE = 32
EPS = 1e-6


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    if a.shape[0] < 2:
        return float("nan")
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def rms_norm_ref(x, w_raw, eps=1e-6):
    """x: [..., hd], w_raw: [hd] (raw weight, zero-centered). Returns same shape."""
    xf = x.float()
    norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * (1.0 + w_raw.float())).to(x.dtype)


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_partial_rope_ref(x, cos, sin, rd=64):
    """x: [B, n, T, hd], cos/sin: [1, 1, T, rd]. Returns [B, n, T, hd]."""
    x_rot = x[..., :rd]
    x_pass = x[..., rd:]
    x_rot_out = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([x_rot_out, x_pass], dim=-1)


def main():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.STRICT_INIT, None, ttnn.FabricTensixConfig.DISABLED
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))
    try:
        _run(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def to_host(t, mesh, n=_B):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:n]


def _run(mesh):
    from models.demos.qwen3_6_27b.reference.hf_loader import load_qwen36_tensors
    from models.demos.qwen3_6_galaxy.tt.llama_attention import (
        TtQwen36GatedAttention,
        _flat_to_heads,
        _make_qknorm_weight_tt,
        _qknorm_flat_to_heads,
    )
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    prefix = f"model.language_model.layers.{_LAYER}.self_attn"
    keys = [
        f"{prefix}.{k}"
        for k in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight"]
    ]
    raw = load_qwen36_tensors(keys)
    sd = {
        k.split(".")[-2] + "." + k.split(".")[-1]: raw[f"{prefix}.{k.split('.')[-2]}.{k.split('.')[-1]}"].float()
        for k in keys
    }
    # Re-build sd properly
    sd = {
        "q_proj.weight": raw[f"{prefix}.q_proj.weight"].float(),
        "k_proj.weight": raw[f"{prefix}.k_proj.weight"].float(),
        "v_proj.weight": raw[f"{prefix}.v_proj.weight"].float(),
        "o_proj.weight": raw[f"{prefix}.o_proj.weight"].float(),
        "q_norm.weight": raw[f"{prefix}.q_norm.weight"].float(),
        "k_norm.weight": raw[f"{prefix}.k_norm.weight"].float(),
    }

    args = TtQwen36ModelArgs(mesh)
    n_cols = 4
    n_q_pc = _N_Q // n_cols  # 6
    n_kv_pc = _N_KV // n_cols  # 1
    q_dim_pc = n_q_pc * _HD  # 1536
    g_dim_pc = n_q_pc * _HD  # 1536
    k_dim_pc = n_kv_pc * _HD  # 256
    v_dim_pc = n_kv_pc * _HD  # 256

    compute = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # Build QK norm weights
    q_norm_w = _make_qknorm_weight_tt(sd["q_norm.weight"], mesh)
    k_norm_w = _make_qknorm_weight_tt(sd["k_norm.weight"], mesh)

    torch.manual_seed(10)
    x_cpu = torch.randn(_B, _T, _H, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_cpu,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # ---- REFERENCE: compute col 0's Q, K, V, gate ----
    wq = sd["q_proj.weight"]  # [12288, 5120]
    wk = sd["k_proj.weight"]  # [1024, 5120]
    wv = sd["v_proj.weight"]  # [1024, 5120]
    wo = sd["o_proj.weight"]  # [5120, 6144]

    # De-interleave q_proj: [n_q, 2, hd, H]
    q_2hd = wq.reshape(_N_Q, 2, _HD, _H)
    wq_native = q_2hd[:, 0, :, :].reshape(_N_Q * _HD, _H)  # [6144, H]
    wgate_native = q_2hd[:, 1, :, :].reshape(_N_Q * _HD, _H)  # [6144, H]

    # Col 0's weights (heads 0-5)
    wq_col0 = wq_native[: n_q_pc * _HD]  # [1536, H]
    wgate_col0 = wgate_native[: n_q_pc * _HD]  # [1536, H]
    wk_col0 = wk[: n_kv_pc * _HD]  # [256, H]
    wv_col0 = wv[: n_kv_pc * _HD]  # [256, H]

    x_f = x_cpu.float()
    q_flat_ref = x_f @ wq_col0.float().T  # [B, T, 1536]
    gate_flat_ref = x_f @ wgate_col0.float().T  # [B, T, 1536]
    k_flat_ref = x_f @ wk_col0.float().T  # [B, T, 256]
    v_flat_ref = x_f @ wv_col0.float().T  # [B, T, 256]

    # Build TTNN wqkvg for col 0 only to debug
    col_block = torch.cat([wq_col0, wgate_col0, wk_col0, wv_col0], dim=0)  # [3584, H]
    wqkvg_col0 = col_block.T.contiguous().bfloat16()  # [H, 3584]
    wqkvg_tt = ttnn.from_torch(
        wqkvg_col0,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    total_pc = q_dim_pc + g_dim_pc + k_dim_pc + v_dim_pc

    # TTNN QKVg projection (col 0)
    xqkvg_tt = ttnn.linear(
        x_tt, wqkvg_tt, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute
    )
    xqkvg_host = to_host(xqkvg_tt, mesh)
    # Build ref
    wqkvg_ref = torch.cat([wq_col0, wgate_col0, wk_col0, wv_col0], dim=0).T.float()
    xqkvg_ref = x_f @ wqkvg_ref  # [B, T, 3584]
    print(f"\n=== Step 1: QKVg projection ===")
    print(f"  PCC(ttnn_qkvg, ref_qkvg) = {pcc(xqkvg_host.float(), xqkvg_ref.bfloat16().float()):.6f}")

    # Split
    q_flat_tt = ttnn.slice(xqkvg_tt, [0, 0, 0], [_B, _T, q_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_flat_tt = ttnn.slice(
        xqkvg_tt, [0, 0, q_dim_pc], [_B, _T, q_dim_pc + g_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_flat_tt = ttnn.slice(
        xqkvg_tt,
        [0, 0, q_dim_pc + g_dim_pc],
        [_B, _T, q_dim_pc + g_dim_pc + k_dim_pc],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v_flat_tt = ttnn.slice(
        xqkvg_tt, [0, 0, q_dim_pc + g_dim_pc + k_dim_pc], [_B, _T, total_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    xqkvg_tt.deallocate(True)

    q_flat_host = to_host(q_flat_tt, mesh)
    k_flat_host = to_host(k_flat_tt, mesh)
    print(f"\n=== Step 2: Split Q, gate, K, V ===")
    print(f"  PCC(q_flat, ref_q_flat) = {pcc(q_flat_host.float(), q_flat_ref.bfloat16().float()):.6f}")
    print(f"  PCC(k_flat, ref_k_flat) = {pcc(k_flat_host.float(), k_flat_ref.bfloat16().float()):.6f}")

    # QK norm
    q_normed_tt = _qknorm_flat_to_heads(q_flat_tt, q_norm_w, EPS, _B, n_q_pc, _T, _HD, compute)
    k_normed_tt = _qknorm_flat_to_heads(k_flat_tt, k_norm_w, EPS, _B, n_kv_pc, _T, _HD, compute)
    q_flat_tt.deallocate(True)
    k_flat_tt.deallocate(True)

    q_normed_host = to_host(q_normed_tt, mesh)  # [1, 6, 32, 256]
    k_normed_host = to_host(k_normed_tt, mesh)  # [1, 1, 32, 256]

    # Reference: q_norm per head
    q_raw_heads = q_flat_ref.bfloat16().reshape(_B, _T, n_q_pc, _HD)  # [1, 32, 6, 256]
    q_normed_ref = rms_norm_ref(q_raw_heads, sd["q_norm.weight"])  # [1, 32, 6, 256]
    q_normed_ref_bhth = q_normed_ref.permute(0, 2, 1, 3)  # [1, 6, 32, 256]
    k_raw_heads = k_flat_ref.bfloat16().reshape(_B, _T, n_kv_pc, _HD)
    k_normed_ref = rms_norm_ref(k_raw_heads, sd["k_norm.weight"])
    k_normed_ref_bhth = k_normed_ref.permute(0, 2, 1, 3)  # [1, 1, 32, 256]

    print(f"\n=== Step 3: QK-norm ===")
    print(f"  PCC(q_normed, ref_q_normed) = {pcc(q_normed_host.float(), q_normed_ref_bhth.bfloat16().float()):.6f}")
    print(f"  PCC(k_normed, ref_k_normed) = {pcc(k_normed_host.float(), k_normed_ref_bhth.bfloat16().float()):.6f}")
    print(f"  q_normed head 0 [0,0,0,:5]: {q_normed_host[0,0,0,:5].float()}")
    print(f"  ref    head 0 [0,0,0,:5]: {q_normed_ref_bhth[0,0,0,:5].float()}")
    print(f"  q_normed head 3 [0,3,0,:5]: {q_normed_host[0,3,0,:5].float()}")
    print(f"  ref    head 3 [0,3,0,:5]: {q_normed_ref_bhth[0,3,0,:5].float()}")

    # RoPE
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(_T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=_HD,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    # cos/sin: [1, T, 64] → [1, 1, T, 64]
    cos_ = cos.unsqueeze(1)
    sin_ = sin.unsqueeze(1)
    cos_tt = ttnn.from_torch(
        cos_,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    sin_tt = ttnn.from_torch(
        sin_,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention

    attn = TtQwen36GatedAttention(mesh, args, sd, layer_num=_LAYER)

    q_rot_tt = attn._apply_partial_rope(q_normed_tt, cos_tt, sin_tt)
    k_rot_tt = attn._apply_partial_rope(k_normed_tt, cos_tt, sin_tt)

    q_rot_host = to_host(q_rot_tt, mesh)
    k_rot_host = to_host(k_rot_tt, mesh)

    q_rot_ref = apply_partial_rope_ref(q_normed_ref_bhth.float(), cos_.float(), sin_.float())
    k_rot_ref = apply_partial_rope_ref(k_normed_ref_bhth.float(), cos_.float(), sin_.float())

    print(f"\n=== Step 4: RoPE ===")
    print(f"  PCC(q_rot, ref_q_rot) = {pcc(q_rot_host.float(), q_rot_ref.bfloat16().float()):.6f}")
    print(f"  PCC(k_rot, ref_k_rot) = {pcc(k_rot_host.float(), k_rot_ref.bfloat16().float()):.6f}")

    # V
    v_normed_tt = _flat_to_heads(v_flat_tt, _B, n_kv_pc, _T, _HD)
    v_flat_tt.deallocate(True)
    v_normed_host = to_host(v_normed_tt, mesh)
    v_ref = v_flat_ref.bfloat16().reshape(_B, _T, n_kv_pc, _HD).permute(0, 2, 1, 3)
    print(f"\n=== V (no norm) ===")
    print(f"  PCC(v, ref_v) = {pcc(v_normed_host.float(), v_ref.bfloat16().float()):.6f}")

    # SDPA for col 0 only (prefill, no KV cache)
    gqa_pc = n_q_pc // n_kv_pc  # 6
    k_exp_tt = ttnn.repeat_interleave(k_rot_tt, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_exp_tt = ttnn.repeat_interleave(v_normed_tt, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_rot_tt.deallocate(True)
    v_normed_tt.deallocate(True)

    # Causal mask
    causal = torch.zeros(_B, 1, _T, _T, dtype=torch.bfloat16)
    causal = causal.masked_fill(torch.triu(torch.ones(_T, _T, dtype=torch.bool), diagonal=1), float("-inf"))
    mask_tt = ttnn.from_torch(
        causal,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    scale = _HD**-0.5
    attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_rot_tt,
        k_exp_tt,
        v_exp_tt,
        is_causal=False,
        attn_mask=mask_tt,
        scale=scale,
        compute_kernel_config=compute,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    mask_tt.deallocate(True)
    k_exp_tt.deallocate(True)
    v_exp_tt.deallocate(True)
    q_rot_tt.deallocate(True)

    attn_out_host = to_host(attn_out_tt, mesh)  # [1, 6, 32, 256]

    # Reference SDPA
    k_exp_ref = k_rot_ref.repeat_interleave(gqa_pc, dim=1)  # [1, 6, 32, 256]
    v_exp_ref = v_ref.float().repeat_interleave(gqa_pc, dim=1)
    scores_ref = torch.matmul(q_rot_ref, k_exp_ref.transpose(-2, -1)) * scale  # [1, 6, 32, 32]
    causal_f = causal.expand(1, 6, _T, _T).float()
    scores_ref = scores_ref + causal_f
    attn_ref = torch.softmax(scores_ref, dim=-1)
    attn_out_ref = torch.matmul(attn_ref, v_exp_ref)  # [1, 6, 32, 256]

    print(f"\n=== Step 5: SDPA ===")
    print(f"  PCC(attn_out, ref) = {pcc(attn_out_host.float(), attn_out_ref.bfloat16().float()):.6f}")

    # Output gate: _heads_to_flat then sigmoid(gate) * attn_flat
    from models.demos.qwen3_6_galaxy.tt.llama_attention import _heads_to_flat

    attn_flat_tt = _heads_to_flat(attn_out_tt, _B, n_q_pc, _T, _HD)
    attn_out_tt.deallocate(True)
    attn_flat_host = to_host(attn_flat_tt, mesh)

    # Reference: attn_out.transpose(1,2).reshape(B, T, -1)
    attn_out_ref_3d = attn_out_ref.transpose(1, 2).reshape(_B, _T, n_q_pc * _HD)  # [1, 32, 1536]
    print(f"\n=== Step 6: _heads_to_flat (attn_out → [B,T,n*hd]) ===")
    print(f"  PCC(attn_flat, ref) = {pcc(attn_flat_host.float(), attn_out_ref_3d.bfloat16().float()):.6f}")

    # gate_flat_tt: [B, T, 1536] from wqkvg output
    gate_flat_host = to_host(gate_flat_tt, mesh)
    gate_ref_3d = gate_flat_ref.bfloat16()  # [1, 32, 1536]
    print(f"\n=== Step 7: gate_flat ===")
    print(f"  PCC(gate_flat, ref) = {pcc(gate_flat_host.float(), gate_ref_3d.float()):.6f}")

    gate_sig_tt = ttnn.sigmoid(gate_flat_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_flat_tt.deallocate(True)
    gated_tt = ttnn.multiply(attn_flat_tt, gate_sig_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_flat_tt.deallocate(True)
    gate_sig_tt.deallocate(True)

    gated_host = to_host(gated_tt, mesh)
    gate_ref_sig = torch.sigmoid(gate_ref_3d.float())
    gated_ref = attn_out_ref_3d.float() * gate_ref_sig
    print(f"\n=== Step 8: gated = attn * sigmoid(gate) ===")
    print(f"  PCC(gated, ref) = {pcc(gated_host.float(), gated_ref.bfloat16().float()):.6f}")

    # WO (col 0 only)
    wo_col0 = wo[:, : n_q_pc * _HD].T.contiguous().bfloat16()  # [1536, 5120]
    wo_tt = ttnn.from_torch(
        wo_col0,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    dense_partial_tt = ttnn.linear(
        gated_tt, wo_tt, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute
    )
    gated_tt.deallocate(True)
    dense_partial_host = to_host(dense_partial_tt, mesh)

    wo_ref_f = wo[:, : n_q_pc * _HD].float()  # [5120, 1536]
    dense_partial_ref = gated_ref @ wo_ref_f.T  # [B, T, 5120]
    print(f"\n=== Step 9: WO dense_partial (col 0) ===")
    print(f"  PCC(dense_partial, ref) = {pcc(dense_partial_host.float(), dense_partial_ref.bfloat16().float()):.6f}")

    # Cleanup
    dense_partial_tt.deallocate(True)
    wo_tt.deallocate(True)
    q_normed_tt.deallocate(True)
    k_normed_tt.deallocate(True)
    q_norm_w.deallocate(True)
    k_norm_w.deallocate(True)
    x_tt.deallocate(True)
    wqkvg_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
