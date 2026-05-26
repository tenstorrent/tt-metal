"""Debug script: isolate GatedAttention decode PCC failure.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_decode_pcc.py

This script runs prefill + decode on both CPU reference and TTNN, then
prints per-op PCCs to find where things diverge.
"""

import json
import pathlib
import sys

import torch

sys.path.insert(0, "/home/tt-admin/ssinghal/tt-metal")

import ttnn

_SNAPSHOT_DIR = pathlib.Path(
    "/home/tt-admin/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B"
    "/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
)

LAYER_IDX = 3
B = 1
T_PRE = 32
H = 5120
N_KV = 4
HEAD_DIM = 256
ROPE_DIM = 64


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def send(t, mesh, dtype=None):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype or ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def gather(tt_t, mesh, T=None):
    all_dev = ttnn.to_torch(tt_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    r = all_dev[0:1]
    if T is not None:
        r = r[:, :T, :]
    return r


def gather4d(tt_t, mesh):
    all_dev = ttnn.to_torch(tt_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    return all_dev[0:1]


def load_weights():
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    pfx = f"model.language_model.layers.{LAYER_IDX}.self_attn"
    keys = [
        f"{pfx}.{k}"
        for k in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight"]
    ]
    files = sorted({weight_map[k] for k in keys})
    raw = {}
    for fn in files:
        shard = load_st(str(_SNAPSHOT_DIR / fn))
        for k in keys:
            if k in shard:
                raw[k] = shard[k].float()
    return {k.split(".")[-2] + "." + k.split(".")[-1]: v for k, v in raw.items()}


def build_rope(T):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    positions = torch.arange(T, dtype=torch.long)
    positions_3d = torch.stack([positions, positions, positions], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=positions_3d,
        head_dim=256,
        partial_rotary_factor=0.25,
        mrope_section=[11, 11, 10],
        theta=10_000_000.0,
    )
    return cos, sin  # [1, T, 64]


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope_cpu(x, cos, sin, rd=64):
    # x: [B, n_heads, T, hd], cos/sin: [1,1,T, rd]
    x_rot = x[..., :rd]
    x_pass = x[..., rd:]
    rotated = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([rotated, x_pass], dim=-1)


def rms_norm_cpu(x, w, eps=1e-6):
    # zero-centered: weight = 1+w baked in
    # x: [..., d], w: [d]
    orig_dtype = x.dtype
    x = x.float()
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * (1.0 + w.float())).to(orig_dtype)


def main():
    print("Opening mesh...")
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(8, 4))

    try:
        _run(mesh)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _run(mesh):
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = load_weights()
    args = TtQwen36ModelArgs(mesh)
    rope = Qwen36RopeSetup(mesh, args, batch_size=B, max_seq_len=512)

    # -------------------------------------------------------------------------
    # TTNN module
    # -------------------------------------------------------------------------
    attn = TtQwen36GatedAttention(mesh_device=mesh, args=args, state_dict=sd, layer_num=LAYER_IDX)

    # -------------------------------------------------------------------------
    # Prefill
    # -------------------------------------------------------------------------
    torch.manual_seed(20)
    x_pre = torch.randn(B, T_PRE, H, dtype=torch.bfloat16)
    cos_pre, sin_pre = rope.get_cos_sin_for_prefill(seq_len=T_PRE)

    x_pre_tt = send(x_pre, mesh)
    out_pre_tt = attn.forward_prefill(x_pre_tt, rot_mats=(cos_pre, sin_pre), kv_cache=None, user_id=0)
    out_pre_tt.deallocate(True)
    x_pre_tt.deallocate(True)
    cos_pre.deallocate(True)
    sin_pre.deallocate(True)

    print("[prefill] done, KV cache seeded")

    # -------------------------------------------------------------------------
    # Inspect KV cache after prefill (per-col shard, col0 = KV head 0)
    # -------------------------------------------------------------------------
    # layer_past[0]: [B=1, n_kv=4, max_seq=512, hd=256] sharded by n_kv across 4 cols
    # Per-chip (col0): [B=1, n_kv_pc=1, max_seq=512, hd=256]
    keys_cache = attn.layer_past[0]
    vals_cache = attn.layer_past[1]

    # Read KV cache back for col0 only: gather 4 cols concatenated on dim=1 → [B, 4, 512, 256]
    kv_all_cols = ttnn.to_torch(keys_cache, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    # kv_all_cols: [32 rows * 1_batch, 4 cols * 1_kv_head, 512, 256]
    # With 8 rows and 4 cols sharded, the mesh_composer should give [8*1=8, 4*1=4, 512, 256]
    # We want row=0, col=0 → the first device (row0, col0) in row-major order = device 0
    # ConcatMeshToTensor(dim=1) concatenates chips along dim=1
    # For ShardTensor2dMesh with dims=(None, 1), the mesh_composer should be ConcatMeshToTensor on dim=1
    # But for a 8x4 mesh, devices are laid out as [row0col0, row0col1, row0col2, row0col3, row1col0, ...]
    # ConcatMeshToTensor(dim=1) with 32 devices would give shape [B*32, ...] if not handling 2D properly
    # Let's just inspect the raw tensor
    print(f"kv_all_cols shape: {kv_all_cols.shape}")

    # CPU reference: compute expected KV for first head (col0)
    # q_proj.weight: [12288, 5120] → each head uses 2*256=512 rows
    # We need k_proj.weight: [1024, 5120] → 4 heads × 256 = 1024 rows
    # k_norm.weight: [256]
    cos_pre_ref, sin_pre_ref = build_rope(T_PRE)
    cos4d_ref = cos_pre_ref.unsqueeze(1)  # [1, 1, T, 64]
    sin4d_ref = sin_pre_ref.unsqueeze(1)

    x_f = x_pre.float()
    k_raw = x_f @ sd["k_proj.weight"].float().T  # [1, T, 1024]
    k_heads = k_raw.view(B, T_PRE, N_KV, HEAD_DIM)  # [1, T, 4, 256]
    k_normed = rms_norm_cpu(k_heads, sd["k_norm.weight"])  # [1, T, 4, 256]
    k_t = k_normed.transpose(1, 2)  # [1, 4, T, 256]
    k_rot_ref = apply_partial_rope_cpu(k_t, cos4d_ref, sin4d_ref)  # [1, 4, T, 256]

    v_raw = x_f @ sd["v_proj.weight"].float().T  # [1, T, 1024]
    v_heads = v_raw.view(B, T_PRE, N_KV, HEAD_DIM)
    v_t = v_heads.transpose(1, 2)  # [1, 4, T, 256]

    print(f"k_rot_ref (head0): shape {k_rot_ref.shape}, head0 mean={k_rot_ref[0,0].mean():.4f}")
    print(f"v_t (head0): shape {v_t.shape}, head0 mean={v_t[0,0].mean():.4f}")

    # -------------------------------------------------------------------------
    # Decode step
    # -------------------------------------------------------------------------
    cur_pos = T_PRE  # position 32 (0-indexed)
    torch.manual_seed(21)
    x_dec = torch.randn(B, 1, H, dtype=torch.bfloat16)

    cos_dec_tt, sin_dec_tt = rope.get_cos_sin_for_decode(cur_pos)
    x_dec_tt = send(x_dec, mesh)

    cur_pos_tensor = ttnn.from_torch(
        torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    out_dec_tt = attn.forward_decode(x_dec_tt, current_pos=cur_pos_tensor, rot_mats=(cos_dec_tt, sin_dec_tt))
    out_host = gather(out_dec_tt, mesh, T=1)
    print(f"\n[TTNN decode] output: {out_host.shape}, mean={out_host.float().mean():.4f}")

    # -------------------------------------------------------------------------
    # CPU reference decode
    # -------------------------------------------------------------------------
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        cfg = Qwen36Config(json.load(f))
    ref_attn = GatedAttention(cfg)
    ref_attn.eval()
    for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        getattr(ref_attn, key).weight.data.copy_(sd[f"{key}.weight"])
    ref_attn.q_norm.weight.data.copy_(sd["q_norm.weight"])
    ref_attn.k_norm.weight.data.copy_(sd["k_norm.weight"])

    cos_pre_ref2, sin_pre_ref2 = build_rope(T_PRE)
    mask_pre = torch.zeros(1, 1, T_PRE, T_PRE)
    mask_pre = mask_pre.masked_fill(torch.triu(torch.ones(T_PRE, T_PRE), diagonal=1).bool(), float("-inf"))

    with torch.no_grad():
        _, (k_cache, v_cache) = ref_attn(
            x_pre.float(), cos_pre_ref2, sin_pre_ref2, kv_cache=None, attention_mask=mask_pre
        )
    print(f"\n[CPU] k_cache after prefill: {k_cache.shape}, mean={k_cache.mean():.4f}")
    print(f"[CPU] v_cache after prefill: {v_cache.shape}, mean={v_cache.mean():.4f}")

    cos_dec_ref_full, sin_dec_ref_full = build_rope(cur_pos + 1)
    cos_dec_ref = cos_dec_ref_full[:, cur_pos : cur_pos + 1, :]
    sin_dec_ref = sin_dec_ref_full[:, cur_pos : cur_pos + 1, :]

    with torch.no_grad():
        ref_dec_out, _ = ref_attn(
            x_dec.float(), cos_dec_ref, sin_dec_ref, kv_cache=(k_cache, v_cache), attention_mask=None
        )
    print(f"[CPU] ref decode output: {ref_dec_out.shape}, mean={ref_dec_out.mean():.4f}")

    p = pcc(out_host.float(), ref_dec_out.float())
    print(f"\n=== FINAL PCC: {p:.6f} ===")

    # -------------------------------------------------------------------------
    # Step-by-step decode comparison (manual TTNN decode)
    # -------------------------------------------------------------------------
    print("\n--- Manual decode step-by-step ---")
    hd = HEAD_DIM
    n_q_pc = 6  # n_q_per_col
    n_kv_pc = 1
    q_dim_pc = n_q_pc * hd  # 1536
    g_dim_pc = q_dim_pc  # 1536
    k_dim_pc = n_kv_pc * hd  # 256
    v_dim_pc = n_kv_pc * hd  # 256
    total_pc = q_dim_pc + g_dim_pc + k_dim_pc + v_dim_pc  # 3584

    # CPU decode step
    x_d = x_dec.float()
    # wqkvg for col0: first 3584 cols of wqkvg transpose
    # wqkvg was built as [14336, 5120] for all cols; col0 gets first 3584 rows
    # In TTNN, wqkvg_T is [H=5120, 14336], sharded on dim=1 across 4 cols
    # Col0 sees: x @ wqkvg_T[:, :3584] = x @ [5120, 3584]

    # Reconstruct wqkvg for col0 manually
    q_2hd = sd["q_proj.weight"].float().reshape(N_KV * 6, 2, hd, H)  # n_q=24, n_q_per_col=6
    # Actually n_q=24, but that's the full model. Let me redo this properly.
    n_q_total = 24
    q_2hd = sd["q_proj.weight"].float().reshape(n_q_total, 2, hd, H)
    wq_native = q_2hd[:, 0, :, :].reshape(n_q_total * hd, H)  # [6144, 5120]
    wgate_native = q_2hd[:, 1, :, :].reshape(n_q_total * hd, H)  # [6144, 5120]
    wk_native = sd["k_proj.weight"].float()  # [1024, 5120]
    wv_native = sd["v_proj.weight"].float()  # [1024, 5120]

    # Col0 wqkvg: [Q_h0..h5, gate_h0..h5, K_h0, V_h0] → [3584, 5120]
    c = 0
    qs, qe = c * n_q_pc * hd, (c + 1) * n_q_pc * hd
    ks_idx, ke_idx = c * n_kv_pc * hd, (c + 1) * n_kv_pc * hd
    wqkvg_col0 = torch.cat(
        [
            wq_native[qs:qe],
            wgate_native[qs:qe],
            wk_native[ks_idx:ke_idx],
            wv_native[ks_idx:ke_idx],
        ],
        dim=0,
    )  # [3584, 5120]

    xqkvg_col0 = x_d @ wqkvg_col0.T  # [1, 1, 3584]
    q_flat_cpu = xqkvg_col0[:, :, :q_dim_pc]  # [1, 1, 1536]
    gate_flat_cpu = xqkvg_col0[:, :, q_dim_pc : q_dim_pc + g_dim_pc]  # [1, 1, 1536]
    k_flat_cpu = xqkvg_col0[:, :, q_dim_pc + g_dim_pc : q_dim_pc + g_dim_pc + k_dim_pc]  # [1, 1, 256]
    v_flat_cpu = xqkvg_col0[:, :, q_dim_pc + g_dim_pc + k_dim_pc :]  # [1, 1, 256]

    q_h_cpu = q_flat_cpu.view(B, 1, n_q_pc, hd)  # [1, 1, 6, 256]
    k_h_cpu = k_flat_cpu.view(B, 1, n_kv_pc, hd)  # [1, 1, 1, 256]
    v_h_cpu = v_flat_cpu.view(B, 1, n_kv_pc, hd)  # [1, 1, 1, 256]

    # QK norm
    q_normed_cpu = rms_norm_cpu(q_h_cpu, sd["q_norm.weight"])
    k_normed_cpu = rms_norm_cpu(k_h_cpu, sd["k_norm.weight"])

    # Transpose to [B, heads, T, hd]
    q_t_cpu = q_normed_cpu.transpose(1, 2)  # [1, 6, 1, 256]
    k_t_cpu = k_normed_cpu.transpose(1, 2)  # [1, 1, 1, 256]
    v_t_cpu = v_h_cpu.transpose(1, 2)  # [1, 1, 1, 256]

    # Partial RoPE at position cur_pos=32
    cos4d_dec = cos_dec_ref.unsqueeze(1)  # [1,1,1,64]
    sin4d_dec = sin_dec_ref.unsqueeze(1)
    q_rot_cpu = apply_partial_rope_cpu(q_t_cpu, cos4d_dec, sin4d_dec)  # [1, 6, 1, 256]
    k_rot_dec_cpu = apply_partial_rope_cpu(k_t_cpu, cos4d_dec, sin4d_dec)  # [1, 1, 1, 256]

    # Expected KV at decode position: k_cache was stored as concat([k_prefill, k_dec], dim=2)
    # k_cache shape after prefill: [B, n_kv=4, T_PRE, hd] (FULL model KV cache in CPU reference)
    # For col0, we need head 0: k_cache[0, 0, :, :]
    # The reference KV cache is [B, 4, T_PRE, 256], so head 0 is k_cache[0, 0, :, :]
    k_cache_head0 = k_cache[0, 0:1]  # [1, T_PRE, 256]
    v_cache_head0 = v_cache[0, 0:1]  # [1, T_PRE, 256]

    # Expand to match TTNN col0 shape
    k_cache_col0 = k_cache_head0.unsqueeze(0)  # [1, 1, T_PRE, 256]
    v_cache_col0 = v_cache_head0.unsqueeze(0)  # [1, 1, T_PRE, 256]

    # Full KV for decode: concat([k_cache, k_dec], dim=2)
    k_full_col0 = torch.cat([k_cache_col0, k_rot_dec_cpu], dim=2)  # [1, 1, T_PRE+1, 256]
    v_full_col0 = torch.cat([v_cache_col0, v_t_cpu], dim=2)  # [1, 1, T_PRE+1, 256]

    # GQA expand
    gqa = n_q_pc // n_kv_pc  # 6
    k_exp_col0 = k_full_col0.repeat_interleave(gqa, dim=1)  # [1, 6, T_PRE+1, 256]
    v_exp_col0 = v_full_col0.repeat_interleave(gqa, dim=1)  # [1, 6, T_PRE+1, 256]

    # SDPA (no mask needed: decode q attends to all KV)
    scale = hd**-0.5
    attn_scores = torch.matmul(q_rot_cpu, k_exp_col0.transpose(-2, -1)) * scale
    attn_probs = torch.softmax(attn_scores.float(), dim=-1).to(q_rot_cpu.dtype)
    attn_out_col0 = torch.matmul(attn_probs, v_exp_col0)  # [1, 6, 1, 256]

    # Output gate
    attn_t_col0 = attn_out_col0.transpose(1, 2)  # [1, 1, 6, 256]
    attn_flat_col0 = attn_t_col0.reshape(B, 1, q_dim_pc)  # [1, 1, 1536]
    gate_sig_col0 = torch.sigmoid(gate_flat_cpu)
    gated_col0 = attn_flat_col0 * gate_sig_col0

    # WO (col0 portion): first 1536 rows of wo_T = first 1536 input dim
    wo_T = sd["o_proj.weight"].float().T  # [6144, 5120]
    wo_col0 = wo_T[qs:qe]  # [1536, 5120]
    dense_col0 = gated_col0 @ wo_col0  # [1, 1, 5120]

    print(f"[CPU col0] attn_out: {attn_out_col0.shape}, mean={attn_out_col0.mean():.4f}")
    print(f"[CPU col0] dense partial: {dense_col0.shape}, mean={dense_col0.mean():.4f}")

    # Sum all 4 cols
    dense_total_cpu = torch.zeros(B, 1, H)
    for c in range(4):
        qsc = c * n_q_pc * hd
        qec = (c + 1) * n_q_pc * hd
        ksc = c * n_kv_pc * hd
        kec = (c + 1) * n_kv_pc * hd

        wqkvg_colc = torch.cat(
            [
                wq_native[qsc:qec],
                wgate_native[qsc:qec],
                wk_native[ksc:kec],
                wv_native[ksc:kec],
            ],
            dim=0,
        )
        xqkvg_c = x_d @ wqkvg_colc.T
        q_f_c = xqkvg_c[:, :, :q_dim_pc]
        gate_f_c = xqkvg_c[:, :, q_dim_pc : q_dim_pc + g_dim_pc]
        k_f_c = xqkvg_c[:, :, q_dim_pc + g_dim_pc : q_dim_pc + g_dim_pc + k_dim_pc]
        v_f_c = xqkvg_c[:, :, q_dim_pc + g_dim_pc + k_dim_pc :]

        q_h_c = q_f_c.view(B, 1, n_q_pc, hd)
        k_h_c = k_f_c.view(B, 1, n_kv_pc, hd)
        v_h_c = v_f_c.view(B, 1, n_kv_pc, hd)
        q_n_c = rms_norm_cpu(q_h_c, sd["q_norm.weight"])
        k_n_c = rms_norm_cpu(k_h_c, sd["k_norm.weight"])

        q_tc = q_n_c.transpose(1, 2)  # [1, 6, 1, 256]
        k_tc = k_n_c.transpose(1, 2)
        v_tc = v_h_c.transpose(1, 2)

        q_rc = apply_partial_rope_cpu(q_tc, cos4d_dec, sin4d_dec)
        k_rc = apply_partial_rope_cpu(k_tc, cos4d_dec, sin4d_dec)

        # Use cpu ref k/v cache for this head
        k_cc = k_cache[0, c : c + 1].unsqueeze(0)  # [1, 1, T_PRE, 256]
        v_cc = v_cache[0, c : c + 1].unsqueeze(0)
        k_full_c = torch.cat([k_cc, k_rc], dim=2)
        v_full_c = torch.cat([v_cc, v_tc], dim=2)

        k_exp_c = k_full_c.repeat_interleave(gqa, dim=1)
        v_exp_c = v_full_c.repeat_interleave(gqa, dim=1)

        scores_c = torch.matmul(q_rc, k_exp_c.transpose(-2, -1)) * scale
        probs_c = torch.softmax(scores_c.float(), dim=-1).to(q_rc.dtype)
        attn_c = torch.matmul(probs_c, v_exp_c)

        attn_tc = attn_c.transpose(1, 2).reshape(B, 1, q_dim_pc)
        gate_sc = torch.sigmoid(gate_f_c)
        gated_c = attn_tc * gate_sc

        wo_c = wo_T[qsc:qec]
        dense_total_cpu += gated_c @ wo_c

    print(f"\n[CPU] total sum across 4 cols: {dense_total_cpu.shape}, mean={dense_total_cpu.mean():.4f}")
    print(f"[CPU ref GatedAttention output] mean={ref_dec_out.mean():.4f}")
    p_cpu_sum_vs_ref = pcc(dense_total_cpu.float(), ref_dec_out.float())
    print(f"[CPU col-sum vs ref] PCC = {p_cpu_sum_vs_ref:.6f}")

    print(f"\n[TTNN output] mean={out_host.float().mean():.4f}")
    p_ttnn_vs_col_sum = pcc(out_host.float(), dense_total_cpu.float())
    print(f"[TTNN vs CPU col-sum] PCC = {p_ttnn_vs_col_sum:.6f}")
    p_ttnn_vs_ref = pcc(out_host.float(), ref_dec_out.float())
    print(f"[TTNN vs CPU ref] PCC = {p_ttnn_vs_ref:.6f}")


if __name__ == "__main__":
    main()
