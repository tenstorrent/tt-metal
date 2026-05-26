"""Debug script v2: check if fill_cache is writing the right values.

The question is: does the TTNN KV cache have correct values after prefill?
We compare the TTNN cache content with the CPU reference.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_decode2.py
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
    return cos, sin


def rms_norm_cpu(x, w, eps=1e-6):
    orig_dtype = x.dtype
    x = x.float()
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * (1.0 + w.float())).to(orig_dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_partial_rope_cpu(x, cos, sin, rd=64):
    x_rot, x_pass = x[..., :rd], x[..., rd:]
    rotated = x_rot * cos + rotate_half(x_rot) * sin
    return torch.cat([rotated, x_pass], dim=-1)


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


def _run(mesh):
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = load_weights()
    args = TtQwen36ModelArgs(mesh)
    rope = Qwen36RopeSetup(mesh, args, batch_size=B, max_seq_len=512)
    attn = TtQwen36GatedAttention(mesh_device=mesh, args=args, state_dict=sd, layer_num=LAYER_IDX)

    torch.manual_seed(20)
    x_pre = torch.randn(B, T_PRE, H, dtype=torch.bfloat16)
    cos_pre_tt, sin_pre_tt = rope.get_cos_sin_for_prefill(seq_len=T_PRE)
    x_pre_tt = send(x_pre, mesh)
    out_pre_tt = attn.forward_prefill(x_pre_tt, rot_mats=(cos_pre_tt, sin_pre_tt), kv_cache=None, user_id=0)
    out_pre_tt.deallocate(True)
    x_pre_tt.deallocate(True)
    cos_pre_tt.deallocate(True)
    sin_pre_tt.deallocate(True)

    # -------------------------------------------------------------------------
    # Compare KV cache content: TTNN vs CPU reference
    # -------------------------------------------------------------------------
    # KV cache is sharded: [B=1, n_kv=4, max_seq=512, hd=256] sharded on dim=1 across 4 cols.
    # Gather from col 0 only. To get just col0, we use the first device in each row.
    # ConcatMeshToTensor with dim=1 on the full mesh would give [1, 32, 512, 256]
    # (8 rows * 4 cols = 32 shards, each with [1, 1, 512, 256]).
    #
    # Device ordering on 8x4 mesh: devices 0,1,2,3=row0col0,row0col1,row0col2,row0col3;
    # devices 4,5,6,7=row1col0..row1col3; etc.
    #
    # Using ConcatMeshToTensor(dim=1) → shape [1, 32, 512, 256]
    # Devices for col0: 0, 4, 8, 12, 16, 20, 24, 28 (8 row devices, col 0)
    # They all have the SAME content (replicated across rows), so we can take device 0's shard.
    #
    # With ConcatMeshToTensor(dim=1): device 0 is at index 0 of dim=1
    k_cache_all = ttnn.to_torch(attn.layer_past[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    v_cache_all = ttnn.to_torch(attn.layer_past[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    print(f"k_cache_all shape: {k_cache_all.shape}")  # [1, 32, 512, 256]

    # Shard 0 = first device (row0, col0) = KV head 0
    k_cache_head0_tt = k_cache_all[:, 0:1, :T_PRE, :]  # [1, 1, T_PRE, 256]
    v_cache_head0_tt = v_cache_all[:, 0:1, :T_PRE, :]  # [1, 1, T_PRE, 256]

    # CPU reference KV for head 0 (after prefill)
    cos_pre_ref, sin_pre_ref = build_rope(T_PRE)
    cos4d = cos_pre_ref.unsqueeze(1)  # [1, 1, T, 64]
    sin4d = sin_pre_ref.unsqueeze(1)

    x_f = x_pre.float()
    k_raw = x_f @ sd["k_proj.weight"].T
    k_heads = k_raw.view(B, T_PRE, N_KV, HEAD_DIM)
    k_normed = rms_norm_cpu(k_heads, sd["k_norm.weight"])
    k_t = k_normed.transpose(1, 2)  # [1, 4, T, 256]
    k_rot_ref = apply_partial_rope_cpu(k_t, cos4d, sin4d)

    v_raw = x_f @ sd["v_proj.weight"].T
    v_heads = v_raw.view(B, T_PRE, N_KV, HEAD_DIM)
    v_t = v_heads.transpose(1, 2)

    k_ref_head0 = k_rot_ref[:, 0:1, :, :]  # [1, 1, T_PRE, 256]
    v_ref_head0 = v_t[:, 0:1, :, :]  # [1, 1, T_PRE, 256]

    p_k = pcc(k_cache_head0_tt.float(), k_ref_head0.float())
    p_v = pcc(v_cache_head0_tt.float(), v_ref_head0.float())
    print(f"[KV cache compare] K head0 PCC={p_k:.6f}, V head0 PCC={p_v:.6f}")
    print(f"  TTNN k mean={k_cache_head0_tt.float().mean():.4f}, ref k mean={k_ref_head0.float().mean():.4f}")
    print(f"  TTNN v mean={v_cache_head0_tt.float().mean():.4f}, ref v mean={v_ref_head0.float().mean():.4f}")

    # Also check all 4 heads
    for h in range(4):
        k_h_tt = k_cache_all[:, h * 8 : (h * 8 + 1), :T_PRE, :]  # col h device 0
        v_h_tt = v_cache_all[:, h * 8 : (h * 8 + 1), :T_PRE, :]
        p_kh = pcc(k_h_tt.float(), k_rot_ref[:, h : h + 1, :, :].float())
        p_vh = pcc(v_h_tt.float(), v_t[:, h : h + 1, :, :].float())
        print(f"  Head {h}: K PCC={p_kh:.6f}, V PCC={p_vh:.6f}")

    # -------------------------------------------------------------------------
    # Test update_cache behavior
    # -------------------------------------------------------------------------
    print("\n--- Test update_cache ---")
    # Create a fresh KV cache (all zeros)
    test_cache = torch.zeros(B, 1, 512, HEAD_DIM)  # single KV head (col0)
    test_cache_tt = ttnn.from_torch(
        test_cache,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    # Token to write at position 32
    test_k = torch.randn(B, 1, 1, HEAD_DIM, dtype=torch.bfloat16)  # [1, 1, T=1, 256]
    test_k_tt = send(test_k, mesh)

    print(f"  Before update_cache: test_cache_tt shape={list(test_cache_tt.shape)}")
    print(f"  test_k_tt shape={list(test_k_tt.shape)}")

    ttnn.update_cache(test_cache_tt, test_k_tt, 32, batch_offset=0)

    # Read back
    after = ttnn.to_torch(test_cache_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    after = after[0:1]  # device 0
    # Check position 32
    written = after[0, 0, 32, :]
    expected = test_k[0, 0, 0, :]
    p_write = pcc(written.float(), expected.float())
    print(f"  After update_cache at pos 32: PCC={p_write:.6f}")
    print(f"  Written  mean={written.float().mean():.4f}")
    print(f"  Expected mean={expected.float().mean():.4f}")

    # Check positions 0..31 are still zeros
    zeros_region = after[0, 0, :32, :]
    max_in_zeros = zeros_region.abs().max().item()
    print(f"  Positions 0..31 max abs (should be 0): {max_in_zeros:.4f}")

    # -------------------------------------------------------------------------
    # Manual decode: skip update_cache and feed CPU-correct KV directly
    # -------------------------------------------------------------------------
    print("\n--- Manual decode with TTNN SDPA using CPU KV ---")
    cur_pos = T_PRE
    torch.manual_seed(21)
    x_dec = torch.randn(B, 1, H, dtype=torch.bfloat16)

    # Build decode step KV on CPU
    x_d = x_dec.float()
    cos_dec_ref_full, sin_dec_ref_full = build_rope(cur_pos + 1)
    cos_dec_ref = cos_dec_ref_full[:, cur_pos : cur_pos + 1, :]
    sin_dec_ref = sin_dec_ref_full[:, cur_pos : cur_pos + 1, :]
    cos4d_dec = cos_dec_ref.unsqueeze(1)  # [1,1,1,64]
    sin4d_dec = sin_dec_ref.unsqueeze(1)

    n_q_pc = 6
    n_kv_pc = 1
    q_dim_pc = n_q_pc * HEAD_DIM
    g_dim_pc = q_dim_pc
    k_dim_pc = HEAD_DIM
    v_dim_pc = HEAD_DIM
    n_q_total = 24

    q_2hd = sd["q_proj.weight"].reshape(n_q_total, 2, HEAD_DIM, H)
    wq_native = q_2hd[:, 0, :, :].reshape(n_q_total * HEAD_DIM, H)
    wgate_native = q_2hd[:, 1, :, :].reshape(n_q_total * HEAD_DIM, H)
    wk_native = sd["k_proj.weight"]
    wv_native = sd["v_proj.weight"]

    # Col0
    c = 0
    qs, qe = c * n_q_pc * HEAD_DIM, (c + 1) * n_q_pc * HEAD_DIM
    ksc, kec = c * n_kv_pc * HEAD_DIM, (c + 1) * n_kv_pc * HEAD_DIM
    wqkvg_col0 = torch.cat([wq_native[qs:qe], wgate_native[qs:qe], wk_native[ksc:kec], wv_native[ksc:kec]], dim=0)

    xqkvg_c0 = x_d @ wqkvg_col0.T
    q_f = xqkvg_c0[:, :, :q_dim_pc]
    gate_f = xqkvg_c0[:, :, q_dim_pc : q_dim_pc + g_dim_pc]
    k_f = xqkvg_c0[:, :, q_dim_pc + g_dim_pc : q_dim_pc + g_dim_pc + k_dim_pc]
    v_f = xqkvg_c0[:, :, q_dim_pc + g_dim_pc + k_dim_pc :]

    q_n = rms_norm_cpu(q_f.view(B, 1, n_q_pc, HEAD_DIM), sd["q_norm.weight"])
    k_n = rms_norm_cpu(k_f.view(B, 1, n_kv_pc, HEAD_DIM), sd["k_norm.weight"])

    q_t = q_n.transpose(1, 2)  # [1, 6, 1, 256]
    k_t = k_n.transpose(1, 2)
    v_t2 = v_f.view(B, 1, n_kv_pc, HEAD_DIM).transpose(1, 2)

    q_r = apply_partial_rope_cpu(q_t, cos4d_dec, sin4d_dec)
    k_r = apply_partial_rope_cpu(k_t, cos4d_dec, sin4d_dec)

    # Build full KV from CPU ref cache + this decode token
    # k_cache_ref_head0: [1, 1, T_PRE, 256]
    # k_full: [1, 1, T_PRE+1, 256]
    k_full = torch.cat([k_ref_head0, k_r], dim=2)  # [1, 1, 33, 256]
    v_full = torch.cat([v_ref_head0, v_t2], dim=2)

    # Send Q, K_full, V_full to TTNN and run SDPA directly
    gqa = n_q_pc // n_kv_pc  # 6
    k_exp = k_full.repeat_interleave(gqa, dim=1)  # [1, 6, 33, 256]
    v_exp = v_full.repeat_interleave(gqa, dim=1)

    # Pad KV to tile multiple
    T_kv = k_full.shape[2]  # 33
    T_kv_pad = ((T_kv + 31) // 32) * 32  # 64
    k_exp_padded = torch.zeros(B, n_q_pc, T_kv_pad, HEAD_DIM, dtype=torch.bfloat16)
    v_exp_padded = torch.zeros(B, n_q_pc, T_kv_pad, HEAD_DIM, dtype=torch.bfloat16)
    k_exp_padded[:, :, :T_kv, :] = k_exp.bfloat16()
    v_exp_padded[:, :, :T_kv, :] = v_exp.bfloat16()

    q_tt2 = send(q_r.bfloat16(), mesh)
    k_tt2 = send(k_exp_padded, mesh)
    v_tt2 = send(v_exp_padded, mesh)

    # SDPA with -inf mask for padded positions
    mask_cpu = torch.zeros(B, 1, 1, T_kv_pad, dtype=torch.bfloat16)
    mask_cpu[:, :, :, T_kv:] = float("-inf")
    mask_tt2 = send(mask_cpu, mesh)

    compute_kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    sdpa_out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_tt2,
        k_tt2,
        v_tt2,
        is_causal=False,
        attn_mask=mask_tt2,
        scale=HEAD_DIM**-0.5,
        compute_kernel_config=compute_kernel,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    sdpa_all = ttnn.to_torch(sdpa_out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    sdpa_col0 = sdpa_all[0:1]  # [1, 6, 1, 256] (first device, col0)

    # CPU SDPA reference
    scale = HEAD_DIM**-0.5
    scores = torch.matmul(q_r, k_exp.transpose(-2, -1)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(q_r.dtype)
    attn_ref = torch.matmul(probs, v_exp)  # [1, 6, 1, 256]

    p_sdpa_direct = pcc(sdpa_col0.float(), attn_ref.float())
    print(f"[Direct SDPA with CPU KV] PCC={p_sdpa_direct:.6f}")
    print(f"  TTNN SDPA mean={sdpa_col0.float().mean():.4f}, CPU ref mean={attn_ref.float().mean():.4f}")

    sdpa_out_tt.deallocate(True)
    q_tt2.deallocate(True)
    k_tt2.deallocate(True)
    v_tt2.deallocate(True)
    mask_tt2.deallocate(True)

    # -------------------------------------------------------------------------
    # Full decode using forward_decode, check intermediate Q/K values
    # -------------------------------------------------------------------------
    print("\n--- Full forward_decode with intermediate checks ---")
    x_dec_tt = send(x_dec, mesh)
    cos_dec_tt, sin_dec_tt = rope.get_cos_sin_for_decode(cur_pos)

    cur_pos_tensor = ttnn.from_torch(
        torch.tensor([cur_pos] * args.max_batch_size, dtype=torch.int32),
        device=mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    out_tt = attn.forward_decode(x_dec_tt, current_pos=cur_pos_tensor, rot_mats=(cos_dec_tt, sin_dec_tt))
    out_cpu = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]

    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config

    with open(_SNAPSHOT_DIR / "config.json") as f:
        cfg = Qwen36Config(json.load(f))
    ref_attn = GatedAttention(cfg)
    ref_attn.eval()
    for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        getattr(ref_attn, key).weight.data.copy_(sd[f"{key}.weight"])
    ref_attn.q_norm.weight.data.copy_(sd["q_norm.weight"])
    ref_attn.k_norm.weight.data.copy_(sd["k_norm.weight"])

    with torch.no_grad():
        _, (k_cache_ref2, v_cache_ref2) = ref_attn(
            x_pre.float(),
            cos_dec_ref_full[:, :T_PRE, :],
            sin_dec_ref_full[:, :T_PRE, :],
            kv_cache=None,
            attention_mask=torch.zeros(1, 1, T_PRE, T_PRE).masked_fill(
                torch.triu(torch.ones(T_PRE, T_PRE), diagonal=1).bool(), float("-inf")
            ),
        )
        ref_dec_out2, _ = ref_attn(
            x_dec.float(), cos_dec_ref, sin_dec_ref, kv_cache=(k_cache_ref2, v_cache_ref2), attention_mask=None
        )

    p_final = pcc(out_cpu.float(), ref_dec_out2.float())
    print(f"[forward_decode] PCC={p_final:.6f}")

    out_tt.deallocate(True)
    x_dec_tt.deallocate(True)
    cos_dec_tt.deallocate(True)
    sin_dec_tt.deallocate(True)


if __name__ == "__main__":
    main()
