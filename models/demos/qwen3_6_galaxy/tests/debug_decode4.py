"""Debug script v4: step through forward_decode manually to find the divergence.

Since even CPU-injected KV gives PCC=0.775, the bug is in the decode ops themselves.
We reproduce each step of forward_decode manually and check PCCs.

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_decode4.py
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
B, T_PRE, H = 1, 32, 5120
N_KV, HEAD_DIM, ROPE_DIM = 4, 256, 64
N_Q_PC, N_KV_PC = 6, 1  # per col
Q_DIM_PC = N_Q_PC * HEAD_DIM  # 1536
G_DIM_PC = Q_DIM_PC  # 1536
K_DIM_PC = N_KV_PC * HEAD_DIM  # 256
V_DIM_PC = K_DIM_PC  # 256
TOTAL_PC = Q_DIM_PC + G_DIM_PC + K_DIM_PC + V_DIM_PC  # 3584
N_Q_TOTAL = 24


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def send(t, mesh, dtype=None, layout=None):
    return ttnn.from_torch(
        t,
        device=mesh,
        dtype=dtype or ttnn.bfloat16,
        layout=layout or ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_host(tt_t, mesh):
    return ttnn.to_torch(tt_t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]


def build_rope(T):
    from models.demos.qwen3_6_galaxy.reference.qwen36 import build_mrope_cos_sin

    p = torch.arange(T, dtype=torch.long)
    p3 = torch.stack([p, p, p], dim=0)
    cos, sin = build_mrope_cos_sin(
        positions_3d=p3, head_dim=256, partial_rotary_factor=0.25, mrope_section=[11, 11, 10], theta=10_000_000.0
    )
    return cos, sin


def rms_norm_cpu(x, w, eps=1e-6):
    xf = x.float()
    norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * (1.0 + w.float())).to(x.dtype)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin, rd=64):
    xr, xp = x[..., :rd], x[..., rd:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)


def load_weights():
    from safetensors.torch import load_file as load_st

    with open(_SNAPSHOT_DIR / "model.safetensors.index.json") as f:
        idx = json.load(f)
    wm = idx["weight_map"]
    pfx = f"model.language_model.layers.{LAYER_IDX}.self_attn"
    ks = [
        f"{pfx}.{k}"
        for k in ["q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight"]
    ]
    files = sorted({wm[k] for k in ks})
    raw = {}
    for fn in files:
        s = load_st(str(_SNAPSHOT_DIR / fn))
        for k in ks:
            if k in s:
                raw[k] = s[k].float()
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
    from models.demos.qwen3_6_galaxy.reference.qwen36 import GatedAttention, Qwen36Config
    from models.demos.qwen3_6_galaxy.tt.llama_attention import TtQwen36GatedAttention
    from models.demos.qwen3_6_galaxy.tt.llama_rope import Qwen36RopeSetup
    from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

    sd = load_weights()
    args = TtQwen36ModelArgs(mesh)
    rope = Qwen36RopeSetup(mesh, args, batch_size=B, max_seq_len=512)
    attn = TtQwen36GatedAttention(mesh_device=mesh, args=args, state_dict=sd, layer_num=LAYER_IDX)

    # Prefill
    torch.manual_seed(20)
    x_pre = torch.randn(B, T_PRE, H, dtype=torch.bfloat16)
    cos_pre_tt, sin_pre_tt = rope.get_cos_sin_for_prefill(seq_len=T_PRE)
    x_pre_tt = send(x_pre, mesh)
    out = attn.forward_prefill(x_pre_tt, rot_mats=(cos_pre_tt, sin_pre_tt), kv_cache=None, user_id=0)
    out.deallocate(True)
    x_pre_tt.deallocate(True)
    cos_pre_tt.deallocate(True)
    sin_pre_tt.deallocate(True)
    print("[prefill] done")

    # Setup
    cur_pos = T_PRE
    torch.manual_seed(21)
    x_dec = torch.randn(B, 1, H, dtype=torch.bfloat16)

    # CPU reference
    cos_full, sin_full = build_rope(cur_pos + 1)
    cos_d = cos_full[:, cur_pos : cur_pos + 1, :]  # [1, 1, 64]
    sin_d = sin_full[:, cur_pos : cur_pos + 1, :]
    cos4d_d = cos_d.unsqueeze(1)
    sin4d_d = sin_d.unsqueeze(1)

    with open(_SNAPSHOT_DIR / "config.json") as f:
        cfg = Qwen36Config(json.load(f))
    ref_attn = GatedAttention(cfg)
    ref_attn.eval()
    for key in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        getattr(ref_attn, key).weight.data.copy_(sd[f"{key}.weight"])
    ref_attn.q_norm.weight.data.copy_(sd["q_norm.weight"])
    ref_attn.k_norm.weight.data.copy_(sd["k_norm.weight"])

    cos_pre_ref, sin_pre_ref = build_rope(T_PRE)
    mask_pre = torch.zeros(1, 1, T_PRE, T_PRE).masked_fill(
        torch.triu(torch.ones(T_PRE, T_PRE), diagonal=1).bool(), float("-inf")
    )
    with torch.no_grad():
        _, (k_cache_ref, v_cache_ref) = ref_attn(
            x_pre.float(), cos_pre_ref, sin_pre_ref, kv_cache=None, attention_mask=mask_pre
        )
        ref_dec_out, _ = ref_attn(x_dec.float(), cos_d, sin_d, kv_cache=(k_cache_ref, v_cache_ref), attention_mask=None)
    print(f"CPU ref decode: mean={ref_dec_out.mean():.4f}")

    # ---------------------------------------------------------------
    # Manual TTNN decode step-by-step using col-sharded weights
    # ---------------------------------------------------------------
    compute_kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )

    # CPU per-col reference
    q_2hd = sd["q_proj.weight"].reshape(N_Q_TOTAL, 2, HEAD_DIM, H)
    wq = q_2hd[:, 0, :, :].reshape(N_Q_TOTAL * HEAD_DIM, H)
    wg = q_2hd[:, 1, :, :].reshape(N_Q_TOTAL * HEAD_DIM, H)
    wk_nat = sd["k_proj.weight"]
    wv_nat = sd["v_proj.weight"]
    wo_T = sd["o_proj.weight"].T.contiguous()  # [6144, 5120]

    x_d = x_dec.float()

    # STEP 1: xqkvg for col0
    print("\n--- Step 1: xqkvg ---")
    c = 0
    qs, qe = c * N_Q_PC * HEAD_DIM, (c + 1) * N_Q_PC * HEAD_DIM
    ksc, kec = c * N_KV_PC * HEAD_DIM, (c + 1) * N_KV_PC * HEAD_DIM
    wqkvg_c = torch.cat([wq[qs:qe], wg[qs:qe], wk_nat[ksc:kec], wv_nat[ksc:kec]], dim=0)  # [3584, 5120]
    xqkvg_cpu = x_d @ wqkvg_c.T  # [1, 1, 3584]

    # TTNN xqkvg for col0 (the wqkvg weight is sharded across cols)
    x_dec_tt = send(x_dec, mesh)
    # T_logical=1 after send; shape=[1,1,5120] TILE_LAYOUT → actual T=32 tile-padded
    x_shape = list(x_dec_tt.shape)
    print(f"  x_dec_tt.shape={x_shape}")

    xqkvg_tt = ttnn.linear(
        x_dec_tt,
        attn.wqkvg,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )
    xqkvg_host = to_host(xqkvg_tt, mesh)  # [1, 32, 3584] (T=32 due to tile-pad)
    print(f"  xqkvg_tt.shape={list(xqkvg_tt.shape)}, host.shape={xqkvg_host.shape}")
    # Compare T=0 token (the first row)
    p_xqkvg = pcc(xqkvg_host[:, 0:1, :].float(), xqkvg_cpu.float())
    print(f"  TTNN xqkvg row0 vs CPU: PCC={p_xqkvg:.6f}")

    # STEP 2: split and check
    print("\n--- Step 2: split q/gate/k/v ---")
    T_tt = list(xqkvg_tt.shape)[1]  # 32 (tile-padded)
    q_flat_tt = ttnn.slice(xqkvg_tt, [0, 0, 0], [B, T_tt, Q_DIM_PC], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_flat_tt = ttnn.slice(
        xqkvg_tt, [0, 0, Q_DIM_PC], [B, T_tt, Q_DIM_PC + G_DIM_PC], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_flat_tt = ttnn.slice(
        xqkvg_tt,
        [0, 0, Q_DIM_PC + G_DIM_PC],
        [B, T_tt, Q_DIM_PC + G_DIM_PC + K_DIM_PC],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    v_flat_tt = ttnn.slice(
        xqkvg_tt, [0, 0, Q_DIM_PC + G_DIM_PC + K_DIM_PC], [B, T_tt, TOTAL_PC], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    xqkvg_tt.deallocate(True)

    q_flat_host = to_host(q_flat_tt, mesh)[:, 0:1, :]
    k_flat_host = to_host(k_flat_tt, mesh)[:, 0:1, :]
    v_flat_host = to_host(v_flat_tt, mesh)[:, 0:1, :]
    gate_flat_host = to_host(gate_flat_tt, mesh)[:, 0:1, :]

    q_flat_cpu = xqkvg_cpu[:, :, :Q_DIM_PC]
    k_flat_cpu = xqkvg_cpu[:, :, Q_DIM_PC + G_DIM_PC : Q_DIM_PC + G_DIM_PC + K_DIM_PC]
    v_flat_cpu = xqkvg_cpu[:, :, Q_DIM_PC + G_DIM_PC + K_DIM_PC :]
    gate_flat_cpu = xqkvg_cpu[:, :, Q_DIM_PC : Q_DIM_PC + G_DIM_PC]

    print(f"  q_flat PCC={pcc(q_flat_host.float(), q_flat_cpu.float()):.6f}")
    print(f"  k_flat PCC={pcc(k_flat_host.float(), k_flat_cpu.float()):.6f}")
    print(f"  v_flat PCC={pcc(v_flat_host.float(), v_flat_cpu.float()):.6f}")
    print(f"  gate PCC={pcc(gate_flat_host.float(), gate_flat_cpu.float()):.6f}")

    # STEP 3: reshape and QK-norm
    print("\n--- Step 3: reshape + QK-norm ---")
    q_h_tt = ttnn.reshape(q_flat_tt, [B, T_tt, N_Q_PC, HEAD_DIM])
    k_h_tt = ttnn.reshape(k_flat_tt, [B, T_tt, N_KV_PC, HEAD_DIM])
    v_h_tt = ttnn.reshape(v_flat_tt, [B, T_tt, N_KV_PC, HEAD_DIM])
    q_flat_tt.deallocate(True)
    k_flat_tt.deallocate(True)
    v_flat_tt.deallocate(True)

    q_normed_tt = ttnn.rms_norm(
        q_h_tt,
        weight=attn.q_norm_w,
        epsilon=attn.eps,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )
    k_normed_tt = ttnn.rms_norm(
        k_h_tt,
        weight=attn.k_norm_w,
        epsilon=attn.eps,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )
    q_h_tt.deallocate(True)
    k_h_tt.deallocate(True)

    q_normed_host = to_host(q_normed_tt, mesh)[:, 0:1, :, :]  # [1, 1, 6, 256]
    k_normed_host = to_host(k_normed_tt, mesh)[:, 0:1, :, :]  # [1, 1, 1, 256]

    q_n_cpu = rms_norm_cpu(q_flat_cpu.view(B, 1, N_Q_PC, HEAD_DIM), sd["q_norm.weight"])
    k_n_cpu = rms_norm_cpu(k_flat_cpu.view(B, 1, N_KV_PC, HEAD_DIM), sd["k_norm.weight"])
    print(f"  q_normed PCC={pcc(q_normed_host.float(), q_n_cpu.float()):.6f}")
    print(f"  k_normed PCC={pcc(k_normed_host.float(), k_n_cpu.float()):.6f}")

    # STEP 4: transpose and RoPE
    print("\n--- Step 4: transpose + RoPE ---")
    q_t_tt = ttnn.transpose(q_normed_tt, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_t_tt = ttnn.transpose(k_normed_tt, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_t_tt = ttnn.transpose(v_h_tt, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q_normed_tt.deallocate(True)
    k_normed_tt.deallocate(True)

    cos_dec_tt, sin_dec_tt = rope.get_cos_sin_for_decode(cur_pos)
    q_rot_tt = attn._apply_partial_rope(q_t_tt, cos_dec_tt, sin_dec_tt)
    k_rot_tt = attn._apply_partial_rope(k_t_tt, cos_dec_tt, sin_dec_tt)
    q_t_tt.deallocate(True)
    k_t_tt.deallocate(True)

    q_rot_host = to_host(q_rot_tt, mesh)[:, :, 0:1, :]  # [1, 6, 1, 256]
    k_rot_host = to_host(k_rot_tt, mesh)[:, :, 0:1, :]  # [1, 1, 1, 256]

    q_t_cpu = q_n_cpu.transpose(1, 2)  # [1, 6, 1, 256]
    k_t_cpu = k_n_cpu.transpose(1, 2)  # [1, 1, 1, 256]
    q_rot_cpu = apply_rope(q_t_cpu, cos4d_d, sin4d_d)
    k_rot_cpu = apply_rope(k_t_cpu, cos4d_d, sin4d_d)

    print(f"  q_rot PCC={pcc(q_rot_host.float(), q_rot_cpu.float()):.6f}")
    print(f"  k_rot PCC={pcc(k_rot_host.float(), k_rot_cpu.float()):.6f}")

    # STEP 5: update_cache + slice
    print("\n--- Step 5: update_cache + KV slice ---")
    ttnn.update_cache(attn.layer_past[0], k_rot_tt, cur_pos, batch_offset=0)
    ttnn.update_cache(attn.layer_past[1], v_t_tt, cur_pos, batch_offset=0)
    k_rot_tt.deallocate(True)
    v_t_tt.deallocate(True)

    T_kv = cur_pos + 1  # 33
    T_kv_pad = ((T_kv + 31) // 32) * 32  # 64
    k_cached_tt = ttnn.slice(
        attn.layer_past[0], [0, 0, 0, 0], [B, N_KV_PC, T_kv_pad, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    v_cached_tt = ttnn.slice(
        attn.layer_past[1], [0, 0, 0, 0], [B, N_KV_PC, T_kv_pad, HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    k_cached_host = to_host(k_cached_tt, mesh)  # [1, 1, 64, 256]
    v_cached_host = to_host(v_cached_tt, mesh)

    # CPU reference: full KV at decode position
    k_full_ref = torch.cat([k_cache_ref[:, 0:1, :, :], k_rot_cpu], dim=2)  # [1, 1, 33, 256]
    v_full_ref = torch.cat(
        [v_cache_ref[:, 0:1, :, :], v_h_tt if False else v_flat_cpu.view(B, 1, N_KV_PC, HEAD_DIM).transpose(1, 2)],
        dim=2,
    )

    k_cached_host_valid = k_cached_host[:, :, :T_kv, :]
    p_k_cache = pcc(k_cached_host_valid.float(), k_full_ref.float())
    print(f"  k_cached valid PCC={p_k_cache:.6f}")
    print(f"  k_cached shape: {k_cached_host.shape}, valid T={T_kv}")

    # STEP 6: GQA expand
    gqa = N_Q_PC // N_KV_PC  # 6
    k_exp_tt = ttnn.repeat_interleave(k_cached_tt, gqa, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_exp_tt = ttnn.repeat_interleave(v_cached_tt, gqa, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k_cached_tt.deallocate(True)
    v_cached_tt.deallocate(True)

    # STEP 7: SDPA
    print("\n--- Step 7: SDPA ---")
    # T_q from q_rot: what does TTNN report?
    print(f"  q_rot_tt.shape={list(q_rot_tt.shape)}")
    print(f"  k_exp_tt.shape={list(k_exp_tt.shape)}")
    print(f"  v_exp_tt.shape={list(v_exp_tt.shape)}")

    attn_mask_tt = None
    if T_kv_pad > T_kv:
        mask_cpu = torch.zeros(B, 1, 1, T_kv_pad, dtype=torch.bfloat16)
        mask_cpu[:, :, :, T_kv:] = float("-inf")
        attn_mask_tt = send(mask_cpu, mesh)

    attn_out_tt = ttnn.transformer.scaled_dot_product_attention(
        q_rot_tt,
        k_exp_tt,
        v_exp_tt,
        is_causal=False,
        attn_mask=attn_mask_tt,
        scale=HEAD_DIM**-0.5,
        compute_kernel_config=compute_kernel,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if attn_mask_tt is not None:
        attn_mask_tt.deallocate(True)
    k_exp_tt.deallocate(True)
    v_exp_tt.deallocate(True)

    attn_out_host = to_host(attn_out_tt, mesh)  # [1, 6, T_q_padded, 256]
    print(f"  attn_out_tt.shape={list(attn_out_tt.shape)}, host.shape={attn_out_host.shape}")

    # CPU SDPA
    v_flat_cpu_reshaped = v_flat_cpu.view(B, 1, N_KV_PC, HEAD_DIM).transpose(1, 2)  # [1,1,1,256]
    k_full_ref2 = torch.cat([k_cache_ref[:, 0:1, :, :], k_rot_cpu], dim=2)
    v_full_ref2 = torch.cat([v_cache_ref[:, 0:1, :, :], v_flat_cpu_reshaped], dim=2)
    k_exp_ref = k_full_ref2.repeat_interleave(gqa, dim=1)  # [1, 6, 33, 256]
    v_exp_ref = v_full_ref2.repeat_interleave(gqa, dim=1)
    scale = HEAD_DIM**-0.5
    scores = torch.matmul(q_rot_cpu, k_exp_ref.transpose(-2, -1)) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(q_rot_cpu.dtype)
    attn_ref = torch.matmul(probs, v_exp_ref)  # [1, 6, 1, 256]

    # Compare: take T=0 from TTNN result (which might be padded)
    T_q_tt = attn_out_host.shape[2]
    print(f"  T_q_tt={T_q_tt} (should be 1 or 32 tile-padded)")
    attn_out_host_t0 = attn_out_host[:, :, 0:1, :]  # take first token
    p_sdpa = pcc(attn_out_host_t0.float(), attn_ref.float())
    print(f"  SDPA PCC (token 0 vs ref)={p_sdpa:.6f}")

    # STEP 8: output gate
    print("\n--- Step 8: output gate ---")
    attn_t_tt = ttnn.transpose(attn_out_tt, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_out_tt.deallocate(True)
    attn_flat_tt = ttnn.reshape(attn_t_tt, [B, T_tt, Q_DIM_PC])
    attn_t_tt.deallocate(True)

    attn_flat_host = to_host(attn_flat_tt, mesh)[:, 0:1, :]  # [1, 1, 1536]
    attn_ref_flat = attn_ref.transpose(1, 2).reshape(B, 1, Q_DIM_PC)
    print(f"  attn_flat (gate-ready) PCC={pcc(attn_flat_host.float(), attn_ref_flat.float()):.6f}")

    gate_sig_tt = ttnn.sigmoid(gate_flat_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gate_flat_tt.deallocate(True)
    gated_tt = ttnn.multiply(attn_flat_tt, gate_sig_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn_flat_tt.deallocate(True)
    gate_sig_tt.deallocate(True)

    gated_host = to_host(gated_tt, mesh)[:, 0:1, :]
    gate_sig_cpu = torch.sigmoid(gate_flat_cpu)
    gated_cpu = attn_ref_flat * gate_sig_cpu
    print(f"  gated PCC={pcc(gated_host.float(), gated_cpu.float()):.6f}")

    # STEP 9: WO + all_gather + fast_reduce_nc
    print("\n--- Step 9: WO + reduce ---")
    dense_partial_tt = ttnn.linear(
        gated_tt,
        attn.wo,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel,
    )
    gated_tt.deallocate(True)

    dense_partial_host = to_host(dense_partial_tt, mesh)[:, 0:1, :]  # col0's partial
    wo_col0 = wo_T[qs:qe]  # [1536, 5120]
    dense_partial_cpu = gated_cpu @ wo_col0
    print(f"  dense_partial (col0) PCC={pcc(dense_partial_host.float(), dense_partial_cpu.float()):.6f}")

    gathered_tt = ttnn.all_gather(
        dense_partial_tt,
        dim=0,
        num_links=1,
        cluster_axis=1,
        topology=ttnn.Topology.Linear,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    dense_partial_tt.deallocate(True)
    print(f"  gathered.shape={list(gathered_tt.shape)}")

    dense_out_full_tt = ttnn.experimental.fast_reduce_nc(
        gathered_tt, dims=[0], output=None, compute_kernel_config=None, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    gathered_tt.deallocate(True)
    print(f"  dense_out_full.shape={list(dense_out_full_tt.shape)}")

    dense_out_host = to_host(dense_out_full_tt, mesh)
    print(f"  dense_out host.shape={dense_out_host.shape}")
    dense_out_host_t0 = dense_out_host[:, 0:1, :]

    # CPU full sum
    total_cpu = torch.zeros(B, 1, H)
    for cc in range(4):
        qsc = cc * N_Q_PC * HEAD_DIM
        qec = (cc + 1) * N_Q_PC * HEAD_DIM
        ksc2 = cc * N_KV_PC * HEAD_DIM
        kec2 = (cc + 1) * N_KV_PC * HEAD_DIM
        wqkvg_cc = torch.cat([wq[qsc:qec], wg[qsc:qec], wk_nat[ksc2:kec2], wv_nat[ksc2:kec2]], dim=0)
        xqkvg_cc = x_d @ wqkvg_cc.T
        q_f = xqkvg_cc[:, :, :Q_DIM_PC]
        gf = xqkvg_cc[:, :, Q_DIM_PC : Q_DIM_PC + G_DIM_PC]
        kf = xqkvg_cc[:, :, Q_DIM_PC + G_DIM_PC : Q_DIM_PC + G_DIM_PC + K_DIM_PC]
        vf = xqkvg_cc[:, :, Q_DIM_PC + G_DIM_PC + K_DIM_PC :]
        q_nc = rms_norm_cpu(q_f.view(B, 1, N_Q_PC, HEAD_DIM), sd["q_norm.weight"])
        k_nc = rms_norm_cpu(kf.view(B, 1, N_KV_PC, HEAD_DIM), sd["k_norm.weight"])
        qt = q_nc.transpose(1, 2)
        kt = k_nc.transpose(1, 2)
        vt = vf.view(B, 1, N_KV_PC, HEAD_DIM).transpose(1, 2)
        qr = apply_rope(qt, cos4d_d, sin4d_d)
        kr = apply_rope(kt, cos4d_d, sin4d_d)
        k_full_c = torch.cat([k_cache_ref[:, cc : cc + 1, :, :], kr], dim=2)
        v_full_c = torch.cat([v_cache_ref[:, cc : cc + 1, :, :], vt], dim=2)
        ke = k_full_c.repeat_interleave(gqa, dim=1)
        ve = v_full_c.repeat_interleave(gqa, dim=1)
        sc = torch.matmul(qr, ke.transpose(-2, -1)) * scale
        pr = torch.softmax(sc.float(), dim=-1).to(qr.dtype)
        ao = torch.matmul(pr, ve)
        af = ao.transpose(1, 2).reshape(B, 1, Q_DIM_PC)
        gs = torch.sigmoid(gf)
        gd = af * gs
        wo_c = wo_T[qsc:qec]
        total_cpu += gd @ wo_c

    p_final = pcc(dense_out_host_t0.float(), total_cpu.float())
    print(f"\n  final output PCC vs CPU col-sum={p_final:.6f}")
    p_vs_ref = pcc(dense_out_host_t0.float(), ref_dec_out.float())
    print(f"  final output PCC vs CPU ref   ={p_vs_ref:.6f}")

    dense_out_full_tt.deallocate(True)
    x_dec_tt.deallocate(True)
    cos_dec_tt.deallocate(True)
    sin_dec_tt.deallocate(True)


if __name__ == "__main__":
    main()
