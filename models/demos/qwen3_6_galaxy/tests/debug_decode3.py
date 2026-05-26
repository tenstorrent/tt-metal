"""Debug script v3: correct KV cache inspection.

Check KV cache for all 4 heads using correct device indices.
Device ordering for 8x4 mesh (row-major):
  row0: dev0(col0), dev1(col1), dev2(col2), dev3(col3)
  row1: dev4(col0), dev5(col1), dev6(col2), dev7(col3)
  ...

ConcatMeshToTensor(dim=1) → shape [1, 32, 512, 256]
Head h is at index h (col h of row 0).

Run:
    source python_env/bin/activate && export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd)
    python3 models/demos/qwen3_6_galaxy/tests/debug_decode3.py
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

    # CPU reference KV
    cos_ref, sin_ref = build_rope(T_PRE)
    cos4d = cos_ref.unsqueeze(1)
    sin4d = sin_ref.unsqueeze(1)
    x_f = x_pre.float()
    k_raw = x_f @ sd["k_proj.weight"].T
    k_h = k_raw.view(B, T_PRE, N_KV, HEAD_DIM)
    k_n = rms_norm_cpu(k_h, sd["k_norm.weight"])
    k_t = k_n.transpose(1, 2)  # [1, 4, T_PRE, 256]
    k_rot = apply_rope(k_t, cos4d, sin4d)
    v_raw = x_f @ sd["v_proj.weight"].T
    v_h = v_raw.view(B, T_PRE, N_KV, HEAD_DIM)
    v_t = v_h.transpose(1, 2)  # [1, 4, T_PRE, 256]

    # TTNN KV cache: [1, 32, 512, 256] (all 32 devices, each holding 1 KV head shard)
    # Device h (for h in 0..3) = row0col{h} = KV head h
    k_all = ttnn.to_torch(attn.layer_past[0], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    v_all = ttnn.to_torch(attn.layer_past[1], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1))
    print(f"KV cache shape from device: {k_all.shape}")  # [1, 32, 512, 256]

    # CORRECT device index for KV head h: use device h (col h of row 0)
    # The KV cache has shape=[1, n_kv_pc=1, 512, 256] per chip.
    # ConcatMeshToTensor(dim=1) → all 32 chips' dim=1 concatenated → [1, 32, 512, 256]
    # Device 0=col0, device 1=col1, device 2=col2, device 3=col3 (row 0)
    print("\n=== KV cache PCC by head (correct indices) ===")
    for h in range(4):
        k_h_tt = k_all[:, h : h + 1, :T_PRE, :]  # device h, head
        v_h_tt = v_all[:, h : h + 1, :T_PRE, :]
        pk = pcc(k_h_tt.float(), k_rot[:, h : h + 1].float())
        pv = pcc(v_h_tt.float(), v_t[:, h : h + 1].float())
        print(f"  Head {h} (device {h}): K PCC={pk:.6f}, V PCC={pv:.6f}")

    # -------------------------------------------------------------------------
    # Now do the decode step manually but feeding TTNN the CORRECT CPU KV
    # to isolate whether the decode SDPA + gate + WO are correct.
    # -------------------------------------------------------------------------
    print("\n=== Decode step with CPU-injected KV cache ===")
    cur_pos = T_PRE
    torch.manual_seed(21)
    x_dec = torch.randn(B, 1, H, dtype=torch.bfloat16)

    cos_full, sin_full = build_rope(cur_pos + 1)
    cos_d_ref = cos_full[:, cur_pos : cur_pos + 1, :]  # [1, 1, 64]
    sin_d_ref = sin_full[:, cur_pos : cur_pos + 1, :]
    cos4d_d = cos_d_ref.unsqueeze(1)
    sin4d_d = sin_d_ref.unsqueeze(1)

    n_q_pc = 6
    n_kv_pc = 1
    q_dim = n_q_pc * HEAD_DIM
    g_dim = q_dim
    k_dim = n_kv_pc * HEAD_DIM
    v_dim = k_dim
    n_q_total = 24

    q_2hd = sd["q_proj.weight"].reshape(n_q_total, 2, HEAD_DIM, H)
    wq = q_2hd[:, 0, :, :].reshape(n_q_total * HEAD_DIM, H)
    wg = q_2hd[:, 1, :, :].reshape(n_q_total * HEAD_DIM, H)
    wk = sd["k_proj.weight"]
    wv = sd["v_proj.weight"]

    x_d = x_dec.float()

    # Compute all 4 col partials on CPU, sum, compare to full reference
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
    mask_pre = torch.zeros(1, 1, T_PRE, T_PRE).masked_fill(
        torch.triu(torch.ones(T_PRE, T_PRE), diagonal=1).bool(), float("-inf")
    )
    with torch.no_grad():
        _, (k_cache_ref, v_cache_ref) = ref_attn(
            x_pre.float(), cos_pre_ref2, sin_pre_ref2, kv_cache=None, attention_mask=mask_pre
        )
        ref_dec_out, _ = ref_attn(
            x_dec.float(), cos_d_ref, sin_d_ref, kv_cache=(k_cache_ref, v_cache_ref), attention_mask=None
        )
    print(f"CPU ref decode output mean: {ref_dec_out.mean():.4f}")

    # Now build a "fake" KV cache with correct values and run TTNN forward_decode
    # We REPLACE the TTNN KV cache with the CPU-computed values
    # Shape: [B=1, n_kv=4, max_seq=512, hd=256] sharded on dim=1 across 4 cols
    cpu_k_cache = torch.zeros(B, N_KV, 512, HEAD_DIM, dtype=torch.bfloat16)
    cpu_v_cache = torch.zeros(B, N_KV, 512, HEAD_DIM, dtype=torch.bfloat16)
    cpu_k_cache[:, :, :T_PRE, :] = k_cache_ref.bfloat16()
    cpu_v_cache[:, :, :T_PRE, :] = v_cache_ref.bfloat16()

    col_shard_kv = ttnn.ShardTensor2dMesh(mesh, dims=(None, 1), mesh_shape=[8, 4])
    attn.layer_past[0] = ttnn.from_torch(
        cpu_k_cache,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=col_shard_kv,
    )
    attn.layer_past[1] = ttnn.from_torch(
        cpu_v_cache,
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=col_shard_kv,
    )
    print("Replaced TTNN KV cache with CPU reference values.")

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

    out_injected_tt = attn.forward_decode(x_dec_tt, current_pos=cur_pos_tensor, rot_mats=(cos_dec_tt, sin_dec_tt))
    out_injected_cpu = ttnn.to_torch(out_injected_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[0:1]
    p_injected = pcc(out_injected_cpu.float(), ref_dec_out.float())
    print(f"[CPU-injected KV decode] PCC={p_injected:.6f}")
    print(f"  TTNN mean={out_injected_cpu.float().mean():.4f}, CPU ref mean={ref_dec_out.float().mean():.4f}")

    out_injected_tt.deallocate(True)
    x_dec_tt.deallocate(True)
    cos_dec_tt.deallocate(True)
    sin_dec_tt.deallocate(True)


if __name__ == "__main__":
    main()
