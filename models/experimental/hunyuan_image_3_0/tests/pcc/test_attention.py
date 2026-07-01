# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — HunyuanTtAttention vs PyTorch reference
# Uses real layer-0 attention weights from HunyuanImage-3.0
#
# Run:
#   cd /home/iguser/ign-tt/tt-metal
#   source python_env/bin/activate
#   python3 models/experimental/hunyuan_image_3_0/tests/pcc/test_attention.py

import sys, json
import torch
from safetensors import safe_open

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
WEIGHTS = "/home/iguser/ign-tt/base"

for p in [ROOT, HUNYUAN]:
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.attention.attention import (
    HunyuanImage3SDPAAttention as RefAttn,
    AttentionConfig,
    build_causal_mask,
)
from models.experimental.hunyuan_image_3_0.tt.attention.attention import HunyuanTtAttention


# ---------------------------------------------------------------------------
def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def check(name, ref, tt_out, pcc_thr=0.99, atol=1.0):
    p = pcc(ref, tt_out)
    d = (ref.float() - tt_out.float()).abs().max().item()
    ok = (p >= pcc_thr) and (d <= atol)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    print(f"         PCC={p:.8f}  max_abs_diff={d:.6f}  (pcc>={pcc_thr}, atol<={atol})")
    return ok


# ---------------------------------------------------------------------------
# Load config + weights
with open(f"{WEIGHTS}/config.json") as f:
    cfg = json.load(f)

SHARDS = [
    f"{WEIGHTS}/model-0001-of-0032.safetensors",
    f"{WEIGHTS}/model-0002-of-0032.safetensors",
]
WKEYS = [
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.query_layernorm.weight",
    "model.layers.0.self_attn.key_layernorm.weight",
]
state_dict = {}
for shard in SHARDS:
    with safe_open(shard, framework="pt") as f:
        for k in f.keys():
            if k in WKEYS:
                state_dict[k] = f.get_tensor(k)

B = 1
S = 256
H = cfg["hidden_size"]  # 4096
n_h = cfg["num_attention_heads"]  # 32
n_kv = cfg["num_key_value_heads"]  # 8
d_h = cfg["attention_head_dim"]  # 128
eps = cfg.get("rms_norm_eps", 1e-5)

# Build PyTorch ref model (loaded once, reused across tests)
ref_cfg = AttentionConfig(
    hidden_size=H,
    num_attention_heads=n_h,
    attention_head_dim=d_h,
    num_key_value_heads=n_kv,
    use_qk_norm=True,
    use_rotary_pos_emb=True,
    rms_norm_eps=eps,
)
ref_attn = RefAttn(ref_cfg, layer_idx=0).to(torch.bfloat16).eval()
ref_attn.qkv_proj.weight = torch.nn.Parameter(state_dict["model.layers.0.self_attn.qkv_proj.weight"].clone())
ref_attn.o_proj.weight = torch.nn.Parameter(state_dict["model.layers.0.self_attn.o_proj.weight"].clone())
ref_attn.query_layernorm.weight = torch.nn.Parameter(
    state_dict["model.layers.0.self_attn.query_layernorm.weight"].clone()
)
ref_attn.key_layernorm.weight = torch.nn.Parameter(state_dict["model.layers.0.self_attn.key_layernorm.weight"].clone())

pt_mask = build_causal_mask(S, dtype=torch.bfloat16)  # [1,1,S,S]

# ---------------------------------------------------------------------------
print("Opening device …")
device = ttnn.open_device(device_id=0)
# ttnn.enable_program_cache not available in this build

tt_attn = HunyuanTtAttention(
    device=device,
    state_dict=state_dict,
    layer_num=0,
    hidden_size=H,
    num_heads=n_h,
    num_kv_heads=n_kv,
    head_dim=d_h,
    use_qk_norm=True,
    eps=eps,
)

results = []


# helper: run one forward through both ref and TTNN
def run(label, x_pt, image_infos, pcc_thr=0.99, atol=0.1):
    cos_pt, sin_pt = build_batch_2d_rope(S, d_h, image_infos=image_infos)

    with torch.no_grad():
        pt_out, _, _ = ref_attn(x_pt, attention_mask=pt_mask, custom_pos_emb=(cos_pt, sin_pt))

    cos_tt, sin_tt = tt_attn.rope.prepare_cos_sin(S, image_infos=image_infos)
    x_tt = ttnn.from_torch(
        x_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    # Pass attention_mask=None → is_causal=True (matches causal ref mask)
    out_tt = tt_attn.forward(x_tt, cos_tt, sin_tt, attention_mask=None)
    tt_out = ttnn.to_torch(out_tt)

    x_tt.deallocate(True)
    out_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)

    return check(label, pt_out, tt_out, pcc_thr=pcc_thr, atol=atol)


# ── Test 1: text-only, random input [1, 256, 4096] ─────────────────────────
print("\n" + "=" * 60)
print("Attention — text-only RoPE  [1, 256, 4096]")
print("=" * 60)
torch.manual_seed(3)
x1 = torch.randn(B, S, H, dtype=torch.bfloat16)
results.append(run("text-only  [1,256,4096]", x1, image_infos=None, pcc_thr=0.99, atol=1.0))

# ── Test 2: image-region RoPE (8×8 patch at token 10) ──────────────────────
print("\n" + "=" * 60)
print("Attention — 8×8 image patch RoPE  [1, 256, 4096]")
print("=" * 60)
img_infos = [[(slice(10, 74), (8, 8))]]
torch.manual_seed(3)  # same input, different RoPE
x2 = torch.randn(B, S, H, dtype=torch.bfloat16)
results.append(run("image-region RoPE  [1,256,4096]", x2, image_infos=img_infos, pcc_thr=0.99, atol=1.0))

# ── Test 3: different random seed ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Attention — text-only RoPE  [1, 256, 4096]  seed=99")
print("=" * 60)
torch.manual_seed(99)
x3 = torch.randn(B, S, H, dtype=torch.bfloat16)
results.append(run("text-only  [1,256,4096]  seed=99", x3, image_infos=None, pcc_thr=0.99, atol=1.0))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n = sum(results)
print(f"Attention: {n}/{len(results)} PASSED")
print("ALL PASS ✓" if n == len(results) else "SOME FAILED ✗")
print("=" * 60)

ttnn.close_device(device)
sys.exit(0 if n == len(results) else 1)
