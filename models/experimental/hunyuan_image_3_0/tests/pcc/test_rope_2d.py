# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — HunyuanTtRoPE2D vs PyTorch reference
# No model weights needed; tests cos/sin build + on-device apply.
#
# Run:
#   cd /home/iguser/Christy/tt-metal
#   source python_env/bin/activate
#   python3 models/experimental/hunyuan_image_3_0/tests/pcc/test_rope_2d.py

import sys, json
import torch

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"

for p in [ROOT, HUNYUAN]:
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import (
    build_batch_2d_rope,
    apply_rotary_pos_emb,
)
from models.experimental.hunyuan_image_3_0.tt.attention.rope_2d import HunyuanTtRoPE2D


# ---------------------------------------------------------------------------
def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a -= a.mean()
    b -= b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def check(name, ref, tt_out, pcc_thr=0.9999, atol=0.05):
    p = pcc(ref, tt_out)
    d = (ref.float() - tt_out.float()).abs().max().item()
    ok = (p >= pcc_thr) and (d <= atol)
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    print(f"         PCC={p:.8f}  max_abs_diff={d:.6f}  (pcc>={pcc_thr}, atol<={atol})")
    return ok


def rope_run(tt_rope, q_pt, k_pt, cos_pt, sin_pt, image_infos, device):
    """Run ref and TTNN RoPE on the same Q/K and return both outputs."""
    # ref
    q_ref, k_ref = apply_rotary_pos_emb(q_pt, k_pt, cos_pt, sin_pt)

    # TTNN: prepare cos/sin on device then apply
    S = q_pt.shape[2]
    cos_tt, sin_tt = tt_rope.prepare_cos_sin(S, image_infos=image_infos)

    q_tt = ttnn.from_torch(
        q_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_tt = ttnn.from_torch(
        k_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    q_rot_tt, k_rot_tt = tt_rope.forward(q_tt, k_tt, cos_tt, sin_tt)

    q_out = ttnn.to_torch(q_rot_tt)
    k_out = ttnn.to_torch(k_rot_tt)

    q_tt.deallocate(True)
    k_tt.deallocate(True)
    q_rot_tt.deallocate(True)
    k_rot_tt.deallocate(True)
    cos_tt.deallocate(True)
    sin_tt.deallocate(True)

    return q_ref, k_ref, q_out, k_out


# ---------------------------------------------------------------------------
with open(f"{WEIGHTS}/config.json") as f:
    cfg = json.load(f)

B = 1
S = 256
n_h = cfg["num_attention_heads"]  # 32
n_kv = cfg["num_key_value_heads"]  # 8
d_h = cfg["attention_head_dim"]  # 128

print("Opening device …")
device = ttnn.open_device(device_id=0)
# ttnn.enable_program_cache not available in this build
tt_rope = HunyuanTtRoPE2D(device=device, head_dim=d_h)

results = []
torch.manual_seed(7)
q_pt = torch.randn(B, n_h, S, d_h, dtype=torch.bfloat16)
k_pt = torch.randn(B, n_kv, S, d_h, dtype=torch.bfloat16)

# ── Test 1: text-only (no image region) ─────────────────────────────────────
print("\n" + "=" * 60)
print("RoPE 2D — text-only  Q[1,32,256,128]  K[1,8,256,128]")
print("=" * 60)
cos_t, sin_t = build_batch_2d_rope(S, d_h, image_infos=None)
q_ref, k_ref, q_tt, k_tt = rope_run(tt_rope, q_pt, k_pt, cos_t, sin_t, None, device)
results.append(check("Q text-only", q_ref, q_tt))
results.append(check("K text-only", k_ref, k_tt))

# ── Test 2: 8×8 image patch starting at token 10 ───────────────────────────
print("\n" + "=" * 60)
print("RoPE 2D — 8×8 image patch (tokens 10..73)  S=256")
print("=" * 60)
ih, iw = 8, 8
img_infos = [[(slice(10, 10 + ih * iw), (ih, iw))]]
cos_i, sin_i = build_batch_2d_rope(S, d_h, image_infos=img_infos)
q_ref_i, k_ref_i, q_tt_i, k_tt_i = rope_run(tt_rope, q_pt, k_pt, cos_i, sin_i, img_infos, device)
results.append(check("Q with 8×8 image patch", q_ref_i, q_tt_i))
results.append(check("K with 8×8 image patch", k_ref_i, k_tt_i))

# ── Test 3: larger 12×12 image patch starting at token 20 ──────────────────
print("\n" + "=" * 60)
print("RoPE 2D — 12×12 image patch (tokens 20..163)  S=256")
print("=" * 60)
ih2, iw2 = 12, 12
img_infos2 = [[(slice(20, 20 + ih2 * iw2), (ih2, iw2))]]
cos_i2, sin_i2 = build_batch_2d_rope(S, d_h, image_infos=img_infos2)
q_ref_i2, k_ref_i2, q_tt_i2, k_tt_i2 = rope_run(tt_rope, q_pt, k_pt, cos_i2, sin_i2, img_infos2, device)
results.append(check("Q with 12×12 image patch", q_ref_i2, q_tt_i2))
results.append(check("K with 12×12 image patch", k_ref_i2, k_tt_i2))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n = sum(results)
print(f"RoPE 2D: {n}/{len(results)} PASSED")
print("ALL PASS ✓" if n == len(results) else "SOME FAILED ✗")
print("=" * 60)

ttnn.close_device(device)
sys.exit(0 if n == len(results) else 1)
