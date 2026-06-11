# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — HunyuanTtRMSNorm vs PyTorch reference
# Uses real layer-0 input_layernorm weight from HunyuanImage-3.0
#
# Run:
#   cd /home/iguser/Christy/tt-metal
#   source python_env/bin/activate
#   python3 models/experimental/hunyuan_image_3_0/tests/pcc/test_rms_norm.py

import sys, json
import torch
from safetensors import safe_open

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"

for p in [ROOT, HUNYUAN]:
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.tt.attention.rms_norm import HunyuanTtRMSNorm


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


# ---------------------------------------------------------------------------
# Load config + weight
with open(f"{WEIGHTS}/config.json") as f:
    cfg = json.load(f)

SHARD = f"{WEIGHTS}/model-0001-of-0032.safetensors"
with safe_open(SHARD, framework="pt") as f:
    w = f.get_tensor("model.layers.0.input_layernorm.weight")  # [4096] bf16

H = cfg["hidden_size"]  # 4096
eps = cfg.get("rms_norm_eps", 1e-5)

state_dict = {"model.layers.0.input_layernorm.weight": w}

# ---------------------------------------------------------------------------
print("Opening device …")
device = ttnn.open_device(device_id=0)
# ttnn.enable_program_cache not available in this build

results = []

torch.manual_seed(0)

# ── Test 1: bf16 input  [1, 256, 4096] ─────────────────────────────────────
print("\n" + "=" * 60)
print("RMSNorm — [1, 256, 4096] bfloat16")
print("=" * 60)
x = torch.randn(1, 256, H, dtype=torch.bfloat16)

ref_norm = HunyuanRMSNorm(H, eps=eps)
ref_norm.weight = torch.nn.Parameter(w.clone())
ref_norm.eval()
with torch.no_grad():
    pt_out = ref_norm(x)

tt_norm = HunyuanTtRMSNorm(device, H, state_dict, "model.layers.0.input_layernorm.weight", eps=eps)
x_tt = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out_tt = tt_norm(x_tt)
tt_out = ttnn.to_torch(out_tt)
x_tt.deallocate(True)
out_tt.deallocate(True)

results.append(check("bf16 in  → output matches ref", pt_out, tt_out))

# ── Test 2: longer sequence  [1, 512, 4096] ────────────────────────────────
print("\n" + "=" * 60)
print("RMSNorm — [1, 512, 4096] bfloat16")
print("=" * 60)
x2 = torch.randn(1, 512, H, dtype=torch.bfloat16)
with torch.no_grad():
    pt_out2 = ref_norm(x2)

x_tt2 = ttnn.from_torch(
    x2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out_tt2 = tt_norm(x_tt2)
tt_out2 = ttnn.to_torch(out_tt2)
x_tt2.deallocate(True)
out_tt2.deallocate(True)

results.append(check("bf16 in  seq=512 → output matches ref", pt_out2, tt_out2))

# ── Test 3: batch=2  [2, 256, 4096] ────────────────────────────────────────
print("\n" + "=" * 60)
print("RMSNorm — [2, 256, 4096] bfloat16")
print("=" * 60)
x3 = torch.randn(2, 256, H, dtype=torch.bfloat16)
with torch.no_grad():
    pt_out3 = ref_norm(x3)

x_tt3 = ttnn.from_torch(
    x3, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
out_tt3 = tt_norm(x_tt3)
tt_out3 = ttnn.to_torch(out_tt3)
x_tt3.deallocate(True)
out_tt3.deallocate(True)

results.append(check("bf16 in  batch=2 → output matches ref", pt_out3, tt_out3))

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
n = sum(results)
print(f"RMSNorm: {n}/{len(results)} PASSED")
print("ALL PASS ✓" if n == len(results) else "SOME FAILED ✗")
print("=" * 60)

ttnn.close_device(device)
sys.exit(0 if n == len(results) else 1)
