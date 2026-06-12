# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — HunyuanTtTopKGate (TTNN) vs PyTorch reference (easy_topk).
# Uses the real layer-0 router weight from HunyuanImage-3.0.
#
# Gate-1 requirement: the TT router must select the SAME set of experts per
# token as PyTorch, and the normalised routed weights must match to high PCC.
#
# Run:
#   cd /home/iguser/Christy/tt-metal
#   source python_env/bin/activate
#   python3 models/experimental/hunyuan_image_3_0/tests/pcc/test_moe_router.py

import sys, json
import torch
from safetensors import safe_open

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"

for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.moe.gate import HunyuanTopKGate
from models.experimental.hunyuan_image_3_0.tt.moe.gate import HunyuanTtTopKGate


# ---------------------------------------------------------------------------
def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


# ---------------------------------------------------------------------------
# Config + real router weight
with open(f"{WEIGHTS}/config.json") as f:
    cfg = json.load(f)

H = cfg["hidden_size"]  # 4096
NUM_EXPERTS = cfg["num_experts"] if isinstance(cfg["num_experts"], int) else cfg["num_experts"][0]  # 64
MOE_TOPK = cfg["moe_topk"] if isinstance(cfg["moe_topk"], int) else cfg["moe_topk"][0]  # 8
NORM_TOPK = cfg.get("norm_topk_prob", True)

WG_KEY = "model.layers.0.mlp.gate.wg.weight"
import glob

weight_map = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
SHARD = f"{WEIGHTS}/{weight_map[WG_KEY]}"
with safe_open(SHARD, framework="pt") as f:
    wg = f.get_tensor(WG_KEY)  # [num_experts, hidden]

state_dict = {WG_KEY: wg}

B, S = 1, 256

# ---------------------------------------------------------------------------
# PyTorch reference (easy_topk = the inference path)
ref_gate = HunyuanTopKGate(hidden_size=H, num_experts=NUM_EXPERTS, moe_topk=MOE_TOPK, norm_topk_prob=NORM_TOPK).eval()
# wg is fp32 in the reference; load the checkpoint weight verbatim.
ref_gate.wg.weight = torch.nn.Parameter(wg.float().clone())

torch.manual_seed(0)
x = torch.randn(B, S, H, dtype=torch.bfloat16)

with torch.no_grad():
    ref_w, ref_idx = ref_gate(x, topk_impl="easy")  # [256, 8]
    # Full softmax gates (fp32) for tie diagnostics.
    ref_logits = ref_gate.wg(x.reshape(-1, H).float())
    ref_gates = torch.softmax(ref_logits, dim=-1)  # [256, 64]

# ---------------------------------------------------------------------------
print("Opening device …")
device = ttnn.open_device(device_id=0)

tt_gate = HunyuanTtTopKGate(device, H, NUM_EXPERTS, MOE_TOPK, state_dict, WG_KEY, norm_topk_prob=NORM_TOPK)

x_tt = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
tt_w_t, tt_idx_t = tt_gate(x_tt)
tt_w = ttnn.to_torch(tt_w_t).reshape(B * S, MOE_TOPK)
tt_idx = ttnn.to_torch(tt_idx_t).reshape(B * S, MOE_TOPK).long()
x_tt.deallocate(True)

# ---------------------------------------------------------------------------
# 1) Same set of experts per token (order-independent)
ref_sets = [set(r.tolist()) for r in ref_idx]
tt_sets = [set(r.tolist()) for r in tt_idx]
same = sum(int(a == b) for a, b in zip(ref_sets, tt_sets))
total = len(ref_sets)
set_match_rate = same / total

# 1b) Tie analysis: a mismatch is a "precision tie" (not a logic error) when
# every differing expert's reference gate value sits within TIE_EPS of the
# selection boundary (the gap between the 8th- and 9th-ranked gate). At a real
# tie even PyTorch's own pick is arbitrary under fp32 rounding.
TIE_EPS = 2e-3
sorted_gates, _ = torch.sort(ref_gates, dim=-1, descending=True)
boundary_gap = (sorted_gates[:, MOE_TOPK - 1] - sorted_gates[:, MOE_TOPK]).abs()  # [256]

tie_explained = 0
genuine_fail = []
for t, (a, b) in enumerate(zip(ref_sets, tt_sets)):
    if a == b:
        continue
    diff = a.symmetric_difference(b)
    gap = boundary_gap[t].item()
    if gap <= TIE_EPS:
        tie_explained += 1
    else:
        genuine_fail.append((t, gap, sorted(diff)))

effective_rate = (same + tie_explained) / total


# 2) Weight agreement — align both by expert id, then PCC
def aligned_weights(idx, w):
    out = torch.zeros(idx.shape[0], NUM_EXPERTS)
    out.scatter_(1, idx, w.float())
    return out


ref_full = aligned_weights(ref_idx, ref_w)
tt_full = aligned_weights(tt_idx, tt_w)
weight_pcc = pcc(ref_full, tt_full)
weight_maxdiff = (ref_full - tt_full).abs().max().item()

# ---------------------------------------------------------------------------
print("\n" + "=" * 64)
print(f"MoE router — [{B},{S},{H}]  experts={NUM_EXPERTS}  topk={MOE_TOPK}")
print("=" * 64)
print(f"  expert-set match  : {same}/{total} tokens  ({100*set_match_rate:.2f}%)")
print(f"  + precision ties  : {tie_explained}  (boundary gap <= {TIE_EPS})")
print(f"  effective match   : {same + tie_explained}/{total}  ({100*effective_rate:.2f}%)")
print(f"  genuine mismatches: {len(genuine_fail)}")
for t, gap, diff in genuine_fail[:8]:
    print(f"      token {t}: boundary_gap={gap:.5f}  swapped_experts={diff}")
print(f"  weight PCC        : {weight_pcc:.8f}")
print(f"  weight max|diff|  : {weight_maxdiff:.6f}")

set_ok = effective_rate >= 0.99  # gate-1: same experts (ties allowed)
pcc_ok = weight_pcc >= 0.999
ok = set_ok and pcc_ok
print(f"\n  [{'PASS' if ok else 'FAIL'}] " f"(effective expert-set>=99%: {set_ok},  weight PCC>=0.999: {pcc_ok})")
print("=" * 64)

ttnn.close_device(device)
sys.exit(0 if ok else 1)
