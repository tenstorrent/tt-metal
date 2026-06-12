# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — HunyuanTtTopKGate (TTNN) vs PyTorch reference (easy_topk).
# Uses the real layer-0 router weight from HunyuanImage-3.0.
#
# Gate-1 requirement: the TT router must select the SAME set of experts per
# token as PyTorch (precision ties at the top-k boundary allowed), and the
# normalised routed weights must match to high PCC.
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_moe_router.py -v -s
#   # or directly:
#   python_env/bin/python models/experimental/hunyuan_image_3_0/tests/pcc/test_moe_router.py

import sys, json, glob
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

TIE_EPS = 2e-3  # top-k boundary tie tolerance
SET_THR = 0.99  # min effective expert-set match rate
WPCC_THR = 0.999  # min routed-weight PCC
B, S = 1, 256


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _measure(device):
    cfg = json.load(open(f"{WEIGHTS}/config.json"))
    H = cfg["hidden_size"]
    NUM_EXPERTS = cfg["num_experts"] if isinstance(cfg["num_experts"], int) else cfg["num_experts"][0]
    MOE_TOPK = cfg["moe_topk"] if isinstance(cfg["moe_topk"], int) else cfg["moe_topk"][0]
    NORM_TOPK = cfg.get("norm_topk_prob", True)

    WG_KEY = "model.layers.0.mlp.gate.wg.weight"
    wmap = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
    with safe_open(f"{WEIGHTS}/{wmap[WG_KEY]}", framework="pt") as f:
        wg = f.get_tensor(WG_KEY)  # [num_experts, hidden]
    state_dict = {WG_KEY: wg}

    # PyTorch reference (easy_topk = the inference path).
    ref_gate = HunyuanTopKGate(
        hidden_size=H, num_experts=NUM_EXPERTS, moe_topk=MOE_TOPK, norm_topk_prob=NORM_TOPK
    ).eval()
    ref_gate.wg.weight = torch.nn.Parameter(wg.float().clone())

    torch.manual_seed(0)
    x = torch.randn(B, S, H, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_w, ref_idx = ref_gate(x, topk_impl="easy")
        ref_gates = torch.softmax(ref_gate.wg(x.reshape(-1, H).float()), dim=-1)

    tt_gate = HunyuanTtTopKGate(device, H, NUM_EXPERTS, MOE_TOPK, state_dict, WG_KEY, norm_topk_prob=NORM_TOPK)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_w_t, tt_idx_t = tt_gate(x_tt)
    tt_w = ttnn.to_torch(tt_w_t).reshape(B * S, MOE_TOPK)
    tt_idx = ttnn.to_torch(tt_idx_t).reshape(B * S, MOE_TOPK).long()
    x_tt.deallocate(True)

    # Same expert set per token (ties at the boundary excused).
    ref_sets = [set(r.tolist()) for r in ref_idx]
    tt_sets = [set(r.tolist()) for r in tt_idx]
    same = sum(int(a == b) for a, b in zip(ref_sets, tt_sets))
    total = len(ref_sets)

    sorted_gates, _ = torch.sort(ref_gates, dim=-1, descending=True)
    boundary_gap = (sorted_gates[:, MOE_TOPK - 1] - sorted_gates[:, MOE_TOPK]).abs()
    tie_explained = sum(
        1 for t, (a, b) in enumerate(zip(ref_sets, tt_sets)) if a != b and boundary_gap[t].item() <= TIE_EPS
    )
    genuine = total - same - tie_explained
    effective_rate = (same + tie_explained) / total

    def aligned(idx, w):
        out = torch.zeros(idx.shape[0], NUM_EXPERTS)
        out.scatter_(1, idx, w.float())
        return out

    weight_pcc = _pcc(aligned(ref_idx, ref_w), aligned(tt_idx, tt_w))

    print(
        f"  experts={NUM_EXPERTS} topk={MOE_TOPK}  set_match={same}/{total} "
        f"+ties={tie_explained}  effective={100*effective_rate:.2f}%  "
        f"genuine_mismatch={genuine}  weight_PCC={weight_pcc:.6f}"
    )
    return effective_rate, weight_pcc, genuine


def test_moe_router_same_experts(device):
    effective_rate, weight_pcc, genuine = _measure(device)
    assert genuine == 0, f"{genuine} genuine (non-tie) expert-set mismatches"
    assert effective_rate >= SET_THR, f"effective expert-set match {effective_rate:.4f} < {SET_THR}"
    assert weight_pcc >= WPCC_THR, f"routed-weight PCC {weight_pcc:.6f} < {WPCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        eff, wpcc, gen = _measure(dev)
    finally:
        ttnn.close_device(dev)
    ok = gen == 0 and eff >= SET_THR and wpcc >= WPCC_THR
    print(
        f"\n  [{'PASS' if ok else 'FAIL'}]  effective>={SET_THR}: {eff>=SET_THR}  "
        f"genuine==0: {gen==0}  weight_PCC>={WPCC_THR}: {wpcc>=WPCC_THR}"
    )
    sys.exit(0 if ok else 1)
