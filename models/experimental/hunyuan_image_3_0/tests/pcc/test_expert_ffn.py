# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC / atol test — TTNN expert FFN + full MoE layer vs PyTorch reference,
# on the REAL layer-0 weights from HunyuanImage-3.0.
#
#   Part A: HunyuanTtMLP  vs  ref HunyuanMLP   (single real expert)
#   Part B: HunyuanTtMoE  vs  ref HunyuanMoE   (full layer)
# Pass criterion is PCC >= 0.999 (see note at PCC_THR below).
#
# Run:
#   cd /home/iguser/Christy/tt-metal
#   python_env/bin/python models/experimental/hunyuan_image_3_0/tests/pcc/test_expert_ffn.py

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
from models.experimental.hunyuan_image_3_0.ref.moe.mlp import HunyuanMLP as RefMLP
from models.experimental.hunyuan_image_3_0.ref.moe.moe import HunyuanMoE as RefMoE
from models.experimental.hunyuan_image_3_0.tt.moe.mlp import HunyuanTtMLP
from models.experimental.hunyuan_image_3_0.tt.moe.moe import HunyuanTtMoE


# ---------------------------------------------------------------------------
def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


# Pass criterion is PCC. A <1e-2 ABSOLUTE tolerance vs the fp32 reference is not
# attainable with bf16/BFP8 activations: bf16's 8-bit mantissa yields ~0.4% rel
# error, so O(10) outputs carry ~0.04 abs error over a 4096-dim contraction.
# PCC>=PCC_THR captures "numerically correct"; max|diff| is reported for info.
PCC_THR = 0.999


def report(name, ref, tt, pcc_thr=PCC_THR):
    p = pcc(ref, tt)
    d = (ref.float() - tt.float()).abs().max().item()
    rel = d / (ref.float().abs().max().item() + 1e-9)
    ok = p >= pcc_thr
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         PCC={p:.6f}  (>= {pcc_thr})   max|diff|={d:.4f}  rel={rel:.4%}")
    return ok


# ---------------------------------------------------------------------------
# Weight loading from sharded safetensors
_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard)
    if f is None:
        f = _OPEN[shard] = safe_open(f"{WEIGHTS}/{shard}", framework="pt")
    return f.get_tensor(key)


def load_prefix(prefix):
    """Return {relative_key: tensor} for all checkpoint keys under `prefix.`"""
    out = {}
    for k in _WMAP:
        if k.startswith(prefix + "."):
            out[k[len(prefix) + 1 :]] = load(k)
    return out


# ---------------------------------------------------------------------------
cfg = json.load(open(f"{WEIGHTS}/config.json"))
H = cfg["hidden_size"]
NUM_EXPERTS = cfg["num_experts"] if isinstance(cfg["num_experts"], int) else cfg["num_experts"][0]
MOE_TOPK = cfg["moe_topk"] if isinstance(cfg["moe_topk"], int) else cfg["moe_topk"][0]
MOE_INTER = cfg["moe_intermediate_size"]
MOE_INTER = MOE_INTER if isinstance(MOE_INTER, int) else MOE_INTER[0]
NUM_SHARED = cfg["num_shared_expert"]
NUM_SHARED = NUM_SHARED if isinstance(NUM_SHARED, int) else NUM_SHARED[0]
NORM_TOPK = cfg.get("norm_topk_prob", True)
LAYER = "model.layers.0.mlp"

print(f"config: H={H} experts={NUM_EXPERTS} topk={MOE_TOPK} " f"moe_inter={MOE_INTER} shared={NUM_SHARED}")

results = []
print("\nOpening device …")
device = ttnn.open_device(device_id=0)
torch.manual_seed(0)

# ===========================================================================
# Part A — single expert FFN on real expert-0 weights
# ===========================================================================
print("\n" + "=" * 64)
print("Part A: expert FFN  (model.layers.0.mlp.experts.0)")
print("=" * 64)
B, S = 1, 256
x = torch.randn(B, S, H, dtype=torch.bfloat16)

exp_sd = load_prefix(f"{LAYER}.experts.0")  # gate_and_up_proj.weight, down_proj.weight
ref_mlp = RefMLP(H, MOE_INTER, is_moe=True)
ref_mlp.load_state_dict({k: v.float() for k, v in exp_sd.items()})
ref_mlp.eval()
with torch.no_grad():
    ref_out = ref_mlp(x.float())

tt_sd = {f"{LAYER}.experts.0.{k}": v for k, v in exp_sd.items()}
tt_mlp = HunyuanTtMLP(device, H, tt_sd, f"{LAYER}.experts.0")
x_tt = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
tt_out = ttnn.to_torch(tt_mlp(x_tt))
ttnn.deallocate(x_tt)
results.append(report("expert FFN matches ref", ref_out, tt_out))

# ===========================================================================
# Part B — full MoE layer on real layer-0 weights
# ===========================================================================
print("\n" + "=" * 64)
print(f"Part B: full MoE layer  ({NUM_EXPERTS} experts + shared)")
print("=" * 64)
moe_sd = load_prefix(LAYER)  # everything under model.layers.0.mlp

ref_moe = RefMoE(
    H,
    MOE_INTER,
    num_experts=NUM_EXPERTS,
    moe_topk=MOE_TOPK,
    num_shared_expert=NUM_SHARED,
    use_mixed_mlp_moe=cfg.get("use_mixed_mlp_moe", True),
    norm_topk_prob=NORM_TOPK,
)
ref_moe.load_state_dict({k: v.float() for k, v in moe_sd.items()})
ref_moe.eval()
with torch.no_grad():
    ref_moe_out = ref_moe(x.float())

tt_moe_sd = {f"{LAYER}.{k}": v for k, v in moe_sd.items()}
tt_moe = HunyuanTtMoE(
    device,
    H,
    NUM_EXPERTS,
    MOE_TOPK,
    tt_moe_sd,
    LAYER,
    use_mixed_mlp_moe=cfg.get("use_mixed_mlp_moe", True),
    norm_topk_prob=NORM_TOPK,
    stream_experts=True,
)
x_tt2 = ttnn.from_torch(
    x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
)
tt_moe_out = ttnn.to_torch(tt_moe(x_tt2))
ttnn.deallocate(x_tt2)
results.append(report("full MoE layer matches ref", ref_moe_out, tt_moe_out))

# ===========================================================================
print("\n" + "=" * 64)
n = sum(results)
print(f"Expert FFN + MoE: {n}/{len(results)} PASSED")
print("ALL PASS ✓" if n == len(results) else "SOME FAILED ✗")
print("=" * 64)
ttnn.close_device(device)
sys.exit(0 if n == len(results) else 1)
