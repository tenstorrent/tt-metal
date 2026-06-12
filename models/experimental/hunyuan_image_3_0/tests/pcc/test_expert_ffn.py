# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC tests — TTNN expert FFN + full MoE layer vs PyTorch reference,
# on the REAL layer-0 weights from HunyuanImage-3.0.
#
#   test_expert_ffn_pcc : HunyuanTtMLP  vs ref HunyuanMLP   (single real expert)
#   test_moe_layer_pcc  : HunyuanTtMoE  vs ref HunyuanMoE   (full layer)
#
# Pass criterion is PCC >= 0.999. A <1e-2 ABSOLUTE tolerance vs the fp32 reference
# is not attainable with bf16/BFP8 activations (bf16's 8-bit mantissa gives ~0.4%
# rel error over a 4096-dim contraction); PCC captures "numerically correct".
#
# Run:
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_expert_ffn.py -v -s
#   # or directly:
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

PCC_THR = 0.999
B, S = 1, 256


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


def _summary(name, ref, tt):
    p = _pcc(ref, tt)
    d = (ref.float() - tt.float()).abs().max().item()
    rel = d / (ref.float().abs().max().item() + 1e-9)
    print(f"  {name}: PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    return p


# --- sharded weight loading -------------------------------------------------
_WMAP = json.load(open(glob.glob(f"{WEIGHTS}/*.index.json")[0]))["weight_map"]
_OPEN = {}


def _load(key):
    shard = _WMAP[key]
    f = _OPEN.get(shard) or _OPEN.setdefault(shard, safe_open(f"{WEIGHTS}/{shard}", framework="pt"))
    return f.get_tensor(key)


def _load_prefix(prefix):
    return {k[len(prefix) + 1 :]: _load(k) for k in _WMAP if k.startswith(prefix + ".")}


def _cfg():
    cfg = json.load(open(f"{WEIGHTS}/config.json"))
    first = lambda v: v if isinstance(v, int) else v[0]
    return dict(
        H=cfg["hidden_size"],
        NUM_EXPERTS=first(cfg["num_experts"]),
        MOE_TOPK=first(cfg["moe_topk"]),
        MOE_INTER=first(cfg["moe_intermediate_size"]),
        NUM_SHARED=first(cfg["num_shared_expert"]),
        NORM_TOPK=cfg.get("norm_topk_prob", True),
        USE_MIXED=cfg.get("use_mixed_mlp_moe", True),
    )


LAYER = "model.layers.0"


def _expert_ffn_pcc(device):
    c = _cfg()
    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"], dtype=torch.bfloat16)

    expert_prefix = f"{LAYER}.mlp.experts.0"
    exp_sd = _load_prefix(expert_prefix)
    ref_mlp = RefMLP(c["H"], c["MOE_INTER"], is_moe=True)
    ref_mlp.load_state_dict({k: v.float() for k, v in exp_sd.items()})
    ref_mlp.eval()
    with torch.no_grad():
        ref_out = ref_mlp(x.float())

    tt_sd = {f"{expert_prefix}.{k}": v for k, v in exp_sd.items()}
    tt_mlp = HunyuanTtMLP(device, c["H"], tt_sd, expert_prefix)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.to_torch(tt_mlp(x_tt))
    ttnn.deallocate(x_tt)
    return _summary("expert FFN", ref_out, tt_out)


def _moe_layer_pcc(device):
    c = _cfg()
    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"], dtype=torch.bfloat16)

    sd = _load_prefix(LAYER)  # under model.layers.0 — includes attn keys, MoE only used below
    moe_sd = {k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")}

    ref_moe = RefMoE(
        c["H"],
        c["MOE_INTER"],
        num_experts=c["NUM_EXPERTS"],
        moe_topk=c["MOE_TOPK"],
        num_shared_expert=c["NUM_SHARED"],
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
    )
    ref_moe.load_state_dict({k: v.float() for k, v in moe_sd.items()})
    ref_moe.eval()
    with torch.no_grad():
        ref_out = ref_moe(x.float())

    tt_moe_sd = {f"{LAYER}.mlp.{k}": v for k, v in moe_sd.items()}
    tt_moe = HunyuanTtMoE(
        device,
        c["H"],
        c["NUM_EXPERTS"],
        c["MOE_TOPK"],
        tt_moe_sd,
        f"{LAYER}.mlp",
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        stream_experts=True,
    )
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out = ttnn.to_torch(tt_moe(x_tt))
    ttnn.deallocate(x_tt)
    return _summary("full MoE layer", ref_out, tt_out)


# --- pytest entry points ----------------------------------------------------
def test_expert_ffn_pcc(device):
    assert _expert_ffn_pcc(device) >= PCC_THR


def test_moe_layer_pcc(device):
    assert _moe_layer_pcc(device) >= PCC_THR


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        print("Part A: expert FFN")
        a = _expert_ffn_pcc(dev)
        print("Part B: full MoE layer")
        b = _moe_layer_pcc(dev)
    finally:
        ttnn.close_device(dev)
    n = int(a >= PCC_THR) + int(b >= PCC_THR)
    print(f"\nExpert FFN + MoE: {n}/2 PASSED")
    sys.exit(0 if n == 2 else 1)
