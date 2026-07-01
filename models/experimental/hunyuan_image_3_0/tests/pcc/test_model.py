# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — TTNN transformer backbone (stack of decoder layers) vs PyTorch
# reference, on the REAL weights from HunyuanImage-3.0.
#
#   HunyuanTtModel  vs  [wte -> N x HunyuanImage3DecoderLayer -> ln_f]
#
# This stacks the already-verified single-layer port (test_transformer_layer.py)
# and adds the token embedding (model.wte) and final norm (model.ln_f). It runs
# a SMALL number of layers by default (HY_NUM_LAYERS, default 2) because with
# stream_experts=True every layer holds its torch expert weights in host RAM;
# the full 32-layer run needs disk-backed expert streaming (a future change).
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model.py -v
# Run (script, choose layer count):
#   HY_NUM_LAYERS=4 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model.py

import os, sys, json, glob
import torch
from safetensors import safe_open

ROOT = "/home/iguser/ign-tt/tt-metal"
HUNYUAN = "/home/iguser/ign-tt/hunyan_instruct"
WEIGHTS = "/home/iguser/ign-tt/base"
for p in (ROOT, HUNYUAN):
    if p not in sys.path:
        sys.path.insert(0, p)

import ttnn
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

PCC_THR = 0.99
B, S = 1, 256
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))


def _pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm()).clamp(min=1e-12)).item()


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
        HEADS=cfg["num_attention_heads"],
        KV_HEADS=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        HEAD_DIM=cfg.get("attention_head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
        NUM_EXPERTS=first(cfg["num_experts"]),
        MOE_TOPK=first(cfg["moe_topk"]),
        MOE_INTER=first(cfg["moe_intermediate_size"]),
        NUM_SHARED=first(cfg["num_shared_expert"]),
        NORM_TOPK=cfg.get("norm_topk_prob", True),
        EPS=cfg.get("rms_norm_eps", 1e-5),
        USE_QK_NORM=cfg.get("use_qk_norm", True),
        USE_MIXED=cfg.get("use_mixed_mlp_moe", True),
    )


def _run(device):
    """Returns (pcc, max_abs_diff, rel) for the TT backbone vs ref over NUM_LAYERS."""
    c = _cfg()
    print(
        f"config: H={c['H']} heads={c['HEADS']}/{c['KV_HEADS']} head_dim={c['HEAD_DIM']} "
        f"experts={c['NUM_EXPERTS']} topk={c['MOE_TOPK']} eps={c['EPS']}  layers={NUM_LAYERS}"
    )

    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (B, S), dtype=torch.long)

    # ---- reference: wte -> N decoder layers -> ln_f (all fp32) ----
    wte_w = _load("model.wte.weight")
    lnf_w = _load("model.ln_f.weight")

    def make_ref_layer(i):
        sd = _load_prefix(f"model.layers.{i}")
        layer = RefLayer(
            hidden_size=c["H"],
            num_attention_heads=c["HEADS"],
            num_key_value_heads=c["KV_HEADS"],
            attention_head_dim=c["HEAD_DIM"],
            num_experts=c["NUM_EXPERTS"],
            moe_topk=c["MOE_TOPK"],
            moe_intermediate_size=c["MOE_INTER"],
            num_shared_expert=c["NUM_SHARED"],
            use_mixed_mlp_moe=c["USE_MIXED"],
            norm_topk_prob=c["NORM_TOPK"],
            use_qk_norm=c["USE_QK_NORM"],
            rms_norm_eps=c["EPS"],
            layer_idx=i,
        )
        layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
        layer.eval()
        return layer

    ref_layers = [make_ref_layer(i) for i in range(NUM_LAYERS)]
    ref_lnf = HunyuanRMSNorm(c["H"], eps=c["EPS"])
    ref_lnf.load_state_dict({"weight": lnf_w.float()})
    ref_lnf.eval()

    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=None)
    mask_add = to_additive(build_attention_mask(S, image_slices=None, bsz=B), dtype=torch.float32)
    with torch.no_grad():
        h = torch.nn.functional.embedding(input_ids, wte_w.float())  # [B,S,H]
        for layer in ref_layers:
            h = layer(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
        ref_out = ref_lnf(h)

    # ---- TT backbone on real weights ----
    layer_sds = {
        i: {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()} for i in range(NUM_LAYERS)
    }
    tt_model = HunyuanTtModel(
        device,
        num_layers=NUM_LAYERS,
        hidden_size=c["H"],
        num_heads=c["HEADS"],
        num_kv_heads=c["KV_HEADS"],
        head_dim=c["HEAD_DIM"],
        num_experts=c["NUM_EXPERTS"],
        moe_topk=c["MOE_TOPK"],
        use_qk_norm=c["USE_QK_NORM"],
        use_mixed_mlp_moe=c["USE_MIXED"],
        norm_topk_prob=c["NORM_TOPK"],
        rms_norm_eps=c["EPS"],
        stream_experts=True,
        layer_loader=lambda i: layer_sds[i],
        embed_state_dict={"model.wte.weight": wte_w},
        norm_state_dict={"model.ln_f.weight": lnf_w},
        apply_final_norm=True,
    )

    ids_tt = ttnn.from_torch(
        input_ids.to(torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out_tt = tt_model(ids_tt, seq_len=S, image_infos=None, attention_mask=None)  # causal
    tt_out = ttnn.to_torch(out_tt)[..., : c["H"]]
    ttnn.deallocate(ids_tt)

    p = _pcc(ref_out, tt_out)
    d = (ref_out.float() - tt_out.float()).abs().max().item()
    rel = d / (ref_out.float().abs().max().item() + 1e-9)
    return p, d, rel


def test_model_pcc(device):
    p, d, rel = _run(device)
    print(f"PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    assert p >= PCC_THR, f"backbone PCC {p:.6f} below {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        p, d, rel = _run(dev)
    finally:
        ttnn.close_device(dev)
    ok = p >= PCC_THR
    print("\n" + "=" * 64)
    print(f"Transformer backbone — [{B},{S}] x {NUM_LAYERS} layers")
    print(f"  [{'PASS' if ok else 'FAIL'}] PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    print("=" * 64)
    sys.exit(0 if ok else 1)
