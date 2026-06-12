# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PCC test — TTNN single transformer decoder layer vs PyTorch reference,
# on the REAL layer-0 weights from HunyuanImage-3.0.
#
#   HunyuanTtDecoderLayer  vs  ref HunyuanImage3DecoderLayer
#   (input_norm -> attn(+2D RoPE) -> +residual -> post_norm -> MoE -> +residual)
#
# The reference is itself proven bit-exact vs the actual upstream
# HunyuanImage3DecoderLayer in test_ref_layer_bit_exact.py.
#
# Pass criterion is PCC (a tight absolute tolerance is unreachable with bf16/BFP8
# over a 4096-dim contraction; PCC captures "numerically correct").
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer_layer.py -v
# Run (script):
#   python_env/bin/python models/experimental/hunyuan_image_3_0/tests/pcc/test_transformer_layer.py

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
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.tt.transformer_layer import HunyuanTtDecoderLayer

PCC_THR = 0.99
B, S = 1, 256


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
    """Returns (pcc, max_abs_diff, rel) for the TT decoder layer vs ref."""
    c = _cfg()
    LAYER = "model.layers.0"
    print(
        f"config: H={c['H']} heads={c['HEADS']}/{c['KV_HEADS']} head_dim={c['HEAD_DIM']} "
        f"experts={c['NUM_EXPERTS']} topk={c['MOE_TOPK']} eps={c['EPS']}"
    )

    torch.manual_seed(0)
    x = torch.randn(B, S, c["H"], dtype=torch.bfloat16)
    sd = _load_prefix(LAYER)

    # Reference (fp32) on real weights.
    ref = RefLayer(
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
        layer_idx=0,
    )
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    ref.eval()

    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=None)
    mask_add = to_additive(build_attention_mask(S, image_slices=None, bsz=B), dtype=torch.float32)
    with torch.no_grad():
        ref_out = ref(x.float(), attention_mask=mask_add, custom_pos_emb=(cos, sin))

    # TT on real weights.
    tt_sd = {f"{LAYER}.{k}": v for k, v in sd.items()}
    tt_layer = HunyuanTtDecoderLayer(
        device,
        tt_sd,
        layer_num=0,
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
    )
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = tt_layer(x_tt, seq_len=S, image_infos=None, attention_mask=None)  # is_causal
    tt_out = ttnn.to_torch(out_tt)[..., : c["H"]]
    ttnn.deallocate(x_tt)

    p = _pcc(ref_out, tt_out)
    d = (ref_out.float() - tt_out.float()).abs().max().item()
    rel = d / (ref_out.float().abs().max().item() + 1e-9)
    return p, d, rel


def test_decoder_layer_pcc(device):
    p, d, rel = _run(device)
    print(f"PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    assert p >= PCC_THR, f"decoder layer PCC {p:.6f} below {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        p, d, rel = _run(dev)
    finally:
        ttnn.close_device(dev)
    ok = p >= PCC_THR
    print("\n" + "=" * 64)
    print(f"Single transformer layer — [{B},{S}]")
    print(f"  [{'PASS' if ok else 'FAIL'}] PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    print("=" * 64)
    sys.exit(0 if ok else 1)
