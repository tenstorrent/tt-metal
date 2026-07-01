# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Teacher-forced FINAL-output PCC test — TTNN transformer backbone vs PyTorch
# reference, on the REAL HunyuanImage-3.0 weights, across ALL 32 decoder layers.
#
# Unlike test_model_teacher_forced.py (which reports 32 independent per-layer
# PCCs), this script runs the full 32-layer stack under teacher forcing and emits
# ONE number: the PCC of the FINAL output (after ln_f) vs the reference final.
#
# Teacher forcing here means every TT layer i is fed the GOLDEN fp32 reference
# hidden state that fed reference layer i — exactly as tt_transformers feeds the
# golden token into the model at every decode step (tests/test_model.py:399-400).
# Because each layer is re-anchored to golden, no cross-layer error accumulates,
# so the final value reflects only the LAST layer + ln_f numerics — not the bf16
# drift that the free-running chained path accumulates (see
# test_model_teacher_forced.py::test_model_final_output_pcc, ~0.88).
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model_teacher_forced_final.py -v -s --timeout=1800
# Run (script):
#   HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model_teacher_forced_final.py

import os, sys, json, glob, gc
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
from models.experimental.hunyuan_image_3_0.tt.transformer_layer import HunyuanTtDecoderLayer
from models.experimental.hunyuan_image_3_0.tt.attention.rms_norm import HunyuanTtRMSNorm
from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

PCC_THR = 0.99
B, S = 1, 256
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "32"))


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


def _to_tt(device, x):
    """torch [B,S,H] fp32 -> TTNN [B,S,H] TILE bf16 on device."""
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _reference(c, input_ids):
    """Reference fp32 forward over NUM_LAYERS.

    Returns:
        golden:    list len NUM_LAYERS+1; golden[i] = hidden state at input of
                   layer i (golden[0] = embedding); golden[i+1] = layer i output.
        ref_final: ln_f(golden[NUM_LAYERS]) — reference final output.
    """
    wte_w = _load("model.wte.weight")
    lnf_w = _load("model.ln_f.weight")

    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=None)
    mask_add = to_additive(build_attention_mask(S, image_slices=None, bsz=B), dtype=torch.float32)

    golden = []
    with torch.no_grad():
        h = torch.nn.functional.embedding(input_ids, wte_w.float())  # [B,S,H]
        golden.append(h.clone())
        for i in range(NUM_LAYERS):
            sd = _load_prefix(f"model.layers.{i}")
            ref_layer = RefLayer(
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
            ref_layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
            ref_layer.eval()
            h = ref_layer(h, attention_mask=mask_add, custom_pos_emb=(cos, sin))
            golden.append(h.clone())
            del ref_layer
            gc.collect()

        ref_lnf = HunyuanRMSNorm(c["H"], eps=c["EPS"])
        ref_lnf.load_state_dict({"weight": lnf_w.float()})
        ref_lnf.eval()
        ref_final = ref_lnf(golden[NUM_LAYERS])

    return golden, ref_final


def _run(device):
    """Teacher-forced pass over all NUM_LAYERS layers; return the FINAL-output PCC.

    Every layer is fed its golden input. The teacher-forced output of the LAST
    layer is passed through ln_f and compared (single PCC) to the reference final.

    Returns: (final_pcc, max_abs_diff, rel, per_layer_pcc)
    """
    c = _cfg()
    print(
        f"config: H={c['H']} heads={c['HEADS']}/{c['KV_HEADS']} head_dim={c['HEAD_DIM']} "
        f"experts={c['NUM_EXPERTS']} topk={c['MOE_TOPK']} eps={c['EPS']}  layers={NUM_LAYERS}  "
        f"(teacher-forced, final output)"
    )

    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (B, S), dtype=torch.long)
    golden, ref_final = _reference(c, input_ids)
    lnf_w = _load("model.ln_f.weight")

    cos_tt = sin_tt = None
    last_out = None  # teacher-forced output of the most recent layer (kept for ln_f)
    per_layer = []
    for i in range(NUM_LAYERS):
        layer_sd = {f"model.layers.{i}.{k}": v for k, v in _load_prefix(f"model.layers.{i}").items()}
        tt_layer = HunyuanTtDecoderLayer(
            device,
            layer_sd,
            layer_num=i,
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
        if cos_tt is None:
            cos_tt, sin_tt = tt_layer.self_attn.rope.prepare_cos_sin(S, image_infos=None)

        x_tt = _to_tt(device, golden[i])  # teacher forcing: feed the GOLDEN input
        out_tt = tt_layer(x_tt, seq_len=S, image_infos=None, attention_mask=None, cos_sin=(cos_tt, sin_tt))
        ttnn.deallocate(x_tt)

        # Track per-layer PCC (diagnostic) and keep only the last layer's output.
        per_layer.append(_pcc(golden[i + 1], ttnn.to_torch(out_tt)[..., : c["H"]]))
        if last_out is not None:
            ttnn.deallocate(last_out)
        last_out = out_tt
        print(f"  layer {i:2d}: teacher-forced PCC = {per_layer[-1]:.6f}")

        del tt_layer
        gc.collect()

    # ---- final norm on the teacher-forced last-layer output ----
    ln_f = HunyuanTtRMSNorm(device, c["H"], {"model.ln_f.weight": lnf_w}, "model.ln_f", eps=c["EPS"])
    final_tt = ln_f(last_out)
    ttnn.deallocate(last_out)
    final = ttnn.to_torch(final_tt)[..., : c["H"]]
    ttnn.deallocate(final_tt)
    if cos_tt is not None:
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    final_pcc = _pcc(ref_final, final)
    d = (ref_final.float() - final.float()).abs().max().item()
    rel = d / (ref_final.float().abs().max().item() + 1e-9)
    return final_pcc, d, rel, per_layer


def test_model_teacher_forced_final_pcc(device):
    final_pcc, d, rel, per_layer = _run(device)
    print(
        f"\nTEACHER-FORCED FINAL output after {NUM_LAYERS} layers + ln_f: "
        f"PCC={final_pcc:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}"
    )
    assert final_pcc >= PCC_THR, f"teacher-forced final-output PCC {final_pcc:.6f} below {PCC_THR}"


if __name__ == "__main__":
    dev = ttnn.open_device(device_id=0)
    try:
        final_pcc, d, rel, per_layer = _run(dev)
    finally:
        ttnn.close_device(dev)
    ok = final_pcc >= PCC_THR
    print("\n" + "=" * 64)
    print(f"Teacher-forced FINAL output — [{B},{S}] x {NUM_LAYERS} layers + ln_f")
    print(f"  [{'PASS' if ok else 'FAIL'}] PCC={final_pcc:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    print("=" * 64)
    sys.exit(0 if ok else 1)
