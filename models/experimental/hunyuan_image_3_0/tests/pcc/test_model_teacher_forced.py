# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Teacher-forced PCC test — TTNN transformer backbone vs PyTorch reference, on
# the REAL weights from HunyuanImage-3.0, across ALL 32 decoder layers.
#
# Motivation (mirrors tt_transformers/tests/test_model.py):
#   The tt_transformers accuracy test runs autoregressive decode under "teacher
#   forcing": at every step it feeds the GOLDEN token sampled from the reference
#   model into BOTH models (test_model.py:399-400), so the TT model never drifts
#   off the reference trajectory and the per-step PCC is not polluted by error
#   accumulated over previous steps.
#
#   The Hunyuan backbone has no autoregressive decode — its "steps" are the 32
#   stacked decoder layers. The analogous teacher forcing is therefore applied at
#   the LAYER boundary: TT layer i is fed the reference (golden) hidden state that
#   the fp32 reference produced as the input to layer i, and its output is checked
#   against the reference output of layer i. No TT layer ever sees another TT
#   layer's (already-degraded) output, so each layer is scored independently and
#   the per-layer PCC reflects that single layer's numerics — exactly as the
#   tt_transformers per-step PCC does.
#
# This also sidesteps the ~150GB host-RAM problem of holding 32 stream_experts
# layers at once (see tt/model.py header): we build, run, score, and free ONE TT
# layer at a time.
#
# Threshold: PCC >= 0.99 per layer (matches the single-layer / 2-layer Hunyuan
# PCC tests in this directory).
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model_teacher_forced.py -v
# Run (script, choose layer count):
#   HY_NUM_LAYERS=32 python_env/bin/python \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model_teacher_forced.py
#
# bf16-vs-bf8 mixed-precision boundary audit (which layers must stay bf16):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_model_teacher_forced.py \
#     -k bf8_mixed_precision_audit -v -s
#   # or as a script:  HY_BF8_AUDIT=1 ... test_model_teacher_forced.py --audit

import os, sys, json, glob, gc
import torch
from safetensors import safe_open

ROOT = "/home/iguser/Christy/tt-metal"
HUNYUAN = "/home/iguser/Christy/tt-metal/HunyuanImage-3.0"
WEIGHTS = "/home/iguser/Christy/HunyuanImage-3"
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

PCC_THR = 0.99  # strict per-layer gate (teacher-forced) — each layer scored in isolation
# Chained end-to-end gate. Looser than PCC_THR because running all NUM_LAYERS TT
# layers back-to-back accumulates bf16 rounding (the teacher-forced test confirms
# every layer is individually faithful at >= 0.99; the drop is pure accumulation).
# 0.86 matches the tt_transformers multi-layer floor (tests/test_model.py:136).
CHAIN_PCC_THR = 0.86
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
        golden:    list of length NUM_LAYERS+1; golden[i] is the hidden state at
                   the INPUT of layer i (golden[0] = embedding) and golden[i+1]
                   is layer i's reference output.
        ref_final: ln_f applied to golden[NUM_LAYERS] — the reference final
                   output after all layers.
    """
    wte_w = _load("model.wte.weight")
    lnf_w = _load("model.ln_f.weight")

    cos, sin = build_batch_2d_rope(S, c["HEAD_DIM"], image_infos=None)
    mask_add = to_additive(build_attention_mask(S, image_slices=None, bsz=B), dtype=torch.float32)

    golden = []  # golden[i] = reference hidden state at the input of layer i
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


def _run(device, weight_dtype=ttnn.bfloat16, golden=None):
    """Teacher-forced per-layer PCC at a given weight dtype.

    Returns list of (layer_idx, pcc, max_abs_diff, rel). Pass `golden` (the fp32
    reference states from `_reference`) to avoid recomputing it across a dtype
    sweep; if None it is computed here.
    """
    c = _cfg()
    print(
        f"config: H={c['H']} heads={c['HEADS']}/{c['KV_HEADS']} head_dim={c['HEAD_DIM']} "
        f"experts={c['NUM_EXPERTS']} topk={c['MOE_TOPK']} eps={c['EPS']}  layers={NUM_LAYERS}  "
        f"dtype={weight_dtype}  (teacher-forced)"
    )

    if golden is None:
        torch.manual_seed(0)
        input_ids = torch.randint(0, 130000, (B, S), dtype=torch.long)
        golden, _ = _reference(c, input_ids)

    # ---- teacher-forced TT pass: one layer at a time ----
    # Build the shared 2D RoPE tables once from the first TT layer's rope and
    # reuse them for every layer (they depend only on S and head_dim).
    cos_tt = sin_tt = None
    results = []
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
            weight_dtype=weight_dtype,
        )
        if cos_tt is None:
            cos_tt, sin_tt = tt_layer.self_attn.rope.prepare_cos_sin(S, image_infos=None)

        x_tt = _to_tt(device, golden[i])  # teacher forcing: feed the GOLDEN input
        out_tt = tt_layer(x_tt, seq_len=S, image_infos=None, attention_mask=None, cos_sin=(cos_tt, sin_tt))
        out = ttnn.to_torch(out_tt)[..., : c["H"]]
        ttnn.deallocate(out_tt)
        ttnn.deallocate(x_tt)

        ref_out = golden[i + 1]
        p = _pcc(ref_out, out)
        d = (ref_out.float() - out.float()).abs().max().item()
        rel = d / (ref_out.float().abs().max().item() + 1e-9)
        results.append((i, p, d, rel))
        print(f"  layer {i:2d}: PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")

        del tt_layer
        gc.collect()

    if cos_tt is not None:
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    return results


def _run_dtype_audit(device):
    """bf16-vs-bf8 per-layer drift audit — the mixed-precision boundary finder.

    Runs the teacher-forced per-layer PCC at BOTH ttnn.bfloat16 and ttnn.bfloat8_b
    against the SAME fp32 golden, isolating weight-quantization precision from
    parallelism (single device, dense MoE). Recommends the set of layers that must
    stay bf16 — those whose bf8 per-layer PCC drops below PCC_THR — which is exactly
    the `bf16_layers=` argument HunyuanTtModel / demo.py take. See MEMORY_FIT_PLAN.md
    step 1 ("de-risk bf8 backbone accuracy; decide mixed-precision boundaries").

    Returns:
        rows:      list of (layer_idx, pcc_bf16, pcc_bf8, drift=pcc_bf16-pcc_bf8).
        need_bf16: sorted list of layer indices whose bf8 PCC < PCC_THR.
    """
    c = _cfg()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (B, S), dtype=torch.long)
    golden, _ = _reference(c, input_ids)  # computed once, reused for both dtypes

    res16 = _run(device, weight_dtype=ttnn.bfloat16, golden=golden)
    res8 = _run(device, weight_dtype=ttnn.bfloat8_b, golden=golden)

    p16 = {i: p for (i, p, _, _) in res16}
    p8 = {i: p for (i, p, _, _) in res8}
    rows = [(i, p16[i], p8[i], p16[i] - p8[i]) for i in range(NUM_LAYERS)]
    need_bf16 = sorted(i for (i, _, b, _) in rows if b < PCC_THR)
    return rows, need_bf16


def _print_audit(rows, need_bf16):
    print("\nbf16-vs-bf8 per-layer drift audit (teacher-forced, fp32 golden):")
    print(f"  {'layer':>5}  {'bf16 PCC':>10}  {'bf8 PCC':>10}  {'drift':>10}")
    for i, a, b, d in rows:
        flag = "   <-- bf8 below threshold (keep bf16)" if b < PCC_THR else ""
        print(f"  {i:5d}  {a:10.6f}  {b:10.6f}  {d:10.6f}{flag}")
    print(f"\nPCC_THR={PCC_THR}.  Recommended bf16_layers = {need_bf16}")
    print(f"  ({len(need_bf16)}/{NUM_LAYERS} layers must stay bf16; the rest are bf8-safe per-layer.)")


def _run_chained(device):
    """End-to-end chained pass: each TT layer is fed the PREVIOUS TT layer's own
    output (NOT the golden state), then ln_f is applied. This is the real
    deployment path and shows error ACCUMULATED across all NUM_LAYERS layers —
    the contrast to the teacher-forced per-layer scoring in _run().

    Builds/runs/frees one layer at a time so all 32 layers fit in host RAM.

    Returns:
        trace:     list of (layer_idx, pcc_vs_golden) tracking drift per layer.
        final_pcc: PCC of the final output (after ln_f) vs the reference final.
        d, rel:    max abs diff / relative error of the final output.
    """
    c = _cfg()
    print(
        f"config: H={c['H']} heads={c['HEADS']}/{c['KV_HEADS']} head_dim={c['HEAD_DIM']} "
        f"experts={c['NUM_EXPERTS']} topk={c['MOE_TOPK']} eps={c['EPS']}  layers={NUM_LAYERS}  (chained end-to-end)"
    )

    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (B, S), dtype=torch.long)
    golden, ref_final = _reference(c, input_ids)
    lnf_w = _load("model.ln_f.weight")

    # Start from the (fp32) embedding so the layer stack is measured in isolation
    # from embedding numerics, then never reset — each layer consumes the prior
    # TT output.
    hidden = _to_tt(device, golden[0])

    cos_tt = sin_tt = None
    trace = []
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

        nxt = tt_layer(hidden, seq_len=S, image_infos=None, attention_mask=None, cos_sin=(cos_tt, sin_tt))
        ttnn.deallocate(hidden)
        hidden = nxt

        p = _pcc(golden[i + 1], ttnn.to_torch(hidden)[..., : c["H"]])
        trace.append((i, p))
        print(f"  layer {i:2d}: chained PCC vs golden = {p:.6f}")

        del tt_layer
        gc.collect()

    # ---- final norm (ln_f) on the chained hidden ----
    ln_f = HunyuanTtRMSNorm(device, c["H"], {"model.ln_f.weight": lnf_w}, "model.ln_f", eps=c["EPS"])
    out_tt = ln_f(hidden)
    ttnn.deallocate(hidden)
    final = ttnn.to_torch(out_tt)[..., : c["H"]]
    ttnn.deallocate(out_tt)
    if cos_tt is not None:
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

    final_pcc = _pcc(ref_final, final)
    d = (ref_final.float() - final.float()).abs().max().item()
    rel = d / (ref_final.float().abs().max().item() + 1e-9)
    return trace, final_pcc, d, rel


def test_model_teacher_forced_pcc(device):
    results = _run(device)
    failing = [(i, p) for (i, p, _, _) in results if p < PCC_THR]
    for i, p, d, rel in results:
        print(f"layer {i:2d}: PCC={p:.6f} (>= {PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}")
    assert not failing, f"teacher-forced per-layer PCC below {PCC_THR}: " + ", ".join(
        f"L{i}={p:.6f}" for i, p in failing
    )


def test_model_final_output_pcc(device):
    """Final output after all NUM_LAYERS layers (chained) + ln_f vs reference."""
    trace, final_pcc, d, rel = _run_chained(device)
    print(
        f"FINAL output after {NUM_LAYERS} layers: PCC={final_pcc:.6f} "
        f"(>= {CHAIN_PCC_THR})  max|diff|={d:.4f}  rel={rel:.4%}"
    )
    assert final_pcc >= CHAIN_PCC_THR, f"final-output PCC {final_pcc:.6f} below {CHAIN_PCC_THR}"


def test_bf8_mixed_precision_audit(device):
    """Informational: report per-layer bf16-vs-bf8 drift and the recommended
    bf16_layers set. Gates only the bf16 baseline (each layer must be >= PCC_THR
    in bf16; a bf16 regression is a real bug, not a quantization choice). bf8
    drops are expected and reported, not failed — the recommendation is the output.
    """
    rows, need_bf16 = _run_dtype_audit(device)
    _print_audit(rows, need_bf16)
    bf16_fail = [(i, a) for (i, a, _, _) in rows if a < PCC_THR]
    assert not bf16_fail, "bf16 baseline regressed (not a bf8 issue): " + ", ".join(
        f"L{i}={a:.6f}" for i, a in bf16_fail
    )


if __name__ == "__main__":
    audit = "--audit" in sys.argv or os.environ.get("HY_BF8_AUDIT") == "1"
    dev = ttnn.open_device(device_id=0)
    try:
        if audit:
            rows, need_bf16 = _run_dtype_audit(dev)
        results = _run(dev)
        trace, final_pcc, fd, frel = _run_chained(dev)
    finally:
        ttnn.close_device(dev)

    if audit:
        _print_audit(rows, need_bf16)

    failing = [(i, p) for (i, p, _, _) in results if p < PCC_THR]
    worst = min(results, key=lambda r: r[1])
    print("\n" + "=" * 64)
    print(f"Teacher-forced backbone — [{B},{S}] x {NUM_LAYERS} layers, per-layer PCC")
    print(f"  worst: layer {worst[0]} PCC={worst[1]:.6f} (>= {PCC_THR})")
    print(f"  [{'PASS' if not failing else 'FAIL'}] {len(results) - len(failing)}/{len(results)} layers >= {PCC_THR}")
    if failing:
        print("  failing: " + ", ".join(f"L{i}={p:.6f}" for i, p in failing))
    print("-" * 64)
    final_ok = final_pcc >= CHAIN_PCC_THR
    print(f"Chained end-to-end final output (after {NUM_LAYERS} layers + ln_f)")
    print(
        f"  [{'PASS' if final_ok else 'FAIL'}] PCC={final_pcc:.6f} "
        f"(>= {CHAIN_PCC_THR})  max|diff|={fd:.4f}  rel={frel:.4%}"
    )
    print("=" * 64)
    sys.exit(0 if (not failing and final_ok) else 1)
