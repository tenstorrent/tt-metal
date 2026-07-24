# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated PCC tests for the attention stack (gpt_oss test_modules.py pattern):
#   RMSNorm, 2D RoPE, GQA attention, TT attention mask.
#
# Lean ISL: S=1, 32, 4096, 4160 (+ S=22784 max-context slow tests).
#
# Run (pytest):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_attention_modules.py -v -s
# Fast only (skip S>=4096 mask / max-context):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_attention_modules.py -m "not slow" -v
# ISL sweep CSV:
#   HY_PCC_CSV=/tmp/hunyuan_isl.csv python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_attention_modules.py -k isl_sweep -v -s

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

import ttnn
from models.experimental.hunyuan_image_3_0.ref.attention.attention import (
    AttentionConfig,
    HunyuanImage3SDPAAttention as RefAttn,
    build_causal_mask,
)
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import (
    apply_rotary_pos_emb,
    build_batch_2d_rope,
)
from models.experimental.hunyuan_image_3_0.ref.weights import load_tensors, resolve_base_model_dir
from models.experimental.hunyuan_image_3_0.tt.attention.attention import HunyuanTtAttention
from models.experimental.hunyuan_image_3_0.tt.attention.mask import _NEG, build_attention_mask_tt
from models.experimental.hunyuan_image_3_0.tt.attention.rms_norm import HunyuanTtRMSNorm
from models.experimental.hunyuan_image_3_0.tt.attention.rope_2d import HunyuanTtRoPE2D
from pcc_common import (
    BATCH_CASE,
    LEAN_ISL_CASES,
    MASK_CASES,
    PRODUCTION_IMAGE_SPAN,
    PRODUCTION_MODULE_CASES,
    PRODUCTION_PHASE_CASES,
    PRODUCTION_SEQ,
    PCC_BLOCK,
    PCC_STRICT,
    ROPE_ATTN_PCC_CASES,
    isl_csv_path,
    load_config,
    max_seq_tile_aligned,
    model_dims,
    pcc_metrics,
    rope_image_infos,
    write_isl_csv,
)

INPUT_NORM_KEY = "model.layers.0.input_layernorm.weight"
ATTN_WEIGHT_KEYS = [
    "model.layers.0.self_attn.qkv_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.query_layernorm.weight",
    "model.layers.0.self_attn.key_layernorm.weight",
]

CAUSAL_MASK_FAST = [(s, label) for _, s, label in LEAN_ISL_CASES if s < 4096]
CAUSAL_MASK_SLOW = [(s, label) for _, s, label in LEAN_ISL_CASES if s >= 4096]
IMAGE_SPAN_MASK_CASES = [
    (name, s, per_batch) for name, s, per_batch in MASK_CASES if per_batch != [[]] and s != PRODUCTION_SEQ
]

_ATTN_CACHE: dict[int, tuple] = {}


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
def _rms_norm_run(device, batch, seq_len, *, seed=0):
    cfg = load_config()
    w = load_tensors(resolve_base_model_dir(), [INPUT_NORM_KEY])[INPUT_NORM_KEY]
    h, eps = cfg["hidden_size"], cfg.get("rms_norm_eps", 1e-5)
    state_dict = {INPUT_NORM_KEY: w}

    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, h, dtype=torch.bfloat16)
    ref = HunyuanRMSNorm(h, eps=eps)
    ref.weight = torch.nn.Parameter(w.clone())
    ref.eval()
    with torch.no_grad():
        pt_out = ref(x)

    tt = HunyuanTtRMSNorm(device, h, state_dict, INPUT_NORM_KEY, eps=eps)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = tt(x_tt)
    tt_out = ttnn.to_torch(out_tt)
    x_tt.deallocate(True)
    out_tt.deallocate(True)
    return pcc_metrics(pt_out, tt_out, PCC_STRICT)


@pytest.mark.parametrize("batch,seq_len,label", LEAN_ISL_CASES)
def test_rms_norm_pcc(device, batch, seq_len, label):
    p, d = _rms_norm_run(device, batch, seq_len)
    print(f"RMSNorm [{batch}, {seq_len}] ({label}): PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_STRICT, f"RMSNorm {label} PCC {p:.8f} < {PCC_STRICT}"


def test_rms_norm_batch_pcc(device):
    batch, seq_len, label = BATCH_CASE
    p, d = _rms_norm_run(device, batch, seq_len)
    print(f"RMSNorm batch [{batch}, {seq_len}] ({label}): PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_STRICT


@pytest.mark.slow
def test_rms_norm_max_context_pcc(device):
    max_seq = max_seq_tile_aligned()
    p, d = _rms_norm_run(device, 1, max_seq)
    print(f"RMSNorm max context S={max_seq}: PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_STRICT


@pytest.mark.slow
@pytest.mark.parametrize("seq_len,label", PRODUCTION_PHASE_CASES)
def test_rms_norm_production_pcc(device, seq_len, label):
    """Production submodule gate: decode S=1 and prefill S=4160."""
    p, d = _rms_norm_run(device, 1, seq_len)
    phase = "decode" if seq_len == 1 else "prefill"
    print(f"RMSNorm production {phase} [{label}] S={seq_len}: PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_STRICT


# ---------------------------------------------------------------------------
# 2D RoPE
# ---------------------------------------------------------------------------
def _rope_run(device, batch, seq_len, image_infos=None, *, seed=0):
    _, n_h, n_kv, d_h = model_dims()
    tt_rope = HunyuanTtRoPE2D(device=device, head_dim=d_h)
    infos = rope_image_infos(image_infos, batch) if image_infos is not None else None
    if image_infos is None and batch > 1:
        infos = [None] * batch

    torch.manual_seed(seed)
    q_pt = torch.randn(batch, n_h, seq_len, d_h, dtype=torch.bfloat16)
    k_pt = torch.randn(batch, n_kv, seq_len, d_h, dtype=torch.bfloat16)
    cos_pt, sin_pt = build_batch_2d_rope(seq_len, d_h, image_infos=infos)
    q_ref, k_ref = apply_rotary_pos_emb(q_pt, k_pt, cos_pt, sin_pt)

    cos_tt, sin_tt = tt_rope.prepare_cos_sin(seq_len, image_infos=infos)
    q_tt = ttnn.from_torch(
        q_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    k_tt = ttnn.from_torch(
        k_pt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    q_rot, k_rot = tt_rope.forward(q_tt, k_tt, cos_tt, sin_tt)
    q_out, k_out = ttnn.to_torch(q_rot), ttnn.to_torch(k_rot)
    for t in (q_tt, k_tt, q_rot, k_rot, cos_tt, sin_tt):
        t.deallocate(True)
    return pcc_metrics(q_ref, q_out, PCC_STRICT), pcc_metrics(k_ref, k_out, PCC_STRICT)


def _assert_rope(label, q_m, k_m):
    (qp, qd), (kp, kd) = q_m, k_m
    print(f"RoPE {label}: Q PCC={qp:.8f} K PCC={kp:.8f}")
    assert qp >= PCC_STRICT and kp >= PCC_STRICT


@pytest.mark.parametrize("mode,batch,seq_len,image_infos,label", ROPE_ATTN_PCC_CASES)
def test_rope_2d_pcc(device, mode, batch, seq_len, image_infos, label):
    _, n_h, n_kv, d_h = model_dims()
    infos = None if mode == "text" else image_infos
    _assert_rope(
        f"{mode} [{batch}, {n_h}/{n_kv}, {seq_len}, {d_h}] ({label})",
        *_rope_run(device, batch, seq_len, infos),
    )


def test_rope_2d_batch_pcc(device):
    batch, seq_len, label = BATCH_CASE
    _, n_h, n_kv, d_h = model_dims()
    _assert_rope(
        f"batch [{batch}, {n_h}/{n_kv}, {seq_len}, {d_h}] ({label})",
        *_rope_run(device, batch, seq_len),
    )


@pytest.mark.slow
def test_rope_2d_max_context_pcc(device):
    max_seq = max_seq_tile_aligned()
    _, n_h, n_kv, d_h = model_dims()
    _assert_rope(f"max context [{1}, {n_h}/{n_kv}, {max_seq}, {d_h}]", *_rope_run(device, 1, max_seq))


@pytest.mark.slow
@pytest.mark.parametrize("mode,seq_len,image_infos,label", PRODUCTION_MODULE_CASES)
def test_rope_2d_production_pcc(device, mode, seq_len, image_infos, label):
    """Production submodule gate: decode S=1 and prefill S=4160 (text + image layout)."""
    _, n_h, n_kv, d_h = model_dims()
    infos = None if mode == "text" else image_infos
    _assert_rope(
        f"{mode} production [{1}, {n_h}/{n_kv}, {seq_len}, {d_h}] ({label})",
        *_rope_run(device, 1, seq_len, infos),
    )


# ---------------------------------------------------------------------------
# Attention (B=1; decode S=1 + prefill S>1)
# ---------------------------------------------------------------------------
def _get_attn(device):
    if id(device) not in _ATTN_CACHE:
        cfg = load_config()
        sd = load_tensors(resolve_base_model_dir(), ATTN_WEIGHT_KEYS)
        h, n_h, n_kv, d_h = model_dims(cfg)
        eps = cfg.get("rms_norm_eps", 1e-5)
        ref_cfg = AttentionConfig(
            hidden_size=h,
            num_attention_heads=n_h,
            attention_head_dim=d_h,
            num_key_value_heads=n_kv,
            use_qk_norm=True,
            use_rotary_pos_emb=True,
            rms_norm_eps=eps,
        )
        ref = RefAttn(ref_cfg, layer_idx=0).to(torch.bfloat16).eval()
        ref.qkv_proj.weight = torch.nn.Parameter(sd["model.layers.0.self_attn.qkv_proj.weight"].clone())
        ref.o_proj.weight = torch.nn.Parameter(sd["model.layers.0.self_attn.o_proj.weight"].clone())
        ref.query_layernorm.weight = torch.nn.Parameter(sd["model.layers.0.self_attn.query_layernorm.weight"].clone())
        ref.key_layernorm.weight = torch.nn.Parameter(sd["model.layers.0.self_attn.key_layernorm.weight"].clone())
        tt = HunyuanTtAttention(
            device, sd, layer_num=0, hidden_size=h, num_heads=n_h, num_kv_heads=n_kv, head_dim=d_h, eps=eps
        )
        _ATTN_CACHE[id(device)] = (ref, tt, cfg)
    return _ATTN_CACHE[id(device)]


def _attention_run(device, batch, seq_len, image_infos=None, *, seed=0):
    ref, tt, cfg = _get_attn(device)
    h, _, _, d_h = model_dims(cfg)
    infos = rope_image_infos(image_infos, batch)
    text_only = image_infos is None
    torch.manual_seed(seed)
    x = torch.randn(batch, seq_len, h, dtype=torch.bfloat16)
    mask = None if text_only else build_causal_mask(seq_len, dtype=torch.bfloat16)
    cos, sin = build_batch_2d_rope(seq_len, d_h, image_infos=infos)
    with torch.no_grad():
        pt_out, _, _ = ref(
            x,
            attention_mask=mask,
            custom_pos_emb=(cos, sin),
            is_causal=text_only,
        )
    cos_tt, sin_tt = tt.rope.prepare_cos_sin(seq_len, image_infos=infos)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out_tt = tt.forward(x_tt, cos_tt, sin_tt, attention_mask=None)
    tt_out = ttnn.to_torch(out_tt)
    for t in (x_tt, out_tt, cos_tt, sin_tt):
        t.deallocate(True)
    return pcc_metrics(pt_out, tt_out, PCC_BLOCK)


@pytest.mark.parametrize("mode,batch,seq_len,image_infos,label", ROPE_ATTN_PCC_CASES)
def test_attention_pcc(device, mode, batch, seq_len, image_infos, label):
    infos = None if mode == "text" else image_infos
    p, d = _attention_run(device, batch, seq_len, infos)
    print(f"Attention {mode} [{batch}, {seq_len}] ({label}): PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_BLOCK


@pytest.mark.slow
def test_attention_max_context_pcc(device):
    max_seq = max_seq_tile_aligned()
    p, d = _attention_run(device, 1, max_seq)
    print(f"Attention max context S={max_seq}: PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_BLOCK, f"Attention max context PCC {p:.8f} < {PCC_BLOCK}"


@pytest.mark.slow
@pytest.mark.parametrize("mode,seq_len,image_infos,label", PRODUCTION_MODULE_CASES)
def test_attention_production_pcc(device, mode, seq_len, image_infos, label):
    """Production submodule gate: decode S=1 and prefill S=4160 (text + image layout)."""
    infos = None if mode == "text" else image_infos
    p, d = _attention_run(device, 1, seq_len, infos)
    phase = "decode" if seq_len == 1 else "prefill"
    print(f"Attention production {mode} {phase} [{label}] S={seq_len}: " f"PCC={p:.8f}  max_abs_diff={d:.6f}")
    assert p >= PCC_BLOCK


# ---------------------------------------------------------------------------
# TT attention mask (bitwise exact, not PCC)
# ---------------------------------------------------------------------------
def _mask_ref_additive(seq_len, per_batch, bsz):
    ref_bool = build_attention_mask(seq_len, per_batch, bsz=bsz)
    add = torch.where(
        ref_bool,
        torch.zeros((), dtype=torch.float32),
        torch.full((), _NEG, dtype=torch.float32),
    )
    return add.to(torch.bfloat16)


def _mask_run(device, seq_len, per_batch, *, label=""):
    bsz = len(per_batch) if per_batch else 1
    per_batch = per_batch or [[]]
    ref_add = _mask_ref_additive(seq_len, per_batch, bsz)
    tt = build_attention_mask_tt(device, seq_len, per_batch, bsz=bsz)
    tt_t = ttnn.to_torch(tt)[..., :seq_len, :seq_len].to(torch.bfloat16)
    tt.deallocate(True)
    ok = torch.equal(tt_t, ref_add) and tuple(tt_t.shape) == (bsz, 1, seq_len, seq_len)
    return {
        "label": label,
        "seq_len": seq_len,
        "batch": bsz,
        "bitwise_ok": torch.equal(tt_t, ref_add),
        "diff_elems": int((tt_t != ref_add).sum().item()),
        "pass": ok,
    }


@pytest.mark.parametrize("seq_len,label", CAUSAL_MASK_FAST)
def test_mask_causal_isl_bitwise(device, seq_len, label):
    mode = "decode" if seq_len == 1 else "prefill"
    r = _mask_run(device, seq_len, [[]], label=f"causal {mode} — {label}")
    print(f"Mask [{r['label']}] S={seq_len}: pass={r['pass']}")
    assert r["pass"]


@pytest.mark.slow
@pytest.mark.parametrize("seq_len,label", CAUSAL_MASK_SLOW)
def test_mask_causal_large_isl_bitwise(device, seq_len, label):
    r = _mask_run(device, seq_len, [[]], label=f"causal prefill — {label}")
    assert r["pass"]


@pytest.mark.parametrize("name,seq_len,per_batch", IMAGE_SPAN_MASK_CASES)
def test_mask_image_span_bitwise(device, name, seq_len, per_batch):
    r = _mask_run(device, seq_len, per_batch, label=name.strip())
    assert r["pass"]


@pytest.mark.slow
def test_mask_production_pcc(device):
    """Production layout mask @ S=4160 with 64×64 image span (bitwise exact)."""
    r = _mask_run(device, PRODUCTION_SEQ, PRODUCTION_IMAGE_SPAN, label="production layout S=4160")
    print(f"Mask production S={PRODUCTION_SEQ}: pass={r['pass']}")
    assert r["pass"]


@pytest.mark.slow
def test_mask_max_context_causal_bitwise(device):
    max_seq = max_seq_tile_aligned()
    r = _mask_run(device, max_seq, [[]], label=f"max context causal S={max_seq}")
    assert r["pass"]


# ---------------------------------------------------------------------------
# ISL sweep + CSV (RMSNorm + mask)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_isl_sweep_table(device, tmp_path):
    rows = []
    for batch, seq_len, label in LEAN_ISL_CASES + [BATCH_CASE]:
        p, d = _rms_norm_run(device, batch, seq_len)
        rows.append(
            {
                "module": "rms_norm",
                "batch": batch,
                "seq_len": seq_len,
                "label": label,
                "pcc": f"{p:.8f}",
                "max_abs_diff": f"{d:.6f}",
                "threshold": PCC_STRICT,
                "pass": p >= PCC_STRICT,
            }
        )
        print(f"  RMSNorm ISL {label:40s}  S={seq_len:5d}  PCC={p:.8f}")

    for seq_len, label in CAUSAL_MASK_FAST + CAUSAL_MASK_SLOW:
        r = _mask_run(device, seq_len, [[]], label=f"mask causal — {label}")
        rows.append(
            {
                "module": "attention_mask",
                "batch": 1,
                "seq_len": seq_len,
                "label": label,
                "pcc": "",
                "max_abs_diff": r["diff_elems"],
                "threshold": "bitwise",
                "pass": r["pass"],
            }
        )

    out = isl_csv_path() or tmp_path / "hunyuan_attention_isl_sweep.csv"
    write_isl_csv(rows, out)
    print(f"\nISL sweep CSV: {out}")
    assert all(r["pass"] for r in rows)
