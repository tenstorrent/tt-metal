# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated teacher-forced backbone PCC tests:
#   - Per-layer teacher-forced PCC (each layer fed golden input)
#   - Chained end-to-end final output (accumulated drift)
#   - Teacher-forced final output (all layers re-anchored, single PCC)
#   - bf16-vs-bf8 mixed-precision boundary audit
#
# Lean ISL: S=32 fast smoke (2 layers); full depth via HY_NUM_LAYERS=32 (slow).
# Production all-layer gates: decode S=1 and prefill S=4160 at 32L (@pytest.mark.slow).
#
# Run (fast smoke):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_teacher_forced.py -m "not slow" -v
# Production 32L decode + prefill:
#   HY_NUM_LAYERS=32 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_teacher_forced.py \
#     -k "production" -v -s

from __future__ import annotations

import gc
import os
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
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.weights import (
    load_prefixed_state_dict,
    load_tensors,
    resolve_base_model_dir,
)
from models.experimental.hunyuan_image_3_0.tt.attention.rms_norm import HunyuanTtRMSNorm
from models.experimental.hunyuan_image_3_0.tt.transformer_layer import HunyuanTtDecoderLayer
from pcc_common import (
    LEAN_ISL_CASES,
    PRODUCTION_SEQ,
    PCC_BLOCK,
    PCC_CHAINED,
    pcc_metrics,
    per_layer_pcc,
    transformer_cfg,
)

BATCH = 1
PCC_PER_LAYER = PCC_BLOCK
NUM_LAYERS_FAST = int(os.environ.get("HY_NUM_LAYERS", "2"))
NUM_LAYERS_FULL = int(os.environ.get("HY_NUM_LAYERS", "32"))

TF_ISL_FAST = [(batch, seq_len, label) for batch, seq_len, label in LEAN_ISL_CASES if seq_len < 4096]
TF_ISL_SLOW = [(batch, seq_len, label) for batch, seq_len, label in LEAN_ISL_CASES if seq_len >= 4096]
TF_ALL_LAYERS_PRODUCTION = [
    (1, "decode S=1"),
    (PRODUCTION_SEQ, "production prefill S=4160 (32+64×64+32)"),
]


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def _to_tt(device, x):
    return ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _reference(c: dict, input_ids: torch.Tensor, num_layers: int):
    wte_w = load_tensors(resolve_base_model_dir(), ["model.wte.weight"])["model.wte.weight"]
    lnf_w = load_tensors(resolve_base_model_dir(), ["model.ln_f.weight"])["model.ln_f.weight"]
    s = input_ids.shape[1]

    cos, sin = build_batch_2d_rope(s, c["HD"], image_infos=None)
    mask_add = to_additive(build_attention_mask(s, image_slices=None, bsz=BATCH), dtype=torch.float32)

    golden = []
    with torch.no_grad():
        h = torch.nn.functional.embedding(input_ids, wte_w.float())
        golden.append(h.clone())
        for i in range(num_layers):
            sd = load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.")
            ref_layer = RefLayer(
                hidden_size=c["H"],
                num_attention_heads=c["HEADS"],
                num_key_value_heads=c["KV"],
                attention_head_dim=c["HD"],
                num_experts=c["E"],
                moe_topk=c["K"],
                moe_intermediate_size=c["MOE_INTER"],
                num_shared_expert=c["NUM_SHARED"],
                use_mixed_mlp_moe=c["MIXED"],
                norm_topk_prob=c["NORM_TOPK"],
                use_qk_norm=c["QKN"],
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
        ref_final = ref_lnf(golden[num_layers])
    return golden, ref_final


def _teacher_forced_layers(device, c: dict, golden: list, num_layers: int, seq_len: int, weight_dtype=ttnn.bfloat16):
    results = []
    cos_tt = sin_tt = None
    for i in range(num_layers):
        layer_sd = {
            f"model.layers.{i}.{k}": v
            for k, v in load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.").items()
        }
        tt_layer = HunyuanTtDecoderLayer(
            device,
            layer_sd,
            layer_num=i,
            hidden_size=c["H"],
            num_heads=c["HEADS"],
            num_kv_heads=c["KV"],
            head_dim=c["HD"],
            num_experts=c["E"],
            moe_topk=c["K"],
            use_qk_norm=c["QKN"],
            use_mixed_mlp_moe=c["MIXED"],
            norm_topk_prob=c["NORM_TOPK"],
            rms_norm_eps=c["EPS"],
            stream_experts=True,
            weight_dtype=weight_dtype,
        )
        if cos_tt is None:
            cos_tt, sin_tt = tt_layer.self_attn.rope.prepare_cos_sin(seq_len, image_infos=None)
        x_tt = _to_tt(device, golden[i])
        out_tt = tt_layer(x_tt, seq_len=seq_len, image_infos=None, attention_mask=None, cos_sin=(cos_tt, sin_tt))
        out = ttnn.to_torch(out_tt)[..., : c["H"]]
        p, d = pcc_metrics(golden[i + 1], out, PCC_PER_LAYER)
        results.append((i, p, d))
        out_tt.deallocate(True)
        x_tt.deallocate(True)
        del tt_layer
        gc.collect()
    if cos_tt is not None:
        cos_tt.deallocate(True)
        sin_tt.deallocate(True)
    return results


def _chained_final(device, c: dict, golden: list, ref_final: torch.Tensor, num_layers: int, seq_len: int):
    lnf_w = load_tensors(resolve_base_model_dir(), ["model.ln_f.weight"])["model.ln_f.weight"]
    hidden = _to_tt(device, golden[0])
    cos_tt = sin_tt = None
    for i in range(num_layers):
        layer_sd = {
            f"model.layers.{i}.{k}": v
            for k, v in load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.").items()
        }
        tt_layer = HunyuanTtDecoderLayer(
            device,
            layer_sd,
            layer_num=i,
            hidden_size=c["H"],
            num_heads=c["HEADS"],
            num_kv_heads=c["KV"],
            head_dim=c["HD"],
            num_experts=c["E"],
            moe_topk=c["K"],
            use_qk_norm=c["QKN"],
            use_mixed_mlp_moe=c["MIXED"],
            norm_topk_prob=c["NORM_TOPK"],
            rms_norm_eps=c["EPS"],
            stream_experts=True,
        )
        if cos_tt is None:
            cos_tt, sin_tt = tt_layer.self_attn.rope.prepare_cos_sin(seq_len, image_infos=None)
        nxt = tt_layer(hidden, seq_len=seq_len, image_infos=None, attention_mask=None, cos_sin=(cos_tt, sin_tt))
        hidden.deallocate(True)
        hidden = nxt
        del tt_layer
        gc.collect()

    ln_f = HunyuanTtRMSNorm(device, c["H"], {"model.ln_f.weight": lnf_w}, "model.ln_f", eps=c["EPS"])
    out_tt = ln_f(hidden)
    hidden.deallocate(True)
    final = ttnn.to_torch(out_tt)[..., : c["H"]]
    out_tt.deallocate(True)
    if cos_tt is not None:
        cos_tt.deallocate(True)
        sin_tt.deallocate(True)
    return pcc_metrics(ref_final, final, PCC_CHAINED)


def _teacher_forced_final(device, c: dict, golden: list, ref_final: torch.Tensor, num_layers: int, seq_len: int):
    lnf_w = load_tensors(resolve_base_model_dir(), ["model.ln_f.weight"])["model.ln_f.weight"]
    cos_tt = sin_tt = None
    last_out = None
    for i in range(num_layers):
        layer_sd = {
            f"model.layers.{i}.{k}": v
            for k, v in load_prefixed_state_dict(resolve_base_model_dir(), f"model.layers.{i}.").items()
        }
        tt_layer = HunyuanTtDecoderLayer(
            device,
            layer_sd,
            layer_num=i,
            hidden_size=c["H"],
            num_heads=c["HEADS"],
            num_kv_heads=c["KV"],
            head_dim=c["HD"],
            num_experts=c["E"],
            moe_topk=c["K"],
            use_qk_norm=c["QKN"],
            use_mixed_mlp_moe=c["MIXED"],
            norm_topk_prob=c["NORM_TOPK"],
            rms_norm_eps=c["EPS"],
            stream_experts=True,
        )
        if cos_tt is None:
            cos_tt, sin_tt = tt_layer.self_attn.rope.prepare_cos_sin(seq_len, image_infos=None)
        x_tt = _to_tt(device, golden[i])
        out_tt = tt_layer(x_tt, seq_len=seq_len, image_infos=None, attention_mask=None, cos_sin=(cos_tt, sin_tt))
        x_tt.deallocate(True)
        if last_out is not None:
            last_out.deallocate(True)
        last_out = out_tt
        del tt_layer
        gc.collect()

    ln_f = HunyuanTtRMSNorm(device, c["H"], {"model.ln_f.weight": lnf_w}, "model.ln_f", eps=c["EPS"])
    final_tt = ln_f(last_out)
    last_out.deallocate(True)
    final = ttnn.to_torch(final_tt)[..., : c["H"]]
    final_tt.deallocate(True)
    if cos_tt is not None:
        cos_tt.deallocate(True)
        sin_tt.deallocate(True)
    return pcc_metrics(ref_final, final, PCC_BLOCK)


def _teacher_forced_all_layers_run(device, seq_len: int, label: str):
    c = transformer_cfg()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, _ = _reference(c, input_ids, NUM_LAYERS_FULL)
    results = _teacher_forced_layers(device, c, golden, NUM_LAYERS_FULL, seq_len)
    thr = per_layer_pcc(seq_len)
    failing = [(i, p) for i, p, _ in results if p < thr]
    phase = "decode" if seq_len == 1 else "prefill"
    print(
        f"teacher-forced all layers {phase} [{label}] S={seq_len} L={NUM_LAYERS_FULL}: "
        f"worst={min(r[1] for r in results):.6f} thr={thr}"
    )
    assert not failing, f"layers below {thr}: " + ", ".join(f"L{i}={p:.6f}" for i, p in failing)
    return results


def _teacher_forced_final_run(device, seq_len: int, label: str):
    c = transformer_cfg()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, ref_final = _reference(c, input_ids, NUM_LAYERS_FULL)
    p, d = _teacher_forced_final(device, c, golden, ref_final, NUM_LAYERS_FULL, seq_len)
    phase = "decode" if seq_len == 1 else "prefill"
    print(
        f"teacher-forced final {phase} [{label}] S={seq_len} L={NUM_LAYERS_FULL}: "
        f"PCC={p:.8f}  max|diff|={d:.6f}  thr={PCC_BLOCK}"
    )
    assert p >= PCC_BLOCK
    return p, d


# ---------------------------------------------------------------------------
# Per-layer teacher forced (test_model_teacher_forced.py) — fast smoke
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch,seq_len,label", TF_ISL_FAST)
def test_teacher_forced_per_layer_smoke(device, batch, seq_len, label):
    assert batch == 1
    c = transformer_cfg()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, _ = _reference(c, input_ids, NUM_LAYERS_FAST)
    results = _teacher_forced_layers(device, c, golden, NUM_LAYERS_FAST, seq_len)
    thr = per_layer_pcc(seq_len)
    failing = [(i, p) for i, p, _ in results if p < thr]
    print(
        f"teacher-forced smoke [{label}] S={seq_len} layers={NUM_LAYERS_FAST}: "
        f"worst={min(r[1] for r in results):.6f} thr={thr}"
    )
    assert not failing, f"layers below {thr}: " + ", ".join(f"L{i}={p:.6f}" for i, p in failing)


@pytest.mark.slow
@pytest.mark.parametrize("batch,seq_len,label", TF_ISL_SLOW)
def test_teacher_forced_per_layer_large_isl(device, batch, seq_len, label):
    assert batch == 1
    c = transformer_cfg()
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, _ = _reference(c, input_ids, NUM_LAYERS_FAST)
    results = _teacher_forced_layers(device, c, golden, NUM_LAYERS_FAST, seq_len)
    assert all(p >= per_layer_pcc(seq_len) for _, p, _ in results)


@pytest.mark.slow
@pytest.mark.parametrize("batch,seq_len,label", [(1, 32, "one tile S=32")])
def test_teacher_forced_all_layers(device, batch, seq_len, label):
    assert batch == 1
    _teacher_forced_all_layers_run(device, seq_len, label)


@pytest.mark.slow
@pytest.mark.parametrize("seq_len,label", TF_ALL_LAYERS_PRODUCTION)
def test_teacher_forced_all_layers_production(device, seq_len, label):
    """32-layer per-layer teacher-forced PCC at decode (S=1) and production prefill (S=4160)."""
    _teacher_forced_all_layers_run(device, seq_len, label)


@pytest.mark.slow
@pytest.mark.parametrize("seq_len,label", TF_ALL_LAYERS_PRODUCTION)
def test_teacher_forced_final_production(device, seq_len, label):
    """32-layer teacher-forced final hidden PCC at decode (S=1) and production prefill (S=4160)."""
    _teacher_forced_final_run(device, seq_len, label)


# ---------------------------------------------------------------------------
# Chained final output (test_model_teacher_forced.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_teacher_forced_chained_final(device):
    c = transformer_cfg()
    seq_len = 32
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, ref_final = _reference(c, input_ids, NUM_LAYERS_FULL)
    p, d = _chained_final(device, c, golden, ref_final, NUM_LAYERS_FULL, seq_len)
    print(f"chained final {NUM_LAYERS_FULL}L: PCC={p:.8f}  max|diff|={d:.6f}  thr={PCC_CHAINED}")
    assert p >= PCC_CHAINED


# ---------------------------------------------------------------------------
# Teacher-forced final output (test_model_teacher_forced_final.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_teacher_forced_final_output(device):
    c = transformer_cfg()
    seq_len = 32
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, ref_final = _reference(c, input_ids, NUM_LAYERS_FULL)
    p, d = _teacher_forced_final(device, c, golden, ref_final, NUM_LAYERS_FULL, seq_len)
    print(f"teacher-forced final {NUM_LAYERS_FULL}L: PCC={p:.8f}  max|diff|={d:.6f}  thr={PCC_BLOCK}")
    assert p >= PCC_BLOCK


# ---------------------------------------------------------------------------
# bf8 mixed-precision audit (test_model_teacher_forced.py)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_bf8_mixed_precision_audit(device):
    c = transformer_cfg()
    seq_len = 32
    torch.manual_seed(0)
    input_ids = torch.randint(0, 130000, (BATCH, seq_len), dtype=torch.long)
    golden, _ = _reference(c, input_ids, NUM_LAYERS_FULL)
    res16 = _teacher_forced_layers(device, c, golden, NUM_LAYERS_FULL, seq_len, weight_dtype=ttnn.bfloat16)
    res8 = _teacher_forced_layers(device, c, golden, NUM_LAYERS_FULL, seq_len, weight_dtype=ttnn.bfloat8_b)
    p16 = {i: p for i, p, _ in res16}
    p8 = {i: p for i, p, _ in res8}
    need_bf16 = sorted(i for i in range(NUM_LAYERS_FULL) if p8[i] < PCC_PER_LAYER)
    bf16_fail = [(i, p16[i]) for i in range(NUM_LAYERS_FULL) if p16[i] < PCC_PER_LAYER]
    print(f"bf8 audit: recommend bf16_layers={need_bf16} ({len(need_bf16)}/{NUM_LAYERS_FULL} layers)")
    assert not bf16_fail
