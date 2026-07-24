# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for recaption PCC tests.

from __future__ import annotations

import gc
import json
import os
import time

import torch
import torch.nn.functional as F
import ttnn

from models.experimental.hunyuan_image_3_0.ref.attention.rope_2d import build_batch_2d_rope
from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.lm_head import lm_head_logits
from models.experimental.hunyuan_image_3_0.ref.recaption import (
    build_recaption_stage_params,
    default_recaption_sampling_config,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import enrich_bundle_attention
from models.experimental.hunyuan_image_3_0.ref.transformer_layer import HunyuanImage3DecoderLayer as RefLayer
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h

PROMPT = "a cat on a mat"
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
MAX_NEW = int(os.environ.get("HY_MAX_NEW_TOKENS", "6"))
TRACE_REGION = 32 * 1024 * 1024

# Resident bf16 MoE layers for deep AR refs. fp32×32 ≈ 320GB OOMs; bf16×32 ≈ 160GB fits
# and avoids rebuilding/reloading the stack on every generate_text decode step.
_REF_LAYERS_BF16: list = []


def has_weights() -> bool:
    return h.has_weights()


def greedy_config():
    from dataclasses import replace

    return replace(default_recaption_sampling_config(), do_sample=False, max_new_tokens=MAX_NEW)


def prepare_recaption_bundle():
    tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")
    proc = HunyuanImage3ImageProcessor(json.load(open(INSTRUCT_MODEL_DIR / "config.json")))
    bundle = enrich_bundle_attention(
        prepare_recaption_inputs(tok, PROMPT, bot_task="recaption", sequence_template="instruct"),
        proc,
    )
    return tok, proc, bundle


def clear_ref_layers_bf16() -> None:
    _REF_LAYERS_BF16.clear()
    gc.collect()


def _make_ref_layer_bf16(c: dict, i: int) -> RefLayer:
    """Load one MoE layer and cast weights to bf16 (activations stay fp32 via autocast-free fwd)."""
    sd = h.load_prefix(f"model.layers.{i}")
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
    layer.to(dtype=torch.bfloat16)
    layer.eval()
    return layer


def _ensure_ref_layers_bf16(c: dict) -> list:
    if len(_REF_LAYERS_BF16) == NUM_LAYERS:
        return _REF_LAYERS_BF16
    clear_ref_layers_bf16()
    print(
        f"[recaption/generate ref] loading {NUM_LAYERS} layers as resident bf16 "
        f"(~5GB/layer; once per process, then reused for every AR step)",
        flush=True,
    )
    for i in range(NUM_LAYERS):
        t0 = time.time()
        _REF_LAYERS_BF16.append(_make_ref_layer_bf16(c, i))
        gc.collect()
        print(f"[recaption/generate ref] loaded layer {i + 1}/{NUM_LAYERS} in {time.time() - t0:.1f}s", flush=True)
    return _REF_LAYERS_BF16


def ref_logits_fn(tok, bundle, c, ln_f_w):
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"].float()
    lm_w = load_tensors(INSTRUCT_MODEL_DIR, ["lm_head.weight"])["lm_head.weight"].float()
    ln_f = HunyuanRMSNorm(c["H"], eps=c["EPS"])
    ln_f.weight.data = ln_f_w.float()
    # Deep stacks: keep bf16 modules resident (not fp32 — that OOMs). Shallow: fp32 cache.
    deep = NUM_LAYERS > 8
    if deep:
        h._REF_LAYERS.clear()
        layers = _ensure_ref_layers_bf16(c)
    else:
        layers = [h.ref_layer(c, i) for i in range(NUM_LAYERS)]

    step = {"n": 0}

    @torch.no_grad()
    def forward_logits_fn(ids):
        s = ids.shape[1]
        step["n"] += 1
        # generate_text calls this once per new token; S grows by 1 each time (not a sweep).
        print(f"[recaption/generate ref] AR step {step['n']} S={s} layers={NUM_LAYERS}", flush=True)
        hidden = F.embedding(ids.long(), wte)
        if deep:
            # Weights are bf16; match activation dtype for matmuls.
            hidden = hidden.to(torch.bfloat16)
        mask = to_additive(build_attention_mask(s, bundle.full_attn_slices or [[]], bsz=1)).reshape(1, 1, s, s)
        if deep:
            mask = mask.to(torch.bfloat16)
        image_infos = bundle.rope_image_info
        cos, sin = build_batch_2d_rope(s, c["HEAD_DIM"], image_infos=image_infos, device=hidden.device)
        if deep:
            cos, sin = cos.to(torch.bfloat16), sin.to(torch.bfloat16)
        for layer in layers:
            hidden = layer(hidden, attention_mask=mask, custom_pos_emb=(cos, sin))
        hidden = ln_f(hidden.float() if deep else hidden)
        return lm_head_logits(hidden, lm_w)[:, -1, :]

    return forward_logits_fn


def build_tt_backbone_lm(device, c, wte, ln_f_w):
    from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
    from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
    # Deep stacks: stream experts so 32L does not pin all expert weights in DRAM.
    stream_experts = NUM_LAYERS > 8
    backbone = HunyuanTtModel(
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
        stream_experts=stream_experts,
        layer_loader=layer_loader,
        embed_state_dict={"model.wte.weight": wte},
        norm_state_dict={"model.ln_f.weight": ln_f_w},
        apply_final_norm=True,
        weight_dtype=ttnn.bfloat16,
        sp_factor=1,
    )
    lm_head = HunyuanTtLMHead(device, {"lm_head.weight": h.load_tensor("lm_head.weight")})
    return backbone, lm_head


def attention_mask_fn(device, bundle):
    attn_slices = bundle.full_attn_slices or [[]]

    def fn(s: int):
        mask_bool = build_attention_mask(s, attn_slices, bsz=1)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, s, s)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return fn


def stage_params(tok):
    return build_recaption_stage_params(tok, "recaption", image_size=1024, sequence_template="instruct")
