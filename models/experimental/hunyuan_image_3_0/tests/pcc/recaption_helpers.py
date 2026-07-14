# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for recaption PCC tests.

from __future__ import annotations

import json
import os

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
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h

PROMPT = "a cat on a mat"
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
MAX_NEW = int(os.environ.get("HY_MAX_NEW_TOKENS", "6"))
TRACE_REGION = 32 * 1024 * 1024


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


def ref_logits_fn(tok, bundle, c, ln_f_w):
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"].float()
    lm_w = load_tensors(INSTRUCT_MODEL_DIR, ["lm_head.weight"])["lm_head.weight"].float()
    layers = [h.ref_layer(c, i) for i in range(NUM_LAYERS)]
    ln_f = HunyuanRMSNorm(c["H"], eps=c["EPS"])
    ln_f.weight.data = ln_f_w.float()

    def forward_logits_fn(ids):
        s = ids.shape[1]
        hidden = F.embedding(ids.long(), wte)
        mask = to_additive(build_attention_mask(s, bundle.full_attn_slices or [[]], bsz=1)).reshape(1, 1, s, s)
        image_infos = bundle.rope_image_info
        cos, sin = build_batch_2d_rope(s, c["HEAD_DIM"], image_infos=image_infos, device=hidden.device)
        for layer in layers:
            hidden = layer(hidden, attention_mask=mask, custom_pos_emb=(cos, sin))
        hidden = ln_f(hidden)
        return lm_head_logits(hidden, lm_w)[:, -1, :]

    return forward_logits_fn


def build_tt_backbone_lm(device, c, wte, ln_f_w):
    from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
    from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel

    layer_loader = lambda i: {f"model.layers.{i}.{k}": v for k, v in h.load_prefix(f"model.layers.{i}").items()}
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
        stream_experts=False,
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
