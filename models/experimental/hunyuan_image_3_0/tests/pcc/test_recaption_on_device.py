# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device recaption integration: greedy AR tokens vs host ref stack (few layers).
#
# Run:
#   HY_NUM_LAYERS=2 HY_MAX_NEW_TOKENS=6 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_recaption_on_device.py -v -s --timeout=3600

from __future__ import annotations

import os

import pytest
import torch
import ttnn

from models.experimental.hunyuan_image_3_0.ref.generate import generate_text
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.lm_head import lm_head_logits
from models.experimental.hunyuan_image_3_0.ref.recaption import build_recaption_stage_params, decode_cot_text
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tt.generate import make_backbone_logits_fn
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.recaption import run_recaption_on_device

PROMPT = "a cat on a mat"
NUM_LAYERS = int(os.environ.get("HY_NUM_LAYERS", "2"))
MAX_NEW = int(os.environ.get("HY_MAX_NEW_TOKENS", "6"))
USE_KV = os.environ.get("HY_RECAPTION_KV", "1") != "0"
PCC_THR = 0.95 if NUM_LAYERS <= 2 else 0.85


def _ref_logits_fn(tok, bundle, c, ln_f_w):
    """Host ref: N decoder layers + ln_f + lm_head (full-seq forward each step)."""
    import torch.nn.functional as F

    from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
    from models.experimental.hunyuan_image_3_0.ref.attention.rms_norm import HunyuanRMSNorm
    from hunyuan_image_3.modeling_hunyuan_image_3 import build_batch_2d_rope

    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"].float()
    lm_w = load_tensors(INSTRUCT_MODEL_DIR, ["lm_head.weight"])["lm_head.weight"].float()
    layers = [h.ref_layer(c, i) for i in range(NUM_LAYERS)]
    ln_f = HunyuanRMSNorm(c["H"], eps=c["EPS"])
    ln_f.weight.data = ln_f_w.float()

    def forward_logits_fn(ids):
        S = ids.shape[1]
        hidden = F.embedding(ids.long(), wte)
        mask = to_additive(build_attention_mask(S, bundle.full_attn_slices or [[]], bsz=1)).reshape(1, 1, S, S)
        image_infos = bundle.rope_image_info
        cos, sin = build_batch_2d_rope(
            image_infos=image_infos,
            seq_len=S,
            n_elem=c["HEAD_DIM"],
            device=hidden.device,
        )
        custom_pos_emb = (cos, sin)
        for layer in layers:
            hidden = layer(hidden, attention_mask=mask, custom_pos_emb=custom_pos_emb)
        hidden = ln_f(hidden)
        return lm_head_logits(hidden, lm_w)[:, -1, :]

    return forward_logits_fn


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
def test_recaption_on_device_greedy_tokens(device):
    import json

    from dataclasses import replace

    from models.experimental.hunyuan_image_3_0.ref.recaption import default_recaption_sampling_config

    tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")
    proc = HunyuanImage3ImageProcessor(json.load(open(INSTRUCT_MODEL_DIR / "config.json")))
    bundle = prepare_recaption_inputs(tok, PROMPT, bot_task="recaption", sequence_template="instruct")
    from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import enrich_bundle_attention

    bundle = enrich_bundle_attention(bundle, proc)
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")

    cfg = replace(default_recaption_sampling_config(), do_sample=False, max_new_tokens=MAX_NEW)
    params = build_recaption_stage_params(tok, "recaption", image_size=1024, sequence_template="instruct")

    ref_out = generate_text(
        _ref_logits_fn(tok, bundle, c, ln_f_w),
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )

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

    from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive

    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]

    def attention_mask_fn(S: int):
        mask_bool = build_attention_mask(S, attn_slices, bsz=1)
        mask_add = to_additive(mask_bool, dtype=torch.bfloat16).reshape(1, 1, S, S)
        return ttnn.from_torch(
            mask_add,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    forward_logits_fn = make_backbone_logits_fn(
        backbone,
        lm_head,
        device,
        attention_mask_fn=attention_mask_fn,
        image_infos=image_infos,
    )

    tt_out = generate_text(
        forward_logits_fn,
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )

    input_len = bundle.input_ids.shape[1]
    assert (
        tt_out["new_tokens"] == ref_out["new_tokens"]
    ), f"token mismatch kv={USE_KV}: tt={tt_out['new_tokens']} ref={ref_out['new_tokens']}"
    assert torch.equal(tt_out["sequences"], ref_out["sequences"])

    recap_result = run_recaption_on_device(
        backbone,
        lm_head,
        device,
        bundle,
        tok,
        "recaption",
        proc,
        wte,
        image_size=1024,
        config=cfg,
    )
    assert recap_result.new_tokens == ref_out["new_tokens"]
    cot = decode_cot_text(tok, recap_result.sequences, input_len, "recaption")
    assert cot[0]
