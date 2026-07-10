# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify traced KV decode replay matches eager KV greedy tokens.
#
# Run (device required):
#   HY_NUM_LAYERS=2 HY_MAX_NEW_TOKENS=6 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_recaption_trace.py -v -s --timeout=3600

from __future__ import annotations

import json

import pytest
import torch.nn.functional as F
import ttnn

from models.experimental.hunyuan_image_3_0.ref.generate import generate_text
from models.experimental.hunyuan_image_3_0.ref.image_processor import HunyuanImage3ImageProcessor
from models.experimental.hunyuan_image_3_0.ref.recaption import (
    build_recaption_stage_params,
    default_recaption_sampling_config,
)
from models.experimental.hunyuan_image_3_0.ref.tokenizer import HunyuanTokenizer, prepare_recaption_inputs
from models.experimental.hunyuan_image_3_0.ref.tokenizer.gen_image_inputs import enrich_bundle_attention
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h
from models.experimental.hunyuan_image_3_0.tests.pcc.test_recaption_on_device import MAX_NEW, NUM_LAYERS, PROMPT
from models.experimental.hunyuan_image_3_0.tt.generate import make_recaption_logits_fn
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte

TRACE_REGION = 32 * 1024 * 1024


@pytest.fixture(scope="module")
def device_trace():
    """Device with enlarged trace region for decode trace capture."""
    import os

    prev = os.environ.pop("TT_DIT_CACHE_DIR", None)
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=TRACE_REGION)
    yield dev
    ttnn.close_device(dev)
    if prev is not None:
        os.environ["TT_DIT_CACHE_DIR"] = prev


def _greedy_kv_sequences(device, *, use_trace: bool):
    from dataclasses import replace

    tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")
    proc = HunyuanImage3ImageProcessor(json.load(open(INSTRUCT_MODEL_DIR / "config.json")))
    bundle = enrich_bundle_attention(
        prepare_recaption_inputs(tok, PROMPT, bot_task="recaption", sequence_template="instruct"),
        proc,
    )
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    cfg = replace(default_recaption_sampling_config(), do_sample=False, max_new_tokens=MAX_NEW)
    params = build_recaption_stage_params(tok, "recaption", image_size=1024, sequence_template="instruct")

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
    wte_tt = HunyuanTtWte(device, wte)
    prefix_embeds = F.embedding(bundle.input_ids.long(), wte.float())
    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]

    forward_logits_fn = make_recaption_logits_fn(
        backbone,
        lm_head,
        device,
        wte_tt=wte_tt,
        prefix_embeds=prefix_embeds,
        image_infos=image_infos,
        attn_slices=attn_slices,
        use_kv_cache=True,
        max_new_tokens=MAX_NEW,
        sp_factor=1,
        use_trace=use_trace,
    )
    out = generate_text(
        forward_logits_fn,
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )
    tracer = forward_logits_fn.tracer
    replay_steps = tracer.replay_steps if tracer is not None else None
    if tracer is not None:
        tracer.release()
    return out["sequences"], replay_steps


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
def test_recaption_trace_matches_eager_kv_greedy_tokens(device_trace):
    """Traced decode replay must match eager KV greedy token stream."""
    eager_seq, _ = _greedy_kv_sequences(device_trace, use_trace=False)
    trace_seq, replay_steps = _greedy_kv_sequences(device_trace, use_trace=True)
    assert replay_steps == MAX_NEW - 1
    assert trace_seq.tolist() == eager_seq.tolist()
