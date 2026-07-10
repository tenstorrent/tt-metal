# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Verify 2CQ AR recaption produces the same tokens as the default 1CQ D2H path.
#
# Run (device required):
#   HY_NUM_LAYERS=2 HY_MAX_NEW_TOKENS=6 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_recaption_2cq.py -v -s --timeout=3600

from __future__ import annotations

import json

import pytest
import ttnn

from models.experimental.hunyuan_image_3_0.ref.attention.mask import build_attention_mask, to_additive
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
from models.experimental.hunyuan_image_3_0.tests.pcc.test_recaption_on_device import (
    MAX_NEW,
    NUM_LAYERS,
    PROMPT,
    _ref_logits_fn,
)
from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import ArDualCQCoordinator, device_num_command_queues
from models.experimental.hunyuan_image_3_0.tt.generate import make_backbone_logits_fn
from models.experimental.hunyuan_image_3_0.tt.lm_head import HunyuanTtLMHead
from models.experimental.hunyuan_image_3_0.tt.model import HunyuanTtModel


@pytest.fixture(scope="module")
def device_2cq():
    """Device with two command queues for 2CQ AR tests."""
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, num_command_queues=2)
    assert device_num_command_queues(dev) >= 2, "Expected num_command_queues>=2"
    yield dev
    ttnn.close_device(dev)


def _greedy_tt_sequences(device, *, dual_cq: ArDualCQCoordinator | None):
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
    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    attn_slices = bundle.full_attn_slices or [[]]

    def attention_mask_fn(S: int):
        import torch

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
        dual_cq=dual_cq,
    )
    out = generate_text(
        forward_logits_fn,
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )
    return out["sequences"], dual_cq.steps if dual_cq is not None else None


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
def test_recaption_2cq_matches_1cq_greedy_tokens(device_2cq):
    """2CQ logits D2H must match 1CQ blocking read (same greedy token stream)."""
    ref_seq, _ = _greedy_tt_sequences(device_2cq, dual_cq=None)
    dual_cq = ArDualCQCoordinator(device_2cq)
    tt_seq, steps = _greedy_tt_sequences(device_2cq, dual_cq=dual_cq)
    assert steps == MAX_NEW
    assert tt_seq.tolist() == ref_seq.tolist()


@pytest.mark.skipif(not h.has_weights(), reason="Hunyuan checkpoint not available")
def test_recaption_2cq_matches_host_ref(device_2cq):
    """2CQ on-device path matches host ref logits loop."""
    from dataclasses import replace

    tok = HunyuanTokenizer.from_model_dir(INSTRUCT_MODEL_DIR, sequence_template="instruct")
    proc = HunyuanImage3ImageProcessor(json.load(open(INSTRUCT_MODEL_DIR / "config.json")))
    bundle = enrich_bundle_attention(
        prepare_recaption_inputs(tok, PROMPT, bot_task="recaption", sequence_template="instruct"),
        proc,
    )
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
    dual_cq = ArDualCQCoordinator(device_2cq)
    tt_seq, steps = _greedy_tt_sequences(device_2cq, dual_cq=dual_cq)
    assert steps == MAX_NEW
    assert tt_seq.tolist() == ref_out["sequences"].tolist()
