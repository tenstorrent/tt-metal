# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# On-device AR generate at full HF dimensions (vocab × hidden, real backbone + lm_head).
# Complements unit_host mocks in test_generate.py (tiny V=64) and the recaption suite.
#
# Smoke (default 2L, short max_new):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_generate_device.py -m "not slow" -v -s
#
# Full depth (32L):
#   HY_NUM_LAYERS=32 python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_generate_device.py -k production -v -s --timeout=10800

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import ttnn

ROOT = Path(__file__).resolve().parents[5]
PCC_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PCC_DIR) not in sys.path:
    sys.path.insert(0, str(PCC_DIR))

from models.experimental.hunyuan_image_3_0.ref.generate import generate_text
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tt.generate import make_backbone_logits_fn
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte
from recaption_helpers import (
    NUM_LAYERS,
    attention_mask_fn,
    build_tt_backbone_lm,
    greedy_config,
    has_weights,
    prepare_recaption_bundle,
    ref_logits_fn,
    stage_params,
)
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h

NUM_LAYERS_FULL = int(os.environ.get("HY_NUM_LAYERS", "32"))
MAX_NEW = int(os.environ.get("HY_MAX_NEW_TOKENS", "8"))


@pytest.fixture(scope="function")
def device():
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    ttnn.close_device(dev)


def _run_device_vs_host_generate(device):
    """Greedy AR on real instruct weights: full vocab logits, token-stream parity."""
    tok, _proc, bundle = prepare_recaption_bundle()
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    lm_w = load_tensors(INSTRUCT_MODEL_DIR, ["lm_head.weight"])["lm_head.weight"]
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    cfg = greedy_config()
    # Allow override of MAX_NEW without rebinding the recaption_helpers default.
    from dataclasses import replace

    cfg = replace(cfg, do_sample=False, max_new_tokens=MAX_NEW)
    params = stage_params(tok)

    # Full vocabulary size from checkpoint (≈133120), not the V=64 host mock.
    assert lm_w.shape[0] >= 100_000, f"expected full vocab lm_head, got V={lm_w.shape[0]}"
    assert c["H"] == 4096

    ref_out = generate_text(
        ref_logits_fn(tok, bundle, c, ln_f_w),
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )

    backbone, lm_head = build_tt_backbone_lm(device, c, wte, ln_f_w)
    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None
    forward_logits_fn = make_backbone_logits_fn(
        backbone,
        lm_head,
        device,
        attention_mask_fn=attention_mask_fn(device, bundle),
        image_infos=image_infos,
    )
    tt_out = generate_text(
        forward_logits_fn,
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )

    assert tt_out["new_tokens"] == ref_out["new_tokens"], (
        f"AR token mismatch (layers={NUM_LAYERS}, V={lm_w.shape[0]}, "
        f"prefix_S={bundle.seq_len}): tt={tt_out['new_tokens']} ref={ref_out['new_tokens']}"
    )
    assert torch.equal(tt_out["sequences"], ref_out["sequences"])
    # Exercise full-vocab wte+lm_head once more (shape sanity).
    wte_tt = HunyuanTtWte(device, wte)
    embeds = wte_tt.embedding_torch(bundle.input_ids[:1, :1])
    assert embeds.shape == (1, 1, c["H"])
    assert F.embedding(bundle.input_ids[:1, :1].long(), wte.float()).shape == embeds.shape
    return bundle.seq_len, lm_w.shape[0], len(tt_out["new_tokens"][0])


@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
def test_generate_device_full_vocab_greedy(device):
    """Device generate_text vs host at full vocab/H (default HY_NUM_LAYERS)."""
    prefix_s, vocab, n_new = _run_device_vs_host_generate(device)
    print(f"generate device full-vocab: prefix_S={prefix_s} V={vocab} H=4096 new={n_new} layers={NUM_LAYERS}")


@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
@pytest.mark.slow
def test_generate_device_production_32l_greedy(device):
    """Full-depth (32L) on-device AR generate at full vocab — production scale gate."""
    if NUM_LAYERS != NUM_LAYERS_FULL:
        pytest.skip(f"requires HY_NUM_LAYERS={NUM_LAYERS_FULL}, got {NUM_LAYERS}")
    prefix_s, vocab, n_new = _run_device_vs_host_generate(device)
    print(f"generate device production 32L: prefix_S={prefix_s} V={vocab} H=4096 " f"new={n_new} layers={NUM_LAYERS}")
