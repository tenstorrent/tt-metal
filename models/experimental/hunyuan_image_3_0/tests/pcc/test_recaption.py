# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated recaption PCC tests:
#   - On-device greedy AR vs host ref (backbone + lm_head)
#   - KV trace replay vs eager KV
#   - 2CQ vs 1CQ token stream + host ref
#
# Run (fast, requires instruct checkpoint):
#   python_env/bin/python -m pytest \
#     models/experimental/hunyuan_image_3_0/tests/pcc/test_recaption.py -m "not slow" -v

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
from models.experimental.hunyuan_image_3_0.ref.recaption import decode_cot_text
from models.experimental.hunyuan_image_3_0.ref.weights import INSTRUCT_MODEL_DIR, load_tensors
from models.experimental.hunyuan_image_3_0.tt.ar_dual_cq import (
    ArDualCQCoordinator,
    device_num_command_queues,
    open_recaption_mesh,
)
from models.experimental.hunyuan_image_3_0.tt.generate import make_backbone_logits_fn, make_recaption_logits_fn
from models.experimental.hunyuan_image_3_0.tt.recaption import run_recaption_on_device
from models.experimental.hunyuan_image_3_0.tt.wte import HunyuanTtWte
from recaption_helpers import (
    MAX_NEW,
    NUM_LAYERS,
    TRACE_REGION,
    attention_mask_fn,
    build_tt_backbone_lm,
    greedy_config,
    has_weights,
    prepare_recaption_bundle,
    ref_logits_fn,
    stage_params,
)
from models.experimental.hunyuan_image_3_0.tests.pcc import i2i_helpers as h

NUM_LAYERS_PRODUCTION = int(os.environ.get("HY_NUM_LAYERS", "32"))


def _close_device(dev) -> None:
    """Best-effort close; never raise during fixture teardown."""
    try:
        if getattr(dev, "__class__", type(None)).__name__ == "MeshDevice" and hasattr(ttnn, "close_mesh_device"):
            # Prefer mesh close when we opened via open_mesh_device / open_recaption_mesh.
            try:
                ttnn.synchronize_device(dev)
            except Exception:
                pass
            ttnn.close_mesh_device(dev)
        else:
            ttnn.close_device(dev)
    except Exception as e:  # noqa: BLE001 — teardown must not cascade fatals
        print(f"[test_recaption] close_device ignored: {e}", flush=True)


@pytest.fixture(scope="function")
def device():
    # Function-scoped: chip 0 cannot be shared across different open params
    # (trace region / num CQs). Overlapping opens caused context/dispatch fatals.
    dev = ttnn.open_device(device_id=0, l1_small_size=32768)
    yield dev
    _close_device(dev)


@pytest.fixture(scope="function")
def device_trace():
    prev = os.environ.pop("TT_DIT_CACHE_DIR", None)
    dev = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=TRACE_REGION)
    yield dev
    _close_device(dev)
    if prev is not None:
        os.environ["TT_DIT_CACHE_DIR"] = prev


@pytest.fixture(scope="function")
def device_2cq():
    # ``ttnn.open_device(..., num_command_queues=2)`` returns MeshDevice(1x1) with
    # only 1 CQ reported. Use the same ``open_mesh_device`` + stash path as demos.
    mesh = open_recaption_mesh(ttnn.MeshShape(1, 1), l1_small_size=32768, enable_2cq=True)
    assert (
        device_num_command_queues(mesh) >= 2
    ), f"expected >=2 CQs after open_recaption_mesh, got {device_num_command_queues(mesh)}"
    yield mesh
    _close_device(mesh)


def _greedy_kv_sequences(device, *, use_trace: bool):
    tok, proc, bundle = prepare_recaption_bundle()
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    cfg = greedy_config()
    params = stage_params(tok)
    backbone, lm_head = build_tt_backbone_lm(device, c, wte, ln_f_w)
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


def _greedy_tt_sequences(device, *, dual_cq: ArDualCQCoordinator | None):
    tok, proc, bundle = prepare_recaption_bundle()
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    cfg = greedy_config()
    params = stage_params(tok)
    backbone, lm_head = build_tt_backbone_lm(device, c, wte, ln_f_w)
    image_infos = [bundle.rope_image_info[0]] if bundle.rope_image_info else None

    forward_logits_fn = make_backbone_logits_fn(
        backbone,
        lm_head,
        device,
        attention_mask_fn=attention_mask_fn(device, bundle),
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


def _greedy_token_parity(device, *, use_recaption_fn: bool = False):
    """Device greedy AR vs host ref; optional run_recaption_on_device wrapper check."""
    tok, proc, bundle = prepare_recaption_bundle()
    wte = load_tensors(INSTRUCT_MODEL_DIR, ["model.wte.weight"])["model.wte.weight"]
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    cfg = greedy_config()
    params = stage_params(tok)

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

    assert tt_out["new_tokens"] == ref_out["new_tokens"]
    assert torch.equal(tt_out["sequences"], ref_out["sequences"])

    if use_recaption_fn:
        recap_result = run_recaption_on_device(
            backbone,
            lm_head,
            device,
            bundle,
            tok,
            "recaption",
            proc,
            wte_weight=wte,
            image_size=1024,
            config=cfg,
        )
        assert recap_result.new_tokens == ref_out["new_tokens"]
        cot = decode_cot_text(tok, recap_result.sequences, bundle.input_ids.shape[1], "recaption")
        assert cot[0]


# ---------------------------------------------------------------------------
# On-device recaption (test_recaption_on_device.py)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
def test_recaption_on_device_greedy_tokens(device):
    _greedy_token_parity(device, use_recaption_fn=True)


@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
@pytest.mark.slow
def test_recaption_production_greedy_tokens(device):
    """32L instruct recaption greedy token parity vs host ref (production slow CI)."""
    if NUM_LAYERS != NUM_LAYERS_PRODUCTION:
        pytest.skip(f"requires HY_NUM_LAYERS={NUM_LAYERS_PRODUCTION}, got {NUM_LAYERS}")
    _greedy_token_parity(device, use_recaption_fn=False)


# ---------------------------------------------------------------------------
# KV trace (test_recaption_trace.py)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
@pytest.mark.slow
def test_recaption_trace_matches_eager_kv_greedy_tokens(device_trace):
    eager_seq, _ = _greedy_kv_sequences(device_trace, use_trace=False)
    trace_seq, replay_steps = _greedy_kv_sequences(device_trace, use_trace=True)
    assert replay_steps == MAX_NEW - 1
    assert trace_seq.tolist() == eager_seq.tolist()


# ---------------------------------------------------------------------------
# 2CQ AR (test_recaption_2cq.py)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
@pytest.mark.slow
def test_recaption_2cq_matches_1cq_greedy_tokens(device_2cq):
    ref_seq, _ = _greedy_tt_sequences(device_2cq, dual_cq=None)
    dual_cq = ArDualCQCoordinator(device_2cq)
    tt_seq, steps = _greedy_tt_sequences(device_2cq, dual_cq=dual_cq)
    assert steps == MAX_NEW
    assert tt_seq.tolist() == ref_seq.tolist()


@pytest.mark.skipif(not has_weights(), reason="Hunyuan instruct checkpoint not available")
@pytest.mark.slow
def test_recaption_2cq_matches_host_ref(device_2cq):
    tok, proc, bundle = prepare_recaption_bundle()
    c = h.model_cfg()
    ln_f_w = h.load_tensor("model.ln_f.weight")
    cfg = greedy_config()
    params = stage_params(tok)
    ref_out = generate_text(
        ref_logits_fn(tok, bundle, c, ln_f_w),
        bundle.input_ids,
        config=cfg,
        stage_transitions=params.stage_transitions or None,
        final_stop_tokens=params.final_stop_tokens,
    )
    dual_cq = ArDualCQCoordinator(device_2cq)
    tt_seq, steps = _greedy_tt_sequences(device_2cq, dual_cq=dual_cq)
    assert steps == MAX_NEW
    assert tt_seq.tolist() == ref_out["sequences"].tolist()
