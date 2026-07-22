# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bespoke TP generation (``generate_tp``) functional check on the full model.

Runs the complete 27B TP model on a real prompt through the stateful generate
loop: ``prefill_tp`` (fills the concat KV cache + GDN recurrent/conv state) then
incremental ``decode_tp`` steps (non-traced). Asserts the continuation is correct
(first token 'Paris') and non-degenerate — i.e. KV-cache + GDN state continuation is right.

This is the *oracle* path, not the served path. The production/served path (chunk-outer
traced prefill + paged traced decode, what vLLM and demo/text_demo.py run) is validated by
test_model_tp.py (which proves the contract/paged/traced path matches this oracle
per-step). This test anchors that oracle to a real expected answer on the full model.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_generate_tp.py -v -s
"""
import os

from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_generate_tp_stateful(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", model_path())
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=256)

    from transformers import AutoTokenizer

    # model.args.CKPT_DIR is the resolved local snapshot dir (downloaded from the hub id).
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()

    new_ids = model.generate_tp(ids, max_new_tokens=8)
    text = tok.decode(ids + new_ids)
    logger.info(f"GENERATED (bespoke generate_tp): {text!r}")

    assert len(set(new_ids)) > 1, f"degenerate: {new_ids}"
    # First generated token should be ' Paris' (matches the stateless re-prefill run).
    first = tok.decode([new_ids[0]]).strip()
    logger.info(f"first generated token: {first!r}")
    assert first == "Paris", f"expected 'Paris', got {first!r} (stateful decode continuation may be wrong)"
    logger.info("PASSED: bespoke generate_tp produces the correct continuation")
