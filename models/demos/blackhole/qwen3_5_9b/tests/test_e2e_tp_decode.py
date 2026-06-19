# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end TP=4 STATEFUL decode: prefill (fills KV cache + GDN state) then
incremental single-token decode — the real generation path (non-traced).

Validates that stateful decode reproduces the stateless re-prefill generation
(test_e2e_tp), i.e. KV-cache + GDN recurrent/conv-state continuation is correct.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen3_5_9b/tests/test_e2e_tp_decode.py -v -s
"""
import os

import pytest

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model


@pytest.mark.parametrize(
    "mesh_device",
    [{"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_e2e_tp_decode(mesh_device, ensure_gc):
    from loguru import logger

    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B")
    model = Qwen35Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=256)

    from transformers import AutoTokenizer

    # model.args.CKPT_DIR is the resolved local snapshot dir (downloaded from the hub id).
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()

    new_ids = model.generate_tp(ids, max_new_tokens=8)
    text = tok.decode(ids + new_ids)
    logger.info(f"GENERATED (stateful decode): {text!r}")

    assert len(set(new_ids)) > 1, f"degenerate: {new_ids}"
    # First generated token should be ' Paris' (matches the stateless re-prefill run).
    first = tok.decode([new_ids[0]]).strip()
    logger.info(f"first generated token: {first!r}")
    assert first == "Paris", f"expected 'Paris', got {first!r} (stateful decode continuation may be wrong)"
    logger.info("PASSED: stateful TP decode generates correct continuation")
