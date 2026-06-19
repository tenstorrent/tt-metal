# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end TP=4: full 64-layer Qwen3.5-27B-FP8 on a (1,4) Blackhole mesh.

Builds the whole model (all layers sharded across 4 devices) and greedily
generates a few tokens via stateless prefill-only generation (re-prefill the
growing sequence each step — avoids KV-cache/GDN-state continuation). Coherent
output exercises embedding + 64 TP layers (attn + GDN) + distributed norms +
LM head end-to-end.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen3_5_9b/tests/test_e2e_tp.py -v -s
"""
import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.model import Qwen35Model


@pytest.mark.parametrize(
    "mesh_device",
    [{"P150": (1, 1), "P150x4": (1, 4)}.get(os.environ.get("MESH_DEVICE"), (1, min(len(ttnn.get_device_ids()), 4)))],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_e2e_tp(mesh_device, ensure_gc):
    os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.6-27B")

    logger.info("Building full TP model (loads + shards all 64 layers)...")
    model = Qwen35Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=256)

    from transformers import AutoTokenizer

    # model.args.CKPT_DIR is the resolved local snapshot dir (downloaded from the hub id).
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    prompt = "The capital of France is"
    ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    logger.info(f"prompt ids ({len(ids)}): {ids}")

    n_new = 8
    for step in range(n_new):
        T = len(ids)
        T_pad = max(128, math.ceil(T / 128) * 128)
        padded = ids + [0] * (T_pad - T)
        lt = model.prefill_tp(torch.tensor([padded], dtype=torch.long), valid_len=T).float()  # [vocab]
        nxt = int(torch.argmax(lt).item())
        ids.append(nxt)
        logger.info(f"step {step}: next_id={nxt} -> {tok.decode([nxt])!r}")

    text = tok.decode(ids)
    logger.info(f"GENERATED: {text!r}")

    gen = ids[-n_new:]
    assert len(set(gen)) > 1, f"degenerate generation (all same token): {gen}"
    assert all(0 <= t < model.vocab_size for t in gen)
    logger.info("PASSED: e2e TP generation produced valid, non-degenerate tokens")
