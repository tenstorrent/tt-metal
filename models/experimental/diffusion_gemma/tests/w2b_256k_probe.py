# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""256K end-to-end blocker probe for DiffusionGemma denoise (#48549).

Stresses, at REAL DiffusionGemma text dims (hidden 2816, head_dim 256, 16/8
heads, sliding 1024), the memory-scaling pieces the tiny-model tests mask:

  1. create_rope_caches at max_seq=256K  (host x_dummy + device cos/sin caches)
  2. denoise mask  [1,1,256, P+256]       (device bf16)
  3. full-attn KV cache  [1,8,256K,256]x2 (device bf16)

Each step is wrapped so the FIRST failure (host OOM / device alloc / TT_FATAL) is
reported with the step + bytes, instead of aborting the run.

This is an opt-in diagnostic script, intentionally not collected by pytest:

  DG_RUN_DEVICE=1 MESH_DEVICE=P150x4 W2B_SEQ=262144 pytest -q -s w2b_256k_probe.py
"""
import os
import traceback

import pytest
import torch
import ttnn

pytestmark = [
    pytest.mark.skipif(os.environ.get("DG_RUN_DEVICE") != "1", reason="device only"),
    pytest.mark.use_module_device,
]

# Real DiffusionGemma-26B-A4B text dims.
HIDDEN = 2816
HEAD_DIM = 256
GLOBAL_HEAD_DIM = 512
N_HEADS = 16
N_KV = 8
SLIDING_WINDOW = 1024
CANVAS = 256
SEQ = int(os.environ.get("W2B_SEQ", "262144"))


def _gb(nbytes):
    return nbytes / (1024**3)


def _real_text_config(max_seq):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        vocab_size=262144,
        hidden_size=HIDDEN,
        intermediate_size=704,
        num_hidden_layers=2,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV,
        num_global_key_value_heads=N_KV,
        head_dim=HEAD_DIM,
        global_head_dim=GLOBAL_HEAD_DIM,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=SLIDING_WINDOW,
        max_position_embeddings=max_seq,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1000000.0},
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def _step(name, fn):
    try:
        info = fn()
        print(f"\n[256k-probe] OK   {name}  {info}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"\n[256k-probe] FAIL {name}  -> {type(e).__name__}: {str(e)[:200]}")
        traceback.print_exc()
        return False


def test_256k_blocker_probe(device):
    total = SEQ + CANVAS
    print(f"\n[256k-probe] SEQ={SEQ} total={total} hidden={HIDDEN} head_dim={HEAD_DIM}")

    def _rope():
        from models.demos.gemma4.tt.model import create_rope_caches

        cfg = _real_text_config(total)
        x_dummy_gb = _gb(1 * total * HIDDEN * 4)
        c4d, c2d = create_rope_caches(device, cfg, total)
        return f"x_dummy~{x_dummy_gb:.2f}GB host; cos/sin device caches built for {list(c4d)}"

    _step("create_rope_caches@256K (real hidden)", _rope)

    def _mask():
        nbytes = CANVAS * total * 2
        m = torch.zeros(1, 1, CANVAS, total, dtype=torch.bfloat16)
        tt = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        del tt
        return f"mask [1,1,{CANVAS},{total}] ~{_gb(nbytes):.3f}GB device"

    _step("denoise mask [256, P+256]@256K", _mask)

    def _kv():
        per = N_KV * total * HEAD_DIM * 2
        k = torch.zeros(1, N_KV, total, HEAD_DIM, dtype=torch.bfloat16)
        tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        tt_v = ttnn.from_torch(k.clone(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        del tt_k, tt_v
        return f"K+V each [1,{N_KV},{total},{HEAD_DIM}] ~{_gb(per):.2f}GB x2 device"

    _step("full-attn KV cache@256K", _kv)
