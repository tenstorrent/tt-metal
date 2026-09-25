# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 26B-A4B chunked SP prefill, real weights + real prompt, vs the fp32 torch reference.

Checks per-layer hidden-state PCC (every chunk), last-token logits PCC and top-1 agreement.
The fp32 golden is cached under $GEMMA4_D_P_GOLDEN_DIR (default: generated/gemma4_26b_d_p_golden).
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_d_p.bringup.registry import record_result
from models.demos.gemma4_26b_d_p.reference.blocks import Gemma4TextModel
from models.demos.gemma4_26b_d_p.reference.config import DEFAULT_CKPT_DIR, Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader, load_text_model
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.gemma4_26b_d_p.tt.model import TtGemma4Model

GOLDEN_DIR = Path(os.environ.get("GEMMA4_D_P_GOLDEN_DIR", "generated/gemma4_26b_d_p_golden"))
PROMPT = (Path(__file__).parent / "prompt.txt")


def prompt_ids(n_tokens: int) -> torch.Tensor:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(DEFAULT_CKPT_DIR)
    text = PROMPT.read_text()
    ids = tok(text, return_tensors="pt").input_ids[0]
    while ids.numel() < n_tokens:
        ids = torch.cat([ids, ids[1:]])
    return ids[:n_tokens]


def golden(n_layers: int, seq: int, with_kv: bool = False):
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    f = GOLDEN_DIR / f"golden_L{n_layers}_S{seq}{'_kv' if with_kv else ''}.pt"
    if f.exists():
        return torch.load(f)
    ids = prompt_ids(seq)
    cfg = Gemma4TextConfig.from_json().with_layers(n_layers)
    ref = load_text_model(Gemma4TextModel(cfg), CheckpointReader(), dtype=torch.float32).eval()
    hidden = []
    orig = [l.forward for l in ref.layers]
    for i, l in enumerate(ref.layers):
        def fwd(*a, _f=orig[i], **k):
            out = _f(*a, **k)
            hidden.append(out[0][0].bfloat16())
            return out
        l.forward = fwd
    with torch.no_grad():
        logits, kv = ref.prefill(ids[None])
    g = {"ids": ids, "hidden": torch.stack(hidden), "logits": logits[0, -1].float()}
    if with_kv:
        g["kv"] = [(k.bfloat16(), v.bfloat16()) for k, v in kv]  # K post-norm+rope (HF rotate-half order), V post-v_norm
    torch.save(g, f)
    return g


@MESH_PARAMS
@pytest.mark.parametrize("n_layers", [6, 30], ids=["L6", "L30"])
@pytest.mark.parametrize("seq,chunk", [(8192, 4096)], ids=["8k-chunk4k"])
def test_model_prefill(mesh_device, device_params, n_layers, seq, chunk):
    g = golden(n_layers, seq)
    cfg = Gemma4TextConfig.from_json()
    model = TtGemma4Model(mesh_device, cfg, CheckpointReader(), fabric_config=device_params["fabric_config"], max_seq_len=seq,
                          chunk_size=chunk, layers=list(range(n_layers)))
    per_layer = {}

    def capture(li, x, kv_actual):
        h = model.gather_hidden(x, kv_actual)
        per_layer.setdefault(li, []).append(comp_pcc(g["hidden"][li, kv_actual : kv_actual + chunk].float(), h)[1])

    logits, _ = model.prefill(g["ids"], capture=capture)
    for li in sorted(per_layer):
        logger.info(f"layer {li:2d} ({cfg.layer_types[li][:4]}): chunk PCC {[round(p, 5) for p in per_layer[li]]}")
    ok, lpcc = comp_pcc(g["logits"], logits, 0.97)
    top1_ref, top1 = int(g["logits"].argmax()), int(logits.argmax())
    top5 = set(g["logits"].topk(5).indices.tolist()) & set(logits.topk(5).indices.tolist())
    worst = min(min(v) for v in per_layer.values())
    logger.info(f"mesh={mesh_id(mesh_device)} L{n_layers}: logits PCC {lpcc}, top1 ref={top1_ref} tt={top1}, top5 overlap {len(top5)}/5, worst layer PCC {worst}")
    # A truncated model's logits saturate at the softcap (dozens tied at 30.0), so top-1 is only meaningful at full depth.
    check_top1 = n_layers == cfg.num_hidden_layers
    passed = ok and worst > 0.99 and (top1 == top1_ref or not check_top1)
    record_result(f"layer:model_L{n_layers}", mesh_id(mesh_device), float(lpcc), bool(passed),
                  f"{seq} tokens, {seq // chunk}x{chunk} chunks; top1 {'match' if top1 == top1_ref else 'MISMATCH'}{'' if n_layers == cfg.num_hidden_layers else ' (not checked: softcap-saturated)'}; worst layer PCC {worst:.4f}")
    assert ok, lpcc
    assert worst > 0.99, worst
    if check_top1:
        assert top1 == top1_ref, (top1, top1_ref)
