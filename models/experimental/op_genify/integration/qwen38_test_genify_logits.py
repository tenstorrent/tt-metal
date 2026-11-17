# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.8-27B QB2: logits with the stock matmuls vs genify_matmul (#65150).

Builds the demo's reduced stack (layer 0 = linear attention, layer 3 = full attention, real weights,
(1, 4) mesh, TP4) once, with the genify-only weights (interleaved gate_up) present, then runs the
same prompts with TT_GENIFY_MATMUL=0 and =1:
  - prefill at M = 128 (ttnn.linear 1D mcast path) and M = 512 (minimal_matmul path), all-position
    logits (interleaved LM head);
  - one eager decode step after each prefill (DRAM-sharded projections, DRAM-sharded LM head).
genify must route every projection (counted), and its logits must match stock.

Run (needs MODEL_WEIGHTS_DIR):
    MODEL_WEIGHTS_DIR=/localdev/avelickov/hf_models/Qwen3.8-27B \
    scripts/run_safe_pytest.sh models/experimental/op_genify/integration/qwen38_test_genify_logits.py -s
"""

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.qwen38_27b_qb2.tt.generator import build_generator, configure_fabric
from models.experimental.op_genify import genify_matmul_adapter as genify

MODEL_DIR = Path(__file__).resolve().parents[3] / "demos" / "qwen38_27b_qb2"
LAYERS = [0, 3]
LENGTHS = (128, 512)
MIN_PCC = 0.99
MIN_TOP1 = 0.9


def _prompt(gen, length):
    tok = gen.tokenizer
    base = tok.apply_chat_template(
        [{"role": "user", "content": "Explain why the sky is blue."}], tokenize=False, add_generation_prompt=True
    )
    base = tok(base, add_special_tokens=False)["input_ids"]
    return (base * ((length + len(base) - 1) // len(base)))[:length]


def _prefill_then_decode(gen, prompt):
    """All-position prefill logits, then one eager decode step on the greedy next token."""
    length = len(prompt)
    prefill = gen.prefill_logits(prompt)[0].float()  # [L, vocab]
    nxt = int(prefill[length - 1].argmax())
    gen._copy(torch.tensor([length], dtype=torch.int32), gen.positions, "position_refreshes")
    gen._copy(torch.tensor([length], dtype=torch.int32), gen.rope_indices, "rope_refreshes")
    ids = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
    ids.reshape(-1)[0] = nxt
    gen._copy(ids, gen.tokens, "token_refreshes")
    decode = gen._host_logits(gen._model_step())  # [1, 1, 32, vocab]; row 0 is the request
    return prefill, decode.reshape(-1, decode.shape[-1])[0].float()


def test_qwen38_genify_logits():
    torch.set_num_threads(8)
    calls = {"n": 0}
    stock_matmul = genify.matmul

    def counting(*args, **kwargs):
        calls["n"] += 1
        return stock_matmul(*args, **kwargs)

    genify.matmul = counting
    os.environ[genify.ENV] = "1"  # build the genify-only weights too
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(MODEL_DIR, mesh, layer_indices=LAYERS, sampling_strategy="split", head_strategy="dram")
        gen._ensure_cache(1, max(LENGTHS) + 64)
        results = {}
        for mode in ("0", "1"):
            os.environ[genify.ENV] = mode
            before = calls["n"]
            for length in LENGTHS:
                results[mode, length] = _prefill_then_decode(gen, _prompt(gen, length))
            routed = calls["n"] - before
            logger.info(f"TT_GENIFY_MATMUL={mode}: {routed} genify_matmul calls")
            assert (routed > 0) == (mode == "1"), f"mode {mode}: {routed} genify calls"

        failures = []
        for length in LENGTHS:
            (sp, sd), (gp, gd) = results["0", length], results["1", length]
            _, p_pcc = comp_pcc(sp, gp, MIN_PCC)
            _, d_pcc = comp_pcc(sd, gd, MIN_PCC)
            top1 = (sp.argmax(-1) == gp.argmax(-1)).float().mean().item()
            same_next = int(sd.argmax()) == int(gd.argmax())
            logger.info(
                f"M={length}: prefill PCC {p_pcc:.5f}, top-1 agreement {top1:.3f} over {length} positions; "
                f"decode PCC {d_pcc:.5f}, same greedy token: {same_next}"
            )
            if p_pcc < MIN_PCC or d_pcc < MIN_PCC or top1 < MIN_TOP1:
                failures.append(length)
        assert not failures, f"genify logits differ from stock at M = {failures}"
    finally:
        genify.matmul = stock_matmul
        os.environ.pop(genify.ENV, None)
        if gen is not None:
            gen.close()
        ttnn.close_mesh_device(mesh)
