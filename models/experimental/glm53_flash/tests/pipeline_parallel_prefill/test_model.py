# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The pipeline-parallel prefill model on the Galaxy: 8 pipeline stages, each a 1 x 4 TP submesh.

* ``test_pipeline_matches_hf`` -- the first ``GLM_PP_LAYERS`` (default 8: one per stage, KDA dense / MLA / KDA MoE)
  layers plus the head, a multi-chunk prompt so several chunks are in flight; pipelined and sequential schedules
  must agree exactly, and the last-token logits are graded against the streaming HF reference.
* ``test_decode_matches_prefill`` -- the same layers: prefill a prefix into a session, then decode the remaining
  prompt tokens one step at a time; the final decode logits must match a one-shot prefill and the HF reference.
* ``test_full_model_capital_of_france`` -- all 45 layers: the greedy next token of "The capital of France is".
* ``test_full_model_generate`` -- all 45 layers: greedy generation continuing that prompt, with decode throughput.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import pytest
import torch

from models.experimental.glm53_flash.tests.test_kda_block import pcc
from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.model import GLMPipelinePrefill
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.weights import ModelSource

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from checkpoint import default_snapshot_dir  # noqa: E402
from hf_layer_reference import build_layer, load_config  # noqa: E402

try:
    SNAPSHOT = default_snapshot_dir()
except FileNotFoundError:
    SNAPSHOT = None

pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="GLM-5.3-Flash checkpoint not available")

CACHE = Path(os.environ.get("GLM_PP_CACHE", Path.home() / ".cache/ttnn/models/glm53_flash_pp"))
N_LAYERS = int(os.environ.get("GLM_PP_LAYERS", "8"))
SEQ, CHUNK = 300, 128
PROMPT = (
    "Write a detailed travel guide for a three-day visit to Kyoto in autumn. For each day give a morning, afternoon "
    "and evening plan, name the temples, gardens and neighbourhoods to visit, suggest where to eat and what local "
    "dishes to try, estimate the walking distances and public transport options between stops, and add practical "
    "advice about tickets, opening hours, crowds and etiquette. Close with a packing list and a short note on the "
    "history that makes the city's architecture so distinctive."
)


@torch.no_grad()
def hf_prefill_logits(ids: torch.Tensor, n_layers: int, source: ModelSource) -> torch.Tensor:
    """Streaming HF reference: layers 0 .. n_layers-1 one at a time (bf16 streams between layers, fp32 math), then
    the HyperHead mean, final RMSNorm and lm_head -> last-token fp32 logits."""
    _, text = load_config(SNAPSHOT)
    s = ids.numel()
    x = source.embedding()[ids.reshape(1, -1)][:, :, None, :].expand(-1, -1, text.hc_mult, -1).float()
    for idx in range(n_layers):
        layer, _ = build_layer(SNAPSHOT, text, idx, source.headers)
        layer = layer.float()
        x = layer(
            x,
            attention_mask=torch.ones(1, s, dtype=torch.bool),
            position_ids=torch.arange(s)[None],
            position_embeddings=None,
        )[0]
        x = x.to(torch.bfloat16).float()
        del layer
        gc.collect()
    h = x[0, -1].mean(0).to(torch.bfloat16).float()
    h = h * torch.rsqrt(h.pow(2).mean() + text.rms_norm_eps)
    h = source.final_norm().float() * h.to(torch.bfloat16).float()
    return h @ source.lm_head().float().t()


@pytest.fixture(scope="module")
def tokenizer():
    from models.experimental.glm53_flash.tt.tokenizer_chat import load_tokenizer

    return load_tokenizer(SNAPSHOT)


def _build(stages, n_layers):
    cfg = Glm5NextTextConfig.from_json(str(SNAPSHOT))
    source = ModelSource(SNAPSHOT)
    t0 = time.time()
    model = GLMPipelinePrefill(stages, cfg, source, n_layers=n_layers, cache_dir=CACHE)
    n = sum(len(p) for p in model.placement)
    print(f"\nbuilt {n} layers on {len(model.stages)} stages in {time.time() - t0:.0f}s: {model.placement}")
    return model, source


@pytest.mark.timeout(3600)
def test_pipeline_matches_hf(stages, tokenizer):
    model, source = _build(stages, N_LAYERS)
    ids = tokenizer.encode_chat([{"role": "user", "content": PROMPT}])
    while len(ids) < SEQ:
        ids = ids + ids
    ids = torch.tensor(ids[:SEQ])

    model.prefill(ids, chunk=CHUNK, pipelined=False)  # program-cache warm-up
    t0 = time.time()
    seq_logits = model.prefill(ids, chunk=CHUNK, pipelined=False)
    t_seq = time.time() - t0
    t0 = time.time()
    pipe_logits = model.prefill(ids, chunk=CHUNK, pipelined=True)
    t_pipe = time.time() - t0
    print(f"prefill {SEQ} tokens in chunks of {CHUNK}: sequential {t_seq:.2f}s, pipelined {t_pipe:.2f}s")
    assert torch.equal(seq_logits, pipe_logits), "pipelined and sequential schedules disagree"

    ref = hf_prefill_logits(ids, N_LAYERS, source)
    p = pcc(pipe_logits, ref)
    top_dev, top_ref = pipe_logits.topk(5).indices.tolist(), ref.topk(5).indices.tolist()
    gap = (ref[top_ref[0]] - ref[top_dev[0]]).item()
    print(
        f"last-token logits PCC {p:.5f}; top-5 device {top_dev} vs HF {top_ref}; "
        f"HF logit of device top-1 is {gap:.3f} below HF top-1 (HF top-1 / top-2 gap {(ref[top_ref[0]] - ref[top_ref[1]]).item():.3f})"
    )
    assert p > 0.99
    # a partial-depth head can sit on a near-tie: the device's pick must be one HF rates within a small margin
    assert top_dev[0] in top_ref[:3] and gap < 0.25


@pytest.mark.timeout(7200)  # a cold weight cache converts ~300 GB of checkpoint
def test_full_model_capital_of_france(stages, tokenizer):
    model, source = _build(stages, None)
    ids = torch.tensor(tokenizer.encode("The capital of France is"))
    t0 = time.time()
    logits = model.prefill(ids, chunk=128)
    print(f"prefill {ids.numel()} tokens through 45 layers: {time.time() - t0:.2f}s")
    top = logits.topk(5).indices.tolist()
    words = [tokenizer.decode([t]) for t in top]
    print(f"top-5 next tokens: {list(zip(top, words))}")
    assert words[0].strip() == "Paris", words
    if os.environ.get("GLM_PP_FULL_REF"):
        ref = hf_prefill_logits(ids, 45, source)
        print(f"logits PCC vs HF {pcc(logits, ref):.5f}; HF top-5 {ref.topk(5).indices.tolist()}")


N_DECODE = 11  # SEQ - N_DECODE = 289 prefilled (one token into an indexer pool)


@pytest.mark.timeout(3600)
def test_decode_matches_prefill(stages, tokenizer):
    model, source = _build(stages, N_LAYERS)
    ids = tokenizer.encode_chat([{"role": "user", "content": PROMPT}])
    while len(ids) < SEQ:
        ids = ids + ids
    ids = torch.tensor(ids[:SEQ])
    n_prefill = SEQ - N_DECODE

    full = model.prefill(ids, chunk=CHUNK)
    session = model.new_session()
    model.prefill(ids[:n_prefill], chunk=CHUNK, session=session)
    times = []
    for tok in ids[n_prefill:].tolist():
        t0 = time.time()
        logits = model.decode(tok, session)
        times.append(time.time() - t0)
    assert session.pos == SEQ and session.tokens == ids.tolist()
    print(
        f"prefill {n_prefill} + decode {N_DECODE} through {N_LAYERS} layers: decode step "
        f"{1e3 * sum(times[1:]) / (len(times) - 1):.0f} ms (first {1e3 * times[0]:.0f} ms)"
    )
    p_full = pcc(logits, full)
    ref = hf_prefill_logits(ids, N_LAYERS, source)
    p_ref = pcc(logits, ref)
    print(
        f"decode vs one-shot prefill logits PCC {p_full:.5f} (top-5 {logits.topk(5).indices.tolist()} vs "
        f"{full.topk(5).indices.tolist()}); vs HF {p_ref:.5f} (HF top-5 {ref.topk(5).indices.tolist()})"
    )
    assert p_full > 0.995
    assert p_ref > 0.99


N_GENERATE = 16


@pytest.mark.timeout(7200)
def test_full_model_generate(stages, tokenizer):
    model, _ = _build(stages, None)
    ids = torch.tensor(tokenizer.encode("The capital of France is"))
    model.generate(ids, 2)  # program-cache warm-up of the prefill and decode shapes
    t0 = time.time()
    out, session = model.generate(ids, N_GENERATE)
    dt = time.time() - t0
    text = tokenizer.decode(out)
    print(f"generated {len(out)} tokens in {dt:.2f}s ({len(out) / dt:.2f} tok/s incl. prefill): {text!r}")
    assert session.pos == ids.numel() + len(out) - 1
    assert text.strip().startswith("Paris"), text
