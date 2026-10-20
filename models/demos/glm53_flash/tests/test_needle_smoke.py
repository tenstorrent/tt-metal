# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model single-chip needle-in-a-haystack smoke: ``GLM53Model`` (all 45 layers streamed through one
chip, ``tt/model.py``) prefills a ~16 K-token needle prompt (``tests/needle.py``) in consecutive chunks
(``prefill(start=)``, the chunked-prefill contract), then decodes greedily and reports whether the needle's
answer is retrieved, the wall time of every chunk pass and of every decode pass (one pass = 45 layer
builds + runs + releases, so this is a functionality check, not a latency figure).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_needle_smoke.py -s --timeout 1500

Knobs: GLM53_NEEDLE_TOKENS (prompt length, default 16384), GLM53_NEEDLE_CHUNK (8192), GLM53_NEEDLE_N
(greedy tokens, 16), GLM53_NEEDLE_DEPTH (0.5), GLM53_NEEDLE_SEED (0), GLM53_NEEDLE_DEADLINE_S (stop
decoding when the next pass would not finish before this many seconds after the test start; 0 = none),
GLM53_FULLMODEL_CACHE (tensorbin cache root as in test_full_model.py), GLM53_NEEDLE_LAYERS (a layer prefix
such as ``0-4`` for mechanics checks: the needle cannot be retrieved then), GLM53_NEEDLE_RESULTS (JSON
lines file).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, shard_headers
from models.demos.glm53_flash.reference.hf_layer_reference import parse_layers
from models.demos.glm53_flash.tests.needle import DEFAULT_ANSWER, build_needle_prompt_info, retrieved
from models.demos.glm53_flash.tt.model import GLM53Model, dram_view

N_TOKENS = int(os.environ.get("GLM53_NEEDLE_TOKENS", "16384"))
CHUNK = int(os.environ.get("GLM53_NEEDLE_CHUNK", "8192"))
N_GEN = int(os.environ.get("GLM53_NEEDLE_N", "16"))
DEPTH = float(os.environ.get("GLM53_NEEDLE_DEPTH", "0.5"))
SEED = int(os.environ.get("GLM53_NEEDLE_SEED", "0"))
DEADLINE = float(os.environ.get("GLM53_NEEDLE_DEADLINE_S", "0"))
_cache_env = os.environ.get("GLM53_FULLMODEL_CACHE", "/home/ttuser/ashai/glm53/_cache/fullmodel")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
LAYERS = parse_layers(os.environ["GLM53_NEEDLE_LAYERS"]) if os.environ.get("GLM53_NEEDLE_LAYERS") else None
RESULTS = os.environ.get("GLM53_NEEDLE_RESULTS")


def _dram(device, when: str):
    a, f = dram_view(device)
    logger.info(f"DRAM {when}: {a:.3f} GiB allocated, {f:.2f} GiB free")


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_needle_chunked_prefill(mesh_device):
    from transformers import AutoTokenizer

    t_start = time.perf_counter()
    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    np_ = build_needle_prompt_info(tok, N_TOKENS, DEPTH, seed=SEED)
    ids = np_.ids
    S = int(ids.numel())
    chunks = []
    while sum(chunks) < S:
        chunks.append(min(CHUNK, S - sum(chunks)))
    logger.info(
        f"needle prompt: {S} tokens ({np_.n_filler} filler sentences, needle {np_.needle!r} at sentence {np_.needle_sentence_index} "
        f"~ token {np_.needle_token_start}, depth {DEPTH}); chunks {chunks}; head {tok.decode(ids[:24].tolist())!r} ... "
        f"tail {tok.decode(ids[-40:].tolist())!r}"
    )
    t0 = time.perf_counter()
    model = GLM53Model.from_snapshot(
        mesh_device,
        snapshot,
        headers=shard_headers(snapshot),
        layers=LAYERS,
        max_batch=1,
        max_seq_len=S + N_GEN + 32,
        weight_cache_root=CACHE_ROOT,
    )
    logger.info(
        f"model: layers {model.layers[0]}..{model.layers[-1]} ({len(model.layers)}), cache root {CACHE_ROOT}, "
        f"embedding + head built in {model.embed_head_build_s:.1f} s (total {time.perf_counter() - t0:.1f} s)"
    )
    _dram(mesh_device, "embedding + head resident")

    # ---- chunked prefill: one pass over the stack per chunk; the last chunk's logits give the first token
    pass_times = []
    logits = None
    start = 0
    for i, n in enumerate(chunks):
        t0 = time.perf_counter()
        out = model.prefill(ids[start : start + n], user=0, start=start)
        dt = time.perf_counter() - t0
        pass_times.append(("prefill", start, start + n, dt))
        runs = [t.run_s[-1][1] for t in model.timing.values()]
        builds = [t.build_s[-1] for t in model.timing.values()]
        last_builds = sum(builds)
        logger.info(
            f"prefill chunk {i} [{start}, {start + n}): pass {dt:.0f} s (layer builds {sum(builds):.0f} s, layer runs {sum(runs):.0f} s, "
            f"slowest run layer {max(range(len(runs)), key=runs.__getitem__)} {max(runs):.1f} s)"
        )
        _dram(mesh_device, f"after chunk {i} (weights released)")
        if logits is not None:
            ttnn.deallocate(logits)
        logits = out
        start += n
    for dec in model.decoders.values():
        assert dec.next_start[0] == S, (dec.layer_idx, dec.next_start)
    lh = model.host_logits(logits)[0]
    ttnn.deallocate(logits)
    out_ids = [int(lh.argmax())]
    top5 = lh.topk(5)
    logger.info(
        f"first token (from the prefill logits): {out_ids[0]} {tok.decode(out_ids)!r}; top-5 {top5.indices.tolist()} "
        f"{[round(float(v), 3) for v in top5.values]}"
    )

    # ---- greedy decode, one pass over the stack per token, under the deadline
    decode_times = []
    stopped_early = False
    while len(out_ids) < N_GEN:
        if DEADLINE:
            # a decode pass costs about the layer builds of a pass (+ the first pass's compiles)
            est = sum(decode_times) / len(decode_times) if decode_times else 1.5 * last_builds + 60.0
            if time.perf_counter() - t_start + est * 1.1 > DEADLINE:
                stopped_early = True
                logger.warning(
                    f"deadline {DEADLINE:.0f} s: stopping after {len(out_ids)} tokens (a pass takes ~{est:.0f} s)"
                )
                break
        if tok.eos_token_id is not None and out_ids[-1] == tok.eos_token_id:
            break
        pos = S + len(out_ids) - 1
        t0 = time.perf_counter()
        lg, dev_ids = model.decode([out_ids[-1]], [pos])
        dt = time.perf_counter() - t0
        decode_times.append(dt)
        lh = model.host_logits(lg)[0]
        dev_id = int(ttnn.to_torch(dev_ids).view(torch.int32).reshape(-1)[0])
        ttnn.deallocate(lg)
        ttnn.deallocate(dev_ids)
        nxt = int(lh.argmax())
        out_ids.append(nxt)
        logger.info(
            f"decode step {len(out_ids) - 1} (pos {pos}): token {nxt} {tok.decode([nxt])!r} (device greedy {dev_id}), pass {dt:.0f} s; "
            f"text so far {tok.decode(out_ids)!r}"
        )
    text = tok.decode(out_ids)
    got = retrieved(text, np_.answer or DEFAULT_ANSWER)
    logger.info(
        f"generated {len(out_ids)} tokens: {out_ids}\ntext: {text!r}\nneedle answer {np_.answer!r} retrieved: {got}"
    )
    logger.info("per-layer timing (seconds, means over the passes):\n" + model.timing_table())
    summary = {
        "prompt_tokens": S,
        "chunks": chunks,
        "needle_token_start": np_.needle_token_start,
        "prefill_pass_s": [round(t, 1) for _, _, _, t in pass_times],
        "decode_pass_s": [round(t, 1) for t in decode_times],
        "generated": out_ids,
        "text": text,
        "retrieved": got,
        "stopped_early": stopped_early,
        "layers": len(model.layers),
    }
    logger.info(f"NEEDLE_RESULT {json.dumps(summary)}")
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(summary) + "\n")
    assert len(set(out_ids)) > 1 or len(out_ids) == 1, "degenerate generation (one token repeated)"
    if LAYERS is None and not stopped_early:
        assert got, f"needle {np_.answer!r} not retrieved in {text!r}"
    elif LAYERS is None:
        logger.warning("retrieval not asserted: the generation stopped at the deadline")
