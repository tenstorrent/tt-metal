# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model single-chip needle-in-a-haystack smoke: ``GLM53Model`` (all 45 layers streamed through one
chip, ``tt/model.py``) prefills the canonical ~16 K-token needle prompt (``tests/needle.py`` =
``test_mesh_long_context.build_needle_ids``, the mesh run's and the HF references' ids) in consecutive
chunks (``prefill(start=)``, the chunked-prefill contract), then decodes greedily and reports whether the
needle key is retrieved, the wall time of every chunk pass and of every decode pass (one pass = 45 layer
builds + runs + releases, so this is a functionality check, not a latency figure).

Deadline-bounded jobs (the device protocol caps a job at 1500 s, a pass takes minutes): with
``GLM53_NEEDLE_CKPT`` the generated ids are checkpointed after every token and a later job resumes by
prefilling prompt + generated ids (still 2 chunks of 8192 for 16359 + 16 ids) and decoding on; the resumed
tokens then come from the prefill kernels where a continuous run would have decoded them (greedy near-ties may
differ), which the report states.

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/test_needle_smoke.py -s --timeout 1500

Knobs: GLM53_NEEDLE_TOKENS (prompt length, default 16384), GLM53_NEEDLE_CHUNK (8192), GLM53_NEEDLE_N
(greedy tokens, 16), GLM53_NEEDLE_DEPTH (0.5), GLM53_NEEDLE_DEADLINE_S (stop decoding when the next pass
would not finish before this many seconds after the test start; 0 = none), GLM53_NEEDLE_CKPT (checkpoint
JSON path), GLM53_FULLMODEL_CACHE (tensorbin cache root as in test_full_model.py), GLM53_NEEDLE_LAYERS (a
layer prefix such as ``0-4`` for mechanics checks: the needle cannot be retrieved then), GLM53_NEEDLE_RESULTS
(JSON lines file).
"""

from __future__ import annotations

import hashlib
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
from models.demos.glm53_flash.tests.needle import NEEDLE_KEY, build_needle_ids, eos_ids, needle_token_start, retrieved
from models.demos.glm53_flash.tt.model import GLM53Model, dram_view

N_TOKENS = int(os.environ.get("GLM53_NEEDLE_TOKENS", "16384"))
CHUNK = int(os.environ.get("GLM53_NEEDLE_CHUNK", "8192"))
N_GEN = int(os.environ.get("GLM53_NEEDLE_N", "16"))
DEPTH = float(os.environ.get("GLM53_NEEDLE_DEPTH", "0.5"))
DEADLINE = float(os.environ.get("GLM53_NEEDLE_DEADLINE_S", "0"))
CKPT = Path(os.environ["GLM53_NEEDLE_CKPT"]) if os.environ.get("GLM53_NEEDLE_CKPT") else None
_cache_env = os.environ.get("GLM53_FULLMODEL_CACHE", "/home/ttuser/ashai/glm53/_cache/fullmodel")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
LAYERS = parse_layers(os.environ["GLM53_NEEDLE_LAYERS"]) if os.environ.get("GLM53_NEEDLE_LAYERS") else None
RESULTS = os.environ.get("GLM53_NEEDLE_RESULTS")


def _dram(device, when: str):
    a, f = dram_view(device)
    logger.info(f"DRAM {when}: {a:.3f} GiB allocated, {f:.2f} GiB free")


def _chunks(total: int) -> list[int]:
    out = []
    while sum(out) < total:
        out.append(min(CHUNK, total - sum(out)))
    return out


def _load_ckpt(key: str) -> list[int]:
    if CKPT is None or not CKPT.exists():
        return []
    d = json.loads(CKPT.read_text())
    if d.get("key") != key:
        logger.warning(f"checkpoint {CKPT} is for another prompt / configuration: ignored")
        return []
    return [int(t) for t in d["generated"]]


def _save_ckpt(key: str, generated: list[int], text: str):
    if CKPT is not None:
        CKPT.parent.mkdir(parents=True, exist_ok=True)
        CKPT.write_text(json.dumps({"key": key, "generated": generated, "text": text, "time": time.time()}))


@pytest.mark.timeout(1500)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_needle_chunked_prefill(mesh_device):
    from transformers import AutoTokenizer

    t_start = time.perf_counter()
    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    prompt, text = build_needle_ids(tok, N_TOKENS, DEPTH)
    S = len(prompt)
    eos = eos_ids(snapshot, tok)
    key = hashlib.sha1(f"{prompt}|{CHUNK}|{LAYERS}".encode()).hexdigest()[:16]
    generated = _load_ckpt(key)
    n_prev = len(generated)
    seq = prompt + generated  # a resumed job re-prefills the tokens generated so far
    chunks = _chunks(len(seq))
    logger.info(
        f"needle prompt: {S} ids (needle {NEEDLE_KEY!r} ~ token {needle_token_start(tok, N_TOKENS, DEPTH)}, depth {DEPTH}); "
        f"resuming with {n_prev} generated ids {generated}; prefill {len(seq)} ids in chunks {chunks}; "
        f"tail {tok.decode(seq[-24:])!r}"
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

    # ---- chunked prefill: one pass over the stack per chunk; the last chunk's logits give the next token
    ids_t = torch.tensor(seq, dtype=torch.int64)
    pass_times, last_builds = [], 0.0
    logits = None
    start = 0
    for i, n in enumerate(chunks):
        t0 = time.perf_counter()
        out = model.prefill(ids_t[start : start + n], user=0, start=start)
        dt = time.perf_counter() - t0
        pass_times.append(dt)
        runs = [t.run_s[-1][1] for t in model.timing.values()]
        builds = [t.build_s[-1] for t in model.timing.values()]
        last_builds = sum(builds)
        slow = max(range(len(runs)), key=runs.__getitem__)
        logger.info(
            f"prefill chunk {i} [{start}, {start + n}): pass {dt:.0f} s (layer builds {last_builds:.0f} s, layer runs {sum(runs):.0f} s, "
            f"slowest run layer {model.layers[slow]} {runs[slow]:.1f} s)"
        )
        _dram(mesh_device, f"after chunk {i} (weights released)")
        if logits is not None:
            ttnn.deallocate(logits)
        logits = out
        start += n
    for dec in model.decoders.values():
        assert dec.next_start[0] == len(seq), (dec.layer_idx, dec.next_start)
    lh = model.host_logits(logits)[0]
    ttnn.deallocate(logits)
    nxt = int(lh.argmax())
    top5 = lh.topk(5)
    generated.append(nxt)
    _save_ckpt(key, generated, tok.decode(generated))
    logger.info(
        f"token {len(generated) - 1} (from the prefill logits at position {len(seq) - 1}): {nxt} {tok.decode([nxt])!r}; "
        f"top-5 {top5.indices.tolist()} {[round(float(v), 3) for v in top5.values]}; text so far {tok.decode(generated)!r}"
    )

    # ---- greedy decode, one pass over the stack per token, under the deadline
    decode_times = []
    stopped_early = False
    while len(generated) < N_GEN and generated[-1] not in eos:
        if DEADLINE:
            # a decode pass costs about the layer builds of a pass (+ the first pass's compiles)
            est = sum(decode_times) / len(decode_times) if decode_times else 1.5 * last_builds + 60.0
            if time.perf_counter() - t_start + est * 1.1 > DEADLINE:
                stopped_early = True
                logger.warning(
                    f"deadline {DEADLINE:.0f} s: stopping after {len(generated)} tokens (a pass takes ~{est:.0f} s)"
                )
                break
        pos = S + len(generated) - 1
        t0 = time.perf_counter()
        lg, dev_ids = model.decode([generated[-1]], [pos])
        dt = time.perf_counter() - t0
        decode_times.append(dt)
        lh = model.host_logits(lg)[0]
        dev_id = int(ttnn.to_torch(dev_ids).view(torch.int32).reshape(-1)[0])
        ttnn.deallocate(lg)
        ttnn.deallocate(dev_ids)
        nxt = int(lh.argmax())
        generated.append(nxt)
        _save_ckpt(key, generated, tok.decode(generated))
        logger.info(
            f"token {len(generated) - 1} (decode at pos {pos}): {nxt} {tok.decode([nxt])!r} (device greedy {dev_id}), pass {dt:.0f} s; "
            f"text so far {tok.decode(generated)!r}"
        )
    gen_text = tok.decode(generated)
    got = retrieved(gen_text, NEEDLE_KEY)
    done = len(generated) >= N_GEN or generated[-1] in eos
    logger.info(
        f"generated {len(generated)} tokens ({n_prev} resumed from the checkpoint, {len(generated) - n_prev} new): {generated}\n"
        f"text: {gen_text!r}\nneedle key {NEEDLE_KEY!r} retrieved: {got}; complete: {done}"
    )
    logger.info("per-layer timing (seconds, means over the passes):\n" + model.timing_table())
    summary = {
        "prompt_tokens": S,
        "prefilled_tokens": len(seq),
        "chunks": chunks,
        "resumed": n_prev,
        "prefill_pass_s": [round(t, 1) for t in pass_times],
        "decode_pass_s": [round(t, 1) for t in decode_times],
        "generated": generated,
        "text": gen_text,
        "retrieved": got,
        "complete": done,
        "stopped_early": stopped_early,
        "layers": len(model.layers),
    }
    logger.info(f"NEEDLE_RESULT {json.dumps(summary)}")
    if RESULTS:
        with open(RESULTS, "a") as f:
            f.write(json.dumps(summary) + "\n")
    if LAYERS is None and (done or len(generated) >= 8):
        assert got, f"needle key {NEEDLE_KEY!r} not retrieved in {gen_text!r}"
    elif LAYERS is None:
        logger.warning("retrieval not asserted yet: fewer than 8 tokens so far (resume with the checkpoint)")
