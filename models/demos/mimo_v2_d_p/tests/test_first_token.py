# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill -> the first generated token, with the final norm + lm_head on device.

test_first_token: a chat prompt in one chunk; the device logits are checked against the same norm + lm_head in
fp32 on host (from the device's own final hidden), the greedy token printed. MIMO_FT_STEPS > 1 appends tokens by
re-prefilling the last chunk (its prefix KV stays in the cache), so a few words come out without a decode path.

test_kv_reload: a prompt over two chunks. The prefix chunk's KV is saved to disk after it is prefilled; then
  A  the last chunk on top of the live cache (the reference),
  C  the last chunk on fresh (zero) caches: must change the logits, else the cache is not being read,
  B  the saved KV loaded into fresh caches (bit-exact, checked), the last chunk again: must give A's token / logits.

    MIMO_FT_LAYERS (default all 48), MIMO_FT_CHUNK (default 1024), MIMO_FT_STEPS (default 1), MIMO_FT_QUESTION
"""

import math
import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mimo_v2_d_p.reference.config import MiMoTextConfig
from models.demos.mimo_v2_d_p.reference.remote_st import LOCAL
from models.demos.mimo_v2_d_p.reference.weights import global_state, layer_state
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS, mesh_id
from models.demos.mimo_v2_d_p.tt.attention.kv_cache import load_kv_cache, save_kv_cache
from models.demos.mimo_v2_d_p.tt.model import PAD_TOKEN_ID, TtMiMoModel
from models.demos.mimo_v2_d_p.tt.options import MiMoRuntimeOptions

N_LAYERS = int(os.environ.get("MIMO_FT_LAYERS", "48"))
CHUNK = int(os.environ.get("MIMO_FT_CHUNK", "1024"))
STEPS = int(os.environ.get("MIMO_FT_STEPS", "1"))
QUESTION = os.environ.get("MIMO_FT_QUESTION", "What is the capital of Paris ?")
THINKING = (
    os.environ.get("MIMO_FT_THINKING", "1") != "0"
)  # 0: the template's enable_thinking=False (empty <think></think>)
DOC = Path(__file__).parent / "prompt.txt"
OUT_DIR = Path(os.environ.get("MIMO_FT_OUT", Path(__file__).parents[4] / "generated" / "mimo_first_token"))


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(LOCAL), trust_remote_code=True)


def chat_ids(tok, content):
    s = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True, enable_thinking=THINKING
    )
    return tok(s, add_special_tokens=False)["input_ids"]


def doc_prompt(tok, n_tokens):
    """A chat prompt of exactly ``n_tokens``: the start of tests/prompt.txt as a document, then QUESTION."""
    doc = tok(DOC.read_text(), add_special_tokens=False)["input_ids"]
    frame = len(chat_ids(tok, f"Here is a document:\n\n\n\nQuestion: {QUESTION}"))
    lo, hi = 0, len(doc)
    while lo < hi:  # the longest document prefix that fits (token counts are not additive across the seam)
        mid = (lo + hi + 1) // 2
        n = len(chat_ids(tok, f"Here is a document:\n\n{tok.decode(doc[:mid])}\n\nQuestion: {QUESTION}"))
        lo, hi = (mid, hi) if n <= n_tokens else (lo, mid - 1)
    ids = chat_ids(tok, f"Here is a document:\n\n{tok.decode(doc[:lo])}\n\nQuestion: {QUESTION}")
    assert frame < len(ids) <= n_tokens, (frame, len(ids), n_tokens)
    return ids


def build(mesh_device, device_params, max_seq):
    cfg = MiMoTextConfig.from_json()
    return cfg, TtMiMoModel(
        mesh_device,
        cfg,
        lambda i: layer_state(i, cfg),
        fabric_config=device_params["fabric_config"],
        max_seq_len=max_seq,
        chunk_size=CHUNK,
        layers=list(range(N_LAYERS)),
        global_state=global_state,
        lm_head=True,
        options=MiMoRuntimeOptions.from_env(),
    )


def run_chunk(model, ids, c, *, keep_hidden=False):
    """Prefill chunk ``c`` of ``ids`` (the last one padded); -> (logits after the chunk's last real token, hidden)."""
    C = model.chunk_size
    end = min(len(ids), (c + 1) * C)
    chunk = torch.full((C,), PAD_TOKEN_ID, dtype=torch.long)
    chunk[: end - c * C] = torch.tensor(ids[c * C : end])
    x = model.prefill_chunk(chunk, c * C, valid_end=end)
    logits = model.next_token_logits(x, c * C, end - 1)
    hidden = model.gather_hidden(x, c * C)[end - 1 - c * C] if keep_hidden else None
    x.deallocate(True)
    return logits, hidden


def host_lm_head(cfg, hidden):
    g = global_state(("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"))
    h = hidden.float()
    h = h * torch.rsqrt(h.pow(2).mean() + cfg.layernorm_epsilon) * g["norm.weight"].float()
    return g["lm_head.weight"].float() @ h


def top(tok, logits, k=5):
    v, i = logits.topk(k)
    return ", ".join(f"{tok.decode([t])!r} {x:.2f}" for t, x in zip(i.tolist(), v.tolist()))


def generate(tok, model, ids, logits):
    """Greedy tokens after ``ids`` until <|im_end|> (or MIMO_FT_STEPS), from the logits of the prompt's last token: each
    step re-prefills the chunk holding the newest token (earlier chunks stay in the KV cache)."""
    out, step_ms = [int(logits.argmax())], []
    while len(out) < STEPS and out[-1] != tok.eos_token_id:
        seq = ids + out
        t0 = time.perf_counter()
        logits, _ = run_chunk(model, seq, (len(seq) - 1) // CHUNK)
        step_ms.append((time.perf_counter() - t0) * 1e3)
        out.append(int(logits.argmax()))
        if len(out) % 64 == 0:
            logger.info(f"{len(out)} tokens: ...{tok.decode(out[-64:])!r}")
    if step_ms:
        med = sorted(step_ms)[len(step_ms) // 2]
        logger.info(
            f"generated {len(out)} tokens (stop: {'eos' if out[-1] == tok.eos_token_id else 'MIMO_FT_STEPS'}); per token "
            f"median {med:.1f} ms -> {1e3 / med:.2f} tok/s, total {sum(step_ms) / 1e3:.1f} s"
        )
    return out


DOCS = os.environ.get(
    "MIMO_FT_DOCS",
    "METALIUM_GUIDE.md:tech_reports/tensor_layouts/tensor_layouts.md:tech_reports/tensor_accessor/tensor_accessor.md",
)
DOC_QUESTION = os.environ.get(
    "MIMO_FT_DOC_QUESTION",
    "Using only the documents above: what is a circular buffer in TT-Metalium, and how do the reader, compute and writer "
    "kernels of a Tensix core use circular buffers to pass tiles between each other?",
)


@pytest.mark.timeout(14400)
@MESH_PARAMS
def test_long_doc(mesh_device, device_params):
    """A long prompt (MIMO_FT_DOCS: ':'-separated files of the repo, stitched) prefilled chunk by chunk, then a greedy
    answer to MIMO_FT_DOC_QUESTION."""
    tok = tokenizer()
    root = Path(__file__).parents[4]
    parts = [f"=== File: {f} ===\n\n{(root / f).read_text()}" for f in DOCS.split(":")]
    ids = chat_ids(tok, "\n\n".join(parts) + f"\n\n=== End of documents ===\n\n{DOC_QUESTION}")
    n_prompt_chunks = math.ceil(len(ids) / CHUNK)
    n_chunks = math.ceil((len(ids) + STEPS) / CHUNK)
    cfg, model = build(mesh_device, device_params, max_seq=(n_chunks + 1) * CHUNK)
    logger.info(f"prompt {len(ids)} tokens ({DOCS}) = {n_prompt_chunks} chunks of {CHUNK}")
    t0 = time.perf_counter()
    for c in range(n_prompt_chunks - 1):
        run_chunk(model, ids, c)
    logits, _ = run_chunk(model, ids, n_prompt_chunks - 1)
    logger.info(f"prefill {len(ids)} tokens: {(time.perf_counter() - t0) * 1e3:.0f} ms (eager, incl. the logits read)")
    logger.info(f"top-5: {top(tok, logits)}")
    out = generate(tok, model, ids, logits)
    logger.info(
        f"mesh={mesh_id(mesh_device)} layers={N_LAYERS} thinking={THINKING} Q: {DOC_QUESTION!r} ->\n{tok.decode(out)}"
    )
    assert logits.isfinite().all()


@pytest.mark.timeout(14400)
@MESH_PARAMS
def test_first_token(mesh_device, device_params):
    tok = tokenizer()
    ids = chat_ids(tok, QUESTION)
    n_chunks = math.ceil((len(ids) + STEPS) / CHUNK)
    cfg, model = build(mesh_device, device_params, max_seq=(n_chunks + 1) * CHUNK)
    for c in range(math.ceil(len(ids) / CHUNK) - 1):
        run_chunk(model, ids, c)
    c = (len(ids) - 1) // CHUNK
    logits, hidden = run_chunk(model, ids, c, keep_hidden=True)

    ref = host_lm_head(cfg, hidden)
    _, pcc = comp_pcc(ref, logits)
    logger.info(f"device lm_head vs host fp32 (same hidden): PCC {pcc:.6f}, argmax {logits.argmax()} / {ref.argmax()}")
    logger.info(f"top-5: {top(tok, logits)}")
    out = generate(tok, model, ids, logits)
    logger.info(
        f"mesh={mesh_id(mesh_device)} layers={N_LAYERS} thinking={THINKING} prompt {len(ids)} tokens: {QUESTION!r} ->\n"
        f"{tok.decode(out)}"
    )
    assert pcc > 0.999 and logits.isfinite().all(), pcc


@pytest.mark.timeout(14400)
@MESH_PARAMS
def test_kv_reload(mesh_device, device_params):
    tok = tokenizer()
    ids = doc_prompt(tok, CHUNK + CHUNK // 2)
    assert len(ids) > CHUNK, len(ids)
    cfg, model = build(mesh_device, device_params, max_seq=2 * CHUNK)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {t: OUT_DIR / f"kv_{mesh_id(mesh_device)}_L{N_LAYERS}_{t}.pt" for t in model.kv}

    run_chunk(model, ids, 0)
    for t, cache in model.kv.items():
        save_kv_cache(cache, paths[t])
    logits_a, _ = run_chunk(model, ids, 1)

    def fresh_caches():
        for cache in model.kv.values():
            cache.k.deallocate(True)
            cache.v.deallocate(True)
        model.kv = model.allocate_kv_caches()

    fresh_caches()
    logits_c, _ = run_chunk(model, ids, 1)

    fresh_caches()
    for t, cache in model.kv.items():
        load_kv_cache(cache, paths[t])
        saved = torch.load(paths[t])
        for n in ("k", "v"):
            back = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(getattr(cache, n))]
            assert all(torch.equal(a, b) for a, b in zip(saved[n], back)), f"{t}.{n}: reloaded KV differs from saved"
    logits_b, _ = run_chunk(model, ids, 1)

    _, pcc_ab = comp_pcc(logits_a, logits_b)
    _, pcc_ac = comp_pcc(logits_a, logits_c)
    exact = torch.equal(logits_a, logits_b)
    logger.info(f"prompt {len(ids)} tokens = chunk 0 (cached) + {len(ids) - CHUNK} in chunk 1")
    logger.info(f"A live cache:      {tok.decode([int(logits_a.argmax())])!r} | top-5 {top(tok, logits_a)}")
    logger.info(
        f"B reloaded cache:  {tok.decode([int(logits_b.argmax())])!r} | PCC vs A {pcc_ab:.6f}, bit-exact {exact}"
    )
    logger.info(f"C empty cache:     {tok.decode([int(logits_c.argmax())])!r} | PCC vs A {pcc_ac:.6f}")
    assert logits_a.argmax() == logits_b.argmax() and pcc_ab > 0.9999, pcc_ab
    assert pcc_ac < pcc_ab, (pcc_ac, pcc_ab)
