# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real GLM-5.3-Flash layers (checkpoint weights, full width) on one TP4 pipeline stage, prefilled chunk by chunk.

The layer inputs are the real residual streams of a chat prompt: embedded and pushed through HF layers 0 .. L-1 on
the host. Each tested layer runs on a 1 x 4 submesh in several prefill chunks (the last one padded), so the KDA conv
and recurrent carries and the MLA latent / pooled-key caches are exercised, and is graded per token against

* implementation -- the HF layer with its routed experts rounded to their bfloat4_b values (isolates the device code);
* deployment     -- the exact HF layer (adds the 4-bit expert storage cost).

Tested layers: 0 (KDA + dense MLP), 3 (MLA / DSA + MoE) and 4 (KDA + MoE). The decode tests prefill a prefix whose
length is not a multiple of the indexer pool size, then feed the remaining tokens one decode step (a 32-row chunk
with one real row) at a time, against the same full-sequence HF forward.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import pytest
import torch

from models.experimental.glm53_flash.tests.test_kda_block import bf4_round_experts, pcc
from models.experimental.glm53_flash.tt.config import Glm5NextTextConfig
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.common import ChunkInfo, TPContext
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.decoder_layer import DecoderLayer
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.model import (
    DECODE_ROWS,
    streams_from_host,
    streams_to_host,
)
from models.experimental.glm53_flash.tt.pipeline_parallel_prefill.weights import LayerSource

pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
from checkpoint import default_snapshot_dir, load_embedding, load_layer_raw, shard_headers  # noqa: E402
from hf_layer_reference import build_layer, load_config  # noqa: E402

try:
    SNAPSHOT = default_snapshot_dir()
except FileNotFoundError:
    SNAPSHOT = None

pytestmark = pytest.mark.skipif(SNAPSHOT is None, reason="GLM-5.3-Flash checkpoint not available")

HC, HIDDEN = 4, 4096
TESTED = (0, 3, 4)
SEQ = 300
CHUNK = 128
CACHE = Path(os.environ.get("GLM_PP_CACHE", Path.home() / ".cache/ttnn/models/glm53_flash_pp"))
PROMPT = (
    "Summarize the history of the printing press, from Gutenberg's movable type through the industrial steam presses "
    "of the nineteenth century, and explain how each advance changed the cost of books, literacy rates, and the spread "
    "of scientific ideas across Europe. Then compare that transition with the arrival of the internet, listing three "
    "similarities and three differences, and finish with a short paragraph on what the comparison suggests about how "
    "new information technologies reshape institutions such as universities, churches, newspapers and governments."
)
ROUTER_TIE_MARGIN = 2e-3
# (output PCC min, median; residual-update PCC min, median) per reference
THRESHOLDS = {"implementation": (0.999, 0.9999, 0.97, 0.995), "deployment": (0.99, 0.995, 0.95, 0.985)}


def _hf_forward(layer, x):
    b, s = x.shape[:2]
    with torch.no_grad():
        return layer(
            x,
            attention_mask=torch.ones(b, s, dtype=torch.bool),
            position_ids=torch.arange(s)[None],
            position_embeddings=None,
        )[0]


def _router_ties(layer, x) -> torch.Tensor:
    """[S] tokens whose HF router is within ROUTER_TIE_MARGIN of a top-k boundary (bf16 inputs on device may pick the
    other expert there, which legitimately changes that token's FFN output)."""
    s = x.shape[1]
    if not hasattr(layer.mlp, "experts"):
        return torch.zeros(s, dtype=torch.bool)
    cap = {}
    hook = layer.mlp.register_forward_hook(lambda m, i, o: cap.setdefault("x", i[0].detach()))
    _hf_forward(layer, x)
    hook.remove()
    h = cap["x"].reshape(s, -1).float()
    gate = layer.mlp.gate
    scores = (h @ gate.weight.float().t()).sigmoid() + gate.e_score_correction_bias.float()
    k = layer.mlp.config.num_experts_per_tok
    top = scores.topk(k + 1, dim=-1).values
    return top[:, k - 1] - top[:, k] < ROUTER_TIE_MARGIN


@pytest.fixture(scope="module")
def chain():
    from models.experimental.glm53_flash.tt.tokenizer_chat import load_tokenizer

    headers = shard_headers(SNAPSHOT)
    _, text = load_config(SNAPSHOT)
    tok = load_tokenizer(SNAPSHOT)
    ids = tok.encode_chat([{"role": "user", "content": PROMPT}])
    while len(ids) < SEQ:
        ids = ids + ids
    ids = torch.tensor([ids[:SEQ]])
    x = load_embedding(SNAPSHOT, headers)[ids].float()[:, :, None, :].expand(-1, -1, HC, -1).contiguous()
    out = {}
    for idx in range(max(TESTED) + 1):
        raw = load_layer_raw(SNAPSHOT, idx, headers)
        layer, _ = build_layer(SNAPSHOT, text, idx, headers, raw=raw)
        layer = layer.float()
        y = _hf_forward(layer, x)
        if idx in TESTED:
            refs = {"deployment": y}
            if hasattr(layer.mlp, "experts"):
                bf4_round_experts(layer)
                refs["implementation"] = _hf_forward(layer, x)
            else:
                refs["implementation"] = y
            out[idx] = (x.to(torch.bfloat16).float(), refs, _router_ties(layer, x), raw)
        else:
            del raw
        del layer
        x = y.to(torch.bfloat16).float()  # the model carries bf16 streams
        gc.collect()
    return out


def _run_schedule(layer: DecoderLayer, ctx: TPContext, x: torch.Tensor, steps) -> torch.Tensor:
    """x [1, S, hc, D] -> device layer output [1, S, hc, D]; ``steps`` are consecutive (real rows, chunk rows)."""
    state = layer.new_state()
    outs, start = [], 0
    for n, rows in steps:
        info = ChunkInfo(ctx, start, rows, n)
        streams = streams_from_host(ctx, x[:, start : start + n].reshape(n, HC * HIDDEN), rows)
        y, state = layer.forward(streams, state, info)
        outs.append(streams_to_host(ctx, y)[:n])
        start += n
    assert start == x.shape[1]
    return torch.cat(outs).reshape(1, start, HC, HIDDEN)


def _prefill_steps(n_tokens: int, chunk: int):
    return [(min(chunk, n_tokens - s), chunk) for s in range(0, n_tokens, chunk)]


def _decode_steps(n_prefill: int, n_decode: int, chunk: int):
    """Prefill ``n_prefill`` tokens in chunks (the last padded to a 128 multiple), then ``n_decode`` decode steps."""
    steps = [(n, -(-n // 128) * 128) for n, _ in _prefill_steps(n_prefill, chunk)]
    return steps + [(1, DECODE_ROWS)] * n_decode


def _run_chunks(layer: DecoderLayer, ctx: TPContext, x: torch.Tensor, chunk: int) -> torch.Tensor:
    """x [1, S, hc, D] -> device layer output [1, S, hc, D], prefilled ``chunk`` rows at a time."""
    return _run_schedule(layer, ctx, x, _prefill_steps(x.shape[1], chunk))


def _grade(got, refs, inp, ties, what):
    """Per-token output and residual-update PCC. Minima are taken over tokens off a router near-tie (where bf16
    rounding may legitimately flip one of the top-8 experts); the all-token minimum and the worst token are printed."""
    failures = []
    for name, ref in refs.items():
        s = got.shape[1]
        out = torch.tensor([pcc(got[0, t], ref[0, t]) for t in range(s)])
        upd = torch.tensor([pcc(got[0, t] - inp[0, t], ref[0, t] - inp[0, t]) for t in range(s)])
        worst = int(out.argmin())
        bounds = THRESHOLDS[name]
        print(
            f"\n{what} vs {name}: output PCC (off-tie) min {out[~ties].min():.5f} median {out.median():.6f} "
            f"[all-token min {out.min():.5f} at token {worst}{' (router near-tie)' if ties[worst] else ''}]; "
            f"update PCC (off-tie) min {upd[~ties].min():.4f} median {upd[~ties].median():.5f}; "
            f"router near-ties {int(ties.sum())}/{s}"
        )
        for value, bound, label in zip(
            (out[~ties].min(), out.median(), upd[~ties].min(), upd[~ties].median()),
            bounds,
            ("min output", "median output", "min update", "median update"),
        ):
            if not value > bound:
                failures.append(f"{name}: {label} PCC {value:.5f} <= {bound}")
    assert not failures, f"{what}: " + "; ".join(failures)


LONG_SEQ, LONG_CHUNK, DSA_LAYER = 2600, 1024, 3


@pytest.fixture(scope="module")
def long_chain():
    """Layer 3's real input / HF outputs for a LONG_SEQ-token prompt: queries past position 2050 see more than
    index_topk tokens, so the DSA indexer's pool selection decides their attention."""
    from models.experimental.glm53_flash.tt.tokenizer_chat import load_tokenizer

    headers = shard_headers(SNAPSHOT)
    _, text = load_config(SNAPSHOT)
    tok = load_tokenizer(SNAPSHOT)
    parts = [tok.encode_chat([{"role": "user", "content": f"Part {i}. {PROMPT}"}]) for i in range(64)]
    ids = torch.tensor([sum(parts, [])[:LONG_SEQ]])
    assert ids.shape[1] == LONG_SEQ
    x = load_embedding(SNAPSHOT, headers)[ids].float()[:, :, None, :].expand(-1, -1, HC, -1).contiguous()
    for idx in range(DSA_LAYER + 1):
        raw = load_layer_raw(SNAPSHOT, idx, headers)
        layer, _ = build_layer(SNAPSHOT, text, idx, headers, raw=raw)
        layer = layer.float()
        if idx == DSA_LAYER:
            bf4_round_experts(layer)
            y = _hf_forward(layer, x)
            out = (x.to(torch.bfloat16).float(), {"implementation": y}, _router_ties(layer, x), raw)
        else:
            x = _hf_forward(layer, x).to(torch.bfloat16).float()
        del layer
        gc.collect()
    return out


def test_dsa_layer_long_prefill_matches_hf(stage, long_chain):
    cfg = Glm5NextTextConfig.from_json(str(SNAPSHOT))
    ctx = TPContext(stage)
    x, refs, ties, raw = long_chain
    layer = DecoderLayer(ctx, cfg, DSA_LAYER, LayerSource(DSA_LAYER, raw=raw), cache_dir=CACHE)
    got = _run_chunks(layer, ctx, x, LONG_CHUNK)
    assert torch.isfinite(got).all()
    sparse = slice(cfg.index_topk + cfg.index_kpool - 1, LONG_SEQ)  # queries whose selection is sparse
    _grade(got, refs, x, ties, f"layer {DSA_LAYER} ({LONG_SEQ} tokens, chunks of {LONG_CHUNK})")
    _grade(got[:, sparse], {k: v[:, sparse] for k, v in refs.items()}, x[:, sparse], ties[sparse], "DSA-sparse queries")


@pytest.mark.parametrize("idx", TESTED, ids=["kda_dense_L0", "mla_moe_L3", "kda_moe_L4"])
def test_layer_prefill_matches_hf(stage, chain, idx):
    cfg = Glm5NextTextConfig.from_json(str(SNAPSHOT))
    ctx = TPContext(stage)
    x, refs, ties, raw = chain[idx]
    layer = DecoderLayer(ctx, cfg, idx, LayerSource(idx, raw=raw), cache_dir=CACHE)
    got = _run_chunks(layer, ctx, x, CHUNK)
    assert torch.isfinite(got).all()
    _grade(got, refs, x, ties, f"layer {idx} ({SEQ} tokens, chunks of {CHUNK})")
    del layer
    gc.collect()


N_DECODE = 13  # SEQ - N_DECODE = 287 prefilled: three tokens of an incomplete indexer pool carry into decode
LONG_DECODE = 10


@pytest.mark.parametrize("idx", TESTED, ids=["kda_dense_L0", "mla_moe_L3", "kda_moe_L4"])
def test_layer_decode_matches_hf(stage, chain, idx):
    cfg = Glm5NextTextConfig.from_json(str(SNAPSHOT))
    ctx = TPContext(stage)
    x, refs, ties, raw = chain[idx]
    layer = DecoderLayer(ctx, cfg, idx, LayerSource(idx, raw=raw), cache_dir=CACHE)
    got = _run_schedule(layer, ctx, x, _decode_steps(SEQ - N_DECODE, N_DECODE, CHUNK))
    assert torch.isfinite(got).all()
    dec = slice(SEQ - N_DECODE, SEQ)
    _grade(got, refs, x, ties, f"layer {idx} (prefill {SEQ - N_DECODE} + decode {N_DECODE})")
    _grade(got[:, dec], {k: v[:, dec] for k, v in refs.items()}, x[:, dec], ties[dec], f"layer {idx} decode steps")
    del layer
    gc.collect()


def test_dsa_layer_long_decode_matches_hf(stage, long_chain):
    """Decode steps whose DSA selection is sparse (positions > index_topk + kpool - 2)."""
    cfg = Glm5NextTextConfig.from_json(str(SNAPSHOT))
    ctx = TPContext(stage)
    x, refs, ties, raw = long_chain
    layer = DecoderLayer(ctx, cfg, DSA_LAYER, LayerSource(DSA_LAYER, raw=raw), cache_dir=CACHE)
    n_prefill = LONG_SEQ - LONG_DECODE
    got = _run_schedule(layer, ctx, x, _decode_steps(n_prefill, LONG_DECODE, LONG_CHUNK))
    assert torch.isfinite(got).all()
    dec = slice(n_prefill, LONG_SEQ)
    _grade(got, refs, x, ties, f"layer {DSA_LAYER} (prefill {n_prefill} + decode {LONG_DECODE})")
    _grade(got[:, dec], {k: v[:, dec] for k, v in refs.items()}, x[:, dec], ties[dec], "DSA-sparse decode steps")
