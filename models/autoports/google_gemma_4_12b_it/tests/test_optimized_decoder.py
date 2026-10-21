# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-vs-TTNN tests for the Gemma4 12B optimized decoder (``tt/optimized_decoder.py``), both layer kinds.

Part 1 re-runs the functional-decoder acceptance suite on ``OptimizedDecoder``: lengths, continuation,
traced decode at batch 1/12/32, determinism, the runtime fallback audit, all 48 layers on real
activations, and the odd-width page table. ``test_functional_decoder.build_decoder`` is patched to
build (and assert) an ``OptimizedDecoder`` with its default policy, at ``prefill_chunk_size=2048`` so the
suite's chunk boundaries stay meaningful. Its trace helper is patched too: the persistent decode input
lives in the decoder's residual layout (``decode_input_memory_config``), as inside a layer stack.

Part 2 covers what is specific to the optimized layer:
- the default 8192-row chunk: its boundaries, continuation across them, and the 262144 full context;
- optimized-vs-fused agreement on identical inputs;
- an op-graph and dtype check that the optimized ops, layouts and weight dtypes are the ones that run;
- a long traced-decode stress run with repeated prefills.
Every HF comparison scores the whole tensor and every logical row (PCC >= 0.995), except two diagnostic groups
recorded with ``diagnostic_bar``: synthetic-activation sliding prefill rows of >= 8192 tokens (0.99) and TT-vs-TT
chunking rows (0.998), each backed by real-activation acceptance checks (see
``SYNTHETIC_SLIDING_PREFILL_DIAG_BAR`` and ``TT_CHUNKING_DIAG_BAR``).

``GEMMA4_OPT_POLICY`` (JSON, dtypes as "bf16"/"bfp8"/"bfp4") overrides the policy for experiments only. The
delivered evidence (``doc/optimized_decoder/evidence.jsonl``) is from runs without it, i.e. the default policy.
"""

import json
import os

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests import test_functional_decoder as fd
from models.autoports.google_gemma_4_12b_it.tests.harness import (
    BLOCK,
    PCC_BAR,
    Pages,
    canary_is_clean,
    check,
    from_tt,
    record,
    to_tt,
    to_tt_int,
)
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    LAYER_KINDS,
    HFLayer,
    real_layer_state_dict,
    synthetic_hidden,
)
from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import FusedDecoder
from models.autoports.google_gemma_4_12b_it.tt.optimized_decoder import DEFAULT_PREFILL_CHUNK, OptimizedDecoder

KINDS = list(LAYER_KINDS)
SUITE_CHUNK = 2048  # chunk used when re-running the functional suite (its boundaries assume 2048)
_chunk = {"value": SUITE_CHUNK}
_DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b}


def env_policy():
    p = json.loads(os.environ.get("GEMMA4_OPT_POLICY", "{}"))
    if "dtype" in p:
        p["dtype"] = {k: _DT[v] for k, v in p["dtype"].items()}
    if "kv_dtype" in p:
        p["kv_dtype"] = _DT[p["kv_dtype"]]
    return p


class TracedDecode(fd.TracedDecode):
    """fd.TracedDecode with the persistent hidden-state input in the decoder's residual layout."""

    def __init__(self, decoder, device, batch):
        super().__init__(decoder, device, batch)
        if hasattr(decoder, "decode_input_memory_config"):
            self.x.deallocate(True)
            self.x = ttnn.from_torch(
                torch.zeros(1, 1, batch, decoder.hidden_size),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=decoder.decode_input_memory_config(batch),
            )


def build_optimized(device, cfg, layer_idx, state_dict, num_blocks, **kwargs):
    kwargs.setdefault("prefill_chunk_size", _chunk["value"])
    kwargs.setdefault("policy", env_policy())
    dec = OptimizedDecoder.from_state_dict(
        state_dict, hf_config=cfg, layer_idx=layer_idx, mesh_device=device, num_blocks=num_blocks, **kwargs
    )
    assert type(dec) is OptimizedDecoder
    return dec


@pytest.fixture(autouse=True)
def _optimized_builder(monkeypatch):
    """Every reused functional test builds the optimized layer and traces it with residual-layout input."""
    monkeypatch.setattr(fd, "build_decoder", build_optimized)
    monkeypatch.setattr(fd, "TracedDecode", TracedDecode)


cfg = fd.cfg

# ---------------------------------------------------------------- part 1: functional suite on OptimizedDecoder
test_rope_tables_match_hf = fd.test_rope_tables_match_hf
test_traced_decode = fd.test_traced_decode
test_prefill_determinism = fd.test_prefill_determinism
test_runtime_fallback_audit = fd.test_runtime_fallback_audit
test_all_layers_real_activations = fd.test_all_layers_real_activations


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("kind", KINDS)
def test_decode_odd_width_page_table(device_1x1, cfg, kind, batch):
    """Decode in the last block of a 17-block page-table row (see fd.test_decode_odd_width_page_table).

    The functional contract kept decode K chunks at one cache block because the decode SDPA op rounds the
    causal K range up to whole K chunks and reads one page-table entry past an odd-width row. The optimized
    decoder uses 128-token K chunks at batch 1 and pads the row (block 0) to a whole number of K chunks
    inside ``decode_forward``. This pins both: the padded width covers the rounded range and decode at the
    last positions of the row matches HF, at batch 1 (128-token chunks) and 32 (one block).
    """
    layer_idx = LAYER_KINDS[kind]
    sd = fd.synthetic_layer_state_dict_cached(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    dec = build_optimized(device_1x1, cfg, layer_idx, sd, num_blocks=17 * batch + 1, max_context=8192)
    k_chunk, _ = dec._sdpa_decode_cfg(batch)
    assert k_chunk % dec.block_size == 0
    gen = torch.Generator().manual_seed(layer_idx)
    table = (torch.randperm(17 * batch, generator=gen) + 1).int().reshape(batch, 17)
    tt_table = to_tt_int(table, device_1x1)
    padded = dec._table_for_k_chunk(tt_table, k_chunk)
    assert (padded.shape[-1] * dec.block_size) % k_chunk == 0
    record(f"odd_width_table[{kind},b{batch}]", k_chunk=k_chunk, width=17, padded_width=padded.shape[-1])
    xs = [synthetic_hidden(layer_idx, 1026, seed=4 + u) for u in range(batch)]
    caches = []
    for u in range(batch):
        rcache = ref.new_cache()
        ref.prefill(xs[u][:, :1024], rcache)
        caches.append(rcache)
        dec.prefill_forward(to_tt(xs[u][:, :1024], device_1x1), page_table=tt_table, user_id=u)
    for p in (1024, 1025):
        x = torch.cat([xs[u][:, p : p + 1] for u in range(batch)], 1)
        out = from_tt(
            dec.decode_forward(
                to_tt(x, device_1x1),
                current_pos=to_tt_int(torch.full((batch,), p), device_1x1),
                page_table=tt_table,
            )
        )
        for u in range(batch):
            want = ref.forward(xs[u][:, p : p + 1], caches[u], p)
            check(f"odd_width_page_table_decode[{kind},b{batch}]", want, out[:, u : u + 1], pos=p, user=u)
    assert canary_is_clean(dec)


# Sliding-layer prefill SDPA uses the op's streaming kernel (BF16 dest). On synthetic hidden states, which are
# i.i.d. Gaussian per channel and give near-uniform attention over long query blocks, the 8192-token default-chunk
# sliding row drops to 0.9945. The same lengths on real activations hold >= 0.9974 on layers 0-7 and 46
# (README "Prefill SDPA"; test_real_activation_long_prefill below). Per the optimize skill (OPT-012) synthetic-
# activation sliding prefill rows of >= 8192 tokens are diagnostics with a 0.99 bar; shorter synthetic rows, every
# other check, and every real-activation check keep the 0.995 bar.
SYNTHETIC_SLIDING_PREFILL_DIAG_BAR = 0.99
SYNTHETIC_SLIDING_PREFILL_DIAG_MIN_LEN = 8192  # shorter synthetic rows keep the 0.995 bar
_SYNTHETIC_SLIDING_PREFILL = (
    "prefill[sliding,real]",  # real weights, synthetic activations
    "prefill[sliding,synthetic]",
    "default_chunk_prefill[sliding]",
    "default_chunk_continuation[sliding]",
)


# TT-vs-TT chunking consistency (continuation vs chunked, chunked vs unchunked). In the functional and fused
# decoders both chunkings ran identical math, so these sat near 1.0 under a 0.999 bar. Here chunks of <= 1280 rows
# (1024 on BFP4-MLP layers) run the 2D-multicast matmul and longer chunks minimal_matmul with BFP8 MLP
# intermediates, so two chunkings of the same prompt legitimately differ by ~1e-3. Both sides are scored
# against HF at the 0.995 bar in the same test (and on real activations in test_real_activation_long_prefill); the
# TT-vs-TT rows are diagnostics with a 0.998 bar.
TT_CHUNKING_DIAG_BAR = 0.998
_TT_CHUNKING = ("continuation_vs_chunked", "chunked_vs_unchunked", "real_chunk_size_consistency")


def _diagnostic_check(name, want, got, bar=PCC_BAR, **kwargs):
    if name in _SYNTHETIC_SLIDING_PREFILL and kwargs.get("seq_len", 0) >= SYNTHETIC_SLIDING_PREFILL_DIAG_MIN_LEN:
        bar = min(bar, SYNTHETIC_SLIDING_PREFILL_DIAG_BAR)
        kwargs["diagnostic_bar"] = bar
    elif name.startswith(_TT_CHUNKING):
        bar = min(bar, TT_CHUNKING_DIAG_BAR)
        kwargs["diagnostic_bar"] = bar
    return check(name, want, got, bar=bar, **kwargs)


@pytest.mark.parametrize("kind", KINDS)
def test_chunked_prefill_continuation(device_1x1, cfg, kind, monkeypatch):
    """fd.test_chunked_prefill_continuation; the TT-vs-TT chunking rows use the diagnostic bar (HF rows keep 0.995)."""
    monkeypatch.setattr(fd, "check", _diagnostic_check)
    fd.test_chunked_prefill_continuation(device_1x1, cfg, kind)


@pytest.mark.parametrize("weights", fd.WEIGHTS)
@pytest.mark.parametrize("kind", KINDS)
def test_prefill_decode_lengths(device_1x1, cfg, kind, weights, monkeypatch):
    """fd.test_prefill_decode_lengths; synthetic-activation sliding prefill rows use the diagnostic bar."""
    monkeypatch.setattr(fd, "check", _diagnostic_check)
    fd.test_prefill_decode_lengths(device_1x1, cfg, kind, weights)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("layer_idx", [0, 1, 5])
def test_real_activation_long_prefill(device_1x1, cfg, layer_idx):
    """Long prefill on real activations at the 0.995 bar: the disputed lengths of the synthetic diagnostics.

    The HF float32 stack runs a 17388-token real prompt (repository text, chat-formatted) up to the layer. Each
    request prefills a prefix of it into fresh pages: 2049 (one chunk, two sliding query blocks), 8192 (one full
    default chunk) and 17384 (two chunks + a tail; sliding layers only), then decodes the next two tokens. Every
    row is scored against HF (one causal HF prefill gives every reference row). The 8192-token prompt is also
    prefilled through 1024-row chunks (the short-chunk 2D-multicast matmul path with BF16 MLP intermediates),
    scored against HF and, as a diagnostic, against the single-chunk result.
    """
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    sliding = cfg.layer_types[layer_idx] == "sliding_attention"
    lengths = [2049, 8192, 17384] if sliding else [2049, 8192]
    x = _real_prompt_hidden(cfg, layer_idx, max(lengths) + 2)
    want = ref.prefill(x, ref.new_cache(), slice_len=2048)
    max_pages = (max(lengths) + 2 + 63) // BLOCK
    dec = build_optimized(
        device_1x1, cfg, layer_idx, sd, num_blocks=max_pages + 1, prefill_chunk_size=DEFAULT_PREFILL_CHUNK
    )
    pages = Pages(dec.num_blocks, max_pages, num_users=1, seed=layer_idx)
    for n in lengths:
        pages.release(0)
        pages.ensure(0, n + 2)
        table = pages.tt(device_1x1)
        got = from_tt(dec.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=table))
        check(f"real_long_prefill[{layer_idx}]", want[:, :n], got, seq_len=n)
        for p in (n, n + 1):
            out = dec.decode_forward(
                to_tt(x[:, p : p + 1], device_1x1),
                current_pos=to_tt_int(torch.tensor([p]), device_1x1),
                page_table=table,
            )
            check(f"real_long_decode[{layer_idx}]", want[:, p : p + 1], from_tt(out), pos=p)
    # The same 8192-token prompt through 1024-row chunks (2D-multicast matmuls, 8 chunks) and one 8192-row chunk
    # (minimal_matmul, BFP8 MLP intermediates): each against HF at the bar, and against each other as a
    # diagnostic at the TT-vs-TT chunking bar.
    n, small_chunk = 8192, 1024
    small = build_optimized(device_1x1, cfg, layer_idx, sd, num_blocks=max_pages + 1, prefill_chunk_size=small_chunk)
    pages.release(0)
    pages.ensure(0, n)
    chunked = from_tt(small.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=pages.tt(device_1x1)))
    check(f"real_chunked_prefill[{layer_idx}]", want[:, :n], chunked, seq_len=n, chunk=small_chunk)
    whole = from_tt(dec.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=pages.tt(device_1x1)))
    _diagnostic_check(
        f"real_chunk_size_consistency[{layer_idx}]",
        whole,
        chunked,
        bar=TT_CHUNKING_DIAG_BAR,
        seq_len=n,
        chunks=[small_chunk, n],
    )
    assert canary_is_clean(dec)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("layer_idx", [0, 13])
def test_prefill_q_floor_exact_table(device_1x1, cfg, layer_idx):
    """Sliding prefill at lengths where the SDPA Q-chunk floor pads Q inside the op, with an exact-width page table.

    Rows that are 64 mod 128 above 512 rows run 128-row Q chunks with the op padding the last one
    (``sdpa_prefill_q_floor``): 575 (non-aligned, 576 rows, 2D matmul path), 1088 (2D on layer 0,
    ``minimal_matmul`` on the BFP4-MLP layer 13), 1300 (non-aligned, 1344 rows, ``minimal_matmul``) and 5184 (query
    blocks 2048 + 2048 + 1088). Each request gets a fresh page-table row exactly as wide as its blocks, so any read past
    the row would hit the canary block. Real activations, scored against HF at the bar; decode of the next token too.
    """
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    lengths = [575, 1088, 1300, 5184]
    x = _real_prompt_hidden(cfg, layer_idx, max(lengths) + 1)
    want = ref.prefill(x, ref.new_cache(), slice_len=2048)
    for n in lengths:
        pages_needed = -(-(n + 1) // BLOCK)
        dec = build_optimized(device_1x1, cfg, layer_idx, sd, num_blocks=pages_needed + 1)
        padded = -(-n // BLOCK) * BLOCK
        q_chunk = dec._sdpa_prefill_cfg(0, padded % dec.sliding_attn_block or dec.sliding_attn_block).q_chunk_size
        assert q_chunk == 128, (n, q_chunk)
        pages = Pages(dec.num_blocks, pages_needed, num_users=1, seed=n)
        pages.ensure(0, n + 1)
        table = pages.tt(device_1x1)
        got = from_tt(dec.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=table))
        check(f"q_floor_exact_table_prefill[{layer_idx}]", want[:, :n], got, seq_len=n, q_chunk=q_chunk)
        out = dec.decode_forward(
            to_tt(x[:, n : n + 1], device_1x1), current_pos=to_tt_int(torch.tensor([n]), device_1x1), page_table=table
        )
        check(f"q_floor_exact_table_decode[{layer_idx}]", want[:, n : n + 1], from_tt(out), pos=n)
        assert canary_is_clean(dec)


# ---------------------------------------------------------------- part 2: optimized-specific coverage
@pytest.mark.parametrize("kind", KINDS)
def test_full_context(device_1x1, cfg, kind, monkeypatch):
    """HF-advertised context 262144 at the default chunk (8192): see fd.test_full_context."""
    monkeypatch.setitem(_chunk, "value", DEFAULT_PREFILL_CHUNK)
    fd.test_full_context(device_1x1, cfg, kind)


DEFAULT_CHUNK_LENGTHS = [8191, 8192, 8193, 16384 + 1000, 5000, 700]


@pytest.mark.parametrize("kind", KINDS)
def test_default_chunk_boundaries(device_1x1, cfg, kind):
    """Default 8192-row chunk: lengths around one and two chunks plus a short reuse, then continuation.

    Same cases as the fused stage: 8192 = 4 sliding query blocks, 5000 -> 2048 + 2048 + 960 (padded tail
    block), 17384 -> two chunks + a 1000-row (padded to 1024) chunk. Real weights. Each request gets fresh
    pages and two decode steps; the continuation starts at 4160 and crosses the 8192 chunk boundary.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    max_pages = 18432 // BLOCK
    dec = build_optimized(
        device_1x1, cfg, layer_idx, sd, num_blocks=2 * max_pages + 1, prefill_chunk_size=DEFAULT_PREFILL_CHUNK
    )
    pages = Pages(dec.num_blocks, max_pages, num_users=2, seed=31 + layer_idx)
    steps = 2
    for i, length in enumerate(DEFAULT_CHUNK_LENGTHS):
        pages.release(0)
        pages.ensure(0, length + steps)
        table = pages.tt(device_1x1, rows=slice(0, 1))
        x = synthetic_hidden(layer_idx, length + steps, seed=200 + i)
        cache = ref.new_cache()
        want = ref.prefill(x[:, :length], cache, slice_len=2048)
        got = from_tt(dec.prefill_forward(to_tt(x[:, :length], device_1x1), page_table=table))
        _diagnostic_check(f"default_chunk_prefill[{kind}]", want, got, seq_len=length, chunk=DEFAULT_PREFILL_CHUNK)
        for s in range(steps):
            p = length + s
            want = ref.forward(x[:, p : p + 1], cache, p)
            out = dec.decode_forward(
                to_tt(x[:, p : p + 1], device_1x1),
                current_pos=to_tt_int(torch.tensor([p]), device_1x1),
                page_table=table,
            )
            check(f"default_chunk_decode[{kind}]", want, from_tt(out), pos=p)

    length = 12000
    x = synthetic_hidden(layer_idx, length, seed=300)
    cache = ref.new_cache()
    want = ref.prefill(x, cache, slice_len=2048)
    pages.ensure(1, length)
    table = pages.tt(device_1x1)
    parts = []
    for a, b in ((0, 4160), (4160, length)):
        parts.append(
            from_tt(dec.prefill_forward(to_tt(x[:, a:b], device_1x1), page_table=table, user_id=1, start_pos=a))
        )
    _diagnostic_check(
        f"default_chunk_continuation[{kind}]", want, torch.cat(parts, dim=1), seq_len=length, starts=[0, 4160]
    )
    assert canary_is_clean(dec)


@pytest.mark.parametrize("kind", KINDS)
def test_optimized_matches_fused(device_1x1, cfg, kind):
    """Identical inputs, pages and positions through FusedDecoder and OptimizedDecoder (real weights).

    Prefill lengths cover a tile tail, the 2048 suite chunk and the 8192 default chunk; traced decode runs
    at batch 1 and 32 on the filled caches. The optimized layer changes weight/KV precision and fidelity,
    so the two are not bit-exact; they must agree at PCC >= 0.995 on every row (the HF bar).
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    batch = 32
    max_pages = 9216 // BLOCK
    shared = {}
    kw = dict(hf_config=cfg, layer_idx=layer_idx, mesh_device=device_1x1, num_blocks=batch * max_pages + 1)
    fus = FusedDecoder.from_state_dict(sd, shared=shared, **kw)
    opt = OptimizedDecoder.from_state_dict(sd, shared=shared, policy=env_policy(), **kw)
    pages = Pages(fus.num_blocks, max_pages, num_users=batch, seed=41)
    lengths = [100, 2049, 8193] + [
        int(n) for n in torch.randint(1, 1300, (batch - 3,), generator=torch.Generator().manual_seed(5))
    ]
    for u, n in enumerate(lengths):
        pages.ensure(u, n + 1)
    table = pages.tt(device_1x1)
    for u, n in enumerate(lengths):
        x = to_tt(synthetic_hidden(layer_idx, n, seed=400 + u), device_1x1)
        a = from_tt(fus.prefill_forward(x, page_table=table, user_id=u))
        b = from_tt(opt.prefill_forward(x, page_table=table, user_id=u))
        if u < 3:
            check(f"optimized_vs_fused_prefill[{kind}]", a, b, seq_len=n)
    for b_ in (1, batch):
        xs = torch.cat([synthetic_hidden(layer_idx, 1, seed=900 + u) for u in range(b_)], 1)
        pos = torch.tensor(lengths[:b_])
        rows = pages.table[:b_]
        outs = []
        for dec in (fus, opt):
            tr = TracedDecode(dec, device_1x1, b_)
            tr.set_inputs(xs, pos, rows)
            tr.capture()
            outs.append(from_tt(tr.replay()))
            tr.release()
        check(f"optimized_vs_fused_traced_decode[{kind},b{b_}]", outs[0], outs[1], batch=b_)


def _device_ops(fn):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    fn()
    graph = ttnn.graph.end_graph_capture()
    names = [n["params"]["name"] for n in graph if n.get("node_type") == "function_start"]
    return [n for n in names if n.endswith("DeviceOperation") or n == "SDPAOperation"]


@pytest.mark.parametrize("kind", KINDS)
def test_optimized_op_graph(device_1x1, cfg, kind):
    """The optimized ops, layouts and dtypes are the ones that run (no fused/functional fallback).

    Weights: every projection is width-sharded over the DRAM banks in its policy dtype; the KV cache
    is in the policy dtype. Prefill: five MinimalMatmuls for a 2100-row chunk (sliding QKV is the 4-way
    split), 2D-multicast matmuls for a 1000-row chunk. Decode at
    batch 1 and 32 with residual-layout input: five DRAM-sharded matmuls, the residual and every
    residual-grid norm/add output stay width-sharded in L1, no interleaved<->sharded conversion of the
    residual stream, no tilize. Records the device-op sequences as evidence.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = fd.synthetic_layer_state_dict_cached(layer_idx)
    dec = build_optimized(
        device_1x1, cfg, layer_idx, sd, num_blocks=32 * 40 + 1, max_context=8192, prefill_chunk_size=8192
    )
    p = dec.policy
    for role, w in (("qkv", dec.wqkv), ("o", dec.wo), ("gate", dec.w_gate), ("up", dec.w_up), ("down", dec.w_down)):
        mc = w.memory_config()
        assert w.dtype == p["dtype"][role], role
        assert mc.buffer_type == ttnn.BufferType.DRAM and mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    assert all(c.dtype == p["kv_dtype"] for c in dec.kv_cache)
    pages = Pages(dec.num_blocks, 40, num_users=32)
    for u in range(32):
        pages.ensure(u, 2101)
    x = to_tt(synthetic_hidden(layer_idx, 2100), device_1x1)
    table = pages.tt(device_1x1)
    dec.prefill_forward(x, page_table=table)  # warm
    pre = _device_ops(lambda: dec.prefill_forward(x, page_table=table))
    record(f"opt_op_graph[{kind}]", path="prefill", batch=1, num_device_ops=len(pre), ops=pre)
    # 2100 rows (> prefill_linear_max_rows): every projection on minimal_matmul (sliding QKV: the 4-way split)
    # (minimal_matmul_split runs as a MinimalMatmulDeviceOperation)
    assert "MatmulDeviceOperation" not in pre and pre.count("MinimalMatmulDeviceOperation") == 5
    # a short chunk (<= prefill_linear_max_rows) runs the 2D-multicast linear instead (sliding QKV keeps the split)
    xs = to_tt(synthetic_hidden(layer_idx, 1000), device_1x1)
    dec.prefill_forward(xs, page_table=table)  # warm
    short = _device_ops(lambda: dec.prefill_forward(xs, page_table=table))
    record(f"opt_op_graph[{kind}]", path="prefill_short", batch=1, num_device_ops=len(short), ops=short)
    assert short.count("MatmulDeviceOperation") == (4 if dec.meta_rope else 5)
    assert short.count("MinimalMatmulDeviceOperation") == (1 if dec.meta_rope else 0)
    for batch in (1, 32):
        tr = TracedDecode(dec, device_1x1, batch)
        tr.set_inputs(synthetic_hidden(layer_idx, batch, seed=1), torch.full((batch,), 2100), pages.table[:batch])
        dec.decode_forward(tr.x, current_pos=tr.pos, page_table=tr.page_table)  # warm
        holder = {}
        ops = _device_ops(
            lambda: holder.setdefault("out", dec.decode_forward(tr.x, current_pos=tr.pos, page_table=tr.page_table))
        )
        out = holder["out"]
        record(f"opt_op_graph[{kind}]", path="decode", batch=batch, num_device_ops=len(ops), ops=ops)
        assert out.memory_config() == dec.decode_input_memory_config(batch)
        assert ops.count("MatmulDeviceOperation") == 5
        assert ops.count("PagedFusedUpdateCacheDeviceOperation") == 1
        assert not any("Tilize" in op for op in ops)
        assert ops.count("LayerNormDeviceOperation") >= 4
        tr.release()


def _real_prompt_hidden(cfg, layer_idx, tokens):
    from models.autoports.google_gemma_4_12b_it.tests.hf_reference import real_hidden, real_prompt_ids

    return real_hidden(cfg, layer_idx, real_prompt_ids(tokens))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("layer_idx", [13, 17, 24, 25, 36])
def test_bfp4_group_traced_decode(device_1x1, cfg, layer_idx, batch):
    """Layers with per-layer BFP4 exceptions on real activations: prefill, then repeated traced decode vs HF.

    Layer 13 (sliding) and 17 (full) run BFP4 attention and MLP weights with 2-3 DRAM readers per bank on a
    60-core residual; layer 24 BFP4 MLP + BFP4 QKV; layer 25 BFP4 gate/up + BFP4 attention; layer 36 BFP4 gate/up +
    BFP4 QKV. Inputs are the HF float32 stack's real
    activations of a chat prompt. Every user prefills a different prefix of it (one HF causal prefill gives
    every reference row). Then the trace is replayed on the prompt's next tokens with positions and pages
    refreshed between replays: 2 replays at batch 1; 120 at batch 32 (crossing page edges), scored on the
    first two and every 20th replay for every user.
    """
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    steps, every = (2, 1) if batch == 1 else (120, 20)
    lengths = [1000 + 5 * u for u in range(batch)] if batch > 1 else [1086]
    x = _real_prompt_hidden(cfg, layer_idx, max(lengths) + steps + 1)
    want = ref.prefill(x, ref.new_cache())
    max_pages = (max(lengths) + steps + 64) // BLOCK
    dec = build_optimized(device_1x1, cfg, layer_idx, sd, num_blocks=batch * max_pages + 1)
    assert dec.policy["dtype"]["gate"] == ttnn.bfloat4_b
    pages = Pages(dec.num_blocks, max_pages, num_users=batch, seed=layer_idx)
    for u, n in enumerate(lengths):
        pages.ensure(u, n + 1)
        got = from_tt(dec.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=pages.tt(device_1x1), user_id=u))
        if u == 0:
            check(f"bfp4_group_prefill[{layer_idx}]", want[:, :n], got, seq_len=n)
    tr = TracedDecode(dec, device_1x1, batch)
    for step in range(steps):
        pos = torch.tensor([n + step for n in lengths])
        for u in range(batch):
            pages.ensure(u, int(pos[u]) + 1)
        tr.set_inputs(torch.cat([x[:, int(p) : int(p) + 1] for p in pos], 1), pos, pages.table)
        if step == 0:
            tr.capture()
        out = tr.replay()
        if step >= 2 and step % every != every - 1:
            continue
        got = from_tt(out)
        for u in range(batch):
            p = int(pos[u])
            check(
                f"bfp4_group_traced_decode[{layer_idx},b{batch}]",
                want[:, p : p + 1],
                got[:, u : u + 1],
                user=u,
                pos=p,
                step=step,
            )
    tr.release()
    assert canary_is_clean(dec)


@pytest.mark.parametrize("batch", [1, 32])
def test_layer_stack_cross_group(device_1x1, cfg, batch):
    """Two chained layers across a residual-grid boundary, traced, vs chained HF layers (real activations).

    Layer 12 (BFP8, 15-core residual) feeds layer 13 (BFP4, 60-core residual): layer 13 reshards its input on
    entry, as in a full stack. Prefill chains the two layers' TT outputs; the decode trace chains them in one
    capture. The same step is captured twice: once with every layer computing its own RoPE rows and once with
    rows computed once per step and shared through ``decode_rope_rows`` / ``rope_rows=``; the two outputs must
    be bitwise equal, and both must match HF.
    """
    layers = (12, 13)
    sds = {li: real_layer_state_dict(li) for li in layers}
    refs = [HFLayer(cfg, li, sds[li]) for li in layers]
    lengths = [1000 + 5 * u for u in range(batch)] if batch > 1 else [1086]
    x = _real_prompt_hidden(cfg, layers[0], max(lengths) + 2)
    want = x
    for ref in refs:
        want = ref.prefill(want, ref.new_cache())
    max_pages = (max(lengths) + 64) // BLOCK + 1
    shared = {}
    decs = [
        build_optimized(device_1x1, cfg, li, sds[li], num_blocks=batch * max_pages + 1, shared=shared) for li in layers
    ]
    assert decs[0].decode_input_memory_config() != decs[1].decode_input_memory_config()
    pages = Pages(decs[0].num_blocks, max_pages, num_users=batch, seed=12)
    for u, n in enumerate(lengths):
        pages.ensure(u, n + 1)
        h = to_tt(x[:, :n], device_1x1)
        for dec in decs:
            h = dec.prefill_forward(h, page_table=pages.tt(device_1x1), user_id=u)
        if u == 0:
            check(f"stack_cross_group_prefill[b{batch}]", want[:, :n], from_tt(h), seq_len=n)
    pos = torch.tensor(lengths)
    tr = TracedDecode(decs[0], device_1x1, batch)
    tr.set_inputs(torch.cat([x[:, int(p) : int(p) + 1] for p in pos], 1), pos, pages.table)

    def step(shared_rows):
        rows = decs[0].decode_rope_rows(tr.pos, batch) if shared_rows else None
        h = tr.x
        for dec in decs:
            h = dec.decode_forward(h, current_pos=tr.pos, page_table=tr.page_table, rope_rows=rows)
        return h

    outs = {}
    for shared_rows in (False, True):
        step(shared_rows)  # compile
        ttnn.synchronize_device(device_1x1)
        tid = ttnn.begin_trace_capture(device_1x1, cq_id=0)
        out = step(shared_rows)
        ttnn.end_trace_capture(device_1x1, tid, cq_id=0)
        ttnn.execute_trace(device_1x1, tid, cq_id=0, blocking=True)
        outs[shared_rows] = ttnn.to_torch(out).clone()
        ttnn.release_trace(device_1x1, tid)
    assert torch.equal(outs[False], outs[True]), "shared RoPE rows change the stack output"
    got = outs[True].float().reshape(1, batch, -1)
    for u in range(batch):
        p = int(pos[u])
        check(f"stack_cross_group_traced_decode[b{batch}]", want[:, p : p + 1], got[:, u : u + 1], user=u, pos=p)
    record(f"stack_cross_group_shared_rope_bitwise[b{batch}]", bitwise_equal=True)
    for dec in decs:
        assert canary_is_clean(dec)


L1_RESERVE_KB = 128  # the fused decoder's own headroom: its prefill fails with 192 KB held


@pytest.mark.parametrize("layer_idx", [0, 5, 13, 17, 21, 24, 25])
def test_l1_headroom(device_1x1, cfg, layer_idx):
    """Every program of the optimized layer still fits with 128 KB of L1 per core held by other tensors.

    A layer stack keeps persistent L1 tensors (decode trace inputs/outputs in the residual layout) while
    it prefills and decodes, so large static circular buffers (prefill matmul blockings, SDPA chunks,
    DRAM-sharded decode CBs) must leave room. With a 128 KB-per-core L1 tensor allocated: prefill of a
    one-chunk (4100), a two-chunk (8193) and two short requests (1000: the 2D-multicast linear path at its
    largest blocks, per_core_M 4; 1280: the last 2D length on BFP8-MLP layers, minimal_matmul on BFP4-MLP
    layers), a continuation at 4160, and traced decode at batch 1
    and 32, for both layer kinds and each per-layer exception configuration (13, 17: all-BFP4; 21: BFP4 MLP with
    BFP8 attention; 24: BFP4 QKV only; 25: BFP4 gate/up + BFP8 down + BFP4 attention). Prefill outputs must be bitwise
    equal to the same requests run without the reserve; layers 0 and 5 (whose synthetic input statistics
    were recorded) are also checked against HF.
    """
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd) if layer_idx in LAYER_KINDS.values() else None
    stats = 0 if cfg.layer_types[layer_idx] == "sliding_attention" else 5
    max_pages = 8256 // BLOCK
    dec = build_optimized(
        device_1x1, cfg, layer_idx, sd, num_blocks=32 * max_pages + 1, prefill_chunk_size=DEFAULT_PREFILL_CHUNK
    )
    pages = Pages(dec.num_blocks, max_pages, num_users=32, seed=layer_idx)
    xs = [synthetic_hidden(stats, n, seed=60 + u) for u, n in enumerate((4100, 8193, 1000, 1280))]
    for u, x in enumerate(xs):
        pages.ensure(u, x.shape[1] + 1)
    free = [
        ttnn.to_torch(dec.prefill_forward(to_tt(x, device_1x1), page_table=pages.tt(device_1x1), user_id=u))
        for u, x in enumerate(xs)
    ]
    cores = device_1x1.compute_with_storage_grid_size().x * device_1x1.compute_with_storage_grid_size().y
    blocker = ttnn.zeros(
        [1, 1, 32, L1_RESERVE_KB * 1024 // 2048 * cores * 32],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device_1x1,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    for u, x in enumerate(xs):
        got = dec.prefill_forward(to_tt(x, device_1x1), page_table=pages.tt(device_1x1), user_id=u)
        assert torch.equal(ttnn.to_torch(got), free[u]), "prefill under the L1 reserve differs"
        if ref is not None:
            want = ref.prefill(x, ref.new_cache(), slice_len=2048)
            check(f"l1_headroom_prefill[{layer_idx}]", want, from_tt(got), seq_len=x.shape[1], reserve_kb=L1_RESERVE_KB)
    x = synthetic_hidden(stats, 6000, seed=70)
    pages.ensure(4, 6001)
    table = pages.tt(device_1x1)
    for a, b in ((0, 4160), (4160, 6000)):
        dec.prefill_forward(to_tt(x[:, a:b], device_1x1), page_table=table, user_id=4, start_pos=a)
    for batch in (1, 32):
        for u in range(5, batch):
            pages.ensure(u, 101)
        tr = TracedDecode(dec, device_1x1, batch)
        pos = torch.tensor([4100, 8193, 1000, 1280, 6000] + [100] * 27)[:batch]
        tr.set_inputs(synthetic_hidden(stats, batch, seed=80), pos, pages.table[:batch])
        tr.capture()
        out = from_tt(tr.replay())
        assert torch.isfinite(out).all()
        tr.release()
    record(f"l1_headroom[{layer_idx}]", reserve_kb=L1_RESERVE_KB, passed=True)
    blocker.deallocate(True)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", KINDS)
def test_stress_repeated(device_1x1, cfg, kind):
    """Long traced-decode run and repeated prefills on one instance (real weights).

    Batch 32: users start at mixed lengths, then 300 trace replays advance every user one token per
    replay, crossing page edges and the sliding window, with positions/pages refreshed between replays.
    Every 60th replay is checked against HF for every user. Afterwards the same three prompts are
    prefilled three times into fresh pages and must be bitwise identical each round.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    batch, replays, every = 32, 300, 60
    max_pages = 2048 // BLOCK
    dec = build_optimized(
        device_1x1, cfg, layer_idx, sd, num_blocks=batch * max_pages + 1, prefill_chunk_size=DEFAULT_PREFILL_CHUNK
    )
    pages = Pages(dec.num_blocks, max_pages, num_users=batch, seed=77)
    g = torch.Generator().manual_seed(17 + layer_idx)
    lengths = torch.randint(1, 1200, (batch,), generator=g).tolist()
    lengths[:3] = [1, 960, 1023]
    xs, caches = [], []
    for u, n in enumerate(lengths):
        x = synthetic_hidden(layer_idx, n + replays, seed=2000 + u)
        xs.append(x)
        cache = ref.new_cache()
        ref.prefill(x[:, :n], cache)
        caches.append(cache)
        pages.ensure(u, n)
        dec.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=pages.tt(device_1x1), user_id=u)
    trace = TracedDecode(dec, device_1x1, batch)
    worst = 1.0
    done = list(lengths)
    for step in range(replays):
        pos = torch.tensor([n + step for n in lengths])
        for u in range(batch):
            pages.ensure(u, int(pos[u]) + 1)
        x = torch.cat([xs[u][:, int(pos[u]) : int(pos[u]) + 1] for u in range(batch)], 1)
        trace.set_inputs(x, pos, pages.table)
        if step == 0:
            trace.capture()
        out = trace.replay()
        if step % every != every - 1:
            continue
        got = from_tt(out)[0]
        for u in range(batch):
            p = int(pos[u])
            ref.inject_prefix_kv(xs[u][:, done[u] : p], caches[u], start_pos=done[u])
            want = ref.forward(xs[u][:, p : p + 1], caches[u], p)
            done[u] = p + 1
            _, w = check(f"stress_traced_decode[{kind}]", want[0], got[u : u + 1], user=u, pos=p, step=step)
            worst = min(worst, w)
    trace.release()
    record(
        f"stress_traced_decode_summary[{kind}]", replays=replays, batch=batch, scored_every=every, worst_row_pcc=worst
    )

    prompts = [synthetic_hidden(layer_idx, n, seed=3000 + n) for n in (37, 1500, 2047)]
    rounds = []
    for r in range(3):
        outs = []
        for i, p in enumerate(prompts):
            pages.release(i)
            pages.ensure(i, p.shape[1])
            outs.append(
                ttnn.to_torch(dec.prefill_forward(to_tt(p, device_1x1), page_table=pages.tt(device_1x1), user_id=i))
            )
        rounds.append(outs)
    for r in (1, 2):
        assert all(torch.equal(a, b) for a, b in zip(rounds[0], rounds[r])), f"prefill round {r} differs"
    record(f"stress_repeated_prefill[{kind}]", rounds=3, lengths=[p.shape[1] for p in prompts], bitwise_equal=True)
    assert canary_is_clean(dec)
