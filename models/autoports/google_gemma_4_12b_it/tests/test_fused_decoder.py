# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-vs-TTNN tests for the Gemma4 12B fused decoder (``tt/fused_decoder.py``), both layer kinds.

Part 1 re-runs the functional-decoder acceptance suite on ``FusedDecoder``: lengths, continuation,
traced decode at batch 1/12/32, determinism, the runtime fallback audit, all 48 layers on real
activations, and the odd-width page table. ``test_functional_decoder.build_decoder`` is patched to
build (and assert) a ``FusedDecoder``. That suite was written around 2048-row chunks, so it builds the
fused layer with ``prefill_chunk_size=2048`` to keep every chunk boundary meaningful.

Part 2 covers what is specific to the fused layer:
- the default 8192-row chunk: its boundaries, continuation across them, and the 262144 full context;
- fused-vs-functional equivalence on identical inputs;
- an op-graph check that the fused ops are the ones that run;
- a long traced-decode stress run with repeated prefills.
Every HF comparison scores the whole tensor and every logical row (PCC >= 0.995).
"""

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests import test_functional_decoder as fd
from models.autoports.google_gemma_4_12b_it.tests.harness import (
    BLOCK,
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
from models.autoports.google_gemma_4_12b_it.tests.test_functional_decoder import TracedDecode
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import FunctionalDecoder
from models.autoports.google_gemma_4_12b_it.tt.fused_decoder import DEFAULT_PREFILL_CHUNK, FusedDecoder

KINDS = list(LAYER_KINDS)
SUITE_CHUNK = 2048  # chunk used when re-running the functional suite (its boundaries assume 2048)
_chunk = {"value": SUITE_CHUNK}


def build_fused(device, cfg, layer_idx, state_dict, num_blocks, **kwargs):
    kwargs.setdefault("prefill_chunk_size", _chunk["value"])
    dec = FusedDecoder.from_state_dict(
        state_dict, hf_config=cfg, layer_idx=layer_idx, mesh_device=device, num_blocks=num_blocks, **kwargs
    )
    assert type(dec) is FusedDecoder
    return dec


@pytest.fixture(autouse=True)
def _fused_builder(monkeypatch):
    """Every reused functional test builds the fused layer."""
    monkeypatch.setattr(fd, "build_decoder", build_fused)


cfg = fd.cfg

# ---------------------------------------------------------------- part 1: functional suite on FusedDecoder
test_rope_tables_match_hf = fd.test_rope_tables_match_hf
test_prefill_decode_lengths = fd.test_prefill_decode_lengths
test_chunked_prefill_continuation = fd.test_chunked_prefill_continuation
test_traced_decode = fd.test_traced_decode
test_prefill_determinism = fd.test_prefill_determinism
test_runtime_fallback_audit = fd.test_runtime_fallback_audit
test_all_layers_real_activations = fd.test_all_layers_real_activations
test_decode_odd_width_page_table = fd.test_decode_odd_width_page_table


# ---------------------------------------------------------------- part 2: fused-specific coverage
@pytest.mark.parametrize("kind", KINDS)
def test_full_context(device_1x1, cfg, kind, monkeypatch):
    """HF-advertised context 262144 at the fused default chunk (8192): see fd.test_full_context."""
    monkeypatch.setitem(_chunk, "value", DEFAULT_PREFILL_CHUNK)
    fd.test_full_context(device_1x1, cfg, kind)


DEFAULT_CHUNK_LENGTHS = [8191, 8192, 8193, 16384 + 1000, 5000, 700]


@pytest.mark.parametrize("kind", KINDS)
def test_default_chunk_boundaries(device_1x1, cfg, kind):
    """Default 8192-row chunk: lengths around one and two chunks plus a short reuse, then continuation.

    Sliding layers split each chunk's attention into 2048-row query blocks with window-shifted
    page-table origins: 8192 = 4 full blocks, 5000 -> 2048 + 2048 + 960 (padded tail block),
    17384 -> two 8192 chunks + a 1000-row (padded to 1024) chunk deep in the context. Real weights. Each request gets fresh pages and is followed by two decode steps. The continuation
    starts at 4160 (page- but not chunk-aligned) and spans the 8192 chunk boundary inside one call.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    max_pages = 18432 // BLOCK
    dec = FusedDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=device_1x1, num_blocks=2 * max_pages + 1
    )
    assert dec.prefill_chunk_size == DEFAULT_PREFILL_CHUNK
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
        check(f"default_chunk_prefill[{kind}]", want, got, seq_len=length, chunk=DEFAULT_PREFILL_CHUNK)
        for s in range(steps):
            p = length + s
            want = ref.forward(x[:, p : p + 1], cache, p)
            out = dec.decode_forward(
                to_tt(x[:, p : p + 1], device_1x1),
                current_pos=to_tt_int(torch.tensor([p]), device_1x1),
                page_table=table,
            )
            check(f"default_chunk_decode[{kind}]", want, from_tt(out), pos=p)

    # Explicit continuation: [0, 4160) then [4160, 12000), the second call crossing the 8192 boundary.
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
    check(f"default_chunk_continuation[{kind}]", want, torch.cat(parts, dim=1), seq_len=length, starts=[0, 4160])
    assert canary_is_clean(dec)


def _pair(device, cfg, layer_idx, sd, num_blocks):
    shared = {}
    kw = dict(hf_config=cfg, layer_idx=layer_idx, mesh_device=device, num_blocks=num_blocks, shared=shared)
    return FunctionalDecoder.from_state_dict(sd, **kw), FusedDecoder.from_state_dict(sd, **kw)


@pytest.mark.parametrize("kind", KINDS)
def test_fused_matches_functional(device_1x1, cfg, kind):
    """Identical inputs, pages and positions through FunctionalDecoder and FusedDecoder (real weights).

    Prefill lengths cover a tile tail, the functional 2048 chunk and the fused 8192 chunk. Traced
    decode runs at batch 1 and 32 on the filled caches. Fused-vs-functional PCC must stay >= 0.999
    on every row: the rewrites are not bit-exact (minimal_matmul accumulation order, one fewer BF16
    rounding in the fused activations, k = v_norm * w_k), but must not drift from the functional result.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    batch = 32
    max_pages = 9216 // BLOCK
    fun, fus = _pair(device_1x1, cfg, layer_idx, sd, num_blocks=batch * max_pages + 1)
    pages = Pages(fun.num_blocks, max_pages, num_users=batch, seed=41)
    lengths = [100, 2049, 8193] + [
        int(n) for n in torch.randint(1, 1300, (batch - 3,), generator=torch.Generator().manual_seed(5))
    ]
    for u, n in enumerate(lengths):
        pages.ensure(u, n + 1)
    table = pages.tt(device_1x1)
    for u, n in enumerate(lengths):
        x = to_tt(synthetic_hidden(layer_idx, n, seed=400 + u), device_1x1)
        a = from_tt(fun.prefill_forward(x, page_table=table, user_id=u))
        b = from_tt(fus.prefill_forward(x, page_table=table, user_id=u))
        if u < 3:
            check(f"fused_vs_functional_prefill[{kind}]", a, b, bar=0.999, seq_len=n)
    for b_ in (1, batch):
        xs = torch.cat([synthetic_hidden(layer_idx, 1, seed=900 + u) for u in range(b_)], 1)
        pos = torch.tensor(lengths[:b_])
        rows = pages.table[:b_]
        outs = []
        for dec in (fun, fus):
            tr = TracedDecode(dec, device_1x1, b_)
            tr.set_inputs(xs, pos, rows)
            tr.capture()  # eager compile step + captured step both rewrite the same cache slot
            outs.append(from_tt(tr.replay()))
            tr.release()
        check(f"fused_vs_functional_traced_decode[{kind},b{b_}]", outs[0], outs[1], bar=0.999, batch=b_)


def _device_ops(fn):
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    fn()
    graph = ttnn.graph.end_graph_capture()
    names = [n["params"]["name"] for n in graph if n.get("node_type") == "function_start"]
    return [n for n in names if n.endswith("DeviceOperation") or n == "SDPAOperation"]


@pytest.mark.parametrize("kind", KINDS)
def test_fused_op_graph(device_1x1, cfg, kind):
    """The fused ops are the ones that run (no functional fallback) and the graph is smaller.

    Prefill: matmuls are MinimalMatmul, no standalone GELU/scale; sliding layers run the fused
    head-split + norm + RoPE op (4x) instead of head split and HF RoPE. Decode: one fused K/V cache
    update, no tilize, no standalone GELU/scale; sliding layers rotate with rotary_embedding_llama.
    Records both device-op sequences as evidence.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = fd.synthetic_layer_state_dict_cached(layer_idx)
    out = {}
    for name, cls in (("functional", FunctionalDecoder), ("fused", FusedDecoder)):
        dec = cls.from_state_dict(
            sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=device_1x1, num_blocks=32 * 40 + 1, max_context=8192
        )
        pages = Pages(dec.num_blocks, 40, num_users=32)
        for u in range(32):
            pages.ensure(u, 2101)
        x = to_tt(synthetic_hidden(layer_idx, 2100), device_1x1)
        table = pages.tt(device_1x1)
        dec.prefill_forward(x, page_table=table)  # warm
        out[(name, "prefill", 1)] = _device_ops(lambda: dec.prefill_forward(x, page_table=table))
        for batch in (1, 32):
            xd = to_tt(synthetic_hidden(layer_idx, batch, seed=1), device_1x1)
            pos = to_tt_int(torch.full((batch,), 2100), device_1x1)
            tbl = pages.tt(device_1x1, rows=slice(0, batch))
            dec.decode_forward(xd, current_pos=pos, page_table=tbl)  # warm
            out[(name, "decode", batch)] = _device_ops(lambda: dec.decode_forward(xd, current_pos=pos, page_table=tbl))
        del dec
    for key, ops in out.items():
        record(f"op_graph[{kind}]", impl=key[0], path=key[1], batch=key[2], num_device_ops=len(ops), ops=ops)
    pre = out[("fused", "prefill", 1)]
    assert "MatmulDeviceOperation" not in pre and pre.count("MinimalMatmulDeviceOperation") == 5
    assert "UnaryDeviceOperation" not in pre, "standalone GELU/scale in fused prefill"
    sliding = cfg.layer_types[layer_idx] == "sliding_attention"
    if sliding:
        # Interleaved-RoPE path: fused head split + norm + RoPE (Q halves, K, V), no separate head split/RoPE.
        assert pre.count("DitFusedDistributedRmsnormDeviceOperation") == 4
        assert "NlpCreateHeadsDeviceOperation" not in pre and "RotaryEmbeddingHfDeviceOperation" not in pre
    else:
        assert (
            "DitFusedDistributedRmsnormDeviceOperation" not in pre
            and pre.count("RotaryEmbeddingHfDeviceOperation") == 2
        )
    for batch in (1, 32):
        dec_ops = out[("fused", "decode", batch)]
        rope_op = "RotaryEmbeddingLlamaDeviceOperation" if sliding else "RotaryEmbeddingHfDeviceOperation"
        assert dec_ops.count(rope_op) == 2, f"decode RoPE should be {rope_op}"
        assert dec_ops.count("PagedFusedUpdateCacheDeviceOperation") == 1
        assert "PagedUpdateCacheDeviceOperation" not in dec_ops
        assert not any("Tilize" in op for op in dec_ops)
        # The only unary op left is the relu clamping inactive (-1) positions before the index typecast.
        assert dec_ops.count("UnaryDeviceOperation") == 1
        assert len(dec_ops) < len(out[("functional", "decode", batch)])
    assert len(pre) < len(out[("functional", "prefill", 1)])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", KINDS)
def test_stress_repeated(device_1x1, cfg, kind):
    """Long traced-decode run and repeated prefills on one instance (real weights).

    Batch 32: users start at mixed lengths, then 300 trace replays advance every user one token per
    replay, crossing page edges and the sliding window, with positions/pages refreshed between
    replays. Every 60th replay is checked against HF for every user (the HF cache catches up on the
    tokens in between with HF's K/V computation). Afterwards the same three prompts are
    prefilled three times into fresh pages and must be bitwise identical each round.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    batch, replays, every = 32, 300, 60
    max_pages = 2048 // BLOCK
    dec = FusedDecoder.from_state_dict(
        sd, hf_config=cfg, layer_idx=layer_idx, mesh_device=device_1x1, num_blocks=batch * max_pages + 1
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
    done = list(lengths)  # tokens of each user already in the HF cache
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
            # Unscored tokens since the last check enter the HF cache through HF's own K/V math
            # (HFLayer.inject_prefix_kv); the scored token runs through the full HF layer forward.
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
