# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HF-vs-TTNN tests for the Gemma4 12B functional decoder (both layer kinds, real config shapes).

Layer kinds: ``sliding`` = layer 0 (sliding_attention), ``full`` = layer 5 (full_attention).
``synthetic`` weights are generated from recorded checkpoint stats; ``real`` weights come from the
pinned checkpoint. Every comparison scores the whole tensor and every logical row (PCC >= 0.995).
"""

import os
import sys
import time

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.harness import (
    BLOCK,
    Pages,
    build_decoder,
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
    layer_state_dict,
    real_hidden,
    real_layer_state_dict,
    real_prompt_ids,
    synthetic_hidden,
    text_config,
)
from models.autoports.google_gemma_4_12b_it.tt.functional_decoder import rope_tables

KINDS = list(LAYER_KINDS)
WEIGHTS = ["synthetic", "real"]


@pytest.fixture(scope="module")
def cfg():
    return text_config()


class TracedDecode:
    """Decode trace with persistent device inputs refreshed by host copies outside the trace."""

    def __init__(self, decoder, device, batch):
        self.decoder = decoder
        self.device = device
        self.batch = batch
        h = decoder.hidden_size
        self.x = to_tt(torch.zeros(1, batch, h), device)
        self.pos = to_tt_int(torch.zeros(batch), device)
        self.page_table = None
        self.trace_id = None
        self.out = None

    def set_inputs(self, x, pos, table):
        """x ``[1, B, H]`` torch, pos ``[B]``, table ``[B, pages]`` (host -> persistent device buffers)."""
        host_x = ttnn.from_torch(x.reshape(1, 1, self.batch, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_x, self.x)
        host_pos = ttnn.from_torch(pos.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host_pos, self.pos)
        host_table = ttnn.from_torch(table.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.page_table is None:
            self.page_table = ttnn.from_torch(
                table.to(torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
        else:
            ttnn.copy_host_to_device_tensor(host_table, self.page_table)

    def capture(self):
        """Compile with one eager step on the current inputs, then capture the same step."""
        self.decoder.decode_forward(self.x, current_pos=self.pos, page_table=self.page_table)
        ttnn.synchronize_device(self.device)
        self.trace_id = ttnn.begin_trace_capture(self.device, cq_id=0)
        self.out = self.decoder.decode_forward(self.x, current_pos=self.pos, page_table=self.page_table)
        ttnn.end_trace_capture(self.device, self.trace_id, cq_id=0)
        ttnn.synchronize_device(self.device)

    def replay(self):
        ttnn.execute_trace(self.device, self.trace_id, cq_id=0, blocking=False)
        return self.out

    def release(self):
        if self.trace_id is not None:
            ttnn.release_trace(self.device, self.trace_id)
            self.trace_id = None


def test_rope_tables_match_hf(cfg):
    """Setup-time RoPE tables equal HF's rotary embedding for both kinds, up to the last position."""
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedTextRotaryEmbedding

    hf = Gemma4UnifiedTextRotaryEmbedding(cfg)
    pos = torch.tensor([0, 1, 1023, 1024, 4097, 131071, 262143])
    for kind in ("sliding_attention", "full_attention"):
        cos, sin = rope_tables(cfg, kind, cfg.max_position_embeddings)
        hcos, hsin = hf(torch.zeros(1, dtype=torch.float32), pos[None], kind)
        assert torch.equal(cos[pos], hcos[0]) and torch.equal(sin[pos], hsin[0]), kind
        record("rope_tables", kind=kind, positions=pos.tolist(), exact=True)


LENGTHS = [1, 31, 64, 65, 100, 1023, 1024, 1025, 2047, 2048, 2049, 4097, 6000, 17]


@pytest.mark.parametrize("weights", WEIGHTS)
@pytest.mark.parametrize("kind", KINDS)
def test_prefill_decode_lengths(device_1x1, cfg, kind, weights):
    """Fresh requests of increasing non-aligned lengths on one loaded instance, then a short reuse.

    Covers tile (32), page (64), sliding window (1024) and prefill chunk (2048) boundaries plus
    multi-chunk tails; each request gets freshly shuffled pages (reusing released ones), its prefill
    is scored on every row and three follow-on decode steps consume the filled cache.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = layer_state_dict(layer_idx, weights)
    ref = HFLayer(cfg, layer_idx, sd)
    max_pages = 8192 // BLOCK
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=2 * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=1, seed=layer_idx)
    steps = 3
    for i, length in enumerate(LENGTHS):
        pages.release(0)
        pages.ensure(0, length + steps)
        table = pages.tt(device_1x1)
        x = synthetic_hidden(layer_idx, length + steps, seed=i)
        cache = ref.new_cache()
        want = ref.prefill(x[:, :length], cache)
        got = from_tt(dec.prefill_forward(to_tt(x[:, :length], device_1x1), page_table=table))
        check(f"prefill[{kind},{weights}]", want, got, seq_len=length, kind=kind, weights=weights)
        for s in range(steps):
            p = length + s
            want = ref.forward(x[:, p : p + 1], cache, p)
            out = dec.decode_forward(
                to_tt(x[:, p : p + 1], device_1x1),
                current_pos=to_tt_int(torch.tensor([p]), device_1x1),
                page_table=table,
            )
            check(f"decode[{kind},{weights}]", want, from_tt(out), pos=p, kind=kind, weights=weights)
    assert canary_is_clean(dec)


@pytest.mark.parametrize("kind", KINDS)
def test_chunked_prefill_continuation(device_1x1, cfg, kind):
    """Chunked vs explicit-continuation vs unchunked prefill of one prompt, then stale-state reuse."""
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    length, steps = 5000, 2
    max_pages = 8192 // BLOCK
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=3 * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=3, seed=11)
    x = synthetic_hidden(layer_idx, length + steps, seed=5)
    cache = ref.new_cache()
    want = ref.prefill(x[:, :length], cache)
    want_dec = [ref.forward(x[:, p : p + 1], cache, p) for p in range(length, length + steps)]

    # User 0: one call, internally 2048 + 2048 + 904 (padded to 960) rows.
    pages.ensure(0, length + steps)
    # User 1: explicit prefix continuation; starts 1088 and 3200 are page- but not chunk-aligned,
    # 1088 lies past the sliding window and the middle call spans a chunk boundary.
    pages.ensure(1, length + steps)
    table = pages.tt(device_1x1)
    chunked = from_tt(dec.prefill_forward(to_tt(x[:, :length], device_1x1), page_table=table, user_id=0))
    check(f"chunked_prefill[{kind}]", want, chunked, seq_len=length)
    parts = []
    for a, b in ((0, 1088), (1088, 3200), (3200, length)):
        parts.append(
            from_tt(dec.prefill_forward(to_tt(x[:, a:b], device_1x1), page_table=table, user_id=1, start_pos=a))
        )
    continued = torch.cat(parts, dim=1)
    check(f"continuation_prefill[{kind}]", want, continued, seq_len=length, starts=[0, 1088, 3200])
    check(f"continuation_vs_chunked[{kind}]", chunked, continued, bar=0.999)
    for user in (0, 1):
        row_table = to_tt_int(pages.table[user : user + 1], device_1x1)
        for s, p in enumerate(range(length, length + steps)):
            out = dec.decode_forward(
                to_tt(x[:, p : p + 1], device_1x1),
                current_pos=to_tt_int(torch.tensor([p]), device_1x1),
                page_table=row_table,
            )
            check(f"decode_after_prefill[{kind}]", want_dec[s], from_tt(out), user=user, pos=p)

    # Unchunked control: one physical chunk covers the whole prompt.
    whole = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=max_pages + 1, prefill_chunk_size=8192)
    wpages = Pages(whole.num_blocks, max_pages, num_users=1, seed=12)
    wpages.ensure(0, length)
    unchunked = from_tt(whole.prefill_forward(to_tt(x[:, :length], device_1x1), page_table=wpages.tt(device_1x1)))
    check(f"unchunked_prefill[{kind}]", want, unchunked, seq_len=length, chunk=8192)
    check(f"chunked_vs_unchunked[{kind}]", unchunked, chunked, bar=0.999)
    del whole

    # Stale-state check: release user 0, give its pages to a new short request with new tokens.
    pages.release(0)
    pages.ensure(2, 300 + 1)
    table = pages.tt(device_1x1)
    y = synthetic_hidden(layer_idx, 301, seed=99)
    ycache = ref.new_cache()
    want = ref.prefill(y[:, :300], ycache)
    got = from_tt(dec.prefill_forward(to_tt(y[:, :300], device_1x1), page_table=table, user_id=2))
    check(f"reused_pages_prefill[{kind}]", want, got, seq_len=300)
    out = dec.decode_forward(
        to_tt(y[:, 300:301], device_1x1),
        current_pos=to_tt_int(torch.tensor([300]), device_1x1),
        page_table=to_tt_int(pages.table[2:3], device_1x1),
    )
    check(f"reused_pages_decode[{kind}]", ref.forward(y[:, 300:301], ycache, 300), from_tt(out), pos=300)
    assert canary_is_clean(dec)


@pytest.mark.parametrize("batch", [1, 12, 32])
@pytest.mark.parametrize("kind", KINDS)
def test_traced_decode(device_1x1, cfg, kind, batch):
    """Traced decode replays scored against HF per user, with changing tokens, positions and pages.

    Users are prefilled to different lengths (some past the sliding window and around page edges).
    Between replays the host refreshes hidden states, positions and the page table (new pages are
    appended when a user crosses a page boundary). With batch 32, user 3 is inactive (-1) for the
    first replays. The final replay is repeated with identical inputs and must match bitwise.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    steps = 6
    g = torch.Generator().manual_seed(batch + layer_idx)
    if batch == 1:
        lengths = [1086]
    else:
        lengths = torch.randint(1, 1300, (batch,), generator=g).tolist()
        lengths[:4] = [1, 63, 1020, 1279]  # page edge and window crossings during decode
    max_pages = 2048 // BLOCK
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=batch * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=batch, seed=batch)
    xs, caches = [], []
    for u, n in enumerate(lengths):
        x = synthetic_hidden(layer_idx, n + steps, seed=1000 + u)
        xs.append(x)
        cache = ref.new_cache()
        ref.prefill(x[:, :n], cache)
        caches.append(cache)
        pages.ensure(u, n)  # decode pages are appended on demand below
        dec.prefill_forward(to_tt(x[:, :n], device_1x1), page_table=pages.tt(device_1x1), user_id=u)

    inactive = {3} if batch > 3 else set()
    trace = TracedDecode(dec, device_1x1, batch)
    pos = torch.tensor(lengths)
    done = [0] * batch  # decode steps consumed per user

    def inputs():
        cur = []
        for u in range(batch):
            if u in inactive:
                cur.append(-1)
            else:
                pages.ensure(u, int(pos[u]) + 1)
                cur.append(int(pos[u]))
        x = torch.cat([xs[u][:, int(lengths[u]) + done[u] : int(lengths[u]) + done[u] + 1] for u in range(batch)], 1)
        return x, torch.tensor(cur)

    for step in range(steps):
        if step == 3:
            inactive.clear()  # user 3 resumes with its first pending token
        x, cur = inputs()
        trace.set_inputs(x, cur, pages.table)
        if step == 0:
            trace.capture()
        got = from_tt(trace.replay())[0]
        for u in range(batch):
            if cur[u] < 0:
                continue
            p = int(cur[u])
            want = ref.forward(xs[u][:, p : p + 1], caches[u], p)
            check(f"traced_decode[{kind},b{batch}]", want[0], got[u : u + 1], user=u, pos=p, step=step)
            done[u] += 1
            pos[u] += 1
        last = (x, cur)

    # Determinism: replay the last step again with identical inputs (same cache slot rewritten).
    x, cur = last
    trace.set_inputs(x, cur, pages.table)
    a = ttnn.to_torch(trace.replay()).clone()
    trace.set_inputs(x, cur, pages.table)
    b = ttnn.to_torch(trace.replay())
    assert torch.equal(a, b), "trace replay with identical inputs is not bitwise deterministic"
    record(f"traced_decode_determinism[{kind},b{batch}]", bitwise_equal=True)
    trace.release()
    assert canary_is_clean(dec)


@pytest.mark.parametrize("kind", KINDS)
def test_prefill_determinism(device_1x1, cfg, kind):
    """Identical prefill inputs (fresh request each time, different pages) give bitwise-equal outputs."""
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    max_pages = 4096 // BLOCK
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=2 * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=2, seed=3)
    x = synthetic_hidden(layer_idx, 3000, seed=8)
    outs = []
    for u in range(2):
        pages.ensure(u, 3000)
        outs.append(
            ttnn.to_torch(dec.prefill_forward(to_tt(x, device_1x1), page_table=pages.tt(device_1x1), user_id=u))
        )
    assert torch.equal(outs[0], outs[1])
    record(f"prefill_determinism[{kind}]", seq_len=3000, bitwise_equal=True)


@pytest.mark.parametrize("kind", KINDS)
def test_runtime_fallback_audit(device_1x1, cfg, kind):
    """No torch / host-transfer call happens inside one prefill or one decode forward.

    A Python profiler hook records every Python/C call made while the forward runs; any frame from
    the torch package or any ttnn host-transfer entry point fails the test. TTNN graph capture also
    lists the device ops of each pass as evidence.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = synthetic_layer_state_dict_cached(layer_idx)
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=129, max_context=8192)
    pages = Pages(dec.num_blocks, 128, num_users=1)
    pages.ensure(0, 2101)
    table = pages.tt(device_1x1)
    x = to_tt(synthetic_hidden(layer_idx, 2100), device_1x1)
    xd = to_tt(synthetic_hidden(layer_idx, 1, seed=1), device_1x1)
    pos = to_tt_int(torch.tensor([2100]), device_1x1)
    forbidden_ttnn = {
        "from_torch",
        "to_torch",
        "cpu",
        "copy_device_to_host_tensor",
        "copy_host_to_device_tensor",
        "to_device",
        "from_device",
    }
    torch_dir = os.path.dirname(torch.__file__)

    def audited(fn):
        bad = []

        def hook(frame, event, arg):
            if event == "call" and frame.f_code.co_filename.startswith(torch_dir):
                bad.append(f"torch:{frame.f_code.co_name}")
            if event in ("call", "c_call"):
                name = getattr(arg, "__name__", None) if event == "c_call" else frame.f_code.co_name
                if name in forbidden_ttnn:
                    bad.append(f"ttnn:{name}")

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
        sys.setprofile(hook)
        try:
            fn()
        finally:
            sys.setprofile(None)
            graph = ttnn.graph.end_graph_capture()
        ops = [n["params"]["name"] for n in graph if n.get("node_type") == "function_start"]
        return bad, ops

    # Warm both paths first so compilation is outside the audited pass.
    dec.prefill_forward(x, page_table=table)
    dec.decode_forward(xd, current_pos=pos, page_table=table)
    bad_p, ops_p = audited(lambda: dec.prefill_forward(x, page_table=table))
    bad_d, ops_d = audited(lambda: dec.decode_forward(xd, current_pos=pos, page_table=table))
    host_ops = [o for o in ops_p + ops_d if any(k in o.lower() for k in ("to_torch", "from_torch", "cpu", "host"))]
    record(
        f"fallback_audit[{kind}]",
        prefill_python_violations=bad_p,
        decode_python_violations=bad_d,
        prefill_top_level_ops=sorted(set(ops_p)),
        decode_top_level_ops=sorted(set(ops_d)),
        host_ops=host_ops,
    )
    assert not bad_p and not bad_d and not host_ops


_SD_CACHE = {}


def synthetic_layer_state_dict_cached(layer_idx):
    if layer_idx not in _SD_CACHE:
        _SD_CACHE[layer_idx] = layer_state_dict(layer_idx, "synthetic")
    return _SD_CACHE[layer_idx]


@pytest.mark.parametrize("kind", KINDS)
def test_full_context(device_1x1, cfg, kind):
    """HF-advertised context (262144): prefill 262140 tokens (non-aligned tail; a fresh request plus an
    explicit continuation at 131136), then 4 traced decode steps at positions 262140..262143, i.e. the
    last decode attends the full 262144-token context.

    Real weights. The checked prefill windows (start, middle across 131072, last 1024 rows) and every
    decode step run through the HF decoder layer; the unchecked prefix K/V between windows is appended
    with ``HFLayer.inject_prefix_kv`` (HF's own K/V computation) so the HF reference stays tractable.
    """
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    context = cfg.max_position_embeddings
    steps = 4
    length = context - steps
    max_pages = context // BLOCK
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=1, seed=21)
    pages.ensure(0, context)
    table = pages.tt(device_1x1)
    x = synthetic_hidden(layer_idx, context, seed=77)

    # Two calls: a fresh 131136-token request, then an explicit continuation starting at 131136
    # (page-aligned but not a multiple of the 2048-row prefill chunk or the 128-row full-attention Q
    # tile, so its chunks start mid-chunk and use 64-row Q tiles) up to 262140 tokens.
    split = 131136
    t0 = time.time()
    got = []
    for a, b in ((0, split), (split, length)):
        out = dec.prefill_forward(to_tt(x[:, a:b], device_1x1), page_table=table, start_pos=a)
        ttnn.synchronize_device(device_1x1)
        got.append(from_tt(out))
        out.deallocate(True)
    prefill_s = time.time() - t0
    got = torch.cat(got, dim=1)

    windows = [(0, 512), (131072 - 256, 131072 + 256), (length - 1024, length)]
    cache = ref.new_cache()
    done = 0
    for a, b in windows:
        if a > done:
            ref.inject_prefix_kv(x[:, done:a], cache, start_pos=done)
        want = ref.prefill(x[:, a:b], cache, start_pos=a, slice_len=256)
        check(f"full_context_prefill[{kind}]", want, got[:, a:b], rows_from=a, rows_to=b, seq_len=length)
        done = b

    trace = TracedDecode(dec, device_1x1, 1)
    for s in range(steps):
        p = length + s
        trace.set_inputs(x[:, p : p + 1], torch.tensor([p]), pages.table)
        if s == 0:
            trace.capture()
        got_d = from_tt(trace.replay())
        want_d = ref.forward(x[:, p : p + 1], cache, p)
        check(f"full_context_traced_decode[{kind}]", want_d, got_d, pos=p, attended=p + 1)
    trace.release()
    record(
        f"full_context[{kind}]", prefill_tokens=length, decode_last_pos=context - 1, prefill_wall_s=round(prefill_s, 1)
    )
    assert canary_is_clean(dec)


def test_all_layers_real_activations(device_1x1, cfg):
    """Every one of the 48 checkpoint layers on real activations of a real chat-formatted prompt.

    The HF float32 text stack runs the prompt layer by layer; each TT layer receives the HF input of
    that layer, prefills 1024 tokens and decodes tokens 1024 and 1025 (eager), scored against HF.
    """
    ids = real_prompt_ids(1026)
    x = real_hidden(cfg, 0, ids)
    shared = {}
    for layer_idx in range(cfg.num_hidden_layers):
        sd = real_layer_state_dict(layer_idx)
        ref = HFLayer(cfg, layer_idx, sd)
        want = ref.prefill(x, ref.new_cache())
        dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=18, shared=shared)
        pages = Pages(dec.num_blocks, 17, num_users=1, seed=layer_idx)
        pages.ensure(0, 1026)
        table = pages.tt(device_1x1)
        got = from_tt(dec.prefill_forward(to_tt(x[:, :1024], device_1x1), page_table=table))
        kind = cfg.layer_types[layer_idx]
        check("real_activations_prefill", want[:, :1024], got, layer=layer_idx, kind=kind, seq_len=1024)
        for p in (1024, 1025):
            out = dec.decode_forward(
                to_tt(x[:, p : p + 1], device_1x1),
                current_pos=to_tt_int(torch.tensor([p]), device_1x1),
                page_table=table,
            )
            check("real_activations_decode", want[:, p : p + 1], from_tt(out), layer=layer_idx, kind=kind, pos=p)
        del dec
        x = want


@pytest.mark.parametrize("kind", KINDS)
def test_decode_odd_width_page_table(device_1x1, cfg, kind):
    """Decode in the last block of a page-table row with an odd number of blocks (17).

    paged_scaled_dot_product_attention_decode rounds the causal K range up to whole K tiles and does
    not bound page-table reads by the row width: with a 128-token K tile over 64-token blocks it
    dereferences entry 17 of a 17-entry row (DRAM page padding holding stale command-queue bytes) and
    can pull a garbage block whose +Inf-saturating keys escape the additive mask. That intermittently
    broke test_all_layers_real_activations (5/5 long pytest sessions, layers 4/19/21/24/31, PCC
    0.66-0.995). The decoder therefore uses one K tile per cache block; this test pins that contract
    and the odd-width shape (the stale padding itself cannot be planted deterministically from Python).
    """
    layer_idx = LAYER_KINDS[kind]
    sd = synthetic_layer_state_dict_cached(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    dec = build_decoder(device_1x1, cfg, layer_idx, sd, num_blocks=18, max_context=8192)
    assert dec.block_size % dec.sdpa_decode_cfg.k_chunk_size == 0
    table = (torch.randperm(17, generator=torch.Generator().manual_seed(layer_idx)) + 1).int().reshape(1, 17)
    tt_table = to_tt_int(table, device_1x1)
    x = synthetic_hidden(layer_idx, 1026, seed=4)
    rcache = ref.new_cache()
    ref.prefill(x[:, :1024], rcache)
    dec.prefill_forward(to_tt(x[:, :1024], device_1x1), page_table=tt_table)
    for p in (1024, 1025):
        out = dec.decode_forward(
            to_tt(x[:, p : p + 1], device_1x1),
            current_pos=to_tt_int(torch.tensor([p]), device_1x1),
            page_table=tt_table,
        )
        check(f"odd_width_page_table_decode[{kind}]", ref.forward(x[:, p : p + 1], rcache, p), from_tt(out), pos=p)
