# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-model generator tests on the QB2 mesh (48 real layers unless ``GEMMA4_FM_LAYERS`` selects a reduced model).

One generator (build + prepare + trace capture) serves the whole session; every test leaves it reset.

* split-sampling trace contract: token feedback through ``tt_out_tok``, device-side positions, unchanged / changed
  page tables, no host refresh in the steady state, greedy <-> sampled parameter switches without recapture;
* host-sampling compatibility mode against the on-device greedy path;
* batch: 32 identical rows (bitwise-equal logits and tokens across rows), mixed prompts vs one-at-a-time, inactive
  rows, every decode bucket;
* non-aligned prompt lengths around bucket and chunk boundaries through the public generator;
* determinism of repeated prefill logits and generation;
* (optimized full model) traced prefill vs the eager bucketed path, the pipelined free-running ``generate`` (on-device
  token feedback, asynchronous token copies, EOS deactivation of rows in flight) vs the blocking path, and the split
  ``decode_forward(read_from_device=False)`` / ``read_decode_output`` / ``process_decode_output_host`` API.

  FD_EVIDENCE=doc/full_model/evidence.jsonl pytest models/autoports/google_gemma_4_12b_it/tests/test_full_model.py
"""

import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_12b_it.tests.harness import record
from models.autoports.google_gemma_4_12b_it.tt.generator import GREEDY, SamplingParams, build_generator
from models.autoports.google_gemma_4_12b_it.tt.model import CANARY_BLOCK


@pytest.fixture(scope="session")
def gen():
    from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh

    torch.set_num_threads(16)
    layers = os.environ.get("GEMMA4_FM_LAYERS")
    kwargs = {}
    if layers:
        kwargs = dict(layer_indices=[int(v) for v in layers.split(",")], max_context=16384, num_blocks=1200)
    if os.environ.get("GEMMA4_FM_BLOCKS"):
        kwargs["num_blocks"] = int(os.environ["GEMMA4_FM_BLOCKS"])
    mesh = open_qb2_mesh(trace_region_size=0)
    g = build_generator(None, mesh, **kwargs)
    yield g
    g.teardown()
    close_qb2_mesh(mesh)


def chat(gen, text):
    return gen.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True, return_dict=False
    )


PROMPTS = [
    "Write a haiku about machine learning.",
    "Explain the difference between supervised and unsupervised learning in simple terms.",
    "What are the three laws of thermodynamics?",
    'Translate the following to French: "Hello, how are you today?"',
    "Write a Python function to calculate the Fibonacci sequence.",
    "Name five rivers in Europe and the countries they flow through.",
    "What is the capital of Australia, and why was it chosen?",
    "Give me three tips for writing clear technical documentation.",
]


# Long natural text for the length / path tests (tests/text/long_natural_text.md: a pinned copy of
# tech_reports/LLMs/llms.md at commit 3be67656, ~33K tokens). The full-model stage tiled
# PROMPTS instead: on that repetitive text 48 chained layers amplify rounding (HF bf16 itself drops to worst-row PCC
# 0.95-0.98 vs fp32, doc/full_model README "Precision"), which made the split-vs-unsplit and cross-prefill-path
# ranking comparisons below a test of one precision policy (doc/datatype_sweep work log 7-8).
# GEMMA4_FM_REPETITIVE=1 restores the tiled text as a stress run.
LONG_TEXT = Path(__file__).resolve().parent / "text/long_natural_text.md"


def long_base(gen, repeats):
    if os.environ.get("GEMMA4_FM_REPETITIVE"):
        return chat(gen, " ".join(PROMPTS) * repeats)
    return chat(gen, LONG_TEXT.read_text())


def near_tie(a, b, k=3.0):
    """Two logit rows of the same position computed along two paths (placements, prefill matmul paths): equal argmax,
    or argmaxes that differ only within the rows' own rounding noise. Each row's margin between its argmax and the other
    row's argmax must be at most ``k`` standard deviations of ``a - b`` over the vocabulary. Returns (ok, details)."""
    ia, ib = int(a.argmax()), int(b.argmax())
    if ia == ib:
        return True, None
    noise = float((a.float() - b.float()).std())
    ma, mb = float(a[ia] - a[ib]), float(b[ib] - b[ia])
    return ma <= k * noise and mb <= k * noise, {"margins": [round(ma, 4), round(mb, 4)], "noise_std": round(noise, 4)}


def device_tokens(gen):
    return ttnn.to_torch(ttnn.get_device_tensors(gen.tokens)[0]).reshape(-1).to(torch.int64)


def device_ints(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).to(torch.int64)


def row_logits(gen, ids, index):
    """Logits of position ``index`` of prompt ``ids`` (prefill chunk by chunk, one 32-row logits readback)."""
    m = gen.model
    row = gen._alloc_row(len(ids))
    plan = m.chunk_plan(0, len(ids))
    table = m.device_ints(m.prefill_page_table(row, len(ids), plan[-1][0] + plan[-1][2]))
    out = None
    for c0, real, bucket in plan:
        chunk = torch.zeros(bucket, dtype=torch.int32)
        chunk[:real] = torch.tensor(ids[c0 : c0 + real])
        hidden = m.prefill_chunk_hidden(chunk, table, c0)
        if c0 <= index < c0 + real:
            out = gen._rows_logits(hidden, torch.full((32,), index - c0, dtype=torch.int32))[0]
        ttnn.deallocate(hidden)
    gen._release(row)
    return out


def test_split_sampling_trace_contract(gen):
    """Two+ traced steps with different tokens / positions: the sampled token of replay N is the token input of
    replay N+1 (read from the persistent buffer), positions advance on device, nothing is refreshed from the host in
    the steady state, and a changed page table is refreshed exactly once."""
    ids = chat(gen, PROMPTS[0])
    n = len(ids)
    row = gen._alloc_row(n + 64)
    t0 = int(gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)[0])
    c0 = dict(gen.counters)
    tok, pos = t0, n
    seen_tokens, seen_logits = [], []
    for step in range(6):
        out = gen.decode_forward(torch.tensor([tok]), torch.tensor([pos]), page_table=row[None])
        nxt = int(out[0])
        # the token the next replay consumes is exactly the sampled one, in the persistent decode token buffer
        assert int(device_tokens(gen)[0]) == nxt
        # positions advanced on device inside the model trace
        assert int(device_ints(gen.positions[1])[0]) == pos + 1
        seen_tokens.append(nxt)
        seen_logits.append(gen.model.logits_to_torch(gen.logits, 1)[0])
        tok, pos = nxt, pos + 1
    c1 = dict(gen.counters)
    delta = {k: c1[k] - c0[k] for k in c1}
    # first decode step refreshes the request's position (request boundary); afterwards nothing
    assert delta["token_refreshes"] == 0, delta
    assert delta["position_refreshes"] == 1, delta
    assert delta["page_table_refreshes"] == 1, delta
    assert delta["model_replays"] == 6 and delta["sampling_replays"] == 6 and delta["trace_captures"] == 0
    assert all(not torch.equal(seen_logits[i], seen_logits[i + 1]) for i in range(5)), "logits did not change"
    # pure replay loop (no generator bookkeeping at all): 4 more steps from device state only
    for _ in range(4):
        ttnn.execute_trace(gen.mesh, gen.traces[("decode", 1, False)], cq_id=0, blocking=False)
        ttnn.execute_trace(gen.mesh, gen.traces[("sample",)], cq_id=0, blocking=False)
        seen_tokens.append(int(device_tokens(gen)[0]))
    assert int(device_ints(gen.positions[1])[0]) == pos + 4
    # the same 10 tokens from a host-driven teacher-forced replay (tokens and positions from the host every step)
    gen.reset()
    row2 = gen._alloc_row(n + 64)
    t0b = int(
        gen.prefill_forward(torch.tensor([ids]), page_table=row2[None], prompt_lens=[n], sample_on_device=True)[0]
    )
    assert t0b == t0
    ref, tok, pos = [], t0, n
    for i in range(10):
        out = gen.decode_forward(torch.tensor([tok]), torch.tensor([pos]), page_table=row2[None], host_sampling=True)
        ref.append(int(out[0][0]))
        tok, pos = ref[-1], pos + 1
    record("split_sampling_trace", feedback_tokens=seen_tokens, host_sampled_tokens=ref, counters_delta=delta)
    assert seen_tokens == ref, (seen_tokens, ref)
    gen._release(row2)
    gen.reset()


def test_page_table_change_and_slot_switch(gen):
    """Two requests in different blocks; switching the decode page table between them refreshes it once per change
    and each request continues exactly as it does alone."""
    a, b = chat(gen, PROMPTS[1]), chat(gen, PROMPTS[2])
    alone = {}
    for name, ids in (("a", a), ("b", b)):
        alone[name] = gen.generate(ids, 8, stop_on_eos=False)
        gen.reset()
    rows = {"a": gen._alloc_row(len(a) + 16), "b": gen._alloc_row(len(b) + 16)}
    state = {}
    for name, ids in (("a", a), ("b", b)):
        first = gen.prefill_forward(
            torch.tensor([ids]), page_table=rows[name][None], prompt_lens=[len(ids)], sample_on_device=True
        )
        state[name] = [int(first[0]), len(ids), [int(first[0])]]
    c0 = dict(gen.counters)
    # the device may already hold request a's table (the previous decode used the same blocks)
    held = gen._table_host[1][0][0, : rows["a"].numel()]  # bucket 1, page-table group 0
    expected = 14 - int(torch.equal(held, rows["a"]))
    for step in range(7):
        for name in ("a", "b"):  # alternate requests: the page table changes on every call
            tok, pos, out = state[name]
            nxt = int(gen.decode_forward(torch.tensor([tok]), torch.tensor([pos]), page_table=rows[name][None])[0])
            out.append(nxt)
            state[name] = [nxt, pos + 1, out]
    delta = {k: gen.counters[k] - c0[k] for k in c0}
    record("page_table_switch", alone=alone, interleaved={k: v[2] for k, v in state.items()}, delta=delta)
    assert state["a"][2] == alone["a"] and state["b"][2] == alone["b"]
    assert delta["page_table_refreshes"] == expected and delta["trace_captures"] == 0
    gen.reset()


def test_sampling_modes_reuse_traces(gen):
    """greedy -> top-k/top-p sampling -> greedy reuses the same traces; greedy output is identical before and after,
    seeded sampling is reproducible, and sampled tokens differ from greedy."""
    ids = chat(gen, PROMPTS[3])
    captures = gen.counters["trace_captures"]
    g1 = gen.generate(ids, 16, stop_on_eos=False)
    gen.reset()
    sp = SamplingParams(temperature=1.0, top_k=32, top_p=0.95, seed=1234)
    s1 = gen.generate(ids, 16, sampling_params=sp, stop_on_eos=False)
    gen.reset()
    s2 = gen.generate(
        ids, 16, sampling_params=SamplingParams(temperature=1.0, top_k=32, top_p=0.95, seed=1234), stop_on_eos=False
    )
    gen.reset()
    g2 = gen.generate(ids, 16, sampling_params=GREEDY, stop_on_eos=False)
    gen.reset()
    record("sampling_modes", greedy=g1, sampled=s1, sampled_again=s2, greedy_again=g2)
    assert g1 == g2
    assert gen.counters["trace_captures"] == captures
    assert all(0 <= t < gen.model.vocab_size for t in s1)
    assert s1 == s2, "seeded sampling must be reproducible"


def test_sampled_default_top_p(gen):
    """A sampled request that leaves top_p unset samples (top_p 1.0 = no nucleus filter), on device and on the host,
    reproducibly with a seed; it is not silently greedy."""
    ids = chat(gen, PROMPTS[7])
    greedy = gen.generate(ids, 32, stop_on_eos=False)
    gen.reset()
    sp = SamplingParams(temperature=1.5, top_k=32, seed=7)
    s1 = gen.generate(ids, 32, sampling_params=sp, stop_on_eos=False)
    rows = (gen._row_k[0].item(), gen._row_p[0].item(), gen._row_t[0].item())
    gen.reset()
    s2 = gen.generate(ids, 32, sampling_params=sp, stop_on_eos=False)
    gen.reset()
    host = gen.generate(ids, 32, sampling_params=sp, host_sampling=True, stop_on_eos=False)
    gen.reset()
    gen.set_sampling_params(GREEDY)
    record("sampled_default_top_p", greedy=greedy, sampled=s1, sampled_again=s2, host_sampled=host, row0_k_p_t=rows)
    assert rows == (32, 1.0, 1.5)
    assert s1 == s2 and s1 != greedy and host != greedy
    assert all(0 <= t < gen.model.vocab_size for t in s1 + host)


def test_prefill_sampling_params_by_slot(gen):
    """prefill_forward(slots=..., sampling_params=...) sets only the named slots: a greedy request decoding in slot 0
    keeps its parameters (identical tokens to its solo run) while a sampled request joins in slot 5."""
    a, b = chat(gen, PROMPTS[1]), chat(gen, PROMPTS[6])
    solo = gen.generate(a, 8, stop_on_eos=False)
    gen.reset()
    gen.set_sampling_params(GREEDY)
    ra, rb = gen._alloc_row(len(a) + 16), gen._alloc_row(len(b) + 16)
    table = torch.full((8, gen.pages), CANARY_BLOCK, dtype=torch.int32)
    table[0], table[5] = ra, rb
    out_a = [
        int(gen.prefill_forward(torch.tensor([a]), page_table=ra[None], prompt_lens=[len(a)], sample_on_device=True)[0])
    ]
    toks = torch.zeros(8, dtype=torch.int64)
    pos = torch.full((8,), -1, dtype=torch.int64)
    toks[0], pos[0] = out_a[0], len(a)
    for _ in range(3):
        out = gen.decode_forward(toks, pos, page_table=table)
        out_a.append(int(out[0]))
        toks[0], pos[0] = out[0], pos[0] + 1
    sp = SamplingParams(temperature=2.0, top_k=32, seed=11)
    first_b = gen.prefill_forward(
        torch.tensor([b]),
        page_table=rb[None],
        prompt_lens=[len(b)],
        slots=[5],
        sampling_params=sp,
        sample_on_device=True,
    )
    params = [(int(gen._row_k[i]), float(gen._row_p[i]), float(gen._row_t[i])) for i in (0, 5, 6)]
    toks[5], pos[5] = int(first_b[0]), len(b)
    out_b = [int(first_b[0])]
    for _ in range(4):
        out = gen.decode_forward(toks, pos, page_table=table)
        out_a.append(int(out[0]))
        out_b.append(int(out[5]))
        toks[0], toks[5] = out[0], out[5]
        pos[0] += 1
        pos[5] += 1
    gen.reset()
    gen.set_sampling_params(GREEDY)
    record("prefill_sampling_params_by_slot", solo=solo, slot0=out_a, slot5=out_b, k_p_t_slots_0_5_6=params)
    assert params == [(1, 0.0, 1.0), (32, 1.0, 2.0), (1, 0.0, 1.0)]
    assert out_a == solo
    assert all(0 <= t < gen.model.vocab_size for t in out_b)


def test_host_sampling_matches_device_greedy(gen):
    ids = chat(gen, PROMPTS[4])
    dev = gen.generate(ids, 24, stop_on_eos=False)
    gen.reset()
    host = gen.generate(ids, 24, host_sampling=True, stop_on_eos=False)
    gen.reset()
    record("host_vs_device_greedy", device=dev, host=host)
    assert dev == host


def test_batch32_identical_rows(gen):
    """32 rows of the same prompt: bitwise-equal last-token logits across rows (one decode step through every row) and
    identical generations; equal to the batch-1 generation."""
    ids = chat(gen, PROMPTS[5])
    single = gen.generate(ids, 12, stop_on_eos=False)
    gen.reset()
    batch = gen.generate([ids] * 32, 12, stop_on_eos=False)
    gen.reset()
    # decode logits across rows at B=32 (one teacher-forced step from the same state)
    rows = [gen._alloc_row(len(ids) + 4) for _ in range(32)]
    table = torch.stack(rows)
    first = gen.prefill_forward(
        torch.tensor([ids] * 32), page_table=table, prompt_lens=[len(ids)] * 32, sample_on_device=True
    )
    lg = gen.decode_forward(first, torch.tensor([len(ids)] * 32), page_table=table, return_logits=True)
    same = all(torch.equal(lg[0], lg[i]) for i in range(32))
    record(
        "batch32_identical_rows",
        single=single,
        rows_equal=all(r == batch[0] for r in batch),
        logits_rows_bitwise_equal=same,
        batch_row0=batch[0],
    )
    assert all(r == batch[0] for r in batch)
    assert same
    assert batch[0] == single
    gen.reset()


def test_mixed_batch_inactive_rows(gen):
    """Mixed prompt lengths in fixed slots with an inactive middle row: active rows match their one-at-a-time
    generations (bucket 4 and bucket 8)."""
    prompts = [chat(gen, p) for p in PROMPTS[:3]]
    alone = []
    for ids in prompts:
        alone.append(gen.generate(ids, 10, stop_on_eos=False))
        gen.reset()
    for bucket_rows in (4, 8):
        rows = [gen._alloc_row(len(p) + 16) for p in prompts]
        table = torch.full((bucket_rows, gen.pages), CANARY_BLOCK, dtype=torch.int32)
        slots = [0, 2, bucket_rows - 1]  # slot 1 (and the rest) inactive
        for s, r in zip(slots, rows):
            table[s] = r
        firsts = gen.prefill_forward(
            torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in prompts], batch_first=True),
            page_table=torch.stack(rows),
            prompt_lens=[len(p) for p in prompts],
            sample_on_device=True,
            slots=slots,
        )
        toks = torch.zeros(bucket_rows, dtype=torch.int64)
        pos = torch.full((bucket_rows,), -1, dtype=torch.int64)
        outs = [[int(f)] for f in firsts]
        for s, f, p in zip(slots, firsts, prompts):
            toks[s], pos[s] = int(f), len(p)
        for _ in range(9):
            out = gen.decode_forward(toks, pos, page_table=table)
            for j, s in enumerate(slots):
                outs[j].append(int(out[s]))
                toks[s] = int(out[s])
                pos[s] += 1
        record(f"mixed_batch_inactive_b{bucket_rows}", alone=alone, batch=outs)
        for j in range(3):
            assert outs[j] == alone[j], (bucket_rows, j, outs[j], alone[j])
        gen.reset()


@pytest.mark.parametrize("length", [1, 2, 127, 128, 129, 1000, 1024, 1281, 8191, 8192, 8193, 16385])
def test_non_aligned_prompt_lengths(gen, length):
    """Public generator path at lengths around bucket / chunk boundaries: last-token logits of a length-L prompt
    equal the L-th row of the all-logits prefill of a longer prompt (same program family), and decode continues."""
    if length > gen.model.max_context - 64:
        pytest.skip("beyond the configured context")
    base = long_base(gen, 200)
    ids = (base * (1 + length // len(base)))[:length]
    row = gen._alloc_row(length + 8)
    last = gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[length])[0, 0]
    toks = [int(last.argmax())]
    for i in range(3):
        toks.append(
            int(gen.decode_forward(torch.tensor([toks[-1]]), torch.tensor([length + i]), page_table=row[None])[0])
        )
    gen._release(row)
    gen.reset()
    # the same logical position computed as part of a longer prompt. Within one prefill matmul path (buckets of
    # <= 1280 rows: 2D-multicast BF16 MLP; larger: minimal_matmul with BFP8 MLP intermediates) the logits must agree
    # to rounding; across the two paths only the precision policy differs (argmax / top-5 compared).
    from models.autoports.google_gemma_4_12b_it.tt.model import prefill_bucket

    def short_path(n):
        return prefill_bucket(min(n, 8192)) <= 1280 if n <= 8192 else prefill_bucket(n - (n - 1) // 8192 * 8192) <= 1280

    extra = 37
    if short_path(length) != short_path(length + extra):
        extra = max(1, prefill_bucket(min(length, 8192)) - length) if length < 8192 else 37
    longer = (base * (2 + length // len(base)))[: length + extra]
    allg = row_logits(gen, longer, length - 1)
    gen.reset()
    from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc

    same_path = short_path(length) == short_path(length + extra)
    p = pcc(allg, last)
    top5 = len(set(allg.topk(5).indices.tolist()) & set(last.topk(5).indices.tolist()))
    record(
        "non_aligned_length",
        length=length,
        longer=length + extra,
        same_prefill_path=same_path,
        pcc_vs_longer_prompt=round(p, 6),
        top5_overlap=top5,
        argmax_equal=int(allg.argmax()) == toks[0],
        argmax_near_tie=near_tie(allg, last),
        tokens=toks,
    )
    assert torch.isfinite(last).all()
    if same_path:
        assert p > 0.999
        assert int(allg.argmax()) == toks[0]
    # across the two prefill matmul paths the argmax may differ only as a near tie within the paths' rounding noise
    assert near_tie(allg, last)[0] and top5 >= 4 and p > 0.9


def test_determinism(gen):
    ids = chat(gen, PROMPTS[6])
    a = gen.prefill_logits(ids)
    gen.reset()
    b = gen.prefill_logits(ids)
    gen.reset()
    g1 = gen.generate(ids, 20, stop_on_eos=False)
    gen.reset()
    g2 = gen.generate(ids, 20, stop_on_eos=False)
    gen.reset()
    record("determinism", prefill_bitwise=torch.equal(a, b), generate_equal=g1 == g2)
    assert torch.equal(a, b) and g1 == g2


@pytest.mark.parametrize("batch,length", [(4, 8200), (8, 4100), (16, 2100), (32, 1300)])
def test_user_split_traces(gen, batch, length):
    """Beyond the measured break-even the generator replays the user-split full-layer decode trace; the same step
    through the unsplit trace (same tokens, positions, cache) gives the same logits up to rounding."""
    from models.autoports.google_gemma_4_12b_it.tests.hf_reference import pcc

    if gen.model.full_layer is None or length + 8 > gen.model.max_context:
        pytest.skip("no full-attention layer / context too short")
    base = long_base(gen, 300)
    # rows of different lengths, every one starting with <bos> and the chat header (prefixes of the long prompt)
    prompts = [base[: length + i] for i in range(batch)]
    rows = [gen._alloc_row(length + batch + 8) for _ in range(batch)]
    table = torch.stack(rows)
    first = gen.prefill_forward(
        torch.nn.utils.rnn.pad_sequence([torch.tensor(p) for p in prompts], batch_first=True),
        page_table=table,
        prompt_lens=[len(p) for p in prompts],
        sample_on_device=True,
    )
    pos = torch.tensor([len(p) for p in prompts])
    assert gen.model.use_user_split(batch, int(pos.max()))
    c0 = gen.counters["model_replays"]
    split = gen.decode_forward(first, pos, page_table=table, return_logits=True)
    assert gen.counters["model_replays"] == c0 + 1
    # the same step again through the unsplit trace (rewrites identical K/V at the same positions)
    gen._write(gen.positions[batch], pos.to(torch.int32))
    ttnn.execute_trace(gen.mesh, gen.traces[("decode", batch, False)], cq_id=0, blocking=True)
    unsplit = gen.model.logits_to_torch(gen.logits, batch)
    gen._pos_host[batch] = pos.to(torch.int32) + 1
    p = min(pcc(split[b], unsplit[b]) for b in range(batch))
    agree = float((split.argmax(-1) == unsplit.argmax(-1)).float().mean())
    top5 = sum(
        len(set(split[b].topk(5).indices.tolist()) & set(unsplit[b].topk(5).indices.tolist())) for b in range(batch)
    ) / (5 * batch)
    ties = [near_tie(split[b], unsplit[b]) for b in range(batch)]
    flips = [t[1] for t in ties if t[1] is not None]
    swaps = all(t[0] for t in ties)
    record(
        "user_split_traces",
        batch=batch,
        max_position=int(pos.max()),
        min_row_pcc=round(p, 6),
        argmax_agree=agree,
        argmax_flips_near_ties=swaps,
        argmax_flips=flips,
        top5_overlap=round(top5, 4),
    )
    # The two attention placements differ per layer by rounding only (split vs unsplit layer output PCC
    # 0.9997-0.9999, optimized-multichip README); 48 chained layers amplify that, so the logits are compared by their
    # ranking. A row may change its argmax only as a near tie within the two placements' own rounding noise
    # (``near_tie``). The exact-agreement version failed the stage-7 policy on the earlier natural-text rows (1 flat row
    # of 32); with the <bos>-prefixed rows the stage-7 policy has no flips, and the rule admits the lower-precision
    # policies' near ties while the PCC / top-5 floors stay (doc/datatype_sweep work log 9-10, 17).
    assert swaps and top5 >= 0.8 and p > 0.9
    gen.reset()


@pytest.mark.parametrize("length", [1, 127, 128, 129, 257, 300, 512, 1000, 1025, 8192])
def test_traced_prefill_matches_eager(gen, length):
    """A single-chunk request replays its bucket's prefill trace; its last-token logits are bitwise equal to the eager
    bucketed path (same programs) and the first sampled token and the decode that follows agree. Re-prefilling the
    same request refreshes no page table / row ids; another request refreshes them once."""
    base = chat(gen, " ".join(PROMPTS) * 200)
    ids = (base * (1 + length // len(base)))[:length]
    row = gen._alloc_row(length + 8)
    kw = dict(page_table=row[None], prompt_lens=[length])
    c0 = dict(gen.counters)
    traced = gen.prefill_forward(torch.tensor([ids]), **kw)[0, 0]
    c1 = dict(gen.counters)
    traced_again = gen.prefill_forward(torch.tensor([ids]), **kw)[0, 0]
    c2 = dict(gen.counters)
    eager = gen.prefill_forward(torch.tensor([ids]), enable_trace=False, **kw)[0, 0]
    toks = {}
    for et in (True, False):
        first = gen.prefill_forward(torch.tensor([ids]), sample_on_device=True, enable_trace=et, **kw)
        t = [int(first[0])]
        for i in range(3):
            t.append(
                int(gen.decode_forward(torch.tensor([t[-1]]), torch.tensor([length + i]), page_table=row[None])[0])
            )
        toks[et] = t
    gen._release(row)
    gen.reset()
    record(
        "traced_prefill",
        length=length,
        bitwise_equal=bool(torch.equal(traced, eager)),
        replays=c1["prefill_replays"] - c0["prefill_replays"],
        input_refreshes_first=c1["prefill_input_refreshes"] - c0["prefill_input_refreshes"],
        input_refreshes_repeat=c2["prefill_input_refreshes"] - c1["prefill_input_refreshes"],
        tokens=toks[True],
    )
    assert c1["prefill_replays"] - c0["prefill_replays"] == 1 and c2["prefill_replays"] - c1["prefill_replays"] == 1
    assert c2["prefill_input_refreshes"] == c1["prefill_input_refreshes"]
    assert torch.equal(traced, eager) and torch.equal(traced, traced_again)
    assert toks[True] == toks[False]


def test_pipelined_generate_matches_blocking(gen):
    """Free-running generate() (pipelined: tokens=None feedback, async token copies, the host PIPELINE_LAG steps
    behind) produces exactly the tokens of the blocking per-step path, with no token write, one position refresh and
    one asynchronous copy per decode step; with EOS stopping, the output is the blocking output cut at the first EOS
    (the in-flight steps after it are discarded)."""
    ids = chat(gen, PROMPTS[2])
    c0 = dict(gen.counters)
    pipe = gen.generate(ids, 40, stop_on_eos=False)
    c1 = dict(gen.counters)
    gen.reset()
    # the blocking path: generate(next_input=...) returning the prediction itself (no teacher-forced change)
    block = gen.generate(ids, 40, next_input=lambda step, pred: pred, stop_on_eos=False)
    gen.reset()
    delta = {k: c1[k] - c0[k] for k in c0}
    # seeded top-k / top-p: the pipelined loop replays the same sampling traces in the same order as the blocking one
    sp = SamplingParams(temperature=1.0, top_k=32, top_p=0.9, seed=99)
    pipe_s = gen.generate(ids, 24, sampling_params=sp, stop_on_eos=False)
    gen.reset()
    block_s = gen.generate(ids, 24, sampling_params=sp, next_input=lambda step, pred: pred, stop_on_eos=False)
    gen.reset()
    gen.set_sampling_params(GREEDY)
    eos_ids = chat(gen, "Reply with exactly one word: yes or no. Is water wet?")
    pipe_eos = gen.generate(eos_ids, 48)  # free-running, stops at EOS
    gen.reset()
    block_eos = gen.generate(eos_ids, 48, next_input=lambda step, pred: pred, stop_on_eos=False)
    gen.reset()
    cut = next((i + 1 for i, t in enumerate(block_eos) if t in gen.eos_token_ids), len(block_eos))
    record(
        "pipelined_generate",
        pipelined=pipe,
        blocking=block,
        counters_delta=delta,
        eos_pipelined=pipe_eos,
        eos_blocking_cut=block_eos[:cut],
        sampled_pipelined=pipe_s,
        sampled_blocking=block_s,
    )
    assert pipe == block
    assert pipe_s == block_s
    # the seeded request must really sample: top-p 0.9 keeps a single candidate wherever the model puts > 90% on its
    # argmax, so on a confident prompt it can reproduce the greedy tokens for every seed; the diversity check samples
    # the full top-32 at temperature 1.5 instead
    sampled_differs = pipe_s != pipe[:24]
    for seed in (7, 1234):
        if sampled_differs:
            break
        sp_k = SamplingParams(temperature=1.5, top_k=32, top_p=1.0, seed=seed)
        sampled_differs = gen.generate(ids, 24, sampling_params=sp_k, stop_on_eos=False) != pipe[:24]
        gen.reset()
    gen.set_sampling_params(GREEDY)
    assert sampled_differs
    assert delta["token_refreshes"] == 0 and delta["position_refreshes"] == 1 and delta["synchronizations"] == 0
    assert delta["token_readbacks"] == 40 and delta["model_replays"] == 39 and delta["sampling_replays"] == 40
    assert cut < 48, "the EOS prompt should finish early"
    assert pipe_eos == block_eos[:cut]


def test_pipelined_batch_rows_finish_at_different_steps(gen):
    """Batch of mixed prompts with EOS stopping in the pipelined loop: rows that finish early go inactive while the
    others continue; every row equals its one-at-a-time (pipelined) generation."""
    prompts = [
        chat(gen, "Reply with exactly one word: yes or no. Is fire hot?"),
        chat(gen, PROMPTS[0]),
        chat(gen, "Answer with a single number: what is 2 + 2?"),
        chat(gen, PROMPTS[5]),
    ]
    alone = []
    for ids in prompts:
        alone.append(gen.generate(ids, 40))
        gen.reset()
    batch = gen.generate(prompts, 40)
    gen.reset()
    record("pipelined_batch_eos", alone=alone, batch=batch, lengths=[len(b) for b in batch])
    assert batch == alone
    assert min(len(b) for b in batch) < 40 and max(len(b) for b in batch) == 40


def test_split_decode_api(gen):
    """decode_forward(read_from_device=False) + read_decode_output(async_read=True) + process_decode_output_host,
    driven like a serving loop with tokens=None (device feedback) after the first step, equals the blocking loop."""
    ids = chat(gen, PROMPTS[4])
    n = len(ids)
    row = gen._alloc_row(n + 16)
    first = int(
        gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)[0]
    )
    reads = []
    for i in range(8):
        gen.decode_forward(
            torch.tensor([first]) if i == 0 else None,
            torch.tensor([n + i]),
            page_table=row[None],
            read_from_device=False,
        )
        reads.append(gen.read_decode_output(async_read=True))
    split = [first] + [int(gen.process_decode_output_host(r, 1)[0]) for r in reads]
    gen.reset()
    # the same split API with host tokens (a scheduler that feeds back what it read): the processed copy of the
    # latest step refreshes the host mirror, so the host tokens are never written back
    row = gen._alloc_row(n + 16)
    host_loop = [
        int(gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)[0])
    ]
    c0 = gen.counters["token_refreshes"]
    for i in range(8):
        gen.decode_forward(
            torch.tensor([host_loop[-1]]), torch.tensor([n + i]), page_table=row[None], read_from_device=False
        )
        host_loop.append(int(gen.process_decode_output_host(gen.read_decode_output(async_read=True), 1)[0]))
    host_refreshes = gen.counters["token_refreshes"] - c0
    gen.reset()
    row = gen._alloc_row(n + 16)
    ref = [
        int(gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)[0])
    ]
    for i in range(8):
        ref.append(int(gen.decode_forward(torch.tensor([ref[-1]]), torch.tensor([n + i]), page_table=row[None])[0]))
    gen.reset()
    record(
        "split_decode_api",
        split=split,
        blocking=ref,
        host_tokens_loop=host_loop,
        host_loop_token_refreshes=host_refreshes,
    )
    assert split == ref and host_loop == ref and host_refreshes == 0


def test_deferred_decode_reads(gen):
    """read_decode_output(defer=True) (serving): the token copy is enqueued only at the generator's next device call.
    In the serving order (submit step N + 1, then process step N), processed with no later device call, and with a
    prefill of another slot submitted while a deferred read is pending (its sample overwrites every token row), every
    processed step equals the blocking loop."""
    ids = chat(gen, PROMPTS[4])
    other = chat(gen, PROMPTS[1])
    n = len(ids)
    row = gen._alloc_row(n + 16)
    ref = [
        int(gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)[0])
    ]
    for i in range(8):
        ref.append(int(gen.decode_forward(torch.tensor([ref[-1]]), torch.tensor([n + i]), page_table=row[None])[0]))
    gen.reset()
    row = gen._alloc_row(n + 16)
    orow = gen._alloc_row(len(other) + 16)
    first = int(
        gen.prefill_forward(torch.tensor([ids]), page_table=row[None], prompt_lens=[n], sample_on_device=True)[0]
    )
    out = [first]
    pending = None
    deferred_before_call = []
    host_token = first  # host input for the next step (request start, and after the other slot's prefill)
    for i in range(8):
        gen.decode_forward(
            torch.tensor([host_token]) if host_token is not None else None,
            torch.tensor([n + i]),
            page_table=row[None],
            read_from_device=False,
        )
        host_token = None
        if pending is not None:  # serving order: step i submitted before step i - 1 is processed
            out.append(int(gen.process_decode_output_host(pending, 1)[0]))
        pending = gen.read_decode_output(async_read=True, defer=True)
        deferred_before_call.append(pending.read is None)
        if i == 3:  # another request's prefill (slot 5) while step 3's copy is still deferred
            gen.prefill_forward(
                torch.tensor([other]), page_table=orow[None], prompt_lens=[len(other)], slots=[5], sample_on_device=True
            )
            assert pending.read is not None, "the prefill must enqueue the deferred copy before its sample"
            # the prefill sample overwrote row 0 of the token buffer: drain, then pass the host token once
            out.append(int(gen.process_decode_output_host(pending, 1)[0]))
            pending, host_token = None, out[-1]
    out.append(int(gen.process_decode_output_host(pending, 1)[0]))  # no later device call: issued here
    gen.reset()
    record("deferred_decode_reads", deferred=out, blocking=ref)
    assert all(deferred_before_call)
    assert out == ref


def test_feedback_rows_after_slot_prefill(gen, expect_error):
    """tokens=None is only accepted for rows whose own latest sample sits in the token buffer: after another slot's
    prefill sampled its first token (overwriting every row), a feedback decode of the other slots is refused; with
    host tokens for that one step, every slot continues exactly as it does alone."""
    a, b = chat(gen, PROMPTS[1]), chat(gen, PROMPTS[6])
    solo = {}
    for name, ids in (("a", a), ("b", b)):
        solo[name] = gen.generate(ids, 12, stop_on_eos=False)
        gen.reset()
    ra, rb = gen._alloc_row(len(a) + 16), gen._alloc_row(len(b) + 16)
    table = torch.full((8, gen.pages), CANARY_BLOCK, dtype=torch.int32)
    table[0], table[5] = ra, rb
    pos = torch.full((8,), -1, dtype=torch.int64)
    out_a = [
        int(gen.prefill_forward(torch.tensor([a]), page_table=ra[None], prompt_lens=[len(a)], sample_on_device=True)[0])
    ]
    toks = torch.zeros(8, dtype=torch.int64)
    toks[0], pos[0] = out_a[0], len(a)
    out = gen.decode_forward(toks, pos, page_table=table)  # host tokens (request start)
    out_a.append(int(out[0]))
    pos[0] += 1
    out_a.append(int(gen.decode_forward(None, pos, page_table=table)[0]))  # device feedback
    pos[0] += 1
    first_b = int(
        gen.prefill_forward(
            torch.tensor([b]), page_table=rb[None], prompt_lens=[len(b)], slots=[5], sample_on_device=True
        )[0]
    )
    out_b = [first_b]
    pos[5] = len(b)
    with expect_error(AssertionError, "tokens=None decodes each active row"):
        gen.decode_forward(None, pos, page_table=table)
    toks[0], toks[5] = out_a[-1], first_b
    out = gen.decode_forward(toks, pos, page_table=table)
    out_a.append(int(out[0]))
    out_b.append(int(out[5]))
    pos[0] += 1
    pos[5] += 1
    for _ in range(4):  # both rows valid again: device feedback
        out = gen.decode_forward(None, pos, page_table=table)
        out_a.append(int(out[0]))
        out_b.append(int(out[5]))
        pos[0] += 1
        pos[5] += 1
    # a new request c replaces b in slot 5 through a logits prefill (first token chosen on the host): slot 5's row
    # still holds b's last sample, so a feedback decode of slot 5 is refused until the host token is passed once
    c = chat(gen, PROMPTS[3])
    rc = gen._alloc_row(len(c) + 16)
    table[5] = rc
    lg = gen.prefill_forward(torch.tensor([c]), page_table=rc[None], prompt_lens=[len(c)], slots=[5])[0, 0]
    lg[gen.suppress_ids] = -1e4
    out_c = [int(lg.argmax())]
    pos[5] = len(c)
    with expect_error(AssertionError, "tokens=None decodes each active row"):
        gen.decode_forward(None, pos, page_table=table)
    toks[0], toks[5] = out_a[-1], out_c[0]
    out = gen.decode_forward(toks, pos, page_table=table)
    out_a.append(int(out[0]))
    out_c.append(int(out[5]))
    pos[0] += 1
    pos[5] += 1
    for _ in range(2):
        out = gen.decode_forward(None, pos, page_table=table)
        out_a.append(int(out[0]))
        out_c.append(int(out[5]))
        pos[0] += 1
        pos[5] += 1
    gen.reset()
    solo_c = gen.generate(c, 4, stop_on_eos=False)
    gen.reset()
    record("feedback_rows_after_slot_prefill", solo=solo, solo_c=solo_c, slot0=out_a, slot5=out_b, slot5_c=out_c)
    assert out_a == solo["a"][: len(out_a)] and out_b == solo["b"][: len(out_b)] and out_c == solo_c
