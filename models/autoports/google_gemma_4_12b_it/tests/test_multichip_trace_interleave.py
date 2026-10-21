# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill between replays of a live decode trace on the QB2 mesh (the serving pattern of a full model).

Every prefill length (1000, 2100, 4100, 8193 tokens: the 2D-matmul path, one and two minimal_matmul chunks, all four
norm-stats size classes) is first run once into a scratch user, which compiles its programs. Then a decode trace is
captured, and new requests of those lengths with new content are prefilled into other users' pages, with a trace
replay after each one. Run with ``TT_METAL_TRACE_ALLOC_TRACKING=1``: a persistent device allocation made by a
prefill after the capture (e.g. a lazily created CCL buffer) would be alive at the next replay and make
``execute_trace`` refuse. Every replay (the decoding user's next token) and every prefill is scored against HF.

Programs compiled after a capture (a prefill shape or slice start never seen before) keep device allocations of their
own (the program cache), which the tracker refuses too. A generator therefore prepares every signature before the first
capture and never retires or recaptures traces: fresh requests follow the bucket rule of
``test_unseen_length_in_bucket`` (zero-pad to a warmed physical bucket, own page-table row with user_id 0), and
continuation starts become tensor-valued inputs or warmed starts (README "Prefill program signatures").
"""

import pytest
import torch

from models.autoports.google_gemma_4_12b_it.tests import mc_cases
from models.autoports.google_gemma_4_12b_it.tests.harness import BLOCK, PCC_BAR, Pages, record, row_pcc
from models.autoports.google_gemma_4_12b_it.tests.hf_reference import (
    LAYER_KINDS,
    HFLayer,
    pcc,
    real_layer_state_dict,
    synthetic_hidden,
    text_config,
)


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", list(LAYER_KINDS))
def test_prefill_between_trace_replays(qb2, kind):
    cfg = text_config()
    be = mc_cases.Multichip(qb2)
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    lengths = [1000, 2100, 4100, 8193]
    max_pages = (max(lengths) + 64) // BLOCK + 1
    scratch = len(lengths) + 1  # warm-up user
    dec = be.build(cfg, layer_idx, sd, num_blocks=(len(lengths) + 2) * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=len(lengths) + 2, seed=7)
    pages.ensure(scratch, max(lengths))
    for n in lengths:  # compile every prefill length before the capture
        warm = synthetic_hidden(layer_idx, n, seed=50 + n)
        dec.prefill_forward(be.prefill_in(dec, warm), page_table=be.ints(pages.table[scratch : scratch + 1]))
    # user 0: the decoding request, prefilled before the capture
    x0 = synthetic_hidden(layer_idx, 600 + len(lengths) + 1, seed=1)
    pages.ensure(0, 600 + len(lengths) + 1)
    dec.prefill_forward(be.prefill_in(dec, x0[:, :600]), page_table=be.ints(pages.table), user_id=0)
    cache = ref.new_cache()
    ref.prefill(x0[:, :600], cache)
    trace = mc_cases.TracedDecode(be, dec, 1, pages.table[:1])
    for step, n in enumerate([None] + lengths):
        if n is not None:
            u = step
            x = synthetic_hidden(layer_idx, n, seed=10 + u)
            pages.ensure(u, n)
            # each request passes its own page-table row (user_id 0): the row-slice programs are the warmed ones
            row = be.ints(pages.table[u : u + 1])
            got = be.prefill_out(dec.prefill_forward(be.prefill_in(dec, x), page_table=row))
            row.deallocate(True)  # test-side tensor created after the capture: freed before the next replay
            del row
            want = ref.prefill(x, ref.new_cache(), slice_len=2048)
            whole, worst = pcc(want, got), float(row_pcc(want, got).min())
            record(
                f"mc_trace_interleave_prefill[{kind}]", seq_len=n, pcc=round(whole, 6), worst_row_pcc=round(worst, 6)
            )
            bar = 0.99 if (kind == "sliding" and n >= 8192) else PCC_BAR  # synthetic sliding >= 8192 diagnostic bar
            assert whole >= bar and worst >= bar, (n, whole, worst)
        p = 600 + step
        trace.set_inputs(x0[:, p : p + 1], torch.tensor([p]), pages.table[:1])
        if step == 0:
            trace.capture()
        got_d = be.decode_out(trace.replay())
        want_d = ref.forward(x0[:, p : p + 1], cache, p)
        whole = pcc(want_d, got_d)
        record(f"mc_trace_interleave_decode[{kind}]", pos=p, after_prefill_of=n, pcc=round(whole, 6))
        assert whole >= PCC_BAR, (p, whole)
    trace.release()
    assert all(mc_cases._canary_clean(be, dec))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("kind", list(LAYER_KINDS))
def test_unseen_length_in_bucket(qb2, kind):
    """The bucket rule a generator follows (README "Prefill program signatures"): a fresh request (start 0) of an
    unseen logical length L is zero-padded by the caller to a warmed physical bucket P (here 1000 -> 1024 and
    3000 -> 4096) and passes its own page-table row (user_id 0). Its prefill then runs only programs compiled for P
    before the decode capture, so a trace replay after it is accepted under ``TT_METAL_TRACE_ALLOC_TRACKING=1``.
    Rows [0, L) are scored against HF; the request then decodes from position L through the same trace (its
    page-table row copied into the persistent input), which overwrites the pad rows' cache slots before reading them.
    """
    cfg = text_config()
    be = mc_cases.Multichip(qb2)
    layer_idx = LAYER_KINDS[kind]
    sd = real_layer_state_dict(layer_idx)
    ref = HFLayer(cfg, layer_idx, sd)
    buckets = {1000: 1024, 3000: 4096}
    max_pages = max(buckets.values()) // BLOCK + 2
    dec = be.build(cfg, layer_idx, sd, num_blocks=4 * max_pages + 1)
    pages = Pages(dec.num_blocks, max_pages, num_users=4, seed=9)
    scratch = 3
    pages.ensure(scratch, max(buckets.values()))
    for p_len in buckets.values():  # warm each bucket with a different request
        warm = synthetic_hidden(layer_idx, p_len, seed=60 + p_len)
        dec.prefill_forward(be.prefill_in(dec, warm), page_table=be.ints(pages.table[scratch : scratch + 1]))
    # user 0 decodes through the trace before and between the new prefills
    x0 = synthetic_hidden(layer_idx, 300 + 8, seed=2)
    pages.ensure(0, 308)
    dec.prefill_forward(be.prefill_in(dec, x0[:, :300]), page_table=be.ints(pages.table[0:1]))
    cache0 = ref.new_cache()
    ref.prefill(x0[:, :300], cache0)
    trace = mc_cases.TracedDecode(be, dec, 1, pages.table[0:1])
    trace.set_inputs(x0[:, 300:301], torch.tensor([300]), pages.table[0:1])
    trace.capture()
    assert pcc(ref.forward(x0[:, 300:301], cache0, 300), be.decode_out(trace.replay())) >= PCC_BAR
    p0 = 301
    for u, (n, p_len) in enumerate(buckets.items(), start=1):
        x = synthetic_hidden(layer_idx, n + 2, seed=70 + u)
        padded = torch.cat([x[:, :n], torch.zeros(1, p_len - n, x.shape[-1], dtype=x.dtype)], dim=1)
        pages.ensure(u, p_len)
        row = be.ints(pages.table[u : u + 1])
        got = be.prefill_out(dec.prefill_forward(be.prefill_in(dec, padded), page_table=row))[:, :n]
        row.deallocate(True)
        del row
        cache = ref.new_cache()
        want = ref.prefill(x[:, :n], cache, slice_len=2048)
        whole, worst = pcc(want, got), float(row_pcc(want, got).min())
        record(
            f"mc_bucket_prefill[{kind}]",
            logical_len=n,
            bucket=p_len,
            pcc=round(whole, 6),
            worst_row_pcc=round(worst, 6),
        )
        assert whole >= PCC_BAR and worst >= PCC_BAR, (n, whole, worst)
        # user 0 replay after the new prefill (no new persistent allocation, or the tracker refuses)
        trace.set_inputs(x0[:, p0 : p0 + 1], torch.tensor([p0]), pages.table[0:1])
        assert pcc(ref.forward(x0[:, p0 : p0 + 1], cache0, p0), be.decode_out(trace.replay())) >= PCC_BAR
        p0 += 1
        # the new request decodes from its logical length through the same trace
        for p in (n, n + 1):
            trace.set_inputs(x[:, p : p + 1], torch.tensor([p]), pages.table[u : u + 1])
            got_d = be.decode_out(trace.replay())
            whole_d = pcc(ref.forward(x[:, p : p + 1], cache, p), got_d)
            record(f"mc_bucket_decode[{kind}]", logical_len=n, bucket=p_len, pos=p, pcc=round(whole_d, 6))
            assert whole_d >= PCC_BAR, (n, p, whole_d)
    trace.release()
    assert all(mc_cases._canary_clean(be, dec))
