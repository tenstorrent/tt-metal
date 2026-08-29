# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Batched decode: what running several utterances through one decode step is worth.

A decode step at one row is **bound by reading the AR decoder's weights out of
DRAM**. Every matmul is a matrix against a single row, so nothing amortises the read;
`test_device_decode_bfloat8_weights` measures the same bottleneck from the other side
by halving the weight width. Batching attacks the numerator instead -- one weight read
serves `B` rows -- so the interesting quantity is not the step time, which must grow,
but the *per-utterance* cost, which should fall until the step becomes compute bound.

Two tests, in the order that makes the second meaningful:

1. **Correctness first.** A batched step must compute, for each row, what the same
   row computes alone. Batching is exactly the kind of change that produces plausible
   audio from subtly wrong attention, so this is a PCC gate against the single-row
   path, not a smoke test.
2. **Then the curve.** Step time and per-utterance time at `B = 1, 2, 4, 8`, with the
   crossover reported rather than assumed.

Both run on the moving KV cache: it is the mechanism `TracedDecodeStep` batches, and
`TracedDecodeStepInPlace` is deliberately not batched (65 traces times a batch is a
trace region no board here has -- see `TracedDecodeStep.__init__`).
"""
from __future__ import annotations

import os
import time

import pytest
import torch

from models.demos.cosyvoice.tt.weights import default_weights_path

LLM_WEIGHTS = default_weights_path().replace("hift_", "llm_")

needs_weights = pytest.mark.skipif(not os.path.exists(LLM_WEIGHTS), reason="export llm weights first")
# A batched trace holds `B` copies of the KV buffers and of the widened positional
# projection, so the region is sized for the largest batch in the sweep rather than
# for one utterance.
needs_trace = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 131072, "trace_region_size": 402653184}], indirect=True
)

BATCHES = (1, 2, 4, 8)


def _decoder(device, ttnn):
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta["ar_decoder"]
    return TtARDecoder(device, bag.sub("llm"), meta), meta


def _prefill_rows(device, ttnn, dec, prefix_lens, max_len, d, seeds=None):
    """One right-aligned prefill per row, stacked on the batch axis.

    Deliberately ragged: the rows have different prompt lengths, which is the case a
    real batch presents and the case a left-aligned cache could not serve.

    **`seeds` is explicit, not derived from the loop index.** The correctness test
    calls this once for the whole batch and then once per row, and a seed taken from
    the enclosing loop is 0 on every single-row call -- so rows 1..n would be
    prefilled with different content in the two runs, and the comparison would report
    a model bug that is really a harness bug. (It did, first time round: PCC 0.80 on
    exactly rows 1..3 and 1.0 on row 0.)
    """
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    seeds = seeds if seeds is not None else list(range(len(prefix_lens)))
    per_row = []
    for i, pl in enumerate(prefix_lens):
        torch.manual_seed(100 + seeds[i])
        ys, caches = dec.forward_chunk_fixed(
            dev(torch.randn(1, pl, d) * 0.1),
            dec.empty_cache(max_len, pl),
            max_len,
            valid=pl,
            mask=dev(right_aligned_bias(max_len, pl, pl, causal=True)),
        )
        ttnn.deallocate(ys)
        per_row.append(caches)

    if len(per_row) == 1:
        # `ttnn.concat` of a single tensor returns an **alias** of it, so stacking
        # and then freeing the sources would free the result. One row needs no
        # stacking anyway.
        return per_row[0]
    stacked = []
    for layer in range(len(per_row[0])):
        stacked.append(
            (
                ttnn.concat([c[layer][0] for c in per_row], dim=0),
                ttnn.concat([c[layer][1] for c in per_row], dim=0),
            )
        )
    for caches in per_row:
        TtARDecoder.free_caches(caches)
    return stacked


@needs_weights
@needs_trace
def test_device_batched_decode_matches_single(device):
    """Row `i` of a batched step equals the same row stepped on its own.

    The batch is ragged on purpose -- prefixes of 209, 177, 241 and 193 -- because the
    equal-length case cannot distinguish a correct per-row mask from a mask that
    happens to suit every row. If `right_aligned_bias`'s per-row `valid` were dropped
    or transposed, this is the test that fails; a uniform batch would pass it.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder

    dec, meta = _decoder(device, ttnn)
    d, max_len = meta["input_size"], 384
    prefix_lens = [209, 177, 241, 193]
    b, n_steps = len(prefix_lens), 6

    torch.manual_seed(7)
    tokens = [torch.randn(1, 1, d) * 0.1 for _ in range(n_steps * b)]

    # --- batched
    stacked = _prefill_rows(device, ttnn, dec, prefix_lens, max_len, d, seeds=list(range(b)))
    batched = TracedDecodeStep(dec, max_len, batch=b).capture()
    batched.seed(stacked)
    TtARDecoder.free_caches(stacked)
    got = []
    for s in range(n_steps):
        rows = torch.cat([tokens[s * b + i] for i in range(b)], dim=0)  # [B, 1, d]
        ys = batched.step(rows, [pl + 1 + s for pl in prefix_lens])
        ttnn.synchronize_device(device)
        got.append(ttnn.to_torch(ys).float().clone())
    batched.release()

    # --- one row at a time, same inputs, same steps
    want = []
    for i, pl in enumerate(prefix_lens):
        single = _prefill_rows(device, ttnn, dec, [pl], max_len, d, seeds=[i])
        step = TracedDecodeStep(dec, max_len, batch=1).capture()
        step.seed(single)
        TtARDecoder.free_caches(single)
        rows = []
        for s in range(n_steps):
            ys = step.step(tokens[s * b + i], pl + 1 + s)
            ttnn.synchronize_device(device)
            rows.append(ttnn.to_torch(ys).float().clone())
        step.release()
        want.append(rows)

    worst, worst_at = 1.0, None
    for s in range(n_steps):
        for i in range(b):
            a = got[s][i].reshape(-1)
            e = want[i][s].reshape(-1)
            pcc = float(torch.corrcoef(torch.stack([a, e]))[0, 1])
            if pcc < worst:
                worst, worst_at = pcc, (s, i)
    print(f"\n  batched vs single-row, B={b} ragged prefixes {prefix_lens}, {n_steps} steps")
    print(f"    worst hidden-state PCC {worst:.10f} at step {worst_at[0]} row {worst_at[1]}")
    assert worst >= 0.999, f"batched row diverges from single-row at {worst_at}: PCC {worst}"


@needs_weights
@needs_trace
def test_device_batched_decode_throughput(device):
    """Step time and per-utterance time across the batch sweep.

    Per-utterance time is the figure that matters: a batched step is *slower* than a
    single-row step by construction, and reading only that would say batching hurts.
    What batching changes is how many utterances that step serves.

    Prefixes are equal here, unlike the correctness test -- a throughput number is a
    property of the shapes, and raggedness only affects how much of the batch is
    useful, which `generate_batch` accounts for separately and its docstring explains.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TracedDecodeStep, TtARDecoder

    dec, meta = _decoder(device, ttnn)
    d, max_len, prefix_len, n = meta["input_size"], 384, 209, 32

    rows = []
    for b in BATCHES:
        stacked = _prefill_rows(device, ttnn, dec, [prefix_len] * b, max_len, d)
        step = TracedDecodeStep(dec, max_len, batch=b).capture()
        step.seed(stacked)
        TtARDecoder.free_caches(stacked)
        torch.manual_seed(0)
        x = torch.randn(b, 1, d) * 0.1
        for i in range(2):  # warm the replay path
            step.step(x, [prefix_len + 1 + i] * b)
        ttnn.synchronize_device(device)

        t0 = time.perf_counter()
        for i in range(n):
            step.step(x, [prefix_len + 3 + i] * b)
            ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) / n * 1e3
        step.release()
        dec.release_pos_proj_cache()
        rows.append((b, ms))

    base = rows[0][1]
    print(f"\n  batched AR decode step, max_len={max_len}, mean of {n}")
    print("    batch   step ms   per-utterance ms   utterances/s   speedup vs B=1")
    for b, ms in rows:
        print(f"    {b:5d}   {ms:7.2f}   {ms/b:16.2f}   {1e3*b/ms:12.1f}   {base/(ms/b):14.2f}x")
    best_b, best_ms = min(rows, key=lambda r: r[1] / r[0])
    print(f"    best per-utterance cost at B={best_b}: {best_ms/best_b:.2f} ms ({1e3*best_b/best_ms:.1f} utt/s)")
    print(f"    equivalently {1e3/(best_ms/best_b):.1f} tok/s of aggregate semantic-token throughput")

    # Batching must actually amortise something. If the per-utterance cost at the
    # sweep's best batch is no better than at B=1, the step was not weight-bound
    # after all and this whole path is dead weight -- that is a result worth failing
    # on, not printing.
    assert best_ms / best_b < base * 0.9, (
        f"batching bought nothing: best per-utterance {best_ms/best_b:.2f} ms at B={best_b} "
        f"against {base:.2f} ms at B=1"
    )
