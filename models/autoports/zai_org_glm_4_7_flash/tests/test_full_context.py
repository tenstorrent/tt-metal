# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The advertised 202752-token context, exercised through the 47-layer model.

The decoder stages proved 202751 prefill + decode one layer at a time. This is
the full-stack version: the whole model, the real terminal path, the real
sampler, decoding right up to max_valid_position (202751).

It is slow by construction and by a wide margin the longest thing in this
stage: the optimized decoder prefills 202751 tokens in ~51 s *per MoE layer*,
so 47 layers is around 40 minutes of device time, and the host spends about as
long enqueueing it. Run it deliberately, with the environment below so the
progress lines are not swallowed by stdout buffering:

    PYTHONUNBUFFERED=1 pytest \\
        models/autoports/zai_org_glm_4_7_flash/tests/test_full_context.py -q -s

The repo's ``pytest.ini`` sets a 300 s per-test timeout, which this exceeds by
design, hence the explicit ``@pytest.mark.timeout``.

Correctness at that depth is checkable without an HF reference because the
prompt is an exactly periodic token sequence: a model whose cache, page table
and positions are healthy at position 202751 continues the period, and one
whose are not cannot. Results land in ``doc/full_model/full_context.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "full_model"
DECODE_STEPS = 8
#: Longest valid non-aligned prompt that still leaves room for DECODE_STEPS
#: decode positions inside the context: the last step lands exactly on
#: max_valid_position (202751). 202744 is not a multiple of 32, 64, any prefill
#: bucket, or the 2048 prefill chunk, and still buckets to a full 202752
#: physical prefill.
PROMPT_LEN = 202752 - DECODE_STEPS
TRACE_REGION_SIZE = 350_000_000


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768, trace_region_size=TRACE_REGION_SIZE
    )
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def gen(device):
    generator = build_generator(MODEL_DIR, device, warmup_prefill_lens=())
    yield generator
    generator.teardown()


def _periodic_ids(gen, length):
    """An exactly periodic token stream, plus its period."""
    text = (
        "The quick brown fox jumps over the lazy dog. Tenstorrent builds AI accelerators in Toronto. "
        "Seventeen ravens counted the blue kettle and then flew north over the quiet river. "
    )
    period = gen.tokenizer.encode(text, add_special_tokens=False)
    assert len(period) > 16, len(period)
    reps = -(-length // len(period)) + 1
    return (period * reps)[:length], period


@pytest.mark.timeout(10800)
def test_full_context_prefill_and_decode(gen):
    """One 202744-token prefill and 8 traced decode steps ending at 202751.

    Deliberately a single prefill: a second one would double a 40-minute run to
    prove something this one already covers.
    """
    model = gen.model
    assert model.max_seq_len == 202752
    stream, period = _periodic_ids(gen, PROMPT_LEN + DECODE_STEPS + 1)
    prompt = stream[:PROMPT_LEN]
    physical = model.prefill_physical_len(PROMPT_LEN)
    assert physical == 202752, physical

    gen.reset()
    started = time.perf_counter()
    marks = []

    def progress(layer_idx, total):
        now = time.perf_counter() - started
        marks.append(round(now, 1))
        print(f"  prefill layer {layer_idx + 1}/{total} enqueued at {now:7.1f}s", flush=True)

    logits, row = model.prefill_forward_last_logits_device(
        prompt,
        kv_cache=gen._kv_cache,
        page_table=gen._page_table_dev,
        seq_len=PROMPT_LEN,
        progress_cb=progress,
    )
    enqueue_s = time.perf_counter() - started
    print(f"  prefill enqueued in {enqueue_s:.1f}s; waiting for the device to drain...", flush=True)
    ttnn.synchronize_device(model.mesh_device)
    prefill_s = time.perf_counter() - started
    print(f"  prefill complete in {prefill_s:.1f}s ({PROMPT_LEN / prefill_s:.1f} tok/s)", flush=True)

    # The final prompt position's logits must be a real distribution, not drift.
    host_row = ttnn.to_torch(logits).to(torch.float32)[0, 0, row, : model.vocab_size]
    assert torch.isfinite(host_row).all()
    top5 = host_row.topk(5)
    margin = float(top5.values[0] - host_row.mean())
    print(f"  last-position top5 {top5.indices.tolist()} margin {margin:.2f}", flush=True)

    # Same on-device sampler the decode loop uses, on the same logits.
    gen.sampling.sample(logits=logits, tt_out_tok=gen._prefill_tokens_dev, enable_trace=False)
    ttnn.deallocate(logits)
    first = int(ttnn.to_torch(gen._prefill_tokens_dev).reshape(-1)[row].item())
    gen.set_decode_tokens([first])
    gen.set_decode_positions([PROMPT_LEN])
    gen.reset_counters()

    predictions = [first]
    t0 = time.perf_counter()
    for step in range(DECODE_STEPS):
        gen.decode_step_traced()
        predictions.append(gen.read_decode_tokens(1)[0])
        print(f"  decode step {step + 1}/{DECODE_STEPS} at position {PROMPT_LEN + step}", flush=True)
    decode_s = time.perf_counter() - t0

    expected = stream[PROMPT_LEN : PROMPT_LEN + len(predictions)]
    matches = sum(int(a == b) for a, b in zip(predictions, expected))
    payload = {
        "prompt_len": PROMPT_LEN,
        "physical_prefill_len": physical,
        "period_tokens": len(period),
        "prefill_enqueue_s": round(enqueue_s, 1),
        "prefill_total_s": round(prefill_s, 1),
        "prefill_tokens_per_s": round(PROMPT_LEN / prefill_s, 1),
        "per_layer_enqueue_marks_s": marks,
        "decode_steps": DECODE_STEPS,
        "decode_ms_per_token": round(decode_s / DECODE_STEPS * 1000, 1),
        "first_decode_position": PROMPT_LEN,
        "last_decode_position": PROMPT_LEN + DECODE_STEPS - 1,
        "last_prompt_position_top5": top5.indices.tolist(),
        "last_prompt_position_margin_over_mean": round(margin, 3),
        "predicted": predictions,
        "expected_periodic_continuation": expected,
        "matches": f"{matches}/{len(predictions)}",
        "decoded": gen.tokenizer.decode(predictions),
        "counters": dict(gen.counters),
    }
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "full_context.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_layer_enqueue_marks_s"}, indent=2), flush=True)

    assert margin > 5.0, margin
    assert all(0 <= t < model.vocab_size for t in predictions)
    # A healthy cache/page-table/position path at position 202751 continues the
    # period; a broken one cannot. Allow one miss for a bf16 near-tie.
    assert matches >= len(predictions) - 1, (predictions, expected)
    assert gen.counters["token_input_refreshes"] == 0
    assert gen.counters["position_refreshes"] == 0
    assert gen.counters["page_table_refreshes"] == 0
    assert gen.counters["full_logits_readbacks"] == 0
    assert PROMPT_LEN + DECODE_STEPS - 1 == model.max_seq_len - 1, "last decode step must be max_valid_position"


def test_decode_position_past_context_is_rejected(gen, expect_error):
    """A position the paged cache cannot represent must fail on the host.

    Driving `paged_update_cache` past the end of the page table does not fail
    on device - it wedges it, and every later read hangs behind the wedged
    queue. This stage hit that twice while building the test above, so the
    generator now refuses the position instead.
    """
    with expect_error(ValueError, "decode positions must be in"):
        gen.set_decode_positions([gen.max_seq_len])
    with expect_error(ValueError, "decode positions must be in"):
        gen.set_decode_positions([-2])
    gen.set_decode_positions([gen.max_seq_len - 1])  # the last valid position is fine
