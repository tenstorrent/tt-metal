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

Two things come off the single prefill, in order:

**periodic continuation** (the hard gate). The prompt is an exactly periodic
token sequence, so correctness at that depth is checkable without an HF
reference: a model whose cache, page table and positions are healthy at
position 202751 continues the period, and one whose are not cannot.

**needle retrieval** (recorded, not gated). A distinctive sentence is planted
near the start of the prompt and queried at the very end, so the query reaches
~200k positions back into the compressed-latent cache. Whether a 30.6B model
with bfloat4_b routed experts *succeeds* at that recall distance is a property
of the checkpoint, not of this port, so the outcome is recorded rather than
asserted; what is asserted is that the deep-cache read yields a sane, peaked
distribution at all.

Results land in ``doc/full_model/full_context.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.model import source_manifest

MODEL_DIR = Path(__file__).resolve().parents[1]
DOC_DIR = MODEL_DIR / "doc" / "full_model"
CONTEXT = 202752
DECODE_STEPS = 8
#: Where the needle goes. Early enough that retrieving it at the end of the
#: prompt is a ~200k-position reach across the paged compressed-latent cache.
NEEDLE_AT = 1024
NEEDLE = " Remember this: the vault passphrase is jade lantern seventeen. "
QUERY = "\nWhat was the vault passphrase? The vault passphrase is"
#: First token of the expected answer, as it follows QUERY.
ANSWER_PREFIX = " jade"
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
    """One prefill, 8 periodic decode steps, then a needle query, ending at 202751.

    Deliberately a single prefill: a second one would double a 40-minute run to
    prove something this one already covers. The phases are laid out so the
    *last* decode position of the whole run is exactly max_valid_position.
    """
    model = gen.model
    assert model.max_seq_len == CONTEXT

    def encode(text):
        return gen.tokenizer.encode(text, add_special_tokens=False)

    needle, query = encode(NEEDLE), encode(QUERY)
    answer_tok = encode(ANSWER_PREFIX)[0]

    # Longest valid non-aligned prompt that still leaves room for the periodic
    # decode steps *and* the teacher-forced needle query inside the context.
    prompt_len = CONTEXT - DECODE_STEPS - len(query)
    stream, period = _periodic_ids(gen, prompt_len + DECODE_STEPS + 1)
    prompt = list(stream[:prompt_len])
    # Planted in place, so every token after it keeps its periodic phase and
    # the continuation check below is unaffected.
    prompt[NEEDLE_AT : NEEDLE_AT + len(needle)] = needle
    assert len(prompt) == prompt_len
    assert prompt_len % 32 and prompt_len % 64 and prompt_len % 2048, "prompt must be non-aligned"
    physical = model.prefill_physical_len(prompt_len)
    assert physical == CONTEXT, physical

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
        seq_len=prompt_len,
        progress_cb=progress,
    )
    enqueue_s = time.perf_counter() - started
    print(f"  prefill enqueued in {enqueue_s:.1f}s; waiting for the device to drain...", flush=True)
    ttnn.synchronize_device(model.mesh_device)
    prefill_s = time.perf_counter() - started
    print(f"  prefill complete in {prefill_s:.1f}s ({prompt_len / prefill_s:.1f} tok/s)", flush=True)

    # This prefill went through the model-level entry point (for the per-layer
    # progress callback), which has no post-compile hook, and a 202733-token
    # prompt compiles programs no bucket warmed. Re-capture before replaying
    # over them: work log FM-016.
    recaptured = gen._maybe_recapture_after_compile(warm_at=prompt_len)
    print(f"  post-prefill trace recapture: {recaptured}", flush=True)

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
    gen.set_decode_positions([prompt_len])
    gen.reset_counters()

    # Phase 1: free-running periodic continuation at positions prompt_len..+7.
    predictions = [first]
    t0 = time.perf_counter()
    for step in range(DECODE_STEPS):
        gen.decode_step_traced()
        predictions.append(gen.read_decode_tokens(1)[0])
        print(f"  decode step {step + 1}/{DECODE_STEPS} at position {prompt_len + step}", flush=True)
    decode_s = time.perf_counter() - t0
    periodic_counters = dict(gen.counters)

    expected = stream[prompt_len : prompt_len + len(predictions)]
    matches = sum(int(a == b) for a, b in zip(predictions, expected))

    # Phase 2: teacher-force the needle query over the remaining positions, so
    # the last one is max_valid_position. Host token refreshes here are teacher
    # forcing by definition; the counter assertions use phase 1.
    last_position = prompt_len + DECODE_STEPS + len(query) - 1
    assert last_position == CONTEXT - 1, last_position
    answer_row = None
    for i, tok in enumerate(query):
        gen.set_decode_tokens([int(tok)])
        if i < len(query) - 1:
            gen.decode_step_traced()
            continue
        # Read the whole distribution for the answer position rather than one
        # sampled token, so a near-miss stays visible. No position advance:
        # this is the last position the cache can represent.
        answer_logits = gen._decode_logits_device(advance_positions=False)
        ttnn.synchronize_device(model.mesh_device)
        answer_row = ttnn.to_torch(answer_logits).to(torch.float32)[0, 0, 0, : model.vocab_size]
        ttnn.deallocate(answer_logits)
    answer_top5 = answer_row.topk(5)
    answer_margin = float(answer_top5.values[0] - answer_row.mean())

    payload = {
        "source_manifest": source_manifest([__file__]),
        "prompt_len": prompt_len,
        "physical_prefill_len": physical,
        "period_tokens": len(period),
        "prefill_enqueue_s": round(enqueue_s, 1),
        "prefill_total_s": round(prefill_s, 1),
        "prefill_tokens_per_s": round(prompt_len / prefill_s, 1),
        "per_layer_enqueue_marks_s": marks,
        "decode_steps": DECODE_STEPS,
        "decode_ms_per_token": round(decode_s / DECODE_STEPS * 1000, 1),
        "first_decode_position": prompt_len,
        "last_decode_position": last_position,
        "last_prompt_position_top5": top5.indices.tolist(),
        "last_prompt_position_margin_over_mean": round(margin, 3),
        "predicted": predictions,
        "expected_periodic_continuation": expected,
        "matches": f"{matches}/{len(predictions)}",
        "decoded": gen.tokenizer.decode(predictions),
        "needle": {
            "gated": False,
            "why_not_gated": (
                "200k-distance recall is a property of the checkpoint (and of the bfloat4_b routed "
                "experts), not of the port; the mechanical gate is the periodic continuation above."
            ),
            "planted_at_position": NEEDLE_AT,
            "planted_text": NEEDLE,
            "query_text": QUERY,
            "query_tokens": len(query),
            "reach_positions": last_position - NEEDLE_AT,
            "expected_first_answer_token": answer_tok,
            "expected_first_answer_text": ANSWER_PREFIX,
            "top5_tokens": answer_top5.indices.tolist(),
            "top5_text": [gen.tokenizer.decode([int(t)]) for t in answer_top5.indices.tolist()],
            "expected_in_top5": answer_tok in answer_top5.indices.tolist(),
            "margin_over_mean": round(answer_margin, 3),
        },
        "post_prefill_trace_recapture": bool(recaptured),
        "counters": {"periodic_phase": periodic_counters, "end_of_run": dict(gen.counters)},
    }
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    (DOC_DIR / "full_context.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: v for k, v in payload.items() if k != "per_layer_enqueue_marks_s"}, indent=2), flush=True)

    assert margin > 5.0, margin
    assert all(0 <= t < model.vocab_size for t in predictions)
    # A healthy cache/page-table/position path at position 202751 continues the
    # period; a broken one cannot. Allow one miss for a bf16 near-tie.
    assert matches >= len(predictions) - 1, (predictions, expected)
    assert periodic_counters["token_input_refreshes"] == 0
    assert periodic_counters["position_refreshes"] == 0
    assert periodic_counters["page_table_refreshes"] == 0
    assert periodic_counters["full_logits_readbacks"] == 0
    # The needle query is not gated on recall, but reading the cache at
    # max_valid_position must still produce a sane, peaked distribution.
    assert torch.isfinite(answer_row).all()
    assert answer_margin > 5.0, answer_margin


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


def test_traced_decode_loop_stops_at_context_end(gen, expect_error):
    """The same limit, but for the *traced* loop, which takes no arguments.

    ``set_decode_positions`` cannot cover this case: a captured trace advances
    the device position itself, so a loop started legally inside the context
    walks off the end without ever calling back into it. The generator mirrors
    the positions on the host and refuses the replay that would step out.
    """
    gen.set_decode_positions([gen.max_seq_len - 1])
    gen.replay_decode_trace()  # the step *at* the last valid position is fine
    with expect_error(ValueError, "step past the supported context"):
        gen.replay_decode_trace()
    with expect_error(ValueError, "step past the supported context"):
        gen.decode_step_traced()
    # An inactive slot is never out of range, whatever the active slots did.
    gen.set_decode_positions([-1] * gen.max_batch_size)
    gen.replay_decode_trace()
    gen.reset()
