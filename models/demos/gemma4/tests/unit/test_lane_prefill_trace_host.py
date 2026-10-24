# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The traced lane prefill (item (a) of the Galaxy plan), host-only: the bucket set (``tt/lane_prefill_trace.py``),
the replay decision, the page-table width fitting, the router's use of a captured frame (a single prompt routed
through the frame, the replay, the eager fallback) and the boot warmup's capture. No device: the capture and the
replay are recorded stubs on a stubbed bridge.

    pytest models/demos/gemma4/tests/unit/test_lane_prefill_trace_host.py -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.lane_prefill_trace import (
    fit_table_width,
    lane_prefill_trace_buckets,
    lane_prefill_trace_choice,
    lane_trace_rounds_for_call,
    parse_lane_prefill_trace_buckets,
)

pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM  # noqa: E402
from models.tt_transformers.tt.common import get_padded_prefill_len  # noqa: E402

LANES, LANE_SLOTS, CHUNK, VOCAB, WIDTH = 4, 32, 4096, 8, 6
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention"]
SCRATCH = {0: 7, 1: 0, 2: 7}


# ── buckets, choice, widths ──────────────────────────────────────────────────────────────────────────────────


def test_bucket_parsing_keeps_frames_within_the_chunk_the_slots_and_the_trace_lengths():
    kept, dropped = parse_lane_prefill_trace_buckets(
        "1x128, 2x128,8x512,3x200,40x128,1x8192,bad,,", max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=False
    )
    assert kept == {(1, 128), (2, 128), (8, 512)}
    assert dropped == ["3x200", "40x128", "1x8192", "bad"]
    kept, dropped = parse_lane_prefill_trace_buckets(
        "1x128,2x128,8x512", max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=True
    )
    assert kept == {(1, 128)}, "bounded sliding: one valid length per device, so one round per column"
    assert dropped == ["2x128", "8x512"]


def test_buckets_follow_the_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_LANE_PREFILL_TRACE", raising=False)
    monkeypatch.delenv("GEMMA4_LANE_PREFILL_TRACE_BUCKETS", raising=False)
    assert lane_prefill_trace_buckets(max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=True) == (set(), [])
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE", "1")
    assert lane_prefill_trace_buckets(max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=True) == ({(1, 128)}, [])
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE_BUCKETS", "1x128,4x512")
    assert lane_prefill_trace_buckets(max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=False)[0] == {
        (1, 128),
        (4, 512),
    }


def test_replay_choice_needs_an_exact_single_chunk_cold_match():
    captured = {(1, 128), (4, 256)}
    assert lane_prefill_trace_choice(captured, 1, 128) == (1, 128)
    assert lane_prefill_trace_choice(captured, 4, 256) == (4, 256)
    assert lane_prefill_trace_choice(captured, 2, 128) is None
    assert lane_prefill_trace_choice(captured, 1, 256) is None
    assert lane_prefill_trace_choice(captured, 1, 128, single_chunk=False) is None
    assert lane_prefill_trace_choice(captured, 1, 128, cached_prefix=True) is None
    assert lane_prefill_trace_choice(set(), 1, 128) is None


def test_traced_call_sizes_win_over_larger_eager_ones():
    captured = {(1, 128), (4, 128), (2, 512)}
    assert lane_trace_rounds_for_call(captured, 128, 32) == 4
    assert lane_trace_rounds_for_call(captured, 128, 3) == 1
    assert lane_trace_rounds_for_call(captured, 512, 8) == 2
    assert lane_trace_rounds_for_call(captured, 1024, 8) == 8, "no bucket at this length: the eager size"
    assert lane_trace_rounds_for_call(set(), 128, 8) == 8


def test_fit_table_width_trims_or_null_pads_the_last_dim_only():
    pt = torch.arange(1, 1 + 4 * 6, dtype=torch.int32).reshape(4, 6)
    assert fit_table_width(pt, 6) is pt
    assert torch.equal(fit_table_width(pt, 2), pt[:, :2])
    wide = fit_table_width(pt, 9)
    assert tuple(wide.shape) == (4, 9) and torch.equal(wide[:, :6], pt) and not wide[:, 6:].any()
    frame = torch.ones(LANES * LANE_SLOTS, 3, dtype=torch.int32)
    assert tuple(fit_table_width(frame, 5).shape) == (LANES * LANE_SLOTS, 5)
    assert fit_table_width(None, 5) is None and fit_table_width("not a table", 5) == "not a table"


# ── the router with a captured frame ─────────────────────────────────────────────────────────────────────────


def _bridge(monkeypatch, *, prompts, captured, replay_returns=True):
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1")
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda cls: True))
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda cls: prompts))
    calls = {"replays": [], "eager": [], "installs": [], "serial": []}
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES),
        lane_slots=LANE_SLOTS,
        hf_config=SimpleNamespace(layer_types=LAYER_TYPES),
        vocab_size=VOCAB,
        _lane_scratch_block_for=SCRATCH.__getitem__,
        update_persistent_per_layer_page_tables=lambda tables, force=False: calls["installs"].append(len(tables)),
    )
    inst = object.__new__(Gemma4ForCausalLM)
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK)]
    inst._g4_lane_prefill_traces = {b: {"trace_id": 1} for b in captured}

    def _replay(bucket, tokens, page_tables, per_layer, kv_cache, prompt_lens):
        calls["replays"].append(
            (
                bucket,
                tuple(tokens.shape),
                tuple(page_tables.shape),
                [tuple(t.shape) for t in per_layer],
                list(prompt_lens),
            )
        )
        if not replay_returns:
            return None
        k = bucket[0]
        return (500 + torch.arange(LANES * k, dtype=torch.float32)).reshape(-1, 1).repeat(1, VOCAB)

    def _eager(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        calls["eager"].append((tuple(tokens.shape), list(prompt_lens)))
        k = int(tokens.shape[1]) if tokens.dim() == 3 else 1
        return (100 + torch.arange(LANES * k, dtype=torch.float32)).reshape(-1, 1).repeat(1, VOCAB)

    def _serial(page_tables_per_layer=None, **kwargs):
        calls["serial"].append(list(kwargs["empty_slots"]))
        return torch.full((int(kwargs["tokens"].shape[0]), 1, VOCAB), -1.0)

    inst._g4_lane_prefill_trace_replay = _replay
    inst.prefill_forward_lanes = _eager
    inst.prefill_forward = _serial
    return inst, calls


def _step(slots, plens, seq=100):
    B = len(slots)
    tokens = torch.zeros(B, seq, dtype=torch.long)
    for q in range(B):
        tokens[q, : plens[q]] = 1 + q
    kwargs = {"tokens": tokens, "empty_slots": list(slots), "prompt_lens": list(plens), "kv_cache": [object()]}
    tables = [torch.arange(1, 1 + B * WIDTH, dtype=torch.int32).reshape(B, WIDTH) * (i + 1) for i in range(3)]
    return kwargs, tables


def test_a_single_prompt_routes_through_the_captured_frame_with_scratch_lanes(monkeypatch):
    inst, calls = _bridge(monkeypatch, prompts=1, captured={(1, 128)})
    kwargs, tables = _step([33], [50])  # one request on lane 1: B < lanes, which the eager router rejects
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and tuple(out.shape) == (1, 1, VOCAB) and calls["serial"] == [] and calls["eager"] == []
    ((bucket, toks, pts, per_layer, plens),) = calls["replays"]
    assert bucket == (1, 128) and toks == (LANES, 1, 128) and pts == (LANES, 1, WIDTH)
    assert per_layer == [(LANES, WIDTH)] * 3 and plens == [50] * LANES, "scratch lanes carry the round's length"
    assert float(out[0, 0, 0]) == 500 + 1 * 1 + 0, "lane 1's row of the replay"
    assert calls["installs"] == [], "the replay uploads the fitted tables itself (the stub does not)"


def test_without_a_captured_frame_a_single_prompt_stays_on_the_eager_owner_path(monkeypatch):
    inst, calls = _bridge(monkeypatch, prompts=1, captured=set())
    kwargs, tables = _step([33], [50])
    assert inst._try_lane_parallel_prefill(kwargs, tables) is None
    assert calls["replays"] == [] and calls["eager"] == []


def test_a_captured_round_size_splits_a_gathered_step_into_traced_calls(monkeypatch):
    inst, calls = _bridge(monkeypatch, prompts=8, captured={(1, 128)})
    slots = [0, 32, 64, 96, 1, 33, 65, 97]
    kwargs, tables = _step(slots, [50] * 8)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert [c[0] for c in calls["replays"]] == [(1, 128), (1, 128)], "two one-round replays, not one 2-round walk"
    assert calls["eager"] == []
    assert float(out[5, 0, 0]) == 500 + 1, "request 5 = lane 1, second round"


def test_a_mismatched_shape_stays_eager_and_a_refused_replay_falls_back(monkeypatch):
    inst, calls = _bridge(monkeypatch, prompts=8, captured={(1, 128)})
    kwargs, tables = _step([0, 32, 64, 96], [300] * 4, seq=300)  # a longer frame: no bucket at its padded length
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert calls["replays"] == [] and [c[0] for c in calls["eager"]] == [(LANES, get_padded_prefill_len(300))]

    inst, calls = _bridge(monkeypatch, prompts=1, captured={(1, 128)}, replay_returns=False)
    kwargs, tables = _step([0, 32, 64, 96], [50] * 4)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert len(calls["replays"]) == 1 and [c[0] for c in calls["eager"]] == [(LANES, 128)]
    assert calls["installs"] == [3], "the eager fallback uploads the call's tables"
    assert float(out[2, 0, 0]) == 100 + 2


# ── the boot warmup's capture ────────────────────────────────────────────────────────────────────────────────


def _warm_instance(monkeypatch, *, trace_env, prompts=1, bounded=True, capture_raises=False):
    from models.demos.gemma4.tests.unit.test_warmup_lane_prefill_group import _instance

    inst, model, calls = _instance(monkeypatch, lane_router=True, multimodal=False, prompts=prompts)
    if trace_env is None:
        monkeypatch.delenv("GEMMA4_LANE_PREFILL_TRACE", raising=False)
    else:
        monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE", trace_env)
    monkeypatch.delenv("GEMMA4_LANE_PREFILL_TRACE_BUCKETS", raising=False)
    inst._bounded_sliding_kv_cache = bounded
    calls["captures"] = []

    def _capture(bucket, kv):
        calls["captures"].append(bucket)
        if capture_raises:
            raise RuntimeError("trace region full")
        return True

    inst._g4_capture_lane_prefill_trace = _capture
    return inst, model, calls


def test_warmup_captures_the_configured_frames_after_the_eager_ones(monkeypatch):
    inst, model, calls = _warm_instance(monkeypatch, trace_env="1")
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert len(calls["lanes"]) == 1 and calls["captures"] == [(1, 128)]
    assert not hasattr(model, "_active_page_tables_per_layer")


def test_warmup_captures_nothing_when_the_trace_is_off_and_survives_a_failed_capture(monkeypatch):
    inst, _model, calls = _warm_instance(monkeypatch, trace_env=None)
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["captures"] == []
    inst, model, calls = _warm_instance(monkeypatch, trace_env="1", capture_raises=True)
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["captures"] == [(1, 128)] and not hasattr(model, "_active_page_tables_per_layer")
