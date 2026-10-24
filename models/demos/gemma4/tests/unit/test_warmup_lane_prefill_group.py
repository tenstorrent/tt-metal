# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Boot-time warmups of the vLLM bridge, host-only.

The shared prefill path re-runs ``warmup_model_prefill`` on every prefill batch (its
``warmup_prefill`` default), so what that override does must be once per process:

* the lane-group prefill path (``lane_prefill_group``) runs once, before the prefill
  sweep, with one slot per lane -- its first group otherwise compiled and allocated
  minutes into serving, behind the live decode traces (26B Galaxy hang);
* the vision dummy warmup runs once (it cost ~170 ms of TTFT per prefill batch).

Also the decode-trace allocation guard: past its re-capture budget it still re-captures,
rate-limited, instead of letting replays scribble over later allocations.
"""
import time
from types import SimpleNamespace

import pytest
import torch

# generator_vllm imports vllm at module scope (through tt_transformers), so
# COLLECTING this file fails on a runner without vLLM -- the tt-metal unit job.
pytest.importorskip("vllm")

from models.demos.gemma4.tt import generator_vllm
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin
from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

LANES, LANE_SLOTS, CHUNK, BLOCKS = 4, 32, 4096, 32


def _instance(monkeypatch, *, lane_router: bool, lane_sharded: bool = True, multimodal: bool = True, prompts: int = 1):
    """A bridge instance with everything ``warmup_model_prefill`` touches stubbed on the host."""
    calls = {"lanes": [], "vision": 0, "sweep": 0}
    monkeypatch.setattr(
        generator_vllm, "warmup_gemma4_model_prefill", lambda *a, **k: calls.__setitem__("sweep", calls["sweep"] + 1)
    )
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda cls: lane_router))
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda cls: int(lane_router) * prompts))
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1")
    monkeypatch.setattr(
        Gemma4ForCausalLM, "warmup_vision", lambda self: calls.__setitem__("vision", calls["vision"] + 1)
    )
    inst = object.__new__(Gemma4ForCausalLM)
    inst._bounded_sliding_kv_cache = True  # eager prefill warmup (no capture decision to make)
    inst._decode_warmup_complete = True  # the plugin's trace pass: the eager decode warmup has run
    inst.is_multimodal = multimodal
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lane_sharded=lane_sharded, lanes=LANES),
        lane_slots=LANE_SLOTS,
        mesh_device=SimpleNamespace(shape=(8, 4)),
    )
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK)]
    inst._mock_tokens = lambda b, s, kv, m: {
        "tokens": torch.zeros(b, s, dtype=torch.long),
        "page_table": torch.arange(b * BLOCKS, dtype=torch.int32).reshape(b, BLOCKS),
    }

    def _lanes(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        model._active_page_tables_per_layer = ["set by the lane path"]
        calls["lanes"].append((tuple(tokens.shape), tuple(page_tables.shape), list(prompt_lens), list(slot_ids)))
        return torch.zeros(tokens.shape[0] * (tokens.shape[1] if tokens.dim() == 3 else 1), 8)

    inst.prefill_forward_lanes = _lanes
    return inst, model, calls


def test_lane_group_and_vision_warm_once_across_repeated_calls(monkeypatch):
    inst, model, calls = _instance(monkeypatch, lane_router=True)
    kv = [object()]
    for _ in range(3):  # plugin phase 1, phase 2, then every serving prefill batch
        Gemma4ForCausalLM.warmup_model_prefill(inst, kv, enable_trace=True, can_sample_on_device=False)
    assert calls["sweep"] == 3, "the prefill sweep keeps its own once-only gate"
    assert calls["lanes"] == [((LANES, CHUNK), (LANES, 1, BLOCKS), [CHUNK] * LANES, [0, 32, 64, 96])]
    assert calls["vision"] == 1
    assert not hasattr(model, "_active_page_tables_per_layer"), "the warmup leaves no per-layer stash behind"


def test_lane_group_warmup_waits_for_the_eager_decode_warmup(monkeypatch):
    """The plugin's first (eager) pass must not run it: a lane-parallel prefill before
    the decode graph's first step hung that step at boot. The second (trace) pass does."""
    inst, _model, calls = _instance(monkeypatch, lane_router=True, multimodal=False)
    inst._decode_warmup_complete = False
    kv = [object()]
    Gemma4ForCausalLM.warmup_model_prefill(inst, kv, enable_trace=False, can_sample_on_device=False)
    assert calls["lanes"] == []
    inst._decode_warmup_complete = True
    Gemma4ForCausalLM.warmup_model_prefill(inst, kv, enable_trace=True, can_sample_on_device=False)
    assert len(calls["lanes"]) == 1


def test_lane_batched_warmup_adds_the_k_users_per_lane_frame(monkeypatch):
    """k prompts per lane: the batched walk's programs and its page-table frame (the lanes' decode frame) must
    exist before the decode traces are captured, so the warmup adds one lane-batched call."""
    inst, model, calls = _instance(monkeypatch, lane_router=True, multimodal=False, prompts=8)
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["lanes"] == [
        ((LANES, CHUNK), (LANES, 1, BLOCKS), [CHUNK] * LANES, [0, 32, 64, 96]),
        (
            (LANES, 8, CHUNK // 8),
            (LANES, 8, BLOCKS),
            [CHUNK // 8] * (LANES * 8),
            [ln * 32 + r for ln in range(LANES) for r in range(8)],
        ),
    ]
    assert not hasattr(model, "_active_page_tables_per_layer")
    inst, _model, calls = _instance(monkeypatch, lane_router=True, multimodal=False, prompts=8)
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "0")
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert len(calls["lanes"]) == 1, "without the batched walk there is no frame to warm"


@pytest.mark.parametrize("why", ["router_off", "not_lane_sharded", "no_kv_cache"])
def test_lane_group_warmup_is_skipped_when_it_cannot_apply(monkeypatch, why):
    inst, _model, calls = _instance(
        monkeypatch, lane_router=why != "router_off", lane_sharded=why != "not_lane_sharded", multimodal=False
    )
    kv = None if why == "no_kv_cache" else [object()]
    Gemma4ForCausalLM.warmup_model_prefill(inst, kv, enable_trace=False, can_sample_on_device=False)
    assert calls["lanes"] == [] and calls["sweep"] == 1 and calls["vision"] == 0


def _guard(monkeypatch, *, recaptures: int, last_recapture_age_s: float):
    g = object.__new__(ChunkedPrefillPageTableGuardMixin)
    g._g4_trace_alloc_baseline = {"k": (100, 0)}
    g._g4_alloc_check_pending = True
    g._g4_trace_guard_recaptures = recaptures
    g._g4_trace_guard_last_recapture_t = time.monotonic() - last_recapture_age_s
    monkeypatch.setattr(ChunkedPrefillPageTableGuardMixin, "_g4_alloc_levels", lambda self: (200, 0))
    return g


def test_guard_recaptures_within_budget_then_rate_limited(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "8")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MIN_INTERVAL_S", "30")
    g = _guard(monkeypatch, recaptures=3, last_recapture_age_s=1.0)
    assert g._g4_trace_guard_should_recapture("k") is True and g._g4_trace_guard_recaptures == 4
    g = _guard(monkeypatch, recaptures=8, last_recapture_age_s=1.0)
    assert g._g4_trace_guard_should_recapture("k") is False, "budget spent and the interval has not elapsed"
    assert g._g4_trace_guard_recaptures == 8
    g = _guard(monkeypatch, recaptures=8, last_recapture_age_s=31.0)
    assert g._g4_trace_guard_should_recapture("k") is True, "budget spent but the interval elapsed"
    assert g._g4_trace_guard_recaptures == 9


def test_guard_interval_zero_keeps_the_hard_limit(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "8")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MIN_INTERVAL_S", "0")
    g = _guard(monkeypatch, recaptures=8, last_recapture_age_s=1e6)
    assert g._g4_trace_guard_should_recapture("k") is False
