# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for image / video requests on the Qwen3.8 MTP spec-decode class (step 1a): the processor-info
mm limits, Qwen36MTPContractForCausalLM.prefill_forward computing vision tokens and handing them to the engine, and
MTPSpecEngine threading ``vision_tokens`` through prefill -> _prefill_row -> the fast / B==1 / eager routes (with the
vision row offset of a segment that starts after some placeholders). Fakes of test_spec_fastprefill_host.py.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_spec_mm_host.py -q
"""
import types

import torch

from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import (
    VOCAB,
    _dispatch_engine,
    _engine,
    _fake_model,
    _row,
    fake_ttnn,
)
from models.demos.blackhole.qwen36.tt import qwen36_vllm
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.qwen36_vllm import (
    Qwen36ForCausalLM,
    Qwen36MTPContractForCausalLM,
    TT_Qwen3_5ProcessingInfo,
    TT_Qwen3_5SpecProcessingInfo,
)
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

__all__ = ["fake_ttnn"]  # fixture re-export

IMG = 7  # image placeholder token id of the fakes
VIS = "VISION_ROWS"  # stands for the packed vision rows (a device tensor in production)


def _mm_prompt(L, n_img, first=130):
    """L tokens (ids >= 10, never IMG) with ``n_img`` IMG placeholders starting at position ``first``."""
    toks = [10 + (i % 90) for i in range(L)]
    toks[first : first + n_img] = [IMG] * n_img
    return toks


def _with_vis_offset(model):
    """Give the fake model the real _vis_row_offset_for (image placeholder counting)."""
    model._req_video_grid_thw = None
    model.args = types.SimpleNamespace(hf_config=types.SimpleNamespace(image_token_id=IMG, video_token_id=IMG + 1))
    model._vision_placeholder_token_id = lambda: Qwen36Model._vision_placeholder_token_id(model)
    model._vis_row_offset_for = lambda ids, cs: Qwen36Model._vis_row_offset_for(model, ids, cs)
    return model


# --------------------------------------------------------------------------- #
# (a) mm limits
# --------------------------------------------------------------------------- #
def test_spec_processing_info_mm_limits_match_non_spec():
    spec = TT_Qwen3_5SpecProcessingInfo.get_supported_mm_limits(None)
    assert spec == {"image": 1, "video": 1}
    assert spec == TT_Qwen3_5ProcessingInfo.get_supported_mm_limits(None)


# --------------------------------------------------------------------------- #
# (b) MTP prefill_forward
# --------------------------------------------------------------------------- #
class _FakeEngine:
    def __init__(self):
        self.calls = []

    def prefill(self, slot, ids, row, *args, **kw):
        self.calls.append((slot, ids.clone(), args, kw))
        return torch.arange(VOCAB, dtype=torch.float32)


def _gen(monkeypatch, prefix_cache=True):
    g = Qwen36MTPContractForCausalLM.__new__(Qwen36MTPContractForCausalLM)
    eng = _FakeEngine()
    tower = []
    model = types.SimpleNamespace()
    model.get_image_features = lambda pix, grid: tower.append(("image", tuple(pix.shape), grid.tolist())) or "IMG_ROWS"
    model.get_video_features = lambda pix, grid: tower.append(("video", tuple(pix.shape), grid.tolist())) or "VID_ROWS"
    g.model = [model]
    g._ensure_ready = lambda kv_cache=None: eng
    g._table = lambda pt: pt.to(torch.int32)
    monkeypatch.setattr(qwen36_vllm, "_spec_prefix_cache_enabled", lambda: prefix_cache)
    return g, eng, tower


def _call(g, N=2, **kw):
    tokens = torch.arange(1, 1 + N * 12, dtype=torch.int32).reshape(N, 12)
    pt = torch.ones(N, 4, dtype=torch.int32)
    return g.prefill_forward(tokens, pt, None, torch.tensor([10] * N), empty_slots=list(range(N)), **kw)


def test_prefill_forward_mm_computes_vision_and_passes_it(monkeypatch):
    g, eng, tower = _gen(monkeypatch)
    pix = [torch.zeros(16, 8), None]  # user 0 carries an image, user 1 is text
    grid = [torch.tensor([[1, 4, 4]]), None]
    logits, deltas = _call(g, pixel_values=pix, image_grid_thw=grid, start_pos=torch.tensor([64, 64]))
    assert tower == [("image", (16, 8), [[1, 4, 4]])]  # lazily, only for the mm user
    (s0, _, a0, k0), (s1, _, a1, k1) = eng.calls
    # mm user: vision tokens, no prefix-cache resume even though vLLM reported start_pos 64 (recompute from 0)
    assert (s0, k0) == (0, {"start_pos": None, "vision_tokens": "IMG_ROWS"})
    # text user: unchanged call, keeps its vLLM start_pos
    assert (s1, a1, k1) == (1, (), {"start_pos": 64})
    assert logits.shape == (2, 1, VOCAB) and deltas.shape == (2,) and deltas.dtype == torch.long and not deltas.any()


def test_prefill_forward_video_takes_precedence_and_no_raise(monkeypatch):
    g, eng, tower = _gen(monkeypatch)
    out = _call(
        g,
        N=1,
        pixel_values=[torch.zeros(4, 8)],
        image_grid_thw=[torch.tensor([[1, 2, 2]])],
        pixel_values_videos=[torch.zeros(8, 8)],
        video_grid_thw=[torch.tensor([[2, 2, 2]])],
    )
    assert [t[0] for t in tower] == ["video"] and eng.calls[0][3]["vision_tokens"] == "VID_ROWS"
    assert out[0].shape == (1, 1, VOCAB)


def test_prefill_forward_empty_placeholder_is_text_exactly_as_before(monkeypatch):
    """vLLM attaches empty pixel_values to text requests; no tower call, calls identical to a plain text request."""
    for pc, start, expect in ((True, None, {}), (True, torch.tensor([0, 64]), None), (False, None, {})):
        g, eng, tower = _gen(monkeypatch, prefix_cache=pc)
        kw = dict(pixel_values=[[], []], image_grid_thw=[None, None])
        if start is not None:
            kw["start_pos"] = start
        _call(g, **kw)
        assert tower == []
        for _, _, args, k in eng.calls:
            assert "vision_tokens" not in k
        if expect is None:
            assert [c[3] for c in eng.calls] == [{"start_pos": 0}, {"start_pos": 64}]
        else:
            assert [(c[2], c[3]) for c in eng.calls] == [((), {}), ((), {})]


def test_mtp_warmup_and_init_reach_vision(monkeypatch):
    """initialize_vllm_model goes through the base (init_vision_model); warmup_model_prefill runs warmup_vision first."""
    import inspect

    assert "init_vision_model" in inspect.getsource(Qwen36ForCausalLM.initialize_vllm_model)
    g = Qwen36MTPContractForCausalLM.__new__(Qwen36MTPContractForCausalLM)
    order = []
    g.warmup_vision = lambda *a, **k: order.append("vision")
    g._maybe_build_spec_prefix_cache = lambda: order.append("pc")
    eng = types.SimpleNamespace(prefill_prepared=True)
    g._engine = lambda: eng
    g.warmup_model_prefill(None, False)
    assert order == ["vision", "pc"]
    order.clear()
    eng.prefill_recorded = True
    g.warmup_model_prefill(None, True)  # the traced call must not run the tower
    assert order == []


# --------------------------------------------------------------------------- #
# Engine routes
# --------------------------------------------------------------------------- #
def _mm_fast_engine(calls, trace_lengths, B=8, replay_none=False):
    model = _with_vis_offset(_fake_model(calls, trace_lengths=trace_lengths))
    spy = {}

    def _replay(tokens, pt, L, vision_tokens=None, strict=True, chunk_start=0, carry_gdn=False, spec_inputs=None, **kw):
        calls.append(("replay", L, chunk_start, vision_tokens, kw.get("vis_row_offset", 0)))
        return None if replay_none else model._short_trace_for(L)

    model._replay_short_trace = _replay
    model._build_request_rope = lambda toks, vt: calls.append(("rope", int(toks.shape[1]), vt))
    model._prefill_traced_chunked_tp = (
        lambda *a, **k: calls.append(
            ("chunks", a[3], a[5], k.get("start", 0), k.get("run_tail"), k.get("vision_tokens"))
        )
        or "REPL"
    )
    model.prefill_masked_bucket = lambda *a, **k: calls.append(
        ("masked", k["actual_len"], k["chunk_start"], k.get("vision_tokens"), k.get("vis_row_offset"))
    ) or ("REPL", "HIDDEN")
    return _engine(B, model=model)


def _fast(e, tokens, **kw):
    Tp = len(tokens)
    return MTPSpecEngine._prefill_fast(
        e, 3, tokens, torch.tensor([tokens], dtype=torch.int32), _row(), Tp, lambda *a: None, [False], **kw
    )


def test_fast_short_path_passes_vision_tokens_to_replay(fake_ttnn):
    calls = []
    e = _mm_fast_engine(calls, trace_lengths=(300,))
    tokens = _mm_prompt(300, 64, first=100)
    assert _fast(e, tokens, vision_tokens=VIS) is not None
    assert ("replay", 300, 0, VIS, 0) in calls  # segment at 0: offset 0


def test_fast_tail_after_chunk_gets_vis_row_offset(fake_ttnn):
    """Prompt 2048 + 300: the chunk replay covers [0, 2048); the short-trace tail starts at 2048, so its vision row
    offset is the number of placeholders before 2048. The chunk loop also gets the vision tokens."""
    calls = []
    e = _mm_fast_engine(calls, trace_lengths=(300,))
    tokens = _mm_prompt(2348, 100, first=1990)  # 58 placeholders before 2048, 42 after
    _fast(e, tokens, vision_tokens=VIS)
    chunks = [c for c in calls if c[0] == "chunks"][0]
    assert chunks[5] == VIS and chunks[3] == 0 and chunks[4] is False
    assert ("replay", 300, 2048, VIS, 58) in calls
    assert ("rope", 2348, VIS) in calls  # M-RoPE table built with the grid, over the whole prompt


def test_fast_masked_fallback_gets_offset(fake_ttnn):
    calls = []
    # the tail's short replay declines (e.g. trash-block collision) -> eager masked bucket at 2048
    e = _mm_fast_engine(calls, trace_lengths=(300,), replay_none=True)
    tokens = _mm_prompt(2348, 100, first=1990)
    _fast(e, tokens, vision_tokens=VIS)
    assert ("masked", 300, 2048, VIS, 58) in calls
    # no trace for the tail at all: the uncached long path (chunks + eager tail, run_tail=True) carries the rows
    calls.clear()
    e = _mm_fast_engine(calls, trace_lengths=())
    _fast(e, tokens, vision_tokens=VIS)
    chunks = [c for c in calls if c[0] == "chunks"][0]
    assert chunks[4] is True and chunks[5] == VIS


def test_fast_text_replay_call_unchanged(fake_ttnn):
    """Text: the replay is called without any vision kwarg (the fake of test_spec_fastprefill_host.py has none)."""
    calls = []
    model = _fake_model(calls, trace_lengths=(300,))
    e = _engine(8, model=model)
    tokens = list(range(1, 301))
    assert _fast(e, tokens) is not None
    assert [c[0] for c in calls] == ["bind", "replay", "unbind", "slot_write"]


def _row_engine(B, calls, plan_spy):
    e = _dispatch_engine(B, fast=True, recorded=True, calls=calls)
    model = e.model
    _with_vis_offset(model)
    model.plan_spec_prefix = lambda *a: plan_spy.append(a) or None
    e._prefill_fast = lambda *a, **k: calls.append(("fast", k)) or (
        a[6].__setitem__(0, True) or torch.arange(VOCAB, dtype=torch.float32)
    )
    return e


def test_mm_row_bypasses_prefix_plan_even_with_start_pos(fake_ttnn):
    calls, plans = [], []
    e = _row_engine(8, calls, plans)
    e._row_gens = None
    MTPSpecEngine._prefill_row(e, 1, _mm_prompt(300, 32), _row(), None, start_pos=128, vision_tokens=VIS)
    assert plans == []  # no prefix lookup
    assert calls[0] == ("fast", {"vision_tokens": VIS})  # no plan / mtp_prompt: no snapshot saves either


def test_text_row_with_start_pos_still_plans(fake_ttnn):
    calls, plans = [], []
    e = _row_engine(8, calls, plans)
    MTPSpecEngine._prefill_row(e, 1, list(range(1, 301)), _row(), None, start_pos=128)
    assert len(plans) == 1 and plans[0][2] == 128
    assert calls[0] == ("fast", {})  # plan None (fake) and no vision kwarg


def test_prefill_row_b1_and_eager_routes_get_vision_tokens(fake_ttnn):
    for B, route in ((1, "b1"), (8, "eager")):
        calls, seen = [], {}
        e = _dispatch_engine(B, fast=True, recorded=True, calls=calls)
        e._prefill_fast = lambda *a, **k: None  # fast declines
        _with_vis_offset(e.model)

        def _b1(prompt, pt_rec, Tp, on_chunk=None, vision_tokens=None):
            seen["b1"] = vision_tokens
            on_chunk("HIDDEN", 0, Tp)
            return torch.arange(VOCAB + 8, dtype=torch.float32)

        def _eager(prompt, pt_row, Tp, on_chunk, slot=0, plan=None, vision_tokens=None):
            seen["eager"] = (vision_tokens, plan)
            on_chunk("HIDDEN", 0, Tp)
            return torch.arange(VOCAB + 8, dtype=torch.float32)

        e.model.prefill_traced_chunked, e.model.prefill_for_spec = _b1, _eager
        e.model.plan_spec_prefix = lambda *a: (_ for _ in ()).throw(AssertionError("mm must not plan"))
        MTPSpecEngine._prefill_row(e, 0, _mm_prompt(300, 32), _row(), None, start_pos=128, vision_tokens=VIS)
        assert seen == ({"b1": VIS} if route == "b1" else {"eager": (VIS, None)})


def test_prefill_forwards_vision_tokens_to_row(fake_ttnn):
    e = _engine(4)
    e._phys = list(range(4))
    e._row_gens = lambda: [0] * 4
    got = {}
    e._prefill_row = lambda slot, ids, row, cap=None, start_pos=None, vision_tokens=None: got.update(
        slot=slot, start_pos=start_pos, vt=vision_tokens
    )
    MTPSpecEngine.prefill(e, 2, [1, 2, 3], _row(), vision_tokens=VIS)
    assert got == dict(slot=2, start_pos=None, vt=VIS)


# --------------------------------------------------------------------------- #
# Model side: vis_row_offset reaches _set_vision_merge; eager spec path threads the rows
# --------------------------------------------------------------------------- #
def test_prefill_for_spec_b1_threads_vision(monkeypatch, expect_error):
    m = types.SimpleNamespace(_chunked_chunk_size=2048, device=None, num_devices=4, args=None)
    _with_vis_offset(m)
    log = []
    m._build_request_rope = lambda toks, vt: log.append(("rope", vt))
    m._reset_gdn_state_for_new_sequence = lambda: None
    m._set_vision_merge = lambda ids, vt, off: log.append(("merge", int(ids.shape[1]), vt, off))
    m._forward_prefill_chunk_masked_tp = lambda *a, **k: "H"
    m._masked_bucket_logits_tp = lambda *a: "LOGITS"
    m.prefill_masked_bucket = lambda ids, pt, **k: log.append(("masked", k["chunk_start"], k["vis_row_offset"])) or (
        "L",
        "H",
    )
    import ttnn

    monkeypatch.setattr(ttnn, "deallocate", lambda *a, **k: None)
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *a, **k: None)
    toks = torch.tensor([_mm_prompt(2348, 100, first=1990)], dtype=torch.int32)
    Qwen36Model._prefill_for_spec_b1(m, toks, None, 2348, lambda *a: None, vision_tokens=VIS)
    assert log == [("rope", VIS), ("merge", 2048, VIS, 0), ("masked", 2048, 58)]
    with expect_error(
        AssertionError, "bypasses the prefix cache"
    ):  # the resumable (prefix-cache) path never serves a multimodal prompt
        Qwen36Model.prefill_for_spec(m, toks, None, 10, None, plan=(0, None, []), vision_tokens=VIS)
