# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests (no device) of the DFlash2 class's scheduler-driven chunked prefill (QWEN36_DFLASH_CHUNKED_PREFILL):
the capability dict with the knob off / on, and the prefill_forward orchestration against a fake model and decoder
(planner order, park / unpark, _pending only on a final chunk, zero logits for an intermediate one, the legacy loop for
whole prompts with no partial held, the loud ownership checks).

Run: QWEN36_DRAFTER=dflash2 python -m pytest models/demos/blackhole/qwen36/tests/test_dflash_chunked_host.py -q
(needs the vLLM + ttnn Python env; opens no device)."""

import os

import pytest
import torch

os.environ.setdefault("QWEN36_DRAFTER", "dflash2")
os.environ.setdefault("QWEN36_DFLASH_SERVE_BLOCK", "32")

from models.demos.blackhole.qwen36.tt import qwen36_vllm_dflash as D  # noqa: E402
from models.demos.blackhole.qwen36.tt.chunked_prefill import ChunkedPrefillPlanner  # noqa: E402

C = 2048
VOCAB = 16
STATIC_KEYS = {
    "supports_prefix_caching",
    "supports_async_decode",
    "supports_sample_on_device",
    "supports_chunked_prefill",
    "output_tokens_per_step",
    "tt_adaptive_block_output",
    "tt_adaptive_block_batched",
    "tt_adaptive_block_ragged",
    "tt_adaptive_block_max_prompt_tokens",
    "tt_block_output_kv_lookahead_tokens",
}


@pytest.fixture
def knob(monkeypatch):
    def _set(on):
        if on:
            monkeypatch.setenv("QWEN36_DFLASH_CHUNKED_PREFILL", "1")
        else:
            monkeypatch.delenv("QWEN36_DFLASH_CHUNKED_PREFILL", raising=False)

    return _set


# ------------------------------------------------------------------ capabilities


def test_capabilities_with_the_knob_off_are_todays_static_dict(knob, monkeypatch, expect_error):
    knob(False)
    # the plain class's knobs must not leak into the DFlash dict (review: _DFlashCapabilities is its own class)
    monkeypatch.setenv("QWEN36_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("QWEN36_ASYNC_DECODE_OK", "1")
    caps = D.Qwen36DFlashForCausalLM.model_capabilities
    assert set(dict(caps)) == STATIC_KEYS
    assert caps["supports_chunked_prefill"] is False
    assert caps.get("supports_chunked_prefill", True) is False
    assert caps["supports_async_decode"] is False
    assert caps.get("supports_async_decode") is False
    for key in ("tt_prefill_chunk_tokens", "tt_block_output_chunked_prefill"):
        assert key not in caps
        assert caps.get(key) is None
        assert caps.get(key, 7) == 7
        with expect_error(KeyError, key):
            caps[key]
    assert caps["output_tokens_per_step"] == D._W > 1


def test_capabilities_with_the_knob_on_declare_the_block_output_chunk_contract(knob):
    knob(True)
    caps = D.Qwen36DFlashForCausalLM.model_capabilities
    assert caps["supports_chunked_prefill"] is True
    assert caps.get("supports_chunked_prefill") is True
    assert caps["tt_prefill_chunk_tokens"] == C
    assert caps.get("tt_prefill_chunk_tokens") == C
    assert "tt_block_output_chunked_prefill" in caps and caps["tt_block_output_chunked_prefill"] is True
    assert caps["supports_async_decode"] is False
    # a copy sees only the static entries (the dynamic keys live in the accessors)
    assert set(dict(caps)) == STATIC_KEYS and dict(caps)["supports_chunked_prefill"] is False


# ------------------------------------------------------------------ orchestration fakes


class _FakeDec:
    def __init__(self, B):
        self.active = [False] * B
        self.ctx_len = [0] * B
        self.ended = []

    def end(self, phys):
        self.ended.append(phys)
        self.active[phys] = False

    def ingest_prompt(self, phys, taps, T, chunk_start=0):
        self.ctx_len[phys] = int(T)


class _FakeModel:
    """The model surface _prefill_planned / _spec_prefill touch; prefill_for_spec feeds on_chunk per 2048 chunk."""

    vocab_size = VOCAB
    mesh_device = None

    def __init__(self):
        self.calls = []
        self._planner = ChunkedPrefillPlanner(C)
        self._dflash_tap = False

    def _chunked_prefill_planner(self):
        return self._planner

    def _park_gdn_scratch(self):
        self.calls.append(("park",))

    def _unpark_gdn_scratch(self):
        self.calls.append(("unpark",))

    def take_dflash_eager_taps(self):
        return []

    def prefill_for_spec(self, prompt, pt, T, on_chunk, slot=0, start=0, final=True):
        self.calls.append(("prefill", slot, start, T, final))
        cs = start
        while cs < T:
            n = min(C, T - cs)
            on_chunk(None, cs, n)
            cs += n
        return None if not final else "logits"


def _obj(B=8, cp_on=True):
    o = D.Qwen36DFlashForCausalLM.__new__(D.Qwen36DFlashForCausalLM)
    o.model = [_FakeModel()]
    o._spec = _FakeDec(B)
    o._in_warmup = False
    o._B = B
    o._phys = list(range(B))
    o._pending = [None] * B
    o._carry = [[] for _ in range(B)]
    o._stopped = [False] * B
    o._prev_tail = [None] * B
    o._cp_on = cp_on
    o._cp_owner_phys = None
    o._forbid_plain = False
    return o


def _fake_final_logits(o):
    """Replace the device logits readback of a FINAL spec prefill (host test: 'logits' marker -> a host row)."""
    real = D.Qwen36DFlashForCausalLM._spec_prefill

    def spec_prefill(self, model, dec, phys, prompt, T, pt_row, start=0, final=True):
        if final:
            # run the real ctx_len bookkeeping on a host-only path: final chunks just advance the drafter frontier
            if not start:
                dec.ctx_len[phys] = 0
            elif dec.ctx_len[phys] != start:
                raise RuntimeError(f"resume at {start} but drafter context ends at {dec.ctx_len[phys]}")
            model.prefill_for_spec(prompt, pt_row, T, lambda h, cs, n: dec.ingest_prompt(phys, [], cs + n, cs), phys)
            model.calls[-1] = ("prefill", phys, start, T, True)
            return torch.full((1, VOCAB), float(phys))
        return real(self, model, dec, phys, prompt, T, pt_row, start=start, final=final)

    o._spec_prefill = spec_prefill.__get__(o)
    return o


def _call(o, rows, slots, starts, ends, resume, final, first_blocks=None):
    """prefill_forward as the plugin runner calls it under the chunk policy (both masks on every step)."""
    N = len(rows)
    T = max(ends)
    tokens = torch.arange(N * T, dtype=torch.int32).reshape(N, T) % 1000
    pt = torch.zeros(N, 64, dtype=torch.int32)
    for u in range(N):
        pt[u, 0] = (first_blocks or [100 + 10 * s for s in slots])[u]
    kw = dict(empty_slots=slots, start_pos=torch.tensor(starts))
    if resume is not None:
        kw.update(prefill_resume_mask=resume, prefill_final_mask=final)
    return o.prefill_forward(tokens, pt, None, torch.tensor(ends), **kw)


def _prefills(o):
    return [c for c in o.model[0].calls if c[0] == "prefill"]


def test_masks_absent_takes_todays_loop():
    o = _fake_final_logits(_obj())
    lg, _ = _call(o, [0, 1], [3, 4], [0, 0], [100, 5000], None, None)
    assert _prefills(o) == [("prefill", 3, 0, 100, True), ("prefill", 4, 0, 5000, True)]
    assert o._pending[3][0] == 100 and o._pending[4][0] == 5000
    assert o.model[0]._planner.owner is None and ("park",) not in o.model[0].calls


def test_whole_prompts_with_no_partial_held_take_todays_loop():
    o = _fake_final_logits(_obj())
    _call(o, [0], [2], [0], [9000], [False], [True])
    assert _prefills(o) == [("prefill", 2, 0, 9000, True)]
    assert o.model[0]._planner.owner is None


def test_a_chunked_prompt_with_riders_parks_and_seats_only_on_its_final_chunk():
    o = _fake_final_logits(_obj())
    m = o.model[0]
    # chunk 1 of a 5000-token prompt in slot 5: intermediate
    lg, _ = _call(o, [0], [5], [0], [C], [False], [False])
    assert torch.count_nonzero(lg) == 0 and lg.shape == (1, 1, VOCAB)
    assert o._pending[5] is None and o._cp_owner_phys == 5
    assert m._planner.owner.next_pos == C and not m._planner.owner.parked
    assert o._spec.ctx_len[5] == C
    # a rider-only call between chunks (trivial masks, but a partial is held): park before the rider
    _call(o, [0], [1], [0], [300], [False], [True])
    assert m.calls[-2] == ("park",) and m.calls[-1] == ("prefill", 1, 0, 300, True)
    assert m._planner.owner.parked and o._cp_owner_phys == 5 and o._pending[1] is not None
    # chunk 2 (intermediate) resumes first with an unpark, then a rider (which parks again)
    _call(o, [0, 1], [5, 2], [C, 0], [2 * C, 200], [True, False], [False, True], first_blocks=[150, 120])
    assert m.calls[-4:] == [
        ("unpark",),
        ("prefill", 5, C, 2 * C, False),
        ("park",),
        ("prefill", 2, 0, 200, True),
    ]
    assert o._pending[5] is None and o._spec.ctx_len[5] == 2 * C
    # final chunk: unpark, resume the tail, seat the slot, no owner left
    lg, _ = _call(o, [0], [5], [2 * C], [5000], [True], [True], first_blocks=[150])
    assert m.calls[-2:] == [("unpark",), ("prefill", 5, 2 * C, 5000, True)]
    assert o._pending[5][0] == 5000 and torch.all(lg == 5.0)
    assert m._planner.owner is None and o._cp_owner_phys is None


def test_a_multi_chunk_resume_in_one_call():
    """The last decoder left while a partial was in flight: the remainder [C, 4C+1) runs in ONE resume call."""
    o = _fake_final_logits(_obj())
    _call(o, [0], [6], [0], [C], [False], [False])
    _call(o, [0], [6], [C], [4 * C + 1], [True], [True], first_blocks=[160])
    assert _prefills(o)[-1] == ("prefill", 6, C, 4 * C + 1, True)
    assert o._spec.ctx_len[6] == 4 * C + 1 and o._pending[6][0] == 4 * C + 1


def test_a_resume_on_another_slot_raises(expect_error):
    """Negative control of the sticky-slot contract: the plugin moved the continuation to slot 3."""
    o = _fake_final_logits(_obj())
    _call(o, [0], [5], [0], [C], [False], [False], first_blocks=[150])
    with expect_error(RuntimeError, "the partial prompt is in slot 5"):
        _call(o, [0], [3], [C], [5000], [True], [True], first_blocks=[150])


def test_a_rider_on_the_partials_slot_breaks_the_resume_loudly(expect_error):
    """Negative control: without sticky slots a rider takes the partial's slot (and its drafter context)."""
    o = _fake_final_logits(_obj())
    _call(o, [0], [5], [0], [C], [False], [False], first_blocks=[150])
    _call(o, [0], [5], [0], [300], [False], [True], first_blocks=[999])  # rider lands on slot 5
    with expect_error(RuntimeError, "slot"):
        _call(o, [0], [5], [C], [5000], [True], [True], first_blocks=[150])


def test_a_drafter_frontier_mismatch_raises(expect_error):
    o = _fake_final_logits(_obj())
    _call(o, [0], [5], [0], [C], [False], [False], first_blocks=[150])
    o._spec.ctx_len[5] = 1000  # a lost chunk / clobbered ring
    with expect_error(RuntimeError, "drafter context"):
        _call(o, [0], [5], [C], [5000], [True], [True], first_blocks=[150])


def test_a_chunked_request_with_the_knob_off_raises(expect_error):
    o = _fake_final_logits(_obj(cp_on=False))
    with expect_error(RuntimeError, "QWEN36_DFLASH_CHUNKED_PREFILL is off"):
        _call(o, [0], [5], [0], [C], [False], [False])


def test_release_of_the_partial_drops_the_scratch_ownership():
    o = _fake_final_logits(_obj())
    _call(o, [0], [5], [0], [C], [False], [False])
    o.release_request(5)
    assert o._cp_owner_phys is None and o.model[0]._planner.owner is None
    # the next whole prompt is not parked for nothing and takes today's loop
    _call(o, [0], [5], [0], [700], [False], [True])
    assert ("park",) not in o.model[0].calls


def test_abort_then_a_new_long_prompt_on_the_same_slot_with_a_rider():
    """Lossless critique m6: abort a partial, reuse its slot (and its first block) for a new long prompt, then a rider."""
    o = _fake_final_logits(_obj())
    _call(o, [0], [5], [0], [C], [False], [False], first_blocks=[150])
    o.release_request(5)
    _call(o, [0], [5], [0], [C], [False], [False], first_blocks=[150])
    assert o._spec.ctx_len[5] == C and o._cp_owner_phys == 5
    _call(o, [0, 1], [5, 1], [C, 0], [3 * C, 300], [True, False], [True, True], first_blocks=[150, 110])
    assert o._pending[5][0] == 3 * C and o._pending[1][0] == 300
    assert o.model[0]._planner.owner is None


def test_release_of_another_slot_keeps_the_partial():
    o = _fake_final_logits(_obj())
    _call(o, [0], [5], [0], [C], [False], [False])
    o.release_request(2)
    assert o._cp_owner_phys == 5 and o.model[0]._planner.owner is not None


def test_plain_trace_tripwire_blocks_the_plain_fallbacks(expect_error):
    o = _obj()
    o._spec = None
    o._in_warmup = True  # the only state in which the class would fall back to the plain path
    o._forbid_plain = True
    with expect_error(RuntimeError, "hang class"):
        o.prefill_forward(torch.zeros(1, 8, dtype=torch.int32), torch.zeros(1, 4), None, torch.tensor([8]))
    with expect_error(RuntimeError, "hang class"):
        o.decode_forward(tokens=torch.zeros(1, 1, dtype=torch.int32))


# ---- lane R review R1: speculative DFlash serving never warms (or runs) the fast GDN slot remap ----
class _FakeWarmModel:
    def __init__(self):
        self.use_tp = True
        self.args = type("A", (), {"max_batch_size": 4})()
        self.calls = []

    def _bind_gdn_prefill_scratch(self):
        return []

    def _unbind_gdn_prefill_scratch(self, prev):
        pass

    def ensure_gdn_park_buffer(self):
        pass

    def capture_prefill_trace_chunked(self, *a, **kw):
        self.calls.append("capture")

    def warmup_gdn_slot_write(self):
        self.calls.append("slot_write")

    def warmup_gdn_remap(self):
        self.calls.append("remap")


@pytest.mark.parametrize("W, expect_remap", [(32, False), (1, True)], ids=["spec_W32", "plain_W1"])
def test_dflash_prefill_warmup_skips_gdn_remap_when_speculative(monkeypatch, W, expect_remap):
    """With _W > 1 (tp2-dflash2 / batch8-dflash2 serving) warmup_model_prefill must not call model.warmup_gdn_remap:
    spec decode composes slot_remap into _phys and never runs remap_slots, and the warmup would run 4 remaps on the
    B=4 state after the spec traces exist. _W == 1 (plain decode, which does remap) keeps it. The plain class always
    warms it."""
    from models.demos.blackhole.qwen36.tt import qwen36_vllm as Q

    monkeypatch.setattr(D, "_W", W)
    monkeypatch.setattr(Q, "_log_device_memory", lambda *a, **kw: None)
    for cls, want in ((D.Qwen36DFlashForCausalLM, expect_remap), (Q.Qwen36ForCausalLM, True)):
        obj = object.__new__(cls)
        fake = _FakeWarmModel()
        obj.model = [fake]
        obj.mesh_device = None
        obj.warmup_model_prefill(None, True)
        assert fake.calls[0] == "capture" and "slot_write" in fake.calls, fake.calls
        assert ("remap" in fake.calls) == want, (cls.__name__, W, fake.calls)
