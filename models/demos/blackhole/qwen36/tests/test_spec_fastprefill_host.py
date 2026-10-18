# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_FAST_PREFILL=1 (MTPSpecEngine._prefill_fast / _replay_spec_short, mtp_prefill_inputs,
the split capture of the spec short traces and the GDN slot-write trace format check). No device: the host math is
checked in torch against the eager path's arithmetic, the engine bookkeeping with fakes.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_spec_fastprefill_host.py -q
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.model import Qwen36Model, _ShortPrefillTrace
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, mtp_prefill_inputs
from models.demos.blackhole.qwen36.tt.tp_common import spec_fast_prefill_enabled

BS = 64  # block size
SCRATCH = 999  # the MTP cache's scratch block in these tests
VOCAB = 40


# --------------------------------------------------------------------------- #
# Flag
# --------------------------------------------------------------------------- #
def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_FAST_PREFILL", raising=False)
    assert spec_fast_prefill_enabled() is False
    monkeypatch.setenv("QWEN36_SPEC_FAST_PREFILL", "1")
    assert spec_fast_prefill_enabled() is True
    monkeypatch.setenv("QWEN36_SPEC_FAST_PREFILL", "0")
    assert spec_fast_prefill_enabled() is False


# --------------------------------------------------------------------------- #
# mtp_prefill_inputs: the eager _warm_mtp_chunk / ttnn_mtp_prefill_forward arithmetic, as host tensors
# --------------------------------------------------------------------------- #
def _eager_reference(prompt, cs, bucket, row, scratch):
    """What _warm_mtp_chunk + ttnn_mtp_prefill_forward upload: toks (slot i <- token i+1, zeros past n), the chunk
    table with blocks >= keep redirected to the scratch block."""
    Tp = len(prompt)
    toks = torch.zeros(1, bucket, dtype=torch.int32)
    n = min(bucket, Tp - 1 - cs)
    toks[0, :n] = torch.tensor(prompt[cs + 1 : cs + 1 + n], dtype=torch.int32)
    blk0 = cs // BS
    blkN = (cs + bucket + BS - 1) // BS  # num_blocks_in_seq(chunk_start + S)
    pt_chunk = row[:, blk0:blkN].clone()
    keep = max(0, -(-(cs + n) // BS) - blk0)
    pt_chunk[:, keep:] = scratch
    return toks, pt_chunk, n, keep


@pytest.mark.parametrize(
    "Tp,cs,bucket",
    [(1600, 0, 2048), (300, 0, 512), (128, 0, 128), (1, 0, 128), (2048 + 1500, 2048, 2048), (4096 + 5, 4096, 128)],
)
def test_mtp_prefill_inputs_match_eager(Tp, cs, bucket):
    prompt = list(range(1000, 1000 + Tp))
    nb = 80
    g = torch.Generator().manual_seed(Tp)
    row = (1 + torch.randperm(nb, generator=g)).to(torch.int32).reshape(1, nb)  # never block 0, never SCRATCH
    buf_blocks = 96
    toks, full, chunk = mtp_prefill_inputs(prompt, cs, bucket, row, BS, SCRATCH, buf_blocks)
    ref_toks, ref_chunk, n, keep = _eager_reference(prompt, cs, bucket, row, SCRATCH)
    assert toks.dtype == full.dtype == chunk.dtype == torch.int32
    assert toks.shape == (1, bucket) and full.shape == (1, buf_blocks) and chunk.shape == (1, bucket // BS)
    assert torch.equal(toks, ref_toks), "slot i must fuse with prompt token i+1, zeros past n"
    assert torch.equal(chunk, ref_chunk), "chunk table: blocks past ceil((cs+n)/block) go to the MTP scratch block"
    blk0, bpb = cs // BS, bucket // BS
    # Full table: the user's row outside the window, the window redirected like the chunk table, zeros past the row.
    assert torch.equal(full[0, :blk0], row[0, :blk0])
    assert torch.equal(full[0, blk0 : blk0 + bpb][: buf_blocks - blk0], chunk[0][: buf_blocks - blk0])
    tail_lo = min(buf_blocks, blk0 + bpb)
    assert torch.equal(full[0, tail_lo:nb], row[0, tail_lo:nb]) if tail_lo < nb else True
    assert not bool(full[0, nb:].ne(0).any()) or bool((full[0, nb:] == SCRATCH).all())
    if n > 0:
        assert int(toks[0, n - 1]) == prompt[cs + n]
    # Slot Tp-1 (row Tp-1-cs of the segment) is never written with a real token here (left to _warm_mtp_last).
    if Tp - 1 - cs < bucket:
        assert int(toks[0, Tp - 1 - cs]) == 0 and n == Tp - 1 - cs


def test_mtp_prefill_inputs_segment_with_only_slot_tp_minus_1():
    """Tp = 2049 at cs = 2048: the segment holds only slot Tp-1 -> n = 0, every window block pads to the scratch."""
    prompt = list(range(2049))
    row = torch.arange(1, 65, dtype=torch.int32).reshape(1, 64)
    toks, full, chunk = mtp_prefill_inputs(prompt, 2048, 128, row, BS, SCRATCH, 64)
    assert not bool(toks.any())
    assert bool((chunk == SCRATCH).all())
    assert torch.equal(full[0, :32], row[0, :32]) and bool((full[0, 32:34] == SCRATCH).all())


def test_mtp_prefill_inputs_row_narrower_than_window():
    """A row narrower than the bucket window is zero-padded (null block) before the redirect; width = buf_blocks."""
    prompt = list(range(100))
    row = torch.arange(1, 3, dtype=torch.int32).reshape(1, 2)  # 2 blocks = 128 tokens
    toks, full, chunk = mtp_prefill_inputs(prompt, 0, 256, row, BS, SCRATCH, 32)
    assert full.shape == (1, 32) and chunk.shape == (1, 4)
    # keep = ceil(99 / 64) = 2 real blocks, then scratch
    assert chunk.tolist() == [[1, 2, SCRATCH, SCRATCH]]
    assert full[0, :4].tolist() == [1, 2, SCRATCH, SCRATCH] and not bool(full[0, 4:].any())


# --------------------------------------------------------------------------- #
# Engine fakes
# --------------------------------------------------------------------------- #
def _engine(B, fast=True, recorded=True, model=None, **kw):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.block_size = B, 2, 3, BS
    e.nb, e.nb_prefill, e.vocab = 64, 64, VOCAB
    e._active, e._fresh, e._offered = set(), {}, set()
    e._parked, e._prompts, e._fresh_dirty = {}, {}, set()
    e._no_draft, e._speculated, e._withheld_streak = set(), set(), {}
    e._phys = list(range(B))
    e._last_hidden = [f"last_hidden[{u}]" for u in range(B)]
    e._scratch_block = SCRATCH
    e._feed_dtype, e._dim_frac = ttnn.bfloat16, 16
    e.fast_prefill, e.prefill_recorded, e.recorded_prefill = fast, recorded, True
    e.mesh = None
    e.model = model if model is not None else types.SimpleNamespace()
    e.__dict__.update(kw)
    return e


def _fake_model(calls, chunk=2048, trace_lengths=(), buf_blocks=64, bucket_for=None, replay_none=False):
    """Fake Qwen36Model: records every call; _short_trace_for serves the lengths in ``trace_lengths``."""
    m = types.SimpleNamespace()
    m._chunked_chunk_size = chunk
    m._chunked_trace_output = "CHUNK_OUT"
    m._logits_read_comp = None
    m._fit_spec_page_table = lambda pt, nb: Qwen36Model._fit_spec_page_table(pt, nb)
    m.spec_feed_rows = lambda h: h

    def _bind():
        calls.append(("bind",))
        return "prev"

    def _unbind(prev):
        calls.append(("unbind", prev))

    def _tr(L):
        if L not in trace_lengths:
            return None
        bucket = bucket_for(L) if bucket_for else 2048
        tr = _ShortPrefillTrace(bucket, buf_blocks, bucket // BS)
        tr.spec = True
        tr.last_feed = types.SimpleNamespace(dtype=ttnn.bfloat16, shape=(1, 1, 1, 16))
        tr.logits = "LOGITS"
        return tr

    def _replay(tokens, pt, L, strict=True, chunk_start=0, carry_gdn=False, spec_inputs=None):
        calls.append(("replay", tokens.clone(), pt.clone(), L, strict, chunk_start, carry_gdn, spec_inputs))
        return None if replay_none else _tr(L)

    m._bind_gdn_prefill_scratch = _bind
    m._unbind_gdn_prefill_scratch = _unbind
    m._short_trace_for = _tr
    m._replay_short_trace = _replay
    m._build_request_rope = lambda toks, vt: calls.append(("rope", toks.clone()))
    m._prefill_traced_chunked_tp = (
        lambda *a, **k: calls.append(("chunks", a[3], a[5], k.get("run_tail"), k.get("on_chunk") is not None)) or "REPL"
    )
    m._masked_bucket_logits_tp = lambda h, L, b: calls.append(("last_logits", h, L, b)) or "REPL"
    m.prefill_masked_bucket = lambda *a, **k: calls.append(("masked", k.get("actual_len"), k.get("chunk_start"))) or (
        "REPL",
        "HIDDEN",
    )
    m._write_gdn_slot_from_scratch = lambda slot: calls.append(("slot_write", slot))
    m.prefill_for_spec = lambda *a, **k: calls.append(("eager",)) or "EAGER"
    m.prefill_traced_chunked = lambda *a, **k: calls.append(("b1_chunked",)) or "B1"
    return m


@pytest.fixture
def fake_ttnn(monkeypatch):
    """ttnn.copy / to_torch / deallocate recorders (no device). to_torch returns VOCAB + 8 fake logits."""
    rec = {"copy": [], "to_torch": [], "dealloc": []}
    monkeypatch.setattr(ttnn, "copy", lambda src, dst: rec["copy"].append((src, dst)))
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: rec["dealloc"].append(t))
    monkeypatch.setattr(ttnn, "synchronize_device", lambda *a, **k: rec.setdefault("sync", []).append(a))
    monkeypatch.setattr(
        ttnn, "to_torch", lambda t, **k: rec["to_torch"].append((t, k)) or torch.arange(VOCAB + 8, dtype=torch.bfloat16)
    )
    monkeypatch.setattr(ttnn, "ConcatMeshToTensor", lambda mesh, dim: ("comp", dim))
    return rec


def _row(nb=64):
    return torch.arange(1, nb + 1, dtype=torch.int32).reshape(1, nb)


# --------------------------------------------------------------------------- #
# _prefill_row dispatch
# --------------------------------------------------------------------------- #
def _dispatch_engine(B, fast, recorded, calls):
    model = _fake_model(calls)
    e = _engine(B, fast=fast, recorded=recorded, model=model)
    # The eager path's on_chunk must be cheap here: no MTP warm / feed-row ops.
    e._warm_mtp_chunk = lambda *a, **k: calls.append(("warm_mtp",))
    e._capture_last_row = lambda *a, **k: calls.append(("capture_last_row",))

    def _eager(prompt, pt_row, Tp, on_chunk, slot=0):
        calls.append(("eager", slot))
        on_chunk("HIDDEN", 0, Tp)
        return torch.arange(VOCAB + 8, dtype=torch.float32)  # host logits are accepted as is

    def _b1(prompt, pt_rec, Tp, on_chunk=None):
        calls.append(("b1_chunked",))
        on_chunk("HIDDEN", 0, Tp)
        return torch.arange(VOCAB + 8, dtype=torch.float32)

    model.prefill_for_spec = _eager
    model.prefill_traced_chunked = _b1
    return e


def test_prefill_row_fast_path_used(monkeypatch):
    calls = []
    e = _dispatch_engine(2, fast=True, recorded=True, calls=calls)
    e._prefill_fast = lambda slot, pl, prompt, pt_row, Tp, on_chunk, captured: (
        calls.append(("fast", slot, Tp)),
        captured.__setitem__(0, True),
        torch.arange(VOCAB, dtype=torch.float32),
    )[2]
    out = MTPSpecEngine._prefill_row(e, 1, list(range(10)), _row())
    assert [c[0] for c in calls] == ["fast"] and calls[0][1:] == (1, 10)
    assert out.shape == (VOCAB,) and out.dtype == torch.float32
    assert e._fresh == {1: 10} and e._active == {1}


def test_prefill_row_fast_none_falls_back_to_eager():
    calls = []
    e = _dispatch_engine(2, fast=True, recorded=True, calls=calls)
    e._prefill_fast = lambda *a: calls.append(("fast",)) or None
    out = MTPSpecEngine._prefill_row(e, 0, list(range(10)), _row())
    assert [c[0] for c in calls] == ["fast", "eager", "warm_mtp", "capture_last_row"]
    assert out.shape == (VOCAB,)


@pytest.mark.parametrize("fast,recorded", [(False, True), (True, False)])
def test_prefill_row_eager_when_flag_off_or_not_recorded(fast, recorded):
    calls = []
    e = _dispatch_engine(2, fast=fast, recorded=recorded, calls=calls)
    e._prefill_fast = lambda *a: (_ for _ in ()).throw(AssertionError("fast path must not run"))
    MTPSpecEngine._prefill_row(e, 0, list(range(10)), _row())
    assert [c[0] for c in calls] == ["eager", "warm_mtp", "capture_last_row"]


def test_prefill_row_b1_keeps_chunk_trace_path_when_fast_declines():
    """B == 1: a fast-path decline still takes the recorded chunk-trace path (as before), not the eager one."""
    calls = []
    e = _dispatch_engine(1, fast=True, recorded=True, calls=calls)
    e._prefill_fast = lambda *a: calls.append(("fast",)) or None
    MTPSpecEngine._prefill_row(e, 0, list(range(10)), _row())
    assert [c[0] for c in calls] == ["fast", "b1_chunked", "warm_mtp", "capture_last_row"]


# --------------------------------------------------------------------------- #
# _prefill_fast bookkeeping
# --------------------------------------------------------------------------- #
def _fast(e, slot, prompt_list, pt_row):
    prompt = torch.tensor([prompt_list], dtype=torch.int32)
    captured = [False]
    on_chunk_calls = []
    on_chunk = lambda h, cs, L: on_chunk_calls.append((h, cs, L))
    out = MTPSpecEngine._prefill_fast(e, slot, prompt_list, prompt, pt_row, len(prompt_list), on_chunk, captured)
    return out, captured[0], on_chunk_calls


@pytest.mark.parametrize("B", [1, 8])
def test_fast_short_prompt_one_replay_then_slot_write(B, fake_ttnn):
    calls = []
    Tp = 1600
    model = _fake_model(calls, trace_lengths=(Tp,))
    e = _engine(B, model=model)
    pt_row = _row()
    out, captured, on_chunk_calls = _fast(e, 3 if B > 1 else 0, list(range(Tp)), pt_row)
    kinds = [c[0] for c in calls]
    if B > 1:
        assert kinds == ["bind", "replay", "unbind", "slot_write"], kinds
        assert calls[2] == ("unbind", "prev") and calls[3] == ("slot_write", 3)
    else:
        assert kinds == ["replay"], kinds  # no scratch binding, no slot write at B == 1
    _, tokens, pt, L, strict, cs, carry, spec_inputs = calls[kinds.index("replay")]
    assert L == Tp and strict is False and cs == 0 and carry is False
    assert torch.equal(tokens, torch.tensor([list(range(Tp))], dtype=torch.int32)) and torch.equal(pt, pt_row)
    toks, full, chunk = spec_inputs
    ref = mtp_prefill_inputs(list(range(Tp)), 0, 2048, pt_row, BS, SCRATCH, 64)
    assert all(torch.equal(a, b) for a, b in zip((toks, full, chunk), ref))
    # The feed row of slot Tp-1 lands in _last_hidden[slot] by one copy, and the row is marked captured.
    assert len(fake_ttnn["copy"]) == 1
    src, dst = fake_ttnn["copy"][0]
    assert src.dtype == ttnn.bfloat16 and dst == e._last_hidden[3 if B > 1 else 0]
    assert captured is True and not on_chunk_calls
    # Logits: the pre-gather shard read (ConcatMeshToTensor dim -1), clipped to the vocab, float.
    assert fake_ttnn["to_torch"][0][0] == "LOGITS" and fake_ttnn["to_torch"][0][1]["mesh_composer"] == ("comp", -1)
    assert out.shape == (VOCAB,) and out.dtype == torch.float32


def test_fast_short_prompt_without_trace_returns_none_without_side_effects(fake_ttnn):
    calls = []
    model = _fake_model(calls, trace_lengths=())  # no spec short trace for this length
    e = _engine(8, model=model)
    out, captured, _ = _fast(e, 2, list(range(300)), _row())
    assert out is None and captured is False
    assert [c[0] for c in calls] == ["bind", "unbind"]  # bound for the trace lookup, nothing staged or written
    assert not fake_ttnn["copy"] and not fake_ttnn["to_torch"]


def test_fast_short_prompt_trash_collision_returns_none(fake_ttnn):
    calls = []
    model = _fake_model(calls, trace_lengths=(300,), replay_none=True)
    e = _engine(8, model=model)
    out, captured, _ = _fast(e, 2, list(range(300)), _row())
    assert out is None and captured is False
    assert [c[0] for c in calls] == ["bind", "replay", "unbind"]


def test_fast_long_prompt_chunks_then_tail_replay(fake_ttnn):
    calls = []
    Tp = 2048 + 1500
    model = _fake_model(calls, trace_lengths=(1500,))
    e = _engine(8, model=model)
    pt_row = _row()
    out, captured, on_chunk_calls = _fast(e, 5, list(range(Tp)), pt_row)
    kinds = [c[0] for c in calls]
    assert kinds == ["bind", "rope", "chunks", "replay", "unbind", "slot_write"], kinds
    assert calls[2] == ("chunks", 1, 1500, False, True)  # num_full, tail, run_tail=False, on_chunk given
    _, tokens, pt, L, strict, cs, carry, spec_inputs = calls[3]
    assert L == 1500 and cs == 2048 and carry is True and strict is False
    assert torch.equal(tokens, torch.tensor([list(range(2048, Tp))], dtype=torch.int32))
    ref = mtp_prefill_inputs(list(range(Tp)), 2048, 2048, pt_row, BS, SCRATCH, 64)
    assert all(torch.equal(a, b) for a, b in zip(spec_inputs, ref))
    assert captured is True and out.shape == (VOCAB,)
    assert not on_chunk_calls  # the full chunks' on_chunk is driven by the (fake) chunk loop


def test_fast_long_prompt_exact_multiple(fake_ttnn):
    calls = []
    Tp = 4096
    model = _fake_model(calls, trace_lengths=())
    e = _engine(8, model=model)
    out, captured, on_chunk_calls = _fast(e, 1, list(range(Tp)), _row())
    kinds = [c[0] for c in calls]
    assert kinds == ["bind", "rope", "chunks", "last_logits", "unbind", "slot_write"], kinds
    assert calls[2] == ("chunks", 2, 0, False, True)
    assert calls[3] == ("last_logits", "CHUNK_OUT", 2048, 2048)
    assert on_chunk_calls == [("CHUNK_OUT", 2048, 2048)]  # the last chunk's output feeds the MTP warm + slot Tp-1
    assert captured is False  # the caller's on_chunk captures the row (it ran above); _prefill_fast does not fake it
    assert fake_ttnn["to_torch"][0][1]["mesh_composer"] == ("comp", 0) and "REPL" in fake_ttnn["dealloc"]
    assert out.shape == (VOCAB,)


@pytest.mark.parametrize("tail", [1, 2])
def test_fast_long_prompt_short_tail_runs_eager_tail(tail, fake_ttnn):
    """A < 3-token tail has no short trace (the conv history would be lost): the chunk path runs its eager tail."""
    calls = []
    Tp = 2048 + tail
    model = _fake_model(calls, trace_lengths=(tail,))  # even if a trace existed for this length
    e = _engine(8, model=model)
    out, captured, _ = _fast(e, 1, list(range(Tp)), _row())
    kinds = [c[0] for c in calls]
    assert kinds == ["bind", "rope", "chunks", "unbind", "slot_write"], kinds
    assert calls[2] == ("chunks", 1, tail, True, True)  # run_tail=True with on_chunk
    assert out.shape == (VOCAB,)


def test_fast_long_prompt_tail_collision_falls_back_to_masked_tail(fake_ttnn):
    calls = []
    Tp = 2048 + 500
    model = _fake_model(calls, trace_lengths=(500,), replay_none=True)
    e = _engine(8, model=model)
    out, captured, on_chunk_calls = _fast(e, 1, list(range(Tp)), _row())
    kinds = [c[0] for c in calls]
    assert kinds == ["bind", "rope", "chunks", "replay", "masked", "unbind", "slot_write"], kinds
    assert calls[4] == ("masked", 500, 2048)
    assert on_chunk_calls == [("HIDDEN", 2048, 500)] and "HIDDEN" in fake_ttnn["dealloc"]
    assert out.shape == (VOCAB,)


# --------------------------------------------------------------------------- #
# record_prefill
# --------------------------------------------------------------------------- #
def _record_engine(B, fast, fmts, calls):
    model = types.SimpleNamespace(
        record_prefill_trace_chunked=lambda mesh: calls.append(("chunk",)),
        record_prefill_traces_short=lambda mesh: calls.append(("short",)),
        capture_gdn_slot_write_traces=lambda conv_fmt="packed": calls.append(("slot_traces", conv_fmt)),
        _bind_gdn_prefill_scratch=lambda: calls.append(("bind",)) or "prev",
        _unbind_gdn_prefill_scratch=lambda prev: calls.append(("unbind", prev)),
    )
    e = _engine(B, fast=fast, recorded=False, model=model)
    e.prefill_prepared = True
    e._gdn = [types.SimpleNamespace(_conv_fmt=f) for f in fmts]
    return e


def test_record_prefill_fast_batched():
    calls = []
    e = _record_engine(8, True, ["both"] * 3, calls)
    MTPSpecEngine.record_prefill(e)
    assert calls == [("bind",), ("chunk",), ("short",), ("unbind", "prev"), ("slot_traces", "both")]
    assert e.prefill_recorded is True
    MTPSpecEngine.record_prefill(e)  # idempotent
    assert len(calls) == 5


def test_record_prefill_fast_batched_skips_slot_traces_on_other_formats():
    calls = []
    e = _record_engine(8, True, ["both", "states"], calls)
    MTPSpecEngine.record_prefill(e)
    assert ("slot_traces", "both") not in calls and calls[-1] == ("unbind", "prev")


def test_record_prefill_fast_b1_no_scratch():
    calls = []
    e = _record_engine(1, True, ["both"], calls)
    MTPSpecEngine.record_prefill(e)
    assert calls == [("chunk",), ("short",)]


@pytest.mark.parametrize("B,expect", [(1, [("chunk",)]), (8, [])])
def test_record_prefill_flag_off_unchanged(B, expect):
    calls = []
    e = _record_engine(B, False, ["both"], calls)
    MTPSpecEngine.record_prefill(e)
    assert calls == expect and e.prefill_recorded is True


# --------------------------------------------------------------------------- #
# GDN slot-write trace: format-aware fingerprint / replay check
# --------------------------------------------------------------------------- #
class _Buf:
    def __init__(self, addr):
        self._addr = addr

    def buffer_address(self):
        return self._addr


def _slot_write_model(fmt, with_hist=True):
    m = Qwen36Model.__new__(Qwen36Model)
    m._gdn_slot_write_traces = {0: "t0"}
    m._gdn_slot_write_warned = False
    m._gdn_slot_write_fmt = None
    m._gdn_slot_write_fp = None
    layers = []
    for li in range(3):
        dn = types.SimpleNamespace(
            _conv_fmt=fmt,
            rec_state=_Buf(100 + li),
            conv_states=[_Buf(200 + 10 * li + j) for j in range(4)],
            _conv_hist_rm=_Buf(300 + li) if with_hist else None,
            _conv_hist_packed=_Buf(400 + li),
            _pack_tables=lambda li=li: {"slot_to_packed": (_Buf(500 + li), _Buf(600 + li)), "zero_row": _Buf(700 + li)},
        )
        layers.append((dn, _Buf(800 + li), [_Buf(900 + 10 * li + j) for j in range(4)], None, None, None))
    m._gdn_prefill_scratch = layers
    return m


@pytest.mark.parametrize("fmt", ["both", "packed"])
def test_slot_write_replayable_tracks_captured_format(fmt):
    m = _slot_write_model(fmt)
    assert m._gdn_slot_write_replayable() is False  # nothing captured yet (fmt None)
    m._gdn_slot_write_fmt = fmt
    m._gdn_slot_write_fp = m._gdn_slot_write_fingerprint(fmt)
    assert m._gdn_slot_write_replayable() is True
    # A layer leaves the captured format: not replayable (no warning, the format is a host flag, not an address move).
    m._gdn_prefill_scratch[1][0]._conv_fmt = "states" if fmt == "both" else "hist"
    assert m._gdn_slot_write_replayable() is False and m._gdn_slot_write_warned is False
    m._gdn_prefill_scratch[1][0]._conv_fmt = fmt
    assert m._gdn_slot_write_replayable() is True
    # A baked address moves: not replayable, warned once.
    moved = _Buf(4242)
    if fmt == "both":
        m._gdn_prefill_scratch[2][0].conv_states[3] = moved
    else:
        m._gdn_prefill_scratch[2][0]._conv_hist_packed = moved
    assert m._gdn_slot_write_replayable() is False and m._gdn_slot_write_warned is True


def test_slot_write_fingerprint_formats_differ():
    m = _slot_write_model("both")
    fp_both, fp_packed = m._gdn_slot_write_fingerprint("both"), m._gdn_slot_write_fingerprint("packed")
    assert fp_both != fp_packed
    assert all(len(layer) == 4 for layer in fp_both)
    # "both" tracks the K tap buffers and the RM history; a model without the history fingerprints None there.
    assert fp_both[0][3][1] == 300 and len(fp_both[0][3][0]) == 4
    assert _slot_write_model("both", with_hist=False)._gdn_slot_write_fingerprint("both")[0][3][1] is None


def test_capture_slot_write_traces_rejects_unknown_format(monkeypatch, expect_error):
    monkeypatch.delenv("QWEN36_GDN_SLOT_WRITE_TRACE", raising=False)
    m = Qwen36Model.__new__(Qwen36Model)
    m._gdn_slot_write_traces = {}
    with expect_error(ValueError, ".*conv_fmt.*"):
        m.capture_gdn_slot_write_traces(conv_fmt="hist")


def test_release_slot_write_traces_clears_format(monkeypatch):
    m = Qwen36Model.__new__(Qwen36Model)
    m.mesh_device = "mesh"
    m._gdn_slot_write_traces = {0: 7}
    m._gdn_slot_write_fp, m._gdn_slot_write_fmt = ["fp"], "both"
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: None)
    m.release_gdn_slot_write_traces()
    assert m._gdn_slot_write_traces == {} and m._gdn_slot_write_fp is None and m._gdn_slot_write_fmt is None


# --------------------------------------------------------------------------- #
# Short-trace record: pending bookkeeping
# --------------------------------------------------------------------------- #
def test_record_prefill_traces_short_noop_when_nothing_pending():
    m = Qwen36Model.__new__(Qwen36Model)
    m._short_traces_pending = {}
    m._short_traces = {}
    m.record_prefill_traces_short("mesh")  # no device call
    assert m._short_traces == {}


def test_release_prefill_trace_short_frees_pending_and_spec_buffers(monkeypatch):
    freed = []
    monkeypatch.setattr(ttnn, "deallocate", lambda t, *a, **k: freed.append(t))
    monkeypatch.setattr(ttnn, "release_trace", lambda dev, tid: freed.append(("trace", tid)))
    m = Qwen36Model.__new__(Qwen36Model)
    m.device = "dev"
    done = _ShortPrefillTrace(2048, 64, 32)
    done.spec, done.trace_id = True, 11
    done.tok, done.mtp_tok, done.mtp_full_pt, done.mtp_chunk_pt, done.last_feed = "tok", "mtok", "mfull", "mchunk", "lf"
    pend = _ShortPrefillTrace(128, 64, 2)
    pend.tok = "ptok"
    m._short_traces, m._short_traces_pending = {2048: done}, {128: pend}
    m.release_prefill_trace_short()
    assert m._short_traces == {} and m._short_traces_pending == {}
    assert {"tok", "mtok", "mfull", "mchunk", "ptok", ("trace", 11)} <= set(freed)
    assert done.last_feed is None and done.logits is None


def test_spec_prepare_is_idempotent_per_variant(expect_error):
    """prepare_prefill_traces_short returns early when every bucket is already prepared/captured for that variant."""
    m = Qwen36Model.__new__(Qwen36Model)
    m.num_devices, m._lmhead_vocab_sharded = 8, True
    m._paged_kv_caches = "kv"
    m.mtp = types.SimpleNamespace(attention=types.SimpleNamespace(use_paged=True, paged_k="k"))
    tr = _ShortPrefillTrace(128, 64, 2)
    tr.spec = True
    m._short_traces, m._short_traces_pending = {128: tr}, {}
    m.short_prefill_trace_buckets = lambda: (128,)
    m.prepare_prefill_traces_short("mesh", torch.zeros(1, 64, dtype=torch.int32), spec=True)  # early return, no device
    with expect_error(Exception, ".*"):
        # the non-spec variant of the same bucket is a different trace: it would re-prepare (and touch the device here)
        m.prepare_prefill_traces_short("mesh", torch.zeros(1, 64, dtype=torch.int32), spec=False)
