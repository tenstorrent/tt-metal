# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device) tests for scheduler-driven chunked prefill on the model side.

1. ChunkedPrefillPlanner (tt/chunked_prefill.py): execution order (the continuation first), park before ANY scratch
   reset while an unparked owner exists -- including a call that carries only short prompts between two chunks
   (review B1) -- unpark before the continuation, ownership checks (wrong request, wrong start, unaligned start,
   non-resume rows ignore their start), at most one partial per call, and the owner a call leaves behind.
2. Qwen36Model.prefill_paged_slots row bookkeeping with a fake ttnn and a recording fake prefill, for all three
   QWEN36_PLAIN_GDN_SLOT_DEVICE_COPY modes (0 host snapshot, 1 clone + row write, 2 fill_cache): logits come back in
   CALL order although the continuation runs first; only final rows write a decode slot (each to its own slot, with its
   own state); an intermediate row returns a zero row; park/unpark happen exactly where the plan says; the owner is
   committed only after the loop.

Run: pytest models/demos/blackhole/qwen36/tests/test_chunked_prefill_plan.py -q
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt.chunked_prefill import ChunkedPrefillPlanner, ScratchOwner

C = 2048


def _plan(planner, starts, ends, resume, final, blocks):
    return planner.plan(starts=starts, ends=ends, resume_mask=resume, final_mask=final, first_blocks=blocks)


def _run(planner, *args):
    plans, owner = _plan(planner, *args)
    planner.owner = owner
    return plans


# ---------------------------------------------------------------------------------------------------------- planner


def test_unchunked_call_is_trivial():
    p = ChunkedPrefillPlanner(C)
    plans = _run(p, [0, 0], [100, 5000], [False, False], [True, True], [7, 9])
    assert [(r.row, r.start, r.final, r.park_before, r.unpark_before) for r in plans] == [
        (0, 0, True, False, False),
        (1, 0, True, False, False),
    ]
    assert p.owner is None


def test_full_chunk_sequence_with_riders_parks_and_unparks():
    p = ChunkedPrefillPlanner(C)
    # call 1: a new long prompt's first chunk shares the call with a short prompt that comes FIRST in call order
    plans = _run(p, [0, 0], [300, C], [False, False], [True, False], [5, 40])
    # no owner at the start: nothing parked; the short row resets first, then the chunk starts from 0
    assert [(r.row, r.park_before) for r in plans] == [(0, False), (1, False)]
    assert p.owner == ScratchOwner(40, C, parked=False)
    # call 2: continuation + a rider after it: continuation first, rider parks the (still intermediate) partial
    plans = _run(p, [0, C], [200, 2 * C], [False, True], [True, False], [6, 40])
    assert [(r.row, r.start, r.resume, r.unpark_before, r.park_before) for r in plans] == [
        (1, C, True, False, False),
        (0, 0, False, False, True),
    ]
    assert p.owner == ScratchOwner(40, 2 * C, parked=True)
    # call 3: final continuation (tail) alone: unpark, then no owner
    plans = _run(p, [2 * C], [2 * C + 77], [True], [True], [40])
    assert [(r.start, r.end, r.unpark_before, r.final) for r in plans] == [(2 * C, 2 * C + 77, True, True)]
    assert p.owner is None


def test_short_prompt_only_call_between_chunks_parks_the_partial():
    """Review B1: a prefill call with ONLY short prompts between two chunks must park the partial before resetting the
    scratch, and the next continuation must unpark it."""
    p = ChunkedPrefillPlanner(C)
    _run(p, [0], [C], [False], [False], [40])
    plans = _run(p, [0, 0], [512, 90], [False, False], [True, True], [3, 4])
    assert [r.park_before for r in plans] == [True, False]  # parked once, before the first reset
    assert p.owner == ScratchOwner(40, C, parked=True)
    plans = _run(p, [C], [2 * C], [True], [False], [40])
    assert plans[0].unpark_before and not plans[0].park_before
    assert p.owner == ScratchOwner(40, 2 * C, parked=False)


def test_continuation_without_riders_needs_no_copy():
    p = ChunkedPrefillPlanner(C)
    _run(p, [0], [C], [False], [False], [40])
    plans = _run(p, [C], [2 * C], [True], [False], [40])
    assert not plans[0].unpark_before and not plans[0].park_before


@pytest.mark.parametrize(
    "starts, ends, blocks, why",
    [
        ([C], [2 * C], [41], "wrong request (first block)"),
        ([2 * C], [3 * C], [40], "wrong start (skips a chunk)"),
        ([C // 2], [C + C // 2], [40], "unaligned start"),
        ([C], [C], [40], "start not below the end"),
    ],
)
def test_bad_continuation_raises(starts, ends, blocks, why, expect_error):
    p = ChunkedPrefillPlanner(C)
    _run(p, [0], [C], [False], [False], [40])
    with expect_error(ValueError, "does not continue|must be a positive|not a multiple"):
        _plan(p, starts, ends, [True], [False], blocks)
    assert p.owner == ScratchOwner(40, C, parked=False), why  # a refused call leaves the owner untouched


def test_continuation_without_owner_raises(expect_error):
    with expect_error(ValueError, "does not continue"):
        _plan(ChunkedPrefillPlanner(C), [C], [2 * C], [True], [True], [40])


def test_non_resume_row_ignores_its_start_and_reprefills():
    """An s > 0 row that is not a flagged continuation (e.g. a resumed streaming session) re-prefills from 0, as the
    model did before chunking; it never resumes a foreign state."""
    p = ChunkedPrefillPlanner(C)
    plans = _run(p, [777], [900], [False], [True], [3])
    assert plans[0].start == 0 and not plans[0].resume


def test_intermediate_row_must_end_on_a_chunk_boundary(expect_error):
    with expect_error(ValueError, "not a multiple"):
        _plan(ChunkedPrefillPlanner(C), [0], [C + 1], [False], [False], [3])


def test_two_partials_in_one_call_raise(expect_error):
    p = ChunkedPrefillPlanner(C)
    with expect_error(ValueError, "more than one partial"):
        _plan(p, [0, 0], [C, C], [False, False], [False, False], [3, 4])
    _run(p, [0], [C], [False], [False], [40])
    with expect_error(ValueError, "more than one partial"):
        _plan(p, [C, 0], [2 * C, C], [True, False], [False, False], [40, 4])
    with expect_error(ValueError, "resume rows"):
        _plan(p, [C, C], [2 * C, 2 * C], [True, True], [True, True], [40, 40])


def test_stale_owner_is_parked_once_then_replaced():
    """An aborted/preempted partial leaves a stale owner: the next reset parks it once (harmless), later resets do not
    copy again, and a new partial (or the same request restarting from 0) replaces it."""
    p = ChunkedPrefillPlanner(C)
    _run(p, [0], [C], [False], [False], [40])
    assert [r.park_before for r in _run(p, [0], [10], [False], [True], [1])] == [True]
    assert [r.park_before for r in _run(p, [0], [10], [False], [True], [2])] == [False]
    _run(p, [0], [2 * C], [False], [False], [40])  # the preempted request restarts from 0
    assert p.owner == ScratchOwner(40, 2 * C, parked=False)


# ------------------------------------------------------------------------------------- prefill_paged_slots bookkeeping


class _FakeTensor:
    def __init__(self, tag):
        self.tag = tag
        self.dtype = "bf16"


class _FakeTtnn(SimpleNamespace):
    """Just enough ttnn for prefill_paged_slots' host-side bookkeeping; every device op is recorded."""

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.ROW_MAJOR_LAYOUT = "rm"
        self.ConcatMeshToTensor = lambda *a, **k: None

    def to_torch(self, t, mesh_composer=None):
        if isinstance(t, _FakeTensor) and t.tag.startswith("logits:"):
            v = float(t.tag.split(":")[1])
            return torch.full((2, 8), v)
        return torch.tensor([hash(getattr(t, "tag", "x")) % 1000], dtype=torch.float32)

    def deallocate(self, t):
        pass

    def clone(self, t):
        return _FakeTensor(f"clone({t.tag})")

    def typecast(self, t, dtype):
        return t

    def fill_cache(self, dst, src, slot):
        self.log.append(("fill_cache", dst.tag, src.tag, slot))

    def synchronize_device(self, dev):
        pass

    def to_layout(self, t, layout):
        return t

    def get_device_tensors(self, t):
        return [t]


class _FakeDN:
    K = 2
    _decode_fused_conv = True

    def __init__(self, i, log):
        self.i = i
        self.log = log
        self.rec_state = _FakeTensor(f"scratch_rec{i}")
        self.conv_states = [_FakeTensor(f"scratch_conv{i}_{m}") for m in range(self.K)]

    def _write_index(self, buf, src, idx, dim):
        self.log.append(("write_index", buf.tag, idx))

    def _sync_conv_hist_packed(self, slot):
        self.log.append(("sync_hist", self.i, slot))


def _fake_model(monkeypatch, log):
    from models.demos.blackhole.qwen36.tt import model as model_mod

    monkeypatch.setattr(model_mod, "ttnn", _FakeTtnn(log))
    dns = [_FakeDN(i, log) for i in range(2)]
    layers = [SimpleNamespace(is_full_attention=False, attention=dn) for dn in dns] + [
        SimpleNamespace(is_full_attention=True, attention=None)
    ]
    m = model_mod.Qwen36Model.__new__(model_mod.Qwen36Model)
    m.use_tp = True
    m.mesh_device = None
    m.args = SimpleNamespace(vocab_size=8)
    m.layers = layers
    m._chunked_chunk_size = C

    def _bind():
        log.append(("bind",))
        return [
            (dn, 8, _FakeTensor(f"batched_rec{dn.i}"), [_FakeTensor(f"batched_conv{dn.i}_{k}") for k in range(2)])
            for dn in dns
        ]

    def _prefill(toks, pt, actual_len, start=0, want_logits=True):
        log.append(("prefill", int(pt[0, 0]), start, actual_len, want_logits))
        for dn in dns:  # the scratch now holds this request's state
            dn.rec_state.tag = f"scratch_rec{dn.i}@{int(pt[0, 0])}"
        return _FakeTensor(f"logits:{int(pt[0, 0])}") if want_logits else None

    m._bind_gdn_prefill_scratch = _bind
    m._unbind_gdn_prefill_scratch = lambda prev: log.append(("unbind",))
    m._park_gdn_scratch = lambda: log.append(("park",))
    m._unpark_gdn_scratch = lambda: log.append(("unpark",))
    m.prefill_traced_chunked = _prefill
    m._write_gdn_slot = lambda slot, rec, conv: log.append(("write_slot", slot, len(rec)))
    return m


@pytest.mark.parametrize("dev_copy", ["0", "1", "2"])
def test_prefill_paged_slots_row_bookkeeping(monkeypatch, dev_copy):
    monkeypatch.setenv("QWEN36_PLAIN_GDN_SLOT_DEVICE_COPY", dev_copy)
    monkeypatch.delenv("QWEN36_PREFILL_LOGITS_FAST", raising=False)
    log = []
    m = _fake_model(monkeypatch, log)
    # call 1: short prompt (block 3, slot 5) then a long prompt's FIRST chunk (block 40, slot 6)
    pt = torch.tensor([[3, 0, 0], [40, 41, 42]], dtype=torch.int32)
    toks = [torch.zeros(1, 300, dtype=torch.long), torch.zeros(1, C, dtype=torch.long)]
    out = m.prefill_paged_slots(
        toks,
        pt,
        [5, 6],
        valid_lens=[300, C],
        start_positions=[0, 0],
        resume_mask=[False, False],
        final_mask=[True, False],
    )
    assert float(out[0].flatten()[0]) == 3.0 and torch.count_nonzero(out[1]) == 0  # call order; zero row
    writes = [e for e in log if e[0] in ("write_slot", "fill_cache", "write_index", "sync_hist")]
    slots_written = {e[1] if e[0] == "write_slot" else e[-1] for e in writes if e[0] != "sync_hist"}
    assert slots_written == {5}, writes  # only the final row writes, to ITS slot
    if dev_copy != "0":  # the host path repacks the history inside write_slot
        assert {e[2] for e in writes if e[0] == "sync_hist"} == {5}
    if dev_copy == "2":
        assert all(e[2] == "scratch_rec0@3" or e[2] == "scratch_rec1@3" for e in writes if e[0] == "fill_cache")
    assert "park" not in [e[0] for e in log]
    assert m._cp_planner.owner == ScratchOwner(40, C, parked=False)

    # call 2: a rider (block 4, slot 7) FIRST in call order + the continuation's final tail (block 40, slot 2)
    log.clear()
    pt = torch.tensor([[4, 0, 0], [40, 41, 42]], dtype=torch.int32)
    toks = [torch.zeros(1, 90, dtype=torch.long), torch.zeros(1, C + 100, dtype=torch.long)]
    out = m.prefill_paged_slots(
        toks,
        pt,
        [7, 2],
        valid_lens=[90, C + 100],
        start_positions=[0, C],
        resume_mask=[False, True],
        final_mask=[True, True],
    )
    order = [e for e in log if e[0] in ("prefill", "park", "unpark")]
    # the continuation runs first (no copy: the scratch still holds it), then the rider (no owner left to park)
    assert order == [("prefill", 40, C, C + 100, True), ("prefill", 4, 0, 90, True)]
    assert [float(o.flatten()[0]) for o in out] == [4.0, 40.0]  # call order, not execution order
    writes = [e for e in log if e[0] in ("write_slot", "fill_cache", "write_index")]
    assert {e[1] if e[0] == "write_slot" else e[-1] for e in writes} == {7, 2}
    if dev_copy == "2":
        by_slot = {(e[3], e[2]) for e in writes if e[0] == "fill_cache"}
        assert (2, "scratch_rec0@40") in by_slot and (7, "scratch_rec0@4") in by_slot
    assert m._cp_planner.owner is None


def test_prefill_paged_slots_parks_before_short_only_call(monkeypatch):
    monkeypatch.setenv("QWEN36_PLAIN_GDN_SLOT_DEVICE_COPY", "0")
    log = []
    m = _fake_model(monkeypatch, log)
    one = torch.tensor([[40, 41]], dtype=torch.int32)
    m.prefill_paged_slots(
        [torch.zeros(1, C, dtype=torch.long)],
        one,
        [1],
        valid_lens=[C],
        start_positions=[0],
        resume_mask=[False],
        final_mask=[False],
    )
    log.clear()
    m.prefill_paged_slots(
        [torch.zeros(1, 512, dtype=torch.long)],
        torch.tensor([[9, 0]], dtype=torch.int32),
        [3],
        valid_lens=[512],
        start_positions=[0],
        resume_mask=[False],
        final_mask=[True],
    )
    assert [e[0] for e in log if e[0] in ("park", "prefill", "unpark")] == ["park", "prefill"]
    log.clear()
    m.prefill_paged_slots(
        [torch.zeros(1, 2 * C, dtype=torch.long)],
        one,
        [1],
        valid_lens=[2 * C],
        start_positions=[C],
        resume_mask=[True],
        final_mask=[True],
    )
    assert [e[0] for e in log if e[0] in ("park", "prefill", "unpark")] == ["unpark", "prefill"]


def test_failed_call_clears_the_owner(monkeypatch, expect_error):
    monkeypatch.setenv("QWEN36_PLAIN_GDN_SLOT_DEVICE_COPY", "0")
    log = []
    m = _fake_model(monkeypatch, log)
    one = torch.tensor([[40, 41]], dtype=torch.int32)
    m.prefill_paged_slots(
        [torch.zeros(1, C, dtype=torch.long)],
        one,
        [1],
        valid_lens=[C],
        start_positions=[0],
        resume_mask=[False],
        final_mask=[False],
    )

    def boom(*a, **k):
        raise RuntimeError("device failure")

    m.prefill_traced_chunked = boom
    with expect_error(RuntimeError, "device failure"):
        m.prefill_paged_slots(
            [torch.zeros(1, 2 * C, dtype=torch.long)],
            one,
            [1],
            valid_lens=[2 * C],
            start_positions=[C],
            resume_mask=[True],
            final_mask=[False],
        )
    assert m._cp_planner.owner is None
    assert log[-1] == ("unbind",)
