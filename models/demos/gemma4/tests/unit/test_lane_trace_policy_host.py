# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Window 11's lessons on the traced lane prefill, host-only: the call plan over the captured frames (the replay
cost model), the gathered lm_head (one projection and one readback per call instead of one per round), the real
rounds handed to a padded replay, the capture that closes and releases a failed trace, the bounded-sliding length
cap at parse time, and unique warmup block rows. No device: ttnn's layout/slice/readback and the bridge are
recording stubs.

    pytest models/demos/gemma4/tests/unit/test_lane_trace_policy_host.py -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.lane_prefill import (
    LANE_TRACE_READBACK_ROWS,
    lane_prefill_trace_plan,
    lane_trace_gather_indices,
    lane_trace_readback_rows,
    lane_unique_block_rows,
    parse_lane_prefill_trace_buckets,
)

pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM  # noqa: E402

LANES, LANE_SLOTS, CHUNK, VOCAB, WIDTH = 4, 32, 4096, 8, 6
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention"]
SCRATCH = {0: 7, 1: 0, 2: 7}


# ── the call plan ──────────────────────────────────────────────────────────────────────────────────


def test_plan_returns_prompts_early_on_the_measured_frame_walls():
    """Window 12: one 32-round frame made every prompt of a 29-round step wait 1.6 s. On the window-13 Galaxy walls
    (1: 63, 8: 325, 16: 650, 32: 1250 ms) the 8-round frame costs 41 ms per round against 39 and the 1-round call
    63, so the plan runs full 8-round calls first and the remainder as 1-round calls when that returns prompts
    earlier within the 10 % slack."""
    f3 = {(1, 128), (8, 128), (32, 128)}
    assert lane_prefill_trace_plan(f3, 11, 128) == [(8, 8), (1, 1), (1, 1), (1, 1)], "8 then three 1-round calls"
    assert lane_prefill_trace_plan(f3, 22, 128) == [(8, 8), (8, 8)] + [(1, 1)] * 6, "six 1s (378 ms) over a padded 8"
    assert lane_prefill_trace_plan(f3, 29, 128) == [(8, 8)] * 3 + [(1, 1)] * 5, "window 12's 113-prompt step"
    assert (
        lane_prefill_trace_plan(f3, 32, 128) == [(8, 8)] * 4
    ), "four 8-round calls beat one 32 (1.30 vs 1.25 s total, 0.8 vs 1.25 s mean wait)"
    assert lane_prefill_trace_plan(f3, 1, 128) == [(1, 1)] and lane_prefill_trace_plan(f3, 2, 128) == [(1, 1), (1, 1)]
    assert lane_prefill_trace_plan(f3, 7, 128) == [(8, 7)], "seven 1-round calls (441 ms) lose to one padded 8"
    f4 = f3 | {(16, 128)}
    assert lane_prefill_trace_plan(f4, 11, 128) == [(8, 8), (1, 1), (1, 1), (1, 1)]
    assert lane_prefill_trace_plan(f4, 22, 128) == [(8, 8), (8, 8)] + [(1, 1)] * 6, "a 16 never beats 8s"
    assert lane_prefill_trace_plan(f4, 25, 128) == [(8, 8)] * 3 + [(1, 1)]
    assert lane_prefill_trace_plan(f4, 40, 128) == [(8, 8)] * 5, "beyond the largest frame: more calls"
    assert lane_prefill_trace_plan({(1, 128), (8, 128)}, 32, 128) == [(8, 8)] * 4, "window 10's four 8-round replays"
    assert lane_prefill_trace_plan(f4, 29, 128, slack=0) == [(32, 29)], "no slack: one 32 is the fastest total (1.25 s)"
    assert lane_prefill_trace_plan(f4, 30, 128, slack=0) == [(32, 30)]
    assert lane_prefill_trace_plan(f4, 11, 512) is None, "no frame at this length"
    assert lane_prefill_trace_plan(f3 | {(8, 512)}, 11, 512) == [(8, 8), (8, 3)], "the only frame at 512"
    assert lane_prefill_trace_plan(f4, 11, 128, single_chunk=False) is None
    assert lane_prefill_trace_plan(f4, 11, 128, cached_prefix=True) is None
    assert lane_prefill_trace_plan(set(), 3, 128) is None and lane_prefill_trace_plan(f4, 0, 128) is None


def test_plan_follows_the_frame_cost_table(monkeypatch):
    f3 = {(1, 128), (8, 128), (32, 128)}
    flat = {1: 1000, 8: 1000, 32: 1000}
    assert lane_prefill_trace_plan(f3, 11, 128, frame_ms=flat) == [(32, 11)], "every call the same: one call"
    cheap_1 = {1: 1, 8: 1000, 32: 100000}
    assert lane_prefill_trace_plan(f3, 11, 128, frame_ms=cheap_1) == [(1, 1)] * 11
    monkeypatch.setenv("GEMMA4_LANE_TRACE_FRAME_MS", "1:1000,8:1000,32:1000")
    assert lane_prefill_trace_plan(f3, 11, 128) == [(32, 11)]
    monkeypatch.setenv("GEMMA4_LANE_TRACE_FRAME_MS", "bad,8:x")
    assert lane_prefill_trace_plan(f3, 11, 128) == [(8, 8), (1, 1), (1, 1), (1, 1)], "malformed entries: defaults"
    monkeypatch.delenv("GEMMA4_LANE_TRACE_FRAME_MS", raising=False)
    monkeypatch.setenv("GEMMA4_LANE_TRACE_PLAN_SLACK", "0")
    assert lane_prefill_trace_plan(f3, 30, 128) == [(32, 30)]
    monkeypatch.setenv("GEMMA4_LANE_TRACE_PLAN_SLACK", "bad")
    assert lane_prefill_trace_plan(f3, 30, 128) == [(8, 8)] * 3 + [(1, 1)] * 6, "a bad slack: the default"
    # a size the table lacks costs the linear fallback (40 + 36 per round): a 4-round frame at 184 ms
    assert lane_prefill_trace_plan({(4, 128), (8, 128)}, 4, 128) == [(4, 4)]
    assert lane_prefill_trace_plan({(4, 128), (8, 128)}, 4, 128, call_ms=1000, round_ms=1) == [(8, 4)]


def test_parse_caps_the_frame_length_under_bounded_sliding(monkeypatch):
    """Above the short-prefill cap the prefill stashes the sliding tail with a host write the capture cannot record
    (TT_FATAL inside begin_trace_capture); such a bucket is dropped at parse time, never captured."""
    kept, dropped = parse_lane_prefill_trace_buckets(
        "1x128,1x512", max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=True
    )
    assert kept == {(1, 128)} and dropped == ["1x512"]
    kept, _ = parse_lane_prefill_trace_buckets(
        "1x128,1x512", max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=True, short_max=512
    )
    assert kept == {(1, 128), (1, 512)}
    monkeypatch.setenv("GEMMA4_TRACE_SHORT_PREFILL_MAX", "1024")
    kept, _ = parse_lane_prefill_trace_buckets(
        "1x128,1x512,1x2048", max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=True
    )
    assert kept == {(1, 128), (1, 512)}
    kept, _ = parse_lane_prefill_trace_buckets(
        "1x512,2x512", max_chunk=CHUNK, lane_slots=LANE_SLOTS, bounded_sliding=False
    )
    assert kept == {(1, 512), (2, 512)}, "hybrid groups stash nothing: every trace length"


def test_unique_block_rows_give_every_user_its_own_blocks():
    t = lane_unique_block_rows(LANES * 8, 2, pool=1000, width=6)
    assert tuple(t.shape) == (32, 6) and t.dtype == torch.int32
    used = t[:, :2].flatten().tolist()
    assert len(set(used)) == 64 and 0 not in used and max(used) < 1000, "distinct, never the null block"
    assert not t[:, 2:].any(), "the columns past the frame's blocks stay null"
    wrap = lane_unique_block_rows(4, 3, pool=5)
    assert tuple(wrap.shape) == (4, 3) and set(wrap.flatten().tolist()) == {1, 2, 3, 4}, "wraps inside the pool"
    assert not lane_unique_block_rows(4, 3, pool=0).any() and not lane_unique_block_rows(4, 3, pool=1).any()
    assert tuple(lane_unique_block_rows(0, 3, pool=10).shape) == (0, 3)


# ── the gathered lm_head ────────────────────────────────────────────────────────────────────────────


def _fake(shape, layout="TILE"):
    import ttnn

    return SimpleNamespace(shape=tuple(shape), dtype=ttnn.bfloat16, layout=layout, deallocate=lambda *a: None)


def _lm_head_rig(monkeypatch, *, concat_raises=False):
    """ttnn's layout/slice/concat/pad/readback as recording fakes; the readback row j of lane device d carries
    ``1000 * j + d`` in every column, so a wrong row or lane shows in the output."""
    import ttnn

    rec = {"slices": [], "reads": [], "lm_head": [], "concat": 0}
    V = VOCAB

    def _to_layout(t, layout, **kw):
        return _fake(t.shape, "RM" if layout == ttnn.ROW_MAJOR_LAYOUT else "TILE")

    def _slice(t, start, end, *a, **kw):
        rec["slices"].append((t.layout, tuple(start), tuple(end)))
        return _fake((1, 1, end[2] - start[2], end[3] - start[3]), t.layout)

    def _concat(ts, dim):
        if concat_raises:
            raise RuntimeError("no row-major concat today")
        rec["concat"] += 1
        return _fake((1, 1, sum(t.shape[2] for t in ts), ts[0].shape[3]), ts[0].layout)

    def _pad(t, padding, value):
        return _fake((1, 1, t.shape[2] + padding[2][1], t.shape[3]), t.layout)

    def _lm_head(x, is_decode=False, **kw):
        rec["lm_head"].append((tuple(x.shape), x.layout))
        return _fake((1, 1, 32, V))

    def _shards(t):
        return [SimpleNamespace(dev=d, rows=t.shape[2]) for d in range(32)]

    def _to_torch(sh):
        rec["reads"].append((sh.dev, sh.rows))
        rows = torch.arange(sh.rows, dtype=torch.float32).reshape(-1, 1) * 1000 + sh.dev
        return rows.repeat(1, V).reshape(1, 1, sh.rows, V)

    monkeypatch.setattr(ttnn, "to_layout", _to_layout)
    monkeypatch.setattr(ttnn, "slice", _slice)
    monkeypatch.setattr(ttnn, "concat", _concat)
    monkeypatch.setattr(ttnn, "pad", _pad)
    monkeypatch.setattr(ttnn, "get_device_tensors", _shards)
    monkeypatch.setattr(ttnn, "to_torch", _to_torch)
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lanes=LANES),
        vocab_size=V,
        lane_device_indices=lambda: [0, 1, 2, 3],
        _apply_lm_head=_lm_head,
    )
    return model, rec


def test_gathered_lm_head_projects_once_and_reads_only_the_real_rounds_rows(monkeypatch):
    from models.demos.gemma4.tt.generator import Gemma4Generator

    monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "1")
    model, rec = _lm_head_rig(monkeypatch)
    k, S = 8, 128
    last_r = [24, 40, 24, 24, 100, 24, 24, 24]
    hidden = _fake((1, 1, k * S, 2816))
    out = Gemma4Generator._lane_logits_from_hidden(model, hidden, k, S, last_r, deallocate=False, rounds=[0, 1, 4])
    assert tuple(out.shape) == (LANES * k, VOCAB)
    rm_slices = [sl for sl in rec["slices"] if sl[0] == "RM"]
    assert rm_slices[:3] == [
        ("RM", (0, 0, 24, 0), (1, 1, 25, 2816)),
        ("RM", (0, 0, 128 + 40, 0), (1, 1, 128 + 41, 2816)),
        ("RM", (0, 0, 4 * 128 + 100, 0), (1, 1, 4 * 128 + 101, 2816)),
    ], "each real round's last-token row, from the row-major hidden states"
    assert rec["concat"] == 1 and rec["lm_head"] == [((1, 1, 32, 2816), "TILE")], "one 32-row tile, one projection"
    assert rm_slices[3] == ("RM", (0, 0, 0, 0), (1, 1, 3, VOCAB)), "only the three rows are read back"
    assert rec["reads"] == [(0, 3), (1, 3), (2, 3), (3, 3)], "one read per lane"
    for ln in range(LANES):
        for j, r in enumerate([0, 1, 4]):
            assert float(out[ln * k + r, 0]) == 1000 * j + ln, "row j of lane ln's read lands at (lane, round)"
        for r in (2, 3, 5, 6, 7):
            assert not out[ln * k + r].any(), "scratch rounds stay zero"

    # a single real round reads one row; 32 real rounds read the whole tile without the row-major detour
    model, rec = _lm_head_rig(monkeypatch)
    Gemma4Generator._lane_logits_from_hidden(model, _fake((1, 1, 128, 2816)), 1, 128, [24], deallocate=False)
    assert rec["concat"] == 0 and rec["reads"] == [(d, 1) for d in range(4)]
    model, rec = _lm_head_rig(monkeypatch)
    Gemma4Generator._lane_logits_from_hidden(model, _fake((1, 1, 32 * 128, 2816)), 32, 128, [5] * 32, deallocate=False)
    assert rec["lm_head"] == [((1, 1, 32, 2816), "TILE")] and rec["reads"] == [(d, 32) for d in range(4)]


def test_gathered_lm_head_falls_back_to_the_per_round_form(monkeypatch):
    from models.demos.gemma4.tt.generator import Gemma4Generator

    monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "1")
    model, rec = _lm_head_rig(monkeypatch, concat_raises=True)
    k, S = 4, 128
    out = Gemma4Generator._lane_logits_from_hidden(
        model, _fake((1, 1, k * S, 2816)), k, S, [40, 24, 24, 70], deallocate=False, rounds=[0, 3]
    )
    assert getattr(model, "_g4_lane_lmhead_gather_warned", False), "warned once"
    tile_slices = [sl for sl in rec["slices"] if sl[0] == "TILE"]
    assert tile_slices == [
        ("TILE", (0, 0, 32, 0), (1, 1, 64, 2816)),
        ("TILE", (0, 0, 3 * 128 + 64, 0), (1, 1, 3 * 128 + 96, 2816)),
    ], "the real rounds' tile-aligned windows only"
    assert len(rec["lm_head"]) == 2 and rec["reads"] == [(d, 32) for d in range(4)] * 2
    for ln in range(LANES):
        assert float(out[ln * k + 0, 0]) == 1000 * (40 % 32) + ln and float(out[ln * k + 3, 0]) == 1000 * (70 % 32) + ln
        assert not out[ln * k + 1].any() and not out[ln * k + 2].any()

    monkeypatch.setenv("GEMMA4_LANE_LMHEAD_GATHER", "0")
    model, rec = _lm_head_rig(monkeypatch)
    Gemma4Generator._lane_logits_from_hidden(model, _fake((1, 1, 128, 2816)), 1, 128, [24], deallocate=False)
    assert rec["concat"] == 0 and len(rec["lm_head"]) == 1 and rec["reads"] == [(d, 32) for d in range(4)]


# ── the lm_head inside the trace: the gather index, the warmed readback sizes, the prompt lanes ────────


def test_readback_rows_and_gather_indices_are_fixed_shapes(expect_error):
    """Window 12: the eager gathered lm_head's concat / pad / slice shapes followed the real-round count, so every
    first-seen count compiled on the mesh inside the call (8x128 calls of 860-1125 ms against 295). The trace now
    gathers 32 rows by a fixed-shape index and a replay reads one of six warmed row counts."""
    assert LANE_TRACE_READBACK_ROWS == (1, 2, 4, 8, 16, 32)
    assert [lane_trace_readback_rows(n) for n in (1, 2, 3, 4, 5, 8, 9, 16, 17, 29, 32)] == [
        1, 2, 4, 4, 8, 8, 16, 16, 32, 32, 32,
    ]  # fmt: skip
    assert lane_trace_readback_rows(3, rows=2) == 2 and lane_trace_readback_rows(0) == 1
    idx = lane_trace_gather_indices([0, 1, 4], [24, 40, 24, 24, 100, 24, 24, 24], 128)
    assert len(idx) == 32 and idx[:3] == [24, 128 + 40, 4 * 128 + 100], "round r's last token at r * S + last_r[r]"
    assert idx[3:] == [24] * 29, "the padding rows repeat the first real row (never read)"
    assert lane_trace_gather_indices([0], [5], 128) == [5] * 32
    with expect_error(ValueError, "at least one real round"):
        lane_trace_gather_indices([], [5], 128)
    with expect_error(ValueError, "exceed the 32-row gather"):
        lane_trace_gather_indices(list(range(33)), [5] * 33, 128)


def test_trace_readback_reads_a_warmed_row_count_and_only_the_prompt_lanes(monkeypatch):
    from models.demos.gemma4.tt.generator import Gemma4Generator

    model, rec = _lm_head_rig(monkeypatch)
    k = 8
    logits = _fake((1, 1, 32, VOCAB), "RM")
    out = torch.zeros(LANES * k, VOCAB)
    Gemma4Generator._lane_trace_logits_readback(model, logits, k, [0, 1, 4], [1, 3], out)
    assert rec["slices"] == [("RM", (0, 0, 0, 0), (1, 1, 4, VOCAB))], "3 real rounds read the warmed 4-row slice"
    assert rec["reads"] == [(1, 4), (3, 4)] and rec["lm_head"] == [], "the prompt lanes only; no projection here"
    for ln in (1, 3):
        for j, r in enumerate([0, 1, 4]):
            assert float(out[ln * k + r, 0]) == 1000 * j + ln, "row j of lane ln's read lands at (lane, round)"
        for r in (2, 3, 5, 6, 7):
            assert not out[ln * k + r].any(), "scratch rounds stay zero"
    assert not out[0:k].any() and not out[2 * k : 3 * k].any(), "scratch lanes stay zero"

    # the served c1 frame: one real round on one lane reads one row of one shard
    model, rec = _lm_head_rig(monkeypatch)
    out = torch.zeros(LANES, VOCAB)
    Gemma4Generator._lane_trace_logits_readback(model, _fake((1, 1, 32, VOCAB), "RM"), 1, [0], [2], out)
    assert rec["slices"] == [("RM", (0, 0, 0, 0), (1, 1, 1, VOCAB))] and rec["reads"] == [(2, 1)]
    assert float(out[2, 0]) == 2 and not out[0].any() and not out[1].any() and not out[3].any()

    # a full 32-round frame reads the whole tile without a slice
    model, rec = _lm_head_rig(monkeypatch)
    out = torch.zeros(LANES * 32, VOCAB)
    Gemma4Generator._lane_trace_logits_readback(
        model, _fake((1, 1, 32, VOCAB), "RM"), 32, list(range(32)), [0, 1, 2, 3], out
    )
    assert rec["slices"] == [] and rec["reads"] == [(d, 32) for d in range(4)]
    assert float(out[3 * 32 + 31, 0]) == 1000 * 31 + 3


def test_gather_lm_head_inside_the_trace_runs_fixed_shapes(monkeypatch):
    """The trace body's tail: hidden [1, 1, k * S, H] -> row-major -> embedding over the 32-row index -> one
    32-row tile -> lm_head -> row-major logits; no shape follows the call."""
    import ttnn
    from models.demos.gemma4.tt.generator import Gemma4Generator

    model, rec = _lm_head_rig(monkeypatch)
    rec["embedding"] = []
    monkeypatch.setattr(
        ttnn,
        "embedding",
        lambda idx, w, dtype=None: (
            rec["embedding"].append((idx, w.shape, w.layout)),
            _fake((1, 32, w.shape[-1]), "RM"),
        )[1],
    )
    monkeypatch.setattr(ttnn, "unsqueeze_to_4D", lambda t: _fake((1, 1) + tuple(t.shape[-2:]), t.layout))
    out = Gemma4Generator._lane_trace_gather_lm_head(model, _fake((1, 1, 8 * 128, 2816)), "index")
    assert rec["embedding"] == [("index", (1, 1, 1024, 2816), "RM")], "the row-major hidden states are the table"
    assert rec["lm_head"] == [((1, 1, 32, 2816), "TILE")] and out.shape == (1, 1, 32, VOCAB) and out.layout == "RM"


def test_capture_adds_the_index_input_and_warms_the_readback(monkeypatch):
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt import common as tt_common

    monkeypatch.setenv("GEMMA4_LANE_LMHEAD_IN_TRACE", "1")
    inst, model, rec = _capture_rig(monkeypatch, fail_in=0)
    copied = []
    monkeypatch.setattr(
        tt_common, "copy_host_to_device", lambda host, **kw: (copied.append(len(host)), ["t", "p", "i"])[1]
    )
    warmed = []
    monkeypatch.setattr(Gemma4Generator, "_g4_warm_lane_trace_readback", lambda self, m, out: warmed.append(out.shape))
    assert Gemma4Generator._g4_capture_lane_prefill_trace(inst, (1, 128), [[]]) is True
    entry = inst._g4_lane_trace_registry()[(1, 128)]
    assert copied == [3] and entry["logits_in_trace"] is True and entry["inputs"] == ["t", "p", "i"]
    assert warmed == [(1, 1, 128, 2816)], "the readback slices are warmed on the trace's output once"

    monkeypatch.setenv("GEMMA4_LANE_LMHEAD_IN_TRACE", "0")
    inst, model, rec = _capture_rig(monkeypatch, fail_in=0)
    copied.clear()
    monkeypatch.setattr(tt_common, "copy_host_to_device", lambda host, **kw: (copied.append(len(host)), ["t", "p"])[1])
    assert Gemma4Generator._g4_capture_lane_prefill_trace(inst, (1, 128), [[]]) is True
    assert copied == [2] and inst._g4_lane_trace_registry()[(1, 128)]["logits_in_trace"] is False


# ── the replay: real rounds, the stash on an error ───────────────────────────────────────────────────


def _buffers(tag):
    return [SimpleNamespace(tag=f"{tag}{i}", buffer_address=lambda i=i: 0x1000 * (i + 1)) for i in range(3)]


def _replay_instance(monkeypatch, *, in_trace=False):
    import ttnn
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt import common as tt_common

    calls = {"executed": [], "rounds": [], "copied": [], "reads": []}
    model = SimpleNamespace(
        mesh_device="mesh",
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES),
        lane_slots=LANE_SLOTS,
        vocab_size=VOCAB,
        _g4_pf_ms={},
        _persistent_pt_by_batch={LANES: _buffers("a")},
        update_persistent_per_layer_page_tables=lambda tables, force=False: None,
        prepare_prefill_inputs_trace=lambda tokens, **kw: ("tokens", None, None, "legacy"),
    )
    model._g4_pf_add = lambda key, seconds: model._g4_pf_ms.__setitem__(
        key, model._g4_pf_ms.get(key, 0.0) + seconds * 1000.0
    )
    inst = object.__new__(Gemma4Generator)
    inst.model = [model]
    inst._g4_lane_prefill_traces = {
        (1, 128): {
            "trace_id": 7,
            "inputs": ["dev_tokens", "dev_table"] + (["dev_index"] if in_trace else []),
            "output": "logits" if in_trace else "hidden",
            "widths": [WIDTH] * 3,
            "needed_blocks": 2,
            "rows": LANES,
            "tables": Gemma4Generator._g4_lane_trace_table_fingerprint(model, LANES),
            "logits_in_trace": in_trace,
        }
    }
    monkeypatch.setattr(
        tt_common,
        "copy_host_to_device",
        lambda host, device_tensors=None, **kw: calls["copied"].append((list(host), list(device_tensors or []))),
    )
    monkeypatch.setattr(Gemma4Generator, "_lane_trace_index_host", staticmethod(lambda model, idx: list(idx)))

    def _readback(model, logits, k, real, lane_ids, out):
        calls["reads"].append((logits, k, list(real), list(lane_ids)))
        out[:] = 9.0

    monkeypatch.setattr(Gemma4Generator, "_lane_trace_logits_readback", staticmethod(_readback))
    monkeypatch.setattr(ttnn, "execute_trace", lambda mesh, tid, cq_id=0, blocking=False: calls["executed"].append(tid))

    def _logits(model, hidden, k, seq_len, last_r, *, deallocate, rounds=None, lanes=None):
        calls["rounds"].append(rounds)
        return torch.full((LANES * k, VOCAB), 9.0)

    inst._lane_logits_from_hidden = _logits
    return inst, model, calls


def _replay_call(inst, **kw):
    tokens = torch.zeros(LANES, 1, 128, dtype=torch.long)
    tables = torch.ones(LANES, 1, 2, dtype=torch.int32)
    per_layer = [torch.ones(LANES, WIDTH, dtype=torch.int32) * (i + 1) for i in range(3)]
    return inst._g4_lane_prefill_trace_replay((1, 128), tokens, tables, per_layer, [object()], [25] * LANES, **kw)


def test_replay_with_the_lm_head_in_the_trace_writes_the_index_and_reads_the_prompt_lanes(monkeypatch):
    inst, model, calls = _replay_instance(monkeypatch, in_trace=True)
    out = _replay_call(inst, rounds=[0], lanes=[2])
    assert out is not None and float(out[0, 0]) == 9.0 and calls["executed"] == [7]
    host, dev = calls["copied"][-1]
    assert dev == ["dev_tokens", "dev_table", "dev_index"] and host[2] == [24] * 32, "the gather index: last token 24"
    assert calls["reads"] == [("logits", 1, [0], [2])], "one real round on lane 2: one shard read"
    assert calls["rounds"] == [], "the eager gathered lm_head never runs for a trace that projects itself"
    _replay_call(inst)
    assert calls["reads"][-1] == ("logits", 1, [0], [0, 1, 2, 3]), "all lanes by default"
    assert "stage" in model._g4_pf_ms and "replay" in model._g4_pf_ms and "readback" in model._g4_pf_ms

    # a frame captured without the index keeps the eager gathered lm_head, and names its lanes to it
    inst, model, calls = _replay_instance(monkeypatch)
    _replay_call(inst, rounds=[0], lanes=[1])
    assert calls["rounds"] == [[0]] and calls["copied"][-1][1] == ["dev_tokens", "dev_table"] and calls["reads"] == []


def test_replay_names_its_real_rounds_and_clears_the_stash_on_an_error(monkeypatch):
    import ttnn

    inst, model, calls = _replay_instance(monkeypatch)
    assert _replay_call(inst, rounds=[0]) is not None and calls["rounds"] == [[0]]
    assert _replay_call(inst) is not None and calls["rounds"][-1] is None, "all rounds by default"
    assert not hasattr(model, "_active_page_tables_per_layer")

    def _boom(mesh, tid, cq_id=0, blocking=False):
        raise RuntimeError("execute_trace boom")

    monkeypatch.setattr(ttnn, "execute_trace", _boom)
    try:
        _replay_call(inst)
    except RuntimeError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("the error must travel")
    assert not hasattr(model, "_active_page_tables_per_layer"), "no stale per-layer stash after an error"
    assert (1, 128) in inst._g4_lane_prefill_traces, "an execution error is not a buffer move: the frame stays"


# ── the capture: a failed capture is closed and released ──────────────────────────────────────────


def _capture_rig(monkeypatch, *, fail_in):
    """The capture over host stubs: the trace body raises in the compile pass (1), inside the capture (2) or not
    at all (0)."""
    import ttnn
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt import common as tt_common

    rec = {"begin": 0, "end": [], "release": [], "forward": 0}
    model = SimpleNamespace(
        mesh_device="mesh",
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES),
        lane_slots=LANE_SLOTS,
        layers=[],
        prepare_prefill_inputs_trace=lambda tokens, **kw: ("tokens", None, None, "legacy"),
        _replicate_to_mesh_mapper=lambda: None,
    )
    inst = object.__new__(Gemma4Generator)
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK, max_seq_len=4096)]
    inst._effective_paged_block_size = lambda kv: 64
    inst._release_all_sliding_prefill_tails = lambda *a, **k: None
    inst._g4_arm_alloc_check = lambda *a, **k: None

    def _forward(m, inputs, k, kv):
        rec["forward"] += 1
        if rec["forward"] == fail_in:
            raise RuntimeError(f"TT_FATAL in pass {fail_in}")
        return _fake((1, 1, k * 128, 2816))

    monkeypatch.setattr(Gemma4Generator, "_g4_lane_trace_forward", staticmethod(_forward))
    monkeypatch.setattr(tt_common, "copy_host_to_device", lambda host, **kw: ["dev_tokens", "dev_table"])
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(
        ttnn, "begin_trace_capture", lambda mesh, cq_id=0: (rec.__setitem__("begin", rec["begin"] + 1), 5)[1]
    )
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, tid, cq_id=0: rec["end"].append(tid))
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: rec["release"].append(tid))
    return inst, model, rec


def test_capture_closes_and_releases_a_capture_that_failed_midway(monkeypatch):
    from models.demos.gemma4.tt.generator import Gemma4Generator

    inst, model, rec = _capture_rig(monkeypatch, fail_in=2)
    try:
        Gemma4Generator._g4_capture_lane_prefill_trace(inst, (1, 128), [[]])
    except RuntimeError as e:
        assert "pass 2" in str(e), "the error travels"
    else:
        raise AssertionError("the capture must re-raise")
    assert rec["begin"] == 1 and rec["end"] == [5] and rec["release"] == [5], "closed and released, never left open"
    assert model._prefill_trace_mode is False and not hasattr(model, "_active_page_tables_per_layer")
    assert (1, 128) not in inst._g4_lane_trace_registry()

    inst, model, rec = _capture_rig(monkeypatch, fail_in=1)
    try:
        Gemma4Generator._g4_capture_lane_prefill_trace(inst, (1, 128), [[]])
    except RuntimeError:
        pass
    assert rec["begin"] == 0 and rec["end"] == [] and rec["release"] == [], "a compile-pass failure opened nothing"

    inst, model, rec = _capture_rig(monkeypatch, fail_in=0)
    assert Gemma4Generator._g4_capture_lane_prefill_trace(inst, (1, 128), [[]]) is True
    entry = inst._g4_lane_trace_registry()[(1, 128)]
    assert rec["end"] == [5] and rec["release"] == [] and entry["trace_id"] == 5
    assert entry["needed_blocks"] == 2 and entry["rows"] is None, "no layers on the stub: nothing keyed"


# ── the router under the plan ───────────────────────────────────────────────────────────────────────


def _bridge(monkeypatch, *, prompts, captured):
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1")
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda cls: True))
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda cls: prompts))
    calls = {"replays": [], "rounds": [], "eager": [], "lanes": []}
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES),
        lane_slots=LANE_SLOTS,
        hf_config=SimpleNamespace(layer_types=LAYER_TYPES),
        vocab_size=VOCAB,
        _lane_scratch_block_for=SCRATCH.__getitem__,
        update_persistent_per_layer_page_tables=lambda tables, force=False: None,
    )
    inst = object.__new__(Gemma4ForCausalLM)
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK)]
    inst._g4_lane_prefill_traces = {b: {"trace_id": 1} for b in captured}

    def _replay(bucket, tokens, page_tables, per_layer, kv_cache, prompt_lens, rounds=None, lanes=None):
        calls["replays"].append((bucket, tuple(tokens.shape)))
        calls["rounds"].append(list(rounds))
        calls["lanes"].append(lanes)
        k = bucket[0]
        return (500 + torch.arange(LANES * k, dtype=torch.float32)).reshape(-1, 1).repeat(1, VOCAB)

    def _eager(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        calls["eager"].append(tuple(tokens.shape))
        k = int(tokens.shape[1]) if tokens.dim() == 3 else 1
        return (100 + torch.arange(LANES * k, dtype=torch.float32)).reshape(-1, 1).repeat(1, VOCAB)

    inst._g4_lane_prefill_trace_replay = _replay
    inst.prefill_forward_lanes = _eager
    inst.prefill_forward = lambda **kw: torch.full((int(kw["tokens"].shape[0]), 1, VOCAB), -1.0)
    return inst, calls


def _step(slots, plens, seq=100):
    B = len(slots)
    tokens = torch.zeros(B, seq, dtype=torch.long)
    for q in range(B):
        tokens[q, : plens[q]] = 1 + q
    kwargs = {"tokens": tokens, "empty_slots": list(slots), "prompt_lens": list(plens), "kv_cache": [object()]}
    tables = [torch.arange(1, 1 + B * WIDTH, dtype=torch.int32).reshape(B, WIDTH) * (i + 1) for i in range(3)]
    return kwargs, tables


def test_router_plans_calls_over_the_captured_frames(monkeypatch):
    """Window 11's step histogram under the plan (frames 1, 8, 32; the window-13 walls): 11 rounds -> 8 + 1 + 1 + 1,
    22 -> 8 + 8 + six 1s; a 16-round frame changes nothing. Every replay is handed the real rounds of its call."""
    f3 = {(1, 128), (8, 128), (32, 128)}
    inst, calls = _bridge(monkeypatch, prompts=8, captured=f3)
    kwargs, tables = _step([ln * 32 + r for r in range(11) for ln in range(4)], [50] * 44)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [c[0] for c in calls["replays"]] == [(8, 128), (1, 128), (1, 128), (1, 128)]
    assert calls["rounds"] == [list(range(8)), [0], [0], [0]], "a full 8-round call, then the three 1-round calls"

    inst, calls = _bridge(monkeypatch, prompts=8, captured=f3)
    kwargs, tables = _step([ln * 32 + r for r in range(22) for ln in range(4)], [50] * 88)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert [c[0] for c in calls["replays"]] == [(8, 128)] * 2 + [(1, 128)] * 6 and calls["rounds"] == [
        list(range(8))
    ] * 2 + [[0]] * 6, "two full 8-round calls, the 6-round remainder as 1-round calls"
    for q in range(88):
        ln, r = q % 4, q // 4
        expect = 500 + ln * 8 + (r % 8) if r < 16 else 500 + ln
        assert float(out[q, 0, 0]) == expect, "rows of every call land on their requests"

    f4 = f3 | {(16, 128)}
    inst, calls = _bridge(monkeypatch, prompts=8, captured=f4)
    kwargs, tables = _step([ln * 32 + r for r in range(11) for ln in range(4)], [50] * 44)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [c[0] for c in calls["replays"]] == [(8, 128), (1, 128), (1, 128), (1, 128)], "a 16 never beats 8s"
    assert calls["eager"] == []

    # a single prompt after a wave: the same 1-round frame, three scratch lanes
    inst, calls = _bridge(monkeypatch, prompts=8, captured=f4)
    for _ in range(3):
        kwargs, tables = _step([ln * 32 + r for r in range(32) for ln in range(4)], [50] * 128)
        assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    kwargs, tables = _step([70], [25])
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert calls["replays"][-1] == ((1, 128), (LANES, 1, 128)) and calls["rounds"][-1] == [0]
    assert float(out[0, 0, 0]) == 500 + 2 and [c[0] for c in calls["replays"][:-1]] == [(8, 128)] * 12
    assert calls["lanes"] == [None] * 12 + [[2]], "full rounds read every lane; the single prompt names lane 2"
