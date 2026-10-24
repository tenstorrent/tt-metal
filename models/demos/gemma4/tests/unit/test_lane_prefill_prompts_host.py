# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""k prompts per lane per gathered prefill step (the c128 prefill-call policy), host-only.

``model_capabilities["lane_prefill_group"]`` is a count (GEMMA4_LANE_PREFILL_PROMPTS under the lane router): the
vllm-tt-plugin caps each lane's new prefill admissions per step at it, and the router prefills the gathered step
as ``[lanes, k, seq]`` lane-batched walks -- every request routed, lanes short of a round on scratch rows, the
per-layer tables in the lanes' decode frame and uploaded per call. No device: the count, the round and frame
arithmetic, the warmup shape and the router's calls on a stubbed bridge.

    pytest models/demos/gemma4/tests/unit/test_lane_prefill_prompts_host.py -p no:cacheprovider
"""

from types import SimpleNamespace

import pytest
import torch

# generator_vllm imports vLLM at module scope; the tt-metal unit-test runners have no vLLM.
pytest.importorskip("vllm")
from models.demos.gemma4.tt import generator_vllm as gv  # noqa: E402
from models.demos.gemma4.tt.generator_vllm import (  # noqa: E402
    Gemma4ForCausalLM,
    _capability_count,
    galaxy_lane_prefill_prompts,
    lane_batched_warmup_shape,
    lane_prefill_frame_tables,
    lane_prefill_rounds,
)

LANES, LANE_SLOTS, CHUNK, VOCAB, WIDTH = 4, 32, 4096, 8, 6
LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention"]
SCRATCH = {0: 7, 1: 0, 2: 7}  # bounded sliding rings keep a reserved scratch block; full layers use null block 0


# ── the declaration ──────────────────────────────────────────────────────────────────────────────────────────


def test_prompts_follow_the_env_rails(monkeypatch):
    for key in ("GEMMA4_GALAXY_LANES", "GEMMA4_LANE_PREFILL_ROUTER", "GEMMA4_LANE_PREFILL_PROMPTS"):
        monkeypatch.delenv(key, raising=False)
    assert galaxy_lane_prefill_prompts() == 0
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_ROUTER", "1")
    assert galaxy_lane_prefill_prompts() == 1, "the router alone is one prompt per lane, as before"
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_PROMPTS", "8")
    assert galaxy_lane_prefill_prompts() == 8
    assert Gemma4ForCausalLM._lane_prefill_prompts() == 8 and Gemma4ForCausalLM._lane_prefill_group()
    # A plugin that reads the declaration as a Boolean (the stock admission cap) still sees it on.
    assert bool(Gemma4ForCausalLM.model_capabilities["lane_prefill_group"]()) is True
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_ROUTER", "0")
    assert galaxy_lane_prefill_prompts() == 0, "the count needs the router"


@pytest.mark.parametrize("raw", ["0", "-2", "eight", "1.5"])
def test_malformed_prompt_counts_fail_the_launch(monkeypatch, raw, expect_error):
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_ROUTER", "1")
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_PROMPTS", raw)
    with expect_error(ValueError, "GEMMA4_LANE_PREFILL_PROMPTS"):
        galaxy_lane_prefill_prompts()


def test_capability_count_reads_booleans_ints_and_callables():
    assert _capability_count({"k": True}, "k") == 1
    assert _capability_count({"k": False}, "k") == 0
    assert _capability_count({"k": 6}, "k") == 6
    assert _capability_count({"k": -3}, "k") == 0
    assert _capability_count({"k": lambda: 4}, "k") == 4
    assert _capability_count({}, "k") == 0 and _capability_count(None, "k") == 0


# ── rounds and frames ────────────────────────────────────────────────────────────────────────────────────────


def test_rounds_without_scratch_are_full_rounds_in_bucket_order_until_lengths_differ():
    plens = {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 90}
    buckets = [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert lane_prefill_rounds(buckets, plens, allow_scratch=False) == [[0, 1, 2, 3]]
    buckets = [[0, 4], [1, 5], [2, 6], [3]]
    assert lane_prefill_rounds(buckets, plens, allow_scratch=False) == [[0, 1, 2, 3]], "lane 3 has no second"
    assert lane_prefill_rounds([[0], [1], [2], []], plens, allow_scratch=False) == [], "no full round"


def test_rounds_with_scratch_route_every_request_by_length():
    plens = {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 90, 6: 90, 7: 50}
    buckets = [[0, 4], [1, 5], [2, 6, 7], [3]]
    rounds = lane_prefill_rounds(buckets, plens, allow_scratch=True)
    assert rounds == [[0, 1, 2, 3], [4, None, 7, None], [None, 5, 6, None]]
    assert sorted(q for rnd in rounds for q in rnd if q is not None) == list(range(8))
    for rnd in rounds:
        assert len({plens[q] for q in rnd if q is not None}) == 1


def test_frame_tables_place_users_per_lane_and_scratch_the_missing_ones():
    pts = [torch.arange(1, 1 + 8 * WIDTH, dtype=torch.int32).reshape(8, WIDTH) for _ in LAYER_TYPES]
    one = lane_prefill_frame_tables(pts, [[0, None, 2, 3]], LANE_SLOTS, SCRATCH.__getitem__)
    assert [tuple(t.shape) for t in one] == [(LANES, WIDTH)] * 3, "k == 1 keeps the one-row-per-lane layout"
    assert torch.equal(one[0][1], torch.full((WIDTH,), 7, dtype=torch.int32)) and int(one[1][1, 0]) == 0
    assert torch.equal(one[1][2], pts[1][2])

    two = lane_prefill_frame_tables(pts, [[0, 1, 2, 3], [4, None, 5, None]], LANE_SLOTS, SCRATCH.__getitem__)
    assert [tuple(t.shape) for t in two] == [(LANES * LANE_SLOTS, WIDTH)] * 3, "k > 1 uses the lanes' decode frame"
    frame = two[0]
    for ln, q in enumerate([0, 1, 2, 3]):
        assert torch.equal(frame[ln * LANE_SLOTS], pts[0][q])
    assert torch.equal(frame[0 * LANE_SLOTS + 1], pts[0][4]) and torch.equal(frame[2 * LANE_SLOTS + 1], pts[0][5])
    assert torch.equal(frame[1 * LANE_SLOTS + 1], torch.full((WIDTH,), 7, dtype=torch.int32)), "scratch round"
    assert int(two[1][3 * LANE_SLOTS + 1, 0]) == 0, "full layers scratch through null block 0"
    assert not frame[2:LANE_SLOTS].any() and not frame[LANE_SLOTS + 2 : 2 * LANE_SLOTS].any(), "unused rows null"
    assert lane_prefill_frame_tables([None], [[0, 1, 2, 3]], LANE_SLOTS, SCRATCH.__getitem__) == [None]


@pytest.mark.parametrize(
    ("prompts", "expected"),
    [(0, (1, CHUNK)), (1, (1, CHUNK)), (3, (3, 1024)), (8, (8, 512)), (64, (32, 128))],
)
def test_batched_warmup_shape_keeps_the_stream_within_the_chunk(prompts, expected):
    k, seq = lane_batched_warmup_shape(prompts, LANE_SLOTS, CHUNK)
    assert (k, seq) == expected
    assert k * seq <= CHUNK and seq % 128 == 0


# ── the router ───────────────────────────────────────────────────────────────────────────────────────────────


def _bridge(monkeypatch, *, prompts, batched=True):
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1" if batched else "0")
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda cls: True))
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda cls: prompts))
    calls = {"lanes": [], "installs": [], "serial": []}
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES),
        lane_slots=LANE_SLOTS,
        hf_config=SimpleNamespace(layer_types=LAYER_TYPES),
        vocab_size=VOCAB,
        _lane_scratch_block_for=SCRATCH.__getitem__,
        update_persistent_per_layer_page_tables=lambda tables, force=False: calls["installs"].append(
            ([None if t is None else t.clone() for t in tables], force)
        ),
    )
    inst = object.__new__(Gemma4ForCausalLM)
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK)]

    def _lanes(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        assert hasattr(model, "_active_page_tables_per_layer")
        assert calls["installs"] and calls["installs"][-1][0][1] is not None
        calls["lanes"].append((tokens.clone(), page_tables.clone(), list(prompt_lens), slot_ids))
        k = int(tokens.shape[1]) if tokens.dim() == 3 else 1
        base = 100 * len(calls["lanes"])
        return (base + torch.arange(LANES * k, dtype=torch.float32)).reshape(-1, 1).repeat(1, VOCAB)

    def _serial(page_tables_per_layer=None, **kwargs):
        calls["serial"].append((kwargs["tokens"].clone(), list(kwargs["prompt_lens"]), list(kwargs["empty_slots"])))
        return torch.full((int(kwargs["tokens"].shape[0]), 1, VOCAB), -1.0)

    inst.prefill_forward_lanes = _lanes
    inst.prefill_forward = _serial
    return inst, model, calls


def _step(slots, plens, seq=100):
    B = len(slots)
    tokens = torch.zeros(B, seq, dtype=torch.long)
    for q in range(B):
        tokens[q, : plens[q]] = 1 + q
    kwargs = {"tokens": tokens, "empty_slots": list(slots), "prompt_lens": list(plens), "kv_cache": [object()]}
    tables = [torch.arange(1, 1 + B * WIDTH, dtype=torch.int32).reshape(B, WIDTH) * (i + 1) for i in range(3)]
    return kwargs, tables


def test_router_prefills_two_prompts_per_lane_as_one_lane_batched_call(monkeypatch):
    inst, model, calls = _bridge(monkeypatch, prompts=2)
    slots = [0, 32, 64, 96, 1, 33, 65, 97]  # two per lane, the step in lane-merge order
    kwargs, tables = _step(slots, [50] * 8)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and tuple(out.shape) == (8, 1, VOCAB) and calls["serial"] == []
    assert len(calls["lanes"]) == 1
    toks, pts, plens, slot_ids = calls["lanes"][0]
    assert tuple(toks.shape) == (LANES, 2, 128) and tuple(pts.shape) == (LANES, 2, WIDTH) and slot_ids is None
    assert plens == [50] * 8
    for ln in range(LANES):
        for r in range(2):
            q = r * LANES + ln
            assert int(toks[ln, r, 0]) == 1 + q and int(toks[ln, r, 49]) == 1 + q and int(toks[ln, r, 50]) == 0
            assert torch.equal(pts[ln, r], tables[1][q])
            assert float(out[q, 0, 0]) == 100 + ln * 2 + r, "row lane * k + r of the call's logits"
    frame, force = calls["installs"][0]
    assert [tuple(t.shape) for t in frame] == [(LANES * LANE_SLOTS, WIDTH)] * 3
    assert torch.equal(frame[0][1 * LANE_SLOTS + 1], tables[0][5]) and torch.equal(
        frame[2][3 * LANE_SLOTS], tables[2][3]
    )
    assert force is True, "the first prefill upload after another writer is forced"
    assert model._g4_page_table_writer_by_rows == {LANES * LANE_SLOTS: "prefill"}
    assert not hasattr(model, "_active_page_tables_per_layer"), "the router leaves no stash behind"


def test_router_fills_short_lanes_with_scratch_rows_and_splits_lengths_into_rounds(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=2)
    slots = [0, 32, 64, 96, 1, 33]  # lanes 2 and 3 have one prompt; lane 1's second is longer
    kwargs, tables = _step(slots, [50, 50, 50, 50, 50, 90])
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and calls["serial"] == [] and len(calls["lanes"]) == 1
    toks, pts, plens, _ = calls["lanes"][0]
    assert tuple(toks.shape) == (LANES, 3, 128), "rounds: [0,1,2,3], [4,-,-,-], [-,5,-,-]"
    for ln in range(LANES):
        assert plens[ln * 3 + 0] == 50
        assert plens[ln * 3 + 1] == 50, "scratch rows carry the round's length"
        assert plens[ln * 3 + 2] == 90
    assert int(toks[0, 1, 0]) == 5 and not toks[1, 1].any() and not toks[2, 1].any() and not toks[3, 1].any()
    assert int(toks[1, 2, 0]) == 6 and int(toks[1, 2, 89]) == 6 and not toks[0, 2].any()
    assert torch.equal(pts[0, 1], tables[1][4]) and not pts[1, 1].any(), "scratch legacy rows are null"
    frame = calls["installs"][0][0]
    assert torch.equal(frame[0][1 * LANE_SLOTS + 1], torch.full((WIDTH,), 7, dtype=torch.int32))
    assert torch.equal(frame[0][1 * LANE_SLOTS + 2], tables[0][5])
    assert float(out[4, 0, 0]) == 100 + 0 * 3 + 1 and float(out[5, 0, 0]) == 100 + 1 * 3 + 2


def test_router_splits_calls_by_the_prefill_chunk(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=2)
    slots = [0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98]  # three per lane (a stock plugin admits past the cap)
    kwargs, tables = _step(slots, [50] * 12)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [(LANES, 3, 128)], "every round that fits the chunk"

    inst, _model, calls = _bridge(monkeypatch, prompts=4)
    kwargs, tables = _step(slots, [2000] * 12, seq=2000)  # 2048-token prompts: two rounds per 4096 chunk
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [(LANES, 2, 2048), (LANES, 2048)]
    assert [tuple(t.shape) for t in calls["installs"][1][0]] == [(LANES, WIDTH)] * 3
    assert [f for _t, f in calls["installs"]] == [True, True], "each row count's first prefill upload is forced"

    inst, _model, calls = _bridge(monkeypatch, prompts=4)
    kwargs, tables = _step(slots[:8], [3000] * 8, seq=3000)  # 4096-token prompts: one round per call
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [(LANES, 4096), (LANES, 4096)]
    assert [f for _t, f in calls["installs"]] == [True, False], "same rows, same writer: no forced re-upload"


def test_router_without_the_batched_walk_runs_every_round_in_the_one_user_layout(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=2, batched=False)
    kwargs, tables = _step([0, 32, 64, 96, 1, 33], [50] * 6)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [(LANES, 128), (LANES, 128)]
    assert calls["lanes"][1][2] == [50] * LANES and not calls["lanes"][1][0][2].any(), "scratch rows still fill"
    assert calls["serial"] == []


def test_router_with_one_prompt_per_lane_keeps_full_rounds_and_the_serial_remainder(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=1)
    kwargs, tables = _step([0, 32, 64, 96, 1, 33, 64 + 1, 97, 2], [50] * 9)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [(LANES, 128), (LANES, 128)]
    assert [f for _t, f in calls["installs"]] == [True, False], "every call uploads its tables"
    assert calls["serial"] and calls["serial"][0][2] == [2] and float(out[8, 0, 0]) == -1.0
    inst, _model, calls = _bridge(monkeypatch, prompts=1)
    kwargs, tables = _step([0, 32, 64], [50] * 3)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is None, "B < lanes falls through"
