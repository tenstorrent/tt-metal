# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The lane engine's decode-frame shapes (``tt/lane_frame.py``), host-only: the engine's own host tensors pass, the
window-5 mistake (a ``[1, rows]`` tensor handed to the paged cache update as its positions) and every other
frame-vs-row slip are refused with the tensor named.

    pytest models/demos/gemma4/tests/unit/test_lane_frame_host.py -p no:cacheprovider
"""

import time
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin
from models.demos.gemma4.tt.lane_prefill import (
    check_lane_decode_frame,
    fit_table_width,
    lane_decode_frame_shapes,
    lane_prefill_trace_bucket_for,
    lane_prefill_trace_buckets,
    lane_prefill_trace_choice,
    lane_trace_rounds_for_call,
    parse_lane_prefill_trace_buckets,
)
from models.tt_transformers.tt.common import get_padded_prefill_len

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:  # the CI unit runners have no vLLM: those sections skip
    _HAS_VLLM = False

LANES, ROWS, BLOCKS, HIDDEN = 4, 32, 3, 2560


def _engine_shapes():
    return {name: host for name, (host, _col, _dim) in lane_decode_frame_shapes(LANES, ROWS, BLOCKS, HIDDEN).items()}


def test_the_engines_host_tensors_shard_to_the_column_frame():
    columns = check_lane_decode_frame(LANES, ROWS, **_engine_shapes())
    assert columns == {
        "hidden": (1, 1, ROWS, HIDDEN),
        "rope_positions": (1, ROWS),
        "cache_positions": (ROWS,),
        "page_table": (ROWS, BLOCKS),
    }
    # the (2,4) block: 4 lanes of 32 rows as well; one lane of 32 rows is the smallest frame
    assert check_lane_decode_frame(1, ROWS, cache_positions=(ROWS,))["cache_positions"] == (ROWS,)


def test_two_dimensional_cache_positions_are_the_window_5_mistake(expect_error):
    shapes = _engine_shapes()
    shapes["cache_positions"] = (1, ROWS)  # one row per column: the update reads ONE index for 32 rows
    with expect_error(ValueError, "cache_positions: rank 2"):
        check_lane_decode_frame(LANES, ROWS, **shapes)


@pytest.mark.parametrize(
    ("name", "shape", "message"),
    [
        ("rope_positions", (1, ROWS), "rope_positions: 32 rows on the lane-shard dim 1"),  # per column, not the frame
        ("page_table", (ROWS, BLOCKS), "page_table: 32 rows on the lane-shard dim 0"),
        ("hidden", (1, 1, ROWS, HIDDEN), "hidden: 32 rows on the lane-shard dim 2"),
        ("page_table", (LANES * ROWS,), "page_table: rank 1"),
    ],
)
def test_per_column_or_misranked_tensors_are_refused(expect_error, name, shape, message):
    shapes = _engine_shapes()
    shapes[name] = shape
    with expect_error(ValueError, message):
        check_lane_decode_frame(LANES, ROWS, **shapes)


def test_unknown_tensor_names_are_refused(expect_error):
    with expect_error(ValueError, "unknown decode-frame tensor"):
        check_lane_decode_frame(LANES, ROWS, masks=(1, ROWS))


# ===== merged from tests/unit/test_lane_capabilities.py: The lane-serving declarations the vllm-tt-plugin reads from ``model_capabilities``
try:
    from models.demos.gemma4.tt import generator_vllm as gv
except ImportError:  # vLLM absent: this section's tests skip
    pass


if _HAS_VLLM:
    CLASSES = [
        gv.Gemma4ForCausalLM,
        gv.Gemma4VisionForConditionalGeneration,
        gv.Gemma4MTPForCausalLM,
        gv.Gemma4DFlashForCausalLM,
    ]


def _declared(cls, key):
    value = cls.model_capabilities[key]
    return value() if callable(value) else value


if _HAS_VLLM:

    @pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
    @pytest.mark.parametrize("cls", CLASSES, ids=lambda c: c.__name__)
    def test_lane_declarations_follow_the_env_rails(cls, monkeypatch):
        monkeypatch.delenv("GEMMA4_GALAXY_LANES", raising=False)
        monkeypatch.delenv("GEMMA4_LANE_PREFILL_ROUTER", raising=False)
        monkeypatch.delenv("GEMMA4_LANE_PREFILL_PROMPTS", raising=False)
        monkeypatch.delenv("GEMMA4_LANE_PREFILL_CAPABILITY", raising=False)
        assert _declared(cls, "folds_dp_into_lanes") is False
        assert _declared(cls, "lane_prefill_group") == 0  # a count: the plugin caps admissions per lane at it
        assert not cls._lanes_fold() and not cls._lane_prefill_group() and cls._lane_prefill_prompts() == 0

        monkeypatch.setenv("GEMMA4_LANE_PREFILL_ROUTER", "1")
        assert _declared(cls, "lane_prefill_group") == 0  # needs the fold

        monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
        assert _declared(cls, "folds_dp_into_lanes") is True
        assert _declared(cls, "lane_prefill_group") == 1
        assert cls._lanes_fold() and cls._lane_prefill_group() and cls._lane_prefill_prompts() == 1

        monkeypatch.setenv("GEMMA4_LANE_PREFILL_PROMPTS", "8")
        assert _declared(cls, "lane_prefill_group") == 8 and cls._lane_prefill_prompts() == 8
        assert _declared(cls, "lane_prefill_router") is True and _declared(cls, "lane_prefill_prompts") == 8

        # GEMMA4_LANE_PREFILL_CAPABILITY=0: the plugin is told 0 (no per-lane cap, no gather) while the router and its
        # prompts-per-lane policy stay on (window 10: uncapped steps into the lane-batched frame)
        monkeypatch.setenv("GEMMA4_LANE_PREFILL_CAPABILITY", "0")
        assert _declared(cls, "lane_prefill_group") == 0
        assert bool(cls.model_capabilities["lane_prefill_group"]()) is False, "what the stock plugin reads"
        assert cls._lane_prefill_group() and cls._lane_prefill_prompts() == 8
        monkeypatch.setenv("GEMMA4_LANE_PREFILL_CAPABILITY", "1")
        assert _declared(cls, "lane_prefill_group") == 8


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_lane_declarations_are_resolved_at_config_time():
    # Callables, not import-time values: the plugin evaluates them in
    # check_and_update_config, once the catalog's env is in place.
    caps = gv.Gemma4ForCausalLM.model_capabilities
    assert callable(caps["folds_dp_into_lanes"]) and callable(caps["lane_prefill_group"])
    assert callable(caps["lane_prefill_router"]) and callable(caps["lane_prefill_prompts"])


# ===== merged from tests/unit/test_lane_prefill_prompts_host.py: k prompts per lane per gathered prefill step (the c128 prefill-call policy), host-only.
try:
    from models.demos.gemma4.tt import generator_vllm as gv
    from models.demos.gemma4.tt.generator_vllm import (
        Gemma4ForCausalLM,
        _capability_count,
        galaxy_lane_prefill_prompts,
        lane_batched_warmup_shape,
        lane_prefill_frame_tables,
        lane_prefill_rounds,
    )
except ImportError:  # vLLM absent: this section's tests skip
    pass


LANES_lane_prefill_prompts_host, LANE_SLOTS, CHUNK, VOCAB, WIDTH = 4, 32, 4096, 8, 6


LAYER_TYPES = ["sliding_attention", "full_attention", "sliding_attention"]


SCRATCH = {0: 7, 1: 0, 2: 7}  # bounded sliding rings keep a reserved scratch block; full layers use null block 0


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("raw", ["0", "-2", "eight", "1.5"])
def test_malformed_prompt_counts_fail_the_launch(monkeypatch, raw, expect_error):
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_ROUTER", "1")
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_PROMPTS", raw)
    with expect_error(ValueError, "GEMMA4_LANE_PREFILL_PROMPTS"):
        galaxy_lane_prefill_prompts()


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_capability_count_reads_booleans_ints_and_callables():
    assert _capability_count({"k": True}, "k") == 1
    assert _capability_count({"k": False}, "k") == 0
    assert _capability_count({"k": 6}, "k") == 6
    assert _capability_count({"k": -3}, "k") == 0
    assert _capability_count({"k": lambda: 4}, "k") == 4
    assert _capability_count({}, "k") == 0 and _capability_count(None, "k") == 0


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_rounds_without_scratch_are_full_rounds_in_bucket_order_until_lengths_differ():
    plens = {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 50, 6: 50, 7: 90}
    buckets = [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert lane_prefill_rounds(buckets, plens, allow_scratch=False) == [[0, 1, 2, 3]]
    buckets = [[0, 4], [1, 5], [2, 6], [3]]
    assert lane_prefill_rounds(buckets, plens, allow_scratch=False) == [[0, 1, 2, 3]], "lane 3 has no second"
    assert lane_prefill_rounds([[0], [1], [2], []], plens, allow_scratch=False) == [], "no full round"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_rounds_with_scratch_route_every_request_by_length():
    plens = {0: 50, 1: 50, 2: 50, 3: 50, 4: 50, 5: 90, 6: 90, 7: 50}
    buckets = [[0, 4], [1, 5], [2, 6, 7], [3]]
    rounds = lane_prefill_rounds(buckets, plens, allow_scratch=True)
    assert rounds == [[0, 1, 2, 3], [4, None, 7, None], [None, 5, 6, None]]
    assert sorted(q for rnd in rounds for q in rnd if q is not None) == list(range(8))
    for rnd in rounds:
        assert len({plens[q] for q in rnd if q is not None}) == 1


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_frame_tables_place_users_per_lane_and_scratch_the_missing_ones():
    pts = [torch.arange(1, 1 + 8 * WIDTH, dtype=torch.int32).reshape(8, WIDTH) for _ in LAYER_TYPES]
    one = lane_prefill_frame_tables(pts, [[0, None, 2, 3]], LANE_SLOTS, SCRATCH.__getitem__)
    assert [tuple(t.shape) for t in one] == [
        (LANES_lane_prefill_prompts_host, WIDTH)
    ] * 3, "k == 1 keeps the one-row-per-lane layout"
    assert torch.equal(one[0][1], torch.full((WIDTH,), 7, dtype=torch.int32)) and int(one[1][1, 0]) == 0
    assert torch.equal(one[1][2], pts[1][2])

    two = lane_prefill_frame_tables(pts, [[0, 1, 2, 3], [4, None, 5, None]], LANE_SLOTS, SCRATCH.__getitem__)
    assert [tuple(t.shape) for t in two] == [
        (LANES_lane_prefill_prompts_host * LANE_SLOTS, WIDTH)
    ] * 3, "k > 1 uses the lanes' decode frame"
    frame = two[0]
    for ln, q in enumerate([0, 1, 2, 3]):
        assert torch.equal(frame[ln * LANE_SLOTS], pts[0][q])
    assert torch.equal(frame[0 * LANE_SLOTS + 1], pts[0][4]) and torch.equal(frame[2 * LANE_SLOTS + 1], pts[0][5])
    assert torch.equal(frame[1 * LANE_SLOTS + 1], torch.full((WIDTH,), 7, dtype=torch.int32)), "scratch round"
    assert int(two[1][3 * LANE_SLOTS + 1, 0]) == 0, "full layers scratch through null block 0"
    assert not frame[2:LANE_SLOTS].any() and not frame[LANE_SLOTS + 2 : 2 * LANE_SLOTS].any(), "unused rows null"
    assert lane_prefill_frame_tables([None], [[0, 1, 2, 3]], LANE_SLOTS, SCRATCH.__getitem__) == [None]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize(
    ("prompts", "expected"),
    [(0, (1, CHUNK)), (1, (1, CHUNK)), (3, (3, 1024)), (8, (8, 512)), (64, (32, 128))],
)
def test_batched_warmup_shape_keeps_the_stream_within_the_chunk(prompts, expected):
    k, seq = lane_batched_warmup_shape(prompts, LANE_SLOTS, CHUNK)
    assert (k, seq) == expected
    assert k * seq <= CHUNK and seq % 128 == 0


def _bridge(monkeypatch, *, prompts, batched=True):
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1" if batched else "0")
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda cls: True))
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda cls: prompts))
    calls = {"lanes": [], "installs": [], "serial": []}
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES_lane_prefill_prompts_host),
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
        return (
            (base + torch.arange(LANES_lane_prefill_prompts_host * k, dtype=torch.float32))
            .reshape(-1, 1)
            .repeat(1, VOCAB)
        )

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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_prefills_two_prompts_per_lane_as_one_lane_batched_call(monkeypatch):
    inst, model, calls = _bridge(monkeypatch, prompts=2)
    slots = [0, 32, 64, 96, 1, 33, 65, 97]  # two per lane, the step in lane-merge order
    kwargs, tables = _step(slots, [50] * 8)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and tuple(out.shape) == (8, 1, VOCAB) and calls["serial"] == []
    assert len(calls["lanes"]) == 1
    toks, pts, plens, slot_ids = calls["lanes"][0]
    assert (
        tuple(toks.shape) == (LANES_lane_prefill_prompts_host, 2, 128)
        and tuple(pts.shape) == (LANES_lane_prefill_prompts_host, 2, WIDTH)
        and slot_ids is None
    )
    assert plens == [50] * 8
    for ln in range(LANES_lane_prefill_prompts_host):
        for r in range(2):
            q = r * LANES_lane_prefill_prompts_host + ln
            assert int(toks[ln, r, 0]) == 1 + q and int(toks[ln, r, 49]) == 1 + q and int(toks[ln, r, 50]) == 0
            assert torch.equal(pts[ln, r], tables[1][q])
            assert float(out[q, 0, 0]) == 100 + ln * 2 + r, "row lane * k + r of the call's logits"
    frame, force = calls["installs"][0]
    assert [tuple(t.shape) for t in frame] == [(LANES_lane_prefill_prompts_host * LANE_SLOTS, WIDTH)] * 3
    assert torch.equal(frame[0][1 * LANE_SLOTS + 1], tables[0][5]) and torch.equal(
        frame[2][3 * LANE_SLOTS], tables[2][3]
    )
    assert force is True, "the first prefill upload after another writer is forced"
    assert model._g4_page_table_writer_by_rows == {LANES_lane_prefill_prompts_host * LANE_SLOTS: "prefill"}
    assert not hasattr(model, "_active_page_tables_per_layer"), "the router leaves no stash behind"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_fills_short_lanes_with_scratch_rows_and_splits_lengths_into_rounds(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=2)
    slots = [0, 32, 64, 96, 1, 33]  # lanes 2 and 3 have one prompt; lane 1's second is longer
    kwargs, tables = _step(slots, [50, 50, 50, 50, 50, 90])
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and calls["serial"] == [] and len(calls["lanes"]) == 1
    toks, pts, plens, _ = calls["lanes"][0]
    assert tuple(toks.shape) == (LANES_lane_prefill_prompts_host, 3, 128), "rounds: [0,1,2,3], [4,-,-,-], [-,5,-,-]"
    for ln in range(LANES_lane_prefill_prompts_host):
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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_splits_calls_by_the_prefill_chunk(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=2)
    slots = [0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98]  # three per lane (a stock plugin admits past the cap)
    kwargs, tables = _step(slots, [50] * 12)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [
        (LANES_lane_prefill_prompts_host, 3, 128)
    ], "every round that fits the chunk"

    inst, _model, calls = _bridge(monkeypatch, prompts=4)
    kwargs, tables = _step(slots, [2000] * 12, seq=2000)  # 2048-token prompts: two rounds per 4096 chunk
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [
        (LANES_lane_prefill_prompts_host, 2, 2048),
        (LANES_lane_prefill_prompts_host, 2048),
    ]
    assert [tuple(t.shape) for t in calls["installs"][1][0]] == [(LANES_lane_prefill_prompts_host, WIDTH)] * 3
    assert [f for _t, f in calls["installs"]] == [True, True], "each row count's first prefill upload is forced"

    inst, _model, calls = _bridge(monkeypatch, prompts=4)
    kwargs, tables = _step(slots[:8], [3000] * 8, seq=3000)  # 4096-token prompts: one round per call
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [
        (LANES_lane_prefill_prompts_host, 4096),
        (LANES_lane_prefill_prompts_host, 4096),
    ]
    assert [f for _t, f in calls["installs"]] == [True, False], "same rows, same writer: no forced re-upload"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_without_the_batched_walk_runs_every_round_in_the_one_user_layout(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=2, batched=False)
    kwargs, tables = _step([0, 32, 64, 96, 1, 33], [50] * 6)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [
        (LANES_lane_prefill_prompts_host, 128),
        (LANES_lane_prefill_prompts_host, 128),
    ]
    assert (
        calls["lanes"][1][2] == [50] * LANES_lane_prefill_prompts_host and not calls["lanes"][1][0][2].any()
    ), "scratch rows still fill"
    assert calls["serial"] == []


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_with_one_prompt_per_lane_keeps_full_rounds_and_the_serial_remainder(monkeypatch):
    inst, _model, calls = _bridge(monkeypatch, prompts=1)
    kwargs, tables = _step([0, 32, 64, 96, 1, 33, 64 + 1, 97, 2], [50] * 9)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert [tuple(c[0].shape) for c in calls["lanes"]] == [
        (LANES_lane_prefill_prompts_host, 128),
        (LANES_lane_prefill_prompts_host, 128),
    ]
    assert [f for _t, f in calls["installs"]] == [True, False], "every call uploads its tables"
    assert calls["serial"] and calls["serial"][0][2] == [2] and float(out[8, 0, 0]) == -1.0
    inst, _model, calls = _bridge(monkeypatch, prompts=1)
    kwargs, tables = _step([0, 32, 64], [50] * 3)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is None, "B < lanes falls through"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_uploads_every_call_unless_the_evidence_knob_keeps_the_pre_fix_path(monkeypatch):
    """The lane router stashed its per-layer tables without uploading them: ``_page_tables_to_ttnn`` reuses a key's
    buffers unrefreshed, so every group after a key's first prefilled through the first group's block ids.
    GEMMA4_LANE_PREFILL_TABLE_UPLOAD=0 keeps that path for the before/after coherence evidence only."""
    inst, _model, calls = _bridge(monkeypatch, prompts=1)
    kwargs, tables = _step([0, 32, 64, 96, 1, 33, 65, 97], [50] * 8)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert len(calls["lanes"]) == 2 and len(calls["installs"]) == 2

    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TABLE_UPLOAD", "0")
    inst, model, calls = _bridge(monkeypatch, prompts=1)
    inst.prefill_forward_lanes = lambda tokens, page_tables, kv_cache, prompt_lens, slot_ids=None: (
        calls["lanes"].append(tuple(tokens.shape)) or torch.zeros(LANES_lane_prefill_prompts_host, VOCAB)
    )
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert len(calls["lanes"]) == 2 and calls["installs"] == [], "the pre-fix path: stash only"
    assert not hasattr(model, "_g4_page_table_writer_by_rows")


# ===== merged from tests/unit/test_lane_prefill_trace_host.py: The traced lane prefill (item (a) of the Galaxy plan), host-only: the bucket set (``tt/lane_prefill_trace.py``),
try:
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


(
    LANES_lane_prefill_trace_host,
    LANE_SLOTS_lane_prefill_trace_host,
    CHUNK_lane_prefill_trace_host,
    VOCAB_lane_prefill_trace_host,
    WIDTH_lane_prefill_trace_host,
) = (4, 32, 4096, 8, 6)


LAYER_TYPES_lane_prefill_trace_host = ["sliding_attention", "full_attention", "sliding_attention"]


SCRATCH_lane_prefill_trace_host = {0: 7, 1: 0, 2: 7}


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_bucket_parsing_keeps_frames_within_the_chunk_the_slots_and_the_trace_lengths():
    kept, dropped = parse_lane_prefill_trace_buckets(
        "1x128, 2x128,8x512,3x200,40x128,1x8192,bad,,",
        max_chunk=CHUNK_lane_prefill_trace_host,
        lane_slots=LANE_SLOTS_lane_prefill_trace_host,
        bounded_sliding=False,
    )
    assert kept == {(1, 128), (2, 128), (8, 512)}
    assert dropped == ["3x200", "40x128", "1x8192", "bad"]
    kept, dropped = parse_lane_prefill_trace_buckets(
        "1x128,2x128,8x512",
        max_chunk=CHUNK_lane_prefill_trace_host,
        lane_slots=LANE_SLOTS_lane_prefill_trace_host,
        bounded_sliding=True,
    )
    assert kept == {(1, 128)}, "bounded sliding: one valid length per device, so one round per column"
    assert dropped == ["2x128", "8x512"]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_buckets_follow_the_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_LANE_PREFILL_TRACE", raising=False)
    monkeypatch.delenv("GEMMA4_LANE_PREFILL_TRACE_BUCKETS", raising=False)
    assert lane_prefill_trace_buckets(
        max_chunk=CHUNK_lane_prefill_trace_host, lane_slots=LANE_SLOTS_lane_prefill_trace_host, bounded_sliding=True
    ) == (set(), [])
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE", "1")
    assert lane_prefill_trace_buckets(
        max_chunk=CHUNK_lane_prefill_trace_host, lane_slots=LANE_SLOTS_lane_prefill_trace_host, bounded_sliding=True
    ) == ({(1, 128)}, [])
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE_BUCKETS", "1x128,4x512")
    assert lane_prefill_trace_buckets(
        max_chunk=CHUNK_lane_prefill_trace_host, lane_slots=LANE_SLOTS_lane_prefill_trace_host, bounded_sliding=False
    )[0] == {
        (1, 128),
        (4, 512),
    }


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_replay_choice_needs_an_exact_single_chunk_cold_match():
    captured = {(1, 128), (4, 256)}
    assert lane_prefill_trace_choice(captured, 1, 128) == (1, 128)
    assert lane_prefill_trace_choice(captured, 4, 256) == (4, 256)
    assert lane_prefill_trace_choice(captured, 2, 128) is None
    assert lane_prefill_trace_choice(captured, 1, 256) is None
    assert lane_prefill_trace_choice(captured, 1, 128, single_chunk=False) is None
    assert lane_prefill_trace_choice(captured, 1, 128, cached_prefix=True) is None
    assert lane_prefill_trace_choice(set(), 1, 128) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_bucket_for_a_call_pads_up_to_the_next_captured_frame():
    captured = {(1, 128), (8, 128), (32, 128), (2, 512)}
    assert lane_prefill_trace_bucket_for(captured, 8, 128) == (8, 128), "exact first"
    assert lane_prefill_trace_bucket_for(captured, 7, 128) == (8, 128), "the smallest captured frame above"
    assert lane_prefill_trace_bucket_for(captured, 9, 128) == (32, 128)
    assert lane_prefill_trace_bucket_for(captured, 33, 128) is None, "nothing captured covers 33 rounds"
    assert lane_prefill_trace_bucket_for(captured, 1, 512) == (2, 512)
    assert lane_prefill_trace_bucket_for(captured, 1, 1024) is None
    assert lane_prefill_trace_bucket_for(captured, 7, 128, single_chunk=False) is None
    assert lane_prefill_trace_bucket_for(captured, 7, 128, cached_prefix=True) is None
    assert lane_prefill_trace_bucket_for(set(), 7, 128) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_traced_call_sizes_win_over_larger_eager_ones():
    captured = {(1, 128), (4, 128), (2, 512)}
    assert lane_trace_rounds_for_call(captured, 128, 32) == 4
    assert lane_trace_rounds_for_call(captured, 128, 3) == 1
    assert lane_trace_rounds_for_call(captured, 512, 8) == 2
    assert lane_trace_rounds_for_call(captured, 1024, 8) == 8, "no bucket at this length: the eager size"
    assert lane_trace_rounds_for_call(set(), 128, 8) == 8


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_fit_table_width_trims_or_null_pads_the_last_dim_only():
    pt = torch.arange(1, 1 + 4 * 6, dtype=torch.int32).reshape(4, 6)
    assert fit_table_width(pt, 6) is pt
    assert torch.equal(fit_table_width(pt, 2), pt[:, :2])
    wide = fit_table_width(pt, 9)
    assert tuple(wide.shape) == (4, 9) and torch.equal(wide[:, :6], pt) and not wide[:, 6:].any()
    frame = torch.ones(LANES_lane_prefill_trace_host * LANE_SLOTS_lane_prefill_trace_host, 3, dtype=torch.int32)
    assert tuple(fit_table_width(frame, 5).shape) == (
        LANES_lane_prefill_trace_host * LANE_SLOTS_lane_prefill_trace_host,
        5,
    )
    assert fit_table_width(None, 5) is None and fit_table_width("not a table", 5) == "not a table"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_trace_body_embeds_the_token_input_before_the_forward():
    """Window 9 stage 0: the forward was handed the rank-2 token tensor ("ShapeBase[] index out of range. 2 not in
    [-4, 2)" at seq_len = hidden_states.shape[2]); the trace body must run the device embed first, as the traced
    single-user prefill does, and hand the forward the embeddings with the transformed page table."""
    from models.demos.gemma4.tt.generator import Gemma4Generator

    calls = []
    tokens, page_table = SimpleNamespace(shape=(1, 2 * 128), name="tokens"), SimpleNamespace(name="page_table")
    embeds = SimpleNamespace(shape=(1, 1, 2 * 128, 2560), name="embeds")

    def _embed(t, pt, cpt, csi=None):
        calls.append(("embed", t.name, pt.name, cpt, csi))
        return embeds, pt, cpt, csi

    def _forward(x, **kw):
        calls.append(("forward", x.name, kw))
        return SimpleNamespace(shape=(1, 1, 2 * 128, 2560))

    model = SimpleNamespace(transform_and_embed_prefill_inputs_device=_embed, ttnn_prefill_forward=_forward)
    out = Gemma4Generator._g4_lane_trace_forward(model, [tokens, page_table], 2, "kv")
    assert tuple(out.shape) == (1, 1, 256, 2560)
    assert calls[0] == ("embed", "tokens", "page_table", None, None)
    kind, x_name, kw = calls[1]
    assert kind == "forward" and x_name == "embeds", "the forward sees the embeddings, never the token ids"
    assert kw["page_table"].name == "page_table" and kw["batch_size"] == 2 and kw["get_last_token"] == -1
    assert kw["valid_seq_lens"] is None and kw["kv_cache"] == "kv" and kw["user_id"] == 0


def _bridge_lane_prefill_trace_host(monkeypatch, *, prompts, captured, replay_returns=True):
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1")
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda cls: True))
    monkeypatch.setattr(Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda cls: prompts))
    calls = {"replays": [], "eager": [], "installs": [], "serial": []}
    model = SimpleNamespace(
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES_lane_prefill_trace_host),
        lane_slots=LANE_SLOTS_lane_prefill_trace_host,
        hf_config=SimpleNamespace(layer_types=LAYER_TYPES_lane_prefill_trace_host),
        vocab_size=VOCAB_lane_prefill_trace_host,
        _lane_scratch_block_for=SCRATCH_lane_prefill_trace_host.__getitem__,
        update_persistent_per_layer_page_tables=lambda tables, force=False: calls["installs"].append(len(tables)),
    )
    inst = object.__new__(Gemma4ForCausalLM)
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK_lane_prefill_trace_host)]
    inst._g4_lane_prefill_traces = {b: {"trace_id": 1} for b in captured}

    def _replay(bucket, tokens, page_tables, per_layer, kv_cache, prompt_lens, rounds=None, lanes=None):
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
        return (
            (500 + torch.arange(LANES_lane_prefill_trace_host * k, dtype=torch.float32))
            .reshape(-1, 1)
            .repeat(1, VOCAB_lane_prefill_trace_host)
        )

    def _eager(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        calls["eager"].append((tuple(tokens.shape), list(prompt_lens)))
        k = int(tokens.shape[1]) if tokens.dim() == 3 else 1
        return (
            (100 + torch.arange(LANES_lane_prefill_trace_host * k, dtype=torch.float32))
            .reshape(-1, 1)
            .repeat(1, VOCAB_lane_prefill_trace_host)
        )

    def _serial(page_tables_per_layer=None, **kwargs):
        calls["serial"].append(list(kwargs["empty_slots"]))
        return torch.full((int(kwargs["tokens"].shape[0]), 1, VOCAB_lane_prefill_trace_host), -1.0)

    inst._g4_lane_prefill_trace_replay = _replay
    inst.prefill_forward_lanes = _eager
    inst.prefill_forward = _serial
    return inst, calls


def _step_lane_prefill_trace_host(slots, plens, seq=100):
    B = len(slots)
    tokens = torch.zeros(B, seq, dtype=torch.long)
    for q in range(B):
        tokens[q, : plens[q]] = 1 + q
    kwargs = {"tokens": tokens, "empty_slots": list(slots), "prompt_lens": list(plens), "kv_cache": [object()]}
    tables = [
        torch.arange(1, 1 + B * WIDTH_lane_prefill_trace_host, dtype=torch.int32).reshape(
            B, WIDTH_lane_prefill_trace_host
        )
        * (i + 1)
        for i in range(3)
    ]
    return kwargs, tables


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_single_prompt_routes_through_the_captured_frame_with_scratch_lanes(monkeypatch):
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=1, captured={(1, 128)})
    kwargs, tables = _step_lane_prefill_trace_host(
        [33], [50]
    )  # one request on lane 1: B < lanes, which the eager router rejects
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert (
        out is not None
        and tuple(out.shape) == (1, 1, VOCAB_lane_prefill_trace_host)
        and calls["serial"] == []
        and calls["eager"] == []
    )
    ((bucket, toks, pts, per_layer, plens),) = calls["replays"]
    assert (
        bucket == (1, 128)
        and toks == (LANES_lane_prefill_trace_host, 1, 128)
        and pts == (LANES_lane_prefill_trace_host, 1, WIDTH_lane_prefill_trace_host)
    )
    assert (
        per_layer == [(LANES_lane_prefill_trace_host, WIDTH_lane_prefill_trace_host)] * 3
        and plens == [50] * LANES_lane_prefill_trace_host
    ), "scratch lanes carry the round's length"
    assert float(out[0, 0, 0]) == 500 + 1 * 1 + 0, "lane 1's row of the replay"
    assert calls["installs"] == [], "the replay uploads the fitted tables itself (the stub does not)"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_without_a_captured_frame_a_single_prompt_stays_on_the_eager_owner_path(monkeypatch):
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=1, captured=set())
    kwargs, tables = _step_lane_prefill_trace_host([33], [50])
    assert inst._try_lane_parallel_prefill(kwargs, tables) is None
    assert calls["replays"] == [] and calls["eager"] == []


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_captured_round_size_splits_a_gathered_step_into_traced_calls(monkeypatch):
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=8, captured={(1, 128)})
    slots = [0, 32, 64, 96, 1, 33, 65, 97]
    kwargs, tables = _step_lane_prefill_trace_host(slots, [50] * 8)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert [c[0] for c in calls["replays"]] == [(1, 128), (1, 128)], "two one-round replays, not one 2-round walk"
    assert calls["eager"] == []
    assert float(out[5, 0, 0]) == 500 + 1, "request 5 = lane 1, second round"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_mismatched_shape_stays_eager_and_a_refused_replay_falls_back(monkeypatch):
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=8, captured={(1, 128)})
    kwargs, tables = _step_lane_prefill_trace_host(
        [0, 32, 64, 96], [300] * 4, seq=300
    )  # a longer frame: no bucket at its padded length
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert calls["replays"] == [] and [c[0] for c in calls["eager"]] == [
        (LANES_lane_prefill_trace_host, get_padded_prefill_len(300))
    ]

    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=1, captured={(1, 128)}, replay_returns=False)
    kwargs, tables = _step_lane_prefill_trace_host([0, 32, 64, 96], [50] * 4)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert len(calls["replays"]) == 1 and [c[0] for c in calls["eager"]] == [(LANES_lane_prefill_trace_host, 128)]
    assert calls["installs"] == [3], "the eager fallback uploads the call's tables"
    assert float(out[2, 0, 0]) == 100 + 2


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_single_short_prompt_after_a_wave_of_gathered_steps_replays_the_same_frame(monkeypatch):
    """Window 9's last step (2026-10-25 03:26): after the c128 wave's gathered replays, one 25-token prompt alone
    (B=1 < lanes, lane 2) takes the same frame and the same replay path as the wave's steps and the c1 rows before
    it -- the device hang inside that replay was not a different route (the dump: the MoE combine of trace 0)."""
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=1, captured={(1, 128)})
    for _ in range(4):
        kwargs, tables = _step_lane_prefill_trace_host([0, 32, 64, 96], [128] * 4, seq=128)
        assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    kwargs, tables = _step_lane_prefill_trace_host([70], [25])  # slot 70 = lane 2, the first request after the wave
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and calls["eager"] == [] and calls["serial"] == []
    assert [c[0] for c in calls["replays"]] == [(1, 128)] * 5, "the same frame, the same path as the wave's steps"
    bucket, toks, pts, per_layer, plens = calls["replays"][-1]
    assert (
        toks == (LANES_lane_prefill_trace_host, 1, 128) and plens == [25] * LANES_lane_prefill_trace_host
    ), "three scratch lanes carry the prompt's length"
    assert float(out[0, 0, 0]) == 500 + 2, "lane 2's row"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_capture_width_is_the_serving_width_never_below_the_frames_blocks():
    from models.demos.gemma4.tt.lane_prefill import lane_trace_capture_width

    assert lane_trace_capture_width(2, 262144, 64) == 4096, "the plugin's table width at 256K"
    assert lane_trace_capture_width(2, 100, 64) == 2, "never narrower than the frame's own blocks"
    assert lane_trace_capture_width(2, 0, 64) == 2 and lane_trace_capture_width(0, 4096, 64) == 64


def _buffers(tag):
    return [SimpleNamespace(tag=f"{tag}{i}", buffer_address=lambda i=i: 0x1000 * (i + 1)) for i in range(3)]


def _replay_instance(monkeypatch, *, install=None):
    """A generator over a host-stubbed model whose key-``LANES_lane_prefill_trace_host`` persistent tables are three buffers; the replay's
    device steps (the H2D of the trace inputs, execute_trace, the lm_head readback) are recorded stubs. ``install``
    runs inside the per-layer table install (a grow or an allocation event under it)."""
    import ttnn
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt import common as tt_common

    calls = {"installs": [], "copies": 0, "executed": [], "logits": 0}
    model = SimpleNamespace(
        mesh_device="mesh",
        mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES_lane_prefill_trace_host),
        lane_slots=LANE_SLOTS_lane_prefill_trace_host,
        _persistent_pt_by_batch={LANES_lane_prefill_trace_host: _buffers("a")},
    )

    def _update(tables, force=False):
        calls["installs"].append([tuple(t.shape) for t in tables])
        if install is not None:
            install(model)

    model.update_persistent_per_layer_page_tables = _update
    model.prepare_prefill_inputs_trace = lambda tokens, **kw: ("tokens", None, None, "legacy")
    inst = object.__new__(Gemma4Generator)
    inst.model = [model]
    inst._g4_lane_prefill_traces = {
        (1, 128): {
            "trace_id": 7,
            "inputs": ["dev_tokens", "dev_table"],
            "output": "hidden",
            "widths": [WIDTH_lane_prefill_trace_host] * 3,
            "needed_blocks": 2,
            "rows": LANES_lane_prefill_trace_host,
            "tables": Gemma4Generator._g4_lane_trace_table_fingerprint(model, LANES_lane_prefill_trace_host),
        }
    }
    monkeypatch.setattr(
        tt_common, "copy_host_to_device", lambda host, **kw: calls.__setitem__("copies", calls["copies"] + 1)
    )
    monkeypatch.setattr(ttnn, "execute_trace", lambda mesh, tid, cq_id=0, blocking=False: calls["executed"].append(tid))

    def _logits(model, hidden, k, seq_len, last_r, *, deallocate, rounds=None):
        calls["logits"] += 1
        assert hidden == "hidden" and not deallocate and last_r == [24]
        return torch.full((LANES_lane_prefill_trace_host * k, VOCAB_lane_prefill_trace_host), 9.0)

    inst._lane_logits_from_hidden = _logits
    return inst, model, calls


def _replay_call(inst, model):
    tokens = torch.zeros(LANES_lane_prefill_trace_host, 1, 128, dtype=torch.long)
    tables = torch.ones(LANES_lane_prefill_trace_host, 1, 2, dtype=torch.int32)
    per_layer = [
        torch.ones(LANES_lane_prefill_trace_host, WIDTH_lane_prefill_trace_host, dtype=torch.int32) * (i + 1)
        for i in range(3)
    ]
    return inst._g4_lane_prefill_trace_replay(
        (1, 128), tokens, tables, per_layer, [object()], [25] * LANES_lane_prefill_trace_host
    )


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_replay_runs_the_frame_when_its_table_buffers_are_the_captured_ones(monkeypatch):
    inst, model, calls = _replay_instance(monkeypatch)
    out = _replay_call(inst, model)
    assert (
        out is not None
        and tuple(out.shape) == (LANES_lane_prefill_trace_host, VOCAB_lane_prefill_trace_host)
        and float(out[0, 0]) == 9.0
    )
    assert (
        calls["installs"] == [[(LANES_lane_prefill_trace_host, WIDTH_lane_prefill_trace_host)] * 3]
        and calls["copies"] == 1
        and calls["executed"] == [7]
    )
    assert calls["logits"] == 1 and (1, 128) in inst._g4_lane_prefill_traces
    assert not hasattr(model, "_active_page_tables_per_layer")


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_replay_drops_a_frame_whose_table_buffers_moved_between_calls(monkeypatch):
    """A wider eager install (a prompt at an uncaptured length) grew the key's buffers after the capture: the trace
    would read the freed buffer, so the call runs eager and the frame is gone for good."""
    inst, model, calls = _replay_instance(monkeypatch)
    model._persistent_pt_by_batch[LANES_lane_prefill_trace_host] = _buffers("grown")
    assert _replay_call(inst, model) is None
    assert calls["executed"] == [] and calls["copies"] == 0 and len(calls["installs"]) == 1
    assert (1, 128) not in inst._g4_lane_prefill_traces and not hasattr(model, "_active_page_tables_per_layer")
    assert _replay_call(inst, model) is None, "a dropped frame never replays"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_replay_drops_a_frame_reallocated_under_its_own_install(monkeypatch):
    def _grow(model):
        model._persistent_pt_by_batch[LANES_lane_prefill_trace_host] = _buffers("regrown")
        model._g4_persistent_alloc_event = "grown"

    inst, model, calls = _replay_instance(monkeypatch, install=_grow)
    assert _replay_call(inst, model) is None
    assert calls["executed"] == [] and (1, 128) not in inst._g4_lane_prefill_traces

    # an address change alone (same objects, moved buffers) is caught by the fingerprint too
    inst, model, calls = _replay_instance(monkeypatch)
    for b in model._persistent_pt_by_batch[LANES_lane_prefill_trace_host]:
        b.buffer_address = lambda: 0xBEEF
    assert _replay_call(inst, model) is None and calls["executed"] == []


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_fingerprint_is_none_without_persistent_tables_for_the_key():
    from models.demos.gemma4.tt.generator import Gemma4Generator

    assert Gemma4Generator._g4_lane_trace_table_fingerprint(SimpleNamespace(), LANES_lane_prefill_trace_host) is None
    model = SimpleNamespace(_persistent_pt_by_batch={LANES_lane_prefill_trace_host: _buffers("a")})
    assert Gemma4Generator._g4_lane_trace_table_fingerprint(model, None) is None
    assert Gemma4Generator._g4_lane_trace_table_fingerprint(model, 128) is None
    fp = Gemma4Generator._g4_lane_trace_table_fingerprint(model, LANES_lane_prefill_trace_host)
    assert (
        len(fp) == 3
        and fp[1][1] == 0x2000
        and fp == Gemma4Generator._g4_lane_trace_table_fingerprint(model, LANES_lane_prefill_trace_host)
    )


def _warm_instance(monkeypatch, *, trace_env, prompts=1, bounded=True, capture_raises=False):
    pass  # _instance is defined in this module now

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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_warmup_captures_the_configured_frames_after_the_eager_ones(monkeypatch):
    inst, model, calls = _warm_instance(monkeypatch, trace_env="1")
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert len(calls["lanes"]) == 1 and calls["captures"] == [(1, 128)]
    assert not hasattr(model, "_active_page_tables_per_layer")


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_warmup_captures_nothing_when_the_trace_is_off_and_survives_a_failed_capture(monkeypatch):
    inst, _model, calls = _warm_instance(monkeypatch, trace_env=None)
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["captures"] == []
    inst, model, calls = _warm_instance(monkeypatch, trace_env="1", capture_raises=True)
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["captures"] == [(1, 128)] and not hasattr(model, "_active_page_tables_per_layer")


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_router_pads_a_short_call_with_scratch_rounds_to_the_captured_frame(monkeypatch):
    """Window 10's steps (127 prompts = 32 rounds, 31 prompts = 8 rounds with a short last one): with 1x128 and
    8x128 captured, a 7-round call is padded with one all-scratch round and replays the 8-round frame."""
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=8, captured={(1, 128), (8, 128)})
    slots = [ln * 32 + r for r in range(7) for ln in range(4)]  # 7 per lane, lane-merge order
    kwargs, tables = _step_lane_prefill_trace_host(slots, [50] * 28)
    out = inst._try_lane_parallel_prefill(kwargs, tables)
    assert out is not None and calls["eager"] == [] and len(calls["replays"]) == 1
    bucket, toks, pts, per_layer, plens = calls["replays"][0]
    assert (
        bucket == (8, 128)
        and toks == (LANES_lane_prefill_trace_host, 8, 128)
        and pts == (LANES_lane_prefill_trace_host, 8, WIDTH_lane_prefill_trace_host)
    )
    assert (
        per_layer
        == [(LANES_lane_prefill_trace_host * LANE_SLOTS_lane_prefill_trace_host, WIDTH_lane_prefill_trace_host)] * 3
    ), "the 8-round frame's per-layer tables"
    assert plens == [50] * (LANES_lane_prefill_trace_host * 8), "the pad round carries a valid length"
    for q in range(28):
        ln, r = q % 4, q // 4
        assert float(out[q, 0, 0]) == 500 + ln * 8 + r, "rows of the real rounds only"

    # 32 rounds at 8 per call: four replays of the 8-round frame, none padded
    inst, calls = _bridge_lane_prefill_trace_host(monkeypatch, prompts=8, captured={(1, 128), (8, 128)})
    slots = [ln * 32 + r for r in range(32) for ln in range(4)]
    kwargs, tables = _step_lane_prefill_trace_host(slots, [50] * 128)
    assert inst._try_lane_parallel_prefill(kwargs, tables) is not None
    assert [c[0] for c in calls["replays"]] == [(8, 128)] * 4 and calls["eager"] == []


# ===== merged from tests/unit/test_warmup_lane_prefill_group.py: Boot-time warmups of the vLLM bridge, host-only.
try:
    from models.demos.gemma4.tt import generator_vllm
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


(
    LANES_warmup_lane_prefill_group,
    LANE_SLOTS_warmup_lane_prefill_group,
    CHUNK_warmup_lane_prefill_group,
    BLOCKS_warmup_lane_prefill_group,
) = (4, 32, 4096, 32)


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
        mesh_config=SimpleNamespace(lane_sharded=lane_sharded, lanes=LANES_warmup_lane_prefill_group),
        lane_slots=LANE_SLOTS_warmup_lane_prefill_group,
        mesh_device=SimpleNamespace(shape=(8, 4)),
    )
    inst.model = [model]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK_warmup_lane_prefill_group)]
    inst._mock_tokens = lambda b, s, kv, m: {
        "tokens": torch.zeros(b, s, dtype=torch.long),
        "page_table": torch.arange(b * BLOCKS_warmup_lane_prefill_group, dtype=torch.int32).reshape(
            b, BLOCKS_warmup_lane_prefill_group
        ),
    }

    def _lanes(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        model._active_page_tables_per_layer = ["set by the lane path"]
        calls["lanes"].append((tuple(tokens.shape), tuple(page_tables.shape), list(prompt_lens), list(slot_ids)))
        return torch.zeros(tokens.shape[0] * (tokens.shape[1] if tokens.dim() == 3 else 1), 8)

    inst.prefill_forward_lanes = _lanes
    return inst, model, calls


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_lane_group_and_vision_warm_once_across_repeated_calls(monkeypatch):
    inst, model, calls = _instance(monkeypatch, lane_router=True)
    kv = [object()]
    for _ in range(3):  # plugin phase 1, phase 2, then every serving prefill batch
        Gemma4ForCausalLM.warmup_model_prefill(inst, kv, enable_trace=True, can_sample_on_device=False)
    assert calls["sweep"] == 3, "the prefill sweep keeps its own once-only gate"
    assert calls["lanes"] == [
        (
            (LANES_warmup_lane_prefill_group, CHUNK_warmup_lane_prefill_group),
            (LANES_warmup_lane_prefill_group, 1, BLOCKS_warmup_lane_prefill_group),
            [CHUNK_warmup_lane_prefill_group] * LANES_warmup_lane_prefill_group,
            [0, 32, 64, 96],
        )
    ]
    assert calls["vision"] == 1
    assert not hasattr(model, "_active_page_tables_per_layer"), "the warmup leaves no per-layer stash behind"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_lane_batched_warmup_adds_the_k_users_per_lane_frame(monkeypatch):
    """k prompts per lane: the batched walk's programs and its page-table frame (the lanes' decode frame) must
    exist before the decode traces are captured, so the warmup adds one lane-batched call."""
    inst, model, calls = _instance(monkeypatch, lane_router=True, multimodal=False, prompts=8)
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["lanes"] == [
        (
            (LANES_warmup_lane_prefill_group, CHUNK_warmup_lane_prefill_group),
            (LANES_warmup_lane_prefill_group, 1, BLOCKS_warmup_lane_prefill_group),
            [CHUNK_warmup_lane_prefill_group] * LANES_warmup_lane_prefill_group,
            [0, 32, 64, 96],
        ),
        (
            (LANES_warmup_lane_prefill_group, 8, CHUNK_warmup_lane_prefill_group // 8),
            (LANES_warmup_lane_prefill_group, 8, BLOCKS_warmup_lane_prefill_group),
            [CHUNK_warmup_lane_prefill_group // 8] * (LANES_warmup_lane_prefill_group * 8),
            [ln * 32 + r for ln in range(LANES_warmup_lane_prefill_group) for r in range(8)],
        ),
    ]
    assert not hasattr(model, "_active_page_tables_per_layer")
    inst, _model, calls = _instance(monkeypatch, lane_router=True, multimodal=False, prompts=8)
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "0")
    Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert len(calls["lanes"]) == 1, "without the batched walk there is no frame to warm"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
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


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_guard_interval_zero_keeps_the_hard_limit(monkeypatch):
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MAX", "8")
    monkeypatch.setenv("GEMMA4_TRACE_ALLOC_GUARD_MIN_INTERVAL_S", "0")
    g = _guard(monkeypatch, recaptures=8, last_recapture_age_s=1e6)
    assert g._g4_trace_guard_should_recapture("k") is False
