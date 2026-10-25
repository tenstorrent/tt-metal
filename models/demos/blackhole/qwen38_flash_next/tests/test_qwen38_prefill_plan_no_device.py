# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The cost prefill plan without a device: the plan rule (covers, costs, alignment, the context end, the legacy plan
unchanged), the ``QWEN38_PREFILL_PLAN`` knob, and the driver's sequence with a padded 128-row chunk or slab as the last
pass (the pad inputs written before it, the pad rows, the hand-off's ``tail``), recorded through fakes of the model and
ttnn.  The padded passes' device numerics are ``test_prefill_padded_pass_device.py`` and the GDN pad-input images below."""

from __future__ import annotations

import functools
from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    PASS_COST_MS,
    PLAN_COST,
    PLAN_LEGACY,
    PassSpec,
    Qwen38ChunkPrefill,
    legacy_cover,
    plan_cover,
    prefill_plan,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import PADDED_PASS_MIN_ROWS

CONTEXT = 1 << 20
SLAB = 2048


def _price(spec: PassSpec, slab_rows: int = SLAB) -> float:
    return PASS_COST_MS[spec.kind] * (slab_rows / 2048 if spec.kind == "slab" else 1.0)


def _cost(specs, slab_rows: int = SLAB) -> float:
    return sum(_price(spec, slab_rows) for spec in specs)


def _brute_force_cost(count: int, position: int, context: int, slab_rows: int, long_chunks: bool) -> float:
    """The cheapest cover by exhaustive recursion over the pass choices, written apart from the planner's table."""

    forms = [("short", 32)] + ([("long", 128)] if long_chunks else []) + ([("slab", slab_rows)] if slab_rows else [])

    @functools.lru_cache(maxsize=None)
    def best(done: int) -> float:
        if done == count:
            return 0.0
        start, left, found = position + done, count - done, float("inf")
        for kind, rows in forms:
            if start + rows > context or (kind != "short" and start % 128):
                continue
            real = min(rows, left)
            if real < rows and real < (1 if kind == "short" else PADDED_PASS_MIN_ROWS):
                continue
            price = PASS_COST_MS[kind] * (rows / 2048 if kind == "slab" else 1.0)
            found = min(found, price + best(done + real))
        return found

    return best(0)


# --------------------------------------------------------------------------- the knob


@pytest.mark.parametrize(
    "environ, expected",
    (
        ({}, PLAN_COST),
        ({"QWEN38_PREFILL_PLAN": ""}, PLAN_COST),
        ({"QWEN38_PREFILL_PLAN": "cost"}, PLAN_COST),
        ({"QWEN38_PREFILL_PLAN": " COST "}, PLAN_COST),
        ({"QWEN38_PREFILL_PLAN": "legacy"}, PLAN_LEGACY),
        ({"QWEN38_PREFILL_PLAN": " Legacy\n"}, PLAN_LEGACY),
    ),
)
def test_the_knob_defaults_to_cost_and_reads_legacy(environ, expected: str) -> None:
    assert prefill_plan(environ) == expected


@pytest.mark.parametrize("value", ("fast", "0", "1", "costs", "cost,legacy", "legacy1", "off"))
def test_the_knob_refuses_an_unknown_plan_by_name(expect_error, value: str) -> None:
    with expect_error(ValueError, "QWEN38_PREFILL_PLAN must be one of"):
        prefill_plan({"QWEN38_PREFILL_PLAN": value})


def test_the_knob_is_read_from_the_process_environment_by_the_driver(monkeypatch, expect_error) -> None:
    monkeypatch.delenv("QWEN38_PREFILL_PLAN", raising=False)
    model, forced = SimpleNamespace(allocated_context=4096), lambda token, context: context
    assert Qwen38ChunkPrefill(model, "mesh", "s", "c", None, forced_step=forced).plan == PLAN_COST
    monkeypatch.setenv("QWEN38_PREFILL_PLAN", "legacy")
    assert Qwen38ChunkPrefill(model, "mesh", "s", "c", None, forced_step=forced).plan == PLAN_LEGACY
    assert Qwen38ChunkPrefill(model, "mesh", "s", "c", None, forced_step=forced, plan="cost").plan == PLAN_COST
    monkeypatch.setenv("QWEN38_PREFILL_PLAN", "bogus")
    with expect_error(ValueError, "QWEN38_PREFILL_PLAN"):
        Qwen38ChunkPrefill(model, "mesh", "s", "c", None, forced_step=forced)
    with expect_error(ValueError, "prefill plan must be one of"):
        Qwen38ChunkPrefill(model, "mesh", "s", "c", None, forced_step=forced, plan="bogus")


# --------------------------------------------------------------------------- the plan rule


@pytest.mark.parametrize(
    "count, expected",
    (
        (0, []),
        (1, [("short", 1)]),
        (31, [("short", 31)]),
        (32, [("short", 32)]),
        (33, [("short", 32), ("short", 1)]),
        (64, [("short", 32), ("short", 32)]),
        (96, [("long", 96)]),  # 156 ms padded beats 3 x 62
        (100, [("long", 100)]),  # was 3 full chunks and a padded 4-row one: 4 x 62
        (128, [("long", 128)]),
        (129, [("long", 128), ("short", 1)]),
        (1023, [("slab", 1023)]),  # was 7 long chunks and 3 + 1 short ones
        (1024, [("slab", 1024)]),
        (2047, [("slab", 2047)]),  # was 15 long chunks and 4 short ones: 2.6 s against one padded slab
        (2048, [("slab", 2048)]),
        (2049, [("slab", 2048), ("short", 1)]),
        (4095, [("slab", 2048), ("slab", 2047)]),
    ),
)
def test_covers_of_the_default_forms(count: int, expected) -> None:
    cover = plan_cover(count, position=0, context=CONTEXT, slab_rows=SLAB, long_chunks=True)
    assert [(spec.kind, spec.real) for spec in cover] == expected
    assert all(spec.rows == {"short": 32, "long": 128, "slab": SLAB}[spec.kind] for spec in cover)


@pytest.mark.parametrize("slab_rows, long_chunks", ((0, False), (0, True), (256, True), (SLAB, True)))
@pytest.mark.parametrize("position", (0, 32, 96, 128, 4000))
def test_the_cover_is_the_cheapest_and_keeps_its_invariants(position: int, slab_rows: int, long_chunks: bool) -> None:
    counts = list(range(0, 700, 7)) + [127, 128, 129, 255, 256, 257, 511, 512, 513, 1000, 1500]
    for count in counts:
        cover = plan_cover(count, position=position, context=CONTEXT, slab_rows=slab_rows, long_chunks=long_chunks)
        assert sum(spec.real for spec in cover) == count
        padded = [index for index, spec in enumerate(cover) if spec.padded]
        assert padded in ([], [len(cover) - 1])  # at most one padded pass, the last
        at = position
        for spec in cover:
            assert 1 <= spec.real <= spec.rows
            if spec.kind != "short":
                assert at % 128 == 0, (count, position, cover)  # a 128-row chunk or slab starts on a 128 boundary
                assert (spec.real == spec.rows) or spec.real >= PADDED_PASS_MIN_ROWS
            at += spec.real
        assert _cost(cover, slab_rows or SLAB) == pytest.approx(
            _brute_force_cost(count, position, CONTEXT, slab_rows, long_chunks)
        ), (count, position)
        if position % 128 == 0:  # the legacy plan's own start condition: the cost plan is never dearer
            legacy = legacy_cover(count, slab_rows=slab_rows, long_chunks=long_chunks)
            assert _cost(cover, slab_rows or SLAB) <= _cost(legacy, slab_rows or SLAB) + 1e-9


def test_the_remainders_the_issue_names_cost_what_it_says() -> None:
    """Head 2047: one padded slab (648 ms) for 15 x 128 + 4 x 32 (2.6 s); head 1023: 648 ms for 7 x 128 + 4 x 32 (1.34 s);
    head 128 (and 100): one 128-row chunk, 156 ms, for 4 x 32 (248 ms)."""

    cost = lambda count: _cost(plan_cover(count, position=0, context=CONTEXT, slab_rows=SLAB, long_chunks=True))
    legacy = lambda count: _cost(legacy_cover(count, slab_rows=SLAB, long_chunks=True))
    assert (cost(2047), legacy(2047)) == (648.0, 15 * 156.0 + 4 * 62.0)
    assert (cost(1023), legacy(1023)) == (648.0, 7 * 156.0 + 4 * 62.0)
    assert (cost(128), legacy(128)) == (156.0, 156.0)
    assert (cost(100), legacy(100)) == (156.0, 4 * 62.0)


def test_a_long_chunk_or_slab_starts_on_a_128_boundary_so_32_row_chunks_run_up_to_it() -> None:
    cover = plan_cover(300, position=32, context=CONTEXT, slab_rows=SLAB, long_chunks=True)
    assert [(spec.kind, spec.real) for spec in cover] == [("short", 32)] * 3 + [("long", 128), ("long", 76)]
    assert plan_cover(300, position=64, context=CONTEXT, slab_rows=0, long_chunks=True)[0] == PassSpec("short", 32, 32)
    # the legacy plan has no such rule (its long chunks start wherever the alignment steps left the position)
    assert legacy_cover(300, long_chunks=True)[0].kind == "long"


def test_the_cover_never_runs_past_the_context_and_falls_back_to_a_cheaper_fitting_one(expect_error) -> None:
    wide = plan_cover(1000, position=0, context=CONTEXT, slab_rows=SLAB, long_chunks=True)
    assert [spec.kind for spec in wide] == ["slab"]  # the padded slab ends at 2048
    tight = plan_cover(1000, position=0, context=1100, slab_rows=SLAB, long_chunks=True)
    assert tight != wide and all(spec.kind != "slab" for spec in tight)
    end = 0
    for spec in tight:
        end += spec.real
        assert end - spec.real + spec.rows <= 1100  # every pass, padded or not, ends inside the context
    assert sum(spec.real for spec in tight) == 1000
    # near the very end only 32-row chunks fit: 1000 = 31 x 32 + 8 inside a context of exactly 1024
    last = plan_cover(1000, position=0, context=1024, slab_rows=SLAB, long_chunks=True)
    assert max(spec.rows for spec in last) <= 128 and sum(spec.real for spec in last) == 1000
    assert plan_cover(1000, position=0, context=1024, slab_rows=SLAB, long_chunks=True) == last
    assert plan_cover(0, position=40, context=10, slab_rows=SLAB, long_chunks=True) == []
    with expect_error(ValueError, "fits the allocated context"):
        plan_cover(40, position=0, context=33)  # no cover of 40 rows ends inside 33 rows
    with expect_error(ValueError, "32-aligned"):
        plan_cover(40, position=5, context=CONTEXT)


def test_a_cost_table_steers_the_cover_and_the_slab_cost_scales_with_its_rows() -> None:
    free_short = {"slab": 648.0, "long": 156.0, "short": 1.0}
    cover = plan_cover(100, position=0, context=CONTEXT, long_chunks=True, costs=free_short)
    assert [(spec.kind, spec.real) for spec in cover] == [("short", 32)] * 3 + [("short", 4)]
    half_slab = plan_cover(1000, position=0, context=CONTEXT, slab_rows=1024, long_chunks=True)
    assert [(spec.kind, spec.real) for spec in half_slab] == [("slab", 1000)]  # 324 ms: half of the 2048-row slab's
    no_forms = plan_cover(300, position=0, context=CONTEXT)  # 32-row chunks only: the legacy plan
    assert no_forms == legacy_cover(300)


@pytest.mark.parametrize("count", (0, 1, 31, 32, 33, 127, 128, 129, 2047, 2048, 2049, 4097, 30000))
def test_the_legacy_cover_is_the_old_plan(count: int) -> None:
    cover = legacy_cover(count, slab_rows=SLAB, long_chunks=True)
    slabs, rest = divmod(count, SLAB)
    longs, rest = divmod(rest, 128)
    shorts = driver_module.chunk_accepts(rest)
    assert [spec.kind for spec in cover] == ["slab"] * slabs + ["long"] * longs + ["short"] * len(shorts)
    assert [spec.real - 1 for spec in cover if spec.kind == "short"] == shorts
    assert all(spec.real == spec.rows for spec in cover if spec.kind != "short")
    assert sum(spec.real for spec in cover) == count
    assert legacy_cover(count, long_chunks=True) == [PassSpec("long", 128, 128)] * (count // 128) + legacy_cover(
        count % 128
    )
    assert legacy_cover(count) == [PassSpec("short", 32, 32)] * (count // 32) + (
        [PassSpec("short", 32, count % 32)] if count % 32 else []
    )


# --------------------------------------------------------------------------- the GDN pad inputs


def test_the_gdn_pad_input_images_are_the_identity_for_a_full_pass_and_mask_the_pad_rows() -> None:
    for rows in (128, 256, 2048):
        full = gdn_module.pad_input_images(rows)
        assert torch.equal(full["row_mask"], torch.ones(1, 1, rows, 1))
        assert torch.equal(full["row_mask_c"], torch.ones(12, rows // 32, 32, 1))
        if rows == 128:  # the 128-row form's select is the constant history_select_full of a full pass
            reference = gdn_module.rows_window_select_tiles(rows)["history_select_full"]
            assert torch.equal(full["history_select"].reshape(32, 32 + rows), reference)
        else:  # the slab's picks the last three rows of the q|k|v rows
            select = full["history_select"].reshape(32, rows)
            assert select[:3].nonzero()[:, 1].tolist() == [rows - 3, rows - 2, rows - 1]
            assert float(select[3:].abs().sum()) == 0.0
        assert torch.equal(gdn_module.pad_input_images(rows, rows)["history_select"], full["history_select"])
    for rows, real in ((128, 100), (128, 9), (2048, 1000), (2048, 2047), (256, 33)):
        images = gdn_module.pad_input_images(rows, real)
        mask = images["row_mask"].reshape(rows)
        assert float(mask[:real].min()) == 1.0 and float(mask[real:].max()) == 0.0
        assert images["row_mask_c"].shape == (12, rows // 32, 32, 1)
        assert torch.equal(images["row_mask_c"][3].reshape(rows), mask)  # the same mask in the prims' page layout
        select = images["history_select"].reshape(32, -1)
        picked = select[:3].nonzero()
        assert picked[:, 0].tolist() == [0, 1, 2] and float(select[3:].abs().sum()) == 0.0
        # the three real rows before the first pad row: the 128-row form's window holds the history tile first, the
        # slab's columns are the q|k|v rows themselves
        expected = (
            [gdn_module._window_buffer_row(real + index) for index in range(3)]
            if rows == 128
            else [real - 3, real - 2, real - 1]
        )
        assert picked[:, 1].tolist() == expected


@pytest.mark.parametrize("real", (0, 1, 2, 129, 130))
def test_the_gdn_pad_input_images_refuse_a_pass_with_too_few_or_too_many_real_rows(expect_error, real: int) -> None:
    with expect_error(ValueError, "real rows"):
        gdn_module.pad_input_images(128, real)


# --------------------------------------------------------------------------- the driver with a padded last pass

ALLOCATED_CONTEXT = 8192
TRACE, LONG_TRACE, SLAB_TRACE = 77, 128, 2048


class _SlabState(str):
    """The slab chunk state the fake model keys its tables by name; the driver reads its ``rows``."""

    rows = SLAB


class _Model:
    allocated_context = ALLOCATED_CONTEXT

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_chunk_state_inplace(self, state, chunk_state) -> None:
        self.calls.append(("reset", chunk_state))

    def write_chunk_accepted(self, chunk_state, accepted: int) -> None:
        self.calls.append(("accepted", accepted))

    def write_chunk_pad(self, chunk_state, real_rows) -> None:
        self.calls.append(("pad", chunk_state, real_rows))

    def prepare_chunk_inputs(self, chunk_state, token_ids, *, ple_context, positions=None, features=None):
        tokens = list(token_ids)
        assert len(tokens) == {"short_state": 32, "long_state": 128, "slab_state": SLAB}[chunk_state]
        contexts = [ple_context]
        for token in tokens:
            contexts.append((2 if contexts[-1] is None else contexts[-1][1], token))
        self.calls.append(("inputs", chunk_state, tuple(tokens), ple_context, positions))
        return SimpleNamespace(chunk_state=chunk_state, contexts=tuple(contexts))

    def upload_chunk_inputs(self, chunk_state, prepared) -> None:
        self.calls.append(("upload", chunk_state))

    def finish_prefill(self, state, chunk_state, prefilled: int, *, rope_shift: int = 0, tail=None) -> None:
        self.calls.append(("finish", chunk_state, prefilled, tail))


@pytest.fixture
def harness(monkeypatch):
    model = _Model()
    log = model.calls
    fake_ttnn = SimpleNamespace(
        _ttnn_execute_trace=lambda mesh, trace_id, *, cq_id, blocking: log.append(("replay", trace_id)),
        record_event=lambda mesh, cq_id: "event",
        event_synchronize=lambda event: log.append(("event",)),
        synchronize_device=lambda mesh: log.append(("sync",)),
    )
    monkeypatch.setattr(driver_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(
        driver_module,
        "TraceAllocationTracker",
        SimpleNamespace(verify_before_replay=lambda mesh, trace_id: log.append(("verify", trace_id))),
    )

    def make(plan: str | None = "cost", *, slab: bool = True, **kwargs):
        return Qwen38ChunkPrefill(
            model,
            "mesh",
            "state",
            "short_state",
            TRACE,
            forced_step=lambda token, context: (2 if context is None else context[1], token),
            long_chunk_state="long_state",
            long_chunk_trace_id=LONG_TRACE,
            **({"slab_state": _SlabState("slab_state"), "slab_trace_id": SLAB_TRACE} if slab else {}),
            plan=plan,
            **kwargs,
        )

    return SimpleNamespace(model=model, log=log, make=make)


def _tokens(count: int) -> list[int]:
    return [1000 + index for index in range(count)]


@pytest.mark.parametrize(
    "count, real, kind", ((100, 100, "long"), (97, 97, "long"), (2047, 2047, "slab"), (1000, 1000, "slab"))
)
def test_a_padded_last_pass_writes_its_pad_inputs_pads_its_rows_and_hands_over_from_its_state(
    harness, count, real, kind
):
    driver = harness.make()
    result = driver.run(_tokens(count), start_position=0, ple_context=None)
    log = harness.log
    state = {"long": "long_state", "slab": "slab_state"}[kind]
    form_rows = {"long": 128, "slab": SLAB}[kind]
    assert result.position == count and result.timing.plan == "cost"
    assert (result.timing.padded_pass, result.timing.padded_real_rows) == (kind, real)
    assert (result.timing.long_chunks, result.timing.slabs, result.timing.chunks) == (
        int(kind == "long"),
        int(kind == "slab"),
        0,
    )
    assert [entry for entry in log if entry[0] == "accepted"] == []  # the 32-row accept scalar is not written
    assert [entry for entry in log if entry[0] == "pad"] == [("pad", state, real)]
    inputs = [entry for entry in log if entry[0] == "inputs"]
    assert len(inputs) == 1 and inputs[0][1] == state and len(inputs[0][2]) == form_rows
    assert list(inputs[0][2][:real]) == _tokens(count) and set(inputs[0][2][real:]) <= {
        driver_module.CHUNK_PAD_TOKEN_ID
    }
    # the pad inputs are written before the pass's inputs upload and replay; the hand-off carries the tail
    names = [entry[0] for entry in log]
    assert names.index("pad") < names.index("upload") < names.index("replay")
    assert log[-3:] == [("sync",), ("finish", "short_state", count, (state, real)), ("sync",)]
    assert [entry[1] for entry in log if entry[0] == "verify"] == [TRACE, LONG_TRACE] + (
        [SLAB_TRACE] if kind == "slab" else []
    )
    assert [entry[1] for entry in log if entry[0] == "reset"] == ["short_state", "long_state"] + (
        ["slab_state"] if kind == "slab" else []
    )


def test_the_pad_rows_never_enter_the_committed_ple_context(harness) -> None:
    driver = harness.make()
    tokens = _tokens(100)
    result = driver.run(tokens, start_position=0, ple_context=None)
    context = None
    for token in tokens:
        context = (2 if context is None else context[1], token)
    assert result.ple_context == context


def test_a_full_last_pass_has_no_tail_and_no_pad_write(harness) -> None:
    driver = harness.make()
    result = driver.run(_tokens(128), start_position=0, ple_context=None)
    assert (result.timing.padded_pass, result.timing.padded_real_rows, result.timing.long_chunks) == (None, 0, 1)
    assert not [entry for entry in harness.log if entry[0] in ("pad", "accepted")]
    assert harness.log[-2] == ("finish", "short_state", 128, None)


def test_a_padded_32_row_tail_after_full_passes_keeps_its_accept_scalar(harness) -> None:
    driver = harness.make()
    result = driver.run(_tokens(2049), start_position=0, ple_context=None)  # a slab and a 1-row padded chunk
    assert (result.timing.slabs, result.timing.chunks, result.timing.padded_pass, result.timing.padded_real_rows) == (
        1,
        1,
        "short",
        1,
    )
    assert [entry for entry in harness.log if entry[0] == "accepted"] == [("accepted", 0)]
    assert not [entry for entry in harness.log if entry[0] == "pad"]
    assert harness.log[-2][3] is None  # the 32-row state hands over as it always did


def test_the_legacy_plan_and_a_chain_with_mtp_extensions_plan_as_before(harness) -> None:
    legacy = harness.make("legacy").run(_tokens(100), start_position=0, ple_context=None)
    assert legacy.timing.plan == "legacy" and (legacy.timing.long_chunks, legacy.timing.chunks) == (0, 4)
    assert legacy.timing.padded_pass == "short" and legacy.timing.padded_real_rows == 4
    assert not [entry for entry in harness.log if entry[0] == "pad"]
    # an MTP chain keeps the legacy plan whatever the knob says: its extensions have no padded 128-row or slab form
    harness.log.clear()

    class Extension:
        rows = 32
        slab_rows = 0

        def reset_chunk(self):
            pass

        def write_tokens(self, model, tokens, features=None):
            pass

        def write_slab_tokens(self, model, tokens, features=None):
            pass

        def finish_chunk(self, model, prefilled):
            pass

    long_extension = Extension()
    long_extension.rows, long_extension.slab_rows = 128, SLAB
    mtp_driver = harness.make("cost", mtp=Extension(), long_mtp=long_extension)
    result = mtp_driver.run(_tokens(100), start_position=0, ple_context=None, following_token=5)
    assert result.timing.plan == "legacy (mtp)" and (result.timing.long_chunks, result.timing.chunks) == (0, 4)
    assert not [entry for entry in harness.log if entry[0] == "pad"]


def test_a_stop_before_the_padded_pass_hands_over_from_the_32_row_state_without_a_tail(harness) -> None:
    driver = harness.make(event_rows=128)
    result = driver.run(_tokens(2048 + 100), start_position=0, ple_context=None, should_stop=lambda: "stop")
    assert result.stopped == "stop" and result.position == 2048
    assert harness.log[-2][3] is None  # the slab was the last pass replayed: a full one
    assert not [entry for entry in harness.log if entry[0] == "pad"]


def test_a_cover_that_does_not_fit_the_context_is_refused_and_a_tight_context_falls_back(harness, expect_error) -> None:
    harness.model.allocated_context = 1100
    tight = harness.make().run(_tokens(1000), start_position=0, ple_context=None)
    assert tight.timing.slabs == 0 and tight.position == 1000  # the padded slab would end at 2048
    harness.model.allocated_context = 1024  # exactly the 32-row grid of 1000 rows: the last pass ends at 1024
    assert harness.make().run(_tokens(1000), start_position=0, ple_context=None).position == 1000
    harness.model.allocated_context = (
        1000  # the padded last pass would end at 1024: no cover fits (the legacy plan raises too)
    )
    with expect_error(ValueError, "fits the allocated context"):
        harness.make().run(_tokens(1000), start_position=0, ple_context=None)


# --------------------------------------------------------------------------- the model's pad writes and the hand-off dispatch

from models.demos.blackhole.qwen38_flash_next.ttnn import layer as layer_module  # noqa: E402
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module  # noqa: E402
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module  # noqa: E402
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module  # noqa: E402


class _Tensor:
    def __init__(self, name: str) -> None:
        self.name = name


def _pad_chunk_state(rows: int, *, slab: bool):
    pad = SimpleNamespace(
        row_mask=_Tensor("row_mask"),
        history_select=_Tensor("history_select"),
        row_mask_c=_Tensor("row_mask_c") if slab else None,
    )
    ple = SimpleNamespace(
        snapshot=_Tensor("snapshot") if slab else None,
        snapshot_keep=tuple(_Tensor(f"keep{b}") for b in range(rows // 128)) if slab else (),
        snapshot_take=tuple(_Tensor(f"take{b}") for b in range(rows // 128)) if slab else (),
    )
    layers = [SimpleNamespace(ple=None) for _ in range(model_module.PLE_CHECKPOINT_LAYER + 1)]
    layers[model_module.PLE_CHECKPOINT_LAYER] = SimpleNamespace(ple=ple)
    return SimpleNamespace(
        rows=rows,
        rows_constants=SimpleNamespace(pad=pad),
        qsa_chunk_constants=SimpleNamespace(pad_rows_col=_Tensor("qsa_pad") if slab else None),
        layers=tuple(layers),
    )


@pytest.fixture
def pad_writer(monkeypatch):
    """A text model whose ``write_chunk_pad`` runs over fake tensors: the writes land in a list by target name."""

    writes: dict[str, torch.Tensor] = {}
    fake_ttnn = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        uint32="u32",
        TILE_LAYOUT="tile",
        ROW_MAJOR_LAYOUT="rm",
        from_torch=lambda host, dtype, layout, mesh_mapper: SimpleNamespace(host=host, dtype=dtype, layout=layout),
        copy_host_to_device_tensor=lambda tensor, target: writes.__setitem__(target.name, tensor),
    )
    monkeypatch.setattr(model_module, "ttnn", fake_ttnn)
    monkeypatch.setattr(model_module, "replicate_tensor_2d_mesh_mapper", lambda mesh: "mapper")
    model = model_module.Qwen38TTNNTextModel.__new__(model_module.Qwen38TTNNTextModel)
    model.mesh_device = "mesh"
    model._validate_chunk_state = lambda chunk_state: None
    model._chunk_pad_full = set()
    return SimpleNamespace(model=model, writes=writes)


def test_write_chunk_pad_writes_the_pad_inputs_of_a_128_row_chunk(pad_writer) -> None:
    state = _pad_chunk_state(128, slab=False)
    pad_writer.model.write_chunk_pad(state, 100)
    images = gdn_module.pad_input_images(128, 100)
    assert set(pad_writer.writes) == {"row_mask", "history_select"}  # no slab-only inputs on the 128-row form
    assert torch.equal(pad_writer.writes["row_mask"].host, images["row_mask"])
    assert torch.equal(pad_writer.writes["history_select"].host, images["history_select"])
    assert pad_writer.writes["history_select"].dtype == "bf16" and pad_writer.writes["row_mask"].dtype == "fp32"


@pytest.mark.parametrize("real", (1000, 2047, None))
def test_write_chunk_pad_writes_the_slab_inputs_the_row_masks_and_the_ple_block_flags(pad_writer, real) -> None:
    state = _pad_chunk_state(2048, slab=True)
    pad_writer.model.write_chunk_pad(state, real)
    images = gdn_module.pad_input_images(2048, real)
    writes = pad_writer.writes
    assert torch.equal(writes["row_mask_c"].host, images["row_mask_c"])
    assert torch.equal(writes["history_select"].host, images["history_select"])
    keep = writes["qsa_pad"].host.reshape(2048).to(torch.int64)
    assert writes["qsa_pad"].host.dtype == torch.uint32 and writes["qsa_pad"].dtype == "u32"
    assert writes["qsa_pad"].layout == "rm"
    assert keep.tolist() == [1] * (2048 if real is None else real) + [0] * (0 if real is None else 2048 - real)
    # the PLE snapshot: only the block that holds the last real row takes the window (keep 0 / take 1 there)
    block = 15 if real is None else (real - 1) // 128
    assert [float(writes[f"take{b}"].host) for b in range(16)] == [1.0 if b == block else 0.0 for b in range(16)]
    assert [float(writes[f"keep{b}"].host) for b in range(16)] == [0.0 if b == block else 1.0 for b in range(16)]


def test_a_full_pass_write_is_skipped_while_the_inputs_hold_one(pad_writer) -> None:
    """P3c: ``reset_chunk_state_inplace`` writes the full pass at every request; the second and later writes of the same
    state are skipped until a partial write (or the release) drops the flag, and a failed write leaves it unknown."""

    model, writes = pad_writer.model, pad_writer.writes
    state = _pad_chunk_state(2048, slab=True)
    model.write_chunk_pad(state, None)
    assert id(state) in model._chunk_pad_full and writes
    writes.clear()
    model.write_chunk_pad(state, None)  # the inputs already hold the full pass
    assert not writes
    model.write_chunk_pad(state, 1000)  # a padded pass: written, and the flag drops
    assert writes and id(state) not in model._chunk_pad_full
    writes.clear()
    model.write_chunk_pad(state, None)  # back to the full pass: written again
    assert writes and id(state) in model._chunk_pad_full
    writes.clear()
    model.write_chunk_pad(state, 2048)  # every row real, spelled out: conservative, written and not flagged
    assert writes and id(state) not in model._chunk_pad_full
    other = _pad_chunk_state(128, slab=False)  # a flag is per state
    model.write_chunk_pad(state, None)
    writes.clear()
    model.write_chunk_pad(other, None)
    assert writes


def test_a_failed_pad_write_leaves_the_full_pass_unknown(pad_writer, expect_error) -> None:
    model = pad_writer.model
    state = _pad_chunk_state(128, slab=False)
    model.write_chunk_pad(state, None)
    assert id(state) in model._chunk_pad_full
    pad_writer.writes.clear()
    original = model_module.ttnn.copy_host_to_device_tensor

    def failing(tensor, target):
        raise RuntimeError("device write failed")

    model_module.ttnn.copy_host_to_device_tensor = failing
    try:
        with expect_error(RuntimeError, "device write failed"):
            model.write_chunk_pad(state, 100)
    finally:
        model_module.ttnn.copy_host_to_device_tensor = original
    assert id(state) not in model._chunk_pad_full
    model.write_chunk_pad(state, None)  # unknown: rewritten
    assert pad_writer.writes


@pytest.mark.parametrize("real", (0, 8, 129, 2.5, True))
def test_write_chunk_pad_refuses_too_few_or_too_many_real_rows(pad_writer, expect_error, real) -> None:
    with expect_error(ValueError):
        pad_writer.model.write_chunk_pad(_pad_chunk_state(128, slab=False), real)
    assert not pad_writer.writes


def test_write_chunk_pad_refuses_the_32_row_state(pad_writer, expect_error) -> None:
    state = _pad_chunk_state(128, slab=False)
    state.rows, state.rows_constants = 32, SimpleNamespace(pad=None)
    with expect_error(ValueError, "no pad inputs"):
        pad_writer.model.write_chunk_pad(state, 20)


def _layer(attention, ple, calls):
    layer = layer_module.Qwen38TTNNDecoderLayer.__new__(layer_module.Qwen38TTNNDecoderLayer)
    layer.attention, layer.ple = attention, ple
    layer._validate_chunk_state = lambda state: None
    return layer


@pytest.mark.parametrize("real, copies", ((100, True), (96, False), (127, True), (33, True)))
def test_a_qsa_layer_hands_over_the_kept_tile_of_its_last_real_row_unless_the_rows_end_on_a_tile(real, copies) -> None:
    calls = []
    attention = qsa_module.Qwen38TTNNQSA.__new__(qsa_module.Qwen38TTNNQSA)
    attention.copy_kept_tile = lambda source, destination, *, tile: calls.append((source, destination, tile))
    layer = _layer(attention, None, calls)
    state, tail = SimpleNamespace(rows=32, attention="kept32"), SimpleNamespace(rows=128, attention="kept128")
    layer.hand_off_padded_tail(state, tail, real_rows=real)
    assert calls == ([("kept128", "kept32", (real - 1) // 32)] if copies else [])


def test_the_ple_layer_commits_the_padded_history_and_a_gdn_layer_hands_over_nothing(expect_error) -> None:
    calls = []
    gdn = gdn_module.Qwen38TTNNGDN.__new__(gdn_module.Qwen38TTNNGDN)
    ple = SimpleNamespace(commit_padded_history=lambda rows_state, rows: calls.append((rows_state, rows)))
    layer = _layer(gdn, ple, calls)
    state, tail = SimpleNamespace(rows=32, ple="ple32"), SimpleNamespace(rows=2048, ple="ple2048")
    layer.hand_off_padded_tail(state, tail, real_rows=1000)
    assert calls == [("ple2048", 1000)]
    calls.clear()
    layer.ple = None
    layer.hand_off_padded_tail(state, tail, real_rows=1000)
    assert calls == []
    with expect_error(ValueError, "longer form"):
        layer.hand_off_padded_tail(state, SimpleNamespace(rows=32), real_rows=10)
    with expect_error(ValueError, "real rows"):
        layer.hand_off_padded_tail(state, tail, real_rows=2048)


# --------------------------------------------------------------------------- source pins: where the pad inputs act

import inspect  # noqa: E402

from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_prefill_rows as fused_slab_module  # noqa: E402


def _in_order(source: str, *fragments: str) -> None:
    positions = [source.index(fragment) for fragment in fragments]
    assert positions == sorted(positions), fragments


def test_the_pad_inputs_act_between_the_programs_and_the_commits_in_the_traced_bodies() -> None:
    # the fused slab: mask beta / g after the pre program and before the prims; the history select from the projection
    # before it is released
    body = inspect.getsource(fused_slab_module.slab_body)
    _in_order(
        body, "gdn_pre_rows.run(", "ttnn.multiply(buffers.beta, pad_mask_c", "chunk_prims(", "ttnn.deallocate(beta)"
    )
    prefill = inspect.getsource(fused_slab_module.gdn_prefill_rows)
    _in_order(prefill, "slab_body(", "gdn._slab_pad_history(projected, rows_state)", "ttnn.deallocate(projected)")
    # the chain bodies (128-row chunk, unfused slab): the row mask in _chunk_rows, the selects in commit_rows_full
    rows = inspect.getsource(gdn_module.Qwen38TTNNGDN._chunk_rows)
    _in_order(rows, "constants.pad.row_mask", "ttnn.multiply(rows_state.beta, committed_mask", "_chunk_rows_composite(")
    # the slab's PLE: the block's window into the snapshot after its injection and before its history commit
    slab_ple = inspect.getsource(layer_module.Qwen38TTNNDecoderLayer._inject_ple_slab)
    _in_order(slab_ple, "self.ple.inject_rows(", "self.ple.snapshot_window(", "self.ple.commit_rows_full(")
    # the slab's QSA: the pad rows complete no block, before the column is tilized and the masks derived from it
    derive = inspect.getsource(qsa_module._derive_slab_tile_inputs)
    _in_order(derive, "pad_rows_col", "ttnn.multiply(complete_rows_rm, pad_rows_col", "ttnn.to_layout(complete_rows_rm")


def test_the_hand_off_takes_the_real_rows_carries_before_it_reads_them_and_restores_a_full_pass() -> None:
    finish = inspect.getsource(model_module.Qwen38TTNNTextModel.finish_prefill)
    _in_order(
        finish,
        "layer.hand_off_padded_tail(layer_chunk, tail_chunk, real_rows=tail_rows)",
        "self.write_chunk_pad(tail_state, None)",
        "self._ring_selects()[prefilled % CHUNK_ROWS]",
        "layer.finish_chunk_state_inplace(",
    )
    reset = inspect.getsource(model_module.Qwen38TTNNTextModel.reset_chunk_state_inplace)
    assert "self.write_chunk_pad(chunk_state, None)" in reset  # every seed restores a full pass
    ple = inspect.getsource(ple_module.Qwen38TTNNPLE.commit_padded_history)
    # one exact selection with a fixed-shape program (no slice at a variable offset: a closed program-cache guard)
    assert "ttnn.matmul(" in ple and "ttnn.permute(source" in ple and "(0, first, 0, 0)" not in ple
    kept = inspect.getsource(qsa_module.Qwen38TTNNQSA.copy_kept_tile)
    assert "ttnn.matmul(" in kept and "optional_output_tensor=target" in kept and "ttnn.slice(" not in kept
