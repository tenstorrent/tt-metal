# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the dFlash verify-width preparation and capture order.

A trace records buffer addresses. Every width's persistent buffers are
installed and run once eagerly (``prepare_widths``) before any trace of any
kind is captured, so no ordinary decode trace can hold an address a drafter
buffer later occupies; ``capture_widths`` then records one trace per width on
the same decoder, and no new width is admitted afterwards.
"""

import sys
from types import SimpleNamespace

import pytest

from models.demos.gemma4.tests.unit.conftest import build_model
from models.demos.gemma4.tests.unit.dflash_contract_harness import make_expect_error


@pytest.fixture
def expect_error():
    return make_expect_error()


@pytest.fixture
def decoder(drafter_module, monkeypatch):
    cls = drafter_module.DFlashFusedDecoder
    dec = cls.__new__(cls)
    dec.use_packed = True
    dec._pv_widths = {}
    dec._prepared_widths = set()
    dec.pv_sk = None
    dec.trace = None
    dec._out = None
    dec._replay_on_first = False
    dec.events = []
    dec.target = SimpleNamespace(
        dflash_capture_taps=lambda layers, buffers=None, keep_last=None: dec.events.append(
            ("taps", None if layers is None else tuple(layers))
        )
    )
    dec.drafter = SimpleNamespace(target_layer_ids=[1, 2])
    dec.tap_bufs = "bufs"
    dec.mesh_device = "mesh"

    def install(width):
        dec._pv_widths.setdefault(int(width), {"pv_sk": int(width), "trace": None})
        dec.events.append(("install", int(width)))

    def activate(record):
        dec.pv_sk = record["pv_sk"]
        dec.events.append(("activate", record["pv_sk"]))

    def body():
        dec.events.append(("body", dec.pv_sk))
        return ("ids", "fc")

    monkeypatch.setattr(dec, "_pv_ring_meta", lambda: dec.events.append(("ring_meta",)))
    monkeypatch.setattr(dec, "_pv_width_install", install)
    monkeypatch.setattr(dec, "_seed_ctx_cache", lambda: dec.events.append(("seed",)))
    monkeypatch.setattr(dec, "_pv_width_activate", activate)
    monkeypatch.setattr(dec, "_pv_upload", lambda start: dec.events.append(("pv_upload", int(start))))
    monkeypatch.setattr(
        dec, "_upload_iter_inputs", lambda anchor, start: dec.events.append(("iter_inputs", anchor, start))
    )
    monkeypatch.setattr(dec, "_body", body)
    monkeypatch.setattr(dec, "restore_model_logits_mode", lambda: dec.events.append(("restore",)))
    runtime = sys.modules["ttnn"]
    counter = [0]

    def begin(mesh, cq_id=0):
        counter[0] += 1
        dec.events.append(("begin_trace", counter[0]))
        return counter[0]

    monkeypatch.setattr(runtime, "begin_trace_capture", begin)
    monkeypatch.setattr(runtime, "end_trace_capture", lambda mesh, tid, cq_id=0: dec.events.append(("end_trace", tid)))
    monkeypatch.setattr(runtime, "release_trace", lambda mesh, tid: dec.events.append(("release_trace", tid)))
    monkeypatch.setattr(runtime, "synchronize_device", lambda mesh: None)
    return dec


def _kinds(dec, *kinds):
    return [event for event in dec.events if event[0] in kinds]


def test_prepare_installs_every_width_before_seeding_and_running_any_body(decoder):
    decoder.prepare_widths([4096, 2048])
    kinds = [event[0] for event in decoder.events]
    assert kinds[:4] == ["ring_meta", "install", "install", "seed"]
    assert _kinds(decoder, "install") == [("install", 2048), ("install", 4096)]
    assert _kinds(decoder, "body") == [("body", 2048), ("body", 4096)]
    assert _kinds(decoder, "begin_trace") == []
    assert decoder._prepared_widths == {2048, 4096}
    assert decoder.events[-2:] == [("taps", None), ("restore",)]
    assert all(record["trace"] is None for record in decoder._pv_widths.values())


def test_prepare_is_idempotent(decoder):
    decoder.prepare_widths([2048])
    seen = list(decoder.events)
    decoder.prepare_widths([2048])
    assert decoder.events == seen


def test_capture_prepares_first_and_records_one_trace_per_width(decoder):
    cost = decoder.capture_widths([2048, 4096])
    bodies = _kinds(decoder, "body")
    begins = _kinds(decoder, "begin_trace")
    assert [b[1] for b in bodies] == [2048, 4096, 2048, 4096]
    assert decoder.events.index(begins[0]) > decoder.events.index(bodies[1])
    assert sorted(cost) == [2048, 4096]
    assert decoder._pv_widths[2048]["trace"] == 1 and decoder._pv_widths[4096]["trace"] == 2
    assert decoder.trace == 2
    assert decoder._replay_on_first is True
    assert decoder._out == ("ids", "fc")


def test_captured_widths_are_not_captured_again(decoder):
    decoder.capture_widths([2048])
    decoder.capture_widths([2048])
    assert len(_kinds(decoder, "begin_trace")) == 1


def test_a_new_width_after_capture_is_refused(decoder, expect_error):
    decoder.capture_widths([2048])
    with expect_error(RuntimeError, "after trace capture"):
        decoder.prepare_widths([4096])
    with expect_error(RuntimeError, "after trace capture"):
        decoder.capture_widths([2048, 4096])


def test_a_failed_capture_releases_its_trace_and_leaves_the_width_uncaptured(decoder, monkeypatch, expect_error):
    calls = [0]

    def body():
        calls[0] += 1
        if calls[0] == 2:  # the capture pass, after the prepare pass
            raise RuntimeError("boom")
        return ("ids", "fc")

    monkeypatch.setattr(decoder, "_body", body)
    with expect_error(RuntimeError, "boom"):
        decoder.capture_widths([2048])
    assert _kinds(decoder, "release_trace") == [("release_trace", 1)]
    assert decoder._pv_widths[2048]["trace"] is None
    assert decoder.events[-2:] == [("taps", None), ("restore",)]


def test_unpacked_verification_cannot_use_the_width_set(decoder, expect_error):
    decoder.use_packed = False
    with expect_error(NotImplementedError, "packed-verify"):
        decoder.prepare_widths([2048])


# -- verify count -----------------------------------------------------------------


def test_verify_count_from_the_adapter_overrides_the_environment(drafter_module, monkeypatch):
    resolve = drafter_module.DFlashFusedDecoder._resolve_verify_count
    monkeypatch.delenv("GEMMA4_DFLASH_VERIFY", raising=False)
    assert resolve(15, True, 5) == 5
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", "7")
    assert resolve(15, True, 5) == 5
    assert resolve(15, True, None) == 7
    monkeypatch.delenv("GEMMA4_DFLASH_VERIFY", raising=False)
    assert resolve(15, True, None) == 15
    assert resolve(15, False, None) == 15


@pytest.mark.parametrize("bad", [0, 16, -1])
def test_verify_count_outside_the_drafter_block_is_refused(drafter_module, bad, expect_error):
    with expect_error(ValueError, r"\[1, 15\]"):
        drafter_module.DFlashFusedDecoder._resolve_verify_count(15, True, bad)


def test_verify_count_below_the_block_needs_packed_verification(drafter_module, expect_error):
    with expect_error(ValueError, "packed"):
        drafter_module.DFlashFusedDecoder._resolve_verify_count(15, False, 5)
    assert drafter_module.DFlashFusedDecoder._resolve_verify_count(15, False, 15) == 15


# -- warmup order on the adapter -----------------------------------------------------


class _RecordingDecoder:
    instances = []

    def __init__(self, target, drafter, kv_layers, page_table, ctx_cap=2048, *, verify_count=None):
        self.verify_count = verify_count
        self.page_table = page_table
        self.events = []
        self._pv_widths = {}
        _RecordingDecoder.instances.append(self)

    def prepare_widths(self, widths):
        self.events.append(("prepare", tuple(widths)))
        for width in widths:
            self._pv_widths.setdefault(int(width), {"trace": None})

    def capture_widths(self, widths):
        self.events.append(("capture", tuple(widths)))
        for width in widths:
            self._pv_widths[int(width)]["trace"] = object()
        return {int(width): 0.1 for width in widths}


@pytest.fixture
def warm_model(adapter, monkeypatch):
    _RecordingDecoder.instances = []
    monkeypatch.setattr(
        sys.modules["models.demos.gemma4.tt.dflash_drafter"], "DFlashFusedDecoder", _RecordingDecoder, raising=False
    )
    model = build_model(adapter, monkeypatch)
    model._spec_decoder = None
    model._spec_width_ladder = None
    model.model_args[0].max_seq_len = 4096
    monkeypatch.setattr(adapter.Gemma4ForCausalLM, "warmup_model_decode", lambda self, *a, **k: "warm", raising=False)
    return model


def test_eager_warmup_prepares_and_traced_warmup_captures_on_the_same_decoder(warm_model):
    model = warm_model
    assert model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64) == "warm"
    assert len(_RecordingDecoder.instances) == 1
    dec = _RecordingDecoder.instances[0]
    assert dec.verify_count == 5
    assert [e[0] for e in dec.events] == ["prepare"]
    assert model._spec_decoder is dec
    assert model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64) == "warm"
    assert len(_RecordingDecoder.instances) == 1
    assert [e[0] for e in dec.events] == ["prepare", "capture"]
    assert dec.events[0][1] == dec.events[1][1] == tuple(model._spec_width_ladder)


def test_a_changed_ladder_between_preparation_and_capture_is_refused(warm_model, expect_error):
    model = warm_model
    model.warmup_model_decode(enable_trace=False, kv_cache=model.kv_cache, num_blocks=64)
    model._spec_width_ladder = [1234]
    with expect_error(RuntimeError, "widths changed"):
        model.warmup_model_decode(enable_trace=True, kv_cache=model.kv_cache, num_blocks=64)


def test_the_block_rail_passes_no_verify_count(adapter, monkeypatch):
    _RecordingDecoder.instances = []
    monkeypatch.setattr(
        sys.modules["models.demos.gemma4.tt.dflash_drafter"], "DFlashFusedDecoder", _RecordingDecoder, raising=False
    )
    cls = adapter.Gemma4DFlashForCausalLM
    model = cls.__new__(cls)
    model.model = [object()]
    model.model_args = [SimpleNamespace(max_seq_len=4096)]
    model._spec_decoder = None
    model._spec_width_ladder = None
    model._spec_horizon = 256
    model._spec_get_drafter = lambda: SimpleNamespace(target_layer_ids=[1])
    model._spec_capture_width_set([[("k", "v")]], 64, prepare_only=True)
    assert _RecordingDecoder.instances[0].verify_count is None
