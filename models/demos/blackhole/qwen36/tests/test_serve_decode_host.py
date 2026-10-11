# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests of the device-resident serving decode (change 1.6S, QWEN36_SERVE_DEVICE_DECODE).

No device, no ttnn tensors: Generator.decode_forward is monkeypatched with a recorder and the model is a namespace fake.
They check the contract plumbing of ``Qwen36ForCausalLM.decode_forward`` (four commands pass through unchanged, illegal
combinations raise before any state moves, slot_remap applied once, bucket rule identical on remap steps, bucket switch
without reload refused), the model_capabilities per flag, and the pure-torch input helpers of ``Qwen36Model``.
"""

from types import SimpleNamespace

import pytest
import torch

from models.demos.blackhole.qwen36.tt import qwen36_vllm
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36ForCausalLM, _build_model_capabilities
from models.tt_transformers.tt.generator import Generator

WIDTH = 8
COMMANDS = ("reload_inputs", "reload_page_table", "reload_sampling_params", "reset_sampling_state")


class _Recorder:
    def __init__(self):
        self.calls = []  # Generator.decode_forward kwargs, in order
        self.remaps = []  # GDN remaps, in order
        self.rope_remaps = []  # per-slot rope-delta remaps, in order
        self.events = []  # interleaved order of "remap" / "forward"


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.delenv("QWEN36_SERVE_DEVICE_DECODE", raising=False)
    monkeypatch.delenv("TT_DECODE_BUCKETING", raising=False)
    rec = _Recorder()

    def fake_decode_forward(self, *args, **kwargs):
        assert not args
        rec.calls.append(kwargs)
        rec.events.append("forward")
        return "out"

    monkeypatch.setattr(Generator, "decode_forward", fake_decode_forward)

    def remap(r):
        rec.remaps.append(list(r))
        rec.events.append("remap")

    model = SimpleNamespace(
        num_devices=4,
        args=SimpleNamespace(max_batch_size=WIDTH),
        _serve_device_decode=True,
        _remap_gdn_slots=remap,
        remap_slot_rope_delta=lambda r: rec.rope_remaps.append(list(r)),
        sampling=None,
    )
    gen = Qwen36ForCausalLM.__new__(Qwen36ForCausalLM)
    gen.model = [model]
    return gen, rec


def _inputs(num_active, width=WIDTH):
    tokens = torch.arange(1, width + 1, dtype=torch.int32).reshape(width, 1)
    pos = torch.full((width,), -1, dtype=torch.int32)
    pos[:num_active] = 100
    pt = torch.arange(width * 4, dtype=torch.int32).reshape(width, 4)
    return tokens, pos, pt


def _call(gen, num_active, *, sampling=True, remap=None, width=WIDTH, **commands):
    tokens, pos, pt = _inputs(num_active, width)
    kw = dict(
        tokens=tokens,
        start_pos=pos,
        page_table=pt,
        kv_cache=None,
        enable_trace=True,
        read_from_device=False,
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=False,
        reset_sampling_state=False,
    )
    if sampling:
        kw["sampling_params"] = object()
    if remap is not None:
        kw["slot_remap"] = remap
    kw.update(commands)
    return gen.decode_forward(**kw)


# ---------------------------------------------------------------------------------------------------------------------
# model_capabilities
# ---------------------------------------------------------------------------------------------------------------------


def test_capabilities_flag_on(monkeypatch):
    monkeypatch.setenv("QWEN36_SERVE_DEVICE_DECODE", "1")
    caps = _build_model_capabilities()
    assert caps["supports_async_decode"] is True
    assert caps["supports_sample_on_device"] is True
    assert caps["supports_prefix_caching"] is False
    assert caps["max_device_top_k"] == 32
    assert caps["supports_device_penalties"] is False
    assert Qwen36ForCausalLM.decode_input_update_contract == 1


def test_capabilities_default_is_flag_on(monkeypatch):
    monkeypatch.delenv("QWEN36_SERVE_DEVICE_DECODE", raising=False)
    assert _build_model_capabilities()["supports_async_decode"] is True


def test_capabilities_flag_off_is_the_old_dict(monkeypatch):
    monkeypatch.setenv("QWEN36_SERVE_DEVICE_DECODE", "0")
    assert _build_model_capabilities() == {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }


def test_readback_is_the_generators():
    """Async readback / host processing already serve this model through the Generator (token buffer in place)."""
    assert Qwen36ForCausalLM.read_decode_output is Generator.read_decode_output
    assert Qwen36ForCausalLM.process_decode_output_host is Generator.process_decode_output_host


# ---------------------------------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "commands",
    [
        dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=True),
        dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False),
        dict(reload_inputs=False, reload_page_table=True, reload_sampling_params=False, reset_sampling_state=False),
        dict(reload_inputs=False, reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False),
        dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=False),
    ],
)
def test_four_commands_pass_through_unchanged(harness, commands):
    gen, rec = harness
    # steady / page-only steps need a previous accepted submission in the same bucket
    _call(gen, 3)
    rec.calls.clear()
    assert _call(gen, 3, **commands) == "out"
    (kw,) = rec.calls
    for name in COMMANDS:
        assert kw[name] is commands[name], name


def test_direct_caller_defaults_are_host_authoritative(harness):
    gen, rec = harness
    tokens, pos, pt = _inputs(3)
    gen.decode_forward(tokens=tokens, start_pos=pos, page_table=pt, sampling_params=object())
    kw = rec.calls[-1]
    assert (kw["reload_inputs"], kw["reload_page_table"]) == (True, False)
    assert (kw["reload_sampling_params"], kw["reset_sampling_state"]) == (False, False)


def test_positional_arguments_are_bound_by_generator_order(harness):
    gen, rec = harness
    tokens, pos, pt = _inputs(3)
    gen.decode_forward(tokens, pos, pt, None, True, False, object(), reload_inputs=True)
    kw = rec.calls[-1]
    assert kw["enable_trace"] is True and kw["read_from_device"] is False
    assert kw["tokens"].shape[0] == 4


@pytest.mark.parametrize(
    "bad,kwargs",
    [
        ("reload_inputs and reload_page_table", dict(reload_inputs=True, reload_page_table=True)),
        ("reset_sampling_state without reload_inputs", dict(reload_inputs=False, reset_sampling_state=True)),
        ("host sampling without reload_inputs", dict(reload_inputs=False, sampling=False)),
        ("remap without reload_inputs", dict(reload_inputs=False, remap=[1, 0, 2, 3, 4, 5, 6, 7])),
        ("legacy reset_batch", dict(reset_batch=True)),
    ],
)
def test_illegal_combos_raise_before_any_state_moves(harness, expect_error, bad, kwargs):
    gen, rec = harness
    _call(gen, 3)  # a valid previous step in the same bucket, so only the combo itself can fail
    rec.calls.clear()
    rec.remaps.clear()
    with expect_error(
        (ValueError, TypeError),
        "reload_page_table must be false|Resetting sampling state requires|Host sampling requires|slot_remap moves|reset_batch is legacy",
    ):
        _call(gen, 3, **kwargs)
    assert rec.calls == [] and rec.remaps == [], bad


def test_page_table_only_step_and_nothing_step_are_legal_for_device_sampling(harness):
    gen, rec = harness
    _call(gen, 3)
    _call(gen, 3, reload_inputs=False, reload_page_table=True)
    _call(gen, 3, reload_inputs=False)
    assert [c["reload_page_table"] for c in rec.calls] == [False, True, False]
    assert [c["reload_inputs"] for c in rec.calls] == [True, False, False]


def test_host_sampling_with_reload_is_legal(harness):
    gen, rec = harness
    _call(gen, 3, sampling=False)
    assert "sampling_params" not in rec.calls[-1]


# ---------------------------------------------------------------------------------------------------------------------
# remap + bucketing
# ---------------------------------------------------------------------------------------------------------------------


def test_remap_applied_once_before_forward_and_forwarded_to_generator(harness):
    gen, rec = harness
    remap = [1, 2, 0, 3, 4, 5, 6, 7]
    _call(gen, 3, remap=remap)
    assert rec.remaps == [remap]
    assert rec.events == ["remap", "forward"]
    # The Generator owns the sampler-side remap and gets the SAME remap exactly once (it is not applied twice here).
    assert rec.calls[-1]["slot_remap"] == remap


def test_remap_step_uses_the_steady_bucket_rule_and_bucket_is_stable(harness):
    gen, rec = harness
    remap = [1, 2, 0, 3, 4, 5, 6, 7]
    _call(gen, 3, remap=remap)  # remap step: 3 active -> bucket 4 (NOT full width 8, the old hazard)
    _call(gen, 3, reload_inputs=False)  # next steady step replays the same bucket
    widths = [c["tokens"].shape[0] for c in rec.calls]
    assert widths == [4, 4]
    assert [c["start_pos"].shape[0] for c in rec.calls] == [4, 4]
    assert [c["page_table"].shape[0] for c in rec.calls] == [4, 4]
    assert len(rec.remaps) == 1


def test_bucket_follows_active_prefix(harness):
    gen, rec = harness
    for active, expect in ((1, 1), (2, 2), (3, 4), (4, 4), (5, 8), (8, 8)):
        _call(gen, active)
        assert rec.calls[-1]["tokens"].shape[0] == expect


def test_bucket_switch_without_reload_raises_and_with_reload_is_fine(harness, expect_error):
    gen, rec = harness
    _call(gen, 3)  # bucket 4
    n = len(rec.calls)
    with expect_error(ValueError, "bucket changed"):
        _call(gen, 2, reload_inputs=False)  # layout shrank to bucket 2 without a reload
    with expect_error(ValueError, "bucket changed"):
        _call(gen, 5, reload_inputs=False, reload_page_table=True)
    assert len(rec.calls) == n
    _call(gen, 2)  # commanded reload: fine, new bucket
    _call(gen, 2, reload_inputs=False)
    assert [c["tokens"].shape[0] for c in rec.calls[n:]] == [2, 2]


def test_steady_step_without_a_previous_accepted_submission_raises(harness, expect_error):
    gen, rec = harness
    with expect_error(ValueError, "bucket changed"):
        _call(gen, 3, reload_inputs=False)
    assert rec.calls == []


def test_prefill_invalidates_resident_inputs(harness, expect_error, monkeypatch):
    gen, rec = harness
    _call(gen, 3)
    model = gen.model[0]
    model.prefill_paged_slots = lambda *a, **k: [torch.zeros(4)]
    monkeypatch.setattr(Qwen36ForCausalLM, "_has_visual", staticmethod(lambda kwargs, key, u=0: False))
    gen.prefill_forward(
        torch.ones(1, 4, dtype=torch.int32), torch.zeros(1, 4, dtype=torch.int32), None, [4], empty_slots=[0]
    )
    with expect_error(ValueError, "bucket changed"):
        _call(gen, 3, reload_inputs=False)
    _call(gen, 3)  # the commanded reload re-establishes the chain


def test_remap_failure_window_validation_precedes_remap(harness, expect_error):
    gen, rec = harness
    _call(gen, 3)
    rec.remaps.clear()
    with expect_error(ValueError, "reload_page_table must be false"):
        _call(gen, 3, remap=[1, 0, 2, 3, 4, 5, 6, 7], reload_inputs=True, reload_page_table=True)
    assert rec.remaps == []


def test_bucketing_disabled_keeps_full_width(harness, monkeypatch):
    gen, rec = harness
    monkeypatch.setenv("TT_DECODE_BUCKETING", "0")
    _call(gen, 3)
    assert rec.calls[-1]["tokens"].shape[0] == WIDTH


# ---------------------------------------------------------------------------------------------------------------------
# flag off: the pre-1.6S behaviour
# ---------------------------------------------------------------------------------------------------------------------


def test_flag_off_uses_the_legacy_path(harness, monkeypatch):
    gen, rec = harness
    monkeypatch.setenv("QWEN36_SERVE_DEVICE_DECODE", "0")
    remap = [1, 2, 0, 3, 4, 5, 6, 7]
    # Legacy semantics: a remap forces the FULL width bucket, no contract validation, no guard.
    _call(gen, 3, remap=remap)
    assert rec.calls[-1]["tokens"].shape[0] == WIDTH
    _call(gen, 3, reload_inputs=False)  # legacy never raises on the bucket
    assert rec.calls[-1]["tokens"].shape[0] == 4
    assert rec.remaps == [remap]


def test_single_device_model_takes_the_legacy_path(harness):
    gen, rec = harness
    gen.model[0].num_devices = 1
    gen.model[0]._serve_device_decode = False
    _call(gen, 3, remap=[1, 2, 0, 3, 4, 5, 6, 7])
    assert rec.calls[-1]["tokens"].shape[0] == WIDTH
    assert rec.remaps == []  # legacy: GDN remap only for TP batched
    assert rec.rope_remaps == []


# ---------------------------------------------------------------------------------------------------------------------
# pure-torch input helpers of the model
# ---------------------------------------------------------------------------------------------------------------------


def test_resident_token_row_pads_to_32_lanes():
    row = Qwen36Model._resident_token_row(torch.tensor([[7], [8], [9]]), 32)
    assert row.shape == (1, 1, 1, 32) and row.dtype == torch.int32
    assert row[0, 0, 0, :4].tolist() == [7, 8, 9, 0] and int(row[..., 3:].abs().sum()) == 0
    # sentinel / negative tokens on a non-reload step are clamped into the unsigned range, not asserted
    assert int(Qwen36Model._resident_token_row(torch.tensor([[-5]]), 32).min()) == 0


def test_resident_rope_index_matches_host_position_rule():
    pos = torch.tensor([5, 100, -1, 7], dtype=torch.int32)
    idx = Qwen36Model._resident_rope_index(pos, 0, 4096)
    assert idx.shape == (1, 4) and idx.dtype == torch.int32
    assert idx.tolist() == [[5, 100, 0, 7]]  # idle row -> max(-1, 0)
    assert Qwen36Model._resident_rope_index(pos, 10, 4096).tolist() == [[15, 110, 10, 17]]  # M-RoPE delta
    assert Qwen36Model._resident_rope_index(pos, -50, 4096).tolist() == [[0, 50, 0, 0]]  # negative delta clamps at 0
    # per-slot delta vector (batched serving): one delta per row; idle row still clamps its position to 0
    delta = torch.tensor([0, 10, 3, -50], dtype=torch.int64)
    assert Qwen36Model._resident_rope_index(pos, delta, 4096).tolist() == [[5, 110, 3, 0]]
    big = torch.tensor([10**9], dtype=torch.int32)  # stale host position on a non-reload step: still in range
    assert Qwen36Model._resident_rope_index(big, 0, 4096).tolist() == [[4095]]


def test_module_exports():
    assert qwen36_vllm._MAX_DEVICE_TOP_K == 32


def test_slot_rope_delta_remap_follows_gdn_slot_semantics():
    m = Qwen36Model.__new__(Qwen36Model)
    m.args = SimpleNamespace(max_batch_size=4)
    m._slot_rope_delta = torch.zeros(4, dtype=torch.int64)
    for slot, d in enumerate([0, 11, 0, 33]):
        m.set_slot_rope_delta(slot, d)
    m.remap_slot_rope_delta([1, 3, 2, 0])  # slot i takes the delta previously at slot remap[i]
    assert m._slot_rope_delta.tolist() == [11, 33, 0, 0]
    m.remap_slot_rope_delta([1, 0])  # rows beyond len(remap) unchanged
    assert m._slot_rope_delta.tolist() == [33, 11, 0, 0]
    assert m._decode_rope_delta(2).tolist() == [33, 11]
    m.args.max_batch_size = 1
    m.rope = SimpleNamespace(rope_delta=7)
    assert m._decode_rope_delta(1) == 7
