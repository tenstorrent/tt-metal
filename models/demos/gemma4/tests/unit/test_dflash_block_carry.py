# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A dFlash block step must not DROP the tokens it produced past its width.

One fused iteration commits an accepted prefix plus a bonus, up to V+1 tokens,
so the block loop can pass ``_SPEC_BLOCK`` mid-iteration. Truncating with
``block[:K]`` desyncs the stream from the session: the dropped tokens' KV is
already written and ``dec.start`` has advanced past them, so the model keeps
conditioning on tokens the caller never received. In prose that is an invisible
gap, not an error -- which is why it needs a test rather than a warning.

The excess is carried to the next step instead. Review finding on
vllm-tt-plugin#118: "The current dFlash fill/truncate behavior needs resolution
before the combined serving path is treated as correct."

Host-only: the fused decoder is a stub, no device.
"""

import pytest
import torch

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashForCausalLM as DF


class _Dec:
    """Scripted fused decoder: each step() yields one iteration's commit."""

    def __init__(self, script, start=100):
        self.script = list(script)
        self.start = start
        self.anchor = 11
        self.calls = 0

        class _D:
            vocab = 32000

        self.drafter = _D()

    def step(self, first=False):
        self.calls += 1
        committed = self.script.pop(0)
        self.start += len(committed)
        return committed[:-1], committed[-1], len(committed)

    def select_width(self, pos):
        return 4096

    def refresh_page_tables(self, row):
        pass


def _model(monkeypatch, script, block=8, carry=None, budget=10**9):
    from types import SimpleNamespace

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    monkeypatch.setattr(Gemma4ForCausalLM, "decode_forward", lambda self, *a, **k: "baseline")
    m = DF.__new__(DF)
    m._SPEC_BLOCK = block
    m._spec_decoder = _Dec(script)
    m._spec_active = True
    m._spec_active_owner = None
    m._spec_pending = None
    m._spec_pending_owner = None
    m._spec_first_step = False
    m._spec_width_set = True
    m._spec_width_ladder = [4096]
    m._spec_budget_end = budget
    m._spec_carry = list(carry or [])
    m._spec_last_pt = None
    m._bounded_sliding_kv_cache = False
    m.model = [SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=1))]
    return m


def _run(m, pos=100):
    return m.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([pos], dtype=torch.int32),
    )


def test_overshoot_is_carried_not_dropped(monkeypatch):
    """Three 3-token iterations fill a width-8 block to 9. The 9th token must
    survive into the next step, because its KV is already written."""
    m = _model(monkeypatch, [[21, 22, 23], [24, 25, 26], [27, 28, 29]], block=8)
    out = _run(m)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26, 27, 28]
    assert m._spec_carry == [29]


def test_carry_is_delivered_first_on_the_next_step(monkeypatch):
    m = _model(monkeypatch, [[30, 31, 32], [33, 34, 35]], block=8, carry=[28, 29])
    out = _run(m)
    # carry first, then this step's iterations, in order
    assert out[0].tolist() == [28, 29, 30, 31, 32, 33, 34, 35]
    assert m._spec_carry == []


def test_carry_counts_toward_the_width(monkeypatch):
    """A carry large enough to fill the block must emit ZERO iterations: the
    session is already that far ahead."""
    m = _model(monkeypatch, [[99] * 3], block=4, carry=[41, 42, 43, 44])
    out = _run(m)
    assert out[0].tolist() == [41, 42, 43, 44]
    assert m._spec_decoder.calls == 0


def test_exact_fit_leaves_no_carry(monkeypatch):
    m = _model(monkeypatch, [[21, 22, 23, 24], [25, 26, 27, 28]], block=8)
    out = _run(m)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26, 27, 28]
    assert m._spec_carry == []


def test_eos_still_stops_and_fills_the_tail(monkeypatch):
    """EOS ends the request, so the tail is EOS-filled and any carry behind it
    is moot -- the scheduler trims at the first stop token."""
    m = _model(monkeypatch, [[21, 1], [99, 99]], block=8)
    out = _run(m)
    row = out[0].tolist()
    assert row[:2] == [21, 1]
    assert set(row[2:]) == {1}
    assert m._spec_decoder.calls == 1


def test_a_carry_never_crosses_a_session(monkeypatch):
    """_spec_bootstrap clears it: the next request's KV has none of it."""
    m = _model(monkeypatch, [[21, 22]] * 8, block=8, carry=[77])
    assert m._spec_carry == [77]
    m._spec_carry = []  # what bootstrap does
    out = _run(m)
    assert 77 not in out[0].tolist()


@pytest.mark.parametrize("width", [2, 4, 16, 64])
def test_emitted_width_is_always_the_block_width(monkeypatch, width):
    m = _model(monkeypatch, [[21, 22, 23]] * 128, block=width)
    out = _run(m)
    assert out.shape == (1, width)
