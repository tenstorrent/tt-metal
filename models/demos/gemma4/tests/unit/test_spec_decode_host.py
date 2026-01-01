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
import ast
import builtins
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import ttnn
from models.demos.gemma4.tests.unit.dflash_contract_harness import make_expect_error
from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV, bounded_ring_modulo
from models.demos.gemma4.tt.dflash_drafter import DFlashFusedDecoder
from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashForCausalLM as DF
from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder, auto_draft_len, auto_draft_len_batched

try:
    import vllm  # noqa: F401

    _HAS_VLLM = True
except ImportError:  # the CI unit runners have no vLLM: those sections skip
    _HAS_VLLM = False


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
    m._spec_horizon = 2048
    m._spec_decoder_bucket = None
    m._spec_last_pt = None
    m._bounded_sliding_kv_cache = False
    m.model = [SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=1))]
    return m


def _bootstrap_via_width_set(m, anchor=11, start=100):
    """Arm a session through the REAL _spec_bootstrap width-set branch.

    That branch returns early, which is why it needs its own coverage: a reset
    placed only on the capture path never ran for it.
    """
    m._spec_decoder.width_for = lambda pos: 4096
    m._spec_decoder.prefill_ingest = lambda taps, n: None
    m._spec_decoder.reseed = lambda a, s: None
    m._spec_pending = (object(), int(start))
    m._spec_pending_owner = None
    DF._spec_bootstrap(
        m,
        anchor,
        start,
        torch.zeros((1, 8), dtype=torch.int32),
        None,
        page_tables_per_layer=None,
    )


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


def test_release_request_drops_a_carry(monkeypatch):
    """Teardown must drop it: the retained decoder outlives the request.

    The decoder is deliberately kept alive across requests for reuse, so
    anything request-scoped has to die in release_request. A carry left behind
    is emitted as the NEXT request's first tokens.
    """
    m = _model(monkeypatch, [[21, 22]] * 8, block=8, carry=[77])
    assert m._spec_carry == [77]
    DF.release_request(m, 0)
    assert m._spec_carry == []


def test_a_carry_never_crosses_a_session(monkeypatch):
    """Through the REAL release and bootstrap, with no manual reset.

    Both paths matter: release_request ends request A, and the width-set
    branch of _spec_bootstrap arms request B and returns early -- a reset
    placed only on the capture path left that branch leaking A's tokens into
    B's first block.
    """
    m = _model(monkeypatch, [[21, 22]] * 8, block=8, carry=[77])
    assert m._spec_carry == [77]
    DF.release_request(m, 0)
    _bootstrap_via_width_set(m)
    out = _run(m)
    assert m._spec_carry == [] or 77 not in m._spec_carry
    assert 77 not in out[0].tolist()


def test_the_width_set_bootstrap_branch_clears_the_carry(monkeypatch):
    """The early-returning branch specifically, since it is the one that leaked."""
    m = _model(monkeypatch, [[21, 22]] * 8, block=8, carry=[77])
    _bootstrap_via_width_set(m)
    assert m._spec_carry == []


@pytest.mark.parametrize("width", [2, 4, 16, 64])
def test_emitted_width_is_always_the_block_width(monkeypatch, width):
    m = _model(monkeypatch, [[21, 22, 23]] * 128, block=width)
    out = _run(m)
    assert out.shape == (1, width)


def test_releasing_another_request_preserves_the_owner_session(monkeypatch):
    """Adaptive serving admits several live requests; this adapter keeps one
    session. Aborting A must not tear down a session B owns.

    The sequence that broke: A and B prefill separately, the client aborts A,
    the runner calls release_request(A_slot), and release_request cleared the
    GLOBAL session -- B's. B then decoded one baseline token against the full
    block the scheduler had reserved for it, and the scheduler rejected the
    width.
    """
    m = _model(monkeypatch, [[21, 22]] * 8, block=8)
    m._spec_owner_slot = 3  # B owns the session, parked at state slot 3
    m._spec_active = True
    m._spec_pending = ("taps", 10)

    DF.release_request(m, 1)  # A finishes at a different slot

    assert m._spec_active is True, "B's live session must survive A's release"
    assert m._spec_pending == ("taps", 10)
    assert m._spec_owner_slot == 3


def test_releasing_the_owner_still_tears_the_session_down(monkeypatch):
    m = _model(monkeypatch, [[21, 22]] * 8, block=8, carry=[77])
    m._spec_owner_slot = 3
    m._spec_active = True
    m._spec_pending = ("taps", 10)

    DF.release_request(m, 3)

    assert m._spec_active is False
    assert m._spec_pending is None
    assert m._spec_carry == []
    assert m._spec_owner_slot is None


def test_the_owner_slot_follows_a_runner_gather(monkeypatch):
    """The runner permutes per-slot state between decode steps and then releases
    by the CURRENT slot, so the recorded owner has to move with it.

    Without this the runner had to release by the slot the request was PREFILLED
    into, and that is ambiguous: slots are reused, so one request's prefill slot
    can be another live request's owner slot and the release clears the wrong
    session (vllm-tt-plugin#118 review, finding 1).
    """
    m = _model(monkeypatch, [[21, 22]] * 8, block=8)
    m._spec_owner_slot = 1
    m._spec_active = True
    m._spec_pending = ("taps", 10)

    # Row 0 reads slot 1 and row 1 reads slot 0: the whole permutation arrives at
    # once, because applying the pairs in sequence would move the owner twice.
    DF.note_state_slots_moved(m, {1: 0, 0: 1})

    assert m._spec_owner_slot == 0, "the owner followed its state to slot 0"
    assert m._spec_active is True, "a gather is not a release"

    # Releasing the pre-move slot is now somebody else's release.
    DF.release_request(m, 1)
    assert m._spec_active is True, "slot 1 is no longer this session's identity"

    # Releasing the current slot tears it down.
    DF.release_request(m, 0)
    assert m._spec_active is False
    assert m._spec_owner_slot is None


def test_a_gather_that_does_not_touch_the_owner_leaves_it_alone(monkeypatch):
    m = _model(monkeypatch, [[21, 22]] * 8, block=8)
    m._spec_owner_slot = 3
    m._spec_active = True

    DF.note_state_slots_moved(m, {1: 0, 0: 1})

    assert m._spec_owner_slot == 3
    assert m._spec_active is True


def test_a_gather_with_no_session_is_a_no_op(monkeypatch):
    """Nothing armed, so there is no owner to move."""
    m = _model(monkeypatch, [[21, 22]] * 8, block=8)
    m._spec_owner_slot = None

    DF.note_state_slots_moved(m, {1: 0, 0: 1})

    assert m._spec_owner_slot is None


def test_an_unknown_owner_slot_keeps_the_old_unconditional_behaviour(monkeypatch):
    """The runner may supply no empty_slots; that is the pre-existing
    single-session case and must not start leaking sessions."""
    m = _model(monkeypatch, [[21, 22]] * 8, block=8, carry=[77])
    m._spec_owner_slot = None
    m._spec_active = True
    m._spec_pending = ("taps", 10)

    DF.release_request(m, 7)

    assert m._spec_active is False
    assert m._spec_carry == []


# ===== merged from tests/unit/test_dflash_bucket_ladder.py: Tests for the enumerable packed-verify bucket ladder.
try:
    from models.demos.gemma4.tt.generator_vllm import dflash_bucket_for, dflash_pv_bucket_ladder
except ImportError:  # vLLM absent: this section's tests skip
    pass


_H, _V = 2048, 5


_MAX_CTX = 262144


def _exact_bucket(start, horizon=_H, verify=_V):
    """Mirror of DFlashFusedDecoder.pv_bucket -- the value capture would use."""
    return ((start + horizon + (verify + 1) + 64 + 1023) // 1024) * 1024


@pytest.fixture()
def _pin_env(monkeypatch):
    monkeypatch.setenv("GEMMA4_DFLASH_SERVE_HORIZON", str(_H))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", str(_V))


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_ladder_is_enumerable_and_small():
    """Logarithmic in context: a 256K server must not need ~258 captures."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    assert len(ladder) <= 12
    assert ladder == sorted(set(ladder))
    assert all(r % 1024 == 0 for r in ladder)


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("start", [0, 1, 127, 128, 1000, 4096, 8192, 16384, 32768, 65536, 131072, 200000, _MAX_CTX])
def test_every_prompt_length_rounds_up_to_a_rung(start):
    """The whole point: no prompt inside max_context may be uncovered, and the
    chosen rung must be >= the bucket capture would have used (never smaller,
    or the verify would attend past what it captured)."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    rung = dflash_bucket_for(start, ladder)
    assert rung is not None
    assert rung >= _exact_bucket(start)


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_rounding_waste_is_bounded():
    """Rounding up costs verify width, so the ladder may not be arbitrarily
    coarse. Doubling rungs bound it below 2x the exact bucket."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    worst = max(dflash_bucket_for(s, ladder) / _exact_bucket(s) for s in range(0, 65536, 97))
    assert worst < 2.0


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_beyond_ladder_is_none_not_a_wrong_rung():
    """A request past the ladder must signal fallback, NOT silently get the
    largest rung -- that would attend past the captured width."""
    ladder = dflash_pv_bucket_ladder(_MAX_CTX)
    assert dflash_bucket_for(_MAX_CTX * 2, ladder) is None


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_cap_keeps_the_largest_rungs():
    """Capping trades wasted width on SHORT prompts for a bounded capture count.
    Dropping the largest rungs instead would leave long prompts uncovered."""
    full = dflash_pv_bucket_ladder(_MAX_CTX)
    capped = dflash_pv_bucket_ladder(_MAX_CTX, max_rungs=3)
    assert len(capped) == 3
    assert capped == full[-3:]
    # Long prompts stay covered; that is the invariant the cap must not break.
    assert dflash_bucket_for(_MAX_CTX, capped) is not None


@pytest.mark.usefixtures("_pin_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_horizon_and_verify_widen_the_smallest_rung():
    """The bucket carries the generation horizon, so a bigger horizon must not
    silently leave a short prompt under-covered."""
    small = dflash_pv_bucket_ladder(_MAX_CTX, horizon=512, verify=_V)
    large = dflash_pv_bucket_ladder(_MAX_CTX, horizon=8192, verify=_V)
    assert large[0] > small[0]
    assert dflash_bucket_for(0, large) >= _exact_bucket(0, horizon=8192)


# ===== merged from tests/unit/test_dflash_release_runtime.py: ``release_persistent_capture`` on the base ``Generator`` and DiffusionGemma.
pytest.importorskip("ttnn.device", reason="needs the TT runtime")


def _base_generator():
    from models.tt_transformers.tt.generator import Generator

    generator = Generator.__new__(Generator)
    released = []
    generator.model_args = [SimpleNamespace(mesh_device="mesh")]
    generator.model = [SimpleNamespace(sampling=SimpleNamespace(reset_trace=lambda: released.append("sampling")))]
    generator.trace_id_prefill = {"128_0": 11}
    generator.trace_id_prefill_sampling = {}
    generator.trace_ids_decode = {}
    generator.data_parallel = 1
    return generator, released


def test_base_release_runs_once_and_the_destructor_only_runs_the_base(monkeypatch):
    generator, released = _base_generator()
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: released.append(tid))
    generator.release_persistent_capture()
    generator.release_persistent_capture()
    assert released.count(11) == 1 and released.count("sampling") == 1
    generator.__del__()
    assert released.count(11) == 1


@pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="DiffusionGemmaForCausalLM derives from the vLLM generator base in models.tt_transformers.tt.generator_vllm, "
    "which imports vllm at module level; the models unit-test image does not ship vllm",
)
def test_diffusion_gemma_release_reaches_the_base():
    from models.experimental.diffusion_gemma.tt.generator_vllm import DiffusionGemmaForCausalLM

    model = DiffusionGemmaForCausalLM.__new__(DiffusionGemmaForCausalLM)
    model._sessions = {}
    model._persistent_adapter = None
    model.model_args = []
    model.model = []
    model.release_persistent_capture()
    assert getattr(model, "_generator_capture_released", False) is True


# ===== merged from tests/unit/test_dflash_tap_keep.py: ``Gemma4Model.dflash_capture_taps`` stores the ``keep_last`` cap.
def _model_dflash_tap_keep():
    module = pytest.importorskip("models.demos.gemma4.tt.model", reason="needs the TT runtime")
    return module.Gemma4Model.__new__(module.Gemma4Model)


def test_keep_last_is_stored_in_tap_entries():
    model = _model_dflash_tap_keep()
    model.dflash_capture_taps([1, 2, 3], keep_last=4)
    assert model._dflash_tap_layers == {1, 2, 3}
    assert model._dflash_tap_keep == 12
    assert model._dflash_taps == [] and model._dflash_tap_idx == 0


def test_no_keep_last_means_no_cap():
    model = _model_dflash_tap_keep()
    model.dflash_capture_taps([1, 2], buffers=["a", "b"])
    assert model._dflash_tap_keep is None
    assert model._dflash_tap_buffers == ["a", "b"]


def test_disarming_clears_the_cap():
    model = _model_dflash_tap_keep()
    model.dflash_capture_taps([1], keep_last=2)
    model.dflash_capture_taps(None)
    assert model._dflash_tap_layers is None and model._dflash_tap_keep is None


# ===== merged from tests/unit/test_dflash_warmup_prefill_trace.py: dFlash warms prefill eagerly: it never replays a prefill trace (tt-metal#57853).
try:
    from models.demos.gemma4.tt.generator_vllm import (
        Gemma4DFlashContractForCausalLM,
        Gemma4DFlashForCausalLM,
        Gemma4ForCausalLM,
        Gemma4MTPForCausalLM,
    )
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _captured_enable_trace(monkeypatch, cls, *, bounded):
    """Call ``cls``'s warmup override and report the enable_trace it passed up.

    The base ``Gemma4ForCausalLM.warmup_model_prefill`` needs a live model and a
    device, so it is replaced with a recorder: what matters here is the value the
    subclass decides to pass, not what the base then does with it.
    """
    seen = {}

    def _record(self, kv_cache, enable_trace, can_sample_on_device, greedy_only=False):
        seen["enable_trace"] = enable_trace

    monkeypatch.setattr(Gemma4ForCausalLM, "warmup_model_prefill", _record, raising=True)
    instance = object.__new__(cls)
    instance._bounded_sliding_kv_cache = bounded
    cls.warmup_model_prefill(instance, None, True, True)
    return seen.get("enable_trace")


if _HAS_VLLM:

    @pytest.mark.parametrize("bounded", [False, True], ids=["unbounded", "bounded"])
    @pytest.mark.parametrize(
        "cls",
        [Gemma4DFlashForCausalLM, Gemma4DFlashContractForCausalLM],
        ids=["block_rail", "contract_rail"],
    )
    def test_dflash_warmup_never_captures_a_prefill_trace(monkeypatch, cls, bounded):
        """Both rails, both bounded settings: the capture is always off.

        The unbounded case is the regression -- that is the one the inherited
        warmup left at True while runtime forced it False.
        """
        assert _captured_enable_trace(monkeypatch, cls, bounded=bounded) is False


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_override_is_scoped_to_the_dflash_rails():
    """It must not leak to classes that DO replay their prefill traces.

    ``Gemma4ForCausalLM`` replays them, and capturing them at warmup is the
    #49083 fix -- forcing the capture off there would undo it. MTP inherits the
    base warmup for the same reason. Asserting the identity of the bound
    function keeps this honest without re-stubbing the base's internals.
    """
    assert Gemma4DFlashForCausalLM.warmup_model_prefill is not Gemma4ForCausalLM.warmup_model_prefill
    assert Gemma4DFlashContractForCausalLM.warmup_model_prefill is Gemma4DFlashForCausalLM.warmup_model_prefill
    assert Gemma4MTPForCausalLM.warmup_model_prefill is Gemma4ForCausalLM.warmup_model_prefill


# ===== merged from tests/unit/test_dflash_width_set.py: Tests for the dFlash packed-verify WIDTH SET selection and migration.
try:
    from models.demos.gemma4.tt.generator_vllm import dflash_pv_bucket_ladder
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _decoder(widths, captured=None, P_v=6):
    """A decoder carrying a captured width set and nothing else."""
    dec = DFlashFusedDecoder.__new__(DFlashFusedDecoder)
    dec.P_v = P_v
    dec.use_packed = True
    captured = set(widths if captured is None else captured)
    dec._pv_widths = {int(w): {"pv_sk": int(w), "trace": (object() if w in captured else None)} for w in widths}
    dec.pv_sk = min(int(w) for w in widths)
    return dec


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_narrowest_covering_width_is_selected():
    dec = _decoder([2048, 4096, 8192])
    assert dec.width_for(0) == 2048
    assert dec.width_for(2048 - 6 - 64) == 2048
    assert dec.width_for(2048 - 6 - 64 + 1) == 4096
    assert dec.width_for(4096 - 6 - 64 + 1) == 8192


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_uncaptured_widths_are_never_selected():
    """A width whose trace was not captured cannot serve a step: replaying
    another width's trace would read the wrong pv_iota length."""
    dec = _decoder([2048, 4096, 8192], captured=[2048, 8192])
    assert dec.width_for(2048) == 8192


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_past_the_widest_width_is_none_not_a_wrong_width():
    dec = _decoder([2048, 4096])
    assert dec.width_for(4096) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_ladder_covers_every_position_up_to_max_model_len():
    """The migration guarantee: the largest rung covers max_model_len, so a
    request never runs out of widths before vLLM's own length limit."""
    max_model_len = 262144
    ladder = dflash_pv_bucket_ladder(max_model_len, horizon=2048, verify=5)
    dec = _decoder(ladder)
    for pos in (0, 1, 2047, 4096, 131072, max_model_len - 1, max_model_len):
        assert dec.width_for(pos) is not None, pos


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("pos", [0, 1000, 5000, 100000])
def test_selection_is_monotonic_in_position(pos):
    """Width never shrinks as a request generates: a migration only ever moves
    to a wider trace, so the mask and page table only ever grow."""
    ladder = dflash_pv_bucket_ladder(262144, horizon=2048, verify=5)
    dec = _decoder(ladder)
    w = dec.width_for(pos)
    assert dec.width_for(pos + 1) >= w
    assert dec.width_for(pos + 1000) >= w


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_top_width_is_the_block_budget_not_the_rounded_context():
    """Reproduces the warmup kill: the ladder's top rung rounds UP past
    max_model_len (265216 for a 262144 server = 4144 blocks), and
    paged_update_cache requires max_num_blocks_per_seq < max_num_blocks, so
    warmup died with 'max_num_blocks_per_seq=4144, max_num_blocks=4128'."""
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=4096, horizon=2048, verify=5)
    assert max(widths) == 262144
    assert max(widths) // 64 == 4096
    assert all(w // 64 <= 4096 for w in widths)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_set_drops_rungs_past_the_budget_and_keeps_the_budget():
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=1024, horizon=2048, verify=5)
    assert max(widths) == 65536  # 1024 blocks x 64
    assert all(w <= 65536 for w in widths)
    assert widths == sorted(widths)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_set_without_a_block_budget_is_the_plain_ladder():
    from models.demos.gemma4.tt.generator_vllm import dflash_pv_bucket_ladder, dflash_width_set

    assert dflash_width_set(262144, num_blocks=0, horizon=2048, verify=5) == dflash_pv_bucket_ladder(
        262144, horizon=2048, verify=5
    )


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_set_covers_every_position_the_pool_can_hold():
    from models.demos.gemma4.tt.generator_vllm import dflash_width_set

    widths = dflash_width_set(262144, num_blocks=4096, horizon=2048, verify=5)
    dec = _decoder(widths)
    # Everything up to the last verify block's worth of the context is covered;
    # past that vLLM's own max_model_len stop arrives first.
    for pos in (0, 4096, 131072, 262144 - 6 - 64):
        assert dec.width_for(pos) is not None, pos


# ===== merged from tests/unit/test_drafter_snapshot_resolution.py: Drafter checkpoints resolve from the HF cache the RUNNER configured.
try:
    from models.demos.gemma4.tt import generator_vllm as gv
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _mk(root, repo):
    snap = os.path.join(root, repo, "snapshots", "abc123")
    os.makedirs(snap, exist_ok=True)
    return snap


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_hf_hub_cache_is_searched_before_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "mlperf" / "hub"))
    roots = gv._hf_hub_cache_dirs()
    assert roots[0] == str(tmp_path / "mlperf" / "hub")
    assert roots[-1] == os.path.expanduser("~/.cache/huggingface/hub")


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_hf_home_contributes_its_hub_subdirectory(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hfhome"))
    assert str(tmp_path / "hfhome" / "hub") in gv._hf_hub_cache_dirs()


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_roots_are_deduplicated(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "hub"))
    roots = gv._hf_hub_cache_dirs()
    assert len(roots) == len(set(roots))


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_dflash_drafter_is_found_in_the_configured_cache(monkeypatch, tmp_path):
    root = str(tmp_path / "mlperf" / "hub")
    snap = _mk(root, "models--z-lab--gemma-4-31B-it-DFlash")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    got = gv._dflash_default_snapshot()
    assert got is not None and got.rstrip("/") == snap


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_mtp_assistant_is_found_in_the_configured_cache(monkeypatch, tmp_path):
    root = str(tmp_path / "mlperf" / "hub")
    snap = _mk(root, "models--google--gemma-4-12B-it-assistant")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    got = gv._assistant_default_snapshot("google/gemma-4-12B-it")
    assert got.rstrip("/") == snap


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_missing_drafter_still_reports_absence(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty2"))
    # No assertion on the default root: a developer box may legitimately have
    # the drafter cached there. Absence is only asserted for the configured
    # roots, which is what the resolver adds.
    assert gv._hf_snapshot_glob("models--z-lab--does-not-exist") is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_read_only_cache_does_not_attempt_a_fetch(monkeypatch, tmp_path):
    """A :ro weights mount is cache-only.

    HF writes metadata on every HEAD call, so a fetch into a read-only mount
    fails with "Read-only file system" rather than populating anything. The
    caller reports the miss instead.
    """
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    try:
        monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(ro)])
        called = []
        monkeypatch.setattr(
            "huggingface_hub.snapshot_download",
            lambda *a, **k: called.append(k) or "/never",
        )
        assert gv._hf_resolve_repo("z-lab/nope", "models--z-lab--nope") is None
        assert called == []
    finally:
        ro.chmod(0o755)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_writable_cache_fetches_even_with_offline_set(monkeypatch, tmp_path):
    """Write access is the operator asking for the fetch.

    HF_HUB_OFFLINE is set for the read-only case; it must not veto a fetch into
    a cache the operator deliberately made writable, and it is restored after.

    Asserting the ENV VAR alone is not enough, and that is what let the real bug
    ship: huggingface_hub reads HF_HUB_OFFLINE once at import into
    constants.HF_HUB_OFFLINE, and every offline gate goes through
    constants.is_offline_mode(), which returns that module global. Clearing only
    the env var left the hub offline, so a :rw CI run still raised
    LocalEntryNotFoundError and the drafter never downloaded. Assert the gate the
    hub actually consults.
    """
    from huggingface_hub import constants as hf_constants

    root = tmp_path / "rw"
    root.mkdir()
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True)
    seen = {}

    def fake_download(*a, **k):
        seen["env_during_call"] = os.environ.get("HF_HUB_OFFLINE")
        seen["offline_mode_during_call"] = hf_constants.is_offline_mode()
        seen.update(k)
        return str(root / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._hf_resolve_repo("z-lab/x", "models--z-lab--x")
    assert got == str(root / "fetched")
    assert seen["env_during_call"] is None, "the env var is cleared for the call"
    assert seen["offline_mode_during_call"] is False, (
        "the hub's own offline gate must be off for the call, or the fetch raises "
        "LocalEntryNotFoundError however the env var is set"
    )
    assert os.environ.get("HF_HUB_OFFLINE") == "1", "env restored afterwards"
    assert hf_constants.HF_HUB_OFFLINE is True, "and the hub global restored too"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_cache_hit_never_reaches_the_hub(monkeypatch, tmp_path):
    root = str(tmp_path / "hub")
    snap = _mk(root, "models--z-lab--gemma-4-31B-it-DFlash")
    monkeypatch.setenv("HF_HUB_CACHE", root)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    called = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda *a, **k: called.append(k) or "/never",
    )
    assert gv._dflash_default_snapshot().rstrip("/") == snap
    assert called == []


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_online_miss_fetches_the_repo(monkeypatch, tmp_path):
    """Cache-then-hub, the way transformers resolves the target model."""
    # Pin the search roots: ~/.cache is the final fallback and a dev box may
    # genuinely have the drafter cached there, which would mask the fetch.
    (tmp_path / "empty").mkdir(exist_ok=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    asked = {}

    def fake_download(*a, **k):
        asked.update(k)
        return str(tmp_path / "fetched")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    got = gv._dflash_default_snapshot()
    assert got == str(tmp_path / "fetched")
    assert asked.get("repo_id") == "z-lab/gemma-4-31B-it-DFlash"


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_failed_fetch_is_reported_as_a_miss_not_an_exception(monkeypatch, tmp_path):
    (tmp_path / "empty").mkdir(exist_ok=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    def boom(*a, **k):
        raise OSError("no route to host")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    assert gv._dflash_default_snapshot() is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_model_weights_dir_is_used_when_it_exists(monkeypatch, tmp_path):
    """tt-inference-server exports MODEL_WEIGHTS_DIR; prefer it."""
    d = tmp_path / "weights"
    d.mkdir()
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(d))
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert gv._target_weights_dir() == str(d)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_repo_id_in_hf_model_resolves_through_the_hub_cache(monkeypatch, tmp_path):
    """tt-metal's vLLM CI sets HF_MODEL to a REPO ID and never sets
    MODEL_WEIGHTS_DIR, so the only way to the checkpoint is the hub cache.
    Relying on the env var alone built 'None/model.safetensors.index.json' and
    killed the engine during warmup after the drafter had downloaded fine."""
    root = tmp_path / "hub"
    snap = root / "models--google--gemma-4-31B-it" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(root)])
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    got = gv._target_weights_dir()
    assert got is not None, "a repo id must resolve through the hub cache"
    assert got.rstrip("/") == str(snap)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_local_hf_model_directory_is_used_directly(monkeypatch, tmp_path):
    d = tmp_path / "local-ckpt"
    d.mkdir()
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("HF_MODEL", str(d))
    assert gv._target_weights_dir() == str(d)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_nothing_resolvable_reports_absence(monkeypatch, tmp_path):
    """None must be returned so the caller can raise something readable,
    rather than formatting it into a path."""
    monkeypatch.setattr(gv, "_hf_hub_cache_dirs", lambda: [str(tmp_path / "empty")])
    monkeypatch.delenv("MODEL_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("GEMMA4_MODEL_PATH", raising=False)
    monkeypatch.setenv("HF_MODEL", "google/gemma-4-31B-it")
    assert gv._target_weights_dir() is None


# ===== merged from tests/unit/test_mtp_block_loop.py: Tests for the Gemma4 MTP decode block loop.
try:
    from models.demos.gemma4.tt.generator_vllm import Gemma4MTPForCausalLM as MTP
except ImportError:  # vLLM absent: this section's tests skip
    pass


class _StubSession:
    """Returns a scripted (committed, accepted_m) per ``serving_step`` call."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []
        self.released = False
        self.restaged = []

    def serving_step(self, cur_token, cur_pos):
        self.calls.append((cur_token, cur_pos))
        committed = self.scripted.pop(0)
        return committed, len(committed) - 1  # m = accepted, committed = m+1

    def serving_release(self):
        self.released = True

    def refresh_page_tables(self, page_table):
        """The decode path re-stages the captured page tables whenever vLLM's
        block table for the request changes; record the rows it was given."""
        self.restaged.append(page_table.clone())


class _Harness:
    """Minimal stand-in exposing only what ``decode_forward`` touches on the
    solo path with a live session."""

    _SPEC_N = 6
    _spec_budget_end = 10_000
    _spec_horizon = 2048  # only read by the horizon-exhausted log line

    def __init__(self, scripted, start_pos=100):
        self._spec = _StubSession(scripted)
        self._spec_cur = (11, start_pos)
        self._spec_pending = None  # no bootstrap on this path
        self._spec_pending_owner = None
        # No page table is passed on this path, so ownership is unknown on both
        # sides and the live session is served -- see _spec_active_is_mine.
        self._spec_active_owner = None
        self._eos = 1

    def _eos_fill_id(self):
        return self._eos

    # real implementations, so the tests exercise shipped logic
    _spec_pt_identity = staticmethod(MTP._spec_pt_identity)
    # The adaptive gate counts REAL requests, not the padded decode wire width
    # (the runner pads positions with -1); the stub drives it with one row.
    _spec_real_batch = staticmethod(MTP._spec_real_batch)
    _spec_pending_is_mine = MTP._spec_pending_is_mine
    _spec_active_is_mine = MTP._spec_active_is_mine
    run = MTP.decode_forward


def _row(h, tokens=None):
    """Drive just the block loop by calling decode_forward with a B=1 input."""
    toks = torch.tensor([[11]], dtype=torch.int32) if tokens is None else tokens
    return h.run(tokens=toks, start_pos=torch.tensor([[100]], dtype=torch.int32))


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_short_iterations_are_concatenated_to_a_full_row():
    """Three 2-token iterations must produce ONE 6-wide row, not a padded 2."""
    h = _Harness([[21, 22], [23, 24], [25, 26]])
    out = _row(h)
    assert out.shape == (1, 6)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert len(h._spec.calls) == 3  # looped rather than padding after one


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_position_advances_by_tokens_emitted():
    """The session must resume from what was EMITTED. Iterations are driven from
    (token, pos), so a wrong position re-drafts the wrong prefix."""
    h = _Harness([[21, 22], [23, 24], [25, 26]], start_pos=100)
    _row(h)
    tok, pos = h._spec_cur
    assert pos == 100 + 6
    assert tok == 26
    # each iteration was driven from the previous one's last token / next pos
    assert h._spec.calls == [(11, 100), (22, 102), (24, 104)]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_overshoot_truncates_and_resumes_from_emitted():
    """A final iteration may overshoot N. The extra tokens are dropped and the
    position reflects the row, so the next step re-drafts and overwrites them."""
    h = _Harness([[21, 22, 23, 24], [25, 26, 27, 28]], start_pos=200)
    out = _row(h)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    tok, pos = h._spec_cur
    assert pos == 200 + 6 and tok == 26


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_iteration_needs_no_loop():
    """High acceptance fills the row in one call; do not spin further."""
    h = _Harness([[21, 22, 23, 24, 25, 26]])
    out = _row(h)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert len(h._spec.calls) == 1


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_eos_stops_the_loop_and_fills_the_tail():
    """A genuine stop is the ONLY case that may leave a short row; the tail is
    EOS so upstream trims there."""
    h = _Harness([[21, 1], [99, 99]])  # eos id is 1
    out = _row(h)
    assert out[0].tolist()[:2] == [21, 1]
    assert out[0].tolist()[2:] == [1, 1, 1, 1]  # EOS-filled tail
    assert len(h._spec.calls) == 1  # stopped, did not consume the next script


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_horizon_exhaustion_emits_eos_without_stepping():
    """Past the horizon the packed verify would attend past its capture, so the
    request ends cleanly instead of stepping again."""
    h = _Harness([[21, 22]])
    h._spec_budget_end = 100  # cur_pos starts at 100
    out = _row(h)
    assert out[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert h._spec.calls == []


def _live_session_instance(monkeypatch, scripted, owner, baseline="baseline-out"):
    """A real Gemma4MTPForCausalLM with a live session and no device.

    ``__new__`` skips the device-bound ``__init__``; the attributes below are
    everything ``decode_forward`` reads on the solo path. The BASE class's
    ``decode_forward`` is the plain-baseline fallback the ownership check hands
    off to, so it is stubbed to a sentinel -- the stub harness above cannot
    reach it at all (zero-arg ``super()`` needs a real instance).
    """
    from types import SimpleNamespace

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    monkeypatch.setattr(Gemma4ForCausalLM, "decode_forward", lambda self, *a, **k: baseline)
    h = MTP.__new__(MTP)
    h._spec = _StubSession(scripted)
    h._spec_cur = (11, 100)
    h._spec_pending = None
    h._spec_pending_owner = None
    h._spec_active_owner = owner
    h._spec_budget_end = 10_000
    h._spec_horizon = 2048
    h._spec_first_step = False
    h._spec_last_pt = None  # nothing staged yet -> the first step stages
    h._bounded_sliding_kv_cache = False
    h._spec_warm = False  # this harness drives a session directly, not via warmup
    h.model = [SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=1))]
    return h


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_live_session_is_refused_for_a_request_that_does_not_own_it(monkeypatch):
    """A solo decode step can be scheduled for a request that is NOT the live
    session's owner: async scheduling skips a request that has reached
    max_tokens (upstream guards that skip on num_output_placeholders), so the
    owner can drop out of a step while its session is still armed. Serving the
    block loop here emits the OWNER's speculation for this request, against the
    single placeholder the scheduler reserved for a non-owner. The session must
    be released and the step served as plain baseline instead.
    """
    pt = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    other = ("not-this-request",)
    h = _live_session_instance(monkeypatch, [[21, 22, 23, 24, 25, 26]], other)
    session = h._spec

    out = h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        page_table=pt,
    )

    assert out == "baseline-out"
    assert h._spec is None  # released
    assert session.calls == []  # never speculated for the non-owner
    assert h._spec_active_owner is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_live_session_is_served_for_its_own_owner(monkeypatch):
    """The companion: a matching identity keeps the session and blocks."""
    pt = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    h = _live_session_instance(monkeypatch, [[21, 22, 23, 24, 25, 26]], MTP._spec_pt_identity(pt))

    out = h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        page_table=pt,
    )

    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert h._spec is not None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_page_tables_are_restaged_when_the_block_table_changes(monkeypatch):
    """vLLM allocates a KV block only every block_size tokens, so a session
    captured at prefill holds the prompt's blocks and zeros past them. The
    fused verify binds its page tables at capture, so without a re-stage it
    reads AND writes the null block once generation crosses out of the
    prompt's last block (tt-metal#55548 D2 on the qwen36 MTP trace; here it
    also reaches the host map behind the per-iteration hot-block uploads).
    """
    pt = torch.tensor([[3, 4, 5, 0]], dtype=torch.int32)
    h = _live_session_instance(monkeypatch, [[21, 22, 23, 24, 25, 26]] * 4, None)
    session = h._spec

    h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        page_table=pt,
    )
    assert len(session.restaged) == 1  # first step stages the current table

    # Same table on the next step: nothing to do.
    h.decode_forward(
        tokens=torch.tensor([[26]], dtype=torch.int32),
        start_pos=torch.tensor([[106]], dtype=torch.int32),
        page_table=pt,
    )
    assert len(session.restaged) == 1

    # A newly allocated block appears -> re-stage.
    grown = torch.tensor([[3, 4, 5, 9]], dtype=torch.int32)
    h.decode_forward(
        tokens=torch.tensor([[26]], dtype=torch.int32),
        start_pos=torch.tensor([[112]], dtype=torch.int32),
        page_table=grown,
    )
    assert len(session.restaged) == 2
    assert session.restaged[-1].reshape(-1).tolist() == [3, 4, 5, 9]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_warm_capture_with_no_active_request_serves_baseline(monkeypatch):
    """The batch-to-solo transition with _spec_warm=True.

    A warm release deliberately KEEPS the capture -- the warmup traces are a
    process artifact, and freeing them would leave no captured widths and no
    way to recapture -- and nulls the active request only. Gating on `_spec`
    alone therefore fell through to `cur_token, cur_pos = self._spec_cur` and
    raised TypeError on an ordinary two-requests-to-one transition.
    """
    pt = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    owner = tuple(pt.reshape(-1).tolist())
    h = _live_session_instance(monkeypatch, [[21, 22, 23]], owner)
    h._spec_warm = True

    # Step 1: two real requests -> release the session, keep the warm capture.
    batched = h.decode_forward(
        tokens=torch.tensor([[11], [12]], dtype=torch.int32),
        start_pos=torch.tensor([[100], [100]], dtype=torch.int32),
        page_table=pt,
    )
    assert batched == "baseline-out"
    assert h._spec is not None, "a warm capture must survive the release"
    assert h._spec_cur is None, "but no request owns it any more"

    # Step 2: back to one request, no new prefill -> baseline, not TypeError.
    solo = h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[101]], dtype=torch.int32),
        page_table=pt,
    )
    assert solo == "baseline-out"


def _warmup_stub(monkeypatch, tmp_path, assistant_env=None, cache_hit=False):
    """An MTP instance whose warmup can be driven host-only.

    ``__new__`` skips the device-bound ``__init__``. ``_spec_warmup_session`` is
    stubbed to succeed, so anything that raises came from the pre-flight and not
    from a capture failure.
    """
    monkeypatch.delenv("GEMMA4_ASSISTANT_MODEL", raising=False)
    if assistant_env is not None:
        monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_env)
    # HF_MODEL drives the fallback candidate path f"{HF_MODEL}-assistant".
    monkeypatch.setenv("HF_MODEL", str(tmp_path / "gemma-4-12B-it"))
    # This box HAS the assistant in its real HF cache, so without pinning the
    # resolver the "missing" case silently resolves and the test proves nothing.
    import models.demos.gemma4.tt.generator_vllm as gv

    monkeypatch.setattr(
        gv, "_hf_resolve_repo", lambda repo, cache_dir_name: (str(tmp_path / "cached-assistant") if cache_hit else None)
    )
    if cache_hit:
        (tmp_path / "cached-assistant").mkdir(exist_ok=True)
    h = MTP.__new__(MTP)
    h._spec = None
    h._spec_warm = False
    h._decode_warmup_complete = False
    h._mtp_warmup_capture = True  # the default path, and the one CI uses
    monkeypatch.setattr(MTP, "_spec_warmup_session", lambda self, kv, nb: None)
    monkeypatch.setattr(MTP, "_spec_release_session", lambda self, force=False: None)
    return h


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_missing_assistant_fails_warmup_instead_of_the_first_request(monkeypatch, tmp_path, expect_error):
    """The batched benchmark never touches the drafter, so a server that boots
    without one passes every throughput check and then dies on the first SOLO
    request (EngineDeadError 500 at the coherence guard). Fail while starting."""
    h = _warmup_stub(monkeypatch, tmp_path)
    with expect_error(RuntimeError, "snapshot not found -- resolved"):
        h.warmup_model_decode(kv_cache=[[object()]], num_blocks=8, enable_trace=True)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_operator_supplied_assistant_path_is_left_alone(monkeypatch, tmp_path):
    """An explicit GEMMA4_ASSISTANT_MODEL is the operator's business: the
    pre-flight must not second-guess it (it may be a repo id, not a dir)."""
    h = _warmup_stub(monkeypatch, tmp_path, assistant_env="google/gemma-4-12B-it-assistant")
    h.warmup_model_decode(kv_cache=[[object()]], num_blocks=8, enable_trace=True)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_existing_assistant_dir_passes_the_preflight(monkeypatch, tmp_path):
    """The resolver finds a cached snapshot, so warmup proceeds to capture."""
    h = _warmup_stub(monkeypatch, tmp_path, cache_hit=True)
    h.warmup_model_decode(kv_cache=[[object()]], num_blocks=8, enable_trace=True)


# ===== merged from tests/unit/test_mtp_width_ladder.py: Tests for the MTP serving verify-width ladder and its per-step selection.
try:
    from models.demos.gemma4.tt.generator_vllm import mtp_pv_width_ladder
except ImportError:  # vLLM absent: this section's tests skip
    pass


def _session(widths, ladder=None, draft_len=5):
    """A decoder carrying a captured width set and nothing else."""
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec.draft_len = draft_len
    spec._srv_widths = {int(w): {"pv_S_k": int(w)} for w in widths}
    spec._srv_ladder = sorted(int(w) for w in (ladder if ladder is not None else widths))
    return spec


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_ladder_doubles_and_covers_the_whole_context():
    ladder = mtp_pv_width_ladder(262144, draft_len=5)
    assert ladder == sorted(ladder)
    assert ladder[0] == 1024
    assert ladder[-1] >= 262144 + 7
    # doubling -> logarithmic rung count
    assert len(ladder) <= 12
    for a, b in zip(ladder, ladder[1:]):
        assert b >= 2 * a or b == ladder[-1]


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_every_position_in_the_context_has_a_rung():
    ladder = mtp_pv_width_ladder(262144, draft_len=5)
    spec = _session(ladder)
    for pos in (0, 1, 1000, 2048, 4096, 131072, 262143):
        assert spec.srv_width_for(pos) is not None, pos


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_selection_is_the_narrowest_covering_width():
    """Width is per-iteration BANDWIDTH here -- the masks are rebuilt on the
    host and uploaded at [1, 1, P_v, S_k] every replay -- so a request must not
    sit on a wider trace than its position needs."""
    spec = _session([1024, 2048, 4096], draft_len=5)
    assert spec.srv_width_for(0) == 1024
    assert spec.srv_width_for(1024 - 7) == 1024
    assert spec.srv_width_for(1024 - 6) == 2048
    assert spec.srv_width_for(2048 - 6) == 4096


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_position_past_the_captured_set_is_none_so_the_caller_can_capture():
    """None is the signal to take the next rung off the ladder, which is what
    replaces ending the request at the horizon."""
    spec = _session([1024], ladder=[1024, 2048, 4096], draft_len=5)
    assert spec.srv_width_for(2000) is None
    need = 2000 + spec.draft_len + 2  # 2007
    assert next(r for r in spec._srv_ladder if r >= need) == 2048
    # and a position past that rung takes the one after it
    assert next(r for r in spec._srv_ladder if r >= 3000 + spec.draft_len + 2) == 4096


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_no_captured_widths_means_no_width_set():
    """A session set up the old way (single capture) must not take the width
    path: srv_width_for reports None and serving_step leaves it alone."""
    spec = SpeculativeDecoder.__new__(SpeculativeDecoder)
    spec.draft_len = 5
    spec._srv_widths = None
    assert spec.srv_width_for(0) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("draft_len", [3, 5, 7])
def test_tail_follows_the_draft_length(draft_len):
    """A verify block is P_v = K + 1 rows at pos..pos+K, so the width has to
    cover K + 2 past the position, not just the position."""
    ladder = mtp_pv_width_ladder(8192, draft_len=draft_len)
    spec = _session(ladder, draft_len=draft_len)
    top = max(ladder)
    assert spec.srv_width_for(top - draft_len - 2) == top
    assert spec.srv_width_for(top - draft_len - 1) is None


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_cap_keeps_the_largest_rungs_mtp_width_ladder():
    ladder = mtp_pv_width_ladder(262144, draft_len=5, max_rungs=4)
    assert len(ladder) == 4
    assert ladder[-1] == max(mtp_pv_width_ladder(262144, draft_len=5))


# ===== merged from tests/unit/test_spec_decode_no_unbound_names.py: No method in the speculative serving path reads a name it never binds.
SOURCE = Path(__file__).resolve().parents[2] / "tt" / "spec_decode.py"


def _module():
    return ast.parse(SOURCE.read_text())


def _bound_at_module_level(tree):
    names = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
    return names


def _free_names(fn, module_names):
    """Names a function reads without binding them anywhere in its own body."""
    bound = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
    if fn.args.vararg:
        bound.add(fn.args.vararg.arg)
    if fn.args.kwarg:
        bound.add(fn.args.kwarg.arg)
    loads = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else loads).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
    return loads - bound - module_names


def _functions(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def test_srv_upload_iter_binds_every_name_it_reads():
    """The method that actually regressed, named so the failure is unmistakable."""
    tree = _module()
    module_names = _bound_at_module_level(tree)
    target = [fn for fn in _functions(tree) if fn.name == "_srv_upload_iter"]
    assert target, "_srv_upload_iter not found in spec_decode.py"
    assert _free_names(target[0], module_names) == set()


@pytest.mark.parametrize(
    "name",
    [
        "serving_step",
        "serving_reseed",
        "serving_setup",
        "serving_warmup_widths",
        "_srv_upload_iter",
        "_srv_capture_width",
        "srv_width_for",
        "refresh_page_tables",
    ],
)
def test_serving_path_methods_bind_every_name_they_read(name):
    tree = _module()
    module_names = _bound_at_module_level(tree)
    for fn in _functions(tree):
        if fn.name == name:
            free = _free_names(fn, module_names)
            assert free == set(), f"{name} reads unbound name(s): {sorted(free)}"
            return
    pytest.skip(f"{name} is not defined in spec_decode.py")


# ===== merged from tests/unit/test_spec_draft_len.py: Host-only tests for the pure speculative draft-length policies.
@pytest.fixture()
def _no_draft_len_env(monkeypatch):
    monkeypatch.delenv("GEMMA4_SPEC_DRAFT_LEN", raising=False)


@pytest.mark.parametrize(
    ("prompt_len", "expected"),
    [
        (None, 3),  # unknown prompt -> short default
        (128, 3),
        (1023, 3),  # boundary: strictly below the knee
        (1024, 5),  # boundary: at the knee
        (131072, 5),
    ],
)
@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_prompt_boundary(prompt_len, expected):
    assert auto_draft_len(prompt_len) == expected


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_env_override_wins(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "7")
    assert auto_draft_len(128) == 7
    assert auto_draft_len(131072) == 7


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_env_auto_defers_to_policy(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "auto")
    assert auto_draft_len(128) == 3
    assert auto_draft_len(4096) == 5


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_custom_default():
    assert auto_draft_len(None, default=2) == 2
    assert auto_draft_len(64, default=2) == 2


@pytest.mark.parametrize(
    ("batch", "prompt_len", "expected"),
    [
        (None, 4096, 5),  # no batch -> plain policy
        (1, 4096, 5),  # B=1 -> plain policy
        (8, 128, 3),  # short prompt K=3 within the 32-row cap (8*(3+1)=32)
        (8, 4096, 3),  # context K=5 CLIPPED to the row cap 32//8-1=3
        (16, 4096, 1),  # row cap 32//16-1=1
        (32, 4096, 0),  # past the compute knee: speculation OFF
        (33, 128, 0),  # row cap goes negative -> OFF, never negative K
    ],
)
@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_batched_row_cap(batch, prompt_len, expected):
    assert auto_draft_len_batched(prompt_len, batch) == expected


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_batched_env_override_wins(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "2")
    assert auto_draft_len_batched(4096, 32) == 2  # explicit K bypasses the cap


@pytest.mark.usefixtures("_no_draft_len_env")
def test_auto_draft_len_batched_env_auto_defers_to_policy(monkeypatch):
    monkeypatch.setenv("GEMMA4_SPEC_DRAFT_LEN", "auto")
    assert auto_draft_len_batched(4096, 8) == 3
    assert auto_draft_len_batched(4096, 32) == 0


# ===== merged from tests/unit/test_spec_plan.py: Tests for the model side of the plugin speculative contract (``spec_plan``).
try:
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashForCausalLM
except ImportError:  # vLLM absent: this section's tests skip
    pass


_VERIFY = 5


@pytest.fixture
def drafter_snapshot(tmp_path, monkeypatch):
    """A minimal drafter checkpoint dir — only ``config.json`` is read."""
    cfg = {
        "num_hidden_layers": 5,
        "hidden_size": 5376,
        "head_dim": 128,
        "num_key_value_heads": 8,
        "block_size": 16,  # the checkpoint's block geometry; spec_plan reads it (GEMMA4_DFLASH_BLOCK overrides)
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", str(tmp_path))
    monkeypatch.setenv("GEMMA4_DFLASH_VERIFY", str(_VERIFY))
    monkeypatch.setenv("MESH_DEVICE", "P150x8")
    monkeypatch.delenv("GEMMA4_DFLASH_REPLICATED", raising=False)
    monkeypatch.delenv("GEMMA4_DFLASH_CTX_CAP", raising=False)
    return tmp_path, cfg


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_rejects_concurrency_above_one(drafter_snapshot):
    """dFlash captures its fused verify at B=1, so B>1 cannot speculate."""
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=2, requested_k=_VERIFY)
    assert isinstance(out, SpecReject)
    assert "single-stream" in out.reason
    # Nothing is speculable at this concurrency, so no K is offered.
    assert out.supported_k == ()


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_k_is_a_set_not_a_range(drafter_snapshot):
    """A narrower K would need its own packed-verify capture, so it is refused
    and the one supported width is named."""
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY - 2)
    assert isinstance(out, SpecReject)
    assert out.supported_k == (_VERIFY,)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_missing_drafter_is_refused_at_config_time(drafter_snapshot, monkeypatch):
    """Without this the drafter resolves lazily and a missing checkpoint kills
    the engine at the FIRST REQUEST instead (EngineDeadError, no traceback)."""
    monkeypatch.setenv("GEMMA4_DFLASH_DRAFTER", "/nonexistent/dflash-drafter")
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert isinstance(out, SpecReject)
    assert out.supported_k == (_VERIFY,)
    # The message has to be actionable: it is surfaced verbatim to the operator.
    assert "GEMMA4_DFLASH_DRAFTER" in out.reason


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_plan_fields(drafter_snapshot):
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert isinstance(out, SpecPlan)
    assert out.effective_k == _VERIFY
    assert out.lanes_per_request == 1
    # The drafter's context cache is a fixed window, not a per-token paged KV.
    assert out.extra_bytes_per_token == 0
    assert out.accept_modes == ("argmax_ids",)
    assert out.drafter_state == "internal"
    assert out.drafter_target_cache_requires == ()


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_effective_k_never_exceeds_requested(drafter_snapshot):
    """#110: ``effective_k <= requested_k``. The plugin publishes it back to
    ``speculative_config.num_speculative_tokens``, which upstream's scheduler
    reads to size its lookahead reservation."""
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY + 6)
    assert isinstance(out, SpecPlan)
    assert out.effective_k <= _VERIFY + 6


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_fixed_cost_tracks_the_decoder_allocations(drafter_snapshot):
    """``extra_bytes_per_seq`` must be the real per-chip cost: under-reporting
    under-reserves by the whole drafter cache (#110 s.2).

    Mirrors ``DFlashFusedDecoder.__init__``: ctx_dev, ctx_k/ctx_v, ctx_pos,
    fc_prev, commit_pos — bf16 except the int64 position vectors.
    """
    _, cfg = drafter_snapshot
    cap, p_v, tp = 2048, _VERIFY + 1, 8
    local_kv = cfg["num_key_value_heads"] // tp
    expected = (
        cap * cfg["hidden_size"] * 2
        + 2 * cfg["num_hidden_layers"] * local_kv * cap * cfg["head_dim"] * 2
        + cap * 8
        + p_v * cfg["hidden_size"] * 2
        + p_v * 8
    )
    out = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert out.extra_bytes_per_seq == expected


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unparseable_mesh_over_reserves_rather_than_under(drafter_snapshot, monkeypatch):
    """Galaxy DP entries set no MESH_DEVICE. Falling back to tp=1 yields the
    LARGEST per-chip cache, so the plugin over-reserves; the failure the
    contract warns about is under-reserving."""
    sharded = Gemma4DFlashForCausalLM.spec_plan(None, 1, _VERIFY).extra_bytes_per_seq
    monkeypatch.delenv("MESH_DEVICE", raising=False)
    unknown_mesh = Gemma4DFlashForCausalLM.spec_plan(None, 1, _VERIFY).extra_bytes_per_seq
    assert unknown_mesh > sharded


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_is_pure_and_frozen(drafter_snapshot, expect_error):
    """Called several times per launch; must be side-effect free and stable."""
    env_before = dict(os.environ)
    first = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    second = Gemma4DFlashForCausalLM.spec_plan(None, max_num_seqs=1, requested_k=_VERIFY)
    assert first == second
    assert dict(os.environ) == env_before
    with expect_error(AttributeError, "effective_k"):
        first.effective_k = 99  # frozen dataclass


# ===== merged from tests/unit/test_spec_ring_guard.py: The bounded ring is sized for the verify, and refused when it cannot hold it.
try:
    from models.demos.gemma4.tt.generator_vllm import (
        _auto_size_spec_ring,
        _reserve_spec_ring_headroom,
        _spec_ring_blocks_for,
    )
except ImportError:  # vLLM absent: this section's tests skip
    pass


WINDOW = 1024


SHIPPED_WIDTH = 6  # GEMMA4_DFLASH_VERIFY=5 -> P_v = K + 1


@pytest.fixture()
def _clean_env():
    """Restore the headroom env around every test.

    ``_auto_size_spec_ring`` writes ``os.environ`` directly, and monkeypatch
    only restores what monkeypatch itself changed -- ``delenv`` on an unset var
    records nothing. Without this, a sizing test leaks the headroom into every
    later test in the process and silently changes ``bounded_ring_modulo`` for
    the device suites.
    """
    saved = os.environ.get(SPEC_RING_HEADROOM_ENV)
    os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
    yield
    if saved is None:
        os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
    else:
        os.environ[SPEC_RING_HEADROOM_ENV] = saved


def _cls(width, *, declines=False):
    return SimpleNamespace(_SPEC_N=width, __name__="FakeSpecClass", _SPEC_DECLINES_AT_RING_WRAP=declines)


def _cfg(window=WINDOW):
    return SimpleNamespace(text_config=SimpleNamespace(sliding_window=window))


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_speculating_class_gets_a_ring_that_holds_its_drafts():
    """The shipped 31B dFlash / 12B MTP case: 1024 window, K=5 -> ring 2048."""
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=True)
    assert bounded_ring_modulo(WINDOW) == 2048
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")  # must not raise


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_ring_is_sized_from_drafts_not_from_the_anchor():
    """K drafts need K slots, not K + 1.

    The anchor write at p lands in slot p % ring, which held p - ring -- already
    outside the window -- so only the drafts evict live history. This is the same
    test the contract rail applies per request (``_dflash_ring_advice``).

    The shipped 1024 window cannot show the difference (1024 + 5 and 1024 + 6
    both round to 2048), so this uses a window where counting the anchor crosses
    a power of two: 3776 + 320 IS 4096, while 3776 + 321 forces 8192. Counting
    one row too many would double the bounded pool for every sliding layer.
    """
    assert _spec_ring_blocks_for(3776, 320) == (4096 - 3776) // 64
    assert _spec_ring_blocks_for(3776, 321) == (8192 - 3776) // 64
    # The shipped case, for the record: 1024 window, K=5 -> 2048 ring -> 16 blocks.
    assert _spec_ring_blocks_for(WINDOW, 5) == 16


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_baseline_class_is_left_alone():
    """No _SPEC_N: nothing speculates, so the exact-window ring is correct."""
    _auto_size_spec_ring(SimpleNamespace(__name__="Gemma4ForCausalLM"), _cfg(), bounded_sliding=True)
    assert bounded_ring_modulo(WINDOW) == WINDOW


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unbounded_serving_is_left_alone():
    """Without bounded sliding there is no ring to size."""
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=False)
    assert SPEC_RING_HEADROOM_ENV not in os.environ


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_attention_models_are_left_alone():
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(window=None), bounded_sliding=True)
    assert SPEC_RING_HEADROOM_ENV not in os.environ


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_operator_value_wins(monkeypatch):
    """A pinned value is not overwritten -- it is checked instead."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "16")
    _auto_size_spec_ring(_cls(SHIPPED_WIDTH), _cfg(), bounded_sliding=True)
    assert os.environ[SPEC_RING_HEADROOM_ENV] == "16"


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_exact_window_ring_is_refused(expect_error):
    with expect_error(RuntimeError, "cannot hold a verify"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_refusal_names_the_remedy_and_the_issue():
    try:
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("an exact-window ring must not be accepted")
    assert SPEC_RING_HEADROOM_ENV in message, "the remedy must be named"
    assert "57701" in message, "the issue must be findable from the failure"
    assert "16" in message, "the working value must be given, not just described"


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_a_class_that_declines_at_the_wrap_is_not_refused():
    """The contract rail stops proposing before the wrap and carries on as plain
    decode, so an exact ring costs it speed, not correctness."""
    _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlashContract", declines_at_wrap=True)


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_width_one_needs_no_headroom():
    """Width 1 writes only the anchor, so it evicts nothing in the live window."""
    _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_an_invalid_ring_is_reported_whatever_the_width(monkeypatch, expect_error):
    """The ring is validated before the width check.

    Width 1 needs no headroom, but a ring that is not a power of two is wrong for
    any width. Validating after the width check would have made this guard's
    error path depend on verify_width, leaving a bad env value to surface later
    from the model or trace path instead of at config time.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")  # 1024 + 32*64 = 3072
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, 1, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_attention_is_unaffected_by_the_guard():
    _reserve_spec_ring_headroom(None, SHIPPED_WIDTH, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_check_is_on_the_ring_not_on_the_env_var(monkeypatch, expect_error):
    """Headroom set too small is still unsafe.

    The first version of this guard returned as soon as the env var was present,
    so an under-sized headroom passed. A 1024 window with 32 blocks is a 3072
    ring, not a power of two, so the ring builder rejects it rather than the
    guard silently accepting a value that only LOOKS configured.
    """
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, "32")
    with expect_error(ValueError, "power of two"):
        _reserve_spec_ring_headroom(WINDOW, SHIPPED_WIDTH, "Gemma4DFlash")


@pytest.mark.usefixtures("_clean_env")
@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_every_real_speculating_class_gets_sized():
    """The three shipped speculating classes, by their real _SPEC_N.

    The fakes above pin the arithmetic; this pins the wiring. A new serving
    class that forgets ``_SPEC_N``, or one whose K changes, silently falls back
    to the exact-window ring -- which is the #57701 defect, and is invisible
    until someone reads generated text past the ring.
    """
    from models.demos.gemma4.tt.generator_vllm import (
        Gemma4DFlashContractForCausalLM,
        Gemma4DFlashForCausalLM,
        Gemma4MTPForCausalLM,
    )

    for cls in (Gemma4DFlashForCausalLM, Gemma4DFlashContractForCausalLM, Gemma4MTPForCausalLM):
        os.environ.pop(SPEC_RING_HEADROOM_ENV, None)
        assert int(cls._SPEC_N) > 1, f"{cls.__name__} must declare a verify width"
        _auto_size_spec_ring(cls, _cfg(), bounded_sliding=True)
        ring = bounded_ring_modulo(WINDOW)
        assert ring - WINDOW >= cls._SPEC_N - 1, f"{cls.__name__}: ring {ring} cannot hold its drafts"
        # and the guard agrees, including for the rail that declines at the wrap
        _reserve_spec_ring_headroom(WINDOW, cls._SPEC_N, cls.__name__, declines_at_wrap=cls._SPEC_DECLINES_AT_RING_WRAP)


# ===== merged from tests/unit/test_adaptive_real_batch.py: The adaptive gate must count REQUESTS, not the padded decode wire width.
try:
    from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashForCausalLM as DF
except ImportError:  # vLLM absent: this section's tests skip
    pass


if _HAS_VLLM:
    real_batch = DF._spec_real_batch


def _toks(n):
    return torch.zeros(n, 1, dtype=torch.int32)


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_solo_request_padded_to_a_bucket_is_still_solo():
    """The Galaxy case: one request, wire width 32."""
    start = torch.cat([torch.tensor([121004]), torch.full((31,), -1)]).to(torch.int32)
    assert real_batch(_toks(32), start) == 1


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unpadded_solo_is_solo():
    assert real_batch(_toks(1), torch.tensor([512], dtype=torch.int32)) == 1


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
@pytest.mark.parametrize("live", [2, 5, 31, 32])
def test_partially_filled_batch_counts_live_rows(live):
    start = torch.cat([torch.arange(live), torch.full((32 - live,), -1)]).to(torch.int32)
    assert real_batch(_toks(32), start) == live


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_full_batch_has_no_padding():
    assert real_batch(_toks(32), torch.arange(32, dtype=torch.int32)) == 32


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_position_zero_is_a_real_row_not_padding():
    """-1 is the pad sentinel; 0 is a legitimate position and must not be
    mistaken for one."""
    start = torch.cat([torch.tensor([0, 0]), torch.full((30,), -1)]).to(torch.int32)
    assert real_batch(_toks(32), start) == 2


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_two_dimensional_positions_are_flattened():
    start = torch.cat([torch.tensor([7]), torch.full((7,), -1)]).to(torch.int32).reshape(8, 1)
    assert real_batch(_toks(8), start) == 1


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_missing_positions_fall_back_to_the_wire_width():
    """Nothing to count: keep the old behaviour rather than guessing solo."""
    assert real_batch(_toks(4), None) == 4


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_all_rows_padded_never_reports_zero():
    """A zero would make the gate divide-by-zero downstream; the floor is 1."""
    assert real_batch(_toks(8), torch.full((8,), -1, dtype=torch.int32)) == 1


# ===== merged from tests/unit/test_native_context_guard.py: Native-context guard: the serving templates waive vLLM's max_model_len check, so gemma4 must refuse a
try:
    from models.demos.gemma4.tt.generator_vllm import _assert_within_native_context
except ImportError:  # vLLM absent: this section's tests skip
    pass


expect_error_native_context_guard = pytest.fixture(lambda: make_expect_error())


NATIVE = 262_144


def _args(native=NATIVE):
    return SimpleNamespace(_hf_text_config=SimpleNamespace(max_position_embeddings=native))


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_at_native_passes(monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(NATIVE, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_below_native_passes(monkeypatch):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(131_072, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_one_past_native_raises(monkeypatch, expect_error_native_context_guard):
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    with expect_error_native_context_guard(ValueError, "native max_position_embeddings"):
        _assert_within_native_context(NATIVE + 1, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_the_scenario_from_the_report_raises(monkeypatch, expect_error_native_context_guard):
    # An operator writes max_context: 300000; the pool convention follows it.
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    with expect_error_native_context_guard(ValueError, "VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
        _assert_within_native_context(300_000, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_explicit_env_waiver_passes(monkeypatch):
    monkeypatch.setenv("GEMMA4_ALLOW_BEYOND_NATIVE", "1")
    _assert_within_native_context(300_000, _args())


@pytest.mark.skipif(not _HAS_VLLM, reason="needs vLLM (vllm_tt_plugin)")
def test_unknown_native_stays_permissive(monkeypatch):
    # A model_args without the attribute (older adaptors) must not brick boots.
    monkeypatch.delenv("GEMMA4_ALLOW_BEYOND_NATIVE", raising=False)
    _assert_within_native_context(300_000, SimpleNamespace())
