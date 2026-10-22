# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""dFlash warms prefill eagerly: it never replays a prefill trace (tt-metal#57853).

Every prefill on the dFlash rails runs untraced at runtime -- the drafter reads
residual taps from a python hook that a traced replay does not run. The
inherited warmup only cleared ``enable_trace`` for bounded sliding, so an
unbounded dFlash server captured every prefill bucket at warmup and then never
replayed one.
"""

import pytest

# generator_vllm imports vllm at module scope (through tt_transformers), so
# COLLECTING this file fails on a runner without vLLM -- the tt-metal unit job.
pytest.importorskip("vllm")

from models.demos.gemma4.tt.generator_vllm import (
    Gemma4DFlashContractForCausalLM,
    Gemma4DFlashForCausalLM,
    Gemma4ForCausalLM,
    Gemma4MTPForCausalLM,
)


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
