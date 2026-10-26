# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative decoding needs the decode traces (no device; needs vllm-tt-plugin): the verify and the MTP draft run
only as traces, which warmup_model_decode captures with the decode trace. The vLLM adapter's spec_plan refuses a
launch whose tt options capture none (trace_mode "none", or no model warmup), and a draft before the capture raises
instead of reading a trace that does not exist."""

from types import SimpleNamespace

import pytest

from models.experimental.mimo_v2_6_flash.reference.spec_decode import MAX_DRAFTS, DraftSource, plan_mtp
from models.experimental.mimo_v2_6_flash.tt import generator_vllm
from models.experimental.mimo_v2_6_flash.tt.generator_vllm import MiMoV2ForCausalLM
from models.experimental.mimo_v2_6_flash.tt.mtp import TtMiMoMTP

pytest.importorskip("vllm_tt_plugin")
from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject  # noqa: E402


def launch(additional_config):
    """The vllm_config fields spec_plan reads."""
    return SimpleNamespace(additional_config=additional_config)


@pytest.mark.parametrize(
    "additional_config",
    [None, {}, {"tt": {"trace_mode": "all"}}, {"tt": {"trace_mode": "decode_only", "enable_model_warmup": True}}],
)
def test_a_launch_that_captures_decode_traces_speculates(monkeypatch, additional_config):
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    plan = MiMoV2ForCausalLM.spec_plan(launch(additional_config), 16, MAX_DRAFTS)
    assert isinstance(plan, SpecPlan) and plan.effective_k == MAX_DRAFTS


@pytest.mark.parametrize(
    "tt",
    [
        {"trace_mode": "none"},
        {"enable_model_warmup": False},
        {"trace_mode": "decode_only", "enable_model_warmup": False},
    ],
)
def test_a_launch_without_decode_traces_is_refused(monkeypatch, tt):
    """No draft length is served: the plugin raises at configuration time with the reason."""
    monkeypatch.setattr(generator_vllm, "SPEC_DECODE", "mtp")
    out = MiMoV2ForCausalLM.spec_plan(launch({"tt": tt}), 16, MAX_DRAFTS)
    assert isinstance(out, SpecReject) and out.supported_k == ()
    assert "trace_mode" in out.reason and "enable_model_warmup" in out.reason


@pytest.mark.parametrize("step", ["backfill", "draft"])
def test_a_draft_before_the_capture_raises(expect_error, step):
    """What propose_draft_tokens enqueues after a plain decode step of one request (slot 2, position 17), on a replica
    whose warmup captured no draft trace: the backfill (MIMO_MTP_BACKFILL, the first draft after a prompt), then the
    draft."""
    mtp = TtMiMoMTP.__new__(TtMiMoMTP)
    mtp.trace = mtp.backfill_trace = None
    plan = plan_mtp(DraftSource(row=2, slots=[2]), [17] + [-1] * MAX_DRAFTS, 1)
    with expect_error(RuntimeError, "before warmup_model_decode captured the draft trace"):
        if step == "backfill":
            mtp.enqueue_backfill(2, 17)
        else:
            mtp.enqueue([0, 0, 5, 0], plan)
