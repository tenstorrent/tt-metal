# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""On an ``--mtp`` server sampled requests take the host pass loop (no device).

The sampler default is the host sampler under ``--mtp`` (the MTP chain's tail resolves the greedy token for the draft
row), ``--device-sampler`` with ``--mtp`` is refused, the session binds no device policy without a sampler, and the
drafting admission admits a sampled request that arrives without the device loop and names the ones it refuses."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tests.helpers.sampling import sampling_parameters
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as step
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38SamplingParameters


@pytest.mark.parametrize(
    "sampling, mtp, requested, expected",
    [
        (True, None, None, True),  # the device sampler is the one-stream --sampling default
        (True, 4, None, False),  # an --mtp server samples on the host
        (False, None, None, False),  # no sampling, no sampler
        (True, None, False, False),  # --host-sampler
        (True, 4, False, False),
        (True, None, True, True),  # --device-sampler on a one-stream server
    ],
)
def test_the_sampler_default_follows_the_server_kind(sampling, mtp, requested, expected):
    args = SimpleNamespace(sampling=sampling, mtp=mtp, device_sampler=requested)
    assert server.effective_device_sampler(args) is expected


def test_the_device_sampler_is_refused_with_mtp_and_without_sampling(expect_error):
    with expect_error(SystemExit, match="not served with --mtp: the MTP chain samples on the host"):
        server.effective_device_sampler(SimpleNamespace(sampling=True, mtp=4, device_sampler=True))
    with expect_error(SystemExit, match="needs --sampling"):
        server.effective_device_sampler(SimpleNamespace(sampling=False, mtp=None, device_sampler=True))


def test_a_sampled_request_on_an_mtp_server_is_admitted_to_the_host_pass_loop_and_the_refusals_are_named():
    sampled, off = SimpleNamespace(sampled=True), SimpleNamespace(sampled=False)
    for profile in (Qwen38SamplingParameters.official_thinking, Qwen38SamplingParameters.official_non_thinking):
        request = step.Qwen38SamplingRequest(profile(seed=1))
        assert request.uniforms == []  # no sampler: begin_request never ran, no device policy or draw was bound
        assert step.drafting_admission(sampled, request, device_loop=False) is None  # admitted: the pass loop drafts
    thinking = step.Qwen38SamplingRequest(Qwen38SamplingParameters.official_thinking(seed=1))
    assert step.drafting_admission(sampled, thinking) is None
    assert step.drafting_admission(sampled, thinking, device_loop=True) == "refused: device sampler loop"
    assert step.drafting_admission(sampled, None) is None and step.drafting_admission(None, None) is None
    assert step.drafting_admission(None, thinking) == "refused: no MTP chain"
    assert step.drafting_admission(off, thinking) == "refused: no split verify captured"
    for request, reason in (
        (step.Qwen38SamplingRequest(sampling_parameters(top_k=0)), "refused: top_k 0"),
        (step.Qwen38SamplingRequest(sampling_parameters(presence_penalty=-0.5)), "refused: penalty raises logits"),
        (step.Qwen38SamplingRequest(sampling_parameters(repetition_penalty=0.9)), "refused: penalty raises logits"),
        (step.Qwen38SamplingRequest(thinking.parameters, logprobs=True), "refused: logprobs"),
        (step.Qwen38SamplingRequest(thinking.parameters, top_logprobs=2), "refused: logprobs"),
    ):
        assert step.drafting_admission(sampled, request) == reason


def test_the_server_and_the_session_wire_the_host_sampler_on_an_mtp_chain():
    source = inspect.getsource(server.main)
    assert source.index("args.device_sampler = effective_device_sampler(args)") < source.index(
        "device_sampler=bool(args.device_sampler),"
    )
    complete = inspect.getsource(session.Qwen38ChatSession.complete)
    # the request start binds the device policy and the first draw only when the extension has a sampler (or the
    # device acceptance, whose begin_request leaves the uniforms empty); otherwise the request arrives at the drafting
    # admission with device_loop False and the pass loop drafts for it
    assert 'getattr(self.sampling, "sampler", None) is not None' in complete
    assert 'getattr(self.sampling, "device_accept", None) is not None' in complete
    assert "device_loop = sampling is not None and bool(sampling.uniforms)" in complete
    assert "sampling_step.drafting_admission(self.mtp, sampling, device_loop=device_loop)" in complete
