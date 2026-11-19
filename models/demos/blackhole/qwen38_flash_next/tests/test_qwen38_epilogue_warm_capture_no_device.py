# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The TAIL epilogue's warm pass and its trace capture ask for the same programs (no device).

A program cache miss inside a trace capture is fatal, and a fused program's cache key is its compile-time form, not
the buffer addresses of a call. So the "program key" here is the form of each call: the greedy row's resolve form and
the tensors, by role, the sampler reads and writes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_sampling as step

TOKEN_ROW = "fp32 TILE DRAM [1,1,1,32]"


class FakeTensor:
    def __init__(self, role: str, placement: str = TOKEN_ROW):
        self.role = role
        self.placement = placement


class FakeLMHead:
    """Records the resolve form; a fresh row comes back from every resolve, the ``into`` row receives its copy."""

    def __init__(self):
        self.resolves: list[str] = []

    def greedy_candidates(self, logits):
        return SimpleNamespace(local_values=FakeTensor("greedy values"), local_indices=FakeTensor("greedy indices"))

    def resolve_greedy_on_device(self, candidates, *, into=None):
        self.resolves.append("into=None" if into is None else f"into={into.role}")
        return FakeTensor("resolved token row")

    def sampling_candidates(self, logits, constants, *, candidates=None):
        return FakeTensor("candidate row", "fp32 ROW_MAJOR DRAM [1,1,1,256]")


class FakeSamplerConstants:
    def __init__(self):
        self.policies: list = []

    def write_policy(self, policy):
        self.policies.append(policy)

    def write_uniform(self, uniform):
        pass


class FakeTTNN:
    def __init__(self):
        self.copies: list[tuple[str, str]] = []

    def deallocate(self, tensor):
        pass

    def copy(self, source, target):
        self.copies.append((source.placement, target.placement))


class FakeCandidateRow:
    values = ids = None

    def agreement(self, other):
        return {"values_bitwise": [True], "ids_equal_up_to_boundary_ties": [True]}

    def to_host_row(self):
        return None


def test_device_sampler_warm_and_capture_ask_for_the_same_programs(monkeypatch):
    # The extension without a device: the sampler set, every device read replaced, the sample call recorded.
    ext = object.__new__(step.Qwen38SamplingChainExtension)
    ext.lm_head = FakeLMHead()
    ext.constants = SimpleNamespace(readback_row=FakeTensor("readback row"))
    ext.sampler = FakeSamplerConstants()
    ext.presence_on_device = False
    ext.candidate_row = False  # the chain's candidate row (the fold has its own program-key test)
    ext.trace_rows, ext.trace_logits = [], []
    keys: list[tuple[str, str, str]] = []

    def sample(row, greedy_row, constants):
        keys.append((row.role, greedy_row.role, ext.lm_head.resolves[-1]))
        return FakeTensor("token row")

    ext.sample = sample
    monkeypatch.setattr(step, "ttnn", FakeTTNN())
    monkeypatch.setattr(ext, "_gather", lambda logits: SimpleNamespace(to=lambda dtype: None))
    monkeypatch.setattr(ext, "read_candidate_row", lambda: FakeCandidateRow())
    monkeypatch.setattr(ext, "_token_of", lambda row: 7)
    monkeypatch.setattr(step, "candidate_row_lanes", lambda host_row: (None, None))
    monkeypatch.setattr(
        step, "device_sampler_reference", lambda *args, **kwargs: SimpleNamespace(token_id=7, guard=False)
    )
    monkeypatch.setattr(step.Qwen38CandidateRow, "emulate", staticmethod(lambda full: None))
    token_row_io = FakeTensor("resident token row")
    ext.warm(logits=None, token_row_io=token_row_io, label="warm")
    warm_keys, warm_copies = set(keys), set(step.ttnn.copies)
    keys.clear()
    step.ttnn.copies.clear()
    trace_output = SimpleNamespace(logits=FakeTensor("logits"))
    candidates, trace_token_row = ext.capture_epilogue(trace_output, token_row_io)
    assert set(keys) == warm_keys == {("candidate row", "resolved token row", "into=resident token row")}
    assert ext.lm_head.resolves == ["into=resident token row"] * 3  # the warm's two passes, then the capture
    assert set(step.ttnn.copies) == warm_copies == {(TOKEN_ROW, TOKEN_ROW)}  # the token row into the resident row
    assert trace_token_row.role == "token row" and ext.trace_rows[-1].role == "candidate row"


# --- the MTP chain's tail epilogue: one function, warm == capture across the served contexts --------------------------


def _fake_mtp_step_row(model, alignment, inputs, residual, resolved_row, *, rope, qsa_position):
    """Records the MTP layer's row call: the residual / RoPE / QSA position placements it sees."""

    alignment.log.append(
        ("mtp_row", model.context, residual.placement, rope.placement, qsa_position.placement, resolved_row.placement)
    )


class FakeMTPLMHead:
    def __init__(self, log):
        self.log = log

    def greedy_candidates(self, logits, *, candidate_row=None):
        form = "fold" if candidate_row is not None else "plain"
        self.log.append(("candidates", logits.placement, form))
        return FakeTensor("greedy candidates", "bf16 TILE DRAM shards" + (" + shard row" if form == "fold" else ""))

    def sampling_candidates(self, logits, constants, *, candidates=None):
        form = "fold" if candidates is not None and "shard row" in candidates.placement else "chain"
        self.log.append(("row", form))
        return FakeTensor("candidate row", "fp32 ROW_MAJOR DRAM [1,1,1,256]")

    def resolve_greedy_on_device(self, candidates, *, into=None):
        self.log.append(("resolve", "into=None" if into is None else f"into={into.placement}"))
        return FakeTensor("resolved token row")


@pytest.mark.parametrize("context", [32768, 65536, 131072, 262144])
@pytest.mark.parametrize("sampling", ["none", "chain row", "folded row"])
def test_mtp_tail_epilogue_records_the_same_programs_for_the_warm_step_and_the_capture(monkeypatch, context, sampling):
    monkeypatch.setattr(session.mtp_v2, "forward_mtp_step_row", _fake_mtp_step_row)
    monkeypatch.setattr(session, "ttnn", FakeTTNN())
    extension = (
        None if sampling == "none" else SimpleNamespace(candidate_row=sampling == "folded row", constants=object())
    )
    model = SimpleNamespace(context=context, model_io=SimpleNamespace(embedding=SimpleNamespace()))
    logs = []
    for tag in ("warm", "capture"):
        # The step's output as the warm step (eager, retain_mtp_inputs=True) and the capture (retained) hand it over:
        # the same producer, so the same placements; ``context`` rides on the QSA position inputs.
        output = SimpleNamespace(
            logits=FakeTensor("logits", "bf16 TILE DRAM vocab shards"),
            residual=FakeTensor(f"{tag} residual", "bf16 TILE DRAM [1,4,1,640] branch-major"),
            rope=FakeTensor(f"{tag} rope", "bf16 TILE DRAM rope rows"),
            qsa_position=FakeTensor(f"{tag} qsa position", f"int32 ROW_MAJOR DRAM position inputs c{context}"),
        )
        log = []
        chain_mtp = SimpleNamespace(alignment=SimpleNamespace(log=log), step_inputs=SimpleNamespace())
        candidates, token_row, row = session.mtp_tail_epilogue(
            model, FakeMTPLMHead(log), chain_mtp, output, FakeTensor("resident token row"), extension
        )
        log.append(("copy", tuple(session.ttnn.copies[-1])))
        assert candidates.role == "greedy candidates" and token_row.role == "resolved token row"
        assert (row is None) == (extension is None)
        logs.append(log)
    assert logs[0] == logs[1]  # the warm step and the capture ask for the same programs on the same placements
    expected = ["candidates", "resolve", "mtp_row"] + (["row"] if extension is not None else []) + ["copy"]
    assert [entry[0] for entry in logs[0]] == expected
    assert logs[0][1] == ("resolve", "into=None") and logs[0][2][1] == context
    if extension is not None:  # the row's form follows the extension's candidate_row switch: the fold, or the chain
        assert logs[0][0][2] == ("fold" if extension.candidate_row else "plain")
        assert logs[0][3] == ("row", "fold" if extension.candidate_row else "chain")


def test_mtp_tail_epilogue_refuses_a_step_without_its_retained_inputs(expect_error):
    with expect_error(session.Qwen38ChatChainError, match="MTP tail epilogue needs .* retained MTP inputs"):
        session.mtp_tail_epilogue(None, None, None, SimpleNamespace(logits=object(), residual=None), None, None)
