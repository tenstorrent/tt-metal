# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The device-decided sampled form on the chain side (no device): the traces' third form, the statistics row's
parsing, the pass record's fields, the forms table with the admission's count, mtp_enter's routing on the hook, the
request side's dump record against the gate's re-derivation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_mtp_device_accept as device_accept_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.device_sampler import Qwen38DeviceSamplerPolicy
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import VOCAB_SIZE, ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38CandidateRow, Qwen38SamplingParameters

K = 4


# --- the traces, the statistics row, the pass record -------------------------------------------------------------------


def test_traces_third_form_is_the_device_decided_one_and_needs_the_commit_form(expect_error) -> None:
    sampled = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=8, commit=3, verify_sampled=7)
    assert sampled.device_sampled and not sampled.split and sampled.form == "sampled" and sampled.ids() == [7, 3, 8]
    fused = mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3)
    split = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=6, commit=3, verify_head=4, verify_tail=5)
    assert (fused.form, split.form) == ("fused", "split") and not fused.device_sampled and not split.device_sampled
    for kwargs in (
        {"verify_first": 1, "draft": 2, "commit": 3, "verify_sampled": 7},  # two forms
        {"verify_first": None, "draft": 2, "commit": 3, "verify_head": 4, "verify_tail": 5, "verify_sampled": 7},
        {"verify_first": None, "draft": 2, "verify_catch_up": 1, "verify_sampled": 7},  # no commit form
    ):
        with expect_error(ValueError):
            mtp_v2.Qwen38TTNNMTPTraces(**kwargs)


def test_accept_statistics_parse_the_program_row_and_count_the_guard_deviations(expect_error) -> None:
    alignment = (7, 42, *[ZERO_EMBEDDING_TOKEN] * 30)
    reference = mtp_accept.AcceptReference(1, 42, alignment, (5, 0), (9, 11), 0b101, True, 0.25, 3)
    row = reference.statistics_row()
    parsed = mtp_v2.parse_accept_statistics(row)
    assert parsed.row == tuple(row.tolist()) and len(parsed.row) == mtp_accept.STATS_LANES
    assert (parsed.weights, parsed.totals) == ((5, 0), (9, 11))
    assert (parsed.guard_mask, parsed.guard_deviations, parsed.resampled) == (0b101, 2, True)
    assert (parsed.accepted, parsed.token, parsed.theta, parsed.kept) == (1, 42, 0.25, 3)
    all_accepted = mtp_accept.AcceptReference(
        K, 9, (1, 2, 3, 4, 9, *[ZERO_EMBEDDING_TOKEN] * 27), (3, 3, 3, 3), (4, 4, 4, 4), 0, False, 0.5, 2
    )
    parsed = mtp_v2.parse_accept_statistics(all_accepted.statistics_row())
    assert parsed.weights == (3, 3, 3, 3) and parsed.totals == (4, 4, 4, 4) and not parsed.resampled
    assert parsed.guard_deviations == 0 and parsed.accepted == K and parsed.token == 9
    with expect_error(RuntimeError, match="lanes"):
        mtp_v2.parse_accept_statistics(torch.zeros(15))
    assert mtp_v2.DEVICE_ACCEPT_ARITHMETIC == device_accept_module.ARITHMETIC_DEVICE == "device-theta"


def test_pass_record_device_fields_default_to_none() -> None:
    record = mtp_v2.Qwen38TTNNMTPPassRecord(0, 0, (1, 2, 3, 4, 5), 0, (), (), 0, 0, False, {})
    assert (record.decision, record.statistics, record.arithmetic, record.guard_deviations, record.candidate_rows) == (
        None,
        None,
        None,
        None,
        None,
    )
    record = mtp_v2.Qwen38TTNNMTPPassRecord(
        0, 0, (1, 2, 3, 4, 5), 2, (), (), 0, 0, False, {}, statistics=tuple(range(16)), arithmetic="device-theta"
    )
    assert record.arithmetic == mtp_v2.DEVICE_ACCEPT_ARITHMETIC and len(record.statistics) == 16


# --- the forms table, the admission's count -----------------------------------------------------------------------------


def test_forms_table_and_the_admission_count_include_the_device_form(expect_error) -> None:
    assert session_module.MTP_VERIFY_FORMS_BY_SWITCH == {
        (False, False): ("fused",),
        (True, False): ("fused", "split"),
        (True, True): ("fused", "split", "sampled"),
    }
    assert session_module.mtp_verify_forms(True, True) == ("fused", "split", "sampled")
    assert session_module.mtp_verify_forms(True) == ("fused", "split") and session_module.mtp_verify_forms(False) == (
        "fused",
    )
    with expect_error(ValueError, match="needs the split verify"):
        session_module.mtp_verify_forms(False, True)
    with expect_error(ValueError, match="must be a bool"):
        session_module.mtp_verify_forms(True, 1)
    two = session_module.mtp_capacity_admission(32768, drafts=K, verify_forms=2)
    three = session_module.mtp_capacity_admission(32768, drafts=K, verify_forms=3)
    assert three["verify_forms"] == 3 and three["mtp_growth_remainders_bytes_per_bank"]["traces"] == 15_825_664
    assert three["required_free_bytes_per_bank"] == 82_285_051 > two["required_free_bytes_per_bank"] and three["fits"]
    assert len(session_module.MTP_WARM_ACCEPT_UNIFORMS) >= max(mtp_v2.SUPPORTED_DRAFTS) + 1
    for uniform in session_module.MTP_WARM_ACCEPT_UNIFORMS:
        assert 0 <= uniform < 1 and uniform * 2**24 == int(uniform * 2**24)
    assert len(set(session_module.MTP_WARM_ACCEPT_UNIFORMS)) == len(session_module.MTP_WARM_ACCEPT_UNIFORMS)


# --- mtp_enter: the routing on the hook -------------------------------------------------------------------------------


def test_traced_chain_mtp_enter_builds_the_device_form_on_the_hook(expect_error, monkeypatch) -> None:
    built: list[dict] = []

    class RecordingChain:
        def __init__(self, model, verify, draft, traces, verify_output, **kwargs):
            built.append(dict(traces=traces, verify_output=verify_output, **kwargs))

        def bootstrap(self, tokens):
            return SimpleNamespace(
                accepted=0, decision=None, tokens=tuple(tokens), arithmetic=None, guard_deviations=None
            )

    monkeypatch.setattr(mtp_v2, "Qwen38TTNNMTPChain", RecordingChain)
    monkeypatch.setattr(mtp_v2, "enter_verify_mode", lambda *args, **kwargs: None)
    fused = mtp_v2.Qwen38TTNNMTPTraces(verify_first=1, draft=2, commit=3)
    split = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=6, commit=3, verify_head=4, verify_tail=5)
    sampled = mtp_v2.Qwen38TTNNMTPTraces(verify_first=None, draft=8, commit=3, verify_sampled=7)
    mtp = session_module.Qwen38ChainMTP(
        drafts=K,
        anchor="off",
        components=None,
        verify="verify",
        draft="draft",
        step_inputs=None,
        chunk_extension=None,
        traces=fused,
        verify_output="fused-row",
        sampled=True,
        split_traces=split,
        split_verify_output="tail-row",
        head_output="head",
        device_accept=True,
        sampled_traces=sampled,
        sampled_verify_output="sampled-row",
    )
    assert mtp.captured_trace_ids() == [1, 3, 2, 4, 5, 6, 7, 8]
    chain = object.__new__(session_module.Qwen38TracedChain)
    chain.mtp = mtp
    chain.built_target = SimpleNamespace(model="model")
    chain.state = SimpleNamespace(position=SimpleNamespace(read=lambda: 7))
    hook = lambda tokens: None  # noqa: E731
    chain.mtp_enter(11, None, before_verify_sampled=hook, record_candidate_rows=True)
    device = built[-1]
    assert device["traces"] is sampled and device["verify_output"] == "sampled-row" and device["head_output"] is None
    assert device["before_verify_sampled"] is hook and device["record_candidate_rows"] is True
    assert device["decide"] is mtp_v2.decide_greedy
    mtp.chain = None
    chain.mtp_enter(11, None)
    assert built[-1]["traces"] is fused and built[-1]["before_verify_sampled"] is None
    assert built[-1]["record_candidate_rows"] is False
    mtp.chain = None
    with expect_error(session_module.Qwen38ChatChainError, match="not both"):
        chain.mtp_enter(11, None, decide=lambda tokens, head: None, before_verify_sampled=hook)
    mtp.sampled_traces = None
    with expect_error(session_module.Qwen38ChatChainError, match="mtp_device_accept"):
        chain.mtp_enter(11, None, before_verify_sampled=hook)


# --- the request side: the dump, the gate's re-derivation -------------------------------------------------------------


def _candidate_rows(seed: int, rows: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            Qwen38CandidateRow.emulate((torch.randn(VOCAB_SIZE, generator=generator) * 3.0).to(torch.bfloat16))
            .to_host_row()
            .reshape(-1)
            .to(torch.float32)
            for _ in range(rows)
        ]
    )


def test_dump_record_shape_re_derives_under_the_gate(tmp_path: Path) -> None:
    """The records the session writes under the dump switch carry what ``check_records`` re-derives from (the rows the
    device decided on, the pass's tokens and its statistics) plus the pass's own uniforms; a consistent record (the
    statistics the host reference computes for those rows, drafts and uniforms) passes the gate with 0 mismatches."""

    parameters = Qwen38SamplingParameters(temperature=1.0, top_p=0.95, top_k=20, presence_penalty=0.0, seed=5)
    policy = Qwen38DeviceSamplerPolicy.from_parameters(parameters)
    assert policy is not None
    uniforms: list[float] = []
    records: list[dict] = []
    for index in range(3):
        rows = _candidate_rows(100 + index, K + 1)
        drafts = [int(v) for v in rows[:K, 32 : 32 + K].reshape(-1)[:K].tolist()]  # ids from the rows' first shard
        pass_uniforms = [(n + 7 * index) * 2**-24 for n in (1_000_003, 5_000_011, 9_000_017, 13_000_019, 16_000_023)]
        uniforms.extend(pass_uniforms)
        reference = mtp_accept.accept_reference(rows, drafts, policy, pass_uniforms, sentinel=ZERO_EMBEDDING_TOKEN)
        records.append(
            {
                "index": index,
                "candidate_rows": [float(v) for v in rows.reshape(-1).tolist()],
                "tokens": [11, *drafts],
                "statistics": [float(v) for v in reference.statistics_row().tolist()],
                "uniforms": pass_uniforms,
                "committed_tokens": 5 + 3 * index,
            }
        )
    session = object.__new__(session_module.Qwen38ChatSession)
    session.mtp = SimpleNamespace(drafts=K)
    session.device_accept_dump = tmp_path
    session.device_accept_records = records
    session.requests_served = 3
    path = session._write_device_accept_dump(SimpleNamespace(parameters=parameters, uniforms=uniforms))
    run = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "device-accept-00003.json" and set(run) == {"drafts", "parameters", "uniforms", "records"}
    assert run["drafts"] == K and run["parameters"] == {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
    assert len(run["records"]) == 3 and set(run["records"][0]) == {
        "index",
        "candidate_rows",
        "tokens",
        "statistics",
        "uniforms",
        "committed_tokens",
    }
    assert len(run["records"][0]["candidate_rows"]) == (K + 1) * 256 and len(run["records"][0]["statistics"]) == 16
    summary = device_accept_module.check_records(run["records"], policy, drafts=K, uniforms=run["uniforms"])
    assert summary["passes"] == 3 and summary["mismatches"] == 0 and summary["arithmetic"] == "device-theta"
