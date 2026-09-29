# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The device-decided sampled form on the chain side (no device): the traces' third form, the split state's
statistics row and its parsing, the pass record's fields, the composed body (head -> the accept program -> tail, the
head released) and its capture, the pass loop's third branch and hooks, the forms table with the admission's count,
open()'s warm == capture rule and routing, the request side's dump record against the gate's re-derivation."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_mtp_device_accept as device_accept_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.device_sampler import Qwen38DeviceSamplerPolicy
from models.demos.blackhole.qwen38_flash_next.ttnn.embedding import VOCAB_SIZE, ZERO_EMBEDDING_TOKEN
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import mtp_accept
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38CandidateRow, Qwen38SamplingParameters

ROOT = Path(__file__).resolve().parents[1]
MTP_V2_SOURCE = ROOT / "ttnn" / "mtp_v2.py"
SESSION_SOURCE = ROOT / "tools" / "qwen38_chat_session.py"
K = 4


def _functions(source: Path) -> dict[str, ast.FunctionDef]:
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef):
            found.setdefault(node.name, node)
    return found


def _segment(source: Path, node: ast.AST) -> str:
    return " ".join(ast.get_source_segment(source.read_text(encoding="utf-8"), node).split()).replace("( ", "(")


def _calls(node: ast.AST) -> list[str]:
    return [
        ast.unparse(call.func)
        for call in sorted(
            (n for n in ast.walk(node) if isinstance(n, ast.Call)), key=lambda n: (n.lineno, n.col_offset)
        )
    ]


def _in_order(text: str, fragments: list[str]) -> None:
    positions = [text.index(fragment) for fragment in fragments]
    assert positions == sorted(positions), fragments


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


# --- the composed body, its capture, the split state ------------------------------------------------------------------


def test_verify_sampled_body_is_head_then_the_accept_program_then_tail_and_the_capture_wraps_it() -> None:
    functions = _functions(MTP_V2_SOURCE)
    body = _segment(MTP_V2_SOURCE, functions["forward_verify_sampled"])
    calls = _calls(functions["forward_verify_sampled"])
    assert (
        calls.index("forward_verify_head")
        < calls.index("mtp_accept_module.mtp_accept")
        < calls.index("forward_verify_tail")
        < len(calls) - 1
    )
    assert calls.count("head.release_tensors") == 2  # the accept failure path and the end
    assert (
        "mtp_accept_module.mtp_accept(split.candidates_readback, verify.draft_lanes, constants, "
        "accept_tile=split.accept_tile, accept_index=split.accept_index, next_token=split.next_token, "
        "alignment_tokens=split.alignment_tokens, statistics=split.statistics" in body
    )
    assert "_tensor_key(statistics) != _tensor_key(split.statistics)" in body
    assert 'model._mark_poisoned("forward_verify_sampled", BACKBONE_LAYERS, error)' in body
    assert body.index("forward_verify_tail(model, verify, state, head, catch_up=catch_up") < body.index(
        "head.release_tensors() return output"
    )
    # The head and tail bodies and the fused body do not know the program.
    for name in ("forward_verify_head", "forward_verify_tail", "forward_verify", "_split_prologue"):
        assert "mtp_accept" not in _segment(MTP_V2_SOURCE, functions[name]), name
    capture = _segment(MTP_V2_SOURCE, functions["capture_verify_sampled"])
    assert 'guard(f"verify sampled capture catch_up={catch_up}")' in capture
    assert "forward_verify_sampled(model, verify, state, constants, catch_up=catch_up)" in capture
    # the capture runs through mtp_v2._capture_trace, which begins it, ends it and releases a failed one before raising
    assert "_capture_trace(model.mesh_device, cq_id, _body)" in capture and "ttnn.begin_trace_capture" not in capture


def test_split_state_carries_the_statistics_row_and_validates_it() -> None:
    source = MTP_V2_SOURCE.read_text(encoding="utf-8")
    split = source[source.index("class Qwen38TTNNVerifySplit:") : source.index("class Qwen38TTNNVerifyState:")]
    assert "statistics: Any" in split and '"accept statistics"' in split
    assert "def device_written(self)" in split and "return (self.candidates_readback, self.statistics)" in split
    assert "_deallocate(*self.device_written(), *self.host_written())" in split
    validate = _segment(MTP_V2_SOURCE, _functions(MTP_V2_SOURCE)["_validate_verify_state"])
    assert (
        '("accept statistics", split.statistics, mtp_accept_module.STATS_SHAPE, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT)'
        in validate
    )
    assert mtp_accept.STATS_SHAPE == (1, 1, 1, 16)


# --- the pass loop's third branch ---------------------------------------------------------------------------------------


def test_pass_loop_device_branch_launches_one_trace_and_reads_the_statistics_after_the_row() -> None:
    tree = ast.parse(MTP_V2_SOURCE.read_text(encoding="utf-8"))
    chain = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNMTPChain")
    methods = {n.name: n for n in chain.body if isinstance(n, ast.FunctionDef)}
    finish = _segment(MTP_V2_SOURCE, methods["_finish_pass"])
    _in_order(
        finish,
        [
            "if self.traces.device_sampled:",
            "self.before_verify_sampled(list(tokens))",
            "launch(self.traces.verify_sampled)",
            "elif not self.traces.split:",
            "launch(self.traces.verify_head)",
            "launch(self.traces.draft)",
            "read_pass_row(self.verify, self.draft)",
            "read_accept_statistics(self.verify)",
            "is not the device decision",
            "read_candidate_rows(self.verify)",
            "commit_verify_host(",
            "statistics=None if statistics is None else statistics.row",
            "arithmetic=None if statistics is None else DEVICE_ACCEPT_ARITHMETIC",
            "guard_deviations=None if statistics is None else statistics.guard_deviations",
            "candidate_rows=candidate_rows",
        ],
    )
    device_branch = finish[finish.index("if self.traces.device_sampled:") : finish.index("elif not self.traces.split:")]
    for absent in ("read_verify_head", "self.decide(", "write_verify_decision"):
        assert absent not in device_branch, absent
    init = _segment(MTP_V2_SOURCE, methods["__init__"])
    assert "before_verify_sampled: Callable[[list[int]], Any] | None = None" in init
    assert "record_candidate_rows: bool = False" in init
    assert "if traces.device_sampled and verify.split is None:" in init
    assert "not callable(before_verify_sampled) or not traces.device_sampled" in init
    assert "record_candidate_rows and not traces.device_sampled" in init


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


# --- open(): warm == capture, the constants seam, the routing -----------------------------------------------------------


def test_open_warms_and_captures_the_device_form_the_same_way_and_mtp_enter_routes_the_hook() -> None:
    functions = _functions(SESSION_SOURCE)
    opened = _segment(SESSION_SOURCE, functions["open"])
    assert "mtp_device_accept: bool = False" in opened
    assert (
        'raise ValueError("mtp_device_accept needs mtp_sampled: the device decides the split verify\'s pass")' in opened
    )
    assert "forms = mtp_verify_forms(mtp_sampled, mtp_device_accept)" in opened
    # The constants are the sampling extension's device acceptance's; the switch and the build must agree.
    _in_order(
        opened,
        [
            "device_accept=sampling_step.device_acceptance_for_chain(mesh, lm_head.mesh_contract, drafts=mtp)",
            'getattr(sampling_extension, "device_accept", None)',
            "if (device_acceptance is not None) != mtp_device_accept:",
            "accept_constants = None if device_acceptance is None else device_acceptance.constants",
        ],
    )
    assert "accept_constants.mark_corruptible()" not in opened  # the extension marks its own constants
    # The warm: the device form's round after the split's at every residue, the very sequence the capture records,
    # checked against the host reference.
    warm = opened[
        opened.index('marker("before-chat-mtp-warm-pass")') : opened.index('marker("after-chat-mtp-warm-pass")')
    ]
    assert (
        '("fused",) if not target.sampled else ("split", "sampled") if target.device_accept else ("split",)'
        in warm  # the rounds run per drafting chain (warm_mtp_chain) since QWEN38_MTP_DRAFTS_PER_REQUEST
    )
    assert 'elif warm_form == "sampled":' in warm and "accept_constants.write_policy(sampling_step.WARM_POLICY)" in warm
    assert "warm_uniforms = list(MTP_WARM_ACCEPT_UNIFORMS[: target.drafts + 1])" in warm
    assert "accept_constants.write_uniforms(warm_uniforms)" in warm
    eager = "mtp_v2.forward_verify_sampled(model, target.verify, state, accept_constants, catch_up=False"
    assert eager in warm
    _in_order(
        warm,
        [
            eager,
            "mtp_accept_module.accept_reference(",
            "mtp_v2.read_candidate_rows(target.verify)",
            "sentinel=ZERO_EMBEDDING_TOKEN",
            "mtp_v2.read_accept_statistics(target.verify)",
            "reference.statistics_row()",
            "torch.equal(actual.view(torch.int32), expected.view(torch.int32))",
        ],
    )
    # the capture sequence lives in open's per-chain helper (run for the default chain, then each alternate)
    captures = _segment(SESSION_SOURCE, functions["capture_mtp_chain"])
    _in_order(
        captures,
        [
            "mtp_v2.capture_verify_head(",
            "mtp_v2.capture_verify_tail(",
            "dram_after_split = dram_allocated_per_bank()",
            "if target.device_accept:",
            "mtp_v2.capture_verify_sampled(model, target.verify, state, accept_constants, catch_up=False, guard=guard, cq_id=0",
            "sampled_draft = mtp_v2.capture_draft(",
            "sampled_verify_output, guard=guard, cq_id=0",
            "target.sampled_traces = mtp_v2.Qwen38TTNNMTPTraces(",
            "verify_first=None, draft=sampled_draft, commit=commit, verify_sampled=verify_sampled",
            "acknowledge_corruptible(sampled_verify_output.readback)",
        ],
    )
    after = captures[captures.index("dram_after_target = dram_allocated_per_bank()") :]
    assert '"mtp_sampled_traces"] = dram_after_target - dram_after_split' in after
    assert '(["sampled"] if target.sampled_traces is not None else [])' in after
    # The routing: the hook selects the device form; both hooks together are refused; the form needs its traces.
    enter = _segment(SESSION_SOURCE, functions["mtp_enter"])
    _in_order(
        enter,
        [
            "if decide is not None and before_verify_sampled is not None:",
            "if before_verify_sampled is not None and mtp.sampled_traces is None:",
            "mtp_v2.enter_verify_mode(",
            "if before_verify_sampled is not None:",
            "traces, verify_output = mtp.sampled_traces, mtp.sampled_verify_output",
            "head_output, decision = None, mtp_v2.decide_greedy",
            "elif decide is None:",
            "before_verify_sampled=before_verify_sampled,",
            "record_candidate_rows=record_candidate_rows,",
        ],
    )
    close = _segment(SESSION_SOURCE, functions["close"])
    assert "chain_mtp.sampled_traces = None" in close
    assert '("verify_output", "split_verify_output", "sampled_verify_output", "head_output")' in close
    construct = _segment(SESSION_SOURCE, functions["construct_chain"])
    assert "mtp_device_accept: bool = False" in construct and "mtp_device_accept=mtp_device_accept" in construct


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


# --- the request side: the ledger, the dump, the gate's re-derivation --------------------------------------------------


def test_generate_mtp_routes_the_acceptance_records_the_ledger_and_complete_reaches_the_request_start() -> None:
    generate = inspect.getsource(session_module.Qwen38ChatSession._generate_mtp)
    _in_order(
        generate,
        [
            'if sampling is not None and getattr(self.mtp, "device_accept", False):',
            "_refusal, before_verify_sampled = self.chain.sampling.device_acceptance_for(sampling)",
            "record_rows = self.device_accept_dump is not None",
            "uniforms_before = 0 if sampling is None else len(sampling.uniforms)",
            "record = self.chain.mtp_step()",
            "before_verify_sampled=before_verify_sampled,",
            "record_candidate_rows=record_rows,",
            "self.chain.mtp_enter(pending, self.ple_context)",
            'if getattr(record, "arithmetic", None) == mtp_v2.DEVICE_ACCEPT_ARITHMETIC:',
            "sampling.mtp.record_device(record.statistics, self.mtp.drafts)",
            '"index": len(self.device_accept_records)',
            '"candidate_rows": [float(value) for value in record.candidate_rows.reshape(-1).tolist()]',
            '"tokens": [int(token) for token in record.tokens]',
            '"statistics": [float(value) for value in record.statistics]',
            '"uniforms": [float(value) for value in sampling.uniforms[uniforms_before:]]',
            '"committed_tokens": len(self.committed)',
        ],
    )
    complete = inspect.getsource(session_module.Qwen38ChatSession.complete)
    assert 'or getattr(self.sampling, "device_accept", None) is not None' in complete
    assert complete.index("self.sampling.begin_request(sampling)") < complete.index("sampling_step.drafting_admission(")
    assert "self.device_accept_records = []" in complete and "self._write_device_accept_dump(sampling)" in complete
    served = inspect.getsource(server_module)
    assert 'DEVICE_ACCEPT_DUMP_VARIABLE = "QWEN38_MTP_DEVICE_ACCEPT_DUMP"' in served
    assert "session.device_accept_dump = Path(os.environ[DEVICE_ACCEPT_DUMP_VARIABLE])" in served
    main = inspect.getsource(server_module.main)
    assert "mtp_device_accept = device_accept_switch()" in main and "mtp_device_accept_switch" not in served
    assert "forms = mtp_verify_forms(mtp_sampled, mtp_device_accept)" in main
    assert "mtp_device_accept=mtp_device_accept," in main and '"device_accept": mtp_device_accept,' in main
    assert "session.mtp.device_accept != mtp_device_accept" in main


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
