# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The MTP pass's early PLE rows (no device): the verify readback row's fixed lanes read on a second command queue
behind the main queue's event after the verify launch, the next pass's PLE rows 0-1 looked up under the draft, rows
2..k after the pass row, one upload.  Pinned here:
* the two-call lookup (rows 0-1 from the committed context, rows 2..k from their chain's context) is the whole-row
  lookup row for row and context for context, on the real resident lookup over a synthetic table;
* the reader's second-queue work is one event wait and one buffer read (``to_torch`` of coordinate 0 on its queue):
  no program, no trace, nothing else on that queue;
* the pass loop's order: the event right after the verify launch (the fused and the device-decided branches, never
  the host-decided split), the lanes read after the draft launch, the early lookup before the pass row, the lanes
  checked against the pass row, ``step`` writing rows 2..k under the commit; the segment names (their table in the
  chain timing tool is that tool's own static);
* the session's switch: a 2-queue open only when asked, the reader built for the fused and device-decided forms,
  ``QWEN38_MTP_PLE_EARLY`` on by default with ``--mtp`` (``0`` off; an explicit ``1`` without ``--mtp`` refused) and
  its ``/health`` field.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.test_ple_resident_lookup_no_device import (
    EOS,
    _synthetic_corpus,
    _synthetic_host_embedding,
    _SyntheticCheckpoint,
)
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_server as server_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_chat_session as session_module
from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_v2
from models.demos.blackhole.qwen38_flash_next.ttnn.ple import Qwen38ResidentPLELookup

ROOT = Path(__file__).resolve().parents[1]
MTP_V2_SOURCE = ROOT / "ttnn" / "mtp_v2.py"


def _segment(node: ast.AST, source: str) -> str:
    return " ".join(ast.get_source_segment(source, node).split()).replace("( ", "(")


def _chain_methods(source: str) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(source)
    chain = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNMTPChain")
    return {n.name: n for n in chain.body if isinstance(n, ast.FunctionDef)}


def _in_order(text: str, fragments: list[str]) -> None:
    positions = [text.index(fragment) for fragment in fragments]
    assert positions == sorted(positions), [fragment for _, fragment in sorted(zip(positions, fragments))]


# --------------------------------------------------------------------------- the two-call lookup


@pytest.mark.parametrize("rows", (4, 5, 6))
def test_rows_0_1_then_rows_2_k_equal_the_whole_lookup_row_for_row(tmp_path, rows: int) -> None:
    weight_map = _synthetic_corpus(tmp_path, parts_per_file=2)
    checkpoint = _SyntheticCheckpoint(tmp_path, weight_map)
    host = _synthetic_host_embedding(checkpoint)
    resident = Qwen38ResidentPLELookup(host)
    context = None
    for pass_tokens in ([17, 29, 31, EOS, 43, 47][:rows], [0, 998, 12, 12, 7, 999][:rows], [5, 5, 5, 5, 5, 5][:rows]):
        whole_payload, whole_contexts = resident.lookup_tokens(pass_tokens, context)
        early_payload, early_contexts = resident.lookup_tokens(pass_tokens[:2], context)
        late_payload, late_contexts = resident.lookup_tokens(pass_tokens[2:], early_contexts[2])
        assert bytes(early_payload) + bytes(late_payload) == bytes(whole_payload)
        assert early_contexts + late_contexts[1:] == whole_contexts
        assert len(whole_contexts) == rows + 1
        # The chain's join: rows [1,1,2,W] and [1,1,rows-2,W] concatenated on dim 2 are the whole rows' bytes.
        width = 16 * host.embedding_head_dim
        early = torch.frombuffer(bytearray(early_payload), dtype=torch.bfloat16).reshape(1, 1, 2, width)
        late = torch.frombuffer(bytearray(late_payload), dtype=torch.bfloat16).reshape(1, 1, rows - 2, width)
        whole = torch.frombuffer(bytearray(whole_payload), dtype=torch.bfloat16).reshape(1, 1, rows, width)
        assert torch.equal(torch.cat([early, late], dim=2).view(torch.int16), whole.view(torch.int16))
        context = whole_contexts[2]  # a k = 1 commit: the next pass starts from contexts[a + 1]
    resident.close()


def test_write_verify_ple_rows_takes_the_early_rows_and_checks_their_tokens_and_context() -> None:
    source = MTP_V2_SOURCE.read_text(encoding="utf-8")
    functions = {n.name: n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef)}
    write = _segment(functions["write_verify_ple_rows"], source)
    assert "early: Qwen38TTNNPLEEarlyRows | None = None" in write
    _in_order(
        write,
        [
            "if early is None:",
            "rows, contexts = ple.host_rows(tokens, ple_state.token_context)",
            "if early.tokens != tuple(tokens[:2]) or early.contexts[0] != ple_state.token_context:",
            "tail_rows, tail_contexts = ple.host_rows(tokens[2:], early.contexts[2])",
            "rows, contexts = torch.cat([early.rows, tail_rows], dim=2), early.contexts + tail_contexts[1:]",
            "ttnn.copy_host_to_device_tensor(",
            "verify.ple_rows.embedding_rows",
        ],
    )
    early = _segment(functions["lookup_verify_ple_rows_early"], source)
    assert "if first_draft == ZERO_EMBEDDING_TOKEN or next_token == ZERO_EMBEDDING_TOKEN: return None" in early
    assert "rows, early_contexts = ple.host_rows(list(tokens), contexts[accepted + 1])" in early
    assert "contexts = verify.ple_rows.contexts" in early  # this pass's chain: the commit picks contexts[a + 1]


# --------------------------------------------------------------------------- the reader and the pass loop


def test_early_rows_reader_does_one_event_wait_and_one_buffer_read_on_its_queue() -> None:
    source = MTP_V2_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    reader = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "Qwen38TTNNEarlyRowsReader")
    ttnn_calls = sorted(
        {
            ast.unparse(node.func)
            for node in ast.walk(reader)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ttnn"
        }
    )
    assert ttnn_calls == ["ttnn.get_device_tensors", "ttnn.record_event", "ttnn.to_torch", "ttnn.wait_for_event"]
    methods = {n.name: _segment(n, source) for n in reader.body if isinstance(n, ast.FunctionDef)}
    assert "ttnn.record_event(self.mesh_device, cq_id=self.main_cq_id)" in methods["verify_launched"]
    _in_order(
        methods["read_fixed_lanes"],
        [
            'raise RuntimeError("the verify must be launched',
            "ttnn.wait_for_event(self.cq_id, self._event)",
            "self._event = None",
            "ttnn.to_torch(ttnn.get_device_tensors(self.readback)[0], cq_id=self.cq_id)",
            "range(len(READBACK_FIXED_LANES))",
        ],
    )
    for forbidden in ("_ttnn_execute_trace", "begin_trace_capture", "synchronize", "copy_host_to_device_tensor"):
        assert forbidden not in _segment(reader, source), forbidden
    assert "cq_id == main_cq_id" in methods["__init__"]


def test_pass_loop_reads_the_lanes_after_the_draft_and_writes_rows_2_k_under_the_commit() -> None:
    source = MTP_V2_SOURCE.read_text(encoding="utf-8")
    methods = _chain_methods(source)
    finish = _segment(methods["_finish_pass"], source)
    device_branch = finish[finish.index("if self.traces.device_sampled:") : finish.index("elif not self.traces.split:")]
    fused_branch = finish[finish.index("elif not self.traces.split:") : finish.index("else:")]
    split_branch = finish[
        finish.index("launch(self.traces.verify_head)") : finish.index("if self.traces.draft_history")
    ]
    assert "launch(self.traces.verify_sampled)" in device_branch and "early_reader.verify_launched()" in device_branch
    assert "launch(verify_trace)" in fused_branch and "early_reader.verify_launched()" in fused_branch
    assert "verify_launched" not in split_branch  # the host-decided split keeps today's order
    _in_order(
        finish,
        [
            "launch(self.traces.draft)",
            'lanes = self._timed(segments, "early_readback", early_reader.read_fixed_lanes)',
            '"ple_rows_early"',
            "lookup_verify_ple_rows_early(",
            "accepted=lanes[0], next_token=lanes[1], first_draft=lanes[2]",
            "read_pass_row(self.verify, self.draft)",
            "are not the pass row's verify lanes",
            "commit_verify_host(",
            "self._early_rows = early_rows",
        ],
    )
    step = _segment(methods["step"], source)
    _in_order(
        step,
        [
            "early_rows, self._early_rows = self._early_rows, None",
            "self.enqueue(self.traces.commit)",
            '"ple_rows_late"',
            "write_verify_ple_rows(self.model, self.verify, tokens, early=early_rows)",
            '"ple_rows"',
            "write_verify_ple_rows(self.model, self.verify, tokens)",
            "return self._finish_pass(tokens, segments)",
        ],
    )
    init = _segment(methods["__init__"], source)
    assert "early_reader: Qwen38TTNNEarlyRowsReader | None = None" in init
    assert "if traces.split:" in init and "not the split" in init
    assert {"Qwen38TTNNEarlyRowsReader", "Qwen38TTNNPLEEarlyRows", "lookup_verify_ple_rows_early"} <= set(
        mtp_v2.__all__
    )
    for name in ('"early_readback"', '"ple_rows_early"', '"ple_rows_late"', '"ple_rows"', '"readback"'):
        assert name in finish or name in step, name  # the segment names the timing tools read


# --------------------------------------------------------------------------- the session and the server


def test_session_opens_two_queues_only_when_asked_and_builds_the_reader_for_the_forms_that_take_it() -> None:
    opened = inspect.getsource(session_module.open_partition_b_mesh)
    assert "command_queues: int = 1" in opened
    assert '**({"num_command_queues": command_queues} if command_queues != 1 else {}),' in opened
    assert "command_queues not in (1, 2)" in opened and '"command_queues": command_queues,' in opened
    enter = inspect.getsource(session_module.Qwen38TracedChain.mtp_enter)
    _in_order(
        enter,
        [
            "early_reader = None",
            "if mtp.ple_early and decide is None:",
            "mtp_v2.Qwen38TTNNEarlyRowsReader(self.mesh, verify_output.readback, cq_id=1, main_cq_id=0)",
            "early_reader=early_reader,",
        ],
    )
    open_source = inspect.getsource(session_module.Qwen38TracedChain.open)
    assert "mtp_ple_early: bool = False" in open_source
    assert 'raise ValueError("mtp_ple_early needs mtp: the early rows are the MTP pass loop\'s")' in open_source
    assert open_source.count("ple_early=mtp_ple_early,") == 2  # the default chain and every alternate
    construct = inspect.getsource(session_module.construct_chain)
    assert "mtp_ple_early: bool = False" in construct and "mtp_ple_early=mtp_ple_early," in construct
    fields = {field.name: field.default for field in session_module.Qwen38ChainMTP.__dataclass_fields__.values()}
    assert fields["ple_early"] is False


def test_server_switch_is_on_by_default_with_mtp_off_without_and_opens_the_second_queue(expect_error) -> None:
    assert server_module.PLE_EARLY_VARIABLE == "QWEN38_MTP_PLE_EARLY"
    assert server_module.ple_early_switch({}) is True  # unset = the server runs with --mtp: on
    assert server_module.ple_early_switch({}, applicable=False) is False  # unset without --mtp: off, no refusal
    assert server_module.ple_early_switch({"QWEN38_MTP_PLE_EARLY": "0"}) is False
    assert server_module.ple_early_switch({"QWEN38_MTP_PLE_EARLY": "1"}, applicable=False) is True  # main refuses it
    with expect_error(SystemExit, match="must be 0 or 1"):  # allow-pytest.raises: the switch's domain
        server_module.ple_early_switch({"QWEN38_MTP_PLE_EARLY": "2"})
    main = inspect.getsource(server_module.main)
    _in_order(
        main,
        [
            "mtp_ple_early = ple_early_switch(os.environ, applicable=args.mtp is not None)",
            "if mtp_ple_early and args.mtp is None:",
            '"ple_early": mtp_ple_early,',
            "command_queues=2 if mtp_ple_early else 1",
            "mtp_ple_early=mtp_ple_early,",
        ],
    )
