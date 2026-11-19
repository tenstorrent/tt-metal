# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter accounting regressions; AST extraction never imports TTNN or opens devices."""

import ast
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[1] / "tt/generator_vllm.py"


def adapter_class():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    base = type("HostOnlyGenerator", (), {"model_capabilities": {}, "get_max_tokens_all_users": lambda: 0})
    namespace = {
        "torch": torch,
        "hashlib": hashlib,
        "json": json,
        "os": os,
        "time": time,
        "Path": Path,
        "QwenGenerator": base,
    }
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace["QwenFlashNextForCausalLM"]


class Recorder:
    def __init__(self):
        self.rows = []
        self.order = []

    def begin(self, phase, metadata):
        ticket = len(self.rows)
        self.rows.append({"phase": phase, "metadata": copy.deepcopy(metadata), "ends": [], "completions": 0})
        self.order.append((ticket, "begin"))
        return ticket

    def end(self, ticket, event=None):
        self.rows[ticket]["ends"].append(event)
        self.order.append((ticket, "end"))

    def host_complete(self, ticket):
        self.rows[ticket]["completions"] += 1
        self.order.append((ticket, "host_complete"))


class Generator:
    def __init__(self, result):
        self.caches = [SimpleNamespace(position=100), SimpleNamespace(position=200)]
        self.model = SimpleNamespace(config=SimpleNamespace(vocab_size=1000))
        self.result = result
        self.host = SimpleNamespace(is_tokens=False)
        self.read_event = object()
        self.calls = []

    def decode_forward(self, *args, **kwargs):
        self.calls.append("decode")
        return self.result

    def read_decode_output(self, output, async_read=False, **kwargs):
        self.calls.append("read")
        if isinstance(output, torch.Tensor):
            return (output, []) if async_read else output
        return (self.host, [self.read_event]) if async_read else self.host

    def process_decode_output_host(self, output, **kwargs):
        self.calls.append("host_process")
        return output.unsqueeze(1) if isinstance(output, torch.Tensor) else torch.tensor([[3], [4]])


def make_adapter(result):
    adapter = object.__new__(adapter_class())
    adapter.generator = Generator(result)
    adapter._trace_diagnostics = False
    adapter._benchmark = Recorder()
    adapter._benchmark_tickets = {}
    adapter._benchmark_requests = {0: {"request": 10}, 1: {"request": 20}}
    adapter._log_event = lambda event: None
    return adapter


def submit(adapter, **kwargs):
    return adapter.decode_forward(torch.tensor([1, 2]), torch.tensor([100, 200]), **kwargs)


def test_tensor_slot_remap_preserves_request_identity():
    adapter = make_adapter(SimpleNamespace(is_tokens=False))
    submit(adapter, slot_remap=torch.tensor([1, 0]), read_from_device=False)
    requests = adapter._benchmark.rows[0]["metadata"]["requests"]
    assert [(row["slot"], row["request"]) for row in requests] == [(0, 20), (1, 10)]


def test_resident_positions_follow_source_slots_before_remap():
    adapter = make_adapter(SimpleNamespace(is_tokens=False))
    submit(adapter, slot_remap=[1, 0], read_from_device=False, reload_inputs=False)
    requests = adapter._benchmark.rows[0]["metadata"]["requests"]
    assert [(row["request"], row["position"]) for row in requests] == [(20, 200), (10, 100)]


def test_direct_host_return_finishes_ticket_without_later_callback():
    adapter = make_adapter(torch.tensor([3, 4]))
    result = submit(adapter, read_from_device=True)
    assert isinstance(result, torch.Tensor)
    row = adapter._benchmark.rows[0]
    assert len(row["ends"]) == row["completions"] == 1
    assert not adapter._benchmark_tickets
    assert adapter.generator.calls == ["decode"]


def test_deferred_async_read_ends_at_existing_event_and_completes_after_host_processing():
    adapter = make_adapter(SimpleNamespace(is_tokens=False))
    result = submit(adapter, read_from_device=False)
    row = adapter._benchmark.rows[0]
    assert row["ends"] == [] and row["completions"] == 0
    host, events = adapter.read_decode_output(result, async_read=True)
    assert row["ends"] == [adapter.generator.read_event] == events
    assert row["completions"] == 0
    adapter.process_decode_output_host(host)
    assert row["completions"] == 1 and not adapter._benchmark_tickets
    assert adapter._benchmark.order == [(0, "begin"), (0, "end"), (0, "host_complete")]


def test_empty_host_return_and_optional_async_read_do_not_leave_or_double_end_ticket():
    adapter = make_adapter(torch.full((2,), -1))
    result = adapter.decode_forward(torch.tensor([0, 0]), torch.tensor([-1, -1]), read_from_device=False)
    assert adapter._benchmark.rows[0]["metadata"]["requests"] == []
    host, events = adapter.read_decode_output(result, async_read=True)
    assert host is result and events == []
    row = adapter._benchmark.rows[0]
    assert len(row["ends"]) == row["completions"] == 1
    assert not adapter._benchmark_tickets
