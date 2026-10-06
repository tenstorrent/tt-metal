# SPDX-License-Identifier: Apache-2.0
"""Host snapshot ownership regression; no TTNN import or hardware access."""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

source = Path(__file__).resolve().parents[1] / "tt/generator.py"
tree = ast.parse(source.read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "KolibriGenerator")
method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_copy_prefill_input")
ns = {"torch": torch}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
g = SimpleNamespace(prefill_input_snapshots={}, counters=Counter())


def copy(host, target, counter):
    target.copy_(host)
    g.counters[counter] += 1


g.copy = copy
call = lambda bucket, key, host, target: ns["_copy_prefill_input"](g, bucket, key, host, target)
# Views into scheduler-owned rows must be snapshotted, not retained as aliases.
scheduler = torch.arange(32, dtype=torch.int32).reshape(4, 8)
device = torch.zeros(1, 8, dtype=torch.int32)
call(128, ("pages", 0), scheduler[:1], device)
call(128, ("pages", 0), scheduler[:1], device)
assert g.counters == {"prefill_input_refreshes": 1, "prefill_input_copies_skipped": 1}
scheduler[0, 3] = 99
call(128, ("pages", 0), scheduler[:1], device)
assert torch.equal(device, scheduler[:1]) and g.counters["prefill_input_refreshes"] == 2
# Row reorder and a return to an earlier row must both refresh this bucket.
for row in (2, 0):
    call(128, ("pages", 0), scheduler[row : row + 1], device)
    assert torch.equal(device, scheduler[row : row + 1])
# Distinct buckets and layer bindings have independent storage and snapshots.
other = torch.zeros_like(device)
call(512, ("pages", 0), scheduler[:1], other)
call(128, ("pages", 1), scheduler[:1], other)
assert g.counters["prefill_input_refreshes"] == 6
# Repeated novel values replace a bounded slot; snapshots never grow per request.
for i in range(100):
    scheduler[0, 0] = i
    call(128, ("pages", 0), scheduler[:1], device)
assert len(g.prefill_input_snapshots) == 3
print("PASS changed/unchanged tables, caller mutation, row order, bucket/layer isolation, bounded snapshots")
