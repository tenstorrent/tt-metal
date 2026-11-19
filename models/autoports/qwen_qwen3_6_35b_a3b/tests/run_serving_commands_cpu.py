# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU fake-owner tests for serving command routing. Does not import TTNN."""

import ast
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def functions(path, namespace, selected=None):
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and (selected is None or node.name in selected)
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def main():
    events = []
    namespace = dict(
        torch=torch,
        np=np,
        dataclass=dataclass,
        isfinite=isfinite,
        Integral=Integral,
        ttnn=SimpleNamespace(
            execute_trace=lambda mesh, trace, **kwargs: events.append(("trace", trace)),
            deallocate=lambda value: events.append(("free", value)),
        ),
    )
    functions(
        ROOT / "tt/serving_state.py",
        namespace,
        {"ReloadCommands", "_ValidatedRemap", "normalize_remap", "ReloadExecutor"},
    )
    functions(ROOT / "tt/serving.py", namespace)
    decode = namespace["decode_with_commands"]
    assert namespace["_integer_rows"](np.array([2, -1, 7], dtype=np.int32), "positions") == (2, -1, 7)
    for invalid in (np.array([2.0, 3.0]), np.array([True, False])):
        try:
            namespace["_integer_rows"](invalid, "positions")
        except ValueError:
            pass
        else:
            raise AssertionError("Noninteger NumPy rows accepted")

    class Pool:
        def __init__(self):
            self.leases = [None, None]

        def require_drained(self):
            if any(self.leases):
                raise RuntimeError("pending")

        def reserve(self, rows, is_tokens):
            index = self.leases.index(None)
            result = SimpleNamespace(rows=tuple(rows), mode=is_tokens, index=index)
            self.leases[index] = result
            events.append(("reserve", rows, is_tokens))
            return result

        def publish(self, handle, value):
            events.append(("publish", value))

        def process(self, handle, is_tokens):
            assert is_tokens == handle.mode
            self.leases[handle.index] = None
            return (handle.rows, is_tokens)

        def discard(self, handle):
            self.leases[handle.index] = None

    class Owner:
        def __init__(self):
            self.closed = False
            self.prepared = True
            self.max_batch_size = 4
            self.host_sampling = False
            self.mesh = "fake"
            self.model = SimpleNamespace(max_seq_len=128, vocab_size=100)
            self.cache = []
            self.tokens = "tokens"
            self.sampler_logits = "logits"
            self.active_slots = (0,)
            self.sampling_values = {
                "temperature": [0.0] * 32,
                "top_k": [1] * 32,
                "top_p": [1.0] * 32,
                "seed": list(range(32)),
            }
            self.sampler = SimpleNamespace(
                _penalties_active=False,
                tt_sampling=SimpleNamespace(max_top_k=32, log_probs_calculator=SimpleNamespace(enable_log_probs=False)),
                tt_penalties=SimpleNamespace(_prompt_tokens_host=None),
            )
            self.traces = dict(model="solo", model_fixed="fixed", sampling="sample", sampling_64="sample64")
            self.trace_replays = {}
            self.counters = dict(model_replays=0, sampling_replays=0, readbacks=0)
            self.output_pool = Pool()
            self.slot_state = SimpleNamespace(capacity=4, poisoned=False, apply=self.remap)
            self.reload_executor = namespace["ReloadExecutor"](
                self.slot_state, require_drained=self.output_pool.require_drained
            )

        def replay_sampling(self):
            events.append(("trace", "sample"))
            self.counters["sampling_replays"] += 1

        def model_trace_key(self, active_slots):
            return "model_fixed"

        def remap(self, order):
            events.append(("remap", order))
            old_active = set(self.active_slots)
            self.active_slots = tuple(i for i in range(4) if order[i] in old_active)
            for key, old in self.sampling_values.items():
                self.sampling_values[key] = [old[i] for i in order] + old[4:]

        def _set_active_slots(self, rows):
            self.active_slots = tuple(sorted(rows))
            events.append(("active", tuple(rows)))

        def _refresh(self, tokens, positions, rows):
            events.append(("inputs", tokens, positions, tuple(rows)))

        def _bind_pages(self, *args, **kwargs):
            events.append("pages")

        def configure_sampling(self, *, slots, reset_seed_state, **params):
            events.append(("params", tuple(slots), reset_seed_state))
            for key, values in params.items():
                for slot, value in zip(slots, values):
                    self.sampling_values[key][slot] = value

    owner = Owner()
    params = dict(temperature=[0.0, 0.0, 0.7], top_k=[1, 1, 8], top_p=[1.0, 1.0, 0.9], seed=[11, None, 13])
    result = decode(
        owner,
        torch.tensor([[1], [-1], [3]]),
        [7, -1, 9],
        slots=[3, 1, 0],
        sampling_params=params,
        reload_sampling_params=True,
        reset_sampling_state=True,
        read_from_device=False,
    )
    assert result.rows == (3, 1, 0) and result.mode
    assert ("inputs", [1, 3], [7, 9], (3, 0)) in events
    assert ("params", (3, 0), False) in events and ("params", (3, 0), True) in events
    assert owner.sampling_values["seed"][1] == 1, "Inactive request seed was changed"
    assert [event for event in events if event[0] == "trace"] == [("trace", "fixed"), ("trace", "sample")]
    owner.output_pool.process(result, is_tokens=True)

    class Stale:
        def __getattribute__(self, name):
            raise AssertionError("Resident path inspected stale input: " + name)

        def __len__(self):
            raise AssertionError("Resident path inspected stale length")

    events.clear()
    decode(owner, Stale(), Stale(), sampling_params=params, reload_inputs=False)
    assert not any(event[0] in ("inputs", "active", "params") for event in events)
    assert [event for event in events if event[0] == "trace"] == [("trace", "fixed"), ("trace", "sample")]

    # Invalid controls reject before remapping, input upload, or trace execution.
    events.clear()
    try:
        decode(
            owner,
            torch.tensor([[1]]),
            [1],
            sampling_params=dict(temperature=0.7, top_k=100, top_p=1.0),
            reload_sampling_params=True,
            slot_remap=[1, 0, 2, 3],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid device top-k accepted")
    assert events == [] and not owner.reload_executor.poisoned

    # A host call executes commanded reset, but emits no device-sampling trace.
    events.clear()
    decode(
        owner,
        torch.tensor([[1], [2]]),
        [10, 11],
        slots=[0, 3],
        sampling_params=None,
        reload_sampling_params=True,
        reset_sampling_state=True,
        slot_remap=[3, 1, 2, 0],
    )
    assert events[0] == ("remap", (3, 1, 2, 0))
    assert ("params", (0, 3), False) in events and ("params", (0, 3), True) in events
    assert [event for event in events if event[0] == "trace"] == [("trace", "fixed")]
    assert owner.sampling_values["seed"][0] == 11 and owner.sampling_values["seed"][3] == 13

    # Full page tables are normalized on CPU, with decoder-safe causal padding.
    owner.cache = [
        [dict(page_table=SimpleNamespace(shape=(1, 8)), k=SimpleNamespace(shape=(8, 1, 32, 256))) for _ in range(4)]
    ]
    events.clear()
    normalized = namespace["_validated_pages"](owner, torch.tensor([[3, 5], [1, -1]]), None, (0, 3), (0, 3), (33, 2))[0]
    assert normalized.tolist() == [[3, 5, 5, 5, 5, 5, 5, 5], [1, 1, 1, 1, 1, 1, 1, 1]]
    layered = namespace["_validated_pages"](
        owner, torch.tensor([[999]]), [torch.tensor([[3, 5], [1, -1], [999, 999]])], (0, 3), (0, 3), (33, 2)
    )[0]
    assert torch.equal(layered, normalized), "Per-layer tables must override group zero and trim unused rows"
    decode(
        owner, Stale(), Stale(), reload_inputs=False, reload_page_table=True, page_table=torch.tensor([[3, 5], [1, -1]])
    )
    assert "pages" in events and not any(
        isinstance(event, tuple) and event[0] in ("inputs", "active", "params") for event in events
    )
    # Histories for a paused row survive a reset of the active rows.
    old = torch.arange(32).reshape(32, 1)
    history = namespace["_history_after_reset"](old, torch.tensor([[9], [8]]), (0, 1), (1, 0, 2, 3), 100, (0,))
    assert history[0, 0] == 9 and history[1, 0] == 0
    prefill = namespace["prefill_with_sampling"]
    prefill_owner = Owner()
    prefill_owner.cache = owner.cache
    prefill_owner.prefill_logit_rows = SimpleNamespace(
        prepared=True, copy=lambda value, destination, slot: events.append(("stage", value, slot))
    )

    def low_level(tokens, **kwargs):
        prefill_owner.output_pool.require_drained()
        assert kwargs["device_output"] and not kwargs["return_all_logits"]
        events.append(("prefill", tuple(kwargs["slots"])))
        return ["row0", "row1"]

    events.clear()
    pref_params = dict(temperature=[0.0, 0.7], top_k=[1, 8], top_p=[1.0, 0.9], seed=[77, 88])
    result = prefill(
        prefill_owner,
        torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
        low_level=low_level,
        prompt_lens=np.array([3, 4], dtype=np.int64),
        start_pos=np.array([0, 0], dtype=np.int32),
        empty_slots=[2, 0],
        page_table=torch.tensor([[2], [0]]),
        sampling_params=pref_params,
    )
    assert result == ((2, 0), True)
    assert [event for event in events if event[0] == "trace"] == [("trace", "sample")]
    assert events.index(("free", "row1")) < events.index(("trace", "sample"))
    assert ("stage", "row0", 2) in events and ("stage", "row1", 0) in events
    assert prefill_owner.sampling_values["seed"][2] == 77 and prefill_owner.sampling_values["seed"][0] == 88
    assert prefill_owner.sampler.tt_penalties._prompt_tokens_host[2].tolist() == [1, 2, 3, -1]
    events.clear()
    result = prefill(
        prefill_owner,
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        low_level=low_level,
        prompt_lens=[3, 3],
        slots=[2, 0],
        page_table=torch.tensor([[2], [0]]),
    )
    assert result == ((2, 0), False) and not any(event[0] == "trace" for event in events)
    print(
        "SERVING_COMMANDS_CPU_PASS: reload routing, stale inputs, sparse slots, independent reset, remap order, "
        "host output, page-only refresh, safe page padding, history preservation, prefill staging/sampling; "
        "no TTNN import or devices"
    )


if __name__ == "__main__":
    main()
