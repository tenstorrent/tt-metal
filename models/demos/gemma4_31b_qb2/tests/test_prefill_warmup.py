# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Prefill lengths are compiled up front, and a kept decode trace gives the same tokens as a rebuilt one.

A prefill of a length class the program cache has never seen compiles new programs (seconds) and retires the decode
traces; after warm_prefill_shapes it does neither, so the trace survives the prefills of an agent loop.
"""

import time

import pytest
import torch

from models.demos.gemma4_31b_qb2.tt.generator import Gemma4Generator, build_generator

CAPACITY = 2048
NEW_TOKENS = 24


@pytest.fixture
def generator(qb2_mesh):
    gen = build_generator(qb2_mesh, layer_indices=[0, 5], max_seq_len=CAPACITY)
    yield gen
    gen.close()


def _generate(gen, prompt):
    return gen.generate(prompt, NEW_TOKENS, temperature=0.0, top_k=1, stop_on_eos=False)


def test_warmup_removes_compilation_and_keeps_the_trace(generator, qb2_mesh):
    gen = generator
    torch.manual_seed(0)
    prompt = torch.randint(1000, 50000, (700,)).tolist()
    other = torch.randint(1000, 50000, (300,)).tolist()

    # Cold: a new length class compiles programs.
    programs = qb2_mesh.num_program_cache_entries()
    cold = _generate(gen, prompt)
    cold_ttft = gen.last_ttft_s
    assert qb2_mesh.num_program_cache_entries() > programs

    # Warm every class up to the capacity, then serve classes that were never run.
    start = time.perf_counter()
    state = gen._owned(1, CAPACITY)
    gen.warm_prefill_shapes(state, state.page_tables, CAPACITY)
    warm_seconds = time.perf_counter() - start
    programs = qb2_mesh.num_program_cache_entries()
    _generate(gen, prompt)  # builds the decode trace again after warm-up's retire
    entry = next(iter(gen.traces.values()))
    warm_ttfts = []
    for length in (320, 1120, 1920):
        ids = torch.randint(1000, 50000, (length,)).tolist()
        _generate(gen, ids)
        warm_ttfts.append(round(gen.last_ttft_s, 3))
        assert gen.traces and next(iter(gen.traces.values())) is entry, f"length {length} retired the trace"
    assert qb2_mesh.num_program_cache_entries() == programs, "a warmed class compiled new programs"
    # 1000 tokens pad to the warmed 1024: nothing compiles, and its time-to-first-token stays near the warmed ones.
    _generate(gen, torch.randint(1000, 50000, (1000,)).tolist())
    odd_ttft = gen.last_ttft_s
    assert odd_ttft < 10.0
    _generate(gen, prompt)  # rebuild the trace for the identity checks below
    print(f"odd-length ttft(1000)={odd_ttft:.2f}s")
    print(f"cold ttft(700)={cold_ttft:.2f}s warm-up={warm_seconds:.1f}s warm ttfts(320,1120,1920)={warm_ttfts}")

    # The kept trace decodes exactly what a rebuilt one does.
    kept = _generate(gen, prompt)
    _generate(gen, other)
    kept_after_other = _generate(gen, prompt)
    gen._retire_traces()
    rebuilt = _generate(gen, prompt)
    assert kept == cold == kept_after_other == rebuilt


def test_warmup_covers_padded_lengths_last_rows_and_continuations_then_retires_the_traces():
    from types import SimpleNamespace

    generator = object.__new__(Gemma4Generator)
    events = []
    generator.prefill_forward = lambda tokens, **kw: events.append(
        (int(tokens.shape[1]), kw["prompt_lens"], kw.get("start_positions"), kw.get("history_keys"))
    )
    generator._retire_traces = lambda **kwargs: events.append("retire")
    generator.model = SimpleNamespace(tensor=lambda *a, **k: object(), config=SimpleNamespace(vocab_size=8))
    generator._merge_prefill_rows = lambda rows: events.append(("merge", len(rows)))
    state = SimpleNamespace(prefill_history={-1: object()})
    generator.warm_prefill_shapes(state, None, 1408)
    from_zero = [(n, [n], None, None) for n in range(128, 1409, 128)]
    last_rows = [(n, [n], None, None) for n in (*range(1, 33), *range(129, 161))]
    continuations = []
    for chunk in (128, 256):
        for previous_end in (1152, 1153):
            continuations.append((1152 + chunk, [previous_end], None, [-1]))
            continuations.append((1152 + chunk, [1152 + chunk], [previous_end], [-1]))
    merges = [("merge", b) for b in range(2, 33)]
    assert events == from_zero + last_rows + continuations + merges + ["retire"]
    assert state.prefill_history == {}


def _serving_adapter(columns):
    from types import SimpleNamespace

    from models.demos.gemma4_31b_qb2.tt.generator_vllm import Gemma4ForCausalLM

    model = object.__new__(Gemma4ForCausalLM)
    calls = []
    model.generator = SimpleNamespace(
        warm_prefill_shapes=lambda state, tables, max_tokens: calls.append((state, max_tokens, tables))
    )
    state = SimpleNamespace(page_tables={0: torch.zeros(1, columns, dtype=torch.int32)}, capacity=columns * 128)
    return model, state, calls


def test_serving_warmup_stays_within_the_cache_capacity():
    model, state, calls = _serving_adapter(columns=16)  # --max-model-len 2048
    model.warmup_model_prefill(state, True)
    ((_, max_tokens, tables),) = calls
    assert max_tokens == 2048
    assert tables[0][0].tolist() == list(range(1, 17))
    model, state, calls = _serving_adapter(columns=2048)  # 262144: the prefill limit applies
    model.warmup_model_prefill(state, True)
    assert calls[0][1] == 8192 and calls[0][2][0][0, :64].tolist() == list(range(1, 65))


def test_second_warm_up_of_the_same_cache_is_skipped():
    model, state, calls = _serving_adapter(columns=64)
    model.warmup_model_prefill(state, True)
    model.warmup_model_prefill(state, True)  # the plugin's second phase
    assert len(calls) == 1
    other = _serving_adapter(columns=64)[1]
    model.warmup_model_prefill(other, True)
    assert len(calls) == 2


def _sampling_generator(monkeypatch, grows):
    import collections
    from types import SimpleNamespace

    import models.demos.gemma4_31b_qb2.tt.generator as module

    generator = object.__new__(Gemma4Generator)
    events, programs = [], [0]
    generator.mesh = SimpleNamespace(num_program_cache_entries=lambda: programs[0])
    generator.model = SimpleNamespace(config=SimpleNamespace(vocab_size=8))
    generator.counters = collections.defaultdict(int)
    generator.configure_sampling = lambda **kwargs: events.append("configure")

    def merge(outputs):
        programs[0] += grows
        events.append("merge")
        return "merged"

    generator._merge_prefill_rows = merge
    generator._entry = lambda state: events.append("entry") or {"logits": "l", "sampling_trace": "t", "tokens": "k"}
    generator._retire_traces = lambda **kwargs: events.append("retire")
    generator._read_tokens = lambda tokens, batch: torch.zeros(batch, dtype=torch.int32)
    monkeypatch.setattr(module.ttnn, "Shape", lambda dims: dims)
    monkeypatch.setattr(module.ttnn, "reshape", lambda tensor, *args: tensor)
    monkeypatch.setattr(module.ttnn, "copy", lambda source, destination: events.append("copy"))
    monkeypatch.setattr(module.ttnn, "execute_trace", lambda *args, **kwargs: events.append("execute"))
    return generator, events


def test_sampling_a_known_batch_size_compiles_nothing_and_keeps_the_traces(monkeypatch):
    generator, events = _sampling_generator(monkeypatch, grows=0)
    generator.sample_prefill_outputs([object(), object()], object(), {})
    assert events == ["configure", "merge", "entry", "copy", "execute"]


def test_a_program_compiled_while_sampling_retires_the_traces_before_the_sampling_trace_runs(monkeypatch):
    generator, events = _sampling_generator(monkeypatch, grows=2)
    generator.sample_prefill_outputs([object(), object()], object(), {})
    assert events == ["configure", "merge", "entry", "copy", "retire", "entry", "copy", "execute"]
