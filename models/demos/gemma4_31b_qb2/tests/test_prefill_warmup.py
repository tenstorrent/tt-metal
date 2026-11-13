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
    # A length between the warmed ones still creates a few small programs (and retires the trace), but nothing the
    # tens-of-seconds compilation of a first-time shape does.
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


def test_warmup_prefills_every_32_token_length_once_and_retires_the_traces():
    generator = object.__new__(Gemma4Generator)
    events = []
    generator.prefill_forward = lambda tokens, **kw: events.append(
        (int(tokens.shape[1]), kw["prompt_lens"], kw["slots"])
    )
    generator._retire_traces = lambda **kwargs: events.append("retire")
    generator.warm_prefill_shapes(None, None, 128)
    assert events == [(n, [n], [0]) for n in (32, 64, 96, 128)] + ["retire"]
