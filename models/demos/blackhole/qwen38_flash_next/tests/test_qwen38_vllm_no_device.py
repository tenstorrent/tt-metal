# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vLLM adapter's contract on a fake chain (no device).

The op order of both forward calls on a chain that logs every primitive (the reset, the chunk driver over all prompt
tokens but the last, the last token forced under the loop guard, the full-vocabulary gather, decode steps at the
runner's position); every refusal, made before any chain call; the LM-head padding mask; the real chunk driver over
the driver test's fake model; construction with the library's construction functions replaced by recorders (the
context table, the cache layout, the corpus pair, the switches, the checkpoint and cache-identity resolution orders,
the refusals; the prefill-form knobs: the slab rows and the 128-row chunks the chain is opened with, the values
refused by name, the DRAM admission's refusal of the slab re-raised naming the knob); the class surface the plugin
reads; the device-sampling contract (``sampling_params`` present): the int32 token drawn on the host from the same
row, greedy = the host path's argmax, seeded draws deterministic and inside the nucleus, vLLM's penalties from the
contract's history, and the reduced sampler's draws against vLLM's sampler algorithm transcribed (chi-square over
20k draws, every reduced / full-row path).
Two tests need the serving venv and skip elsewhere: the real plugin runner's prefill and decode call shape over a
namespace runner (host steps and device-sampled steps), and vLLM's config resolution against the pinned checkpoint
(``QWEN38_CHECKPOINT``).
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import huggingface_hub
import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.checkpoint import PINNED_CHECKPOINT_REVISION
from models.demos.blackhole.qwen38_flash_next.tests import test_qwen38_prefill_driver_no_device as driver_tests
from models.demos.blackhole.qwen38_flash_next.tests import test_qwen38_sampling_step_no_device as sampling_tests
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_prefill_driver as driver_module
from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as vllm_module
from models.demos.blackhole.qwen38_flash_next.tools.live_decode_diagnostic import MODE
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    CHUNK_PREFILL_MIN_ROWS,
    RESIDUE_CLASSES,
    SEED_TOKEN_ID,
    WARM_CHUNK_TOKEN_IDS,
    Qwen38ChatChainError,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_prefill_driver import (
    CHUNK_PAD_TOKEN_ID,
    Qwen38ChunkPrefill,
    chunk_accepts,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import HF_REPO_ID, Qwen38ForCausalLM, Qwen38VLLMSlot
from models.demos.blackhole.qwen38_flash_next.tools.runtime_admission import RuntimeAdmissionError
from models.demos.blackhole.qwen38_flash_next.ttnn import prefill_dense
from models.demos.blackhole.qwen38_flash_next.ttnn.builder import RESIDENT_CONTEXT_HEADROOM, Qwen38CacheRoots
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import CHUNK_ROWS, DEFAULT_SLAB_ROWS

HF_ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
TT_ARCHITECTURE = "TT" + HF_ARCHITECTURE
CHECKPOINT = Path(os.environ.get("QWEN38_CHECKPOINT", "/nonexistent/Qwen3.8-Flash-Next"))
BUNDLE = Path(vllm_module.__file__).parent / "vllm_bundle" / "qwen38"
ALLOCATED_CONTEXT = 32_768
MAX_MODEL_LEN = ALLOCATED_CONTEXT - RESIDENT_CONTEXT_HEADROOM
PAGE_TABLE = torch.zeros((1, (MAX_MODEL_LEN + 63) // 64), dtype=torch.int32)  # the runner's width at block size 64
TRACE_ID = 77
DIGEST = "0123456789abcdef" * 4


class FakeChain(sampling_tests.FakeChain):
    """The sampling-step fake plus what the adapter calls: the reset, a logging guard, a recording chunk driver that
    advances the inputs as the device would, the open and miss-guard flags, close."""

    def __init__(self) -> None:
        super().__init__([])
        self.sampling = sampling_tests.FakeSampling(self)
        self.closed = False
        self.misses_forbidden = True
        self.chunks: list[tuple[tuple[int, ...], int]] = []
        self.chunk_stopped: str | None = None  # the driver's should_stop reason, reported on the result

    def reset_and_seed(self, token_id: int) -> None:
        self.log.append(f"reset:{token_id}")
        self.inputs = []
        self.row = token_id

    @contextmanager
    def loop_guard(self):
        self.log.append("guard")
        yield
        self.log.append("unguard")

    def chunk_prefill(self, token_ids, *, start_position, ple_context, forced_step):
        ids = tuple(int(token) for token in token_ids)
        assert start_position == len(self.inputs)
        self.log.append(f"chunk_prefill:{len(ids)}@{start_position}")
        self.chunks.append((ids, start_position))
        self.inputs.extend(ids)
        for token in ids:
            ple_context = (token, 0 if ple_context is None else ple_context[0])  # refresh_ple_row's rule
        return SimpleNamespace(position=start_position + len(ids), ple_context=ple_context, stopped=self.chunk_stopped)

    def close(self) -> None:
        self.log.append("close")
        self.closed = True


class DriverChain(FakeChain):
    """``chunk_prefill`` runs the real chunk driver over the driver test's fake model; its replays log here."""

    def __init__(self) -> None:
        super().__init__()
        self.model = driver_tests._FakeModel()

    def chunk_prefill(self, token_ids, *, start_position, ple_context, forced_step):
        assert start_position == len(self.inputs)
        result = Qwen38ChunkPrefill(
            self.model, "mesh", "state", "chunk_state", TRACE_ID, forced_step=forced_step, verify_allocations=False
        ).run(token_ids, start_position=start_position, ple_context=ple_context)
        self.inputs.extend(int(token) for token in token_ids)
        return result


@pytest.fixture
def driver_chain(monkeypatch) -> DriverChain:
    chain = DriverChain()

    def execute(mesh, trace_id, *, cq_id, blocking) -> None:
        assert (mesh, trace_id, cq_id, blocking) == ("mesh", TRACE_ID, 0, False)
        chain.log.append("chunk_replay")

    def logged(name: str):
        return lambda *_args, **_kwargs: chain.log.append(name) or name

    monkeypatch.setattr(
        driver_module,
        "ttnn",
        SimpleNamespace(
            _ttnn_execute_trace=execute,
            record_event=logged("chunk_event"),
            event_synchronize=logged("chunk_wait"),
            synchronize_device=logged("synchronize"),
        ),
    )
    return chain


def _adapter() -> tuple[Qwen38ForCausalLM, FakeChain]:
    chain = FakeChain()
    return Qwen38ForCausalLM(chain, allocated_context=ALLOCATED_CONTEXT), chain


def _prefill(model: Qwen38ForCausalLM, ids: list[int], **overrides):
    kwargs = dict(
        tokens=torch.tensor([ids], dtype=torch.int32),
        page_table=PAGE_TABLE,
        kv_cache=None,
        enable_trace=True,
        prompt_lens=[len(ids)],
        start_pos=torch.zeros(1, dtype=torch.int32),
        empty_slots=[0],
    )
    kwargs.update(overrides)
    return model.prefill_forward(**kwargs)


def _decode(model: Qwen38ForCausalLM, token: int, position: int, **overrides) -> torch.Tensor:
    kwargs = dict(
        tokens=torch.tensor([[token]], dtype=torch.int32),
        start_pos=torch.tensor([position], dtype=torch.int32),
        page_table=PAGE_TABLE,
        kv_cache=None,
        enable_trace=True,
        read_from_device=False,
    )
    kwargs.update(overrides)
    return model.decode_forward(**kwargs)


def _step_ops(token: int, residue: int) -> list[str]:
    return [f"write:{token}", f"head:{residue}", f"ple:{token}", f"tail:{residue}"]


def _guarded_step_ops(token: int, residue: int) -> list[str]:
    return ["guard", *_step_ops(token, residue), f"full_gather:{residue}", "unguard"]


def _masked(row: torch.Tensor) -> torch.Tensor:
    logits = row.to(torch.float32)
    logits[TOKENIZER_SIZE:] = float("-inf")
    return logits


# -- the two forward calls: op order, the returned tensors, the mask --------------------------------------------------


@pytest.mark.parametrize("length", (1, 2, 16, 17, 33, 100))
def test_prefill_then_decode_op_order(length: int) -> None:
    model, chain = _adapter()
    ids = [1000 + index for index in range(length)]
    head, last = ids[:-1], ids[-1]

    logits, rope_deltas = _prefill(model, ids)

    expected = [f"reset:{SEED_TOKEN_ID}"]
    if len(head) >= CHUNK_PREFILL_MIN_ROWS:
        expected.append(f"chunk_prefill:{len(head)}@0")
    else:  # fewer than 16 forced rows under one guard, so the 16-token event cadence never fires
        expected += ["guard"]
        expected += [op for position, token in enumerate(head) for op in _step_ops(token, position % RESIDUE_CLASSES)]
        expected += ["unguard"]
    expected += _guarded_step_ops(last, (length - 1) % RESIDUE_CLASSES)
    assert chain.log == expected
    assert chain.chunks == ([(tuple(head), 0)] if len(head) >= CHUNK_PREFILL_MIN_ROWS else [])
    # HEAD consumed the prompt as sent; the device's resolves were never fed back.
    assert chain.inputs == ids and model.slot.committed == ids and model.slot.failed is False
    assert model.slot.ple_context == ((ids[-1], ids[-2]) if length > 1 else (ids[0], 0))
    assert logits.shape == (1, 1, VOCAB_SIZE) and logits.dtype == torch.float32 and logits.device.type == "cpu"
    assert torch.equal(logits[0, 0], _masked(chain.logits()))
    assert rope_deltas.shape == (1,) and rope_deltas.dtype == torch.long and int(rope_deltas[0]) == 0

    sampled = [int(logits.argmax())]
    for step in range(5):
        chain.log.clear()
        position = length + step
        logits = _decode(model, sampled[-1], position)
        assert chain.log == _guarded_step_ops(sampled[-1], position % RESIDUE_CLASSES)
        assert logits.shape == (1, 1, VOCAB_SIZE) and torch.equal(logits[0, 0], _masked(chain.logits()))
        sampled.append(int(logits.argmax()))
    assert model.slot.committed == ids + sampled[:-1] and chain.inputs == ids + sampled[:-1]


def test_logits_are_the_full_row_with_the_lm_head_padding_masked() -> None:
    model, chain = _adapter()
    ids = [7, 8, 9]
    logits, _ = _prefill(model, ids)
    row = logits[0, 0]
    assert VOCAB_SIZE - TOKENIZER_SIZE == 243
    assert torch.equal(row[:TOKENIZER_SIZE], chain.logits().to(torch.float32)[:TOKENIZER_SIZE])
    assert torch.isinf(row[TOKENIZER_SIZE:]).all() and (row[TOKENIZER_SIZE:] < 0).all()
    assert int(row.argmax()) < TOKENIZER_SIZE


@pytest.mark.parametrize("slot_remap", (None, [0], torch.tensor([0], dtype=torch.int32)))
def test_decode_accepts_the_identity_slot_remap_and_the_rope_deltas(slot_remap) -> None:
    model, chain = _adapter()
    _prefill(model, [1, 2, 3, 4])
    logits = _decode(model, 5, 4, slot_remap=slot_remap, rope_deltas_all_users=[0])
    assert logits.shape == (1, 1, VOCAB_SIZE) and model.slot.committed == [1, 2, 3, 4, 5]
    assert chain.log[-7:] == _guarded_step_ops(5, 0)


# -- the device-sampling contract: the token drawn on the host from the same row --------------------------------------


def _params(**overrides) -> SimpleNamespace:
    """The plugin's ``TTSamplingParams`` as ``decode_forward`` receives them under device sampling: per-row lists, the
    seed sentinel already turned into ``None``; the checkpoint's generation defaults."""

    fields = dict(
        temperature=[1.0],
        top_k=[20],
        top_p=[0.95],
        presence_penalty=[0.0],
        frequency_penalty=[0.0],
        repetition_penalty=[1.0],
        seed=[None],
        num_logprobs=[0],
        enable_log_probs=[False],
    )
    fields.update({name: [value] for name, value in overrides.items()})
    return SimpleNamespace(**fields)


def _vllm_sampler_probabilities(row: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """vLLM's ``apply_top_k_top_p_pytorch`` transcribed (ascending sort, the k-th value's threshold, the cumulative
    sum masked at ``<= 1 - p`` with the last kept) and the softmax ``random_sample`` draws from."""

    logits = row.to(torch.float32) / temperature
    vocab = logits.numel()
    top_k = top_k if 0 < top_k < vocab else vocab
    logits_sort, logits_idx = logits.sort(descending=False)
    logits_sort[logits_sort < logits_sort[vocab - top_k]] = float("-inf")
    probs_sum = logits_sort.softmax(0).cumsum(0)
    top_p_mask = probs_sum <= 1 - top_p
    top_p_mask[-1] = False
    logits_sort[top_p_mask] = float("-inf")
    return torch.empty_like(logits).scatter_(0, logits_idx, logits_sort).softmax(0)


def _vllm_penalized(row, prompt_ids, output_ids, *, presence, frequency, repetition) -> torch.Tensor:
    """vLLM's ``apply_penalties`` transcribed for one row (bin counts, the repetition scale, the two subtractions)."""

    logits = row.to(torch.float32).clone()
    vocab = logits.numel()
    prompt_mask = torch.zeros(vocab, dtype=torch.bool)
    prompt_mask[prompt_ids] = True
    output_counts = torch.bincount(output_ids, minlength=vocab)
    output_mask = output_counts > 0
    penalties = torch.where(prompt_mask | output_mask, repetition, 1.0)
    logits *= torch.where(logits > 0, 1.0 / penalties, penalties)
    logits -= frequency * output_counts
    logits -= presence * output_mask
    return logits


def _chi_square_ok(counts: torch.Tensor, expected: torch.Tensor, draws: int) -> tuple[float, float]:
    """Pearson's statistic over the bins with an expected count of at least 5 (the rest pooled) against the 1e-4
    critical value (Wilson-Hilferty)."""

    expected_counts = expected.to(torch.float64) * draws
    big = expected_counts >= 5
    observed = torch.cat([counts[big].to(torch.float64), counts[~big].sum().view(1).to(torch.float64)])
    predicted = torch.cat([expected_counts[big], expected_counts[~big].sum().view(1)])
    keep = predicted > 0
    statistic = float(((observed[keep] - predicted[keep]) ** 2 / predicted[keep]).sum())
    df = int(keep.sum()) - 1
    critical = df * (1 - 2 / (9 * df) + 3.719 * (2 / (9 * df)) ** 0.5) ** 3
    return statistic, critical


def test_device_sampling_greedy_returns_the_host_path_argmax_as_an_int32_token() -> None:
    model, chain = _adapter()
    reference, reference_chain = _adapter()
    ids = [1, 2, 3, 4]
    _prefill(model, ids)
    _prefill(reference, ids)
    token = 5
    for step in range(4):
        chain.log.clear()
        out = _decode(
            model, token, len(ids) + step, sampling_params=_params(temperature=0.0), reset_sampling_state=step == 0
        )
        logits = _decode(reference, token, len(ids) + step)
        assert chain.log == _guarded_step_ops(token, (len(ids) + step) % RESIDUE_CLASSES) == reference_chain.log[-7:]
        assert out.shape == (1, 1) and out.dtype == torch.int32 and out.device.type == "cpu"
        assert int(out) == int(logits.argmax()) == int(_masked(chain.logits()).argmax()) < TOKENIZER_SIZE
        token = int(out)
    assert model.slot.committed == reference.slot.committed and model.slot.policy.greedy
    assert model.slot.generator is None  # no seed: the global RNG (unused by the greedy path)


def test_device_sampling_is_deterministic_per_seed_and_stays_inside_the_nucleus(monkeypatch) -> None:
    logged: list[str] = []
    monkeypatch.setattr(vllm_module, "logger", SimpleNamespace(info=logged.append, warning=lambda *_: None))
    row = torch.zeros(VOCAB_SIZE)
    row[1000:1040] = 10.0 + torch.arange(40) * 0.01  # 40 near-equal leaders: top-k 32 keeps 32, top-p 0.9 about 29
    sampled = _vllm_sampler_probabilities(_masked(row), temperature=0.7, top_k=32, top_p=0.9)
    support = set(torch.nonzero(sampled).flatten().tolist())
    assert 20 <= len(support) <= 32 and support <= set(range(1008, 1040))

    def run(seed, steps: int = 12, **params) -> list[int]:
        model, chain = _adapter()
        chain.logits = lambda: row.clone()  # the real read is a fresh host tensor per step; the adapter masks in place
        _prefill(model, [1, 2, 3])
        tokens = []
        for step in range(steps):
            out = _decode(
                model,
                tokens[-1] if tokens else 4,
                3 + step,
                sampling_params=_params(**{**dict(temperature=0.7, top_k=32, top_p=0.9, seed=seed), **params}),
            )
            tokens.append(int(out))
        assert model.slot.policy.seed == seed and (model.slot.generator is None) == (seed is None)
        return tokens

    first = run(7)
    assert first == run(7) and set(first) <= support
    assert run(8) != first and set(run(8)) <= support
    assert set(run(None)) <= support and set(run(None, top_p=1.0)) <= set(range(1008, 1040))
    assert set(run(3, top_k=VOCAB_SIZE)) <= set(range(1000, 1040))  # top-k off: the nucleus out of the full row
    assert run(3, top_k=1, top_p=0.0) == [1039] * 12  # the plugin's top_k == 1 form: the argmax, seed-independent
    # one log line per policy adoption: each run adopts once (a new request), the seeds differ between runs
    assert len(logged) == 8 and all("host sampler" in line for line in logged)


def test_device_sampling_reseeds_on_a_new_seed_or_reset_and_logs_once_per_policy(monkeypatch) -> None:
    logged: list[str] = []
    monkeypatch.setattr(vllm_module, "logger", SimpleNamespace(info=logged.append, warning=lambda *_: None))
    model, chain = _adapter()
    row = torch.zeros(VOCAB_SIZE)
    row[1000:1040] = 10.0
    chain.logits = lambda: row.clone()
    _prefill(model, [1, 2, 3])
    params = _params(temperature=1.0, top_k=40, top_p=1.0, seed=11)
    generator_state = lambda: model.slot.generator.get_state().clone()
    first = int(_decode(model, 4, 3, sampling_params=params))
    state = generator_state()
    assert len(logged) == 1 and model.slot.policy.seed == 11
    _decode(model, first, 4, sampling_params=params)  # the same policy: no log, the generator advanced
    assert len(logged) == 1 and not torch.equal(generator_state(), state)
    _decode(model, first, 5, sampling_params=params, reset_sampling_state=True)  # the plugin's reset: reseeded
    assert len(logged) == 1
    _decode(model, first, 6, sampling_params=_params(temperature=1.0, top_k=40, top_p=1.0, seed=12))
    assert len(logged) == 2 and model.slot.policy.seed == 12
    _prefill(model, [1, 2, 3])  # a new request: the slot forgets the policy and the generator
    assert model.slot.policy is None and model.slot.generator is None
    assert int(_decode(model, 4, 3, sampling_params=params)) == first and len(logged) == 3


def test_device_sampling_applies_vllm_penalties_from_the_contract_history() -> None:
    row = torch.zeros(VOCAB_SIZE)
    row[100], row[200], row[300] = 5.0, 4.0, -1.0
    history = dict(
        prompt_tokens=torch.tensor([[100, 300, -1, -1], [-1, -1, -1, -1]], dtype=torch.int32),  # batch pad row
        output_tokens=torch.tensor([[100, 100, -1], [-1, -1, -1]], dtype=torch.int32),
    )

    def greedy(**penalties) -> int:
        model, chain = _adapter()
        chain.logits = lambda: row.clone()  # the adapter penalizes the row it read in place
        _prefill(model, [1, 2, 3])
        return int(_decode(model, 4, 3, sampling_params=_params(temperature=0.0, **penalties), **history))

    assert greedy() == 100
    assert greedy(repetition_penalty=2.0) == 200  # 5 / 2 < 4
    assert greedy(frequency_penalty=0.6) == 200  # 5 - 2 x 0.6 < 4
    assert greedy(frequency_penalty=0.4) == 100  # 5 - 2 x 0.4 > 4
    assert greedy(presence_penalty=0.6) == 100  # presence counts once: 5 - 0.6 > 4
    assert greedy(presence_penalty=1.5) == 200

    policy = vllm_module.Qwen38HostSamplingPolicy(0.0, 20, 0.95, presence_penalty=0.7, frequency_penalty=0.3)
    policy = vllm_module.Qwen38HostSamplingPolicy(**{**vars(policy), "repetition_penalty": 1.8})
    prompt_ids, output_ids = vllm_module.penalty_history(**history)
    assert prompt_ids.tolist() == [100, 300] and output_ids.tolist() == [100, 100]
    penalized = vllm_module.apply_penalties(_masked(row.clone()), prompt_ids, output_ids, policy)  # in place
    expected = _vllm_penalized(
        _masked(row.clone()), prompt_ids, output_ids, presence=0.7, frequency=0.3, repetition=1.8
    )
    assert torch.equal(penalized, expected) and torch.isinf(penalized[TOKENIZER_SIZE:]).all()  # vLLM's expressions
    assert float(penalized[300]) == pytest.approx(-1.8) and float(penalized[100]) == pytest.approx(5 / 1.8 - 1.3)

    generator = torch.Generator().manual_seed(5)
    random_row = torch.randn(4096, generator=generator) * 3
    prompt_ids = torch.randint(0, 4096, (300,), generator=generator)
    output_ids = torch.randint(0, 64, (200,), generator=generator)
    penalized = vllm_module.apply_penalties(random_row.clone(), prompt_ids, output_ids, policy)
    expected = _vllm_penalized(random_row, prompt_ids, output_ids, presence=0.7, frequency=0.3, repetition=1.8)
    assert torch.equal(penalized, expected)


def test_host_steps_and_device_steps_interleave_with_the_full_logits_path_unchanged() -> None:
    model, chain = _adapter()
    ids = [1, 2, 3, 4]
    _prefill(model, ids)
    token = 5
    for step, sampled in enumerate((False, True, True, False)):
        chain.log.clear()
        params = _params(temperature=0.0) if sampled else None
        out = _decode(model, token, len(ids) + step, sampling_params=params)
        assert chain.log == _guarded_step_ops(token, (len(ids) + step) % RESIDUE_CLASSES)
        if sampled:
            assert out.shape == (1, 1) and out.dtype == torch.int32
        else:
            assert out.shape == (1, 1, VOCAB_SIZE) and out.dtype == torch.float32
            assert torch.equal(out[0, 0], _masked(chain.logits()))
        token = int(out.argmax())
    assert model.slot.committed == chain.inputs and len(model.slot.committed) == len(ids) + 4
    assert model.slot.policy.greedy  # the policy outlives the host steps; the next prefill clears it


def test_reduced_sampler_draws_the_vllm_sampler_distribution() -> None:
    """20k draws at T 0.7 / top-k 32 / top-p 0.9 from a peaked synthetic row against vLLM's algorithm transcribed."""

    vocab = 4096
    generator = torch.Generator().manual_seed(1234)
    row = torch.randn(vocab, generator=generator)
    row[torch.randperm(vocab, generator=generator)[:48]] = 5.0 + torch.rand(48, generator=generator) * 1.4  # peaks
    policy = vllm_module.Qwen38HostSamplingPolicy(temperature=0.7, top_k=32, top_p=0.9)
    expected = _vllm_sampler_probabilities(row, temperature=0.7, top_k=32, top_p=0.9)
    draws = 20_000
    draw_generator = torch.Generator().manual_seed(99)
    tokens = torch.tensor([vllm_module.sample_from_row(row, policy, draw_generator) for _ in range(draws)])
    counts = torch.bincount(tokens, minlength=vocab)
    assert int(counts[expected == 0].sum()) == 0 and 8 <= int((expected > 0).sum()) <= 32
    statistic, critical = _chi_square_ok(counts, expected, draws)
    assert statistic < critical, (statistic, critical)
    assert float((counts / draws - expected).abs().sum()) / 2 < 0.03


@pytest.mark.parametrize(
    "top_k, top_p, candidates, path",
    (
        (0, 0.9, 64, "candidates hold the nucleus"),
        (0, 0.9, 8, "nucleus wider than the candidates: full row"),
        (0, 1.0, 64, "top-k off, top-p off: full row without a sort"),
        (200, 0.8, 64, "top-k wider than the candidates: full row"),
        (40, 0.999, 64, "ties at the k-th value are kept"),
    ),
)
def test_reduced_sampler_paths_agree_with_the_vllm_algorithm(top_k, top_p, candidates, path) -> None:
    vocab = 1024
    generator = torch.Generator().manual_seed(4321)
    row = torch.randn(vocab, generator=generator) * 1.5
    row[:40] = 4.0  # forty tied leaders
    policy = vllm_module.Qwen38HostSamplingPolicy(temperature=1.0, top_k=top_k, top_p=top_p)
    expected = _vllm_sampler_probabilities(row, temperature=1.0, top_k=top_k, top_p=top_p)
    draws = 4_000
    draw_generator = torch.Generator().manual_seed(7)
    tokens = torch.tensor(
        [vllm_module.sample_from_row(row, policy, draw_generator, candidates=candidates) for _ in range(draws)]
    )
    counts = torch.bincount(tokens, minlength=vocab)
    assert int(counts[expected == 0].sum()) == 0, path
    statistic, critical = _chi_square_ok(counts, expected, draws)
    assert statistic < critical, (path, statistic, critical)
    if top_k == 40:
        assert set(range(40)) <= set(tokens.tolist())  # the 40 tied values are all the k-th value: every one kept


# -- refusals, all before any chain call -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"sampling_params": _params()}, "decode_only"),
        ({"start_pos": [5]}, "position 0"),
        ({"empty_slots": [1]}, "empty_slots"),
        ({"empty_slots": []}, "empty_slots"),
        ({"tokens": torch.zeros((2, 4), dtype=torch.int32)}, r"\[1, P\]"),
        ({"prompt_lens": [0]}, "prompt length"),
        ({"prompt_lens": [5]}, "prompt length"),  # longer than the tokens row
        ({"tokens": torch.tensor([[1, TOKENIZER_SIZE, 2, 3]], dtype=torch.int32)}, "prompt ids"),
        ({"tokens": torch.tensor([[1, -1, 2, 3]], dtype=torch.int32)}, "prompt ids"),
    ),
)
def test_prefill_refusals_happen_before_any_chain_call(overrides, message, expect_error) -> None:
    model, chain = _adapter()
    with expect_error(ValueError, message):
        _prefill(model, [1, 2, 3, 4], **overrides)
    assert chain.log == [] and model.slot.failed is False


def test_prefill_admits_the_context_limit_and_refuses_one_past_it(expect_error) -> None:
    model, chain = _adapter()
    assert model.context_limit == MAX_MODEL_LEN == 32_704
    with expect_error(ValueError, "prompt length"):
        _prefill(model, [1] * (model.context_limit + 1))
    assert chain.log == []
    _prefill(model, [1] * model.context_limit)
    assert len(model.slot.committed) == model.context_limit and chain.chunks[0][1] == 0


@pytest.mark.parametrize(
    "overrides, error, message",
    (
        ({"reset_batch": True}, TypeError, "reset_batch"),
        ({"reload_inputs": False}, ValueError, "reload_inputs"),
        ({"slot_remap": [1]}, ValueError, "slot_remap"),
        ({"slot_remap": [0, 1]}, ValueError, "slot_remap"),
        ({"tokens": torch.zeros((2, 1), dtype=torch.int32)}, ValueError, r"\[1, 1\]"),
        ({"tokens": torch.tensor([[TOKENIZER_SIZE]], dtype=torch.int32)}, ValueError, "decode token"),
        ({"start_pos": [3]}, Qwen38ChatChainError, "runner position 3 vs device position 4"),
        ({"sampling_params": SimpleNamespace(**{**vars(_params()), "top_k": [20, 20]})}, ValueError, "one row"),
        ({"sampling_params": _params(enable_log_probs=True)}, ValueError, "logprobs"),
        ({"sampling_params": _params(top_p=1.5)}, ValueError, "out of range"),
        ({"sampling_params": _params(temperature=-1.0)}, ValueError, "out of range"),
        ({"sampling_params": _params(repetition_penalty=1.5)}, ValueError, "history"),
        (
            {"sampling_params": _params(presence_penalty=0.5), "prompt_tokens": torch.zeros((1, 2), dtype=torch.int32)},
            ValueError,
            "history",
        ),
        (
            {
                "sampling_params": _params(frequency_penalty=0.5),
                "prompt_tokens": torch.zeros(3, dtype=torch.int32),
                "output_tokens": torch.zeros((1, 2), dtype=torch.int32),
            },
            ValueError,
            r"\[rows, L\]",
        ),
    ),
)
def test_decode_refusals_happen_before_any_chain_call(overrides, error, message, expect_error) -> None:
    model, chain = _adapter()
    _prefill(model, [1, 2, 3, 4])
    chain.log.clear()
    with expect_error(error, message):
        _decode(model, 5, 4, **overrides)
    assert chain.log == [] and model.slot.failed is False and model.slot.committed == [1, 2, 3, 4]
    assert model.slot.policy is None and model.slot.generator is None


def test_prefill_refuses_a_chunk_driver_that_stopped_short(expect_error) -> None:
    model, chain = _adapter()
    chain.chunk_stopped = "budget"
    with expect_error(Qwen38ChatChainError, "stopped 'budget'"):
        _prefill(model, [1000 + index for index in range(20)])
    assert model.slot.failed is True


def test_a_failed_chain_call_refuses_decodes_until_a_prefill_resets_the_device(expect_error) -> None:
    model, chain = _adapter()
    _prefill(model, [1, 2, 3])

    def broken_tail(residue: int) -> None:
        raise RuntimeError("device lost")

    chain.execute_tail = broken_tail
    with expect_error(RuntimeError, "device lost"):
        _decode(model, 4, 3)
    assert model.slot.failed is True
    chain.log.clear()
    with expect_error(Qwen38ChatChainError, "failed earlier"):
        _decode(model, 4, 3)
    with expect_error(RuntimeError, "device lost"):  # the prefill retries the device; the fault persists
        _prefill(model, [1, 2])
    assert model.slot.failed is True
    model.release_request(0)
    assert model.slot.failed is True
    model.release_persistent_capture()
    assert chain.closed is False
    del chain.execute_tail  # the device answers again: the next prefill's reset re-verifies it and clears the flag
    chain.log.clear()
    _prefill(model, [1, 2])
    assert model.slot.failed is False and model.slot.committed == [1, 2]
    _decode(model, 5, 2)
    model.release_persistent_capture()
    assert chain.closed is True


def test_trace_mode_off_is_not_honoured_and_warns_once(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(
        vllm_module, "logger", SimpleNamespace(info=lambda *_: None, warning=lambda message: warnings.append(message))
    )
    model, chain = _adapter()
    _prefill(model, [1, 2, 3], enable_trace=False)
    _decode(model, 4, 3, enable_trace=False)
    assert len(warnings) == 1 and "trace_mode" in warnings[0]
    assert [op for op in chain.log if op.startswith("tail:")] == ["tail:0", "tail:1", "tail:2", "tail:3"]


# -- warmup and release ------------------------------------------------------------------------------------------------


def test_warmups_check_the_chain_and_phase_one_touches_nothing(expect_error) -> None:
    model, chain = _adapter()
    decode_kwargs = dict(kv_cache=None, max_batch_size=1, num_blocks=512, can_sample_on_device=False)
    model.warmup_model_prefill(enable_trace=False, kv_cache=None, can_sample_on_device=False)
    model.warmup_model_prefill(enable_trace=True, kv_cache=None, can_sample_on_device=False)
    model.warmup_model_decode(enable_trace=False, **decode_kwargs)
    # decode_only: the plugin's decode flag is served (the sampled steps draw on the host); the prefill flag (mode
    # all: prefill would return tokens) is refused
    model.warmup_model_decode(enable_trace=False, **{**decode_kwargs, "can_sample_on_device": True})
    assert chain.log == []
    with expect_error(ValueError, "decode_only, not all"):
        model.warmup_model_prefill(enable_trace=True, kv_cache=None, can_sample_on_device=True)
    with expect_error(ValueError, "max_batch_size"):
        model.warmup_model_decode(enable_trace=True, **{**decode_kwargs, "max_batch_size": 2})
    chain.misses_forbidden = False
    with expect_error(Qwen38ChatChainError, "miss guard"):
        model.warmup_model_prefill(enable_trace=True, kv_cache=None, can_sample_on_device=False)
    chain.misses_forbidden = True
    chain.closed = True
    with expect_error(Qwen38ChatChainError, "miss guard"):
        model.warmup_model_decode(enable_trace=True, **decode_kwargs)
    assert chain.log == []


def test_decode_warmup_phase_two_runs_both_paths_once_and_resets() -> None:
    model, chain = _adapter()
    prompt = list(vllm_module.WARMUP_PROMPT_TOKEN_IDS)
    assert len(prompt) == 40 and set(prompt) <= set(WARM_CHUNK_TOKEN_IDS)
    assert chunk_accepts(len(prompt) - 1) == [CHUNK_ROWS - 1, 6]  # one full chunk and a 7-row padded tail
    token = int(_masked(sampling_tests.FakeChain(prompt).logits()).argmax())

    model.warmup_model_decode(
        enable_trace=True, kv_cache=None, max_batch_size=1, num_blocks=512, can_sample_on_device=False
    )

    assert chain.log == [
        f"reset:{SEED_TOKEN_ID}",
        f"chunk_prefill:{len(prompt) - 1}@0",
        *_guarded_step_ops(prompt[-1], (len(prompt) - 1) % RESIDUE_CLASSES),
        *_guarded_step_ops(token, len(prompt) % RESIDUE_CLASSES),
        f"reset:{SEED_TOKEN_ID}",
    ]
    assert model.slot == Qwen38VLLMSlot() and chain.inputs == []


def test_release_request_clears_the_mirror_without_a_device_call(expect_error) -> None:
    model, chain = _adapter()
    _prefill(model, [1, 2, 3])
    chain.log.clear()
    with expect_error(ValueError, "slot must be 0"):
        model.release_request(1)
    model.release_request(0)
    assert model.slot == Qwen38VLLMSlot() and chain.log == []
    _prefill(model, [4, 5])  # the next request resets the device itself
    assert chain.log[0] == f"reset:{SEED_TOKEN_ID}" and model.slot.committed == [4, 5]


def test_release_persistent_capture_closes_the_chain() -> None:
    model, chain = _adapter()
    model.release_persistent_capture()
    assert chain.log == ["close"] and chain.closed


def test_allocate_kv_cache_allocates_nothing() -> None:
    model, chain = _adapter()
    assert model.allocate_kv_cache((512, 1, 64, 256), torch.bfloat16, 48) is None and chain.log == []


# -- the real chunk driver ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("length", (17, 33, 100))
def test_prefill_drives_the_real_chunk_driver_from_position_zero(driver_chain, length: int) -> None:
    model = Qwen38ForCausalLM(driver_chain, allocated_context=ALLOCATED_CONTEXT)
    ids = [1000 + index for index in range(length)]
    head, last = ids[:-1], ids[-1]
    accepts = chunk_accepts(len(head))

    logits, _ = _prefill(model, ids)

    calls = driver_chain.model.calls
    assert calls[0] == ("reset_chunk_state_inplace", "state", "chunk_state")
    assert calls[-1] == ("finish_prefill", len(head))
    assert [call[1] for call in calls if call[0] == "write_chunk_accepted"] == [a for a in accepts if a != 31]
    inputs = [call for call in calls if call[0] == "prepare_chunk_inputs"]
    assert len(inputs) == len(accepts) == driver_chain.log.count("chunk_replay") and inputs[0][2] is None
    assert [call[1] for call in calls if call[0] == "upload_chunk_inputs"] == [tokens for _, tokens, _ in inputs]
    rows = [token for _, tokens, _ in inputs for token in tokens]
    assert rows[: len(head)] == head and set(rows[len(head) :]) <= {CHUNK_PAD_TOKEN_ID}
    # No alignment step from position 0: the only forced step is the last prompt token, under the guard.
    guard = driver_chain.log.index("guard")
    assert driver_chain.log[:2] == [f"reset:{SEED_TOKEN_ID}", "synchronize"]
    assert [op for op in driver_chain.log[:guard] if op.startswith("write:")] == []
    assert driver_chain.log[guard:] == _guarded_step_ops(last, (length - 1) % RESIDUE_CLASSES)
    assert model.slot.committed == ids and driver_chain.inputs == ids
    assert model.slot.ple_context == (last, driver_tests._expected_context(head, None)[0])
    assert torch.equal(logits[0, 0], _masked(driver_chain.logits()))


# -- construction ------------------------------------------------------------------------------------------------------


def _fake_ttnn(*, tracking: bool) -> SimpleNamespace:
    return SimpleNamespace(Arch=ttnn.Arch, Topology=ttnn.Topology, _ttnn=ttnn._ttnn, __file__=ttnn.__file__)


def _mesh(shape=(1, 4), count: int = 4, arch=None, ids=(1, 0, 2, 3)) -> SimpleNamespace:
    return SimpleNamespace(
        shape=shape,
        get_num_devices=lambda: count,
        arch=lambda: ttnn.Arch.BLACKHOLE if arch is None else arch,
        get_device_ids=lambda: list(ids),
    )


@pytest.fixture
def construction(monkeypatch, tmp_path) -> SimpleNamespace:
    """The library's construction functions replaced by recorders; the fake prepare keeps the cache-root check.  The
    hub cache holds the pinned snapshot only once a test creates ``snapshot``."""

    calls: list[tuple] = []
    hub: list[tuple] = []
    chain = FakeChain()
    snapshot = tmp_path / "hub" / "snapshots" / PINNED_CHECKPOINT_REVISION

    def snapshot_download(repo_id, **kwargs):
        hub.append((repo_id, kwargs))
        if not snapshot.is_dir():
            raise FileNotFoundError(f"{repo_id} is not cached")  # LocalEntryNotFoundError is a FileNotFoundError
        return str(snapshot)

    def prepare(**kwargs):
        Qwen38CacheRoots(
            component_weights=kwargs["component_cache_root"],
            routed_bf4=kwargs["routed_bf4_scratch_root"],
            model_io=kwargs["model_io_cache_root"],
        )
        calls.append(("prepare", kwargs))
        return "prepared"

    def construct(prepared, **kwargs):
        assert os.environ.get("QWEN38_HARDWARE_MODE") == MODE  # the gate the real construction reads at call time
        calls.append(("construct", prepared, kwargs))
        return "construction"

    def open_chain(constructed, **kwargs):
        calls.append(("open", constructed, kwargs))
        return chain

    monkeypatch.setattr(vllm_module, "prepare_live_decode_diagnostic", prepare)
    monkeypatch.setattr(vllm_module, "construct_live_decode_diagnostic", construct)
    monkeypatch.setattr(vllm_module, "Qwen38TracedChain", SimpleNamespace(open=open_chain))
    monkeypatch.setattr(vllm_module, "sha256_of", lambda path: calls.append(("sha256_of", path)) or "a" * 64)
    monkeypatch.setattr(
        vllm_module, "git_identity", lambda root: calls.append(("git_identity", root)) or {"head": "b" * 40}
    )
    monkeypatch.setattr(vllm_module, "ttnn", _fake_ttnn(tracking=True))
    monkeypatch.setattr(vllm_module, "TRACE_ALLOC_TRACKING", True)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(checkpoint))
    monkeypatch.setenv("QWEN38_CACHE_ROOT", str(tmp_path / "cache"))
    for name in (
        "QWEN38_CACHE_LABEL",
        "QWEN38_BF4_CORPUS",
        "QWEN38_BF4_CORPUS_VERIFICATION",
        "QWEN38_VLLM_ALLOW_BF4_CONVERSION",
        "QWEN38_LONG_CHUNKS",
        "QWEN38_PREFILL_SLAB",
        "QWEN38_HARDWARE_MODE",
        "QWEN38_TT_METAL_SHA",
        "QWEN38_HF_REPO",
    ):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(
        calls=calls,
        hub=hub,
        chain=chain,
        checkpoint=checkpoint,
        snapshot=snapshot,
        caches=tmp_path / "cache" / "caches",
    )


def _initialize(construction, **overrides) -> Qwen38ForCausalLM:
    kwargs = dict(
        hf_config=None,
        mesh_device=_mesh(),
        max_batch_size=1,
        max_seq_len=MAX_MODEL_LEN,
        tt_data_parallel=1,
        optimizations=None,
    )
    kwargs.update(overrides)
    return Qwen38ForCausalLM.initialize_vllm_model(**kwargs)


def _recorded(construction) -> tuple[dict, dict, dict]:
    """The keyword arguments of the one prepare, construct and open call, in that order."""

    names = [call[0] for call in construction.calls]
    assert names[-3:] == ["prepare", "construct", "open"]
    assert names[:-3] in (["sha256_of"], ["sha256_of", "git_identity"])  # no git call once QWEN38_TT_METAL_SHA is set
    prepared, constructed, opened = construction.calls[-3:]
    assert constructed[1] == "prepared" and opened[1] == "construction"
    return prepared[1], constructed[2], opened[2]


@pytest.mark.parametrize(
    "max_seq_len, allocated",
    (
        (1, 8_192),
        (8_128, 8_192),
        (8_129, 32_768),
        (32_704, 32_768),
        (32_705, 65_536),
        (65_472, 65_536),
        (131_008, 131_072),
        (262_080, 262_144),
    ),
)
def test_construction_picks_the_smallest_resident_context_that_fits(
    construction, max_seq_len: int, allocated: int
) -> None:
    model = _initialize(construction, max_seq_len=max_seq_len)
    assert model.allocated_context == allocated and model.context_limit == allocated - RESIDENT_CONTEXT_HEADROOM
    assert model.context_limit >= max_seq_len
    prepared, _, _ = _recorded(construction)
    assert prepared["allocated_context"] == allocated
    assert prepared["component_cache_root"] == construction.caches / f"c{allocated}-vllm" / "components"


def test_construction_refuses_a_context_no_resident_build_fits(construction, expect_error) -> None:
    with expect_error(ValueError, "8128, 32704, 65472, 131008, 262080"):
        _initialize(construction, max_seq_len=262_081)
    assert construction.calls == []


def test_construction_order_and_forwarded_arguments(construction, monkeypatch) -> None:
    monkeypatch.setenv("QWEN38_BF4_CORPUS", "/corpus/root")
    monkeypatch.setenv("QWEN38_BF4_CORPUS_VERIFICATION", "/corpus/verification.json")
    mesh = _mesh(ids=(1, 0, 2, 3))

    model = _initialize(construction, mesh_device=mesh, optimizations="performance")

    prepared, constructed, opened = _recorded(construction)
    extension = Path(ttnn._ttnn.__file__).resolve()
    digests = dict(construction.calls[:2])
    assert digests["sha256_of"] == extension and digests["git_identity"] == Path(ttnn.__file__).resolve().parents[2]
    assert prepared == {
        "checkpoint_root": construction.checkpoint,
        "component_cache_root": construction.caches / "c32768-vllm" / "components",
        "routed_bf4_scratch_root": construction.caches / "bf4-experts",
        "model_io_cache_root": construction.caches / "c32768-vllm" / "model-io",
        "tt_metal_sha": "b" * 40,
        "runtime_extension": extension,
        "runtime_sha256": "a" * 64,
        "allocated_context": ALLOCATED_CONTEXT,
        "physical_ids": (1, 0, 2, 3),
        "bf4_corpus_root": Path("/corpus/root"),
        "bf4_corpus_verification": Path("/corpus/verification.json"),
        "marker": prepared["marker"],
    }
    assert constructed == {
        "mesh_device": mesh,
        "collective_topology": ttnn.Topology.Linear,
        "marker": constructed["marker"],
        "stage_missing_bf4": False,
    }
    assert DEFAULT_SLAB_ROWS == 2048  # the measured form (docs/PREFILL.md), the default slab
    assert opened == {
        "marker": opened["marker"],
        "chunked_prefill": True,
        "sampling": True,
        "long_chunks": True,
        "slab_rows": DEFAULT_SLAB_ROWS,
    }
    assert all(callable(kwargs["marker"]) for kwargs in (prepared, constructed, opened))
    assert os.environ["QWEN38_HARDWARE_MODE"] == MODE
    assert model.chain is construction.chain and model.slot == Qwen38VLLMSlot()
    assert model.allocated_context == ALLOCATED_CONTEXT and model.context_limit == MAX_MODEL_LEN


@pytest.mark.parametrize("switch, expected", ((None, False), ("0", False), ("1", True)))
def test_construction_honours_the_label_and_the_conversion_and_long_chunk_switches(
    construction, monkeypatch, switch: str | None, expected: bool
) -> None:
    monkeypatch.setenv("QWEN38_CACHE_LABEL", "c32768-custom")
    monkeypatch.setenv("QWEN38_PREFILL_SLAB", "0")  # no slab: the long-chunk switch alone decides the 128-row chunks
    for name in ("QWEN38_VLLM_ALLOW_BF4_CONVERSION", "QWEN38_LONG_CHUNKS"):
        if switch is not None:
            monkeypatch.setenv(name, switch)
    _initialize(construction, optimizations="accuracy")
    prepared, constructed, opened = _recorded(construction)
    assert prepared["component_cache_root"] == construction.caches / "c32768-custom" / "components"
    assert prepared["model_io_cache_root"] == construction.caches / "c32768-custom" / "model-io"
    assert prepared["routed_bf4_scratch_root"] == construction.caches / "bf4-experts"
    assert prepared["bf4_corpus_root"] is None and prepared["bf4_corpus_verification"] is None
    assert constructed["stage_missing_bf4"] is expected and opened["long_chunks"] is expected
    assert opened["slab_rows"] is None


@pytest.mark.parametrize(
    "pinned, failure, expected, source",
    (
        ("c" * 40, None, "c" * 40, "QWEN38_TT_METAL_SHA"),
        (None, None, "b" * 40, "checkout head"),
        (None, RuntimeAdmissionError("no .git"), DIGEST[:40], "runtime extension digest"),
        (None, FileNotFoundError("git"), DIGEST[:40], "runtime extension digest"),
    ),
)
def test_construction_resolves_the_cache_identity_in_order(
    construction, monkeypatch, pinned: str | None, failure: Exception | None, expected: str, source: str
) -> None:
    def git_identity(root):
        construction.calls.append(("git_identity", root))
        if failure is not None:
            raise failure
        return {"head": "b" * 40}

    messages: list[str] = []
    monkeypatch.setattr(vllm_module, "git_identity", git_identity)
    monkeypatch.setattr(vllm_module, "sha256_of", lambda path: construction.calls.append(("sha256_of", path)) or DIGEST)
    monkeypatch.setattr(vllm_module, "logger", SimpleNamespace(info=lambda message: messages.append(message)))
    if pinned is not None:
        monkeypatch.setenv("QWEN38_TT_METAL_SHA", pinned)
    _initialize(construction)
    prepared, _, _ = _recorded(construction)
    assert prepared["tt_metal_sha"] == expected and prepared["runtime_sha256"] == DIGEST
    assert len(expected) == 40 and set(expected) <= set("0123456789abcdef")
    assert ("git_identity" in [call[0] for call in construction.calls]) is (pinned is None)
    assert [message for message in messages if message.startswith("tt_metal_sha")] == [
        f"tt_metal_sha {expected} ({source})"
    ]


def test_construction_refuses_a_malformed_pinned_sha(construction, monkeypatch, expect_error) -> None:
    monkeypatch.setenv("QWEN38_TT_METAL_SHA", "B" * 40)
    with expect_error(ValueError, "QWEN38_TT_METAL_SHA must be lowercase 40-hex"):
        _initialize(construction)
    assert construction.calls == []


def test_construction_resolves_the_checkpoint_in_order(construction, monkeypatch, tmp_path) -> None:
    repo = SimpleNamespace(_name_or_path=HF_REPO_ID)
    _initialize(construction, hf_config=repo)  # MODEL_WEIGHTS_DIR wins over the repo id
    assert _recorded(construction)[0]["checkpoint_root"] == construction.checkpoint and construction.hub == []
    monkeypatch.delenv("MODEL_WEIGHTS_DIR")
    directory = tmp_path / "snapshot"
    directory.mkdir()
    construction.calls.clear()
    _initialize(construction, hf_config=SimpleNamespace(_name_or_path=str(directory)))  # a --model directory
    assert _recorded(construction)[0]["checkpoint_root"] == directory and construction.hub == []
    construction.snapshot.mkdir(parents=True)
    pinned = {"revision": PINNED_CHECKPOINT_REVISION, "local_files_only": True}
    for hf_config, env, repo_id in (
        (repo, None, HF_REPO_ID),
        (None, None, HF_REPO_ID),
        (repo, "org/mirror", "org/mirror"),
    ):
        if env is not None:
            monkeypatch.setenv("QWEN38_HF_REPO", env)
        construction.calls.clear()
        construction.hub.clear()
        _initialize(construction, hf_config=hf_config)  # a repo id: the pinned revision from the local hub cache
        assert _recorded(construction)[0]["checkpoint_root"] == construction.snapshot
        assert construction.hub == [(repo_id, pinned)]


def test_construction_refuses_a_checkpoint_the_hub_cache_does_not_hold(construction, monkeypatch, expect_error) -> None:
    monkeypatch.delenv("MODEL_WEIGHTS_DIR")
    with pytest.raises(ValueError) as refusal:  # allow-pytest.raises: the test reads the exception object
        _initialize(construction, hf_config=SimpleNamespace(_name_or_path=HF_REPO_ID))
    message = str(refusal.value)
    assert all(part in message for part in (HF_REPO_ID, PINNED_CHECKPOINT_REVISION, "MODEL_WEIGHTS_DIR", "HF_HOME"))
    assert isinstance(refusal.value.__cause__, FileNotFoundError) and len(construction.hub) == 1
    monkeypatch.setenv("MODEL_WEIGHTS_DIR", str(construction.checkpoint / "missing"))
    with expect_error(ValueError, "MODEL_WEIGHTS_DIR must name the checkpoint directory"):
        _initialize(construction, hf_config=SimpleNamespace(_name_or_path=HF_REPO_ID))  # no fall-through to the hub
    assert len(construction.hub) == 1 and construction.calls == []


def test_construction_requires_the_cache_root_and_disjoint_cache_roots(construction, monkeypatch, expect_error) -> None:
    monkeypatch.delenv("QWEN38_CACHE_ROOT")
    with expect_error(ValueError, "QWEN38_CACHE_ROOT"):
        _initialize(construction)
    assert construction.calls == []
    monkeypatch.setenv("QWEN38_CACHE_ROOT", str(construction.caches.parent))
    monkeypatch.setenv("QWEN38_CACHE_LABEL", "bf4-experts")  # puts the component root under the BF4 root
    with expect_error(ValueError, "disjoint"):
        _initialize(construction)
    assert [call[0] for call in construction.calls if call[0] in ("prepare", "construct", "open")] == []


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"max_batch_size": 2}, "max_num_seqs 1"),
        ({"tt_data_parallel": 2}, "data parallelism"),
        ({"optimizations": "fast"}, "optimizations"),
        ({"mesh_device": _mesh(shape=(4, 1))}, "mesh of 4 devices"),
        ({"mesh_device": _mesh(count=3)}, "mesh of 4 devices"),
        ({"mesh_device": _mesh(arch="wormhole")}, "Blackhole"),
    ),
)
def test_construction_refusals_come_before_any_library_call(construction, overrides, message, expect_error) -> None:
    with expect_error(ValueError, message):
        _initialize(construction, **overrides)
    assert construction.calls == []


def test_construction_requires_trace_allocation_tracking(construction, monkeypatch, expect_error) -> None:
    monkeypatch.setattr(vllm_module, "ttnn", _fake_ttnn(tracking=False))
    monkeypatch.setattr(vllm_module, "TRACE_ALLOC_TRACKING", False)
    with expect_error(ValueError, "TT_METAL_TRACE_ALLOC_TRACKING=1"):
        _initialize(construction)
    assert construction.calls == []


# -- the prefill form: the slab rows and the 128-row chunks the chain is opened with -----------------------------------


@pytest.mark.parametrize(
    "slab, long_chunks, expected",
    (
        (None, None, (True, DEFAULT_SLAB_ROWS)),
        ("", None, (True, DEFAULT_SLAB_ROWS)),
        (None, "1", (True, DEFAULT_SLAB_ROWS)),
        (None, "0", (True, DEFAULT_SLAB_ROWS)),
        ("0", None, (False, None)),
        ("off", None, (False, None)),
        ("OFF", "0", (False, None)),
        ("0", "1", (True, None)),
        ("0", " 1 ", (True, None)),
        ("1024", None, (True, 1024)),
        (" 256 ", "0", (True, 256)),
        ("4096", "1", (True, 4096)),
    ),
)
def test_construction_opens_the_chain_in_the_prefill_form_the_knobs_name(
    construction, monkeypatch, slab: str | None, long_chunks: str | None, expected: tuple[bool, int | None]
) -> None:
    messages: list[str] = []
    monkeypatch.setattr(vllm_module, "logger", SimpleNamespace(info=messages.append, warning=lambda *_: None))
    environ: dict[str, str] = {}
    if slab is not None:
        monkeypatch.setenv("QWEN38_PREFILL_SLAB", slab)
        environ["QWEN38_PREFILL_SLAB"] = slab
    if long_chunks is not None:
        monkeypatch.setenv("QWEN38_LONG_CHUNKS", long_chunks)
        environ["QWEN38_LONG_CHUNKS"] = long_chunks
    assert vllm_module.prefill_form(environ) == expected  # the parser alone, on the given mapping
    _initialize(construction)
    _, _, opened = _recorded(construction)
    assert (opened["long_chunks"], opened["slab_rows"]) == expected
    assert opened["chunked_prefill"] is True and opened["sampling"] is True
    expected_long_chunks, expected_slab_rows = expected
    form = [message for message in messages if message.startswith("prefill form: ")]
    assert len(form) == 1 and form[0].endswith("32-row chunks")  # one line after the open states the form
    assert (f"{expected_slab_rows}-row slabs, then " in form[0]) is (expected_slab_rows is not None)
    assert ("128-row chunks, then 32-row chunks" in form[0]) is expected_long_chunks


def test_prefill_form_admits_every_slab_row_count_of_the_contract() -> None:
    for rows in range(256, 4096 + 1, 128):
        assert vllm_module.prefill_form({"QWEN38_PREFILL_SLAB": str(rows)}) == (True, rows)
    assert vllm_module.describe_prefill_form(True, 2048) == "2048-row slabs, then 128-row chunks, then 32-row chunks"
    assert vllm_module.describe_prefill_form(True, None) == "128-row chunks, then 32-row chunks"
    assert vllm_module.describe_prefill_form(False, None) == "32-row chunks"


@pytest.mark.parametrize(
    "value", ("100", "abc", "32", "128", "4224", "-256", "1024.0", "2048 rows", "none", "1", "\uff11\uff10\uff12\uff14")
)  # the last: fullwidth digits 1024, str.isdigit() true, not ASCII
def test_construction_refuses_a_slab_row_count_outside_the_contract(
    construction, monkeypatch, value: str, expect_error
) -> None:
    monkeypatch.setenv("QWEN38_PREFILL_SLAB", value)
    with expect_error(ValueError, "QWEN38_PREFILL_SLAB must be the slab's rows, a multiple of 128 in 256..4096"):
        _initialize(construction)
    assert construction.calls == []  # refused before any library call
    with expect_error(ValueError, "or 0 / off for no slab, got " + re.escape(repr(value))):
        vllm_module.prefill_form({"QWEN38_PREFILL_SLAB": value})


def test_construction_names_the_slab_knob_when_the_dram_admission_refuses_the_slab(construction, monkeypatch) -> None:
    refusal = ValueError(
        "QWEN38_PREFILL_DENSE_DTYPE=bf8 / GRID=wide refused: the resident prefill weights need 1177 MiB per device "
        "plus 380 MiB of context state, the 2048-row slab's 1900 MiB working set and a 256 MiB margin = 3713 MiB, "
        "but 3100 MiB are free after the resident build"
    )
    other = ValueError("target graph has 47 layers, expected 48")
    # A resident build failure whose owner cleanup also failed arrives wrapped (BF4CleanupError, a RuntimeError) with
    # the primary error as its cause (ttnn/builder.py): the refusal must be recognised one level down.
    wrapped = RuntimeError("target build failed and resident BF4 cleanup also failed; unreleased_tensor_slots=3")
    wrapped.__cause__ = refusal
    unrelated_wrapped = RuntimeError("target build failed and resident BF4 cleanup also failed")
    unrelated_wrapped.__cause__ = other
    raised: dict[str, BaseException | None] = {"error": refusal}
    messages: list[str] = []
    monkeypatch.setattr(vllm_module, "logger", SimpleNamespace(info=messages.append, warning=lambda *_: None))

    def open_chain(constructed, **kwargs):
        construction.calls.append(("open", constructed, kwargs))
        if raised["error"] is not None:
            raise raised["error"]
        return construction.chain

    monkeypatch.setattr(vllm_module, "Qwen38TracedChain", SimpleNamespace(open=open_chain))
    # The mark is the admission function's own refusal text (ttnn/prefill_dense.py).
    assert vllm_module.PREFILL_DENSE_REFUSAL_MARK in inspect.getsource(prefill_dense.admit_prefill_dense_dram)
    assert vllm_module.is_prefill_dense_refusal(refusal) and not vllm_module.is_prefill_dense_refusal(other)
    assert not vllm_module.is_prefill_dense_refusal(RuntimeError(str(refusal)))  # the type is part of the mark
    assert vllm_module.prefill_dense_refusal(wrapped) is refusal  # one level down
    assert not vllm_module.is_prefill_dense_refusal(unrelated_wrapped)
    with pytest.raises(ValueError) as error:  # allow-pytest.raises: the test reads the exception object
        _initialize(construction)
    message = str(error.value)
    assert message.startswith("the 2048-row prefill slab does not fit beside the resident build at context 32768: ")
    assert str(refusal) in message and "QWEN38_PREFILL_SLAB=0 serves this context without the slab" in message
    assert "QWEN38_LONG_CHUNKS=1 keeps the 128-row chunks" in message and error.value.__cause__ is refusal
    assert _recorded(construction)[2]["slab_rows"] == DEFAULT_SLAB_ROWS  # the open was attempted with the slab
    assert not [line for line in messages if line.startswith("prefill form: ")]  # a refused start logs no form
    # The wrapped form: the hint survives, the message carries the wrapper and the cause, the chain is kept.
    raised["error"] = wrapped
    construction.calls.clear()
    with pytest.raises(ValueError) as error:  # allow-pytest.raises: the test reads the exception object
        _initialize(construction)
    message = str(error.value)
    assert "QWEN38_PREFILL_SLAB=0 serves this context without the slab" in message
    assert str(wrapped) in message and f"(cause: {refusal})" in message
    assert error.value.__cause__ is wrapped and wrapped.__cause__ is refusal
    # A wrapped unrelated failure is not the admission's: it propagates as it is.
    raised["error"] = unrelated_wrapped
    construction.calls.clear()
    with pytest.raises(RuntimeError) as wrapped_error:  # allow-pytest.raises: the test reads the exception object
        _initialize(construction)
    assert wrapped_error.value is unrelated_wrapped
    # Another ValueError out of open is not the admission's: it propagates as it is.
    raised["error"] = other
    construction.calls.clear()
    with pytest.raises(ValueError) as error:  # allow-pytest.raises: the test reads the exception object
        _initialize(construction)
    assert error.value is other
    # Without a slab nothing is re-worded, whatever the text.
    monkeypatch.setenv("QWEN38_PREFILL_SLAB", "off")
    raised["error"] = refusal
    construction.calls.clear()
    with pytest.raises(ValueError) as error:  # allow-pytest.raises: the test reads the exception object
        _initialize(construction)
    assert error.value is refusal
    raised["error"] = None
    construction.calls.clear()
    model = _initialize(construction)  # the knob off: the same construction opens without the slab
    assert _recorded(construction)[2] == {
        "marker": _recorded(construction)[2]["marker"],
        "chunked_prefill": True,
        "sampling": True,
        "long_chunks": False,
        "slab_rows": None,
    }
    assert model.chain is construction.chain
    assert [line for line in messages if line.startswith("prefill form: ")] == ["prefill form: 32-row chunks"]


# -- the class surface and the bundle ----------------------------------------------------------------------------------


def test_class_surface_matches_the_plugin_contract(expect_error) -> None:
    cls = Qwen38ForCausalLM
    assert cls.model_capabilities == {
        "fabric_config": {
            "config": ttnn.FabricConfig.FABRIC_1D,
            "reliability_mode": ttnn.FabricReliabilityMode.STRICT_INIT,
            "fabric_tensix_config": ttnn.FabricTensixConfig.DISABLED,
        },
        "supports_device_penalties": True,
        "supports_chunked_prefill": False,
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,  # the plugin's sample_on_device_mode gate (platform.py); decode_only only
    }
    assert cls.decode_input_update_contract == 1
    for name in (
        "already_warmed_up_prefill",
        "is_hybrid",
        "get_kv_cache_spec",
        "tt_supported_decode_batch_sizes",
        "read_decode_output",
        "process_decode_output_host",
    ):
        assert not hasattr(cls, name)
    parameters = inspect.signature(cls.__init__).parameters
    assert any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    assert list(inspect.signature(cls.forward).parameters)[:3] == ["self", "input_ids", "positions"]
    model, _ = _adapter()
    assert not hasattr(model, "already_warmed_up_prefill") and not hasattr(model, "is_hybrid")
    for shim in (
        lambda: model.embed_input_ids(torch.zeros(1)),
        lambda: model.forward(torch.zeros(1), torch.zeros(1)),
        lambda: model.compute_logits(torch.zeros(1)),
    ):
        with expect_error(  # allow-pytest.raises: pins the exception type only, no message contract
            NotImplementedError
        ):
            shim()
    worker_keywords = dict(model_name="", num_devices=4, tt_data_parallel=1, max_model_len=32_704, max_num_seqs=1)
    assert cls.get_max_tokens_all_users(**worker_keywords) == 32_704


def test_bundle_metadata_names_the_hf_architecture_and_the_adapter() -> None:
    document = json.loads((BUNDLE / "vllm_metadata.json").read_text())
    assert document == {"arch": HF_ARCHITECTURE, "main_class": f"{Qwen38ForCausalLM.__module__}:Qwen38ForCausalLM"}
    module_name, class_name = document["main_class"].split(":")
    assert getattr(importlib.import_module(module_name), class_name) is Qwen38ForCausalLM


# -- the serving venv: the real plugin runner and vLLM's config resolution (skip without vLLM) -------------------------


@pytest.mark.parametrize("launch_override", (False, True))
def test_plugin_applies_model_fabric_before_opening_mesh(monkeypatch, launch_override) -> None:
    pytest.importorskip("vllm")
    from vllm_tt_plugin import worker

    capability = Qwen38ForCausalLM.model_capabilities["fabric_config"]
    original = dict(capability)
    steps = []
    mesh = SimpleNamespace(get_num_devices=lambda: 4)
    monkeypatch.setattr(worker, "get_mesh_grid", lambda: (1, 4))
    monkeypatch.setattr(worker, "get_dispatch_core_config", lambda _config: None)
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: ttnn.cluster.ClusterType.P150_X4)
    monkeypatch.setattr(ttnn, "set_fabric_config", lambda **kwargs: steps.append(("fabric", kwargs)))
    monkeypatch.setattr(ttnn, "open_mesh_device", lambda *args, **kwargs: steps.append(("open", kwargs)) or mesh)
    launch = {"fabric_config": "FABRIC_2D", "fabric_reliability_mode": "RELAXED_INIT"} if launch_override else None
    assert worker.open_mesh_device(launch, "off", model_fabric_config=capability) is mesh
    assert [name for name, _ in steps] == ["fabric", "open"]
    expected = dict(original)
    if launch_override:
        expected.update(config=ttnn.FabricConfig.FABRIC_2D, reliability_mode=ttnn.FabricReliabilityMode.RELAXED_INIT)
    assert steps[0][1] == expected
    assert capability == original


@pytest.mark.parametrize(
    "requirement",
    (
        "ordinary",
        "penalties",
        "logprobs_zero",
        "logprobs_top",
        "allowed_ids",
        "bad_words",
        "logits_processor",
        "custom_processor",
        "structured",
    ),
)
def test_plugin_routes_sampling_requirements(requirement) -> None:
    pytest.importorskip("vllm")
    from vllm_tt_plugin.model_runner import TTModelRunner

    runner = SimpleNamespace(
        sample_on_device_mode="decode_only",
        num_devices=4,
        tt_data_parallel_size=1,
        model=Qwen38ForCausalLM,
        supports_topk_logprobs=False,
        model_config=SimpleNamespace(logits_processors=[object()] if requirement == "custom_processor" else []),
        input_batch=SimpleNamespace(
            no_penalties=requirement != "penalties",
            no_allowed_token_ids=requirement != "allowed_ids",
            max_num_logprobs={"logprobs_zero": 0, "logprobs_top": 5}.get(requirement),
            sampling=SimpleNamespace(
                bad_words_token_ids=[1] if requirement == "bad_words" else [],
                has_active_logitsprocs=lambda: requirement == "logits_processor",
            ),
        ),
    )
    assert not TTModelRunner.check_perform_device_sampling(runner, False, False)
    actual = TTModelRunner.check_perform_device_sampling(runner, True, requirement == "structured")
    assert actual is (requirement in ("ordinary", "penalties"))


def test_plugin_runner_prefill_and_decode_call_shape(monkeypatch) -> None:
    pytest.importorskip("vllm")  # vLLM's own bootstrap must finish before the plugin package imports
    from vllm_tt_plugin.async_decode import TTAsyncDecodeController
    from vllm_tt_plugin.model_input import TTSamplingParams
    from vllm_tt_plugin.model_runner import TTModelRunner

    warnings: list[tuple] = []
    monkeypatch.setattr("vllm_tt_plugin.async_decode.logger.warning", lambda *args: warnings.append(args))
    model, chain = _adapter()
    captured: dict[str, dict] = {}
    for name in ("prefill_forward", "decode_forward"):
        forward = getattr(model, name)
        monkeypatch.setattr(
            model,
            name,
            lambda _forward=forward, _name=name, **kwargs: captured.update({_name: kwargs}) or _forward(**kwargs),
        )
    runner = SimpleNamespace(
        model=model,
        kv_caches=object(),
        trace_mode="all",
        request_specific_rope=True,
        input_batch=SimpleNamespace(req_ids=["request-0"]),
        requests={"request-0": SimpleNamespace(mrope_position_delta=None)},
        tt_per_lane_max_num_seqs=1,
        previous_req_ids=set(),
        note_decode_layout_consumed=lambda: None,
        note_decode_state_slots_settled=lambda: None,
    )
    runner.async_decode = TTAsyncDecodeController(runner)
    prompt = [1000 + index for index in range(20)]
    rows = torch.arange(0, 1, dtype=torch.long)  # the runner's front-packed rows of the one DP rank

    tt_out = TTModelRunner.submit_prefill(
        runner,
        SimpleNamespace(
            input_tokens=torch.tensor([prompt], dtype=torch.int32),
            input_positions=torch.zeros(1, dtype=torch.int32),
            prompt_lens=[len(prompt)],
            block_tables=PAGE_TABLE,
            block_tables_per_layer=None,
            multi_modal_kwargs={},
            perform_device_sampling=False,
            prefill_empty_slots=[0],
        ),
        [1],
    )

    prefill = captured["prefill_forward"]
    assert prefill["enable_trace"] is True and prefill["empty_slots"] == [0] and prefill["kv_cache"] is runner.kv_caches
    assert "sampling_params" not in prefill and "page_tables_per_layer" not in prefill
    assert isinstance(tt_out, torch.Tensor) and tt_out.shape == (1, 1, VOCAB_SIZE) and tt_out.dtype == torch.float32
    assert tt_out.device.type == "cpu" and runner.requests["request-0"].mrope_position_delta == 0
    logits = tt_out[rows, -1, :]
    assert logits.shape == (1, VOCAB_SIZE) and torch.equal(logits[0], _masked(chain.logits()))
    sampled = [int(logits[0].argmax())]

    def decode_input(
        token: int, position: int, *, layout_changed: bool, device_sampling: bool = False, temperature: float = 1.0
    ) -> SimpleNamespace:
        return SimpleNamespace(
            input_tokens=torch.tensor([[token]], dtype=torch.int32),
            input_positions=torch.tensor([position], dtype=torch.int32),
            block_tables=PAGE_TABLE,
            block_tables_per_group=[PAGE_TABLE],
            block_tables_per_layer=None,
            unpadded_batch_size=1,
            tt_sampling_params=TTSamplingParams(
                temperature=torch.full((1,), temperature),
                top_k=torch.full((1,), 32),
                top_p=torch.ones(1),
                presence_penalty=torch.zeros(1),
                frequency_penalty=torch.zeros(1),
                repetition_penalty=torch.ones(1),
                seed=torch.full((1,), -1),
                num_logprobs=torch.full((1,), -2),
                enable_log_probs=torch.zeros(1, dtype=torch.bool),
            ),
            perform_device_sampling=device_sampling,
            prompt_tokens=None,
            output_tokens=None,
            decode_layout_changed=layout_changed,
            slot_remap=None,  # the runner's value for the identity remap
        )

    for step, layout_changed in enumerate((True, False, False)):
        position = len(prompt) + step
        submission = runner.async_decode.submit_decode(
            decode_input(sampled[-1], position, layout_changed=layout_changed), read_from_device=False
        )
        decode = captured["decode_forward"]
        assert "reset_batch" not in decode and "slot_remap" not in decode and "sampling_params" not in decode
        assert decode["reload_inputs"] is True and decode["reload_page_table"] is False
        assert decode["reload_sampling_params"] is False and decode["reset_sampling_state"] is False
        assert decode["rope_deltas_all_users"] == ([0] if step == 0 else None)
        assert decode["enable_trace"] is True and decode["read_from_device"] is False
        assert int(decode["start_pos"][0]) == position and int(decode["tokens"][0, 0]) == sampled[-1]
        assert submission.reload_plan is not None and submission.reload_plan.reload_inputs
        finalized = runner.async_decode.finalize_decode(submission)
        assert finalized.tt_out is submission.tt_out and finalized.tt_log_probs is None
        logits = finalized.tt_out[rows, -1, :]
        assert logits.shape == (1, VOCAB_SIZE) and torch.equal(logits[0], _masked(chain.logits()))
        sampled.append(int(logits[0].argmax()))
    assert model.slot.committed == prompt + sampled[:-1] and warnings == []  # no legacy-contract warning

    # The device-sampling contract (sample_on_device_mode decode_only, the plugin's per-step decision True): the
    # controller hands the params over as per-row lists with the seed sentinel turned into None and asks for a
    # sampling reset on the mode transition; the adapter answers with the int32 [1, 1] token the runner reshapes.
    for step, temperature in enumerate((0.0, 0.0, 0.5)):  # 0.5: exact in the plugin's fp32 tensor
        position = len(prompt) + 3 + step
        submission = runner.async_decode.submit_decode(
            decode_input(sampled[-1], position, layout_changed=False, device_sampling=True, temperature=temperature),
            read_from_device=False,
        )
        decode = captured["decode_forward"]
        params = decode["sampling_params"]
        assert isinstance(params, TTSamplingParams) and params.seed == [None] and params.top_k == [32]
        assert params.temperature == [temperature] and params.enable_log_probs == [False]
        assert "prompt_tokens" not in decode and "output_tokens" not in decode  # no penalty active
        assert decode["reload_inputs"] is True
        assert decode["reload_sampling_params"] is (step == 0) and decode["reset_sampling_state"] is (step == 0)
        assert submission.perform_device_sampling is True
        finalized = runner.async_decode.finalize_decode(submission)
        tokens = finalized.tt_out
        assert isinstance(tokens, torch.Tensor) and tokens.shape == (1, 1) and tokens.dtype == torch.int32
        assert tokens[rows].reshape(1, -1).shape == (1, 1) and finalized.tt_log_probs is None  # the runner's take
        if temperature == 0.0:
            assert int(tokens) == int(_masked(chain.logits()).argmax())
        else:
            support = _vllm_sampler_probabilities(_masked(chain.logits()), temperature=0.5, top_k=32, top_p=1.0)
            assert float(support[int(tokens)]) > 0
        sampled.append(int(tokens))
    assert model.slot.committed == prompt + sampled[:-1] and model.slot.policy.temperature == 0.5 and warnings == []


def test_vllm_config_resolution_in_the_serving_venv() -> None:
    pytest.importorskip("vllm")
    if not (CHECKPOINT / "config.json").is_file():
        pytest.skip("QWEN38_CHECKPOINT must name the pinned checkpoint directory")
    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.model_loader.utils import get_model_architecture
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.interfaces_base import is_text_generation_model
    from vllm.platforms import current_platform
    from vllm.plugins import load_general_plugins
    from vllm_tt_plugin.platform import TTPlatform

    assert current_platform.device_name == "tt"
    assert is_text_generation_model(Qwen38ForCausalLM)
    load_general_plugins()
    for name in (HF_ARCHITECTURE, TT_ARCHITECTURE):
        ModelRegistry.register_model(name, f"{Qwen38ForCausalLM.__module__}:Qwen38ForCausalLM")
    overrides = {"architectures": [TT_ARCHITECTURE]}

    model_config = ModelConfig(
        model=str(CHECKPOINT), tokenizer=str(CHECKPOINT), hf_overrides=overrides, max_model_len=MAX_MODEL_LEN
    )

    assert model_config.hf_config.architectures == [TT_ARCHITECTURE]
    assert model_config.runner_type == "generate" and model_config.uses_mrope and not model_config.is_hybrid
    assert model_config.get_vocab_size() == VOCAB_SIZE and model_config.max_model_len == MAX_MODEL_LEN
    assert get_model_architecture(model_config) == (Qwen38ForCausalLM, TT_ARCHITECTURE)

    engine_config = EngineArgs(
        model=str(CHECKPOINT),
        tokenizer=str(CHECKPOINT),
        hf_overrides=overrides,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=1,
    ).create_engine_config()
    TTPlatform.check_and_update_config(engine_config)  # a re-run of the hook __post_init__ already applied

    scheduler = engine_config.scheduler_config
    assert scheduler.max_num_seqs == 1 and scheduler.enable_chunked_prefill is False
    assert scheduler.max_num_batched_tokens == MAX_MODEL_LEN and scheduler.async_scheduling is False
    assert engine_config.cache_config.enable_prefix_caching is False
    assert engine_config.parallel_config.worker_cls == "vllm_tt_plugin.worker.TTWorker"
    assert get_model_architecture(engine_config.model_config)[0] is Qwen38ForCausalLM
