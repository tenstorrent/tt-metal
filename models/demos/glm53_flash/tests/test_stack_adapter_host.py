# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no ttnn, no device, no vLLM) test of the persistent-stack vLLM adapter
(``tt/generator_vllm_stack.py``): the plugin's hooks driven over the real ``Glm53Generator`` on a CPU ``ModelBackend``
mock. This tree carries no ``tests/mock_model.py`` (the reference workspace's), so the mock below mirrors
``StackBackend``'s contract flags -- fused sampling inside the decode trace, one full-width trace, 64-token pages --
on torch tensors with a deterministic model (next token = (token + 1) % V) and a per-slot ledger of the tokens each
slot's state consumed. Covered: the plugin's zero-padded block tables (trimmed to the live blocks, grown block by
block), chunked prefill (intermediate chunk host-routed, the continuation sliced out of the full history), a decode
between two chunks, releases, the condensing slot remaps the plugin builds after a finish -- absorbed into the
adapter's plugin-slot -> generator-slot alias, so no state moves: a 2-cycle through the freed slot, a 3-chain and a
4-cycle over a full batch (every slot live: what the generator's ``remap_slots`` refuses) -- the lowest-free admission,
the host-sampled logits route, the async read, the teardown hooks and the refusals. GLM53_STACK_PREFIX (the prefix
cache, ``StackPrefixCache``) on a mock with the device side of the snapshot pool and the sequence-sharded pool's owner
blocks: snapshots at the on-grid chunk ends, the hit confirmation, the attach of a new request admitted at start > 0
on shared pages, the release, the LRU reuse of unpinned entries, and the fallbacks to a prefill from 0. Image input
(GLM53_STACK_MM): the generator's per-chunk staging + id remap into the mock's extra embedding rows, image prompts kept
out of the batched / packed replays, ``GLM53StackForConditionalGeneration`` (declaration with / without vLLM, the 0.31 /
0.30 processor module, the plugin's per-row image items routed to the generator through a fake encoder, the device tower's L1 plan and encoder selection) and, with a
vLLM interpreter (``GLM53_VLLM_PYTHON``, a subprocess: this process stays vLLM- and ttnn-free) and the checkpoint, the
real processor the class registers run on the demo image.

    PYTHONPATH=<worktree> <tt-metal>/python_env/bin/pytest --noconftest -p no:cacheprovider -q \\
        models/demos/glm53_flash/tests/test_stack_adapter_host.py
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    DecodeIO,
    DecodeOutput,
    Glm53Generator,
    ModelBackend,
    PendingTokens,
)
from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams
from models.demos.glm53_flash.tt import generator_vllm_stack as gvs
from models.demos.glm53_flash.tt.generator_vllm_stack import (
    LATENT_DIM,
    GLM53StackForCausalLM,
    cache_defaults,
    declare_multimodal_class,
    row_params,
    trim_page_rows,
)
from models.demos.glm53_flash.tt.mm_inputs import IMAGE_TOKEN_ID

VOCAB = 64
B = 4  # max_num_seqs = the backend's slots
SEQ = 2048  # max_model_len
BPS = SEQ // BLOCK_SIZE  # the plugin's block-table width: cdiv(max_model_len, block_size)
NUM_BLOCKS = B * SEQ // BLOCK_SIZE + B  # worker.py:737-785: the budget + one block per user; block 0 = the null block


@contextlib.contextmanager
def expect_error(error, message):
    """The root conftest's ``expect_error`` fixture (the CI triager's [EXPECTED_ERROR] markers), local to this module:
    the host test runs with ``--noconftest`` because the root conftest imports ttnn at collection."""
    logger.info(f'[EXPECTED_ERROR BEGIN] {error.__name__} message="{message}"')
    try:
        with pytest.raises(error, match=message) as exc_info:  # allow-pytest.raises: the fixture body, no conftest here
            yield exc_info
    finally:
        logger.info(f'[EXPECTED_ERROR END] {error.__name__} message="{message}"')


def test_module_imports_without_ttnn():
    """The adapter module must be importable on a host without ttnn (every ttnn use is lazy); without ttnn the fabric
    capability is None (the plugin's hardware default) and the other declarations are the first adapter's."""
    assert "models.demos.glm53_flash.tt.generator_vllm_stack" in sys.modules
    assert "ttnn" not in sys.modules, "importing the adapter must not import ttnn"
    caps = GLM53StackForCausalLM.model_capabilities
    assert caps["supports_chunked_prefill"] is True
    assert (
        caps["supports_sample_on_device"] is True and caps["max_device_top_k"] == gvs.DEVICE_TOP_K == 10
    )  # the 8x4 default
    assert caps["device_top_k_unbounded_as"] == 10  # GLM sets no top-k: default requests stay on the device
    assert caps["supports_device_penalties"] is False and caps["supports_async_decode"] is True
    assert caps["fabric_config"] is None
    for key in (
        "supports_prefix_caching",
        "confirms_prefix_cache_hits",
        "supports_async_spec_decode",
        "supports_spec_decode",
        "supports_device_grammar",
        "supports_compact_host_logits",
        "supports_selective_host_readback",
    ):
        assert caps[key] is False, key
    assert GLM53StackForCausalLM.decode_input_update_contract == 1
    assert GLM53StackForCausalLM.tt_supported_decode_batch_sizes == ()


def test_adapter_satisfies_the_vllm_model_protocol():
    """vLLM >= 0.30 admits the generate runner only for classes that pass ``is_vllm_model``: an ``__init__`` accepting
    ``vllm_config`` (``**kwargs`` counts), ``embed_input_ids``, a ``forward`` with ``input_ids`` and ``positions``,
    plus ``compute_logits``. Signature-only check: the plugin never calls these."""

    def accepts(fn, name):
        params = inspect.signature(fn).parameters
        return name in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    assert accepts(GLM53StackForCausalLM.__init__, "vllm_config")
    assert callable(getattr(GLM53StackForCausalLM, "embed_input_ids", None))
    assert callable(getattr(GLM53StackForCausalLM, "compute_logits", None))
    forward = inspect.signature(GLM53StackForCausalLM.forward).parameters
    assert "input_ids" in forward and "positions" in forward


# ------------------------------------------------------------------------------------------ the CPU backend mock
class _Buf:
    """A DecodeIO field: the host tensor the generator writes / reads (``write`` replaces it whole, as a device copy
    into a same-shaped buffer would)."""

    def __init__(self, t: torch.Tensor):
        self.t = t


@dataclass
class _Cache:
    num_blocks: int
    state: dict = field(default_factory=dict)  # slot -> the tokens its recurrent state consumed, in order
    page_table_host: torch.Tensor | None = None


class _MockStackBackend(ModelBackend):
    """StackBackend's contract on CPU: one full-width decode trace that samples greedily itself (fused sampling,
    every token row rewritten), the TP prefill's 64-token alignment, 128-token chunks (so a 200-token prompt is two
    chunks). Model: next token = (token + 1) % VOCAB; the per-slot ledger asserts positions and page coverage. Image
    input: MM_ROWS extra embedding rows (StackBackend.stage_mm_rows under GLM53_STACK_MM=1); a chunk id VOCAB + j reads
    staged row j, recorded in the ledger as ("img", int(row[0]))."""

    vocab_size = VOCAB
    logits_width = VOCAB
    block_size = BLOCK_SIZE
    prefill_align = 64
    continuation_align = 64
    prefill_chunk_tokens = 128
    supported_context = SEQ
    width_bucketed_decode = False
    samples_in_decode_trace = True
    copy_slot_keeps_sampler_rng = True
    layer_kinds = ("dense_kda",)
    mm_extra_rows = 64  # MM_ROWS

    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.chunks: list = []  # (slot, chunk_start, n_valid, logits_rows, sampled)
        self.chunk_ids: list = []  # every chunk's valid ids as the backend received them
        self.mm_table: dict = {}  # extra row j -> its staged row
        self.mm_stages: list = []  # (chunks run before the stage, rows staged)
        self.traces: list = []
        self.released: list = []
        self.releases: list = []
        self.decodes = 0
        self.prepared = (False, False)

    # ---- state ownership
    def allocate_kv_cache(self, *, max_batch, max_seq_len, num_blocks=None):
        nb = num_blocks or max_batch * (max_seq_len // BLOCK_SIZE)
        return _Cache(num_blocks=nb, state={s: [] for s in range(max_batch)})

    def allocate_decode_io(self, kv):
        n = self.max_batch
        return DecodeIO(
            tokens=_Buf(torch.zeros(n, dtype=torch.int32)),
            cur_pos=_Buf(torch.full((n,), -1, dtype=torch.int32)),
            page_table=_Buf(torch.zeros(n, 1, dtype=torch.int32)),
            logits=_Buf(torch.zeros(n, VOCAB)),
            sampled=_Buf(torch.zeros(n, dtype=torch.int32)),
            k=_Buf(torch.ones(n, dtype=torch.int32)),
            p=_Buf(torch.zeros(n)),
            temp=_Buf(torch.ones(n)),
            seeds=_Buf(torch.full((n,), -2, dtype=torch.int64)),
        )

    def write(self, buf, value):
        buf.t = torch.as_tensor(value).clone()

    def read(self, buf):
        return buf.t.clone()

    def set_page_table_host(self, kv, page_table):
        kv.page_table_host = torch.as_tensor(page_table).clone()

    def reset_slot(self, kv, slot):
        kv.state[slot] = []

    def copy_slot(self, kv, src, dst):
        kv.state[dst] = list(kv.state[src])

    def reset_all(self, kv):
        for s in kv.state:
            kv.state[s] = []

    def release_slot(self, kv, slot):
        self.releases.append(slot)

    def stage_mm_rows(self, rows):
        assert 1 <= int(rows.shape[0]) <= self.mm_extra_rows, tuple(rows.shape)
        self.mm_stages.append((len(self.chunks), rows.clone()))
        for j in range(int(rows.shape[0])):
            self.mm_table[j] = rows[j].clone()
        return VOCAB

    # ---- prefill
    def prepare_prefill(self, kv):
        self.prepared = (True, self.prepared[1])

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None, **sio):
        st = kv_cache.state[slot]
        if chunk_start == 0:
            st.clear()
        assert len(st) == chunk_start, f"slot {slot}: state holds {len(st)} tokens, chunk starts at {chunk_start}"
        self.chunk_ids.append(tokens[:n_valid].clone())
        st.extend(t if t < VOCAB else ("img", int(self.mm_table[t - VOCAB][0])) for t in tokens[:n_valid].tolist())
        self.chunks.append((slot, chunk_start, n_valid, logits_rows, sample is not None))
        if logits_rows == "none":
            return None
        nxt = ((st[-1] if isinstance(st[-1], int) else st[-1][1]) + 1) % VOCAB  # an image row: its position
        if sample is not None:
            io.tokens.t[slot] = nxt
            io.sampled.t[slot] = nxt
            return nxt
        if logits_rows == "all":
            logits = torch.zeros(int(tokens.numel()), VOCAB)
            for i, t in enumerate(st[chunk_start:]):
                logits[i, (t + 1) % VOCAB] = 1.0
            return logits
        logits = torch.zeros(1, VOCAB)
        logits[0, nxt] = 1.0
        return logits

    # ---- decode
    def prepare_decode(self, io, kv, widths):
        self.widths = tuple(widths)
        self.prepared = (self.prepared[0], True)

    def decode(self, io, kv, width):
        self.decodes += 1
        pos, tok = io.cur_pos.t, io.tokens.t
        logits = torch.zeros(self.max_batch, VOCAB)
        sampled = torch.zeros(self.max_batch, dtype=torch.int32)
        for s in range(self.max_batch):
            p = int(pos[s])
            if p < 0:
                continue
            st = kv.state[s]
            assert len(st) == p, f"slot {s}: state holds {len(st)} tokens, decode position {p}"
            blk = int(io.page_table.t[s, p // BLOCK_SIZE])
            assert blk > 0, f"slot {s}: position {p} maps to block {blk} (null / unmapped)"
            st.append(int(tok[s]))
            nxt = (int(tok[s]) + 1) % VOCAB
            logits[s, nxt] = 1.0
            sampled[s] = nxt
        io.logits.t = logits
        io.sampled.t = sampled  # the fused sampler writes every row (0 on inactive rows) ...
        io.tokens.t = sampled.clone()  # ... and feeds it back as the next input
        io.cur_pos.t = torch.where(pos >= 0, pos + 1, pos)

    def sample(self, io, mode):
        return None  # fused into the decode step

    def capture_decode(self, io, kv, width):
        trace = ("decode", width, io, kv)
        self.traces.append(trace)
        return trace

    def capture_sampling(self, io, mode):
        raise AssertionError("fused sampling: the generator captures no sampler trace")

    def replay(self, trace):
        assert trace[0] == "decode"
        self.decode(trace[2], trace[3], trace[1])

    def release(self, trace):
        self.released.append(trace)


# ------------------------------------------------------------------------------------------ helpers
def _adapter():
    backend = _MockStackBackend(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    # model_runner.py:4770-4799: eager phase for both, then the traced phase (prefill first)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=False, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.tt_supported_decode_batch_sizes == ()
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.tt_supported_decode_batch_sizes == (B,)
    assert backend.prepared == (True, True) and len(backend.traces) == 1  # one full-width decode trace
    return ad, backend, gen, kv


def _greedy(n: int):
    """The plugin's TTSamplingParams on a device-sampling call: per-row lists, seed None = vLLM gave none."""
    return SimpleNamespace(
        temperature=[0.0] * n,
        top_k=[1] * n,
        top_p=[1.0] * n,
        presence_penalty=[0.0] * n,
        frequency_penalty=[0.0] * n,
        repetition_penalty=[1.0] * n,
        seed=[None] * n,
    )


def _prompt(n: int, base: int) -> list[int]:
    return [(base + i) % VOCAB for i in range(n)]


def _blocks(req: int, n_tokens: int) -> list[int]:
    """vLLM's block ids for request ``req`` holding ``n_tokens`` positions: a fixed pool per request, never block 0."""
    return [1 + req * BPS + j for j in range(-(-n_tokens // BLOCK_SIZE))]


def _table(rows: list[list[int]], n_rows: int) -> torch.Tensor:
    """The plugin's block table (input_batch.py:789-808): ``[n_rows, BPS]`` int32, zero past each row's live blocks,
    all-zero padding rows."""
    t = torch.zeros(n_rows, BPS, dtype=torch.int32)
    for i, blocks in enumerate(rows):
        t[i, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
    return t


def _history(prompts: list[list[int]]) -> torch.Tensor:
    """``tokens [B, T]``: every row's history from position 0 (model_runner.py:4009-4063), zero past its end."""
    t = torch.zeros(len(prompts), max(len(p) for p in prompts), dtype=torch.int32)
    for i, p in enumerate(prompts):
        t[i, : len(p)] = torch.tensor(p, dtype=torch.int32)
    return t


def _decode(ad, kv, tokens: list[int], positions: list[int], rows: list[list[int]], *, remap=None, read=False):
    """One device-sampled decode step as the plugin submits it (async_decode.py:1335-1440): ``B`` padded rows."""
    out = ad.decode_forward(
        torch.tensor(tokens, dtype=torch.int32).reshape(B, 1),
        torch.tensor(positions, dtype=torch.int32),
        page_table=_table(rows, B),
        kv_cache=kv,
        enable_trace=True,
        read_from_device=read,
        sampling_params=_greedy(B),
        slot_remap=None if remap is None else torch.tensor(remap, dtype=torch.int32),
        reload_inputs=True,
        reload_page_table=False,
        reload_sampling_params=remap is not None,
        reset_sampling_state=remap is not None,
    )
    return out


# ------------------------------------------------------------------------------------------ the hooks
def test_page_rows_trim():
    """Zero padding past the live blocks -> negative-terminated rows; column 0 untouched (a dead row stays [0, -1 ...])."""
    t = trim_page_rows(torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0], [0, 0, 0, 0], [1, 2, 3, 4]], dtype=torch.int32))
    assert t.dtype == torch.int64
    assert t.tolist() == [[5, 6, -1, -1], [7, -1, -1, -1], [0, -1, -1, -1], [1, 2, 3, 4]]
    assert trim_page_rows(torch.tensor([9, 0, 0])).tolist() == [[9, -1, -1]]


def test_row_params_conversion():
    """The plugin's per-row lists -> SamplingParams: greedy rows, vLLM's unset top-k (0 / -1), None seeds."""
    sp = SimpleNamespace(
        temperature=[0.0, 0.7, 0.9, 1.0], top_k=[1, 1, 0, 40], top_p=[1.0, 0.9, 0.95, 1.0], seed=[None, 3, -1, None]
    )
    assert row_params(sp, 0) == SamplingParams(top_k=1) and row_params(sp, 0).is_greedy
    assert (
        row_params(sp, 1) == SamplingParams(temperature=0.7, top_k=1, top_p=0.9, seed=3) and row_params(sp, 1).is_greedy
    )
    assert row_params(sp, 2) == SamplingParams(temperature=0.9, top_k=None, top_p=0.95, seed=None)
    assert row_params(sp, 3) == SamplingParams(temperature=1.0, top_k=40, top_p=1.0, seed=None)
    with expect_error(AttributeError, "temperature"):  # a missing field is a contract break, not a greedy row
        row_params(SimpleNamespace(), 0)
    with expect_error(IndexError, "out of range"):  # a short list likewise
        row_params(sp, 5)


def test_kv_cache_spec_and_budget(monkeypatch):
    """worker.py:737-785 / 630-652: the token budget and the one latent-page spec; the block size must be 64."""
    spec_calls = []

    class FullAttentionSpec:
        def __init__(self, **kw):
            spec_calls.append(kw)

    for name in ("vllm", "vllm.v1", "vllm.v1.kv_cache_interface"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["vllm.v1.kv_cache_interface"].FullAttentionSpec = FullAttentionSpec
    assert GLM53StackForCausalLM.get_max_tokens_all_users(max_model_len=SEQ, max_num_seqs=B) == B * SEQ
    with expect_error(ValueError, "needs max_model_len"):
        GLM53StackForCausalLM.get_max_tokens_all_users(max_model_len=SEQ)
    cfg = SimpleNamespace(cache_config=SimpleNamespace(block_size=64))
    spec = GLM53StackForCausalLM.get_kv_cache_spec(cfg)
    assert list(spec) == ["glm53_latent"]
    assert spec_calls == [dict(block_size=64, num_kv_heads=1, head_size=LATENT_DIM, dtype=torch.bfloat16)]
    with expect_error(ValueError, "must be 64"):
        GLM53StackForCausalLM.get_kv_cache_spec(SimpleNamespace(cache_config=SimpleNamespace(block_size=32)))


def test_cache_defaults_under_tt_cache_path():
    """The serving launcher's TT_CACHE_PATH places the tile cache and the permuted expert copies; an explicit
    environment value wins; without TT_CACHE_PATH nothing is defaulted."""
    assert cache_defaults({}) == {}
    assert cache_defaults({"TT_CACHE_PATH": "/cache/glm"}) == {
        "GLM53_MK_TILE_CACHE_V3": "/cache/glm/glm53_tile_cache_v3",
        "GLM53_FUSED_EXPERTS_CACHE": "/cache/glm/glm53_fused_experts_cache",
    }
    assert cache_defaults({"TT_CACHE_PATH": "/cache/glm", "GLM53_MK_TILE_CACHE_V3": "/data/v3"}) == {
        "GLM53_FUSED_EXPERTS_CACHE": "/cache/glm/glm53_fused_experts_cache",
    }


def test_initialize_refuses_data_parallel_and_bad_batch():
    """loader.py:38-45: refused before any device / ttnn work."""
    with expect_error(ValueError, "tt_data_parallel must be 1"):
        GLM53StackForCausalLM.initialize_vllm_model(None, None, B, SEQ, tt_data_parallel=2)
    with expect_error(ValueError, "outside"):
        GLM53StackForCausalLM.initialize_vllm_model(None, None, 33, SEQ)
    assert "ttnn" not in sys.modules


def test_allocate_kv_cache_enforces_the_block_size():
    backend = _MockStackBackend(B)
    gen = Glm53Generator(backend, tokenizer=None, max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ)
    with expect_error(ValueError, "block size 32"):
        ad.allocate_kv_cache((NUM_BLOCKS, 1, 32, LATENT_DIM), torch.bfloat16, 45)
    with expect_error(RuntimeError, "allocate_kv_cache must run"):
        ad.warmup_model_prefill(enable_trace=False)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    assert kv is ad.kv_cache and kv.num_blocks == NUM_BLOCKS


def test_serving_flow_prefill_decode_remap_release():
    """The plugin's step sequence end to end: three admissions (one chunked by the mock's 128-token chunks),
    padded decode steps with block-by-block table growth, a chunked prefill with a decode in between its chunks,
    the async read, releases and the condensing remaps absorbed into the slot alias (a 2-cycle through the freed
    slot, a 3-chain, then two admissions into the lowest free generator slots and a 4-cycle over the full batch)."""
    ad, backend, gen, kv = _adapter()
    prompts = [_prompt(100, 5), _prompt(200, 11), _prompt(70, 40)]
    last = [p[-1] for p in prompts]
    # ---- step 1: three fresh prompts, device greedy
    out = ad.prefill_forward(
        _history(prompts),
        page_table=_table([_blocks(i, len(p)) for i, p in enumerate(prompts)], 3),
        kv_cache=kv,
        enable_trace=True,
        prompt_lens=[len(p) for p in prompts],
        start_pos=torch.zeros(3, dtype=torch.int32),
        empty_slots=[0, 1, 2],
        sampling_params=_greedy(3),
    )
    assert isinstance(out, tuple) and out[1] is None
    ids = out[0]
    assert ids.shape == (3, 1) and ids.dtype == torch.int32
    assert ids.reshape(-1).tolist() == [(t + 1) % VOCAB for t in last]
    assert backend.chunks == [
        (0, 0, 100, "last", True),
        (1, 0, 128, "none", False),  # the 200-token prompt: an intermediate chunk without the head ...
        (1, 128, 72, "last", True),  # ... and its last chunk, sampled on device
        (2, 0, 70, "last", True),
    ]
    assert gen.live_slots(kv) == [0, 1, 2]
    assert [len(kv.state[s]) for s in range(B)] == [100, 200, 70, 0]
    # ---- decode steps: rows = slots, padding row 3; the tables grow a block every 64 positions, as vLLM's do
    cur = [int(v) for v in ids.reshape(-1).tolist()]
    pos = [100, 200, 70]
    for step in range(30):
        rows = [_blocks(i, pos[i] + 1) for i in range(3)]
        out = _decode(ad, kv, cur + [0], pos + [-1], rows, read=False)
        assert isinstance(out, DecodeOutput)
        if step % 2:  # the sync path: decode_forward's raw handle straight into process_decode_output_host
            got = ad.process_decode_output_host(out, is_tokens=True)
        else:  # the async path: read_decode_output enqueues the copy, the handle is formatted after the events
            handle, events = ad.read_decode_output(out, async_read=True)
            assert isinstance(handle, PendingTokens) and events == []
            got = ad.process_decode_output_host(handle, is_tokens=True)
        assert got.shape == (B, 1) and got.dtype == torch.int32
        assert got.reshape(-1).tolist() == [(t + 1) % VOCAB for t in cur] + [0]
        cur = got.reshape(-1).tolist()[:3]
        pos = [p + 1 for p in pos]
    assert pos == [130, 230, 100]  # slot 0 crossed its second page boundary (position 128 needed a third block)
    assert [len(kv.state[s]) for s in range(B)] == [130, 230, 100, 0]
    assert backend.decodes == 30 and gen.counters["decode_trace_replays"] == 30
    assert gen.counters["page_rows_unchanged"] > 0  # the unchanged-row fast path saw the trimmed rows
    # ---- a 300-token prompt in two scheduler steps (slot 3): the intermediate chunk is host-routed by the plugin
    long = _prompt(300, 23)
    logits = ad.prefill_forward(
        _history([long[:128]]),
        page_table=_table([_blocks(3, 128)], 1),
        kv_cache=kv,
        prompt_lens=[128],
        start_pos=[0],
        empty_slots=[3],
        sampling_params=None,
    )
    assert isinstance(logits, torch.Tensor) and logits.shape == (1, 1, VOCAB) and logits.dtype == torch.float32
    assert int(logits[0, 0].argmax()) == (long[127] + 1) % VOCAB
    assert backend.chunks[-1] == (3, 0, 128, "last", False)
    # a decode of the other three between the two chunks leaves slot 3 alone
    rows = [_blocks(i, pos[i] + 1) for i in range(3)]
    got = ad.process_decode_output_host(_decode(ad, kv, cur + [0], pos + [-1], rows), is_tokens=True)
    cur, pos = got.reshape(-1).tolist()[:3], [p + 1 for p in pos]
    assert len(kv.state[3]) == 128
    out = ad.prefill_forward(
        _history([long]),  # the whole history: the adapter slices [128, 300)
        page_table=_table([_blocks(3, 300)], 1),
        kv_cache=kv,
        prompt_lens=[300],
        start_pos=[128],
        empty_slots=[3],
        sampling_params=_greedy(1),
    )
    assert out[0].reshape(-1).tolist() == [(long[-1] + 1) % VOCAB]
    assert backend.chunks[-2:] == [(3, 128, 128, "none", False), (3, 256, 44, "last", True)]
    assert kv.state[3] == long and gen.live_slots(kv) == [0, 1, 2, 3]
    cur.append(int(out[0][0, 0]))
    pos.append(300)
    got = ad.process_decode_output_host(
        _decode(ad, kv, cur, pos, [_blocks(i, pos[i] + 1) for i in range(4)]), is_tokens=True
    )
    assert got.reshape(-1).tolist() == [(t + 1) % VOCAB for t in cur]
    cur, pos = got.reshape(-1).tolist(), [p + 1 for p in pos]
    # ---- request 1 finishes: release, then the plugin condenses rows [0, 3, 2] -> remap [0, 3, 2, 1]
    #      (a 2-cycle between plugin slot 3 and the freed slot 1): the alias absorbs it, no state moves
    ad.release_request(1)
    assert backend.releases == [1] and gen.live_slots(kv) == [0, 2, 3]
    state3 = list(kv.state[3])
    tokens = [cur[0], cur[3], cur[2], 0]
    positions = [pos[0], pos[3], pos[2], -1]
    rows = [_blocks(0, pos[0] + 1), _blocks(3, pos[3] + 1), _blocks(2, pos[2] + 1)]
    got = ad.process_decode_output_host(_decode(ad, kv, tokens, positions, rows, remap=[0, 3, 2, 1]), is_tokens=True)
    assert got.reshape(-1).tolist() == [(t + 1) % VOCAB for t in tokens[:3]] + [0]
    assert kv.state[3] == state3 + [tokens[1]]  # plugin row 1 decoded in generator slot 3, where its state is
    assert ad._alias == [0, 3, 2, 1] and gen.live_slots(kv) == [0, 2, 3]
    ad.note_state_slots_moved({3: 1, 1: 3})  # the plugin's notification of the applied gather: a no-op
    cur = [got[0, 0].item(), got[1, 0].item(), got[2, 0].item()]
    pos = [positions[0] + 1, positions[1] + 1, positions[2] + 1]
    # ---- request 0 finishes: rows [plugin 1, plugin 2] -> remap [1, 2, 0, 3] (the 3-chain 1 -> 0, 2 -> 1, 0 freed)
    ad.release_request(0)
    assert backend.releases == [1, 0] and gen.live_slots(kv) == [2, 3]
    state3, state2 = list(kv.state[3]), list(kv.state[2])
    tokens = [cur[1], cur[2], 0, 0]
    positions = [pos[1], pos[2], -1, -1]
    rows = [_blocks(3, pos[1] + 1), _blocks(2, pos[2] + 1)]
    got = ad.process_decode_output_host(_decode(ad, kv, tokens, positions, rows, remap=[1, 2, 0, 3]), is_tokens=True)
    assert got.reshape(-1).tolist() == [(t + 1) % VOCAB for t in tokens[:2]] + [0, 0]
    assert kv.state[3] == state3 + [tokens[0]] and kv.state[2] == state2 + [tokens[1]]
    assert ad._alias == [3, 2, 0, 1] and gen.live_slots(kv) == [2, 3]
    cur, pos = [got[0, 0].item(), got[1, 0].item()], [positions[0] + 1, positions[1] + 1]
    # ---- two admissions fill the batch (plugin slots 2, 3 -> the lowest free generator slots 0, 1), then the
    #      plugin's gather is a 4-cycle over the full batch, remap [1, 2, 3, 0]: with every slot live the generator's
    #      remap_slots refuses it ("cyclic slot_remap needs a free scratch slot": the first 4-user served step); the
    #      alias absorbs it and every state decodes in place
    fresh = [_prompt(40, 50), _prompt(30, 60)]
    out = ad.prefill_forward(
        _history(fresh),
        page_table=_table([_blocks(1, 40), _blocks(0, 30)], 2),  # the freed requests' block pools
        kv_cache=kv,
        prompt_lens=[40, 30],
        start_pos=[0, 0],
        empty_slots=[2, 3],
        sampling_params=_greedy(2),
    )
    assert backend.chunks[-2:] == [(0, 0, 40, "last", True), (1, 0, 30, "last", True)]
    assert ad._alias == [3, 2, 0, 1] and gen.live_slots(kv) == [0, 1, 2, 3]
    cur += [int(out[0][0, 0]), int(out[0][1, 0])]
    pos += [40, 30]
    blocks = [_blocks(3, pos[0] + 1), _blocks(2, pos[1] + 1), _blocks(1, pos[2] + 1), _blocks(0, pos[3] + 1)]
    states = {s: list(kv.state[s]) for s in range(B)}
    remap = [1, 2, 3, 0]
    tokens, positions, rows = [cur[r] for r in remap], [pos[r] for r in remap], [blocks[r] for r in remap]
    got = ad.process_decode_output_host(_decode(ad, kv, tokens, positions, rows, remap=remap), is_tokens=True)
    assert got.reshape(-1).tolist() == [(t + 1) % VOCAB for t in tokens]
    assert ad._alias == [2, 0, 1, 3] and gen.live_slots(kv) == [0, 1, 2, 3]
    for r, g in zip(remap, [2, 0, 1, 3]):
        assert kv.state[g] == states[g] + [cur[r]]
    # ---- teardown
    ad.release_persistent_capture()
    assert backend.released == backend.traces and ad.tt_supported_decode_batch_sizes == ()


def test_admission_takes_the_lowest_free_generator_slot():
    """A first chunk lands in the lowest free generator slot even when its plugin slot is aliased higher: the alias
    entries of the two free plugin slots swap (invisible to the plugin); a continuation keeps its slot."""
    ad, backend, gen, kv = _adapter()
    prompts = [_prompt(20, 3), _prompt(20, 7)]
    ad.prefill_forward(
        _history(prompts),
        page_table=_table([_blocks(i, 20) for i in range(2)], 2),
        kv_cache=kv,
        prompt_lens=[20, 20],
        start_pos=[0, 0],
        empty_slots=[0, 1],
        sampling_params=_greedy(2),
    )
    assert gen.live_slots(kv) == [0, 1] and ad._alias == [0, 1, 2, 3]
    ad._alias = [0, 1, 3, 2]  # a legal alias after gathers: the free plugin slots 2, 3 hold generator slots 3, 2
    long = _prompt(200, 9)
    ad.prefill_forward(
        _history([long[:128]]),
        page_table=_table([_blocks(2, 128)], 1),
        kv_cache=kv,
        prompt_lens=[128],
        start_pos=[0],
        empty_slots=[2],
        sampling_params=None,
    )
    assert backend.chunks[-1] == (2, 0, 128, "last", False)  # generator slot 2, the lowest free one
    assert ad._alias == [0, 1, 2, 3] and gen.live_slots(kv) == [0, 1, 2]
    ad.prefill_forward(  # the continuation stays in generator slot 2
        _history([long]),
        page_table=_table([_blocks(2, 200)], 1),
        kv_cache=kv,
        prompt_lens=[200],
        start_pos=[128],
        empty_slots=[2],
        sampling_params=_greedy(1),
    )
    assert backend.chunks[-1] == (2, 128, 72, "last", True) and kv.state[2] == long
    ad.release_request(2)
    assert backend.releases == [2] and gen.live_slots(kv) == [0, 1]


def test_host_sampled_decode_route():
    """Without ``sampling_params`` the plugin samples on the host from ``(logits [B_pad, 1, V], None)`` (read back even
    though DP=1 submits ``read_from_device=False``) and feeds its token back on the next step."""
    ad, backend, gen, kv = _adapter()
    p = _prompt(50, 3)
    out = ad.prefill_forward(
        _history([p]),
        page_table=_table([_blocks(0, 50)], 1),
        kv_cache=kv,
        prompt_lens=[50],
        start_pos=[0],
        empty_slots=[0],
        sampling_params=_greedy(1),
    )
    tok = int(out[0][0, 0])
    out = ad.decode_forward(
        torch.tensor([[tok], [0], [0], [0]], dtype=torch.int32),
        torch.tensor([50, -1, -1, -1], dtype=torch.int32),
        page_table=_table([_blocks(0, 51)], B),
        kv_cache=kv,
        enable_trace=True,
        read_from_device=False,
        sampling_params=None,
        reload_inputs=True,
    )
    assert isinstance(out, tuple) and out[1] is None
    logits = out[0]
    assert logits.shape == (B, 1, VOCAB) and logits.dtype == torch.float32
    assert int(logits[0, 0].argmax()) == (tok + 1) % VOCAB
    assert ad.process_decode_output_host(out, is_tokens=False) is out  # a host output passes through
    passed, events = ad.read_decode_output(out, async_read=True)
    assert passed is out and events == []
    host_pick = (tok + 7) % VOCAB  # the host sampler chose another token: the next step must consume it
    got = ad.process_decode_output_host(
        _decode(ad, kv, [host_pick, 0, 0, 0], [51, -1, -1, -1], [_blocks(0, 52)]), is_tokens=True
    )
    assert got.reshape(-1).tolist() == [(host_pick + 1) % VOCAB, 0, 0, 0]
    assert kv.state[0] == p + [tok, host_pick]
    assert gen.counters["logits_readback"] == 1 and gen.counters["token_refresh"] >= 1


def test_refusals(monkeypatch):
    """Undeclared routes fail loudly: sampled rows on the device route, selective readback, speculative kwargs, the
    legacy reload keyword, a resident-input chain, multimodal rows."""
    monkeypatch.setattr(gvs, "DEVICE_TOP_K", 1)  # the greedy-only construction (GLM53_STACK_DEVICE_SAMPLING=0)
    monkeypatch.setattr(gvs, "ASYNC_DECODE", False)  # (it follows the device sampler)
    ad, backend, gen, kv = _adapter()
    p = _prompt(10, 1)
    sampled = SimpleNamespace(temperature=[0.8], top_k=[50], top_p=[0.9], seed=[None])
    with expect_error(ValueError, "sampled"):
        ad.prefill_forward(
            _history([p]),
            page_table=_table([_blocks(0, 10)], 1),
            kv_cache=kv,
            prompt_lens=[10],
            start_pos=[0],
            empty_slots=[0],
            sampling_params=sampled,
        )
    with expect_error(NotImplementedError, "multimodal"):
        ad.prefill_forward(
            _history([p]),
            page_table=_table([_blocks(0, 10)], 1),
            kv_cache=kv,
            prompt_lens=[10],
            start_pos=[0],
            empty_slots=[0],
            pixel_values=[None],
        )
    out = ad.prefill_forward(
        _history([p]),
        page_table=_table([_blocks(0, 10)], 1),
        kv_cache=kv,
        prompt_lens=[10],
        start_pos=[0],
        empty_slots=[0],
        sampling_params=_greedy(1),
    )
    tok = int(out[0][0, 0])
    kw = dict(page_table=_table([_blocks(0, 11)], B), kv_cache=kv, enable_trace=True, read_from_device=False)
    tokens, positions = torch.tensor([[tok], [0], [0], [0]], dtype=torch.int32), torch.tensor([10, -1, -1, -1])
    with expect_error(NotImplementedError, "speculative"):
        ad.decode_forward(tokens, positions, sampling_params=_greedy(B), spec_mode="verify", num_valid_drafts=[1], **kw)
    with expect_error(TypeError, "reset_batch"):
        ad.decode_forward(tokens, positions, sampling_params=_greedy(B), reset_batch=True, **kw)
    with expect_error(ValueError, "reload_inputs=False"):
        ad.decode_forward(tokens, positions, sampling_params=_greedy(B), reload_inputs=False, **kw)
    with expect_error(ValueError, "permutation"):
        ad.decode_forward(tokens, positions, sampling_params=_greedy(B), slot_remap=torch.tensor([0, 0, 2, 3]), **kw)
    with expect_error(ValueError, "full-width"):  # the plugin's gather names every slot
        ad.decode_forward(tokens, positions, sampling_params=_greedy(B), slot_remap=torch.tensor([1, 0]), **kw)
    assert ad._alias == [0, 1, 2, 3]  # a refused gather leaves the alias alone
    # the alias and the plugin's ownership map must agree: a fresh request into a live generator slot, a continuation
    # into an idle one, are refused at the boundary
    p2 = _prompt(100, 2)
    with expect_error(RuntimeError, "still holds a live request"):
        ad.prefill_forward(_history([p2]), page_table=_table([_blocks(1, 100)], 1), kv_cache=kv, prompt_lens=[100],
                           start_pos=[0], empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    with expect_error(RuntimeError, "holds no live request"):
        ad.prefill_forward(_history([p2]), page_table=_table([_blocks(1, 100)], 1), kv_cache=kv, prompt_lens=[100],
                           start_pos=[64], empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    with expect_error(ValueError, "sampled"):
        ad.decode_forward(
            tokens,
            positions,
            sampling_params=SimpleNamespace(temperature=[0.5] * B, top_k=[0] * B, top_p=[1.0] * B, seed=[None] * B),
            **kw,
        )
    out = ad.decode_forward(tokens, positions, sampling_params=_greedy(B), **kw)
    with expect_error(NotImplementedError, "selective"):
        ad.read_decode_output(out, async_read=True, sample_rows=[0])
    with expect_error(NotImplementedError, "compact"):
        ad.process_decode_output_host(out, is_tokens=True, sample_rows=[0])
    with expect_error(TypeError, "expects a DecodeOutput"):
        ad.read_decode_output(object(), async_read=True)
    assert ad.process_decode_output_host(out, is_tokens=True).reshape(-1).tolist() == [(tok + 1) % VOCAB, 0, 0, 0]
    with expect_error(ValueError, "outside"):  # outside the slot range: a contract break, refused
        ad.release_request(7)
    assert backend.releases == []


def test_fabric_capability_uses_set_fabric_config_keywords(monkeypatch):
    """With ttnn present the capability carries ``config`` from ``stack_model.serving_fabric_config`` and the
    ``reliability_mode`` / ``router_config`` of ``tt/mesh.py::mesh_device_params`` (stubbed: both import ttnn)."""
    stack = types.ModuleType("models.autoports.zai_org_glm_5_3_flash.tt.stack_model")
    stack.serving_fabric_config = lambda: {"config": "FABRIC_1D", "router_config": "default-router"}
    mesh = types.ModuleType("models.demos.glm53_flash.tt.mesh")
    seen = []

    def mesh_device_params(rows, cols, **overrides):
        seen.append((rows, cols))
        return {"fabric_config": "TORUS", "reliability_mode": "RELAXED", "fabric_router_config": "max-payload"}

    mesh.mesh_device_params = mesh_device_params
    monkeypatch.setitem(sys.modules, "ttnn", types.ModuleType("ttnn"))
    monkeypatch.setitem(sys.modules, "models.autoports.zai_org_glm_5_3_flash.tt.stack_model", stack)
    monkeypatch.setitem(sys.modules, "models.demos.glm53_flash.tt.mesh", mesh)
    monkeypatch.setenv("GLM53_MESH_SHAPE", "8,4")
    assert gvs._fabric_capability() == {
        "config": "FABRIC_1D",
        "reliability_mode": "RELAXED",
        "router_config": "max-payload",
    }
    assert seen == [(8, 4)]


def test_zz_no_ttnn_was_imported_by_the_suite():
    """Runs last (file order): every adapter path the mock exercised stayed off ttnn."""
    assert "ttnn" not in sys.modules


def test_sampled_rows_take_the_device_route():
    """With the device sampler (the default construction) a sampled row -- temperature 1.0 / top-p 0.95 and no top-k,
    GLM's generation config -- is passed to the generator on the device route instead of refused."""
    ad, backend, gen, kv = _adapter()
    p = _prompt(10, 1)
    sampled = SimpleNamespace(temperature=[1.0], top_k=[0], top_p=[0.95], seed=[None])
    out = ad.prefill_forward(
        _history([p]),
        page_table=_table([_blocks(0, 10)], 1),
        kv_cache=kv,
        prompt_lens=[10],
        start_pos=[0],
        empty_slots=[0],
        sampling_params=sampled,
    )
    assert isinstance(out, tuple) and out[0].shape == (1, 1)
    assert gen.live_slots(kv) == [0]


def test_device_top_k_follows_the_head_gather(monkeypatch):
    """The bound is the largest top-K the TP head's candidate gather carries (K x 6 fields per chip in 2,048 columns):
    10 on the 32-chip 8x4, 32 on an 8-chip mesh; 1 (greedy only) with GLM53_STACK_DEVICE_SAMPLING=0."""
    monkeypatch.setenv("GLM53_MESH_SHAPE", "8,4")
    assert gvs._device_top_k() == 10
    monkeypatch.setenv("GLM53_MESH_SHAPE", "2,4")
    assert gvs._device_top_k() == 32
    monkeypatch.setenv("GLM53_STACK_DEVICE_SAMPLING", "0")
    assert gvs._device_top_k() == 1


def test_resident_chain_decode_follows_the_reload_steps():
    """The async contract: after a reload step, steady chain steps (reload_inputs=False: nothing staged from the host)
    continue every live slot from its device-resident token, the rows come back in the plugin's order through the slot
    alias, and a page-table-only refresh lets a slot cross a block boundary; a chain step that carries a remap is
    refused."""
    assert gvs.ASYNC_DECODE and GLM53StackForCausalLM.model_capabilities["supports_async_decode"] is True
    ad, backend, gen, kv = _adapter()
    prompts = [_prompt(60, 3), _prompt(62, 9)]
    out = ad.prefill_forward(
        _history(prompts),
        page_table=_table([_blocks(i, len(p)) for i, p in enumerate(prompts)], 2),
        kv_cache=kv,
        prompt_lens=[len(p) for p in prompts],
        start_pos=[0, 0],
        empty_slots=[0, 1],
        sampling_params=_greedy(2),
    )
    cur, pos = [int(v) for v in out[0].reshape(-1).tolist()], [60, 62]
    # the reload step with a 2-cycle gather (plugin rows [1, 0]: the alias absorbs it)
    remap = [1, 0, 2, 3]
    rows = [_blocks(1, pos[1] + 1), _blocks(0, pos[0] + 1)]
    got = ad.process_decode_output_host(
        _decode(ad, kv, [cur[1], cur[0], 0, 0], [pos[1], pos[0], -1, -1], rows, remap=remap), is_tokens=True
    )
    assert got.reshape(-1).tolist()[:2] == [(cur[1] + 1) % VOCAB, (cur[0] + 1) % VOCAB]
    nxt = got.reshape(-1).tolist()[:2]  # plugin row order: [request 1, request 0]
    pos = [pos[0] + 1, pos[1] + 1]

    def chain(page_rows=None):
        kw = dict(
            page_table=None if page_rows is None else _table(page_rows, B),
            kv_cache=kv,
            enable_trace=True,
            read_from_device=False,
            sampling_params=_greedy(B),
            slot_remap=None,
            reload_inputs=False,
            reload_page_table=page_rows is not None,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        h = ad.decode_forward(torch.zeros(B, 1, dtype=torch.int32), torch.full((B,), -1), **kw)
        assert isinstance(h, DecodeOutput) and h.rows == ad._alias[:B]
        return ad.process_decode_output_host(h, is_tokens=True).reshape(-1).tolist()[:2]

    for _ in range(3):  # positions 61..63 / 63..65: request 1 crosses 64 -> its next block arrives by a page refresh
        if pos[1] + 1 > 64:
            res = chain([_blocks(1, pos[1] + 1), _blocks(0, pos[0] + 1)])
        else:
            res = chain()
        assert res == [(t + 1) % VOCAB for t in nxt]
        nxt, pos = res, [pos[0] + 1, pos[1] + 1]
    for row in range(2):  # plugin row r decoded in generator slot alias[r]: its last input = its token before
        assert kv.state[ad._alias[row]][-1] == (nxt[row] - 1) % VOCAB
    # a request mid-prefill (first chunk of a 300-token prompt in plugin slot 2: its generator slot is active) stays out
    # of the chain until a reload step adds it (the served 32 K x 4 crash: the chain decoded it past its page rows)
    long = _prompt(300, 23)
    ad.prefill_forward(
        _history([long[:128]]),
        page_table=_table([_blocks(2, 128)], 1),
        kv_cache=kv,
        prompt_lens=[128],
        start_pos=[0],
        empty_slots=[2],
        sampling_params=None,
    )
    g2 = ad._alias[2]
    assert len(kv.state[g2]) == 128 and g2 not in ad._chain
    res = chain([_blocks(1, pos[1] + 1), _blocks(0, pos[0] + 1)])  # request 0 reaches position 64: its next block
    assert res == [(t + 1) % VOCAB for t in nxt] and len(kv.state[g2]) == 128
    with expect_error(ValueError, "resident chain step"):
        ad.decode_forward(
            torch.zeros(B, 1, dtype=torch.int32),
            torch.full((B,), -1),
            page_table=None,
            kv_cache=kv,
            sampling_params=_greedy(B),
            slot_remap=torch.tensor(remap),
            reload_inputs=False,
        )


# ------------------------------------------------------------------------------------------ the prefix cache
class _MockPrefixBackend(_MockStackBackend):
    """+ the prefix cache's device side: a snapshot pool of ``snapshot_entries`` virtual slots (max_batch + k) read /
    written inside the prefill "trace" (``prefill_state_io``: state_src before the chunk, state_snap after it), the
    sequence-sharded pool's owner-aware blocks (``dsa_shard_owners`` -> the generator's OwnerBlockMap) and page
    contents: every chunk / decode step writes its tokens into the physical pages of its row and first asserts that the
    pages of its context hold exactly the tokens its state consumed (a wrong shared page, an attach from the wrong
    entry or a block evicted under a hit shows)."""

    snapshot_entries = 3
    prefill_state_io = True
    C = 4

    def __init__(self, max_batch: int):
        super().__init__(max_batch)
        self.state_io: list = []  # (slot, chunk_start, state_src, state_snap) of every chunk

    def dsa_shard_owners(self, kv):
        return (self.C, OwnerBlockMap.local_blocks_for(self.C, kv.num_blocks, self.max_batch, 2), 2)

    @staticmethod
    def _page_check(kv, slot: int, n: int):
        row, pages = kv.page_table_host[slot], kv.__dict__.setdefault("pages", {})
        got = [pages.get((int(row[p // BLOCK_SIZE]), p % BLOCK_SIZE)) for p in range(n)]
        assert got == kv.state[slot][:n], f"slot {slot}: its pages do not hold the {n} tokens of its state"

    @staticmethod
    def _page_write(kv, slot: int, start: int, toks):
        row, pages = kv.page_table_host[slot], kv.__dict__.setdefault("pages", {})
        for k, t in enumerate(toks):
            pages[(int(row[(start + k) // BLOCK_SIZE]), (start + k) % BLOCK_SIZE)] = int(t)

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None,
                      state_src=None, state_snap=None, **kw):  # fmt: skip
        self.state_io.append((slot, chunk_start, state_src, state_snap))
        if state_src is not None:  # a fresh request attaching at chunk_start: the entry's state starts the chunk
            assert chunk_start > 0
            kv_cache.state[slot] = list(kv_cache.state[self.max_batch + state_src])
        if chunk_start:
            self._page_check(kv_cache, slot, chunk_start)
        out = super().prefill_chunk(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start, n_valid=n_valid,
                                    logits_rows=logits_rows, sample=sample, io=io)  # fmt: skip
        self._page_write(kv_cache, slot, chunk_start, tokens[:n_valid].tolist())
        if state_snap is not None:
            kv_cache.state[self.max_batch + state_snap] = list(kv_cache.state[slot])
        return out

    def decode(self, io, kv, width):
        for s in range(self.max_batch):
            p = int(io.cur_pos.t[s])
            if p >= 0:
                self._page_check(kv, s, p)
                self._page_write(kv, s, p, [int(io.tokens.t[s])])
        super().decode(io, kv, width)


def _prefix_adapter(monkeypatch):
    """the adapter under GLM53_STACK_PREFIX=1 (read at import: the module flag) on the prefix-capable mock"""
    monkeypatch.setattr(gvs, "PREFIX_CACHE", True)
    backend = _MockPrefixBackend(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.prefix is not None and gen.shared_pages and gen.dsa_shard_map(kv) is not None
    assert ad.prefix.grid == 128 and ad.prefix.store.device_entries == 3
    return ad, backend, gen, kv


def _prefill_steps(ad, kv, prompt: list, row: list, slot: int, ends: list):
    """one prompt through the engine steps that end at ``ends`` (the plugin's chunked prefill: intermediate chunks
    host-routed, the last sampled on device) -> the first token"""
    start = 0
    for end in ends:
        last = end == len(prompt)
        out = ad.prefill_forward(_history([prompt[:end]]), page_table=_table([row[: -(-end // BLOCK_SIZE)]], 1),
                                 kv_cache=kv, prompt_lens=[end], start_pos=[start], empty_slots=[slot],
                                 sampling_params=_greedy(1) if last else None)  # fmt: skip
        start = end
    return int(out[0][0, 0])


def test_prefix_cache_snapshot_confirm_attach_release(monkeypatch):
    """GLM53_STACK_PREFIX: prompt A (a 256-token document + its question) in three engine steps snapshots its state at
    the on-grid chunk ends 128 / 256 (inside the prefill: snapshot_after); after A finished, the scheduler's
    confirmation answers B's proposed hits (B = the document + another question) with the longest resident snapshot
    and pins it; B's prefill at start 256 in a free plugin slot is admitted as a NEW request (the lowest free generator
    slot), attaches the entry (prefix_attach) on A's cached blocks and prefills only its suffix; B and the cold B
    decode alike; B's release detaches; an identical resend of A attaches too."""
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    pc = ad.prefix
    doc = _prompt(256, 5)
    a, b = doc + _prompt(40, 50), doc + _prompt(30, 20)
    row_a = _blocks(0, len(a))
    first_a = _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, len(a)])
    assert backend.state_io == [(0, 0, None, 0), (0, 128, None, 1), (0, 256, None, None)]
    assert pc.stats["snapshots"] == 2 and sorted(e.n_tokens for e in pc.store.entries.values()) == [128, 256]
    key256 = pc.store.key(a, 256)
    e256 = pc.store.entries[key256]
    assert e256.device_entry == 1 and list(e256.pages) == row_a[:4] and kv.state[B + 1] == doc
    ad.release_request(0)  # A finished: its blocks stay mapped (cached), its snapshots stay registered
    assert gen.live_slots(kv) == [] and pc.store.attached == {}
    # ---- the scheduler's confirmation of vLLM's proposed hits for B
    assert ad.confirm_prefix_cache_hit(b, [], 255) == 128  # below 256: the 128 snapshot
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256  # the same prompt confirmed again: one pin, moved
    assert e256.readers == 1 and pc.store.entries[pc.store.key(a, 128)].readers == 0 and len(pc.pins) == 1
    assert ad.confirm_prefix_cache_hit(b, [], 0) == 0 and ad.confirm_prefix_cache_hit(b, [], 64) == 0
    assert ad.confirm_prefix_cache_hit(b, [("img", 3, 4, "image")], 256) == 0  # text only
    assert ad.confirm_prefix_cache_hit(_prompt(300, 33), [], 256) == 0  # another prefix
    # ---- B: admitted at 256 in plugin slot 2 (vLLM's cached blocks first in its row) -> generator slot 0
    row_b = row_a[:4] + _blocks(1, len(b))[4:]
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    gb = ad._alias[2]
    assert gb == 0 and ad._alias == [2, 1, 0, 3]  # the lowest free generator slot, as a first chunk takes
    assert backend.state_io[-1] == (gb, 256, 1, None)  # the suffix only, from entry 1
    assert kv.state[gb] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB
    assert pc.store.attached == {gb: key256} and e256.readers == 1 and not pc.pins and pc.stats["attaches"] == 1
    assert gen.counters["prefix_attaches"] == 1
    # ---- B cold (plugin slot 1, its own blocks), then both decode: the same tokens, each on its own pages
    row_c = _blocks(2, len(b))
    out = ad.prefill_forward(_history([b]), page_table=_table([row_c], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[0], empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    gc = ad._alias[1]
    assert int(out[0][0, 0]) == (b[-1] + 1) % VOCAB and kv.state[gc] == b
    cur, pos = [int(out[0][0, 0])] * 2, len(b)
    for _ in range(3):
        got = ad.process_decode_output_host(_decode(ad, kv, [0, cur[0], cur[1], 0], [-1, pos, pos, -1],
                                                    [[], row_c, row_b]), is_tokens=True).reshape(-1).tolist()  # fmt: skip
        assert got[1] == got[2] == (cur[0] + 1) % VOCAB
        cur, pos = [got[1], got[2]], pos + 1
    assert kv.state[gb] == kv.state[gc]
    # ---- B finishes: its slot detaches from the entry
    ad.release_request(2)
    assert pc.store.attached == {} and e256.readers == 0
    # ---- an identical resend of A: vLLM proposes its 4 full cached blocks, the snapshot at 256 serves them
    assert ad.confirm_prefix_cache_hit(a, [], 256) == 256
    free = [p for p in range(B) if ad._alias[p] not in gen.live_slots(kv)][0]
    out = ad.prefill_forward(_history([a]), page_table=_table([row_a[:4] + _blocks(3, len(a))[4:]], 1), kv_cache=kv,
                             prompt_lens=[len(a)], start_pos=[256], empty_slots=[free], sampling_params=_greedy(1))  # fmt: skip
    assert int(out[0][0, 0]) == first_a and backend.state_io[-1][1:3] == (256, 1)
    assert pc.stats["attaches"] == 2 and pc.stats["fallbacks"] == 0
    assert pc.report()["hits"] == 2


def test_prefix_cache_pins_lru_and_eviction_fallbacks(monkeypatch):
    """A confirmed (pinned) entry survives the LRU reuse of the pool by another prompt's snapshots and is attached
    after it; a snapshot whose pages lost a block (the owner pool evicted it) is not confirmed -- a shorter resident
    one is, else none; a block evicted between the confirmation and the prefill, or a hit at a position without a
    snapshot, prefills the request from position 0 instead (the plugin passes the whole history)."""
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    pc, obm = ad.prefix, gen.dsa_shard_map(kv)
    doc = _prompt(256, 5)
    a, b = doc + _prompt(40, 50), doc + _prompt(30, 20)
    row_a = _blocks(0, len(a))
    _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, len(a)])
    ad.release_request(0)
    key128, key256 = pc.store.key(a, 128), pc.store.key(a, 256)
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256  # pinned until B's prefill
    # ---- prompt C (384 tokens of another document) snapshots at 128 / 256 / 384 with one free entry of three: the
    #      least recently used UNPINNED entries are reused (A's 128, then C's own 128); A's pinned 256 survives
    c = _prompt(384, 9) + _prompt(10, 33)
    row_c = _blocks(1, len(c))
    _prefill_steps(ad, kv, c, row_c, slot=1, ends=[128, 256, 384, len(c)])
    assert [r[3] for r in backend.state_io[-4:]] == [2, 0, 2, None]
    assert key128 not in pc.store.entries and key256 in pc.store.entries and pc.stats["snapshots"] == 5
    row_b = row_a[:4] + _blocks(2, len(b))[4:]
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[0]
    assert backend.state_io[-1] == (g, 256, 1, None) and kv.state[g] == b  # entry 1 was never overwritten
    assert int(out[0][0, 0]) == (b[-1] + 1) % VOCAB
    ad.release_request(0)
    ad.release_request(1)
    # ---- residency: C's 384 snapshot loses a block of [256, 384) -> the 256 one is confirmed; then nothing
    c2 = c[:384] + _prompt(20, 40)
    assert ad.confirm_prefix_cache_hit(c2, [], 384) == 384
    obm.forget(row_c[5])  # the owner pool evicts a released block (OwnerBlockMap: least recently released first)
    assert ad.confirm_prefix_cache_hit(c2, [], 384) == 256 and pc.stats["not_resident"] == 1
    obm.forget(row_c[0])
    assert ad.confirm_prefix_cache_hit(c2, [], 384) == 0 and not pc.pins  # (the 256 answer's pin dropped)
    # ---- the race: B confirmed while A's blocks are resident, one evicted before its prefill -> from 0, correct
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256
    obm.forget(row_a[1])
    n = len(backend.state_io)
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[2]
    assert [r[1:3] for r in backend.state_io[n:]] == [(0, None), (128, None), (256, None)]  # no attach
    assert kv.state[g] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB and obm.physical(row_a[1]) is not None
    assert pc.stats["fallbacks"] == 1 and not pc.pins and pc.store.entries[key256].readers == 0
    assert pc.store.attached == {}  # nothing to detach at its release
    ad.release_request(2)
    # ---- a hit at a position without a snapshot (192: vLLM's proposal not trimmed by this cache) -> from 0
    n = len(backend.state_io)
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[192], empty_slots=[3], sampling_params=_greedy(1))  # fmt: skip
    assert backend.state_io[n][1:3] == (0, None) and kv.state[ad._alias[3]] == b and pc.stats["fallbacks"] == 2

# ------------------------------------------------------------------------------- image input (GLM53_STACK_MM)
MM_WIDTH = 8  # the mock's feature width (the generator never looks at it)


def _img_rows(positions) -> torch.Tensor:
    """feature rows whose first value names the prompt position they belong to (the mock's ledger shows it)"""
    pos = torch.as_tensor(positions, dtype=torch.int64)
    return pos.to(torch.bfloat16)[:, None].expand(-1, MM_WIDTH).contiguous()


def _mm_prompt(n: int, lo: int, hi: int, base: int) -> list[int]:
    """an n-token text prompt whose positions [lo, hi) are the image placeholder run"""
    return [IMAGE_TOKEN_ID if lo <= i < hi else t for i, t in enumerate(_prompt(n, base))]


def _ledger(prompt: list[int], lo: int, hi: int) -> list:
    """what the mock's slot state holds after the prompt: text ids, ("img", position) on the image rows"""
    return [("img", i) if lo <= i < hi else t for i, t in enumerate(prompt)]


def _mm_generator():
    backend = _MockStackBackend(B)
    return backend, Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)


def test_generator_stages_image_rows_per_planned_chunk():
    """generate_batch(mm=...): an image run straddling the 128-token chunk boundary -- each planned chunk stages ITS
    rows (in order, before its replay) into the extra rows and its image positions take the ids VOCAB + j; text ids are
    untouched. The refusals: positions outside the prompt, a feature row count
    that is not the position count, more image rows in one chunk than extra rows, a backend without extra rows."""
    backend, gen = _mm_generator()
    lo, hi = 100, 160
    p = _mm_prompt(200, lo, hi, 7)
    pos = torch.arange(lo, hi)
    out = gen.generate_batch([p], 1, mm=[(pos, _img_rows(pos))])  # (the mock models no generator-owned decode)
    assert out == [[(p[-1] + 1) % VOCAB]]
    assert [c[1:3] for c in backend.chunks] == [(0, 128), (128, 72)]
    assert [(k, r[:, 0].tolist()) for k, r in backend.mm_stages] == [
        (0, list(range(100, 128))),  # chunk 0's rows, staged before chunk 0 ran
        (1, list(range(128, 160))),  # chunk 1's rows, after chunk 0, before chunk 1
    ]
    want0 = torch.tensor(p[:128])
    want0[100:128] = VOCAB + torch.arange(28)
    want1 = torch.tensor(p[128:200])
    want1[:32] = VOCAB + torch.arange(32)
    assert torch.equal(backend.chunk_ids[0], want0) and torch.equal(backend.chunk_ids[1], want1)
    assert gen._own_binding().kv_cache.state[0][:200] == _ledger(p, lo, hi)
    assert gen.counters["mm_chunks"] == 2 and gen.counters["mm_rows"] == 60
    # refusals
    with expect_error(ValueError, "outside its"):
        gen.generate_batch([p], 1, mm=[(torch.tensor([250]), _img_rows([250]))])
    with expect_error(ValueError, "one \\[hidden\\] row each"):
        gen.generate_batch([p], 1, mm=[(pos, _img_rows(pos)[:10])])
    backend2, gen2 = _mm_generator()
    big = torch.arange(10, 110)  # 100 image rows inside the first 128-token chunk > the mock's 64 extra rows
    with expect_error(ValueError, "64 extra rows"):
        gen2.generate_batch([_mm_prompt(200, 10, 110, 3)], 1, mm=[(big, _img_rows(big))])
    backend3, gen3 = _mm_generator()
    backend3.mm_extra_rows = 0  # the text build (GLM53_STACK_MM off)
    with expect_error(ValueError, "GLM53_STACK_MM=1"):
        gen3.generate_batch([p], 1, mm=[(pos, _img_rows(pos))])
    assert backend3.chunks == []  # refused before any chunk ran


def test_image_prompts_stay_out_of_the_batched_and_packed_replays(monkeypatch):
    """A batched admission (generate_batch) and a packed step (prefill_forward) take the text prompts only: the image
    prompt runs the per-user path, where its chunk is staged and remapped."""
    backend, gen = _mm_generator()
    calls = []

    def batch_tp(reqs, *, kv_cache, io, sample=None):
        calls.append(("batch", [int(slot) for _, slot, _ in reqs]))
        firsts = []
        for ids, slot, n in reqs:
            kv_cache.state[slot] = [int(t) for t in ids[:n].tolist()]
            nxt = (int(ids[n - 1]) + 1) % VOCAB
            io.tokens.t[slot], io.sampled.t[slot] = nxt, nxt
            firsts.append(nxt)
        return firsts

    backend.prefill_batch_size = lambda: (128, 2)
    backend.prefill_batch_tp = batch_tp
    lo, hi = 20, 40
    texts = [_prompt(50, 1), _prompt(60, 2)]
    img = _mm_prompt(70, lo, hi, 3)
    pos = torch.arange(lo, hi)
    out = gen.generate_batch([texts[0], img, texts[1]], 1, mm=[None, (pos, _img_rows(pos)), None])
    assert calls == [("batch", [0, 2])]  # the two text prompts in one replay, the image prompt alone
    assert backend.chunks == [(1, 0, 70, "last", True)]
    assert gen._own_binding().kv_cache.state[1] == _ledger(img, lo, hi)
    assert out == [[(texts[0][-1] + 1) % VOCAB], [(img[-1] + 1) % VOCAB], [(texts[1][-1] + 1) % VOCAB]]

    # prefill_forward under a packed-step backend (continuation + head-less segments): the image row runs alone
    backend, gen = _mm_generator()
    packed = []

    def packed_tp(reqs, *, kv_cache, io, t, sample=None, starts=None, heads=None):
        packed.append([int(slot) for _, slot, _ in reqs])
        firsts = []
        for (ids, slot, n), st in zip(reqs, starts):
            kv_cache.state[slot] = kv_cache.state[slot][:st] + [int(x) for x in ids[:n].tolist()]
            nxt = (int(ids[n - 1]) + 1) % VOCAB
            io.tokens.t[slot], io.sampled.t[slot] = nxt, nxt
            firsts.append(nxt)
        return firsts

    backend.prefill_packed_caps = lambda: {"continuation": True, "no_head": True}
    backend.prefill_packed_sizes = lambda: [(256, 4)]
    backend.prefill_packed_tp = packed_tp
    kv = gen.allocate_kv_cache(B, SEQ, num_blocks=NUM_BLOCKS, null_block=0)
    gen.warmup(kv, capture=True)
    prompts = [texts[0], img, texts[1]]
    toks = _history(prompts)
    toks[1, lo:hi] = IMAGE_TOKEN_ID
    got = gen.prefill_forward(
        toks,
        page_table=trim_page_rows(_table([_blocks(i, len(q)) for i, q in enumerate(prompts)], 3)),
        kv_cache=kv,
        prompt_lens=[len(q) for q in prompts],
        empty_slots=[0, 1, 2],
        sampling_params=[SamplingParams()] * 3,
        mm_rows={1: (pos, _img_rows(pos))},
    )
    assert packed == [[0, 2]] and backend.chunks == [(1, 0, 70, "last", True)]
    assert kv.state[1] == _ledger(img, lo, hi)
    assert got.tolist() == [(q[-1] + 1) % VOCAB for q in prompts]


def test_multimodal_class_declaration_with_and_without_vllm():
    """GLM53StackForConditionalGeneration: the text adapter + SupportsMultiModal + the upstream processor registration
    (a fake vLLM api here); the text class stays text-only. Without vLLM (this host) the module-level class is the plain
    declaration with the same flags, reachable through MM_ADAPTER_TARGET."""
    plain = declare_multimodal_class(None)
    assert issubclass(plain, GLM53StackForCausalLM) and plain.supports_multimodal is True
    assert GLM53StackForCausalLM.supports_multimodal is False
    assert plain.__name__ == plain.__qualname__ == "GLM53StackForConditionalGeneration"
    assert plain.get_placeholder_str("image", 0) == "<|begin_of_image|><|image|><|end_of_image|>"
    with expect_error(ValueError, "Only image"):
        plain.get_placeholder_str("video", 0)
    assert plain.model_capabilities is GLM53StackForCausalLM.model_capabilities
    assert plain.decode_input_update_contract == 1 and plain.mm_encoder is None

    class SupportsMultiModal:
        supports_multimodal = True

    seen = {}

    def register_processor(cls):
        seen["cls"] = cls
        cls._processor_factory = "glm5next"
        return cls

    mm = declare_multimodal_class(
        SimpleNamespace(SupportsMultiModal=SupportsMultiModal, register_processor=register_processor)
    )
    assert seen["cls"] is mm and mm._processor_factory == "glm5next"
    assert issubclass(mm, SupportsMultiModal) and issubclass(mm, GLM53StackForCausalLM)
    mod, _, name = gvs.MM_ADAPTER_TARGET.partition(":")
    assert getattr(importlib.import_module(mod), name) is gvs.GLM53StackForConditionalGeneration
    assert gvs.GLM53StackForConditionalGeneration.supports_multimodal is True
    assert "vllm" not in sys.modules  # the host declaration never imported vLLM (its platform plugins import ttnn)


def test_vllm_multimodal_api_takes_the_031_module_then_the_030_one(monkeypatch):
    """_vllm_multimodal_api in a vLLM process: the processor from vllm.models.glm5next.common.multimodal (0.31), else
    .nvidia.multimodal (0.30), registered with Glm4v's dummy-inputs builder; None without either, and None in a process
    that has not imported vLLM (fake vLLM modules here)."""
    reg = []

    class Registry:
        @staticmethod
        def register_processor(proc, *, info, dummy_inputs):
            reg.append((proc.__module__, info.__module__, dummy_inputs))
            return lambda cls: cls

    def module(name, **attrs):
        m = types.ModuleType(name)
        for k, v in attrs.items():
            if isinstance(v, type):
                v.__module__ = name
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    assert gvs._vllm_multimodal_api() is None  # no vLLM in this process
    module("vllm")
    module("vllm.model_executor.models.glm4_1v", Glm4vDummyInputsBuilder="builder")
    module("vllm.model_executor.models.interfaces", SupportsMultiModal=type("SupportsMultiModal", (), {}))
    module("vllm.multimodal", MULTIMODAL_REGISTRY=Registry)
    new, old = gvs.GLM5NEXT_PROCESSOR_MODULES
    assert gvs._vllm_multimodal_api() is None  # neither processor module
    module(old, Glm5NextMultiModalProcessor=type("P", (), {}), Glm5NextProcessingInfo=type("I", (), {}))
    api = gvs._vllm_multimodal_api()
    assert api.processor_module == old and reg[-1] == (old, old, "builder")
    module(new, Glm5NextMultiModalProcessor=type("P", (), {}), Glm5NextProcessingInfo=type("I", (), {}))
    api = gvs._vllm_multimodal_api()
    assert api.processor_module == new == "vllm.models.glm5next.common.multimodal" and reg[-1] == (new, new, "builder")
    cls = declare_multimodal_class(api)
    assert issubclass(cls, sys.modules["vllm.model_executor.models.interfaces"].SupportsMultiModal)


def test_multimodal_adapter_routes_image_items_to_the_generator(monkeypatch):
    """The plugin's per-row image items (pixel_values / image_grid_thw lists + placeholder ranges) on the multimodal
    class: the item's rows found in the row's prefix, the item encoded (fake encoder), the chunk's rows handed to the
    generator -- an image straddling the two scheduler chunks of a 200-token prompt is staged by both; text rows of the
    same batch and a text-only batch (pixel_values=[None]) take the text path; videos, a missing encoder and a backend
    without extra rows are refused; the text class still refuses pixels."""
    cls = declare_multimodal_class(None)
    backend = _MockStackBackend(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = cls(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B)
    encoded = []

    def encoder(items):  # row j of an item = its j-th feature row; the first value names it (offset + j)
        encoded.append([it.n_tokens for it in items])
        return [_img_rows(torch.arange(it.offset, it.offset + it.n_tokens)) for it in items]

    ad.mm_encoder = encoder
    lo, n_img = 112, 32  # grid (1, 8, 16): 128 patches -> 32 image tokens at [112, 144)
    img = _mm_prompt(200, lo, lo + n_img, 9)
    text = _prompt(90, 4)
    grid = torch.tensor([[1, 8, 16]])
    pv = torch.zeros(128, 4)
    kw = dict(pixel_values=[None, [pv]], image_grid_thw=[None, [grid]], mm_placeholder_offsets=[None, [lo]],
              mm_placeholder_lengths=[None, [n_img]])  # fmt: skip
    # step 1: the text prompt whole + the image prompt's first 128 tokens (host-routed intermediate chunk)
    first = ad.prefill_forward(
        _history([text, img[:128]]),
        page_table=_table([_blocks(0, 90), _blocks(1, 128)], 2),
        kv_cache=kv,
        prompt_lens=[90, 128],
        start_pos=[0, 0],
        empty_slots=[0, 1],
        sampling_params=None,
        **kw,
    )
    assert first.shape == (2, 1, VOCAB) and encoded == [[32]]
    assert [k for k, _ in backend.mm_stages] == [1] and backend.mm_stages[0][1][:, 0].tolist() == list(range(112, 128))
    assert kv.state[0] == text and kv.state[1] == _ledger(img[:128], lo, lo + n_img)
    # step 2: the rest of the image prompt; the plugin sends the item again, its remaining rows are staged
    out = ad.prefill_forward(
        _history([img]),
        page_table=_table([_blocks(1, 200)], 1),
        kv_cache=kv,
        prompt_lens=[200],
        start_pos=[128],
        empty_slots=[1],
        sampling_params=_greedy(1),
        **{k: [v[1]] for k, v in kw.items()},
    )
    assert int(out[0][0, 0]) == (img[-1] + 1) % VOCAB and encoded == [[32], [32]]
    assert backend.mm_stages[-1][1][:, 0].tolist() == list(range(128, 144))
    assert kv.state[1] == _ledger(img, lo, lo + n_img)
    # a text-only batch on the multimodal class: the text path (no encoder call, no stage)
    ad.prefill_forward(_history([text]), page_table=_table([_blocks(2, 90)], 1), kv_cache=kv, prompt_lens=[90],
                       start_pos=[0], empty_slots=[2], sampling_params=_greedy(1), pixel_values=[None],
                       image_grid_thw=[None], mm_placeholder_offsets=[None], mm_placeholder_lengths=[None])  # fmt: skip
    assert len(encoded) == 2 and len(backend.mm_stages) == 2
    # refusals
    one = dict(page_table=_table([_blocks(3, 200)], 1), kv_cache=kv, prompt_lens=[200], start_pos=[0], empty_slots=[3],
               sampling_params=_greedy(1))  # fmt: skip
    with expect_error(NotImplementedError, "images only"):
        ad.prefill_forward(_history([img]), pixel_values_videos=[[torch.zeros(256, 4)]],
                           video_grid_thw=[[torch.tensor([[2, 8, 16]])]], **one)  # fmt: skip
    ad.mm_encoder = None
    with expect_error(NotImplementedError, "without an image encoder"):
        ad.prefill_forward(_history([img]), **{k: [v[1]] for k, v in kw.items()}, **one)
    ad.mm_encoder = encoder
    backend.mm_extra_rows = 0
    with expect_error(NotImplementedError, "GLM53_STACK_MM=1"):
        ad.prefill_forward(_history([img]), **{k: [v[1]] for k, v in kw.items()}, **one)
    text_ad, *_ = _adapter()
    with expect_error(NotImplementedError, "text-only"):
        text_ad.prefill_forward(_history([img]), **{k: [v[1]] for k, v in kw.items()}, **one)
    assert "ttnn" not in sys.modules and "vllm" not in sys.modules


# the HF reference prompt of demo/sample_prompts/shapes_hello.png + the question (doc/vision/multimodal.md: 286 tokens,
# the 256 image tokens at rows 11..266, </think> appended)
DEMO_IMAGE = Path(__file__).resolve().parents[1] / "demo" / "sample_prompts" / "shapes_hello.png"
DEMO_QUESTION = "What shapes and what word are in this image? Answer in one sentence."
DEMO_HEAD = [154822, 154824, 154826, 25062, 287, 29905, 371, 25, 7487, 154827, 154830]
DEMO_TAIL = [154831, 3838, 20770, 323, 1128, 3409, 525, 304, 419, 2168, 30, 21754, 304, 825, 11646, 13, 154828, 154841,
             154842]  # fmt: skip
_PROCESSOR_PROBE = r"""
import json, os, sys
from pathlib import Path
from models.demos.glm53_flash.tt import generator_vllm_stack as gvs  # before vLLM: the module's fabric probe stays off
import vllm  # noqa: F401
from PIL import Image
from transformers import AutoTokenizer
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.processing import ProcessorInputs
from vllm.multimodal.processing.context import TimingContext

snap, image, question = sys.argv[1:4]
api = gvs._vllm_multimodal_api()
cls = gvs.declare_multimodal_class(api)  # what the module declares when the plugin imports it inside vLLM
mc = ModelConfig(model=snap, tokenizer=snap, dtype="bfloat16", max_model_len=4096)
MULTIMODAL_REGISTRY._get_model_cls = lambda model_config: cls  # the plugin's override resolves the arch to the adapter
proc = MULTIMODAL_REGISTRY.create_processor(mc)
tk = AutoTokenizer.from_pretrained(snap)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
text = tk.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False) + "</think>"
img = Image.open(image).convert("RGB")
out = proc.apply(ProcessorInputs(tk.encode(text, add_special_tokens=False), proc.info.parse_mm_data({"image": [img]})),
                 TimingContext(enabled=False))
data = out["mm_kwargs"].get_data()
ph = out["mm_placeholders"]["image"][0]
print("PROBE " + json.dumps(dict(module=api.processor_module, mm=bool(cls.supports_multimodal),
      bases=[c.__name__ for c in cls.__mro__], processor=type(proc).__name__, ids=list(out["prompt_token_ids"]),
      grid=data["image_grid_thw"].reshape(-1).tolist(), pixels=list(data["pixel_values"].shape),
      placeholder=[ph.offset, ph.length], ttnn="ttnn" in sys.modules)))
"""


def test_registered_processor_encodes_the_demo_image():
    """The processor GLM53StackForConditionalGeneration registers in a real vLLM (0.31: vllm.models.glm5next.common;
    0.30: .nvidia) on the demo image + question: the HF reference's 286 ids (256 image tokens at rows 11..266), grid
    (1, 32, 32), 1024 patches, placeholder [11, 256) -- the plugin's mm_placeholder_offsets / _lengths. Runs in a
    subprocess of GLM53_VLLM_PYTHON (a python with vLLM; VLLM_PLUGINS="" so no TT platform plugin imports ttnn) over
    the checkpoint (MODEL_WEIGHTS_DIR / the HF cache); skipped without them."""
    from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir

    py = os.environ.get("GLM53_VLLM_PYTHON", "")
    if not py or not Path(py).exists():
        pytest.skip("GLM53_VLLM_PYTHON (a python with vLLM) not set")
    try:
        snap = default_snapshot_dir()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no GLM-5.3-Flash checkpoint ({exc})")
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, VLLM_PLUGINS="", PYTHONPATH=str(root), OMP_NUM_THREADS="8")
    res = subprocess.run([py, "-c", _PROCESSOR_PROBE, str(snap), str(DEMO_IMAGE), DEMO_QUESTION], env=env, cwd=root,
                         capture_output=True, text=True, timeout=600)  # fmt: skip
    line = [ln for ln in res.stdout.splitlines() if ln.startswith("PROBE ")]
    assert res.returncode == 0 and line, f"processor probe failed ({res.returncode}):\n{res.stderr[-4000:]}"
    got = json.loads(line[-1][len("PROBE ") :])
    assert got["module"] in gvs.GLM5NEXT_PROCESSOR_MODULES and got["mm"] is True
    assert "SupportsMultiModal" in got["bases"] and got["processor"] == "Glm5NextMultiModalProcessor"
    assert got["ids"] == DEMO_HEAD + [IMAGE_TOKEN_ID] * 256 + DEMO_TAIL
    assert got["grid"] == [1, 32, 32] and got["pixels"] == [1024, 1176] and got["placeholder"] == [11, 256]
    assert got["ttnn"] is False


# ------------------------------------------------------------------------------- the device tower (GLM53_STACK_MM_ENCODER)
def test_device_tower_l1_plan_fits_beside_the_stack():
    """The stack's device tower (DeviceVisionEncoder's TtVisionTower kwargs) at every patch bucket of a tower call
    (VisionRuntime: 1024 .. 4096 patches + the 32-row pad window): every linear has an explicit 2D-multicast plan
    within the 512 KiB per-core budget on the 8x4's 12 x 10 grid (and 13 / 11 x 10) whose blocks satisfy the matmul
    factory's divisibility rules and core counts, ending below the stack's CB end ruling; the SDPA with the 32-row Q
    chunk ends below it too, while the default tower's 64-row chunk would cross the stack's binding L1 floor."""
    from models.demos.glm53_flash.tt import memory_plan as MP

    assert MP.STACK_CB_BASE == 291_968 and MP.STACK_CB_END == 1_330_000
    budget = 512 * 1024
    for gx, gy in ((12, 10), (13, 10), (11, 10)):
        for patches in (1024, 2048, 3072, 4096):
            rows = patches + 32
            for name, m, k, n, in0, in1, out, bias in MP.tower_matmul_shapes(rows):
                p = MP.tower_mm_plan(gx, gy, m, k, n, in0=in0, in1=in1, out=out, bias=bias, budget=budget)
                pm, pn, oh, ow, bw = (p[f] for f in ("per_core_M", "per_core_N", "out_block_h", "out_block_w",
                                                     "in0_block_w"))  # fmt: skip
                sh, sw = p["out_subblock_h"], p["out_subblock_w"]
                assert pm % oh == 0 and pn % ow == 0 and oh % sh == 0 and ow % sw == 0 and sh * sw <= 4, (name, p)
                assert (
                    k % bw == 0 and -(-m // pm) <= gy and -(-n // pn) <= gx and p["cores"] == (-(-n // pn), -(-m // pm))
                )
                assert p["l1_bytes"] == MP.tower_mm_bytes(oh, ow, bw, in0, in1, out, bias) <= budget, (name, p)
                assert MP.STACK_CB_BASE + p["l1_bytes"] < MP.STACK_CB_END
            sdpa = MP.tower_sdpa_cb_bytes(32, 1024, rows, cores=gx * gy)
            assert MP.STACK_CB_BASE + sdpa < MP.STACK_CB_END, (rows, sdpa)
    assert MP.tower_sdpa_cb_bytes(32, 1024, 1056) == 833_568  # the 448 x 448 call
    assert MP.STACK_CB_BASE + MP.tower_sdpa_cb_bytes(64, 1024, 1056) > 1_334_720  # the validated default-L1 chunk
    with expect_error(ValueError, "no tower matmul plan"):  # the merger's gate | up as ONE weight does not fit
        MP.tower_mm_plan(12, 10, 9, 128, 640, in0=4096, in1=2048, out=4096, bias=0, budget=budget)


def test_mm_encoder_selection():
    """GLM53_STACK_MM_ENCODER: device (the default with a backend) registers the tower's build + warm-up as the
    backend's pre-capture hook and refuses a call before it ran; hf_cpu is the default without one; none disables
    image input; another value is refused. The device encoder's tower kwargs follow the GLM53_STACK_MM_TOWER_* knobs."""
    backend = SimpleNamespace(mesh=object(), pre_capture_hooks=[])
    enc = gvs.make_mm_encoder(Path("/nonexistent"), env={}, backend=backend)
    assert isinstance(enc, gvs.DeviceVisionEncoder) and backend.pre_capture_hooks == [enc.attach]
    assert enc.tower_kw == dict(mm_l1_budget=512 * 1024, sdpa_chunks=(32, 1024), split_swiglu=True)
    assert enc.max_patches == 4096 and enc.fence and enc.seal and enc.runtime is None
    with expect_error(RuntimeError, "pre-capture hook"):
        enc([])
    env = {"GLM53_STACK_MM_TOWER_MM_KB": "384", "GLM53_STACK_MM_TOWER_SDPA": "32,512", "GLM53_STACK_MM_TOWER_PATCHES": "1024",
           "GLM53_STACK_MM_TOWER_FENCE": "0", "GLM53_STACK_MM_TOWER_SEAL": "0"}  # fmt: skip
    enc2 = gvs.make_mm_encoder(Path("/nonexistent"), env=dict(env, GLM53_STACK_MM_ENCODER="device"), backend=backend)
    assert enc2.tower_kw["mm_l1_budget"] == 384 * 1024 and enc2.tower_kw["sdpa_chunks"] == (32, 512)
    assert enc2.max_patches == 1024 and not enc2.fence and not enc2.seal and len(backend.pre_capture_hooks) == 2
    assert gvs.make_mm_encoder(Path("/nonexistent"), env={"GLM53_STACK_MM_ENCODER": "none"}, backend=backend) is None
    with expect_error(ValueError, "needs the stack backend"):
        gvs.make_mm_encoder(Path("/nonexistent"), env={"GLM53_STACK_MM_ENCODER": "device"})
    with expect_error(ValueError, "device \\| hf_cpu \\| none"):
        gvs.make_mm_encoder(Path("/nonexistent"), env={"GLM53_STACK_MM_ENCODER": "gpu"}, backend=backend)
    assert "ttnn" not in sys.modules
