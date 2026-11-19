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
0.30 processor module, the plugin's per-row image items routed to the generator through a fake encoder: the feature cache (keyed by content, bounded in bytes: identical images of 16 requests encode once), an off-grid continuation re-run from 0, a failed image request ending on a stop token beside served rows, image requests kept out of the prefix cache; the device tower's L1 plan and encoder selection) and, with a
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
import threading
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

from models.autoports.zai_org_glm_5_3_flash.tt.generator import (
    BLOCK_SIZE,
    SEED_SKIP,
    DecodeIO,
    DecodeOutput,
    Glm53Generator,
    ModelBackend,
    PendingTokens,
    blocks_per_seq,
    mtp_fill_inputs,
    mtp_prefill_span,
    spec_mtp_pt_width,
    spec_mtp_sparse_refusals,
    spec_multi_alt,
    spec_multi_groups,
    spec_multi_k_by_rows,
    spec_multi_plan,
    spec_partial,
    spec_pick_step,
    spec_sampled_verify,
    spec_sampler_pages,
    spec_step_caps,
    spec_verify_masks,
    spec_verify_row_state,
    spec_widths,
    xoshiro_state,
)
from models.autoports.zai_org_glm_5_3_flash.tt.owner_blocks import OwnerBlockMap
from models.autoports.zai_org_glm_5_3_flash.tt.sampling import SamplingParams
from models.demos.glm53_flash.tt import generator_vllm_stack as gvs
from models.demos.glm53_flash.tt.generator_vllm_stack import (
    LATENT_DIM,
    GLM53StackForCausalLM,
    cache_defaults,
    declare_multimodal_class,
    mtp_fill_ids,
    row_params,
    spec_multi_gate,
    spec_plan_for,
    trim_page_rows,
)
from models.demos.glm53_flash.tt.mm_inputs import IMAGE_TOKEN_ID, MMItem
from models.demos.glm53_flash.tt.vllm_spec import PLACEHOLDER_TOKEN_ID, accept_greedy_walk, contract_types

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
        st.extend(
            t if t < VOCAB or t - VOCAB not in self.mm_table else ("img", int(self.mm_table[t - VOCAB][0]))
            for t in tokens[:n_valid].tolist()
        )  # fmt: skip  (a raw image token: a failed image request's text run)
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
    """The plugin's per-row lists -> SamplingParams: greedy rows, vLLM's unset top-k (0 / -1), None seeds; top_p 0.0
    (the plugin's sampling suite) is a greedy row and top_p > 1 is 1.0 -- never an exception inside the engine."""
    sp = SimpleNamespace(
        temperature=[0.0, 0.7, 0.9, 1.0, 1.0, 0.8],
        top_k=[1, 1, 0, 40, 0, 0],
        top_p=[1.0, 0.9, 0.95, 1.0, 0.0, 1.5],
        seed=[None, 3, -1, None, 7, None],
    )
    assert row_params(sp, 0) == SamplingParams(top_k=1) and row_params(sp, 0).is_greedy
    assert (
        row_params(sp, 1) == SamplingParams(temperature=0.7, top_k=1, top_p=0.9, seed=3) and row_params(sp, 1).is_greedy
    )
    assert row_params(sp, 2) == SamplingParams(temperature=0.9, top_k=None, top_p=0.95, seed=None)
    assert row_params(sp, 3) == SamplingParams(temperature=1.0, top_k=40, top_p=1.0, seed=None)
    assert (
        row_params(sp, 4) == SamplingParams(temperature=0.0, top_k=None, top_p=1.0, seed=7)
        and row_params(sp, 4).is_greedy
    )
    assert row_params(sp, 5) == SamplingParams(temperature=0.8, top_k=None, top_p=1.0, seed=None)
    with expect_error(AttributeError, "temperature"):  # a missing field is a contract break, not a greedy row
        row_params(SimpleNamespace(), 0)
    with expect_error(IndexError, "out of range"):  # a short list likewise
        row_params(sp, 6)


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


def _prefix_adapter(monkeypatch, backend_cls=None):
    """the adapter under GLM53_STACK_PREFIX=1 (read at import: the module flag) on the prefix-capable mock"""
    monkeypatch.setattr(gvs, "PREFIX_CACHE", True)
    backend = (backend_cls or _MockPrefixBackend)(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.prefix is not None and gen.shared_pages and gen.dsa_shard_map(kv) is not None
    assert ad.prefix.grid == 128 and ad.prefix.store.device_entries == backend.snapshot_entries
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
    ad.release_request(3)
    # ---- above the head (GLM53_STACK_PREFIX_HEAD; 128 here) a request keeps only its latest snapshot: D's 256 is
    #      evicted when its 384 commits, the 384 when its 512 does; the 128 (head) stays; the next turn of D (E)
    #      attaches at 512
    pc.head = 128
    d = _prompt(512, 61) + _prompt(12, 70)
    row_d = _blocks(3, len(d))
    _prefill_steps(ad, kv, d, row_d, slot=3, ends=[128, 256, 384, 512, len(d)])
    assert pc.store.key(d, 256) not in pc.store.entries and pc.store.key(d, 384) not in pc.store.entries
    assert pc.store.key(d, 512) in pc.store.entries and pc.stats["tail_superseded"] == 2
    ad.release_request(3)
    assert not pc.tail  # the slot released: its chain ends
    e = d + _prompt(20, 80)
    assert ad.confirm_prefix_cache_hit(e, [], 512) == 512
    # ---- GLM53_STACK_PREFIX_EXACT (grid 512 here, min gain 128): F's last chunk [512, 700) ends off the grid and off
    #      the 128 boundary -> split at 640: [512, 640) head-less with the snapshot at 640 (superseding F's 512 tail),
    #      then [640, 700) sampled; the next turn (vLLM proposes its full 64-token blocks, 640) attaches at 640
    backend.prefill_chunk_tokens, pc.exact, pc.exact_min = 512, True, 128
    f = _prompt(700, 91)
    row_f = [1 + 3 * BPS + 12 + j for j in range(-(-len(f) // BLOCK_SIZE))]  # unused blocks of request 3's pool
    n = len(backend.state_io)
    first_f = _prefill_steps(ad, kv, f, row_f, slot=3, ends=[512, len(f)])
    g = ad._alias[3]
    assert [r[0:2] for r in backend.state_io[n:]] == [(g, 0), (g, 512), (g, 640)]
    assert [r[3] is not None for r in backend.state_io[n:]] == [True, True, False]  # snapshots at 512 and 640
    assert kv.state[g] == f and first_f == (f[-1] + 1) % VOCAB and pc.stats["exact_splits"] == 1
    assert pc.store.key(f, 640) in pc.store.entries and pc.store.key(f, 512) not in pc.store.entries
    ad.release_request(3)
    assert ad.confirm_prefix_cache_hit(f + _prompt(30, 95), [], 640) == 640
    # below the minimum gain (a 600-token prompt: 512 -> its boundary 512 already; 640 > 600) nothing splits
    assert pc.exact_splits([512], [600]) == {} and pc.exact_splits([512], [700]) == {0: 640}
    pc.exact_min = 256
    assert pc.exact_splits([512], [700]) == {}  # 640 gains 128 < 256 over the grid snapshot at 512
    # F44 (GLM53_STACK_PREFIX_EXACT_UNCUT, default on): a last chunk the scheduler leaves uncut across the grid
    # ([384, 700) over 512) has no snapshot at 512 -- the gain is measured over its start (640 - 384 = 256: split);
    # the pre-F44 rule (over 512: 128 < 256) left that turn without any snapshot
    assert pc.exact_splits([384], [700]) == {0: 640} and pc.stats["exact_uncut_splits"] == 1
    pc.exact_uncut = False
    assert pc.exact_splits([384], [700]) == {} and pc.stats["exact_uncut_splits"] == 1
    pc.exact_uncut = True
    # ... end to end: G's chunks [0, 384) (snapshot at 384) and the uncut [384, 700) -> split at 640, snapshot there;
    #     G's next turn resumes at 640 (pre-F44: at 384)
    gp = _prompt(700, 37)
    row_g = [1 + 16 + j for j in range(-(-len(gp) // BLOCK_SIZE))]  # unused blocks of request 0's pool
    n = len(backend.state_io)
    first_g = _prefill_steps(ad, kv, gp, row_g, slot=3, ends=[384, len(gp)])
    g = ad._alias[3]
    assert [r[1] for r in backend.state_io[n:]] == [0, 384, 640]
    assert [r[3] is not None for r in backend.state_io[n:]] == [True, True, False]  # snapshots at 384 and 640
    assert kv.state[g] == gp and first_g == (gp[-1] + 1) % VOCAB and pc.stats["exact_uncut_splits"] == 2
    ad.release_request(3)
    assert ad.confirm_prefix_cache_hit(gp + _prompt(30, 95), [], 640) == 640
    # ---- one call, two rows (an exact-split inner call): a new request's fresh column on a full owner chip and a
    #      confirmed hit listing that chip's least recently released cached block. The hit's row maps first (its
    #      blocks referenced), so the fresh column reclaims another cached block; in the call's order the fresh row
    #      evicted the hit's block and the hit's row raised (served AgentX c32 stopped the engine on it)
    o2 = OwnerBlockMap(2, 3, 1)  # 2 chips x 2 usable blocks (local 0 is the null block), column j on chip j % 2
    o2.set_row(0, [10, 11], 2)
    o2.release_slot(0)  # 10 (chip 0) and 11 (chip 1) cached: 10 is chip 0's least recently released
    o2.set_row(2, [12], 1)
    o2.release_slot(2)  # chip 0 is full of cached blocks (10, 12)
    null4 = torch.tensor([o2.null(j) for j in range(4)], dtype=torch.int64)
    bnd = SimpleNamespace(obm=o2, max_batch=4, null_row=null4, phys=null4.expand(4, 4).clone(),
                          slots=[SimpleNamespace(active=False) for _ in range(4)])  # fmt: skip
    tbl = torch.full((4, 4), -1, dtype=torch.int64)
    tbl[1, 0] = 20  # slot 1: a new request, column 0 on chip 0
    tbl[3, :2] = torch.tensor([10, 11])  # slot 3: a hit on slot 0's cached blocks (2 computed columns)
    _phys, done, err = Glm53Generator._physical_rows(bnd, tbl, torch.tensor([0, 1, 0, 2]), [1, 3], keep={3: 2})
    assert err is None and done == [3, 1]
    assert o2.physical(10) is not None and o2.physical(11) is not None and o2.physical(12) is None

    # ---- two keep > 0 rows (a prefill passes keep = start for every row): the first row's NEW column on a full owner
    #      chip must not reclaim a cached block of the later row's computed columns -- every row's computed columns are
    #      pinned for the whole call, so the fresh column reclaims another cached block (in the call's order slot 1's
    #      column 2 evicted 10 and slot 3 raised "no device content")
    def two_hits(nb_local: int, spare: bool):
        o = OwnerBlockMap(2, nb_local, 1)
        o.set_row(0, [10, 11], 2)
        o.release_slot(0)  # 10 (chip 0) and 11 (chip 1) cached: 10 is chip 0's least recently released
        if spare:
            o.set_row(2, [12], 1)
            o.release_slot(2)  # a cached block of no row of the call (chip 0)
        o.set_row(1, [13], 1)  # chip 0 full: 10, (12,) 13
        bd = SimpleNamespace(obm=o, max_batch=4, null_row=null4, phys=null4.expand(4, 4).clone(),
                             slots=[SimpleNamespace(active=False) for _ in range(4)])  # fmt: skip
        tb = torch.full((4, 4), -1, dtype=torch.int64)
        tb[1, :3] = torch.tensor([13, 21, 22])  # slot 1: its live request (1 computed column) + 2 new columns
        tb[3, :2] = torch.tensor([10, 11])  # slot 3: a hit on slot 0's cached blocks (2 computed columns)
        cv = torch.tensor([0, 3, 0, 2])
        return o, Glm53Generator._physical_rows(bd, tb, cv, [1, 3], keep={1: 1, 3: 2})

    o, (ph, done, err) = two_hits(4, spare=True)  # 3 blocks per chip: chip 0 holds 13, 22 and the hit's 10
    assert err is None and done == [1, 3], err
    assert o.physical(12) is None and o.evictions == 1, "the fresh column reclaims the unrelated cached block"
    assert ph[3, :2].tolist() == [o.physical(10), o.physical(11)] and o.rows[3] == [10, 11]
    o, (ph, done, err) = two_hits(3, spare=False)  # 2 blocks per chip: no room for 22 -- slot 1 is refused, the hit's
    assert isinstance(err, RuntimeError) and "no free physical block" in str(err) and done == []  # blocks stay cached
    assert o.physical(10) is not None and o.physical(11) is not None and o.evictions == 0 and o.rows[1] == [13]


class _MockHostTierBackend(_MockPrefixBackend):
    """+ the host tier's device side: export_snapshot / import_snapshot of a pool entry (virtual slot max_batch + k) as
    a token-list tensor blob, recorded"""

    def __init__(self, max_batch: int):
        super().__init__(max_batch)
        self.exports: list = []
        self.imports: list = []

    def ensure_state_export(self, kv) -> bool:
        return True  # (StackBackend: the KDA pack + DSA row staging, allocated before the first capture)

    def export_snapshot(self, kv, entry: int, n_tokens=None):
        self.exports.append((int(entry), n_tokens))
        return {"state": torch.tensor(list(kv.state[self.max_batch + int(entry)]), dtype=torch.int64)}

    def import_snapshot(self, kv, entry: int, states) -> None:
        self.imports.append(int(entry))
        kv.state[self.max_batch + int(entry)] = [int(t) for t in states["state"].tolist()]


def test_prefix_cache_host_tier_spill_restore_and_dominated(monkeypatch):
    """GLM53_PREFIX_HOST_BYTES: a full pool spills its least recently used entry to the host tier instead of dropping
    it; a hit on a host entry (its pages resident) is confirmed, imported into a free device entry before the prefill
    and attached -- the same first token as a cold prefill; above the head a conversation turn's snapshot becomes
    DOMINATED once its next turn snapshots past it and is then the first victim, dropped (never spilled)."""
    monkeypatch.setattr(gvs, "PREFIX_CACHE", True)
    monkeypatch.setenv("GLM53_PREFIX_HOST_BYTES", str(1 << 30))
    monkeypatch.setenv("GLM53_PREFIX_SPILL_MIN", "0")
    backend = _MockHostTierBackend(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    pc = ad.prefix
    assert pc.host and pc.store.host_bytes == 1 << 30 and pc.store.spill_fn is not None
    doc = _prompt(256, 5)
    a, b = doc + _prompt(40, 50), doc + _prompt(30, 20)
    row_a = _blocks(0, len(a))
    _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, len(a)])
    ad.release_request(0)
    key128, key256 = pc.store.key(a, 128), pc.store.key(a, 256)
    # ---- C (384 tokens of another document) needs three entries: A's 128 and 256 (LRU) are spilled, not dropped
    c = _prompt(384, 9) + _prompt(10, 33)
    _prefill_steps(ad, kv, c, _blocks(1, len(c)), slot=1, ends=[128, 256, 384, len(c)])
    ad.release_request(1)
    assert pc.store.entries[key128].tier == "host" and pc.store.entries[key256].tier == "host"
    assert [x[0] for x in backend.exports] == [0, 1] and pc.stats["spills"] == 2
    assert pc.store.host_used == sum(e.bytes for e in pc.store.entries.values() if e.tier == "host") > 0
    # ---- B: vLLM proposes A's 4 cached blocks; the host entry at 256 is confirmed (its pages are resident) ...
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256 and pc.stats["confirmed_host"] == 1
    # ... and restored into a device entry (C's least recently used 128 spills for it) before B's prefill attaches it
    row_b = row_a[:4] + _blocks(2, len(b))[4:]
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[2]
    e256 = pc.store.entries[key256]
    assert e256.tier == "device" and backend.imports == [e256.device_entry] == [2]
    assert backend.state_io[-1] == (g, 256, 2, None)  # the suffix only, from the restored entry
    assert kv.state[g] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB
    assert pc.stats["restores"] == 1 and pc.stats["attaches"] == 1 and pc.stats["fallbacks"] == 0
    assert pc.store.entries[pc.store.key(c, 128)].tier == "host" and pc.stats["spills"] == 3
    ad.release_request(2)
    # ---- dominated: above the head (128 here) a conversation turn's snapshot is superseded by its next turn's
    pc.head = 128
    d1 = _prompt(300, 61)
    row_d = _blocks(3, 520)
    _prefill_steps(ad, kv, d1, row_d, slot=3, ends=[128, 256, len(d1)])
    ad.release_request(3)
    kd256 = pc.store.key(d1, 256)
    assert kd256 in pc.store.entries and not pc.store.entries[kd256].dominated
    d2 = d1 + _prompt(150, 70)  # the next turn: 450 tokens, resumes at 256, snapshots at 384
    assert ad.confirm_prefix_cache_hit(d2, [], 256) == 256
    start = 256
    for end in (384, len(d2)):
        ad.prefill_forward(_history([d2[:end]]), page_table=_table([row_d[: -(-end // BLOCK_SIZE)]], 1), kv_cache=kv,
                           prompt_lens=[end], start_pos=[start], empty_slots=[0],
                           sampling_params=_greedy(1) if end == len(d2) else None)  # fmt: skip
        start = end
    assert pc.store.entries[kd256].dominated and pc.stats["dominated"] == 1
    assert pc.store.key(d2, 384) in pc.store.entries and pc.stats["attaches"] == 2
    ad.release_request(0)
    # the dominated 256 is the first victim -- dropped, no export -- ahead of the older D 128
    n_exp = len(backend.exports)
    ent = pc.store.reserve_device()
    assert ent is not None and kd256 not in pc.store.entries and len(backend.exports) == n_exp
    assert pc.store.stats["dropped_dominated"] == 1 and pc.store.stats["evicted_dominated"] == 1
    assert pc.store.key(d1, 128) in pc.store.entries
    pc.store.unreserve(ent)
    rep = ad.prefix.report()
    assert rep["host_used"] > 0 and rep["spilled"] == pc.stats["spills"] and rep["restores"] == 1
    # ---- GLM53_STACK_PREFIX_KEEP_PREV (F44): E's next turn supersedes the 256 it attached -> KEPT (not dominated);
    #      a turn that rewrites E2's tail (shares 300 < 384 tokens) resumes at the kept 256 instead of the head 128
    pc.keep_prev = True
    e1 = _prompt(300, 101)
    row_e = [1 + 3 * BPS + 12 + j for j in range(8)]  # unused blocks of request 3's pool
    _prefill_steps(ad, kv, e1, row_e, slot=3, ends=[128, 256, len(e1)])
    ad.release_request(3)
    ke256 = pc.store.key(e1, 256)
    e2 = e1 + _prompt(150, 7)
    assert ad.confirm_prefix_cache_hit(e2, [], 256) == 256
    start = 256
    for end in (384, len(e2)):
        ad.prefill_forward(_history([e2[:end]]), page_table=_table([row_e[: -(-end // BLOCK_SIZE)]], 1), kv_cache=kv,
                           prompt_lens=[end], start_pos=[start], empty_slots=[3],
                           sampling_params=_greedy(1) if end == len(e2) else None)  # fmt: skip
        start = end
    ad.release_request(3)
    e256 = pc.store.entries[ke256]
    assert e256.kept_prev and not e256.dominated and pc.stats["keep_prev_kept"] == 1
    assert pc.prev_of == {pc.store.key(e2, 384): ke256}
    e3 = e2[:300] + _prompt(150, 111)  # the rewritten tail: vLLM proposes its 4 shared full blocks (256)
    row_e3 = row_e[:4] + [1 + 3 * BPS + 20 + j for j in range(4)]
    assert ad.confirm_prefix_cache_hit(e3, [], 256) == 256
    n = len(backend.state_io)
    start = 256
    for end in (384, len(e3)):
        out = ad.prefill_forward(_history([e3[:end]]), page_table=_table([row_e3[: -(-end // BLOCK_SIZE)]], 1),
                                 kv_cache=kv, prompt_lens=[end], start_pos=[start], empty_slots=[3],
                                 sampling_params=_greedy(1) if end == len(e3) else None)  # fmt: skip
        start = end
    g = ad._alias[3]
    assert backend.state_io[n][1] == 256 and backend.state_io[n][2] is not None  # attached the kept 256
    assert kv.state[g] == e3 and int(out[0][0, 0]) == (e3[-1] + 1) % VOCAB and pc.stats["keep_prev_hits"] == 1
    ad.release_request(3)
    # one kept snapshot per chain: E3's 384 keeps the 256 again; E2's 384 (the abandoned tail) stays an LRU entry
    assert pc.prev_of[pc.store.key(e3, 384)] == ke256 and pc.stats["keep_prev_kept"] == 2
    # the device victim order: unread dominated, then unread kept, then LRU; a kept one below keep_prev_min is
    # dropped (no export), from it on spilled
    victim = pc.store._device_victim()
    assert victim.dominated or victim.kept_prev
    pc.store.keep_prev_min = 1 << 30
    n_exp, held = len(backend.exports), []
    while ke256 in pc.store.entries and pc.store.entries[ke256].tier == "device":
        held.append(pc.store.reserve_device())
        assert held[-1] is not None
    assert ke256 not in pc.store.entries and pc.store.stats["dropped_kept_prev"] == 1
    assert len(backend.exports) == n_exp  # (dominated victims are dropped too: no export)
    for ent in held:
        pc.store.unreserve(ent)


def test_prefix_cache_no_host_tier_without_the_env(monkeypatch):
    """GLM53_PREFIX_HOST_BYTES unset (or a backend without export / import): no host tier -- the pool drops its LRU
    entry as before"""
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    assert not ad.prefix.host and ad.prefix.store.host_bytes == 0 and ad.prefix.store.spill_fn is None
    monkeypatch.setenv("GLM53_PREFIX_HOST_BYTES", str(1 << 30))
    pc2 = gvs.StackPrefixCache(gen, kv, B)  # the plain prefix mock cannot stage an export (ensure_state_export)
    assert not pc2.host and pc2.store.spill_fn is None


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


STOP = VOCAB - 1  # the mock tokenizer's stop token (a failed image request ends on it)


def _mm_adapter(encoder=None, backend=None):
    """the multimodal class over the CPU mock (plain declaration: no vLLM here), warmed, with a recording encoder"""
    cls = declare_multimodal_class(None)
    backend = backend or _MockStackBackend(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=(STOP,)), max_batch=B, max_seq_len=SEQ)
    ad = cls(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B)
    encoded = []

    def default_encoder(items):  # row j of an item = its j-th feature row; the first value names it (offset + j)
        encoded.append([it.n_tokens for it in items])
        return [_img_rows(torch.arange(it.offset, it.offset + it.n_tokens)) for it in items]

    ad.mm_encoder = encoder or default_encoder
    return ad, backend, gen, kv, encoded


def _items(lo: int, n_tokens: int, w: int = 16, ident=None, fill: float = 0.0) -> dict:
    """the plugin's per-row lists of ONE row holding one image item of n_tokens tokens at [lo, lo + n_tokens), its
    pixels all ``fill`` (the feature cache's content key tells images apart by them)"""
    h = n_tokens * 4 // w
    out = dict(pixel_values=[[torch.full((h * w, 4), float(fill))]], image_grid_thw=[[torch.tensor([[1, h, w]])]],
               mm_placeholder_offsets=[[lo]], mm_placeholder_lengths=[[n_tokens]])  # fmt: skip
    if ident is not None:
        out["mm_identifiers"] = [[ident]]
    return out


def test_multimodal_adapter_routes_image_items_to_the_generator(monkeypatch):
    """The plugin's per-row image items (pixel_values / image_grid_thw lists + placeholder ranges) on the multimodal
    class: the item's rows found in the row's prefix, the item encoded ONCE (the second chunk reads the feature cache),
    the chunk's rows handed to the generator -- an image straddling the two scheduler chunks of a 200-token prompt is
    staged by both; text rows of the same batch and a text-only batch (pixel_values=[None]) take the text path; a
    video, a missing encoder and a backend without extra rows fail THAT request (the head returns the stop token, the
    slot stays consistent); the text class still refuses pixels."""
    ad, backend, gen, kv, encoded = _mm_adapter()
    lo, n_img = 112, 32  # grid (1, 8, 16): 128 patches -> 32 image tokens at [112, 144)
    img = _mm_prompt(200, lo, lo + n_img, 9)
    text = _prompt(90, 4)
    grid = torch.tensor([[1, 8, 16]])
    pv = torch.zeros(128, 4)
    kw = dict(pixel_values=[None, [pv]], image_grid_thw=[None, [grid]], mm_placeholder_offsets=[None, [lo]],
              mm_placeholder_lengths=[None, [n_img]])  # fmt: skip
    # step 1: the text prompt whole + the image prompt's first 128 tokens (host-routed intermediate chunk)
    first = ad.prefill_forward(_history([text, img[:128]]), page_table=_table([_blocks(0, 90), _blocks(1, 128)], 2),
                               kv_cache=kv, prompt_lens=[90, 128], start_pos=[0, 0], empty_slots=[0, 1],
                               sampling_params=None, **kw)  # fmt: skip
    assert first.shape == (2, 1, VOCAB) and encoded == [[32]]
    assert [k for k, _ in backend.mm_stages] == [1] and backend.mm_stages[0][1][:, 0].tolist() == list(range(112, 128))
    assert kv.state[0] == text and kv.state[1] == _ledger(img[:128], lo, lo + n_img)
    # step 2: the rest of the image prompt; the plugin sends the item again: its features come from the cache
    out = ad.prefill_forward(_history([img]), page_table=_table([_blocks(1, 200)], 1), kv_cache=kv, prompt_lens=[200],
                             start_pos=[128], empty_slots=[1], sampling_params=_greedy(1),
                             **{k: [v[1]] for k, v in kw.items()})  # fmt: skip
    assert int(out[0][0, 0]) == (img[-1] + 1) % VOCAB and encoded == [[32]]
    assert ad.mm_stats["cache_hits"] == 1 and ad.mm_stats["items_encoded"] == 1 and ad.mm_stats["rows_staged"] == 32
    assert backend.mm_stages[-1][1][:, 0].tolist() == list(range(128, 144))
    assert kv.state[1] == _ledger(img, lo, lo + n_img)
    # a text-only batch on the multimodal class: the text path (no encoder call, no stage)
    ad.prefill_forward(_history([text]), page_table=_table([_blocks(2, 90)], 1), kv_cache=kv, prompt_lens=[90],
                       start_pos=[0], empty_slots=[2], sampling_params=_greedy(1), pixel_values=[None],
                       image_grid_thw=[None], mm_placeholder_offsets=[None], mm_placeholder_lengths=[None])  # fmt: skip
    assert len(encoded) == 1 and len(backend.mm_stages) == 2
    # failures: the request ends on the stop token, its chunk ran as text (raw image token ids), nothing raised
    one = dict(page_table=_table([_blocks(3, 200)], 1), kv_cache=kv, prompt_lens=[200], start_pos=[0], empty_slots=[3],
               sampling_params=_greedy(1))  # fmt: skip
    video = dict(pixel_values_videos=[[torch.zeros(256, 4)]], video_grid_thw=[[torch.tensor([[2, 8, 16]])]])
    for setup, args in (
        (lambda: None, video),
        (lambda: setattr(ad, "mm_encoder", None), {k: [v[1]] for k, v in kw.items()}),
        (lambda: setattr(backend, "mm_extra_rows", 0), {k: [v[1]] for k, v in kw.items()}),
    ):
        setup()
        ids, _ = ad.prefill_forward(_history([img]), **args, **one)
        assert int(ids[0, 0]) == STOP and kv.state[ad._alias[3]][lo] == IMAGE_TOKEN_ID
        ad.release_request(3)
    assert ad.mm_stats["failed"] == 3 and ad._mm_failed == set()
    text_ad, *_ = _adapter()
    with expect_error(NotImplementedError, "text-only"):
        text_ad.prefill_forward(_history([img]), **{k: [v[1]] for k, v in kw.items()}, **one)
    assert "ttnn" not in sys.modules and "vllm" not in sys.modules


def test_off_grid_continuation_inside_an_image_reruns_from_zero():
    """vLLM may end an image prompt's chunk anywhere (the plugin aligns text prompts only): a continuation off the
    backend's 64-token grid -- here [96, 260) after [0, 96), the cut inside the image item [80, 112) -- re-runs [0, 260)
    from position 0 in the request's own slot, the image's rows from the feature cache (one encoder call in all);
    the slot's state is the whole prompt with every image row, the head is the prompt's next token."""
    ad, backend, gen, kv, encoded = _mm_adapter()
    lo, n_img = 80, 32
    img = _mm_prompt(260, lo, lo + n_img, 5)
    kw = _items(lo, n_img)
    ad.prefill_forward(_history([img[:96]]), page_table=_table([_blocks(0, 96)], 1), kv_cache=kv, prompt_lens=[96],
                       start_pos=[0], empty_slots=[0], sampling_params=None, **kw)  # fmt: skip
    g = ad._alias[0]
    assert backend.chunks == [(g, 0, 96, "last", False)] and kv.state[g] == _ledger(img[:96], lo, lo + n_img)
    out = ad.prefill_forward(_history([img]), page_table=_table([_blocks(0, 260)], 1), kv_cache=kv, prompt_lens=[260],
                             start_pos=[96], empty_slots=[0], sampling_params=_greedy(1), **kw)  # fmt: skip
    assert backend.chunks[1:] == [(g, 0, 128, "none", False), (g, 128, 128, "none", False), (g, 256, 4, "last", True)]
    assert kv.state[g] == _ledger(img, lo, lo + n_img) and int(out[0][0, 0]) == (img[-1] + 1) % VOCAB
    assert encoded == [[32]] and ad.mm_stats["restarts"] == 1 and ad.mm_stats["cache_hits"] == 1
    assert gen.live_slots(kv) == [g]


def test_failed_image_request_does_not_stop_the_batch_or_the_engine():
    """An encoder error on one row of a batch: that request's head is the stop token (host route: the logits row
    carries it) while the text row beside it is served; its next chunk fails too without another encoder call (the
    failure is kept per generator slot) and release_request clears it -- a fresh image request in the same plugin
    slot is served. Identical images of two requests (vLLM's identifier) are encoded once."""
    calls = []

    def flaky(items):
        calls.append(items[0].grid_thw.reshape(-1).tolist())
        if int(items[0].grid_thw.reshape(-1)[1]) == 4:  # the 4 x 16 grid: the "tower" fails
            raise RuntimeError("tower down")
        return [_img_rows(torch.arange(it.offset, it.offset + it.n_tokens)) for it in items]

    ad, backend, gen, kv, _ = _mm_adapter(flaky)
    bad = _mm_prompt(200, 10, 26, 3)  # 16 image tokens: grid (1, 4, 16)
    text = _prompt(90, 4)
    kw = {k: [v[0], None] for k, v in _items(10, 16).items()}
    out = ad.prefill_forward(_history([bad[:128], text]), page_table=_table([_blocks(0, 128), _blocks(1, 90)], 2),
                             kv_cache=kv, prompt_lens=[128, 90], start_pos=[0, 0], empty_slots=[0, 1],
                             sampling_params=None, **kw)  # fmt: skip
    assert int(out[0, 0].argmax()) == STOP and int(out[1, 0].argmax()) == (text[-1] + 1) % VOCAB
    g = ad._alias[0]
    assert ad._mm_failed == {g} and calls == [[1, 4, 16]] and kv.state[ad._alias[1]] == text
    ids, _ = ad.prefill_forward(_history([bad]), page_table=_table([_blocks(0, 200)], 1), kv_cache=kv,
                                prompt_lens=[200], start_pos=[128], empty_slots=[0], sampling_params=_greedy(1),
                                **_items(10, 16))  # fmt: skip
    assert int(ids[0, 0]) == STOP and calls == [[1, 4, 16]] and len(kv.state[g]) == 200
    ad.release_request(0)
    assert ad._mm_failed == set()
    good = _mm_prompt(120, 20, 52, 7)  # 32 image tokens: grid (1, 8, 16)
    for slot, req in ((0, 0), (2, 2)):  # two requests, the same image (one identifier): one encoder call
        ids, _ = ad.prefill_forward(_history([good]), page_table=_table([_blocks(req, 120)], 1), kv_cache=kv,
                                    prompt_lens=[120], start_pos=[0], empty_slots=[slot], sampling_params=_greedy(1),
                                    **_items(20, 32, ident="img-hash-1"))  # fmt: skip
        assert int(ids[0, 0]) == (good[-1] + 1) % VOCAB and kv.state[ad._alias[slot]] == _ledger(good, 20, 52)
    assert calls == [[1, 4, 16], [1, 8, 16]] and ad.mm_stats["cache_hits"] == 1 and ad.mm_stats["failed"] == 1


def test_identical_images_encode_once_across_requests(monkeypatch):
    """The feature cache is keyed by content (``mm_content_key``: the processed pixels + grid): 16 requests showing the
    same image under 16 identifiers (vLLM's per-request uuids without its processor cache) encode it ONCE; a request's
    later chunk finds the item by its identifier alone (an alias of the key: no re-hash); other pixels encode anew."""
    ad, backend, gen, kv, encoded = _mm_adapter()
    hashed = []
    real = gvs.mm_content_key
    monkeypatch.setattr(gvs, "mm_content_key", lambda it: hashed.append(it.offset) or real(it))
    lo, n_img = 112, 32  # straddles the 128-token chunk end
    img = _mm_prompt(200, lo, lo + n_img, 7)
    for k in range(16):
        slot = k % B
        ids, _ = ad.prefill_forward(_history([img]), page_table=_table([_blocks(slot, 200)], 1), kv_cache=kv,
                                    prompt_lens=[200], start_pos=[0], empty_slots=[slot], sampling_params=_greedy(1),
                                    **_items(lo, n_img, ident=f"req{k}-mm-0", fill=0.5))  # fmt: skip
        assert int(ids[0, 0]) == (img[-1] + 1) % VOCAB and kv.state[ad._alias[slot]] == _ledger(img, lo, lo + n_img)
        ad.release_request(slot)
    assert encoded == [[32]] and len(hashed) == 16
    assert ad.mm_stats["items_encoded"] == 1 and ad.mm_stats["content_hits"] == 15 and ad.mm_stats["cache_hits"] == 15
    # a 17th request in two engine steps: its first chunk hashes the item (a content hit), the second reads the alias
    kw = _items(lo, n_img, ident="req16-mm-0", fill=0.5)
    ad.prefill_forward(_history([img[:128]]), page_table=_table([_blocks(0, 128)], 1), kv_cache=kv, prompt_lens=[128],
                       start_pos=[0], empty_slots=[0], sampling_params=None, **kw)  # fmt: skip
    ids, _ = ad.prefill_forward(_history([img]), page_table=_table([_blocks(0, 200)], 1), kv_cache=kv,
                                prompt_lens=[200], start_pos=[128], empty_slots=[0], sampling_params=_greedy(1), **kw)  # fmt: skip
    assert kv.state[ad._alias[0]] == _ledger(img, lo, lo + n_img) and int(ids[0, 0]) == (img[-1] + 1) % VOCAB
    assert encoded == [[32]] and len(hashed) == 17
    assert ad.mm_stats["content_hits"] == 16 and ad.mm_stats["cache_hits"] == 17
    ad.release_request(0)
    # other pixels on the same grid: another image
    ad.prefill_forward(_history([img]), page_table=_table([_blocks(1, 200)], 1), kv_cache=kv, prompt_lens=[200],
                       start_pos=[0], empty_slots=[1], sampling_params=_greedy(1),
                       **_items(lo, n_img, ident="req17-mm-0", fill=0.25))  # fmt: skip
    assert encoded == [[32], [32]] and ad.mm_stats["items_encoded"] == 2 and len(ad._feat_cache) == 2


def test_feature_cache_is_bounded_in_bytes(monkeypatch):
    """GLM53_STACK_MM_CACHE_MB bounds the feature cache: past it the least recently used item goes (a later request
    showing it encodes again, its identifier's alias only misses), the newest stays even when it alone is larger than
    the bound; a negative bound is refused."""
    item_bytes = 32 * MM_WIDTH * 2  # the mock's [32, MM_WIDTH] bf16 rows
    monkeypatch.setenv(gvs.MM_CACHE_ENV, str(2.5 * item_bytes / 2**20))  # two items, not three
    ad, *_, encoded = _mm_adapter()

    def item(fill: float) -> MMItem:
        return MMItem("image", torch.full((128, 4), fill), torch.tensor([[1, 8, 16]]), offset=0, length=32)

    a, b, c = item(1.0), item(2.0), item(3.0)
    for it, ident in ((a, "a"), (b, "b"), (a, "a"), (c, "c")):  # a read again before c arrives: b is the oldest
        ad._mm_features(it, ident)
    assert len(encoded) == 3 and list(ad._feat_cache) == [gvs.mm_content_key(x) for x in (a, c)]
    assert ad._feat_bytes == 2 * item_bytes and ad.mm_stats["evicted"] == 1 and ad.mm_stats["cache_hits"] == 1
    ad._mm_features(item(2.0), "b")  # b's alias names an evicted key: encoded anew, a (now the oldest) goes
    assert len(encoded) == 4 and list(ad._feat_cache) == [gvs.mm_content_key(x) for x in (c, b)]
    assert ad.mm_stats["evicted"] == 2 and ad._feat_bytes == 2 * item_bytes
    monkeypatch.setenv(gvs.MM_CACHE_ENV, "0")
    tiny, *_, tiny_encoded = _mm_adapter()
    for _ in range(2):
        tiny._mm_features(a, None)
    assert len(tiny_encoded) == 1 and len(tiny._feat_cache) == 1 and tiny.mm_stats["content_hits"] == 1
    monkeypatch.setenv(gvs.MM_CACHE_ENV, "-1")
    with expect_error(ValueError, "must be >= 0 MB"):
        _mm_adapter()


class _MockMMPrefixBackend(_MockPrefixBackend):
    """the prefix mock whose pages hold what the slot's state recorded (an image row: ("img", position))"""

    @staticmethod
    def _page_write(kv, slot: int, start: int, toks):
        st = kv.state[slot][start : start + len(toks)]
        row, pages = kv.page_table_host[slot], kv.__dict__.setdefault("pages", {})
        for k, t in enumerate(st if len(st) == len(toks) else toks):  # (decode writes before its state grows)
            pages[(int(row[(start + k) // BLOCK_SIZE]), (start + k) % BLOCK_SIZE)] = (
                t if isinstance(t, tuple) else int(t)
            )


def test_image_requests_take_no_prefix_snapshot_or_hit(monkeypatch):
    """GLM53_STACK_PREFIX on the multimodal class: the store's key hashes token ids, which carry no pixels (every image
    row is the same <|image|> id), so an image request neither snapshots its on-grid chunk ends nor attaches -- confirm
    refuses its mm items, and a new image request vLLM admitted at start > 0 all the same (a text document's snapshot
    matches its first 128 tokens) re-runs from 0 with its image staged; a text request on the same adapter snapshots,
    is confirmed and attaches as on the text class."""
    monkeypatch.setattr(gvs, "PREFIX_CACHE", True)
    ad, backend, gen, kv, encoded = _mm_adapter(backend=_MockMMPrefixBackend(B))
    pc = ad.prefix
    assert pc is not None and pc.grid == 128 and gen.shared_pages
    none = dict(
        pixel_values=[None], image_grid_thw=[None], mm_placeholder_offsets=[None], mm_placeholder_lengths=[None]
    )

    def steps(prompt, slot, ends, row, **kw):
        start = 0
        for end in ends:
            out = ad.prefill_forward(_history([prompt[:end]]), page_table=_table([row[: -(-end // BLOCK_SIZE)]], 1),
                                     kv_cache=kv, prompt_lens=[end], start_pos=[start], empty_slots=[slot],
                                     sampling_params=_greedy(1) if end == len(prompt) else None, **kw)  # fmt: skip
            start = end
        return out

    # an image prompt in three engine steps: its chunk ends 128 / 256 are on the grid, neither is snapshotted
    lo, n_img = 80, 32
    img = _mm_prompt(300, lo, lo + n_img, 9)
    n0 = len(backend.state_io)
    steps(img, 0, [128, 256, 300], _blocks(0, 300), **_items(lo, n_img, ident="img-a"))
    assert [io[3] for io in backend.state_io[n0:]] == [None] * 3 and kv.state[ad._alias[0]] == _ledger(img, lo, lo + 32)
    assert pc.stats["snapshots"] == 0 and pc.stats["mm_excluded"] == 2 and not pc.store.entries and encoded == [[32]]
    ad.release_request(0)
    assert ad.confirm_prefix_cache_hit(img, [("img-a", lo, n_img, "image")], 256) == 0
    # a text document on the same adapter: a snapshot at 128, confirmed and attached for another question
    doc = _prompt(128, 5)
    t1, t2 = doc + _prompt(40, 50), doc + _prompt(30, 20)
    row1 = _blocks(1, len(t1))
    steps(t1, 1, [128, len(t1)], row1, **none)
    assert pc.stats["snapshots"] == 1 and [e.n_tokens for e in pc.store.entries.values()] == [128]
    e128 = pc.store.entries[pc.store.key(t1, 128)]
    ad.release_request(1)
    assert ad.confirm_prefix_cache_hit(t2, [], 128) == 128
    out = ad.prefill_forward(_history([t2]), page_table=_table([row1[:2] + _blocks(2, len(t2))[2:]], 1), kv_cache=kv,
                             prompt_lens=[len(t2)], start_pos=[128], empty_slots=[2], sampling_params=_greedy(1),
                             **none)  # fmt: skip
    g2 = ad._alias[2]
    assert backend.state_io[-1] == (g2, 128, e128.device_entry, None) and kv.state[g2] == t2
    assert int(out[0][0, 0]) == (t2[-1] + 1) % VOCAB and pc.stats["attaches"] == 1
    # an image request behind the same document: its first 128 tokens match the text snapshot, but the hit is refused
    i2 = doc + _mm_prompt(64, 8, 40, 7)  # the image at [136, 168) (positions below 256: exact in the mock's bf16 rows)
    assert ad.confirm_prefix_cache_hit(i2, [("img-b", 136, 32, "image")], 128) == 0
    # ... and admitted at 128 all the same (in plugin slot 3): re-run from 0, the image staged, no attach, no snapshot
    n0 = len(backend.state_io)
    ids, _ = ad.prefill_forward(_history([i2]), page_table=_table([row1[:2] + _blocks(3, len(i2))[2:]], 1),
                                kv_cache=kv, prompt_lens=[len(i2)], start_pos=[128], empty_slots=[3],
                                sampling_params=_greedy(1), **_items(136, 32, ident="img-b", fill=1.0))  # fmt: skip
    g3 = ad._alias[3]
    assert [io[1:] for io in backend.state_io[n0:]] == [(0, None, None), (128, None, None)]
    assert kv.state[g3] == _ledger(i2, 136, 168) and int(ids[0, 0]) == (i2[-1] + 1) % VOCAB
    assert encoded == [[32], [32]] and ad.mm_stats["restarts"] == 1
    assert pc.stats["attaches"] == 1 and pc.stats["snapshots"] == 1 and pc.stats["mm_excluded"] == 2


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


# ------------------------------------------------------------------------------------------ Stage 17 M0: the spec loop
# Glm53Generator.generate_batch(spec_k=K) over a CPU stand-in of the backend's speculative step (tt/spec_mtp.py SpecMTP's
# host contract: set_page_table / enter / step / read / leave + StackBackend.arm_mtp_prefill / disarm_mtp_prefill).
class _MockSpec:
    """The speculative step of the lone user in slot 0 on the mock: the bootstrap drafts K tokens after the next input;
    a step "verifies" [bonus, d1..dK] with the mock model (a_j = (row j + 1) % V, kernels/spec/spec_ctl.cpp ACCEPT: n =
    the longest prefix with a_(j-1) == d_j), the slot's state ledger takes the committed rows b, d1..dn, the bonus a_n
    goes to token row 0 and the next K drafts follow -- right for the first good(step) of them, then wrong."""

    def __init__(self, be, K: int, good):
        self.be, self.K, self.good = be, K, good
        self.pos_cap, self.mtp, self.mw_cu = None, object(), K + 1
        self.counters = dict(enters=0, steps=0, leaves=0)
        self.rows, self._rec, self._drafts = None, None, []

    def _make_drafts(self, bonus: int, step: int) -> list:
        out, t = [], int(bonus)
        for i in range(self.K):
            t = (t + 1) % VOCAB
            out.append(t if i < self.good(step) else (t + 5) % VOCAB)
        return out

    def set_page_table(self, row):
        self.rows = torch.as_tensor(row).clone()

    def enter(self, row):
        io = self.be.io
        b, c = int(io.tokens.t[0]), int(io.cur_pos.t[0]) - 1  # BOOT: c = the plain position - 1
        self._drafts = self._make_drafts(b, 0)
        self._rec = dict(n=0, c=c, a=[b] * (self.K + 1), committed=[b], drafts=list(self._drafts), steps=0)
        self.counters["enters"] += 1

    def step(self, blocking: bool = False):
        io, st = self.be.io, self.be.kv.state[0]
        c = self._rec["c"] + self._rec["n"] + 1  # pos_v row 0
        rows = [int(io.tokens.t[0])] + self._drafts
        assert len(st) == c, f"verify at {c}: the slot's state holds {len(st)} tokens"
        # the step addresses positions c .. c + 2K (verify + the next drafts): mapped in the published device table (the
        # null block 0 marks the row's uncovered tail; GLM53_SPEC_PT_DEFER publishes a grown row behind the first step)
        pt = getattr(getattr(io, "page_table", None), "t", None)
        if pt is not None and pt.dim() == 2 and int(pt.shape[1]) > (c + 2 * self.K) // BLOCK_SIZE:
            assert int(pt[0, (c + 2 * self.K) // BLOCK_SIZE]) != 0, f"step at {c}: position {c + 2 * self.K} unmapped"
        a = [(r + 1) % VOCAB for r in rows]
        n = 0
        for j in range(1, self.K + 1):
            if a[j - 1] != rows[j]:
                break
            n = j
        st.extend(rows[: n + 1])
        io.tokens.t[0] = a[n]
        self.counters["steps"] += 1
        self._drafts = self._make_drafts(a[n], self.counters["steps"])
        self._rec = dict(n=n, c=c, a=a, committed=a[: n + 1], drafts=list(self._drafts), steps=self.counters["steps"])

    def read(self) -> dict:
        return dict(self._rec)

    def abort(self) -> None:
        self.counters["aborts"] = self.counters.get("aborts", 0) + 1

    def leave(self) -> dict:
        rec = self.read()
        pos = self.be.io.cur_pos.t.clone()
        pos[0] = rec["c"] + rec["n"] + 1
        self.be.io.cur_pos.t = pos
        self.counters["leaves"] += 1
        return rec


class _MockSpecBackend(_MockStackBackend):
    """the mock + a speculative step (K drafts, `good(step)` of them right) + the MTP-cache prefill bookkeeping"""

    def __init__(self, max_batch: int, K: int, good=lambda step: 0):
        super().__init__(max_batch)
        self.K, self.good = K, good
        self.spec, self.armed, self.mtp_spans = None, None, []

    def allocate_kv_cache(self, *, max_batch, max_seq_len, num_blocks=None):
        self.kv = super().allocate_kv_cache(max_batch=max_batch, max_seq_len=max_seq_len, num_blocks=num_blocks)
        return self.kv

    def allocate_decode_io(self, kv):
        self.io = super().allocate_decode_io(kv)
        self.spec = _MockSpec(self, self.K, self.good)  # (StackBackend: SpecMTP built with the decode IO)
        return self.io

    def set_page_table_host(self, kv, page_table):
        super().set_page_table_host(kv, page_table)
        if getattr(self, "io", None) is not None:  # (the generator-owned cache: StackBackend writes the device table)
            self.io.page_table.t = torch.as_tensor(page_table).clone()

    def arm_mtp_prefill(self, slot, prompt) -> bool:
        self.armed = dict(slot=int(slot), ids=torch.as_tensor(prompt).reshape(-1).clone(), rows=0)
        return True

    def disarm_mtp_prefill(self) -> int:
        a, self.armed = self.armed, None
        return int(a["rows"]) if a else 0

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None, **sio):
        out = super().prefill_chunk(tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start, n_valid=n_valid,
                                    logits_rows=logits_rows, sample=sample, io=io, **sio)  # fmt: skip
        a = self.armed
        if a is not None and a["slot"] == slot:  # (StackBackend._mtp_prefill_chunk)
            p0, n = mtp_prefill_span(chunk_start, n_valid, int(a["ids"].numel()))
            if n:
                self.mtp_spans.append((p0, n, a["ids"][p0 + 1 : p0 + 1 + n].tolist()))
                a["rows"] += n
        return out


def _spec_gen(K: int, good=lambda step: 0, seq: int = SEQ):
    backend = _MockSpecBackend(B, K, good)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=seq)
    return backend, gen


def test_spec_widths_rules(monkeypatch):
    """generator.spec_widths: the 8x4 shard defaults (GLM53_SPEC_ON_SHARD=1) run every spec launch at the B = 4 chunk
    (verify width = the lone plain width, MTP rows at it: the sparse_dsa layers need no extra fabric row count);
    elsewhere the TP8 design (verify at K + 1, catch-up K + 1, chain 1) is unchanged"""
    for k in ("GLM53_SPEC_ON_SHARD", "GLM53_SPEC_VERIFY_W", "GLM53_SPEC_MTP_W"):
        monkeypatch.delenv(k, raising=False)
    assert spec_widths(1, 4) == dict(w=2, lone=4, wv=2, cu=2, ch=1, mode="k1", ts=(1, 2))
    assert spec_widths(3, 4) == dict(w=4, lone=4, wv=4, cu=4, ch=1, mode="k1", ts=(1, 4))
    monkeypatch.setenv("GLM53_SPEC_ON_SHARD", "1")
    for K in (1, 2, 3):
        assert spec_widths(K, 4) == dict(w=K + 1, lone=4, wv=4, cu=4, ch=4, mode="verify", ts=(4,)), K
    # a native B = 8 backend's lone width 8 > 4: the MTP rows stay on the TP8 design (stacked residual only <= 4)
    assert spec_widths(1, 8, (8,)) == dict(w=2, lone=8, wv=8, cu=2, ch=1, mode="k1", ts=(1, 2))
    monkeypatch.setenv("GLM53_SPEC_VERIFY_W", "")
    assert spec_widths(1, 4)["wv"] == 2 and spec_widths(1, 4)["ts"] == (2,)
    monkeypatch.setenv("GLM53_SPEC_MTP_W", "k1")
    assert spec_widths(1, 4)["ts"] == (1, 2)
    monkeypatch.setenv("GLM53_SPEC_VERIFY_W", "3")
    with expect_error(AssertionError, "must equal the lone speculative plain width"):
        spec_widths(1, 4)
    monkeypatch.setenv("GLM53_SPEC_VERIFY_W", "plain")
    monkeypatch.setenv("GLM53_SPEC_MTP_W", "verify")
    with expect_error(AssertionError, "verify width <= 4"):
        spec_widths(1, 8, (8,))
    # GLM53_SPEC_MULTI's planner on the B 16 1 M build (narrow 1 / 2 / 4, native 8, full 16), K 3: one user = today's
    # lone narrow step (4 rows); 2 users -> the native 8; 4 users -> all 16 rows; 5..8 users -> K 1 (2 rows each: a
    # group is whole row pairs); 9+ users -> plain decode
    ws = (1, 2, 4, 8)
    assert spec_multi_plan(3, 1, ws) == dict(g=4, k=3, slots=(0,), rows=4, wv=4, cu=4, ch=1)
    assert spec_multi_plan(3, 2, ws) == dict(g=4, k=3, slots=(0, 4), rows=8, wv=8, cu=8, ch=2)
    assert spec_multi_plan(3, 4, ws) == dict(g=4, k=3, slots=(0, 4, 8, 12), rows=16, wv=16, cu=16, ch=4)
    assert spec_multi_plan(3, 5, ws) == dict(g=2, k=1, slots=(0, 2, 4, 6, 8), rows=10, wv=16, cu=16, ch=8)
    assert spec_multi_plan(2, 1, ws)["g"] == 3 and spec_multi_plan(2, 2, ws)["g"] == 2  # (one user: any K + 1)
    assert spec_multi_plan(3, 8, ws)["k"] == 1 and spec_multi_plan(3, 8, ws)["slots"] == (0, 2, 4, 6, 8, 10, 12, 14)
    assert spec_multi_plan(3, 9, ws)["k"] == 0 and spec_multi_plan(3, 16, ws)["wv"] is None
    for live in range(1, 17):  # every group fits the full width, the users sit on distinct group bases
        p = spec_multi_plan(3, live, ws)
        assert p["rows"] <= 16 and len(set(p["slots"])) == live and all(s + p["g"] <= 16 for s in p["slots"])
    assert spec_multi_k_by_rows(3, 16) == ((4, 3), (8, 1), (16, 1))
    assert spec_multi_k_by_rows(1, 16) == ((8, 1), (16, 1)) and spec_multi_k_by_rows(3, 4) == ((4, 3),)
    # batch 2's fixed-G step (spec_widths groups=G, GLM53_SPEC_MULTI_USERS): G (K + 1) rows on a build width, every MTP
    # catch-up row with it (cu = wv), the chained rows at the smallest width >= G (K 1: none -- ch = w); the knobs of the
    # single-user rules
    # are not read; the sparse_dsa layers' extra fabric row counts only <= 4 (wider runs on the full width's phases)
    monkeypatch.setenv("GLM53_SPEC_VERIFY_W", "3")
    assert spec_widths(3, 16, ws, groups=4) == dict(w=16, lone=16, wv=16, cu=16, ch=4, mode="multi", ts=(4,))
    assert spec_widths(3, 16, ws, groups=2) == dict(w=8, lone=8, wv=8, cu=8, ch=2, mode="multi", ts=(2,))
    assert spec_widths(1, 16, ws, groups=2) == dict(w=4, lone=4, wv=4, cu=4, ch=4, mode="multi", ts=(4,))
    assert spec_widths(1, 16, ws, groups=8) == dict(w=16, lone=16, wv=16, cu=16, ch=16, mode="multi", ts=())
    assert spec_widths(1, 8, (), groups=4)["ch"] == 8 and spec_widths(1, 4, (), groups=2)["wv"] == 4
    # GLM53_SPEC_MTP_PT_CAP: the multi-user private MTP tables at one user's cap blocks (16-entry rows), else full width
    assert spec_mtp_pt_width(16384, 512, 8, True, {}) == 512 and spec_mtp_pt_width(16384, 4096, 4, True, {}) == 4096
    assert spec_mtp_pt_width(16384, 1563, 8, True, {}) == 1568 and spec_mtp_pt_width(2048, 4096, 8, True, {}) == 2048
    assert spec_mtp_pt_width(16384, 512, 0, True, {}) == 16384 and spec_mtp_pt_width(16384, 512, 8, False, {}) == 16384
    assert spec_mtp_pt_width(16384, 512, 8, True, {"GLM53_SPEC_MTP_PT_CAP": "0"}) == 16384
    with expect_error(AssertionError, "must be a decode width of the build"):
        spec_widths(3, 16, ws, groups=3)
    with expect_error(AssertionError, "K \\+ 1 even"):
        spec_widths(2, 16, ws, groups=2)
    assert spec_multi_plan(3, 4, ws)["slots"] == (0, 4, 8, 12) and spec_multi_plan(3, 2, ws)["slots"] == (0, 4)
    assert spec_multi_groups({}) == 0 and spec_multi_groups({"GLM53_SPEC_MULTI_USERS": "4"}) == 0
    assert spec_multi_groups({"GLM53_SPEC_MULTI": "1"}) == 4
    assert spec_multi_groups({"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_MULTI_USERS": "2"}) == 2
    with expect_error(AssertionError, ">= 2 users"):
        spec_multi_groups({"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_MULTI_USERS": "1"})


def test_spec_multi_lone_step():
    """GLM53_SPEC_MULTI_LONE (F20): a (K, G 1) multi-user step built under GLM53_SPEC_LONE_BUILD only (G 1 stays refused
    elsewhere), K + 1 rows on a build width (narrow 4 for K 3), listed FIRST so the step pick prefers it for a lone
    user on slot 0 and falls through to the primary / alternate for anything else"""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import spec_multi_lone

    lone_env = {"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_MULTI_USERS": "1", "GLM53_SPEC_LONE_BUILD": "1"}
    assert spec_multi_groups(lone_env) == 1
    with expect_error(AssertionError, ">= 2 users"):
        spec_multi_groups({**lone_env, "GLM53_SPEC_LONE_BUILD": "0"})
    assert spec_widths(3, 16, (4,), groups=1) == dict(w=4, lone=4, wv=4, cu=4, ch=4, mode="multi", ts=(4,))
    with expect_error(AssertionError, "must be a decode width of the build"):
        spec_widths(3, 16, (), groups=1)  # (no narrow 4: the lone step has no 4-row programs)
    on = {"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_MTP_SPARSE": "1", "GLM53_SPEC_MULTI_LONE": "1"}
    assert spec_multi_lone(on) is True and spec_multi_lone({}) is False
    with expect_error(ValueError, "GLM53_SPEC_MULTI_LONE"):
        spec_multi_lone({"GLM53_SPEC_MULTI_LONE": "1", "GLM53_SPEC_MULTI": "1"})
    steps = [(3, 1, (0,)), (3, 4, (0, 4, 8, 12)), (1, 8, (0, 2, 4, 6, 8, 10, 12, 14))]
    assert spec_pick_step(steps, [0]) == (0, [0])  # the lone user on slot 0: the 4-row step
    assert spec_pick_step(steps, [4]) == (1, [4])  # a lone user elsewhere: the primary (partial)
    assert spec_pick_step(steps, [0, 4]) == (1, [0, 4]) and spec_pick_step(steps, [0, 2]) == (2, [0, 2])
    assert spec_pick_step(steps, [0], room=lambda i: i != 0) == (1, [0])  # no room on the lone step: the next one


def test_spec_admit_keeps_speculation_on_attach_with_mtp_snapshots():
    """GLM53_SPEC_SNAPSHOT_MTP (F20): a prefix attach keeps the request speculable when the backend's snapshots carry the
    MTP layer's state (StackBackend.snapshot_has_mtp); without it the attach declines (the MTP history is missing); an
    image request always declines; a fresh start clears an earlier verdict"""
    import types

    def admit(has_mtp, no_spec, starts, attach, vision):
        obj = types.SimpleNamespace(gen=types.SimpleNamespace(backend=types.SimpleNamespace(
            snapshot_has_mtp=lambda: has_mtp)), _no_spec=set(no_spec))  # fmt: skip
        gvs.GLM53StackForCausalLM._spec_admit(obj, [0, 4, 8], starts, attach, vision)
        return obj._no_spec

    assert admit(False, set(), [0, 32768, 0], {1: 3}, ()) == {4}  # attach without MTP state: declines
    assert admit(True, {4}, [0, 32768, 0], {1: 3}, ()) == set()  # attach with MTP state: speculable (verdict cleared)
    assert admit(True, set(), [0, 0, 0], {}, (2,)) == {8}  # image rows: never
    assert admit(True, {0}, [0, 0, 0], {}, ()) == set()  # a fresh start clears it
    assert admit(False, {0}, [128, 0, 0], {}, ()) == {0}  # a continuation keeps it


def test_mtp_prefill_span_covers_the_prompt():
    """the MTP-cache rows of a chunked prompt: every position 0 .. L - 2 exactly once (L - 1 is the bootstrap's)"""
    L, got = 300, []
    for s in range(0, L, 128):
        p0, n = mtp_prefill_span(s, min(128, L - s), L)
        got += list(range(p0, p0 + n))
    assert got == list(range(L - 1))
    assert mtp_prefill_span(256, 44, 300) == (256, 43) and mtp_prefill_span(0, 1, 1) == (0, 0)


def test_mtp_fill_inputs_chunks():
    """the in-trace MTP fill's per-chunk inputs (TPPrefill._mtp_write): over a 300-token prompt in 128-row chunks (the
    last one in a 128 bucket with 44 valid rows) every MTP position 0 .. L - 2 gets the prompt token after it exactly
    once, at the slot's page of that position; pages without a valid row of the chunk -- and rows past the row's
    coverage -- go to scratch pages (an unarmed or padded page never lands on a request's)"""
    L, c, blk = 300, 128, 64
    ids = torch.arange(1000, 1000 + L)
    row = torch.arange(40, 40 + 5)  # the slot's 5 pages (positions 0 .. 319)
    scratch = torch.tensor([900, 901])
    filled = {}
    for s0 in range(0, L, c):
        nv = min(c, L - s0)
        tok, pages, n = mtp_fill_inputs(s0, nv, c, blk, ids, row, scratch)
        assert n == mtp_prefill_span(s0, nv, L)[1]
        for i in range(n):
            filled[s0 + i] = (int(tok[i]), int(pages[i // blk]))
        assert bool((tok[n:] == 0).all())
    assert sorted(filled) == list(range(L - 1))
    assert all(t == 1000 + p + 1 and pg == 40 + p // blk for p, (t, pg) in filled.items())
    _, pages, _ = mtp_fill_inputs(256, 44, 128, blk, ids, row, scratch)  # positions 256..299: page 4 valid, 5 padding
    assert pages.tolist() == [44, 901]
    _, pages, n = mtp_fill_inputs(256, 44, 128, blk, ids, row[:4], scratch)  # a row short of the chunk: scratch
    assert pages.tolist() == [900, 901] and n == 43


@pytest.mark.parametrize("K", [1, 3])
@pytest.mark.parametrize("pattern", ["none", "all", "alternate"])
def test_spec_generate_equals_plain_greedy(K, pattern):
    """generate_batch(spec_k=K): the same ids as plain greedy whatever the drafts' quality (none / all / every other
    step right), the steps' bookkeeping (each commits n + 1, the slot's state ledger and the device positions agree --
    the mock asserts both), tau = committed / steps, the MTP-cache prefill rows = L - 1 over both chunks"""
    good = {"none": lambda s: 0, "all": lambda s: K, "alternate": lambda s: K if s % 2 else 1}[pattern]
    prompt, osl = _prompt(200, 3), 40
    _, plain_gen = _spec_gen(K)
    plain = plain_gen.generate_batch([prompt], osl, enable_trace=True)[0]
    backend, gen = _spec_gen(K, good)
    out = gen.generate_batch([prompt], osl, enable_trace=True, spec_k=K)[0]
    assert out == plain and len(out) == osl
    t = gen.last_batch_timing
    acc = t["spec_accepts"]
    assert t["spec_steps"] == sum(acc) == backend.spec.counters["steps"] > 0
    assert t["spec_committed"] == sum((n + 1) * c for n, c in enumerate(acc))
    assert t["spec_emitted"] == osl - 1 and t["decode_steps"] == osl - 1  # (the prefill's token is not a step's)
    assert t["spec_committed"] >= t["spec_emitted"]  # (the last step may commit past max_new_tokens)
    if pattern == "none":
        assert acc == [t["spec_steps"]] + [0] * K
    if pattern == "all":
        assert acc[K] >= t["spec_steps"] - 1 and t["spec_committed"] / t["spec_steps"] > K
    assert backend.spec.counters["enters"] == backend.spec.counters["leaves"] == 1
    assert t["spec_mtp_prefill_rows"] == len(prompt) - 1
    assert [s[:2] for s in backend.mtp_spans] == [(0, 128), (128, 71)]
    assert backend.mtp_spans[0][2] == prompt[1:129] and backend.mtp_spans[1][2] == prompt[129:200]
    # the speculation left plain decode consistent: a plain request on the same generator decodes from scratch
    assert gen.generate_batch([prompt], 8, enable_trace=True)[0] == plain[:8]


def test_spec_stop_token_inside_a_committed_block():
    """a stop token among a step's committed tokens ends the request there (the rest of the block is dropped), as in
    plain decode"""
    K, prompt = 3, _prompt(70, 10)  # (decode positions past block 0: the mock's null block)
    stop = (prompt[-1] + 7) % VOCAB  # the 7th generated token
    _, plain_gen = _spec_gen(K)
    plain = plain_gen.generate_batch([prompt], 30, stop_token_ids=[stop])[0]
    backend, gen = _spec_gen(K, lambda s: K)
    out = gen.generate_batch([prompt], 30, stop_token_ids=[stop], spec_k=K)[0]
    assert out == plain and out[-1] == stop and len(out) == 7
    assert gen.last_batch_timing["spec_emitted"] == 6 < gen.last_batch_timing["spec_committed"]


def test_spec_falls_back_to_plain_without_room():
    """near the end of the context a step no longer fits (positions up to L + 2 K): the loop leaves and plain decode
    finishes the request with the same ids"""
    K, prompt, seq, osl = 3, _prompt(70, 5), 192, 120  # the request ends at position 189: steps stop past 185
    _, plain_gen = _spec_gen(K, seq=seq)
    plain = plain_gen.generate_batch([prompt], osl, enable_trace=True)[0]
    backend, gen = _spec_gen(K, lambda s: 1, seq=seq)
    out = gen.generate_batch([prompt], osl, enable_trace=True, spec_k=K)[0]
    t = gen.last_batch_timing
    assert out == plain and len(out) == osl
    assert 0 < t["spec_emitted"] < osl - 1 and backend.decodes > 0  # the tail decoded plainly
    assert backend.spec.counters["leaves"] == 1


def test_spec_refusals():
    """generate_batch(spec_k=...) serves one greedy text prompt in slot 0 on a backend built with the same K"""
    backend, gen = _spec_gen(3)
    with expect_error(ValueError, "the backend's speculative step with K = 2"):
        gen.generate_batch([_prompt(10, 1)], 4, spec_k=2)
    with expect_error(ValueError, "one prompt in slot 0"):
        gen.generate_batch([_prompt(10, 1), _prompt(10, 2)], 4, spec_k=3)
    with expect_error(ValueError, "greedy sampling"):
        gen.generate_batch([_prompt(10, 1)], 4, spec_k=3, sampling=[SamplingParams(temperature=1.0, seed=1)])
    img_pos = torch.arange(2, 6)
    with expect_error(ValueError, "a text prompt"):  # (the MTP-cache prefill embeds the placeholder ids)
        gen.generate_batch([_mm_prompt(10, 2, 6, 1)], 4, spec_k=3, mm=[(img_pos, _img_rows(img_pos))])
    plain_gen = Glm53Generator(
        _MockStackBackend(B), tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ
    )  # (no GLM53_SPEC_K build: no speculative step)
    with expect_error(ValueError, "the backend's speculative step with K = 3"):
        plain_gen.generate_batch([_prompt(10, 1)], 4, spec_k=3)
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- GLM53_SPEC_MULTI: the multi-user step (batch 2)
class _MockSpecMulti:
    """tt/spec_mtp.py SpecMTP with groups = G (the fixed-G step, doc/perf/mtp_multi_user_design.md section 6) on the
    mock: user u lives in slot u (K + 1) (its group base); the bootstrap / step / leave of _MockSpec per group (spec_ctl
    SPEC_GROUPS: c_u from the base row's position, the bonus to the base token row, one record per user, group order),
    the drafts right for the first good(step + u) of them; set_page_tables takes the users' rows (group order)."""

    def __init__(self, be, K: int, G: int, good):
        self.be, self.K, self.groups, self.good = be, K, G, good
        self.slots = tuple(u * (K + 1) for u in range(G))
        self.pos_cap, self.mtp, self.mw_cu = None, object(), G * (K + 1)
        self.counters = dict(enters=0, steps=0, leaves=0)
        self.rows, self._recs, self._drafts = None, None, {}

    def group_of(self, slot):
        s = int(slot)
        return s // (self.K + 1) if s % (self.K + 1) == 0 and s // (self.K + 1) < self.groups else None

    def _make_drafts(self, bonus: int, step: int, u: int) -> list:
        out, t = [], int(bonus)
        for i in range(self.K):
            t = (t + 1) % VOCAB
            out.append(t if i < self.good(step + u) else (t + 5) % VOCAB)
        return out

    def set_page_tables(self, rows):
        assert len(rows) == self.groups
        self.rows = [torch.as_tensor(r).clone() for r in rows]

    def enter(self, rows):
        self.set_page_tables(rows)
        io, self._recs = self.be.io, []
        for u, s in enumerate(self.slots):
            b, c = int(io.tokens.t[s]), int(io.cur_pos.t[s]) - 1  # BOOT: c_u = the base row's plain position - 1
            self._drafts[s] = self._make_drafts(b, 0, u)
            self._recs.append(
                dict(n=0, c=c, a=[b] * (self.K + 1), committed=[b], drafts=list(self._drafts[s]), steps=0)
            )
        self.counters["enters"] += 1

    def step(self, blocking: bool = False):
        io, recs = self.be.io, []
        self.counters["steps"] += 1
        for u, s in enumerate(self.slots):
            prev, st = self._recs[u], self.be.kv.state[s]
            c = prev["c"] + prev["n"] + 1
            rows = [int(io.tokens.t[s])] + self._drafts[s]
            assert len(st) == c, f"user {u}: verify at {c}, its state holds {len(st)} tokens"
            a = [(r + 1) % VOCAB for r in rows]
            n = 0
            for j in range(1, self.K + 1):
                if a[j - 1] != rows[j]:
                    break
                n = j
            st.extend(rows[: n + 1])
            io.tokens.t[s] = a[n]
            self._drafts[s] = self._make_drafts(a[n], self.counters["steps"], u)
            recs.append(dict(n=n, c=c, a=a, committed=a[: n + 1], drafts=list(self._drafts[s]),
                             steps=self.counters["steps"]))  # fmt: skip
        self._recs = recs

    def read(self) -> list:
        return [dict(r) for r in self._recs]

    def abort(self) -> None:
        self.counters["aborts"] = self.counters.get("aborts", 0) + 1

    def leave(self) -> list:
        recs = self.read()
        pos = self.be.io.cur_pos.t.clone()
        for s, rec in zip(self.slots, recs):
            pos[s] = rec["c"] + rec["n"] + 1
        self.be.io.cur_pos.t = pos
        self.counters["leaves"] += 1
        return recs


class _MockSpecSwitch:
    """tt/spec_mtp.py SpecSwitch on the mock (GLM53_SPEC_MULTI_ALT): the primary and the alternate multi-user steps, one
    current; every attribute the current step's, group_of any step's"""

    def __init__(self, steps):
        self.__dict__.update(steps=list(steps), cur=0)

    def __getattr__(self, name):
        return getattr(self.__dict__["steps"][self.__dict__["cur"]], name)

    def __setattr__(self, name, value):
        setattr(self.__dict__["steps"][self.__dict__["cur"]], name, value)

    def select(self, i: int) -> None:
        self.__dict__["cur"] = int(i)

    def group_of(self, slot):
        return next((u for u in (x.group_of(slot) for x in self.steps) if u is not None), None)


class _MockSpecMultiBackend(_MockSpecBackend):
    """the spec mock with the multi-user step (G users) and one armed MTP-cache prefill per group base"""

    def __init__(self, max_batch: int, K: int, G: int, good=lambda step: 0):
        super().__init__(max_batch, K, good)
        self.G = G

    def allocate_kv_cache(self, *, max_batch, max_seq_len, num_blocks=None):  # (every slot's full identity row)
        nb = num_blocks or max_batch * blocks_per_seq(max_seq_len)
        return super().allocate_kv_cache(max_batch=max_batch, max_seq_len=max_seq_len, num_blocks=nb)

    def allocate_decode_io(self, kv):
        self.io = _MockStackBackend.allocate_decode_io(self, kv)
        self.spec = _MockSpecMulti(self, self.K, self.G, self.good)
        return self.io

    def arm_mtp_prefill(self, slot, prompt) -> bool:  # (StackBackend: GLM53_SPEC_MULTI arms every group base)
        if self.spec.group_of(slot) is None:
            return False
        self.armed = dict(self.armed or {})
        self.armed[int(slot)] = dict(slot=int(slot), ids=torch.as_tensor(prompt).reshape(-1).clone(), rows=0)
        return True

    def disarm_mtp_prefill(self) -> int:
        a, self.armed = self.armed, None
        return sum(int(r["rows"]) for r in a.values()) if a else 0

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None, **sio):
        out = _MockStackBackend.prefill_chunk(self, tokens, kv_cache=kv_cache, slot=slot, chunk_start=chunk_start,
                                              n_valid=n_valid, logits_rows=logits_rows, sample=sample, io=io, **sio)  # fmt: skip
        a = (self.armed or {}).get(int(slot))
        if a is not None:
            p0, n = mtp_prefill_span(chunk_start, n_valid, int(a["ids"].numel()))
            if n:
                self.mtp_spans.append((int(slot), p0, n))
                a["rows"] += n
        return out


class _SampledVerifyPages:
    """the sampled verify head (GLM53_SPEC_SAMPLED) on a spec mock: set_sampling records the verify rows' pages as
    tt/spec_mtp.py SpecMTP.set_sampling builds them (spec_sampler_pages) from the params rows the generator last wrote
    (k, p x 1000, temperature x 1000) and the backend's per-slot RNG rows"""

    samples = True

    def set_sampling(self, Ls: dict) -> None:
        io, n = self.be.io, self.be.max_batch
        par = torch.zeros(32, 4, dtype=torch.int64)
        par[:n, 0] = io.k.t.reshape(-1)[:n].to(torch.int64)
        par[:n, 1] = (io.p.t.reshape(-1)[:n] * 1000).round().to(torch.int64)
        par[:n, 2] = (io.temp.t.reshape(-1)[:n] * 1000).round().to(torch.int64)
        groups, slots = int(getattr(self, "groups", 0)), tuple(getattr(self, "slots", (0,)))
        pages = spec_sampler_pages(self.K, groups, slots, par, self.be.rng_rows(), Ls)
        self.__dict__.setdefault("pages", []).append((*pages, dict(Ls)))


class _SampledRng:
    """the device sampler's RNG page on a mock backend: a request boundary's seeds row re-seeds each slot's row with
    xoshiro_state(seed) (SEED_UNSEEDED: a fresh state per slot; SEED_SKIP keeps the row)"""

    def rng_rows(self) -> torch.Tensor:
        return self.__dict__.setdefault("_rng", torch.zeros(32, 4, dtype=torch.int64))

    def write(self, buf, value):
        super().write(buf, value)
        io = getattr(self, "io", None)
        if io is not None and buf is io.seeds:
            for s, v in enumerate(torch.as_tensor(value).reshape(-1).tolist()):
                if v != SEED_SKIP:
                    self.rng_rows()[s] = torch.tensor(xoshiro_state(v if v >= 0 else (1 << 40) + s))


class _MockSpecSampled(_SampledVerifyPages, _MockSpec):
    pass


class _MockSpecMultiSampled(_SampledVerifyPages, _MockSpecMulti):
    pass


class _MockSpecSampledBackend(_SampledRng, _MockSpecBackend):
    def allocate_decode_io(self, kv):
        io = super().allocate_decode_io(kv)
        self.spec = _MockSpecSampled(self, self.K, self.good)
        return io


class _MockSpecMultiSampledBackend(_SampledRng, _MockSpecMultiBackend):
    def allocate_decode_io(self, kv):
        io = super().allocate_decode_io(kv)
        self.spec = _MockSpecMultiSampled(self, self.K, self.G, self.good)
        return io


def _spec_multi_gen(K: int, G: int, mb: int, good=lambda step: 0, seq: int = SEQ):
    backend = _MockSpecMultiBackend(mb, K, G, good)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=mb, max_seq_len=seq)
    return backend, gen


@pytest.mark.parametrize("K,G,mb", [(1, 2, 4), (3, 2, 8), (1, 4, 8)])
@pytest.mark.parametrize("pattern", ["none", "all", "alternate"])
def test_spec_multi_generate_equals_plain_greedy(K, G, mb, pattern):
    """GLM53_SPEC_MULTI: generate_batch(spec_k=K) of G prompts on the backend's G-user step places them on the group
    bases u (K + 1), arms each prefill for its user's MTP cache, and every user's ids equal plain greedy whatever the
    drafts' quality; per-user stats (tau per user); a user reaching its token budget first ends the speculation for
    all (v1) and the plain loop decodes the rest; plain decode afterwards is consistent"""
    good = {"none": lambda s: 0, "all": lambda s: K, "alternate": lambda s: K if s % 2 else 1}[pattern]
    prompts = [_prompt(200 - 30 * u, 3 + 7 * u) for u in range(G)]
    mnt = [40 - 9 * u for u in range(G)]  # (the last user's budget ends first)
    plain_gen = Glm53Generator(
        _MockSpecBackend(mb, K), tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=mb, max_seq_len=SEQ
    )  # fmt: skip  (the spec mock: its decode IO holds the page table)
    plain = plain_gen.generate_batch(prompts, mnt, enable_trace=True)
    backend, gen = _spec_multi_gen(K, G, mb, good)
    out = gen.generate_batch(prompts, mnt, enable_trace=True, spec_k=K)  # (the step exists after the warm-up)
    assert gen.spec_groups() == G and gen.spec_slots() == tuple(u * (K + 1) for u in range(G))
    assert out == plain and [len(o) for o in out] == mnt
    assert sorted({c[0] for c in backend.chunks}) == list(gen.spec_slots())  # the users sat on the group bases
    t = gen.last_batch_timing
    assert t["spec_users"] == G and t["spec_steps"] == backend.spec.counters["steps"] > 0
    assert sum(t["spec_accepts"]) == G * t["spec_steps"] and t["spec_committed"] == sum(t["spec_committed_u"])
    assert all(sum(a) == t["spec_steps"] for a in t["spec_accepts_u"])
    assert t["spec_emitted"] == sum(t["spec_emitted_u"]) and t["spec_emitted_u"][-1] == mnt[-1] - 1
    if pattern == "none":
        assert t["spec_accepts"] == [G * t["spec_steps"]] + [0] * K
    if pattern == "all":
        assert all(c / t["spec_steps"] > K for c in t["spec_committed_u"])
    assert backend.spec.counters["enters"] == backend.spec.counters["leaves"] == 1
    assert t["spec_mtp_prefill_rows"] == sum(len(p) - 1 for p in prompts)  # every user's MTP cache filled
    assert sorted({s for s, *_ in backend.mtp_spans}) == list(gen.spec_slots())
    # the speculation left plain decode consistent: plain requests on the same generator decode from scratch
    assert gen.generate_batch(prompts, 8, enable_trace=True) == [p[:8] for p in plain]


def test_spec_multi_refusals_and_room(monkeypatch):
    """GLM53_SPEC_MULTI's generate_batch(spec_k=K): exactly G greedy prompts on the group bases; near the context end a
    step no longer fits for some user: the speculation ends, the plain loop finishes every user with the same ids"""
    K, G, mb = 1, 2, 4
    backend, gen = _spec_multi_gen(K, G, mb, lambda s: 1)
    with expect_error(ValueError, r"2 prompts on the group bases \[0, 2\]"):
        gen.generate_batch([_prompt(10, 1)], 4, spec_k=K)
    with expect_error(ValueError, r"2 prompts on the group bases \[0, 2\]"):
        gen.generate_batch([_prompt(10, 1), _prompt(10, 2)], 4, spec_k=K, slots=[0, 1])
    with expect_error(ValueError, "greedy sampling"):
        gen.generate_batch([_prompt(10, 1), _prompt(10, 2)], 4, spec_k=K,
                           sampling=[SamplingParams(temperature=1.0, seed=1)] * 2)  # fmt: skip
    backend, gen = _spec_multi_gen(K, G, mb, lambda s: 1)
    _, plain_gen = _spec_gen(K)
    prompts, osl = [_prompt(1928, 5), _prompt(1900, 11)], 120  # user 0 ends at position 2047: its steps stop past 2045
    plain = plain_gen.generate_batch(prompts, osl, enable_trace=True)
    out = gen.generate_batch(prompts, osl, enable_trace=True, spec_k=K)
    t = gen.last_batch_timing
    assert out == plain and [len(o) for o in out] == [osl, osl]
    assert 0 < t["spec_emitted_u"][0] < osl - 1 and backend.decodes > 0 and backend.spec.counters["leaves"] == 1
    # GLM53_SPEC_MTP_SPARSE v2: multi-user at any K (the chained rows' slot stride), only G (K + 1) <= 16 refused
    sh = dict(shard=True, paged=False, window=False, prefill="trace")
    assert spec_mtp_sparse_refusals(3, 4, "mtp", **sh) == [] and spec_mtp_sparse_refusals(1, 8, "mtp", **sh) == []
    assert any("> 16 rows" in w for w in spec_mtp_sparse_refusals(3, 8, "mtp", **sh))
    # GLM53_SPEC_MULTI_ALT: the alternate step "G:K" next to the primary; the step pick by where the live users sit
    e = {"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_MTP_SPARSE": "1", "GLM53_SPEC_MULTI_USERS": "4", "GLM53_SPEC_K": "3"}
    assert spec_multi_alt(e) is None and spec_multi_alt({**e, "GLM53_SPEC_MULTI_ALT": "8:1"}) == (8, 1)
    for bad in ("2:3", "4:3", "8:2", "8", "16:1"):  # (narrower, the primary itself, K + 1 odd, no K, > 16 rows)
        with expect_error(ValueError, "GLM53_SPEC_MULTI_ALT"):
            spec_multi_alt({**e, "GLM53_SPEC_MULTI_ALT": bad})
    with expect_error(ValueError, "GLM53_SPEC_MTP_SPARSE=1"):
        spec_multi_alt({**e, "GLM53_SPEC_MTP_SPARSE": "0", "GLM53_SPEC_MULTI_ALT": "8:1"})
    steps = [(3, 4, (0, 4, 8, 12)), (1, 8, tuple(range(0, 16, 2)))]
    assert spec_pick_step(steps, [4, 0]) == (0, [0, 4]) and spec_pick_step(steps, [0, 2]) == (1, [0, 2])
    assert spec_pick_step(steps, [0, 4, 8, 12, 2, 6]) == (1, [0, 2, 4, 6, 8, 12])
    assert spec_pick_step(steps, [1]) is None and spec_pick_step(steps, []) is None
    assert spec_pick_step(steps, [0, 4], room=lambda i: i == 1) == (1, [0, 4])  # (no room on the primary: K 3)
    assert spec_pick_step(steps, [0, 4], partial=False) is None and spec_pick_step(
        steps[:1], [0, 4, 8, 12], partial=False
    ) == (0, [0, 4, 8, 12])
    # the served step's gate (spec_multi_gate): v1 = exactly the G bases; partial groups = any non-empty subset of them;
    # every live slot speculable and allowed, every next step fitting; a live slot off the bases declines for everyone
    sp = {0: True, 2: True}
    assert spec_multi_gate([2, 0], [0, 2], sp, set(), True) == [0, 2]
    assert spec_multi_gate([0], [0, 2], sp, set(), True) is None
    assert spec_multi_gate([0], [0, 2], sp, set(), True, partial=True) == [0]
    assert spec_multi_gate([0, 1], [0, 2], {**sp, 1: True}, set(), True, partial=True) is None
    assert spec_multi_gate([0, 2], [0, 2], {0: True, 2: False}, set(), True) is None
    assert spec_multi_gate([0, 2], [0, 2], sp, {2}, True) is None
    assert spec_multi_gate([0, 2], [0, 2], sp, set(), False) is None
    assert spec_multi_gate([], [0, 2], sp, set(), True, partial=True) is None
    # GLM53_SPEC_THROUGH_PREFILL: an idle live slot (a request mid-prefill) on another base stays out and the others
    # speculate (its own verdicts unread); v1 declines with one; an idle slot off the bases (a candidate row) declines;
    # idle slots alone leave nobody to speculate
    assert spec_multi_gate([0, 2, 4], [0, 2, 4, 6], sp, set(), True, partial=True, idle=[4]) == [0, 2]
    assert spec_multi_gate([0, 2], [0, 2], {0: True}, {2}, True, partial=True, idle=[2]) == [0]
    assert spec_multi_gate([0, 2], [0, 2], sp, set(), True, idle=[2]) is None
    assert spec_multi_gate([0, 1], [0, 2], sp, set(), True, partial=True, idle=[1]) is None
    assert spec_multi_gate([2], [0, 2], sp, set(), True, partial=True, idle=[2]) is None
    assert spec_multi_gate([0, 2], [0, 2], sp, set(), True, partial=True, idle=[6]) == [0, 2]  # (not live: ignored)
    # GLM53_SPEC_SAMPLED over a greedy verify head (spec_samples() False) refuses the launch
    import models.demos.glm53_flash.tt.generator_vllm_stack as gvs

    assert gen.spec_samples() is False and gen.spec_partial_groups() is False
    monkeypatch.setattr(gvs, "SPEC_SAMPLED", True)
    with expect_error(RuntimeError, "GLM53_SPEC_SAMPLED=1 declares spec_ids_are_sampled"):
        gvs.GLM53StackForCausalLM(gen, mb, 4096, spec=True)
    # the SERVED init order (chain_smtp_2 srv a refused to boot): the adapter's init checks spec_samples() before vLLM's
    # allocate_kv_cache builds the step (backend.spec None) -> the capability of the step the backend WILL build
    # (spec_build_caps = spec_step_caps, SpecMTP's own flags); once the cache built it, the built step's flags
    env = {"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_MULTI_USERS": "2", "GLM53_SPEC_SAMPLED": "1", "GLM53_SPEC_PARTIAL": "1"}
    assert spec_step_caps(1, "mtp", True, 10, env) == dict(groups=2, samples=True, partial=True)
    assert spec_step_caps(0, "mtp", True, 10, env) == dict(groups=0, samples=False, partial=False)  # (no step)
    assert not any(spec_step_caps(1, d, s, k, env)["samples"] for d, s, k in (("none", True, 10), ("mtp", False, 10),
                                                                              ("mtp", True, 1)))  # fmt: skip
    assert spec_step_caps(3, "mtp", True, 10, {**env, "GLM53_SPEC_MULTI": "0"}) == dict(groups=0, samples=True,
                                                                                     partial=False)  # fmt: skip
    nb = mb * SEQ // BLOCK_SIZE + mb
    for built_greedy in (False, True):
        be2 = (_MockSpecMultiBackend if built_greedy else _MockSpecMultiSampledBackend)(mb, K, G)
        gen2 = Glm53Generator(be2, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=mb, max_seq_len=SEQ)
        assert be2.spec is None and gen2.spec_samples() is False  # (a backend declaring nothing: greedy)
        be2.spec_build_caps = lambda: spec_step_caps(K, "mtp", True, 10, env)
        assert gen2.spec_samples() is True and gen2.spec_partial_groups() is True
        ad2 = gvs.GLM53StackForCausalLM(gen2, mb, SEQ, spec=True)  # boots: the step it will build samples
        if built_greedy:  # a build that disagrees with its declaration: the allocation refuses
            with expect_error(RuntimeError, "built with the cache has a greedy verify head"):
                ad2.allocate_kv_cache((nb, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
            continue
        ad2.allocate_kv_cache((nb, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
        assert be2.spec is not None and gen2.spec_samples() is True and gen2.spec_partial_groups() is False
    # ---- the sampled verify (GLM53_SPEC_SAMPLED) and partial groups (GLM53_SPEC_PARTIAL): their pure pieces
    assert spec_sampled_verify({}) is False and spec_sampled_verify({"GLM53_SPEC_SAMPLED": "1"}) is True
    assert spec_partial({"GLM53_SPEC_PARTIAL": "1"}) is False  # (multi-user only)
    assert spec_partial({"GLM53_SPEC_MULTI": "1", "GLM53_SPEC_PARTIAL": "1"}) is True
    assert spec_partial({"GLM53_SPEC_MULTI": "1"}) is False
    # a stream per column, per entry length, per user state; deterministic; never the all-zero xoshiro state
    base = [0x12345678, 0x9ABCDEF0, 0x0F0F0F0F, 0xF0F0F0F0]
    st = {(j, L): tuple(spec_verify_row_state(base, j, L)) for j in range(4) for L in (100, 101, 4096)}
    assert len(set(st.values())) == len(st) and all(any(v) for v in st.values())
    assert tuple(spec_verify_row_state(base, 2, 101)) == st[(2, 101)]
    assert tuple(spec_verify_row_state([1, 0, 0, 0], 0, 100)) != st[(0, 100)]
    # the pages: G 2, K 1 on the bases (0, 2), group 0 inactive -> rows 0, 1 greedy (zeros), rows 2, 3 = slot 2's
    # params and two distinct streams; the single-user step: rows 0..K of slot 0
    par_rows = torch.arange(32 * 4, dtype=torch.int64).reshape(32, 4)
    rng_rows = torch.arange(1000, 1000 + 32 * 4, dtype=torch.int64).reshape(32, 4)
    par, rng = spec_sampler_pages(1, 2, (0, 2), par_rows, rng_rows, {2: 77})
    assert par[:2].abs().sum() == 0 and rng[:2].abs().sum() == 0 and par[4:].abs().sum() == 0
    assert torch.equal(par[2], par_rows[2]) and torch.equal(par[3], par_rows[2])
    assert rng[2].tolist() == spec_verify_row_state(rng_rows[2].tolist(), 0, 77) != rng[3].tolist()
    par, rng = spec_sampler_pages(3, 0, (0,), par_rows, rng_rows, {0: 5})
    assert all(torch.equal(par[j], par_rows[0]) for j in range(4)) and par[4:].abs().sum() == 0
    assert len({tuple(rng[j].tolist()) for j in range(4)}) == 4
    # the verify KDA masks of a subset: G 4, K 1 (8 rows), groups 1 and 3 live
    assert spec_verify_masks(1, 4, 8, [1, 3]).reshape(-1).tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    # EXACTNESS of the sampled walk against greedy drafts: a toy target p(. | last token) over V tokens, K = 2 drafts
    # from another table's argmax; per step the columns draw independently from p(. | prefix, d_1..d_j) and the walk
    # emits a_0 .. a_n (+ the stop draw); the first two emitted tokens' joint law equals the plain sampled decode's
    V, N, gen_t = 5, 60000, torch.Generator().manual_seed(7)
    tgt = torch.softmax(torch.randn(V, V, generator=gen_t) * 1.5, dim=-1)  # p(next | last)
    drf = torch.randn(V, V, generator=gen_t).argmax(dim=-1)  # the drafter's greedy next token
    start = 3

    def draw(row):
        return int(torch.multinomial(row, 1, generator=gen_t))

    spec_pairs, plain_pairs = torch.zeros(V, V), torch.zeros(V, V)
    for _ in range(N):
        d1 = int(drf[start])
        d2 = int(drf[d1])
        cols = [draw(tgt[start]), draw(tgt[d1]), draw(tgt[d2])]  # column j conditions on prefix + d_1..d_j
        out = [cols[0]]
        if cols[0] == d1:
            out.append(cols[1])  # d1 accepted: column 1 is the draw after it (accepted or the stop draw)
        else:
            out.append(draw(tgt[cols[0]]))  # the walk stopped at a_0: the next step samples after it
        spec_pairs[out[0], out[1]] += 1
        a = draw(tgt[start])
        plain_pairs[a, draw(tgt[a])] += 1
    tv = 0.5 * float((spec_pairs / N - plain_pairs / N).abs().sum())
    assert tv < 0.02, tv
    # a greedy (argmax) verify answering sampled rows would NOT be exact: its first token is a point mass
    assert float(tgt[start].max()) < 0.99
    # ---- the generator's record mapping under a subset: G 2, K 1, bases (0, 2), only slot 2 entered
    backend.spec.partial = True
    b = SimpleNamespace(
        slots=[SimpleNamespace(seq_len=0, active=False, finished=False) for _ in range(mb)],
        tok_val=[-1] * mb, dev_ok=torch.zeros(mb, dtype=torch.bool), pos_shadow=torch.zeros(mb, dtype=torch.int64),
        step=0, spec=dict(groups={2: dict(drafts=[9], bonus=4)}),
    )  # fmt: skip
    b.slots[2].seq_len = 10
    recs = [dict(n=0, c=-1, a=[0, 0], committed=[0], drafts=[0], steps=1),  # group 0: inactive (spec_ctl c -1)
            dict(n=1, c=10, a=[5, 6], committed=[5, 6], drafts=[7], steps=1)]  # fmt: skip
    out = gen._spec_apply_groups(b, {2: 10}, recs)
    assert list(out) == [2] and b.slots[2].seq_len == 12 and b.tok_val[2] == 6 and b.slots[0].seq_len == 0
    assert b.spec == dict(groups={2: dict(drafts=[7], bonus=6)})
    with expect_error(RuntimeError, "1 records for the 2 groups"):
        gen._spec_apply_groups(b, {2: 12}, recs[1:])
    # ---- the sampled verify's pages carry EACH user's own params and RNG stream, single-user and G users (device
    # 11-13, chain_smtp_2: the enter read the params rows before anything had written them after the prefill -- users
    # 1..G-1 sat on never-written greedy rows and decoded greedy; user 0 inherited the plain run's slot-0 row)
    for K, G, mb in ((1, 4, 8), (3, 0, B)):
        backend = _MockSpecMultiSampledBackend(mb, K, G) if G else _MockSpecSampledBackend(mb, K)
        gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=mb, max_seq_len=SEQ)
        n = max(1, G)
        sps = [SamplingParams(temperature=0.5 + 0.25 * u, top_k=3 + u, top_p=0.9 - 0.05 * u, seed=100 + u)
               for u in range(n)]  # fmt: skip
        out = gen.generate_batch([_prompt(60 + 7 * u, 3 + u) for u in range(n)], 16, sampling=sps, stop_on_eos=False,
                                 enable_trace=True, spec_k=K)  # fmt: skip
        assert gen.spec_samples() and [len(o) for o in out] == [16] * n
        assert 1 <= backend.spec.counters["enters"] == len(backend.spec.pages)
        par, rng, Ls = backend.spec.pages[0]
        bases = list(gen.spec_slots()) if G else [0]
        assert sorted(Ls) == bases
        rows = set()
        for u, s in enumerate(bases):
            k, p, t = sps[u].device_values()
            for j in range(K + 1):
                r = u * (K + 1) + j
                assert par[r, :3].tolist() == [k, round(p * 1000), round(t * 1000)], (G, u, j, par[r].tolist())
                assert rng[r].tolist() == spec_verify_row_state(xoshiro_state(100 + u), j, Ls[s]), (G, u, j)
                rows.add(tuple(rng[r].tolist()))
        assert len(rows) == n * (K + 1) and par[n * (K + 1) :].abs().sum() == 0  # distinct streams; the rest zeros
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- Stage 17: the served single-user speculation
def test_spec_plan_for_the_lone_user():
    """spec_plan: off unless GLM53_STACK_VLLM_SPEC=1 with GLM53_SPEC_K; effective_k = K (num_speculative_tokens >= K),
    B in [K + 1, 4]; argmax_ids, narrow decode kept, k_by_rows reaching max_num_seqs (multi-row steps decline through
    DraftOutput.num_valid anyway)"""
    T = contract_types()
    env = {"GLM53_STACK_VLLM_SPEC": "1", "GLM53_SPEC_K": "3"}
    p = spec_plan_for(4, 3, T, env)
    assert isinstance(p, T["SpecPlan"]) and p.effective_k == 3 and p.accept_modes == ("argmax_ids",)
    assert p.supports_narrow_decode and p.k_by_rows == ((1, 3), (4, 1)) and p.drafter_state == "internal"
    assert isinstance(spec_plan_for(4, 5, T, env), T["SpecPlan"])  # more asked: K served
    r = spec_plan_for(4, 2, T, env)
    assert isinstance(r, T["SpecReject"]) and r.supported_k == (3,)
    r = spec_plan_for(3, 3, T, env)
    assert isinstance(r, T["SpecReject"]) and r.supported_k == (1, 2)
    assert isinstance(spec_plan_for(8, 3, T, env), T["SpecReject"])
    assert isinstance(spec_plan_for(4, 3, T, {"GLM53_SPEC_K": "3"}), T["SpecReject"])
    # GLM53_SPEC_MULTI=1: a B 16 launch is planned with the multi-user caps (k_by_rows from spec_multi_k_by_rows);
    # the single-user plan is unchanged with the knob off
    menv = dict(env, GLM53_SPEC_MULTI="1")
    p = spec_plan_for(16, 3, T, menv)
    assert isinstance(p, T["SpecPlan"]) and p.effective_k == 3 and p.k_by_rows == ((4, 3), (8, 1), (16, 1))
    # the multi-user caps come from the BUILD (GLM53_STACK_BUILD_B, default max_num_seqs): the G 4 x (K 3 + 1) = 16
    # verify rows do not fit a 4-row build; on a 16-row build 4 users take K 3
    assert isinstance(spec_plan_for(4, 3, T, menv), T["SpecReject"])
    assert spec_plan_for(4, 3, T, dict(menv, GLM53_STACK_BUILD_B="16")).k_by_rows == ((4, 3),)
    assert isinstance(spec_plan_for(32, 3, T, menv), T["SpecReject"])
    assert spec_plan_for(4, 3, T, env).k_by_rows == ((1, 3), (4, 1))
    # GLM53_SPEC_STEPS_PER_CALL = M: effective_k M (K + 1) - 1 (num_speculative_tokens must reach it), fused_sample
    # only, every draft cap k -> M (k + 1) - 1; refused with the deferred (async) verify
    e4 = dict(env, GLM53_SPEC_STEPS_PER_CALL="4")
    p = spec_plan_for(4, 15, T, e4)
    assert p.effective_k == 15 and p.accept_modes == ("fused_sample",) and p.k_by_rows == ((1, 15), (4, 1))
    r = spec_plan_for(4, 3, T, e4)
    assert isinstance(r, T["SpecReject"]) and r.supported_k == (15,)
    p = spec_plan_for(16, 15, T, dict(menv, GLM53_SPEC_STEPS_PER_CALL="4"))
    assert p.effective_k == 15 and p.accept_modes == ("fused_sample",) and p.k_by_rows == ((4, 15), (8, 7), (16, 7))
    assert spec_plan_for(16, 15, T, dict(menv, GLM53_SPEC_STEPS_PER_CALL="2")).effective_k == 7
    assert isinstance(spec_plan_for(4, 15, T, dict(e4, GLM53_STACK_VLLM_SPEC_ASYNC="1")), T["SpecReject"])
    assert spec_plan_for(4, 3, T, dict(env, GLM53_SPEC_STEPS_PER_CALL="1")).accept_modes == ("argmax_ids",)
    assert gvs.spec_plugin_k(3, 1) == 3 and gvs.spec_plugin_k(1, 4) == 7
    ids, counts, n = gvs.fused_sample_block([dict(n=1, a=[5, 6, 9]), dict(n=0, a=[7, 1, 1])], 2, 4, 8, -1)
    assert n == 3 and counts.tolist() == [1, 1, 3, 1] and ids[2].tolist() == [5, 6, 7] + [-1] * 5
    assert ids[0].tolist() == [0] + [-1] * 7
    ids, counts, n = gvs.fused_sample_block([{0: dict(n=0, a=[3, 4]), 2: dict(n=1, a=[5, 6])}], {1: 2, 0: 0}, 4, 4, -1)
    assert n == {2: 2, 0: 1} and counts.tolist() == [1, 2, 1, 1] and ids[1].tolist() == [5, 6, -1, -1]
    with expect_error(ValueError, "committed tokens for a 2-column"):
        gvs.fused_sample_block([dict(n=1, a=[5, 6]), dict(n=0, a=[7, 1])], 0, 4, 2, -1)


def _spec_adapter(K: int = 3, good=lambda step: 3, backend_cls=None):
    backend = (backend_cls or _MockSpecBackend)(B, K, good)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ, spec=True)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    return ad, backend, gen, kv


def _verify(ad, kv, K, rows: dict, table):
    """the plugin's verify block (rows: plugin row -> (committed token, its position, drafts)) -> the walk's
    (committed [B, 1+K], counts [B]) as the runner computes them (model_runner trims the drafts to the answer's width)
    """
    tok = torch.zeros(B, 1 + K, dtype=torch.int32)
    pos = torch.full((B, 1 + K), -1, dtype=torch.int32)
    nv = torch.zeros(B, dtype=torch.int32)
    dr = torch.full((B, K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    for i, (t, p, d) in rows.items():
        tok[i, 0], pos[i, 0] = t, p
        for j, x in enumerate(d):
            tok[i, 1 + j], pos[i, 1 + j], dr[i, j] = x, p + 1 + j, x
        nv[i] = len(d)
    out = ad.decode_forward(tok, pos, page_table=table, kv_cache=kv, sampling_params=_greedy(B), num_valid_drafts=nv,
                            accepted_counts=torch.ones(B, dtype=torch.int32), spec_mode="argmax_ids")  # fmt: skip
    ids = out.argmax_ids
    return ids, accept_greedy_walk(ids, dr[:, : int(ids.shape[1]) - 1], nv)


def _block(rows: dict, K: int):
    """the plugin's committed block for propose_draft_tokens (rows: plugin row -> (committed outputs, position of the
    first)): the walk's committed tokens, column j at its position (model_runner._committed_positions), padding -1"""
    ct = torch.full((B, 1 + K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
    cp = torch.full((B, 1 + K), -1, dtype=torch.int32)
    ac = torch.ones(B, dtype=torch.int32)
    for i, (toks, p) in rows.items():
        ct[i, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
        cp[i, :] = torch.arange(p, p + 1 + K)
        ac[i] = len(toks)
    return ct, cp, ac


def _fused_verify(ad, kv, Kp: int, rows: dict, table):
    """the plugin's fused_sample verify (GLM53_SPEC_STEPS_PER_CALL): a [B, 1 + Kp] block (rows: plugin row ->
    (committed token, its position, the K drafts); the columns past them padded, position -1) -> the model's
    (accepted_token_ids, accepted_counts), checked as the runner checks them (each row's prefix ends at its count)"""
    W = 1 + Kp
    tok = torch.zeros(B, W, dtype=torch.int32)
    pos = torch.full((B, W), -1, dtype=torch.int32)
    nv = torch.zeros(B, dtype=torch.int32)
    for i, (t, p, d) in rows.items():
        tok[i, 0], pos[i, 0] = t, p
        for j, x in enumerate(d):
            tok[i, 1 + j], pos[i, 1 + j] = x, p + 1 + j
        nv[i] = len(d)
    out = ad.decode_forward(tok, pos, page_table=table, kv_cache=kv, sampling_params=_greedy(B), num_valid_drafts=nv,
                            accepted_counts=torch.ones(B, dtype=torch.int32), spec_mode="fused_sample")  # fmt: skip
    assert out.spec_mode == "fused_sample"
    ids, counts = out.accepted_token_ids, out.accepted_counts
    for i in range(B):
        c = int(counts[i])
        assert PLACEHOLDER_TOKEN_ID not in ids[i, :c].tolist()
        assert all(int(v) == PLACEHOLDER_TOKEN_ID for v in ids[i, c:].tolist())
    return ids, counts


def _served_chain_lone(monkeypatch, sharded: bool = False):
    """GLM53_SPEC_STEPS_PER_CALL = 3 on the lone-user served path: effective_k 3 (K + 1) - 1 = 11; propose hands the
    device's K drafts in the 11-wide block, the verify (fused_sample) chains 3 device steps -- every record collected
    after the last replay -- and commits all their tokens at once; the stream equals plain greedy and each proposal
    continues the chain's last step (the plugin's count = the 3 steps' total); a second live row: the ordinary decode
    answers one token per row (count 1). The block table grows as vLLM's does (the block's positions allocated per
    verify): the rows that only gained blocks are written behind the first replay (GLM53_SPEC_PT_DEFER, default on) and
    the mock step asserts every position it addresses is mapped in the published table. Never deferred (written before
    the first replay, which reads the new row): a first step that reaches past the published coverage (the first
    verify at 122 over 2 published blocks) and a row that CHANGES a covered entry (column 0 on another block).
    sharded: the owner-aware pool (_MockSpecPrefixBackend; no changed entry: a computed column's block must keep its
    content) -- a deferred publish leaves the covered physical entries the first replay reads as they were"""
    monkeypatch.setenv("GLM53_SPEC_STEPS_PER_CALL", "3")
    if sharded:
        monkeypatch.setenv("GLM53_SPEC_ON_SHARD", "1")  # (the prefix mock's sequence-sharded pool)
    K, M = 3, 3
    Kp = gvs.spec_plugin_k(K, M)
    ad, backend, gen, kv = _spec_adapter(K, good=lambda step: step % (K + 1),
                                         backend_cls=_MockSpecPrefixBackend if sharded else None)  # fmt: skip
    assert ad.spec_steps_per_call == M and Kp == 11
    prompt = _prompt(121, 7)
    table = _table([_blocks(0, 122)], B)
    out, _ = ad.prefill_forward(_history([prompt]), page_table=table, kv_cache=kv, prompt_lens=[121], start_pos=[0],
                                empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    stream, pos = [int(out[0, 0])], 121
    ids = ad.decode_forward(torch.tensor([[stream[-1]], [0], [0], [0]]), torch.tensor([pos, -1, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    stream.append(int(ids[0, 0]))
    pos += 1
    ct, cp, ac = _block({0: ([stream[-1]], pos)}, Kp)
    s0, r0 = backend.spec.counters["steps"], gen.counters["page_table_refresh"]
    b, ev, step0, write0 = gen._binding(kv), [], backend.spec.step, gen._write_page_rows

    def step(*a, **kw):  # (each replay with the published table it reads)
        ev.append(("step", backend.io.page_table.t[0].clone()))
        return step0(*a, **kw)

    def write(*a, **kw):
        ev.append(("write", None))
        return write0(*a, **kw)

    monkeypatch.setattr(backend.spec, "step", step)
    monkeypatch.setattr(gen, "_write_page_rows", write)
    swap = None
    for case in ("past", "grow", "grow", "grow") if sharded else ("past", "change", "grow", "grow"):
        do = ad.propose_draft_tokens(Kp, ct, cp, ac)
        assert do.num_valid.tolist() == [K, 0, 0, 0] and tuple(do.draft_token_ids.shape) == (B, Kp)
        drafts = [int(v) for v in do.draft_token_ids[0, :K].tolist()]
        row = _blocks(0, pos + 2 * (Kp + 1))  # (vLLM: the block's positions + lookahead allocated)
        swap = _blocks(3, 1)[0] if case == "change" else swap
        row[0] = row[0] if swap is None else swap
        table = _table([row], B)
        L0, cov0, pub0, n0 = int(b.slots[0].seq_len), int(b.coverage[0]), backend.io.page_table.t[0].clone(), len(ev)
        d0, ph0 = gen.counters.get("page_rows_deferred", 0), b.phys.clone() if sharded else None
        acc, counts = _fused_verify(ad, kv, Kp, {0: (stream[-1], pos, drafts)}, table)
        kinds, seen = [k for k, _t in ev[n0:]], next(t for k, t in ev[n0:] if k == "step")
        assert (L0 + 2 * K + 1 > cov0 * BLOCK_SIZE) == (case == "past"), (case, L0, cov0)
        if case == "grow":  # behind the first replay, which reads the published table
            assert kinds[:2] == ["step", "write"] and gen.counters["page_rows_deferred"] == d0 + 1, (case, kinds)
            assert torch.equal(seen, pub0)
            if sharded:
                assert torch.equal(b.phys[:, :cov0], ph0[:, :cov0]), "a deferred publish moved a covered entry"
        else:  # before the first replay, which reads the call's row
            assert kinds[:2] == ["write", "step"] and gen.counters.get("page_rows_deferred", 0) == d0, (case, kinds)
            assert torch.equal(seen.to(torch.int64), (b.phys if sharded else b.pt_shadow)[0].to(torch.int64))
            assert sharded or seen[: len(row)].tolist() == row
        new = [int(v) for v in acc[0, : int(counts[0])].tolist()]
        assert M <= len(new) <= M * (K + 1) and counts[1:].tolist() == [1, 1, 1]
        ct, cp, ac = _block({0: (new, pos + 1)}, Kp)
        stream += new
        pos += len(new)
    assert backend.spec.counters["steps"] - s0 == 4 * M == ad.spec_stats["fused"] and ad.spec_stats["calls"] == 4
    assert gen.counters["spec_chains"] == 4 and ad.spec_stats["committed"] == len(stream) - 2
    assert ad.spec_stats["committed"] > 4 * M  # (drafts accepted inside the chain)
    assert stream == [(prompt[-1] + 1 + i) % VOCAB for i in range(len(stream))]  # = plain greedy (next = t + 1)
    assert gen.counters["page_table_refresh"] > r0 and gen.counters.get("page_rows_deferred", 0) >= 1
    do = ad.propose_draft_tokens(Kp, ct, cp, ac)
    drafts = [int(v) for v in do.draft_token_ids[0, :K].tolist()]
    p2 = _prompt(80, 30)
    table2 = _table([_blocks(0, 400), _blocks(1, 300)], B)
    out2, _ = ad.prefill_forward(_history([p2]), page_table=table2[1:2], kv_cache=kv, prompt_lens=[80], start_pos=[0],
                                 empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    tok2 = int(out2[0, 0])
    acc, counts = _fused_verify(ad, kv, Kp, {0: (stream[-1], pos, drafts), 1: (tok2, 80, [])}, table2)
    assert tuple(acc.shape) == (B, 1) and counts.tolist() == [1, 1, 1, 1] and ad.spec_stats["fallback"] == 1
    assert int(acc[0, 0]) == (stream[-1] + 1) % VOCAB and int(acc[1, 0]) == (tok2 + 1) % VOCAB
    ad.release_request(1)
    ad.release_request(0)


def _served_chain_multi(monkeypatch):
    """GLM53_SPEC_STEPS_PER_CALL = 4 with GLM53_SPEC_MULTI (G 2, K 1: effective_k 7): a verify of both users' blocks
    chains 4 multi-user device steps and commits each user's 4 steps' tokens; both streams equal plain greedy"""
    monkeypatch.setenv("GLM53_SPEC_STEPS_PER_CALL", "4")
    K, G, M = 1, 2, 4
    Kp = gvs.spec_plugin_k(K, M)
    backend = _MockSpecMultiBackend(B, K, G, lambda step: K if step % 3 else 0)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ, spec=True)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.spec_multi_capable and ad.spec_steps_per_call == M
    pa, pb = _prompt(100, 7), _prompt(90, 20)
    table = _table([_blocks(0, 400), _blocks(1, 400)], B)
    out, _ = ad.prefill_forward(_history([pa]), page_table=table[0:1], kv_cache=kv, prompt_lens=[100], start_pos=[0],
                                empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    sa = [int(out[0, 0])]
    out, _ = ad.prefill_forward(_history([pb]), page_table=table[1:2], kv_cache=kv, prompt_lens=[90], start_pos=[0],
                                empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    sb = [int(out[0, 0])]
    want = lambda p, s: [(p[-1] + 1 + i) % VOCAB for i in range(len(s))]  # noqa: E731
    ids = ad.decode_forward(torch.tensor([[sa[-1]], [sb[-1]], [0], [0]]), torch.tensor([100, 90, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    sa.append(int(ids[0, 0]))
    sb.append(int(ids[1, 0]))
    pa_, pb_ = 101, 91
    ct, cp, ac = _block({0: ([sa[-1]], pa_), 1: ([sb[-1]], pb_)}, Kp)
    for _ in range(3):
        do = ad.propose_draft_tokens(Kp, ct, cp, ac)
        assert do.num_valid.tolist() == [K, K, 0, 0]
        da, db = ([int(v) for v in do.draft_token_ids[i, :K].tolist()] for i in (0, 1))
        acc, counts = _fused_verify(ad, kv, Kp, {0: (sa[-1], pa_, da), 1: (sb[-1], pb_, db)}, table)
        na, nb = ([int(v) for v in acc[i, : int(counts[i])].tolist()] for i in (0, 1))
        assert M <= len(na) <= M * (K + 1) and M <= len(nb) <= M * (K + 1)
        ct, cp, ac = _block({0: (na, pa_ + 1), 1: (nb, pb_ + 1)}, Kp)
        sa += na
        sb += nb
        pa_, pb_ = pa_ + len(na), pb_ + len(nb)
    assert ad.spec_stats["fused"] == 3 * M == backend.spec.counters["steps"] and ad.spec_stats["calls"] == 3
    assert sa == want(pa, sa) and sb == want(pb, sb)


def test_served_spec_lone_user_flow(monkeypatch):
    """the plugin's calls for one greedy request (SPEC_DECODE_CONTRACT.md section 5): prefill (+ the MTP seed: enter /
    leave), an ordinary decode, propose (the bootstrap: K drafts), verify (the fused step: K + 1 argmax ids, the walk
    accepts what the device accepted), propose (the step's drafts) ... -- the committed stream equals plain greedy, the
    slot state ledger stays consistent (the mock asserts it); then a second request arrives: the speculation is left
    before its prefill, the outstanding drafts' verify is answered by an ordinary decode (one column: one token per row),
    the proposals decline (two live rows); after it is released, the lone request speculates again"""
    K = 3
    ad, backend, gen, kv = _spec_adapter(K)
    prompt = _prompt(100, 7)
    table = _table([_blocks(0, 400)], B)
    out, _ = ad.prefill_forward(_history([prompt]), page_table=table, kv_cache=kv, prompt_lens=[100], start_pos=[0],
                                empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    stream, pos = [int(out[0, 0])], 100  # pos: where the last committed token sits (the next input's position)
    assert ad.spec_stats["seeds"] == 1 and backend.spec.counters["enters"] == backend.spec.counters["leaves"] == 1
    assert [sp[:2] for sp in backend.mtp_spans] == [(0, 99)] and backend.mtp_spans[0][2] == prompt[1:]  # the MTP fill
    ids = ad.decode_forward(torch.tensor([[stream[-1]], [0], [0], [0]]), torch.tensor([pos, -1, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    stream.append(int(ids[0, 0]))
    pos += 1
    ct, cp, ac = _block({0: ([stream[-1]], pos)}, K)
    for _ in range(6):
        do = ad.propose_draft_tokens(K, ct, cp, ac)
        assert do.num_valid.tolist() == [K, 0, 0, 0]
        drafts = [int(v) for v in do.draft_token_ids[0].tolist()]
        ids, (committed, counts) = _verify(ad, kv, K, {0: (stream[-1], pos, drafts)}, table)
        assert tuple(ids.shape) == (B, 1 + K)
        new = [int(v) for v in committed[0, : int(counts[0])].tolist()]
        ct, cp, ac = _block({0: (new, pos + 1)}, K)
        stream += new
        pos += len(new)
    assert ad.spec_stats["fused"] == 6 and ad.spec_stats["enters"] == 1 and ad.spec_stats["committed"] == 6 * (K + 1)
    assert stream == [(prompt[-1] + 1 + i) % VOCAB for i in range(len(stream))]  # = plain greedy (next = t + 1)
    do = ad.propose_draft_tokens(K, ct, cp, ac)  # the drafts the last step made, outstanding
    drafts = [int(v) for v in do.draft_token_ids[0].tolist()]
    # a second request arrives: the speculation is left before its prefill (its slot's state rows are not touched)
    p2 = _prompt(80, 30)
    table2 = _table([_blocks(0, 400), _blocks(1, 300)], B)
    out2, _ = ad.prefill_forward(_history([p2]), page_table=table2[1:2], kv_cache=kv, prompt_lens=[80], start_pos=[0],
                                 empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    assert getattr(gen._binding(kv), "spec", None) is None and backend.spec.counters["leaves"] == 2
    assert len(backend.mtp_spans) == 1  # (slot 1's prefill: not armed -- only slot 0 can speculate)
    # the plugin still verifies the first request's outstanding drafts beside the new row: an ordinary decode answers
    tok2 = int(out2[0, 0])
    ids, (committed, counts) = _verify(ad, kv, K, {0: (stream[-1], pos, drafts), 1: (tok2, 80, [])}, table2)
    assert tuple(ids.shape) == (B, 1) and counts.tolist() == [1, 1, 1, 1] and ad.spec_stats["fallback"] == 1
    assert int(ids[0, 0]) == (stream[-1] + 1) % VOCAB and int(ids[1, 0]) == (tok2 + 1) % VOCAB
    stream.append(int(ids[0, 0]))
    pos += 1
    ct, cp, ac = _block({0: ([stream[-1]], pos), 1: ([int(ids[1, 0])], 81)}, K)
    do = ad.propose_draft_tokens(K, ct, cp, ac)
    assert do.num_valid.tolist() == [0, 0, 0, 0] and ad.spec_stats["declines"] >= 1  # two live rows: no drafts
    ad.release_request(1)  # the second request ends: the first one is alone again
    ids = ad.decode_forward(torch.tensor([[stream[-1]], [0], [0], [0]]), torch.tensor([pos, -1, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    stream.append(int(ids[0, 0]))
    pos += 1
    do = ad.propose_draft_tokens(K, *_block({0: ([stream[-1]], pos)}, K))
    assert do.num_valid.tolist() == [K, 0, 0, 0] and ad.spec_stats["enters"] == 2
    ids, (committed, counts) = _verify(ad, kv, K, {0: (stream[-1], pos, [int(v) for v in do.draft_token_ids[0]])},
                                       table)  # fmt: skip
    assert int(counts[0]) == K + 1 and ad.spec_stats["fused"] == 7
    stream += [int(v) for v in committed[0].tolist()]
    assert stream == [(prompt[-1] + 1 + i) % VOCAB for i in range(len(stream))]
    ad.release_request(0)  # released while speculating: abort, no leave
    assert backend.spec.counters.get("aborts") == 1 and getattr(gen._binding(kv), "spec", None) is None
    _served_chain_lone(monkeypatch)
    _served_chain_lone(monkeypatch, sharded=True)
    assert "ttnn" not in sys.modules


def test_served_spec_declines_sampled_rows():
    """a lone sampled request is never speculated (argmax_ids certifies greedy rows only): proposals decline"""
    K = 3
    ad, backend, gen, kv = _spec_adapter(K)
    table = _table([_blocks(0, 300)], B)
    sampled = SimpleNamespace(temperature=[1.0] * B, top_k=[0] * B, top_p=[0.95] * B, seed=[None] * B)
    out, _ = ad.prefill_forward(_history([_prompt(60, 2)]), page_table=table, kv_cache=kv, prompt_lens=[60],
                                start_pos=[0], empty_slots=[0], sampling_params=sampled)  # fmt: skip
    assert ad.spec_stats["seeds"] == 0
    do = ad.propose_draft_tokens(K, *_block({0: ([int(out[0, 0])], 60)}, K))
    assert do.num_valid.tolist() == [0, 0, 0, 0] and backend.spec.counters["enters"] == 0


def test_served_spec_multi_user_flow(monkeypatch):
    """GLM53_SPEC_MULTI on the served path (the fixed-G v1, G = 2, K = 1 on the 4-slot mock: group bases 0 / 2): a new
    request takes the lowest free group base (the second one lands in generator slot 2, invisible to the plugin), each
    prefill fills its user's MTP cache, the MTP seed runs once both users are live; propose hands K drafts to BOTH rows
    (the bootstrap's, then the last fused step's), a verify of exactly both users' blocks runs ONE fused step (both
    rows' K + 1 argmax ids) -- every stream equals plain greedy; a third live request leaves the speculation (its
    verify answered by an ordinary decode, the proposals decline), after its release the two users speculate again; a
    user released while speculating LEAVES (the other user's committed state moves to its base row), never aborts"""
    K, G = 1, 2
    backend = _MockSpecMultiBackend(B, K, G, lambda step: K if step % 3 else 0)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ, spec=True)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.spec_multi_capable and gen.spec_slots() == (0, 2)
    pa, pb, pc = _prompt(100, 7), _prompt(90, 20), _prompt(80, 30)
    table = _table([_blocks(0, 400), _blocks(1, 400), _blocks(2, 300)], B)
    out, _ = ad.prefill_forward(_history([pa]), page_table=table[0:1], kv_cache=kv, prompt_lens=[100], start_pos=[0],
                                empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    sa = [int(out[0, 0])]
    assert ad.spec_stats["seeds"] == 0  # (one of the G users: no step)
    out, _ = ad.prefill_forward(_history([pb]), page_table=table[1:2], kv_cache=kv, prompt_lens=[90], start_pos=[0],
                                empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    sb = [int(out[0, 0])]
    assert ad._alias[:2] == [0, 2] and ad.spec_stats["seeds"] == 1  # the second group base; both live: the seed
    assert sorted({s for s, *_ in backend.mtp_spans}) == [0, 2]  # (each user's MTP cache filled by its prefill)
    want = lambda p, s: [(p[-1] + 1 + i) % VOCAB for i in range(len(s))]  # noqa: E731  (plain greedy: next = t + 1)
    ids = ad.decode_forward(torch.tensor([[sa[-1]], [sb[-1]], [0], [0]]), torch.tensor([100, 90, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    sa.append(int(ids[0, 0]))
    sb.append(int(ids[1, 0]))
    pa_, pb_ = 101, 91  # the next inputs' positions
    ct, cp, ac = _block({0: ([sa[-1]], pa_), 1: ([sb[-1]], pb_)}, K)

    def fused_steps(n_steps, ct, cp, ac):
        nonlocal pa_, pb_
        for _ in range(n_steps):
            do = ad.propose_draft_tokens(K, ct, cp, ac)
            assert do.num_valid.tolist() == [K, K, 0, 0]
            da, db = ([int(v) for v in do.draft_token_ids[i].tolist()] for i in (0, 1))
            ids, (committed, counts) = _verify(ad, kv, K, {0: (sa[-1], pa_, da), 1: (sb[-1], pb_, db)}, table)
            assert tuple(ids.shape) == (B, 1 + K)
            na, nb = ([int(v) for v in committed[i, : int(counts[i])].tolist()] for i in (0, 1))
            ct, cp, ac = _block({0: (na, pa_ + 1), 1: (nb, pb_ + 1)}, K)
            sa.extend(na)
            sb.extend(nb)
            pa_, pb_ = pa_ + len(na), pb_ + len(nb)
        return ct, cp, ac

    ct, cp, ac = fused_steps(6, ct, cp, ac)
    assert ad.spec_stats["fused"] == 6 and ad.spec_stats["enters"] == 1 and backend.spec.counters["steps"] == 6
    assert ad.spec_stats["committed"] > 12 and sa == want(pa, sa) and sb == want(pb, sb)
    do = ad.propose_draft_tokens(K, ct, cp, ac)  # the drafts the last step made, outstanding
    da, db = ([int(v) for v in do.draft_token_ids[i].tolist()] for i in (0, 1))
    # a third request: the speculation is left before its prefill; it takes a free non-base slot
    out, _ = ad.prefill_forward(_history([pc]), page_table=table[2:3], kv_cache=kv, prompt_lens=[80], start_pos=[0],
                                empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    sc = [int(out[0, 0])]
    assert getattr(gen._binding(kv), "spec", None) is None and ad._alias[2] == 1
    ids, (committed, counts) = _verify(ad, kv, K, {0: (sa[-1], pa_, da), 1: (sb[-1], pb_, db), 2: (sc[-1], 80, [])},
                                       table)  # fmt: skip
    assert tuple(ids.shape) == (B, 1) and ad.spec_stats["fallback"] == 1
    sa.append(int(ids[0, 0]))
    sb.append(int(ids[1, 0]))
    pa_, pb_ = pa_ + 1, pb_ + 1
    do = ad.propose_draft_tokens(K, *_block({0: ([sa[-1]], pa_), 1: ([sb[-1]], pb_), 2: ([int(ids[2, 0])], 81)}, K))
    assert do.num_valid.tolist() == [0, 0, 0, 0]  # three live rows: no drafts
    ad.release_request(2)
    ids = ad.decode_forward(torch.tensor([[sa[-1]], [sb[-1]], [0], [0]]), torch.tensor([pa_, pb_, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    sa.append(int(ids[0, 0]))
    sb.append(int(ids[1, 0]))
    pa_, pb_ = pa_ + 1, pb_ + 1
    ct, cp, ac = fused_steps(3, *_block({0: ([sa[-1]], pa_), 1: ([sb[-1]], pb_)}, K))
    assert ad.spec_stats["enters"] == 2 and sa == want(pa, sa) and sb == want(pb, sb)
    # user A ends while both speculate: leave (B's committed row to its base), no abort; B decodes on plainly
    ad.release_request(0)
    assert backend.spec.counters.get("aborts") is None and getattr(gen._binding(kv), "spec", None) is None
    do = ad.propose_draft_tokens(K, *_block({1: ([sb[-1]], pb_)}, K))
    assert do.num_valid.tolist() == [0, 0, 0, 0]  # (one user of G: no step)
    for _ in range(3):
        ids = ad.decode_forward(torch.tensor([[0], [sb[-1]], [0], [0]]), torch.tensor([-1, pb_, -1, -1]),
                                page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
        sb.append(int(ids[1, 0]))
        pb_ += 1
    assert sb == want(pb, sb)
    _served_alt_flow()
    _served_chain_multi(monkeypatch)
    assert "ttnn" not in sys.modules


def _served_alt_flow():
    """GLM53_SPEC_MULTI_ALT on the served path (tt/spec_mtp.py SpecSwitch, 8-slot mock): the primary 2 x 3 step (bases
    0 / 4) and the alternate 4 x 1 step (bases 0 / 2 / 4 / 6). New requests take the primary's bases, then the
    alternate's; two users speculate on the primary (3 drafts), the third and fourth move everyone to the alternate (the
    seed selects it; 1 draft per row in the plugin's K = 3 block, columns 2..3 padded -1), after two releases the two
    left speculate on the primary again -- every stream equals plain greedy"""
    mb, K0 = 8, 3

    class _AltBackend(_MockSpecMultiBackend):
        def allocate_decode_io(self, kv):
            io = super().allocate_decode_io(kv)
            self.spec = _MockSpecSwitch([self.spec, _MockSpecMulti(self, 1, 4, self.good)])
            return io

    be = _AltBackend(mb, K0, 2, lambda step: K0)
    gen = Glm53Generator(be, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=mb, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, mb, SEQ, spec=True)
    kv = ad.allocate_kv_cache((mb * SEQ // BLOCK_SIZE + mb, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=mb, num_blocks=BPS, can_sample_on_device=True)
    assert gen.spec_steps() == [(3, 2, (0, 4)), (1, 4, (0, 2, 4, 6))] and gen.spec_all_slots() == (0, 4, 2, 6)
    assert gen.spec_k_max() == 3 and gen.spec_current() == 0
    ps = [_prompt(100 - 7 * i, 5 + 11 * i) for i in range(4)]
    tab = torch.zeros(mb, BPS, dtype=torch.int32)
    for i in range(4):
        bl = _blocks(i, 400)
        tab[i, : len(bl)] = torch.tensor(bl, dtype=torch.int32)
    st, nxt = {}, {}
    want = lambda p, s: [(p[-1] + 1 + i) % VOCAB for i in range(len(s))]  # noqa: E731

    def pre(i):
        out, _ = ad.prefill_forward(_history([ps[i]]), page_table=tab[i : i + 1], kv_cache=kv, prompt_lens=[len(ps[i])],
                                    start_pos=[0], empty_slots=[i], sampling_params=_greedy(1))  # fmt: skip
        st[i], nxt[i] = [int(out[0, 0])], len(ps[i])

    def plain(users):
        tok, pos = torch.zeros(mb, 1, dtype=torch.int64), torch.full((mb,), -1, dtype=torch.int64)
        for i in users:
            tok[i, 0], pos[i] = st[i][-1], nxt[i]
        ids = ad.decode_forward(tok, pos, page_table=tab, kv_cache=kv, sampling_params=_greedy(mb))
        for i in users:
            st[i].append(int(ids[i, 0]))
            nxt[i] += 1

    def blk(rows):
        ct = torch.full((mb, 1 + K0), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        cp = torch.full((mb, 1 + K0), -1, dtype=torch.int32)
        ac = torch.ones(mb, dtype=torch.int32)
        for i, (toks, p) in rows.items():
            ct[i, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
            cp[i, :] = torch.arange(p, p + 1 + K0)
            ac[i] = len(toks)
        return ct, cp, ac

    def rounds(n, users, k):
        ct, cp, ac = blk({i: ([st[i][-1]], nxt[i]) for i in users})
        for _ in range(n):
            do = ad.propose_draft_tokens(K0, ct, cp, ac)
            assert do.num_valid.tolist() == [k if i in users else 0 for i in range(mb)], do.num_valid.tolist()
            tok = torch.zeros(mb, 1 + K0, dtype=torch.int32)
            pos = torch.full((mb, 1 + K0), -1, dtype=torch.int32)
            nv = torch.zeros(mb, dtype=torch.int32)
            dr = torch.full((mb, K0), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            for i in users:  # (the runner's block: K0 columns, the row's k real drafts, the rest padded -1)
                tok[i, 0], pos[i, 0], nv[i] = st[i][-1], nxt[i], k
                for j in range(k):
                    x = int(do.draft_token_ids[i, j])
                    tok[i, 1 + j], pos[i, 1 + j], dr[i, j] = x, nxt[i] + 1 + j, x
            out = ad.decode_forward(tok, pos, page_table=tab, kv_cache=kv, sampling_params=_greedy(mb),
                                    num_valid_drafts=nv, accepted_counts=torch.ones(mb, dtype=torch.int32),
                                    spec_mode="argmax_ids")  # fmt: skip
            ids = out.argmax_ids
            assert tuple(ids.shape) == (mb, 1 + K0), tuple(ids.shape)
            committed, counts = accept_greedy_walk(ids, dr[:, : int(ids.shape[1]) - 1], nv)
            new = {i: [int(v) for v in committed[i, : int(counts[i])].tolist()] for i in users}
            ct, cp, ac = blk({i: (new[i], nxt[i] + 1) for i in users})
            for i in users:
                assert 1 <= len(new[i]) <= k + 1, (i, new[i], k)
                st[i].extend(new[i])
                nxt[i] += len(new[i])

    pre(0)
    pre(1)
    assert ad._alias[:2] == [0, 4] and ad.spec_stats["seeds"] == 1  # (the primary's second base; both live: seeded)
    plain([0, 1])
    rounds(4, [0, 1], 3)  # the primary: 3 drafts each, all right (good = K)
    assert be.spec.steps[0].counters["steps"] == 4 and ad.spec_stats["fused"] == 4
    pre(2)  # -> generator slot 2 (the alternate's first other base); 3 live: no step's exact bases -> no seed
    assert ad._alias[2] == 2 and ad.spec_stats["seeds"] == 1 and getattr(gen._binding(kv), "spec", None) is None
    pre(3)  # -> slot 6: the 4 live users are the alternate's bases -> the seed selects it
    assert ad._alias[3] == 6 and ad.spec_stats["seeds"] == 2 and gen.spec_current() == 1
    plain([0, 1, 2, 3])
    rounds(3, [0, 1, 2, 3], 1)  # the alternate: 1 draft per row in the K0 block
    assert be.spec.steps[1].counters["steps"] == 3 and ad.spec_stats["fused"] == 7
    ad.release_request(3)  # (speculating: leave, not abort)
    ad.release_request(2)
    assert getattr(gen._binding(kv), "spec", None) is None
    plain([0, 1])
    rounds(2, [0, 1], 3)  # back on the primary (the proposal selects it)
    assert gen.spec_current() == 0 and be.spec.steps[0].counters["steps"] == 6 and gen.counters["spec_selects"] == 2
    assert ad.spec_stats["fallback"] == 0
    for i in range(4):
        assert st[i] == want(ps[i], st[i]), (i, st[i][:12])


class _MockSpecMultiPartial(_MockSpecMulti):
    """_MockSpecMulti with GLM53_SPEC_PARTIAL (kernels/spec/spec_ctl.cpp SPEC_PARTIAL): a group whose base's plain
    position is < 0 at the bootstrap is inactive for the whole speculation -- record c -1, no token, state or position
    written for its base -- while, as on the device (the chained KDA verify writes every row of a group: an inactive
    group's rows take its base's state), its CANDIDATE rows are clobbered: a live request there would be corrupted"""

    partial = True

    def __init__(self, be, K: int, G: int, good):
        super().__init__(be, K, G, good)
        self.active, self.inactive = None, []  # (set_active's groups; (step, base) of every inactive group stepped)

    def set_active(self, groups) -> None:
        self.active = tuple(sorted(int(u) for u in groups))

    def _idle_rec(self) -> dict:
        return dict(n=0, c=-1, a=[0] * (self.K + 1), committed=[0], drafts=[0] * self.K, steps=self.counters["steps"])

    def enter(self, rows):
        self.set_page_tables(rows)
        io, self._recs = self.be.io, []
        for u, s in enumerate(self.slots):
            p = int(io.cur_pos.t[s])
            if p < 0:  # BOOT: an inactive group
                self._recs.append(self._idle_rec())
                continue
            b = int(io.tokens.t[s])
            self._drafts[s] = self._make_drafts(b, 0, u)
            self._recs.append(dict(n=0, c=p - 1, a=[b] * (self.K + 1), committed=[b], drafts=list(self._drafts[s]),
                                   steps=0))  # fmt: skip
        assert self.active == tuple(u for u, r in enumerate(self._recs) if r["c"] >= 0), (self.active, self._recs)
        self.counters["enters"] += 1

    def step(self, blocking: bool = False):
        io, recs = self.be.io, []
        self.counters["steps"] += 1
        for u, s in enumerate(self.slots):
            prev = self._recs[u]
            if prev["c"] < 0:  # an inactive group: its base untouched, its candidate rows take the base's state
                for j in range(1, self.K + 1):
                    self.be.kv.state[s + j] = list(self.be.kv.state[s])
                self.inactive.append((self.counters["steps"], s))
                recs.append(self._idle_rec())
                continue
            st, c = self.be.kv.state[s], prev["c"] + prev["n"] + 1
            rows = [int(io.tokens.t[s])] + self._drafts[s]
            assert len(st) == c, f"user {u}: verify at {c}, its state holds {len(st)} tokens"
            a = [(r + 1) % VOCAB for r in rows]
            n = 0
            for j in range(1, self.K + 1):
                if a[j - 1] != rows[j]:
                    break
                n = j
            st.extend(rows[: n + 1])
            io.tokens.t[s] = a[n]
            self._drafts[s] = self._make_drafts(a[n], self.counters["steps"], u)
            recs.append(dict(n=n, c=c, a=a, committed=a[: n + 1], drafts=list(self._drafts[s]),
                             steps=self.counters["steps"]))  # fmt: skip
        self._recs = recs

    def leave(self) -> list:
        recs = self.read()
        pos = self.be.io.cur_pos.t.clone()
        for s, rec in zip(self.slots, recs):
            if rec["c"] >= 0:  # (an inactive group: nothing committed, no position)
                pos[s] = rec["c"] + rec["n"] + 1
        self.be.io.cur_pos.t = pos
        self.counters["leaves"] += 1
        return recs


class _MockSpecMultiPartialBackend(_MockSpecMultiBackend):
    def allocate_decode_io(self, kv):
        io = super().allocate_decode_io(kv)
        self.spec = _MockSpecMultiPartial(self, self.K, self.G, self.good)
        return io


def _partial_adapter(mb: int, K: int, G: int, good):
    be = _MockSpecMultiPartialBackend(mb, K, G, good)
    gen = Glm53Generator(be, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=mb, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, mb, SEQ, spec=True)
    kv = ad.allocate_kv_cache((mb * SEQ // BLOCK_SIZE + mb, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=mb, num_blocks=BPS, can_sample_on_device=True)
    assert ad._spec_partial()
    return ad, be, gen, kv


@pytest.mark.parametrize("thru", ["1", "0"])
def test_served_spec_through_a_partial_prefill(monkeypatch, thru):
    """GLM53_SPEC_THROUGH_PREFILL (F46: while a request was partway through a multi-chunk prefill every interleaved
    decode step ran plain for every user -- the binding's live slots included the prefilling one, which the plugin's
    decode rows never do, so every proposal declined): on the partial-groups step (G 4, K 1 on the 8-slot mock: bases
    0 / 2 / 4 / 6) users A and B keep speculating while C prefills in three engine steps (its intermediate chunks
    host-routed, as the plugin sends them). C's generator slot (base 4) is an idle group of every entry: never active,
    its state untouched (each continuation chunk finds exactly its prefix: the mock asserts it), no proposal declines.
    C's last chunk seeds A, B and C together and all three speculate. Every stream equals plain greedy.
    GLM53_SPEC_THROUGH_PREFILL=0 keeps the old gate: every proposal declines while C is mid-prefill. Either way a request
    mid-prefill on a CANDIDATE row (G 2 on 4 slots: base 0's row 1) declines every proposal -- the chained verify would
    write its state -- and the generator refuses such an idle slot."""
    monkeypatch.setenv("GLM53_SPEC_THROUGH_PREFILL", thru)
    mb, K = 8, 1
    ad, be, gen, kv = _partial_adapter(mb, K, 4, lambda step: K if step % 3 else 0)
    assert gen.spec_slots() == (0, 2, 4, 6) and ad.spec_thru_pf == (thru == "1")
    ps = [_prompt(100, 7), _prompt(90, 20), _prompt(300, 33)]
    tab = torch.zeros(mb, BPS, dtype=torch.int32)
    for i in range(3):
        bl = _blocks(i, 400)
        tab[i, : len(bl)] = torch.tensor(bl, dtype=torch.int32)
    st, nxt = {}, {}
    want = lambda p, s: [(p[-1] + 1 + i) % VOCAB for i in range(len(s))]  # noqa: E731  (plain greedy: next = t + 1)

    def chunk(i, start, end):  # one engine step of prompt i: intermediate chunks host-routed, the last device-sampled
        last = end == len(ps[i])
        out = ad.prefill_forward(_history([ps[i][:end]]), page_table=tab[i : i + 1], kv_cache=kv, prompt_lens=[end],
                                 start_pos=[start], empty_slots=[i], sampling_params=_greedy(1) if last else None)  # fmt: skip
        if last:
            st[i], nxt[i] = [int(out[0][0, 0])], end

    def plain(users):
        tok, pos = torch.zeros(mb, 1, dtype=torch.int64), torch.full((mb,), -1, dtype=torch.int64)
        for i in users:
            tok[i, 0], pos[i] = st[i][-1], nxt[i]
        ids = ad.decode_forward(tok, pos, page_table=tab, kv_cache=kv, sampling_params=_greedy(mb))
        for i in users:
            st[i].append(int(ids[i, 0]))
            nxt[i] += 1

    def blk(users):  # the committed block (users: plugin row -> its committed tokens, the last at nxt - 1 + 1)
        ct = torch.full((mb, 1 + K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        cp = torch.full((mb, 1 + K), -1, dtype=torch.int32)
        ac = torch.ones(mb, dtype=torch.int32)
        for i, toks in users.items():
            first = nxt[i] - len(toks) + 1
            ct[i, : len(toks)] = torch.tensor(toks, dtype=torch.int32)
            cp[i, :] = torch.arange(first, first + 1 + K)
            ac[i] = len(toks)
        return ct, cp, ac

    def rounds(n, users, k):  # propose -> verify (a declined proposal: the plugin's ordinary decode)
        ct, cp, ac = blk({i: [st[i][-1]] for i in users})
        for _ in range(n):
            do = ad.propose_draft_tokens(K, ct, cp, ac)
            assert do.num_valid.tolist() == [k if i in users else 0 for i in range(mb)], do.num_valid.tolist()
            if not k:
                plain(users)
                ct, cp, ac = blk({i: [st[i][-1]] for i in users})
                continue
            tok = torch.zeros(mb, 1 + K, dtype=torch.int32)
            pos = torch.full((mb, 1 + K), -1, dtype=torch.int32)
            nv = torch.zeros(mb, dtype=torch.int32)
            dr = torch.full((mb, K), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            for i in users:
                tok[i, 0], pos[i, 0], nv[i] = st[i][-1], nxt[i], k
                for j in range(k):
                    x = int(do.draft_token_ids[i, j])
                    tok[i, 1 + j], pos[i, 1 + j], dr[i, j] = x, nxt[i] + 1 + j, x
            out = ad.decode_forward(tok, pos, page_table=tab, kv_cache=kv, sampling_params=_greedy(mb),
                                    num_valid_drafts=nv, accepted_counts=torch.ones(mb, dtype=torch.int32),
                                    spec_mode="argmax_ids")  # fmt: skip
            ids = out.argmax_ids
            assert tuple(ids.shape) == (mb, 1 + K), tuple(ids.shape)
            committed, counts = accept_greedy_walk(ids, dr[:, : int(ids.shape[1]) - 1], nv)
            new = {i: [int(v) for v in committed[i, : int(counts[i])].tolist()] for i in users}
            for i in users:
                st[i].extend(new[i])
                nxt[i] += len(new[i])
            ct, cp, ac = blk(new)

    chunk(0, 0, 100)
    chunk(1, 0, 90)
    assert ad._alias[:2] == [0, 2] and ad.spec_stats["seeds"] == 2 and ad._pf_open == set()
    plain([0, 1])
    rounds(3, [0, 1], K)
    f0, d0, e0 = ad.spec_stats["fused"], ad.spec_stats["declines"], ad.spec_stats["enters"]
    chunk(2, 0, 128)  # C: generator slot 4 (the third base), mid-prefill
    assert ad._alias[2] == 4 and ad._pf_open == {4} and getattr(gen._binding(kv), "spec", None) is None
    for start, end in ((128, 256), (256, 300)):
        plain([0, 1])  # (the step after a prefill: the outstanding verify's ordinary answer)
        rounds(3, [0, 1], K if thru == "1" else 0)
        assert be.kv.state[4] == ps[2][:start]  # (C's state: exactly its prefix)
        chunk(2, start, end)
    assert ad._pf_open == set() and ad.spec_stats["seeds"] == 3  # (C's last chunk seeds A, B and C together)
    if thru == "1":
        assert ad.spec_stats["fused"] == f0 + 6 and ad.spec_stats["declines"] == d0
        assert ad.spec_stats["enters"] == e0 + 2 and ad.spec_stats["thru_pf"] == 2
        assert {s for _, s in be.spec.inactive} == {4, 6}  # (C's base and the empty one: never active while idle)
    else:
        assert ad.spec_stats["fused"] == f0 and ad.spec_stats["declines"] == d0 + 6 and ad.spec_stats["thru_pf"] == 0
    plain([0, 1, 2])
    rounds(4, [0, 1, 2], K)
    for i in range(3):
        assert st[i] == want(ps[i], st[i]), (i, st[i][:12])
    assert be.kv.state[4][: len(ps[2])] == ps[2]
    # ---- a request mid-prefill on a CANDIDATE row: G 2, K 1 on 4 slots (bases 0 / 2), C lands on base 0's row 1
    ad, be, gen, kv = _partial_adapter(B, K, 2, lambda step: K)
    mb, tab = B, tab[:B].clone()
    chunk(0, 0, 100)
    chunk(1, 0, 90)
    plain([0, 1])
    rounds(2, [0, 1], K)
    chunk(2, 0, 128)
    assert ad._alias[:3] == [0, 2, 1] and ad._pf_open == {1}
    plain([0, 1])
    rounds(2, [0, 1], 0)  # (live 0 / 1 / 2: slot 1 is no base -- every proposal declines)
    assert be.kv.state[1] == ps[2][:128]
    with expect_error(ValueError, "must sit on other group bases"):
        gen.spec_enter(kv, slots=[0, 2], idle=[1])
    chunk(2, 128, 256)  # (its state intact: the continuation finds its prefix)
    for i in range(2):
        assert st[i] == want(ps[i], st[i]), (i, st[i][:12])
    assert "ttnn" not in sys.modules


def test_spec_enter_idle_needs_partial_groups():
    """Glm53Generator.spec_enter(idle=): only the partial-groups step keeps a live slot out (the fixed-G v1 step runs
    every group: an idle request's rows would be verified), only on another group base of the step, and never on the
    single-user step; spec_room(slots=) checks the speculating users only (not an idle one near the context end)"""
    backend, gen = _spec_multi_gen(1, 2, 4)
    kv = gen.allocate_kv_cache(4, SEQ)
    b = gen._binding(kv)
    for s in (0, 1, 2):
        b.slots[s].active, b.slots[s].seq_len = True, 10
    with expect_error(ValueError, "must sit on other group bases"):
        gen.spec_enter(kv, slots=[0, 2], idle=[1])  # (no partial groups, and slot 1 is base 0's candidate row)
    with expect_error(ValueError, "must be exactly the live slots"):
        gen.spec_enter(kv, slots=[0, 2])
    assert backend.spec.counters["enters"] == 0
    b.slots[2].seq_len = SEQ - 2
    assert not gen.spec_room(kv, step=0) and gen.spec_room(kv, step=0, slots=[0])
    assert not gen.spec_room(kv, step=0, slots=[0, 2])
    _, gen1 = _spec_gen(1)
    kv1 = gen1.allocate_kv_cache(B, SEQ)
    with expect_error(ValueError, "need the multi-user step's partial groups"):
        gen1.spec_enter(kv1, idle=[1])
    assert "ttnn" not in sys.modules


def test_served_mtp_fill_takes_the_next_chunk_token():
    """The MTP-cache fill of a chunked prompt (the review's F-ACC): MTP position p takes the token at p + 1, so a
    chunk's last row needs the NEXT chunk's first token. Armed with the chunk's own history only (the plugin's tokens
    stop at the chunk end) the fill skipped that row's token -- mtp_fill_inputs wrote it with token 0 into a real page
    and no later chunk rewrote it: rows 127 / 255 of a 300-token prompt in 128-token chunks. With the plugin's
    next_token_ids (model_capabilities prefill_next_token_ids) every row of an intermediate chunk is filled with its
    true next token, the final chunk stops at L - 2 (L - 1 is the seed's); a call whose history reaches past a row's
    end (an exact split's inner call, a shorter row of a multi-row step) supplies it too. mtp_fill_ids: pure"""
    assert GLM53StackForCausalLM.model_capabilities["prefill_next_token_ids"] == gvs.SPEC
    row = torch.arange(10, 20)
    assert mtp_fill_ids(row, 4, final=False, next_token=77).tolist() == [10, 11, 12, 13, 77]
    assert mtp_fill_ids(row, 4, final=False, next_token=-1).tolist() == [10, 11, 12, 13]  # (the plugin: final)
    assert mtp_fill_ids(row, 4, final=False).tolist() == [10, 11, 12, 13, 14]  # (the call's own history)
    assert mtp_fill_ids(row, 4, final=True).tolist() == [10, 11, 12, 13]  # (a device-sampled call: final rows)
    assert mtp_fill_ids(row, 10, final=False).tolist() == list(range(10, 20))  # (nothing past the end)
    assert mtp_fill_ids(row, 4, final=False, next_token=64, vocab=64).tolist() == [10, 11, 12, 13]  # (no such token)
    prompt, ends = _prompt(300, 9), [128, 256, 300]

    def fill(pass_next: bool, width=None):
        ad, backend, _gen, kv = _spec_adapter(K=3)
        row = _blocks(0, 400)
        start = 0
        for end in ends:
            last = end == len(prompt)
            kw = {"next_token_ids": torch.tensor([-1 if last else prompt[end]])} if pass_next else {}
            hist = prompt[: end if width is None or last else min(len(prompt), end + width)]
            ad.prefill_forward(_history([hist]), page_table=_table([row[: -(-end // BLOCK_SIZE)]], 1), kv_cache=kv,
                               prompt_lens=[end], start_pos=[start], empty_slots=[0],
                               sampling_params=_greedy(1) if last else None, **kw)  # fmt: skip
            start = end
        assert ad.spec_stats["seeds"] == 1
        return backend.mtp_spans

    # the plugin without prefill_next_token_ids: the chunk-end rows 127 / 255 stay unfilled (the old behavior)
    assert [s[:2] for s in fill(False)] == [(0, 127), (128, 127), (256, 43)]
    spans = fill(True)
    assert [s[:2] for s in spans] == [(0, 128), (128, 128), (256, 43)]
    assert spans[0][2] == prompt[1:129] and spans[1][2] == prompt[129:257] and spans[2][2] == prompt[257:300]
    assert fill(False, width=64) == spans  # (the history reaches past each intermediate end)
    assert "ttnn" not in sys.modules


class _MockSpecPrefixBackend(_MockPrefixBackend, _MockSpecBackend):
    """the prefix cache's device side (snapshot pool, sharded pool, page contents) + the speculative step, whose
    committed rows also land in the slot's pages (the verify writes their latents)"""

    def __init__(self, max_batch: int, K: int, good=lambda step: 0):
        _MockSpecBackend.__init__(self, max_batch, K, good)
        self.state_io = []

    def allocate_decode_io(self, kv):
        io = super().allocate_decode_io(kv)
        step = self.spec.step

        def paged_step(blocking: bool = False):
            n0 = len(self.kv.state[0])
            step(blocking)
            self._page_write(self.kv, 0, n0, self.kv.state[0][n0:])

        self.spec.step = paged_step
        return io


def test_served_spec_with_the_prefix_cache(monkeypatch):
    """GLM53_STACK_PREFIX + GLM53_STACK_VLLM_SPEC: a prefix-cache attach / snapshot never meets a speculating slot 0 --
    the generator refuses a prefill (its pool copies) and a pool restore while slot 0 speculates, and the adapter leaves
    the speculation before every prefill, so a new request's snapshot (snapshot_after) follows spec_leave; a request
    that ATTACHED a snapshot entry never speculates (the MTP cache never saw its [0, start)): no MTP fill, no seed, its
    proposals decline and it decodes plainly; a fresh request in the same generator slot speculates again"""
    monkeypatch.setattr(gvs, "PREFIX_CACHE", True)
    monkeypatch.setenv("GLM53_SPEC_ON_SHARD", "1")  # (the prefix mock's sequence-sharded pool)
    K = 3
    backend = _MockSpecPrefixBackend(B, K, good=lambda step: K)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ, spec=True)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    pc, sc = ad.prefix, backend.spec.counters
    doc = _prompt(256, 5)
    a, b = doc + _prompt(40, 50), doc + _prompt(30, 20)
    row_a = _blocks(0, 400)
    # ---- A: three engine steps (snapshots at 128 / 256), the MTP fill of every chunk, the seed
    stream = [_prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, len(a)])]
    assert backend.state_io == [(0, 0, None, 0), (0, 128, None, 1), (0, 256, None, None)]
    assert [s[:2] for s in backend.mtp_spans] == [(0, 127), (128, 127), (256, len(a) - 257)]
    assert ad.spec_stats["seeds"] == 1 and ad._no_spec == set()
    table, pos = _table([row_a], B), len(a)
    ids = ad.decode_forward(torch.tensor([[stream[-1]], [0], [0], [0]]), torch.tensor([pos, -1, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    stream.append(int(ids[0, 0]))
    pos += 1
    do = ad.propose_draft_tokens(K, *_block({0: ([stream[-1]], pos)}, K))
    assert do.num_valid.tolist() == [K, 0, 0, 0] and gen._binding(kv).spec is not None
    # ---- while slot 0 speculates the generator writes no slot state: no prefill (attach / snapshot), no pool restore
    with expect_error(RuntimeError, "prefill_forward while slot 0 speculates"):
        gen.prefill_forward(torch.tensor([[1]]), page_table=_table([_blocks(2, 64)], 1), kv_cache=kv, prompt_lens=[1],
                            empty_slots=[1], snapshot_after={0: 2})  # fmt: skip
    with expect_error(RuntimeError, "restore_slot while slot 0 speculates"):
        gen.restore_slot(kv, 0, 2)
    ids, (committed, counts) = _verify(ad, kv, K, {0: (stream[-1], pos, [int(v) for v in do.draft_token_ids[0]])},
                                       table)  # fmt: skip
    assert int(counts[0]) == K + 1 and ad.spec_stats["fused"] == 1
    new = [int(v) for v in committed[0].tolist()]
    do = ad.propose_draft_tokens(K, *_block({0: (new, pos + 1)}, K))
    stream += new
    pos += len(new)
    drafts = [int(v) for v in do.draft_token_ids[0].tolist()]
    assert do.num_valid.tolist() == [K, 0, 0, 0]
    # ---- C arrives while A speculates: its first chunk ends on the grid (a snapshot) -- the speculation is left first
    leaves = sc["leaves"]
    c = _prompt(128, 40) + _prompt(20, 3)
    ad.prefill_forward(_history([c[:128]]), page_table=_table([_blocks(2, 128)], 1), kv_cache=kv, prompt_lens=[128],
                       start_pos=[0], empty_slots=[1], sampling_params=None)  # fmt: skip
    assert sc["leaves"] == leaves + 1 and gen._binding(kv).spec is None
    assert backend.state_io[-1] == (ad._alias[1], 0, None, 2) and pc.stats["snapshots"] == 3
    ids, (committed, counts) = _verify(ad, kv, K, {0: (stream[-1], pos, drafts)}, table)  # answered by a plain decode
    assert tuple(ids.shape) == (B, 1) and ad.spec_stats["fallback"] == 1
    stream.append(int(ids[0, 0]))
    pos += 1
    do = ad.propose_draft_tokens(K, *_block({0: ([stream[-1]], pos)}, K))
    assert do.num_valid.tolist() == [0] * B  # C is live (mid-prefill): A is not alone
    assert stream == [(a[-1] + 1 + i) % VOCAB for i in range(len(stream))]
    ad.release_request(1)
    ad.release_request(0)
    # ---- B attaches A's snapshot at 256 in generator slot 0: never speculates (no MTP fill, no seed, declines)
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256
    n_spans, seeds = len(backend.mtp_spans), ad.spec_stats["seeds"]
    row_b = row_a[:4] + _blocks(1, 400)[4:]
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    assert ad._alias[2] == 0 and backend.state_io[-1] == (0, 256, 1, None) and ad._no_spec == {0}
    assert len(backend.mtp_spans) == n_spans and ad.spec_stats["seeds"] == seeds
    tb, s_b, pos_b = _table([[], [], row_b], B), [int(out[0][0, 0])], len(b)
    for _ in range(3):
        do = ad.propose_draft_tokens(K, *_block({2: ([s_b[-1]], pos_b)}, K))
        assert do.num_valid.tolist() == [0] * B
        ids = ad.decode_forward(torch.tensor([[0], [0], [s_b[-1]], [0]]), torch.tensor([-1, -1, pos_b, -1]),
                                page_table=tb, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
        s_b.append(int(ids[2, 0]))
        pos_b += 1
    assert s_b == [(b[-1] + 1 + i) % VOCAB for i in range(len(s_b))] and gen._binding(kv).spec is None
    enters = ad.spec_stats["enters"]
    ad.release_request(2)
    assert ad._no_spec == set()
    # ---- a fresh text request in generator slot 0 speculates again
    d = _prompt(100, 9)
    out = ad.prefill_forward(_history([d]), page_table=_table([_blocks(3, 400)], 1), kv_cache=kv, prompt_lens=[100],
                             start_pos=[0], empty_slots=[3], sampling_params=_greedy(1))  # fmt: skip
    assert ad._alias[3] == 0 and backend.mtp_spans[-1][:2] == (0, 99) and ad.spec_stats["seeds"] == seeds + 1
    ids = ad.decode_forward(torch.tensor([[0], [0], [0], [int(out[0][0, 0])]]), torch.tensor([-1, -1, -1, 100]),
                            page_table=_table([[], [], [], _blocks(3, 400)], B), kv_cache=kv,
                            sampling_params=_greedy(B))  # fmt: skip
    do = ad.propose_draft_tokens(K, *_block({3: ([int(ids[3, 0])], 101)}, K))
    assert do.num_valid.tolist() == [0, 0, 0, K] and ad.spec_stats["enters"] == enters + 1
    ad.release_request(3)
    assert sc.get("aborts") == 1 and "ttnn" not in sys.modules


def test_served_spec_declines_image_requests():
    """an image request (the multimodal class) never speculates -- the MTP-cache fill embeds token ids, so its image
    positions would carry the placeholder's embedding, not the feature rows: no MTP fill, no seed, its proposals
    decline and it decodes plainly; a text request in the same generator slot afterwards speculates"""
    K = 3
    backend = _MockSpecBackend(B, K, good=lambda step: K)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=(STOP,)), max_batch=B, max_seq_len=SEQ)
    ad = declare_multimodal_class(None)(gen, B, SEQ, spec=True)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    ad.mm_encoder = lambda items: [_img_rows(torch.arange(it.offset, it.offset + it.n_tokens)) for it in items]
    lo, n_img = 16, 32
    img = _mm_prompt(100, lo, lo + n_img, 9)
    table = _table([_blocks(0, 400)], B)
    out, _ = ad.prefill_forward(_history([img]), page_table=table[:1], kv_cache=kv, prompt_lens=[100], start_pos=[0],
                                empty_slots=[0], sampling_params=_greedy(1), **_items(lo, n_img))  # fmt: skip
    assert kv.state[0] == _ledger(img, lo, lo + n_img) and ad.mm_stats["rows_staged"] == n_img
    assert ad._no_spec == {0} and backend.mtp_spans == [] and ad.spec_stats["seeds"] == 0
    tok, pos = int(out[0, 0]), 100
    for _ in range(3):
        do = ad.propose_draft_tokens(K, *_block({0: ([tok], pos)}, K))
        assert do.num_valid.tolist() == [0] * B
        ids = ad.decode_forward(torch.tensor([[tok], [0], [0], [0]]), torch.tensor([pos, -1, -1, -1]),
                                page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
        assert int(ids[0, 0]) == (tok + 1) % VOCAB
        tok, pos = int(ids[0, 0]), pos + 1
    assert backend.spec.counters["enters"] == 0
    ad.release_request(0)
    # a text request on the multimodal class (pixel_values=[None]): armed, seeded, speculates
    text = _prompt(100, 4)
    table = _table([_blocks(1, 400)], B)
    out, _ = ad.prefill_forward(_history([text]), page_table=table[:1], kv_cache=kv, prompt_lens=[100], start_pos=[0],
                                empty_slots=[0], sampling_params=_greedy(1), pixel_values=[None], image_grid_thw=[None],
                                mm_placeholder_offsets=[None], mm_placeholder_lengths=[None])  # fmt: skip
    assert ad._no_spec == set() and backend.mtp_spans[-1][:2] == (0, 99) and ad.spec_stats["seeds"] == 1
    ids = ad.decode_forward(torch.tensor([[int(out[0, 0])], [0], [0], [0]]), torch.tensor([100, -1, -1, -1]),
                            page_table=table, kv_cache=kv, sampling_params=_greedy(B))  # fmt: skip
    do = ad.propose_draft_tokens(K, *_block({0: ([int(ids[0, 0])], 101)}, K))
    assert do.num_valid.tolist() == [K, 0, 0, 0] and ad.spec_stats["enters"] == 1
    ad.release_request(0)
    assert "ttnn" not in sys.modules and "vllm" not in sys.modules


# ------------------------------------------------- Stage 17: the deferred verify readback (supports_async_spec_decode)
class _MockSpecDeferred(_MockSpec):
    """+ SpecMTP's deferred record read: read_async copies the record at the enqueue point (no event on the CPU
    stand-in), finish_read hands it back -- the generator must go through both"""

    def read_async(self):
        self.counters["async_reads"] = self.counters.get("async_reads", 0) + 1
        return ("rec", dict(self._rec)), []

    def finish_read(self, pending) -> dict:
        assert pending[0] == "rec"
        return pending[1]


class _MockSpecAsyncBackend(_MockSpecBackend):
    def allocate_decode_io(self, kv):
        io = super().allocate_decode_io(kv)
        self.spec = _MockSpecDeferred(self, self.K, self.good)
        return io


def _drain(ad, out):
    """the plugin's async readback of one submission: read_decode_output(async_read=True) at the submission, then
    process_decode_output_host at its finalize -> host ids"""
    h, events = ad.read_decode_output(out, async_read=True)
    assert events == []
    return ad.process_decode_output_host(h, is_tokens=True)


def test_spec_async_capability_knob():
    """supports_async_spec_decode = GLM53_STACK_VLLM_SPEC_ASYNC=1 on a speculating launch that declares the ordinary
    async decode (the resident-input chain); off by default"""
    on = {"GLM53_STACK_VLLM_SPEC": "1", "GLM53_SPEC_K": "3", "GLM53_STACK_VLLM_SPEC_ASYNC": "1"}
    assert gvs.spec_async_enabled(on, async_decode=True)
    assert not gvs.spec_async_enabled(on, async_decode=False)
    assert not gvs.spec_async_enabled(dict(on, GLM53_STACK_VLLM_SPEC_ASYNC="0"), async_decode=True)
    assert not gvs.spec_async_enabled(dict(on, GLM53_STACK_VLLM_SPEC="0"), async_decode=True)
    assert GLM53StackForCausalLM.model_capabilities["supports_async_spec_decode"] is gvs.SPEC_ASYNC is False


def test_served_spec_async_readback():
    """SPEC_DECODE_CONTRACT.md 4c with the deferred readback: the plugin's async verify (read_from_device=False)
    returns a SpecReadback -- the step ran, slot 0's bookkeeping waits -- read_decode_output(async_read=True) hands it
    back with its events and process_decode_output_host completes it into the [B, 1+K] argmax ids before the walk; the
    committed stream equals plain greedy. Ordinary decodes overlap as in a non-speculating launch: a proposal at the
    drain of a step whose successor is already in flight declines (the block's position is behind the slot's), the
    next drain's enters. An entry point that reads slot state completes an outstanding verify itself."""
    K = 3
    backend = _MockSpecAsyncBackend(B, K, good=lambda step: (step % K) + 1)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ, spec=True, spec_async=True)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    prompt, table = _prompt(100, 7), _table([_blocks(0, 400)], B)
    out, _ = ad.prefill_forward(_history([prompt]), page_table=table[:1], kv_cache=kv, prompt_lens=[100],
                                start_pos=[0], empty_slots=[0], sampling_params=_greedy(1))  # fmt: skip
    stream, pos = [int(out[0, 0])], 100
    e0 = backend.spec.counters["enters"]
    assert ad.spec_stats["seeds"] == 1 == e0  # (the MTP seed: enter + leave right after the prefill)
    # ---- two ordinary decodes in flight (a reload step, then a resident chain step submitted before its drain)
    kw = dict(page_table=table, kv_cache=kv, enable_trace=True, sampling_params=_greedy(B), read_from_device=False)
    h1 = ad.decode_forward(torch.tensor([[stream[-1]], [0], [0], [0]]), torch.tensor([pos, -1, -1, -1]), **kw)
    p1 = ad.read_decode_output(h1, async_read=True)
    h2 = ad.decode_forward(torch.zeros(B, 1, dtype=torch.int32), torch.full((B,), -1), reload_inputs=False,
                           reload_page_table=False, reload_sampling_params=False, reset_sampling_state=False,
                           **kw)  # fmt: skip
    p2 = ad.read_decode_output(h2, async_read=True)
    stream.append(int(ad.process_decode_output_host(p1[0], is_tokens=True)[0, 0]))
    do = ad.propose_draft_tokens(K, *_block({0: ([stream[-1]], pos + 1)}, K))  # step 1's drain: step 2 in flight
    assert do.num_valid.tolist() == [0] * B and backend.spec.counters["enters"] == e0
    stream.append(int(ad.process_decode_output_host(p2[0], is_tokens=True)[0, 0]))
    pos += 2
    do = ad.propose_draft_tokens(K, *_block({0: ([stream[-1]], pos)}, K))  # step 2's drain: nothing in flight
    assert do.num_valid.tolist() == [K, 0, 0, 0] and backend.spec.counters["enters"] == e0 + 1
    # ---- deferred verifies
    b = gen._binding(kv)
    for i in range(5):
        drafts = [int(v) for v in do.draft_token_ids[0].tolist()]
        tok = torch.zeros(B, 1 + K, dtype=torch.int32)
        p = torch.full((B, 1 + K), -1, dtype=torch.int32)
        tok[0] = torch.tensor([stream[-1]] + drafts, dtype=torch.int32)
        p[0] = torch.arange(pos, pos + 1 + K)
        nv = torch.tensor([K, 0, 0, 0], dtype=torch.int32)
        vo = ad.decode_forward(tok, p, num_valid_drafts=nv, accepted_counts=torch.ones(B, dtype=torch.int32),
                               spec_mode="argmax_ids", reload_inputs=True, **kw)  # fmt: skip
        rb = vo.argmax_ids
        assert isinstance(rb, gvs.SpecReadback) and ad._spec_pending is rb and rb.ids is None
        assert int(b.slots[0].seq_len) == pos and b.spec_pending is rb.handle  # bookkeeping deferred
        if i == 2:  # a caller that never completes it: the next entry point does (here the proposal)
            ids = None
        else:
            ids = _drain(ad, rb)
            assert tuple(ids.shape) == (B, 1 + K) and ids is rb.ids and ad._spec_pending is None
        nvd = torch.tensor([K, 0, 0, 0])
        if ids is None:
            n_acc = None
        else:
            committed, counts = accept_greedy_walk(ids, tok[:, 1:].clone(), nvd)
            n_acc = int(counts[0])
            new = [int(v) for v in committed[0, :n_acc].tolist()]
        if ids is None:  # (what the walk would have committed: the device's record)
            rec = backend.spec.read()
            new = [int(v) for v in rec["committed"]]
        do = ad.propose_draft_tokens(K, *_block({0: (new, pos + 1)}, K))
        assert do.num_valid.tolist() == [K, 0, 0, 0] and ad._spec_pending is None and b.spec_pending is None
        assert n_acc is None or n_acc == len(new) == ad._spec_n + 1
        stream += new
        pos += len(new)
        assert int(b.slots[0].seq_len) == pos
    assert ad.spec_stats["fused"] == ad.spec_stats["deferred"] == 5 and backend.spec.counters["async_reads"] == 5
    assert stream == [(prompt[-1] + 1 + i) % VOCAB for i in range(len(stream))]  # = plain greedy
    # ---- released with a verify outstanding: completed, then aborted
    drafts = [int(v) for v in do.draft_token_ids[0].tolist()]
    tok = torch.zeros(B, 1 + K, dtype=torch.int32)
    tok[0] = torch.tensor([stream[-1]] + drafts, dtype=torch.int32)
    p = torch.full((B, 1 + K), -1, dtype=torch.int32)
    p[0] = torch.arange(pos, pos + 1 + K)
    vo = ad.decode_forward(tok, p, num_valid_drafts=torch.tensor([K, 0, 0, 0]), spec_mode="argmax_ids",
                           accepted_counts=torch.ones(B, dtype=torch.int32), reload_inputs=True, **kw)  # fmt: skip
    assert isinstance(vo.argmax_ids, gvs.SpecReadback)
    ad.release_request(0)
    assert vo.argmax_ids.ids is not None and backend.spec.counters.get("aborts") == 1 and b.spec_pending is None
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- GLM53_STACK_BUILD_B: a build wider than max_num_seqs
def test_stack_build_b_knob_warmup_and_spec_plan(monkeypatch):
    """GLM53_STACK_BUILD_B: the device build (backend, generator slots, KV cache, prefix pool) 16 rows wide while vLLM
    schedules at most max_num_seqs 8 rows -- the knob (default max_num_seqs; refused below it, past the decode slots or
    not an integer, before any device work), the warm-up accepting the plugin's max_batch_size 8 <= 16 and declaring the
    8-row decode width the plugin pads to, plugin slots < 8 (prefill, decode rows, an 8-wide slot_remap, release), and
    the spec plan: 8 live users x 1 draft on the 16 rows (refused on the default 8-row build); without the knob nothing
    changes (max_num_seqs = the build width)"""
    NB, N = 16, 8
    assert gvs.stack_build_b(N, {}) == N and gvs.stack_build_b(N, {"GLM53_STACK_BUILD_B": "16"}) == NB
    assert (
        gvs.stack_build_b(N, {"GLM53_STACK_BUILD_B": " "}) == N
        and gvs.stack_build_b(N, {"GLM53_STACK_BUILD_B": "8"}) == N
    )
    with expect_error(ValueError, "GLM53_STACK_BUILD_B=4 < max_num_seqs 8"):
        gvs.stack_build_b(N, {"GLM53_STACK_BUILD_B": "4"})
    with expect_error(ValueError, "GLM53_STACK_BUILD_B=33 > 32"):
        gvs.stack_build_b(N, {"GLM53_STACK_BUILD_B": "33"})
    with expect_error(ValueError, "an integer"):
        gvs.stack_build_b(N, {"GLM53_STACK_BUILD_B": "wide"})
    monkeypatch.setenv("GLM53_STACK_BUILD_B", "4")
    with expect_error(ValueError, "< max_num_seqs 8"):  # (before the mesh or any stack module is touched)
        GLM53StackForCausalLM.initialize_vllm_model(None, None, N, SEQ)
    monkeypatch.delenv("GLM53_STACK_BUILD_B")
    with expect_error(ValueError, "max_num_seqs 17 outside"):
        GLM53StackForCausalLM(Glm53Generator(_MockStackBackend(NB), tokenizer=None, max_batch=NB, max_seq_len=SEQ),
                              NB, SEQ, max_num_seqs=17)  # fmt: skip
    # ---- the default: max_num_seqs = the build width, the declared decode width the build's
    ad0, *_ = _adapter()
    assert ad0.max_batch == ad0.max_num_seqs == B and ad0.tt_supported_decode_batch_sizes == (B,)
    # ---- the 16-row build under max_num_seqs 8 (vLLM's block budget for 8 users)
    backend = _MockStackBackend(NB)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=NB, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, NB, SEQ, max_num_seqs=N)
    assert ad.max_batch == NB and ad.max_num_seqs == N
    kv = ad.allocate_kv_cache((N * BPS + N, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    assert ad._alias == list(range(NB)) and gen._binding(kv).max_batch == NB
    for trace in (False, True):  # model_runner.py:4770-4799: the plugin's max_batch_size is its max_num_seqs
        ad.warmup_model_prefill(kv_cache=kv, enable_trace=trace, can_sample_on_device=True)
        ad.warmup_model_decode(kv_cache=kv, enable_trace=trace, max_batch_size=N, num_blocks=BPS,
                               can_sample_on_device=True)  # fmt: skip
    assert ad.tt_supported_decode_batch_sizes == (N,)  # the plugin pads every decode to its 8 rows
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=NB)  # (<= the build: accepted)
    with expect_error(ValueError, "max_batch_size 17 outside"):
        ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=NB + 1)
    assert ad.tt_supported_decode_batch_sizes == (N,)
    # plugin slots 0..7: a request in plugin slot 7, decoded in an 8-row step; slot 8 does not exist for the plugin
    p = _prompt(70, 3)
    row = [1 + j for j in range(-(-200 // BLOCK_SIZE))]
    tab = torch.zeros(N, BPS, dtype=torch.int32)
    tab[7, : len(row)] = torch.tensor(row, dtype=torch.int32)
    out, _ = ad.prefill_forward(_history([p]), page_table=tab[7:8], kv_cache=kv, prompt_lens=[70], start_pos=[0],
                                empty_slots=[7], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[7]
    assert g == 0 and kv.state[g] == p and int(out[0, 0]) == (p[-1] + 1) % VOCAB
    with expect_error(ValueError, r"empty_slots \[8\] outside \[0, 8\)"):
        ad.prefill_forward(_history([p]), page_table=tab[7:8], kv_cache=kv, prompt_lens=[70], start_pos=[0],
                           empty_slots=[8], sampling_params=_greedy(1))  # fmt: skip
    tok, pos = torch.zeros(N, 1, dtype=torch.int32), torch.full((N,), -1, dtype=torch.int32)
    tok[0, 0], pos[0] = int(out[0, 0]), 70
    tab2 = torch.zeros(N, BPS, dtype=torch.int32)
    tab2[0] = tab[7]
    remap = torch.tensor([7, 1, 2, 3, 4, 5, 6, 0], dtype=torch.int32)  # the plugin condenses its row to row 0
    ids = ad.decode_forward(tok, pos, page_table=tab2, kv_cache=kv, sampling_params=_greedy(N), slot_remap=remap,
                            reload_sampling_params=True, reset_sampling_state=True)  # fmt: skip
    assert int(ids[0, 0]) == (int(out[0, 0]) + 1) % VOCAB and ad._alias[0] == g and ad._alias[N:] == list(range(N, NB))
    with expect_error(ValueError, "full-width permutation of the 8 slots"):
        ad.decode_forward(tok, pos, page_table=tab2, kv_cache=kv, sampling_params=_greedy(N),
                          slot_remap=torch.arange(NB, dtype=torch.int32))  # fmt: skip
    with expect_error(ValueError, "16 decode rows exceed the 8 plugin slots"):
        ad.decode_forward(torch.zeros(NB, 1, dtype=torch.int32), torch.full((NB,), -1, dtype=torch.int32),
                          kv_cache=kv, sampling_params=_greedy(NB))  # fmt: skip
    with expect_error(ValueError, r"release_request\(8\) outside the 8 slots"):
        ad.release_request(8)
    ad.release_request(0)
    assert gen.live_slots(kv) == []
    # ---- the spec plan sizes the multi-user caps from the build: 8 live users x 1 draft on the 16 rows
    T = contract_types()
    menv = {"GLM53_STACK_VLLM_SPEC": "1", "GLM53_SPEC_K": "3", "GLM53_SPEC_MULTI": "1"}
    p16 = spec_plan_for(N, 3, T, dict(menv, GLM53_STACK_BUILD_B="16"))
    assert isinstance(p16, T["SpecPlan"]) and p16.k_by_rows == ((4, 3), (8, 1))  # 8 x (1 + 1) = 16 rows
    alt = dict(menv, GLM53_STACK_BUILD_B="16", GLM53_SPEC_MTP_SPARSE="1", GLM53_SPEC_MULTI_ALT="8:1")
    assert spec_plan_for(N, 3, T, alt).k_by_rows == ((4, 3), (8, 1))  # (the served 4 x 3 + 8:1 pair)
    r = spec_plan_for(N, 3, T, menv)  # the default 8-row build: the primary 4 x (3 + 1) = 16 rows do not fit
    assert isinstance(r, T["SpecReject"]) and "16 verify rows > the 8-row build" in r.reason and r.supported_k == (1,)
    small = dict(menv, GLM53_SPEC_MULTI_USERS="2", GLM53_SPEC_MTP_SPARSE="1", GLM53_SPEC_MULTI_ALT="8:1")
    r = spec_plan_for(N, 3, T, small)  # the primary 2 x 4 fits 8 rows, the alternate 8 x 2 does not
    assert isinstance(r, T["SpecReject"]) and "GLM53_SPEC_MULTI_ALT 8:1" in r.reason and r.supported_k == ()
    assert spec_plan_for(N, 3, T, dict(small, GLM53_STACK_BUILD_B="16")).k_by_rows == ((4, 3), (8, 1))
    r = spec_plan_for(N, 3, T, dict(menv, GLM53_STACK_BUILD_B="4"))  # (spec_plan never raises: a refusal)
    assert isinstance(r, T["SpecReject"]) and "< max_num_seqs 8" in r.reason
    assert isinstance(spec_plan_for(N, 3, T, dict(menv, GLM53_STACK_BUILD_B="32")), T["SpecReject"])  # > the 16 W
    lone = {"GLM53_STACK_VLLM_SPEC": "1", "GLM53_SPEC_K": "3"}  # the single-user step: a build of <= 4 rows
    assert spec_plan_for(1, 3, T, dict(lone, GLM53_STACK_BUILD_B="4")).k_by_rows == ()
    assert isinstance(spec_plan_for(1, 3, T, lone), T["SpecReject"])  # (K + 1 = 4 verify rows on a 1-row build)
    assert isinstance(spec_plan_for(4, 3, T, dict(lone, GLM53_STACK_BUILD_B="8")), T["SpecReject"])
    assert "ttnn" not in sys.modules


def test_stack_build_b_served_multi_user_spec_on_spare_slots():
    """GLM53_STACK_BUILD_B = 8 under max_num_seqs 4 (the served 16 under 8, halved): the multi-user step G 4 x K 1
    verifies 8 rows of the 8-row build -- its group bases 0 / 2 / 4 / 6, two of them generator slots past the plugin's 4.
    The 4 requests of plugin slots 0..3 are admitted onto the bases (the alias's spare entries take the generator slots
    they leave), their MTP caches fill, the seed runs once all 4 are live, every verify of the 4 users' blocks runs ONE
    fused step and every stream equals plain greedy; an ordinary decode with the plugin's 4-wide slot_remap moves the
    plugin entries only; a fifth plugin slot does not exist"""
    K, G, NB = 1, 4, 8
    backend = _MockSpecMultiBackend(NB, K, G, lambda step: K if step % 3 else 0)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=NB, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, NB, SEQ, spec=True, max_num_seqs=B)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    assert ad.spec_multi_capable and gen.spec_slots() == (0, 2, 4, 6) and ad.tt_supported_decode_batch_sizes == (B,)
    prompts = [_prompt(100 - 5 * i, 7 + 13 * i) for i in range(G)]
    table = _table([_blocks(i, 400) for i in range(G)], B)
    s = []
    for i, p in enumerate(prompts):
        out, _ = ad.prefill_forward(_history([p]), page_table=table[i : i + 1], kv_cache=kv, prompt_lens=[len(p)],
                                    start_pos=[0], empty_slots=[i], sampling_params=_greedy(1))  # fmt: skip
        s.append([int(out[0, 0])])
    assert ad._alias[:B] == [0, 2, 4, 6] and sorted(ad._alias) == list(range(NB))  # users on the bases 4 / 6 too
    assert sorted({sl for sl, *_ in backend.mtp_spans}) == [0, 2, 4, 6] and ad.spec_stats["seeds"] == 1
    pos = [len(p) for p in prompts]
    ids = ad.decode_forward(torch.tensor([[x[-1]] for x in s]), torch.tensor(pos), page_table=table, kv_cache=kv,
                            sampling_params=_greedy(B))  # fmt: skip
    for i in range(G):
        s[i].append(int(ids[i, 0]))
        pos[i] += 1
    ct, cp, ac = _block({i: ([s[i][-1]], pos[i]) for i in range(G)}, K)
    for _ in range(5):
        do = ad.propose_draft_tokens(K, ct, cp, ac)
        assert do.num_valid.tolist() == [K] * G
        d = {i: [int(v) for v in do.draft_token_ids[i].tolist()] for i in range(G)}
        ids, (committed, counts) = _verify(ad, kv, K, {i: (s[i][-1], pos[i], d[i]) for i in range(G)}, table)
        assert tuple(ids.shape) == (B, 1 + K)
        new = {i: [int(v) for v in committed[i, : int(counts[i])].tolist()] for i in range(G)}
        ct, cp, ac = _block({i: (new[i], pos[i] + 1) for i in range(G)}, K)
        for i in range(G):
            s[i] += new[i]
            pos[i] += len(new[i])
    assert ad.spec_stats["fused"] == 5 == backend.spec.counters["steps"] and ad.spec_stats["enters"] == 1
    want = lambda p, x: [(p[-1] + 1 + j) % VOCAB for j in range(len(x))]  # noqa: E731  (plain greedy: t + 1)
    assert all(s[i] == want(prompts[i], s[i]) for i in range(G))
    # an ordinary decode with the plugin's gather: rows 0 / 1 swapped (a 4-wide permutation of its slots)
    spare = ad._alias[B:]
    order = [1, 0, 2, 3]
    ids = ad.decode_forward(torch.tensor([[s[i][-1]] for i in order]), torch.tensor([pos[i] for i in order]),
                            page_table=table[order], kv_cache=kv, sampling_params=_greedy(B),
                            slot_remap=torch.tensor(order, dtype=torch.int32), reload_sampling_params=True,
                            reset_sampling_state=True)  # fmt: skip
    assert ad._alias[:B] == [2, 0, 4, 6] and ad._alias[B:] == spare
    for r, i in enumerate(order):
        s[i].append(int(ids[r, 0]))
    assert all(s[i] == want(prompts[i], s[i]) for i in range(G))
    with expect_error(ValueError, r"empty_slots \[4\] outside \[0, 4\)"):
        ad.prefill_forward(_history([prompts[0]]), page_table=table[0:1], kv_cache=kv, prompt_lens=[95],
                           start_pos=[0], empty_slots=[4], sampling_params=_greedy(1))  # fmt: skip
    for i in range(B):
        ad.release_request(i)
    assert gen.live_slots(kv) == [] and backend.spec.counters.get("aborts") is None
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- F27: no prefill piece past the last resumable start
_F27_CHILD = r"""
import sys
from types import SimpleNamespace
from unittest import mock

sys.modules["ttnn"] = mock.MagicMock()  # prefill_tp's module-level ttnn import only: nothing below calls it
from models.autoports.zai_org_glm_5_3_flash.tt import galaxy_defaults as GD
from models.autoports.zai_org_glm_5_3_flash.tt import prefill_tp as PT
from models.autoports.zai_org_glm_5_3_flash.tt.config import SPARSE_DSA

SEQ = 1 << 20
tp = object.__new__(PT.TPPrefill)  # the served 8x4 TP prefill's PLANNING state only (no device, no trace)
tp.buckets, tp.max_seq, tp.ndev = (2048, 8192), SEQ, 32  # galaxy_defaults: GLM53_PREFILL_BUCKETS 2048,8192
tp.sparse_ts = tuple(int(v) for v in GD.sparse_t(SEQ).split(","))  # the captured sparse windows
tp.dsa, tp.layers, tp.shard = True, [SimpleNamespace(kind=SPARSE_DSA)], object()  # the sharded sparse_dsa pool
tp.start_align, tp.slice_rule, tp.mla_form = 128, True, "mla"
assert max(tp.sparse_ts) == SEQ


def covers(start, n, pieces):
    s = start
    for c in pieces:
        assert tp._plan_ok(s, c), ("a piece without a captured key", s, c, pieces)
        s += c
    assert s == start + n, ("the pieces do not cover the chunk", start, n, pieces)


# the engine kill's replay: 7,999 tokens at 1,040,512 -- the greedy full-bucket walk alone (2,048 x 3) leaves 1,855 rows
# at 1,046,656, past the last start with a key; tail first: the tail at the latest start that plans, the head before it
pieces = tp.clamp_lengths(1040512, 7999)
assert pieces == [2048, 2048, 1920, 1983], pieces
covers(1040512, 7999, pieces)
lens, branch = tp.slice_plan(1040512, 7999, 128)  # (the generator's per-chunk rule: the clamp, then each piece's)
assert lens == pieces and branch == "clamp", (lens, branch)
# slice_plan's guard: the slice rule's [front, rest] split of a chunk whose last row sits on a chip slice boundary would
# start its rest where no key holds it -- the chunk stays whole (it plans); away from the top the split is kept
lens, branch = tp.slice_plan(1040384, 2305, 128)
assert (lens, branch) == ([2305], "unavoidable"), (lens, branch)
covers(1040384, 2305, lens)
lens, branch = tp.slice_plan(65536, 2305, 128)
assert (lens, branch) == ([128, 2177], "split"), (lens, branch)
covers(65536, 2305, lens)
# the last resumable start: the top window - the smallest sequence-parallel bucket (the adapter keeps every start there)
assert tp.resume_max() == SEQ - 2048 == 1046528
assert tp._plan_ok(1046528, 2048) and not tp._plan_ok(1046656, 1)
assert not tp._plan_ok(1046912, 87)  # the served exact split's 87-row tail: no bucket has a key there
print("F27_OK")
"""


def _stub_child(code: str, ok: str):
    """run `code` in a child python with a stub ttnn (MagicMock: the module-level imports only; this process stays
    ttnn-free) from the repository root; assert it printed `ok`"""
    root = Path(__file__).resolve().parents[4]
    env = dict(os.environ, PYTHONPATH=str(root))
    for k in [k for k in env if k.startswith("GLM53_")]:
        env.pop(k)
    r = subprocess.run([sys.executable, "-c", code], cwd=str(root), env=env, capture_output=True, text=True,
                       timeout=300)  # fmt: skip
    assert r.returncode == 0 and ok in r.stdout, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
    return r


def test_f27_prefill_planning_never_starts_a_piece_past_resume_max():
    """F27 (2026-11-14, the served 1 M engine kill: 'chunk [1046912, 1048960) past the captured sparse windows'):
    TPPrefill's planning on the served 8x4 state (buckets 2,048 / 8,192, the 1 M window ladder, 32 chips) -- the
    replay of 7,999 tokens at 1,040,512 is planned tail-first into pieces that all have a captured key and cover the
    chunk exactly; slice_plan keeps a chunk whole when its slice-rule split would start the rest without a key;
    resume_max = the top window - the smallest sequence-parallel bucket. prefill_tp imports ttnn at module level: a
    child with a stub ttnn runs it"""
    _stub_child(_F27_CHILD, "F27_OK")
    assert "ttnn" not in sys.modules


class _MockPrefixResumeBackend(_MockPrefixBackend):
    """+ StackBackend.prefill_resume_max (TPPrefill.resume_max, F27): the last plannable prefill chunk start"""

    def prefill_resume_max(self):
        return 300


def test_prefix_cache_caps_at_the_last_resumable_start(monkeypatch):
    """F27, the adapter's caps (generator_vllm_stack.StackPrefixCache): with the backend's prefill_resume_max (300 ->
    256 on the snapshot grid) no snapshot is taken past it (snapshots_unresumable: a hit there would leave a
    continuation no bucket plans), a confirmed hit is capped at it (confirm_capped: vLLM's 384 -> the entry at 256,
    then attached), an exact split's tail starts at it (exact_capped: 256, not 384); below it nothing changes"""
    ad, backend, gen, kv = _prefix_adapter(monkeypatch, _MockPrefixResumeBackend)
    pc = ad.prefix
    assert pc.resume_max == 256
    a = _prompt(450, 11)
    row_a = _blocks(0, len(a))
    assert _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, 384, len(a)]) == (a[-1] + 1) % VOCAB
    assert sorted(e.n_tokens for e in pc.store.entries.values()) == [128, 256]  # nothing at 384 (> 256)
    assert pc.stats["snapshots_unresumable"] == 1 and pc.stats["snapshots"] == 2
    ad.release_request(0)
    b = a[:400] + _prompt(30, 40)
    assert ad.confirm_prefix_cache_hit(b, [], 384) == 256 and pc.stats["confirm_capped"] == 1
    assert ad.confirm_prefix_cache_hit(b, [], 255) == 128 and pc.stats["confirm_capped"] == 1  # (below: as before)
    assert ad.confirm_prefix_cache_hit(b, [], 384) == 256
    row_b = row_a[:4] + _blocks(1, len(b))[4:]
    n = len(backend.state_io)
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[1], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[1]
    assert backend.state_io[n][1:3] == (256, pc.store.entries[pc.store.key(a, 256)].device_entry)  # attached at 256
    assert kv.state[g] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB and pc.stats["attaches"] == 1
    ad.release_request(1)
    pc.exact, pc.exact_min = True, 64
    assert pc.exact_splits([128], [450]) == {0: 256} and pc.stats["exact_capped"] == 1  # (the tail [256, 450))
    assert pc.exact_splits([0], [200]) == {0: 128} and pc.stats["exact_capped"] == 1
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- keep-prev (GLM53_STACK_PREFIX_KEEP_PREV)
def test_prefix_keep_prev_attach_makes_the_kept_snapshot_ordinary(monkeypatch):
    """GLM53_STACK_PREFIX_KEEP_PREV: a kept (superseded) snapshot that a new turn attaches is that turn's base, no
    longer a kept previous -- kept_prev cleared, its chain link dropped -- so after the turn it is an LRU entry: not the
    next device victim ahead of an older entry (the eviction order: dominated, kept, then least recently used)"""
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    pc = ad.prefix
    pc.head, pc.keep_prev = 128, True
    e1 = _prompt(300, 101)
    row_e = [1 + 3 * BPS + j for j in range(8)]
    _prefill_steps(ad, kv, e1, row_e, slot=3, ends=[128, 256, len(e1)])
    ad.release_request(3)
    k128, k256 = pc.store.key(e1, 128), pc.store.key(e1, 256)
    e2 = e1 + _prompt(150, 7)  # its next turn: attaches 256, snapshots 384 -> the 256 is KEPT
    assert ad.confirm_prefix_cache_hit(e2, [], 256) == 256
    start = 256
    for end in (384, len(e2)):
        ad.prefill_forward(_history([e2[:end]]), page_table=_table([row_e[: -(-end // BLOCK_SIZE)]], 1), kv_cache=kv,
                           prompt_lens=[end], start_pos=[start], empty_slots=[3],
                           sampling_params=_greedy(1) if end == len(e2) else None)  # fmt: skip
        start = end
    ad.release_request(3)
    e256 = pc.store.entries[k256]
    assert e256.kept_prev and pc.prev_of == {pc.store.key(e2, 384): k256}
    # a turn that rewrites e2's tail (shares 300 < 384 tokens) resumes at the kept 256 and ends below the next grid
    # boundary (no snapshot of its own): the 256 is its base now
    e3 = e2[:300] + _prompt(60, 111)
    row_e3 = row_e[:4] + [1 + 3 * BPS + 20 + j for j in range(4)]
    assert ad.confirm_prefix_cache_hit(e3, [], 256) == 256
    out = ad.prefill_forward(_history([e3]), page_table=_table([row_e3[: -(-len(e3) // BLOCK_SIZE)]], 1), kv_cache=kv,
                             prompt_lens=[len(e3)], start_pos=[256], empty_slots=[3], sampling_params=_greedy(1))  # fmt: skip
    assert int(out[0][0, 0]) == (e3[-1] + 1) % VOCAB and pc.stats["keep_prev_hits"] == 1
    assert not e256.kept_prev and k256 not in pc.prev_of.values()
    ad.release_request(3)
    # the pool (3 entries: 128, 384, 256) is full: the next victim is the least recently used (the head's 128), not the
    # 256 the last turn read
    victim = pc.store._device_victim()
    assert victim.key == k128 and not any(e.kept_prev or e.dominated for e in pc.store.entries.values())
    ent = pc.store.reserve_device()
    assert ent is not None and k128 not in pc.store.entries and k256 in pc.store.entries
    pc.store.unreserve(ent)
    assert "ttnn" not in sys.modules


class _MockPrefix5Backend(_MockPrefixBackend):
    snapshot_entries = 5


def test_prefix_keep_prev_tail_path_releases_two_back(monkeypatch):
    """GLM53_STACK_PREFIX_KEEP_PREV on the TAIL path (one long prompt's own chunk-end snapshots above the head): each
    commit keeps the snapshot it supersedes (not evicted: tail_superseded 0) and releases the one kept two back
    (dominated: keep_prev_released) -- the chain holds ONE prev_of link; the victim order is the released one first,
    then the kept one"""
    ad, backend, gen, kv = _prefix_adapter(monkeypatch, _MockPrefix5Backend)
    pc = ad.prefix
    pc.head, pc.keep_prev = 128, True
    d = _prompt(512, 61) + _prompt(12, 70)
    _prefill_steps(ad, kv, d, _blocks(0, len(d)), slot=0, ends=[128, 256, 384, 512, len(d)])
    k = {n: pc.store.key(d, n) for n in (128, 256, 384, 512)}
    ent = pc.store.entries
    assert all(k[n] in ent for n in k) and pc.stats["snapshots"] == 4
    assert pc.stats["keep_prev_kept"] == 2 and pc.stats["keep_prev_released"] == 1 and pc.stats["tail_superseded"] == 0
    assert pc.prev_of == {k[512]: k[384]}  # one link: 384 kept for 512; the 256 kept for 384 was released
    assert ent[k[384]].kept_prev and not ent[k[384]].dominated
    assert ent[k[256]].dominated and not ent[k[128]].kept_prev and not ent[k[128]].dominated
    assert pc.tail == {ad._alias[0]: k[512]}
    ad.release_request(0)
    assert pc.store._device_victim().key == k[256]  # the released (dominated) one, then the kept one
    pc.store.evict(k[256])
    assert pc.store._device_victim().key == k[384]
    assert "ttnn" not in sys.modules


def test_prefix_cache_fast_hash_round_trip(monkeypatch):
    """GLM53_PREFIX_FAST_HASH: confirm hashes an int64 copy of the plugin's prompt list, plan / commit hash the int32
    rows of the prefill's token tensor (_ids: the row view) -- the same int32 bytes, the same keys: the snapshot /
    confirm / attach round trip hits exactly as under the text keys (and the two schemes' keys differ)"""
    monkeypatch.setenv("GLM53_PREFIX_FAST_HASH", "1")
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    pc = ad.prefix
    assert pc.store.fast
    doc = _prompt(256, 5)
    a, b = doc + _prompt(40, 50), doc + _prompt(30, 20)
    t32 = _history([a])[0]
    assert t32.dtype == torch.int32 and pc._ids(t32[:256]) is not None
    k32 = pc.store.key(pc._ids(t32[:256]), 256)
    assert k32 == pc.store.key(torch.tensor(a, dtype=torch.int64)[:256], 256) == pc.store.key(a, 256)
    from models.autoports.zai_org_glm_5_3_flash.tt.prefix_cache import prefix_key

    assert k32 != prefix_key(a, 256, pc.store.sig, pc.store.boundary, fast=False)
    row_a = _blocks(0, len(a))
    _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, len(a)])
    assert pc.stats["snapshots"] == 2 and k32 in pc.store.entries
    ad.release_request(0)
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256 and ad.confirm_prefix_cache_hit(b, [], 200) == 128
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256
    row_b = row_a[:4] + _blocks(1, len(b))[4:]
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[2]
    assert backend.state_io[-1] == (g, 256, pc.store.entries[k32].device_entry, None)  # attached, the suffix only
    assert kv.state[g] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB
    assert pc.stats["attaches"] == 1 and pc.stats["fallbacks"] == 0 and pc.report()["hits"] == 1
    assert "ttnn" not in sys.modules


@pytest.mark.parametrize("fast", [False, True])
def test_prefix_cache_running_keys_equal_the_confirm_keys(monkeypatch, fast):
    """F46 running keys (prefix_cache.KeyChain via StackPrefixCache._key): plan keys each chunk end by extending the
    generator slot's chain -- every token hashed once (store stats chain_tokens: 384, not 128 + 256 + 384) -- and
    the keys equal store.key over the whole prefix and the scheduler's confirm-side keys of the plugin's Python list
    (under the fast scheme the int32 history, an int64 copy and the list give the same bytes), across an attach (the
    chain starts at the attach key) and on random chunkings; a chain that is not the row's sequence (another prompt's,
    or ahead of the key) restarts from the seed; an id int32 cannot hold is refused on both sides"""
    import random

    from models.autoports.zai_org_glm_5_3_flash.tt.prefix_cache import prefix_key

    monkeypatch.setenv("GLM53_PREFIX_FAST_HASH", "1" if fast else "0")
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    pc = ad.prefix
    assert pc.store.fast is fast
    a = _prompt(600, 3)
    row_a = _blocks(0, len(a))
    _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, 384, len(a)])
    g = ad._alias[0]
    keys = {e.n_tokens: k for k, e in pc.store.entries.items()}
    assert sorted(keys) == [128, 256, 384]
    n_ids, _fp, hs = pc._prompt_keys(list(a))
    for n, k in keys.items():
        assert k == pc.store.key(a, n) == hs[n // 128 - 1] == pc.store.key(torch.tensor(a, dtype=torch.int64), n)
    assert pc.store.stats["chain_tokens"] == 384 and pc.chains[g].n == 384 and "chain_restarts" not in pc.store.stats
    ad.release_request(0)
    assert g not in pc.chains
    # ---- B attaches A's 256 and continues: its chain starts at the attach key, its snapshots key B's own prefix
    b = a[:300] + _prompt(300, 41)
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256
    row_b = row_a[:4] + _blocks(1, len(b))[4:]
    start = 256
    for end in (384, 512, len(b)):
        out = ad.prefill_forward(_history([b[:end]]), page_table=_table([row_b[: -(-end // BLOCK_SIZE)]], 1),
                                 kv_cache=kv, prompt_lens=[end], start_pos=[start], empty_slots=[1],
                                 sampling_params=_greedy(1) if end == len(b) else None)  # fmt: skip
        start = end
    gb = ad._alias[1]
    assert kv.state[gb] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB and pc.stats["attaches"] == 1
    hb = pc._prompt_keys(list(b))[2]
    for n in (384, 512):
        assert pc.store.key(b, n) in pc.store.entries and pc.store.key(b, n) == hb[n // 128 - 1]
    assert pc.store.stats["chain_tokens"] == 384 + 512  # B: its attach key (256) once, then 128 per chunk end
    ad.release_request(1)
    # ---- random chunkings of one prompt: every running key is the whole-prefix key
    rng = random.Random(7)
    p = [rng.randrange(VOCAB) for _ in range(3000)]
    for _ in range(10):
        ch = None
        for e in sorted(rng.sample(range(1, 3000 // 128 + 1), 5)):
            k, ch = pc.store.chain_key(ch, torch.tensor(p, dtype=torch.int32), e * 128)
            assert k == pc.store.key(p, e * 128) == prefix_key(p, e * 128, pc.store.sig, 128, fast)
    # ---- a chain that is not this sequence's restarts: the key is still the row's own
    c = _prompt(300, 17)
    _k, ch = pc.store.chain_key(None, torch.tensor(b, dtype=torch.int32), 256)
    k2, ch2 = pc.store.chain_key(ch, torch.tensor(c, dtype=torch.int32), 256)
    assert k2 == pc.store.key(c, 256) and pc.store.stats["chain_restarts"] == 1
    k3, _ch = pc.store.chain_key(ch2, torch.tensor(c, dtype=torch.int32), 128)  # behind the chain
    assert k3 == pc.store.key(c, 128) and pc.store.stats["chain_restarts"] == 2
    if fast:
        bad = list(a)
        bad[5] = 2**31
        with expect_error(ValueError, "outside"):
            pc.store.chain_key(None, torch.tensor(bad, dtype=torch.int64), 128)
        with expect_error(ValueError, "outside"):
            ad.confirm_prefix_cache_hit(bad, [], 128)
    assert "ttnn" not in sys.modules


def test_prefix_cache_confirm_memoizes_per_token_list(monkeypatch):
    """F46: the scheduler confirms a waiting request again at every step with the same token list -- its block hashes
    are memoized per list and reused only while its content is unchanged (a copy compared exactly): the retries answer
    as the first confirmation; an in-place change of the list (same length) recomputes and answers for the new ids;
    vLLM's ConstantList (the list in _x) is compared through its list"""
    from collections.abc import Sequence

    class _ConstantList(Sequence):
        """vllm.v1.utils.ConstantList: a read-only view of a request's token list (kept in _x)"""

        def __init__(self, x):
            self._x = x

        def __getitem__(self, i):
            return self._x[i]

        def __len__(self):
            return len(self._x)

    monkeypatch.setenv("GLM53_PREFIX_FAST_HASH", "1")
    ad, backend, gen, kv = _prefix_adapter(monkeypatch)
    pc = ad.prefix
    a = _prompt(300, 5)
    _prefill_steps(ad, kv, a, _blocks(0, len(a)), slot=0, ends=[128, 256, len(a)])
    ad.release_request(0)
    q = a[:260] + _prompt(20, 9)
    assert ad.confirm_prefix_cache_hit(q, [], 256) == 256
    assert pc.stats["confirm_memo_misses"] == 1 and pc.stats["confirm_memo_hits"] == 0
    for _ in range(3):  # the scheduler's retries of the waiting request
        assert ad.confirm_prefix_cache_hit(q, [], 256) == 256
    assert pc.stats["confirm_memo_hits"] == 3 and len(pc.pins) == 1  # one pin, moved by every retry
    q[200] = (q[200] + 1) % VOCAB  # changed in place, same length: [128, 256) no longer matches A
    assert ad.confirm_prefix_cache_hit(q, [], 256) == 128 and pc.stats["confirm_memo_misses"] == 2
    cl = _ConstantList(a[:260] + _prompt(20, 9))
    assert ad.confirm_prefix_cache_hit(cl, [], 256) == 256 and ad.confirm_prefix_cache_hit(cl, [], 200) == 128
    assert pc.stats["confirm_memo_hits"] == 4 and pc.stats["confirm_memo_misses"] == 3
    assert "ttnn" not in sys.modules


class _MockAsyncHostTierBackend(_MockHostTierBackend):
    """+ export_snapshot_async: the entry's state is taken when the export is enqueued (StackBackend's queue-ordered
    staging reads), the blob assembled by finish() -- which waits for ``gate`` (the test holds it shut) and fails when
    ``fail`` was set at the enqueue; the admission bound is above the blob's bytes"""

    BOUND = 1 << 16

    def __init__(self, max_batch: int):
        super().__init__(max_batch)
        self.gate = threading.Event()
        self.fail = False
        self.async_exports: list = []

    def export_snapshot_async(self, kv, entry: int, n_tokens=None):
        self.async_exports.append((int(entry), n_tokens))
        state, fail = list(kv.state[self.max_batch + int(entry)]), self.fail

        def finish():
            assert self.gate.wait(30)
            if fail:
                raise RuntimeError("mock: the staging read failed")
            return {"state": torch.tensor(state, dtype=torch.int64)}

        return finish, self.BOUND


_ASYNC_EXPORT_CHILD = r"""
import sys
from types import SimpleNamespace
from unittest import mock

import torch

sys.modules["ttnn"] = mock.MagicMock()  # stack_model's module-level ttnn import; the device reads are faked below
import ttnn  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as STM  # noqa: E402

assert not STM.MOCK  # (the real staging reads: the fake device below, not the mock stub's zeros)
MEM = {}  # buffer address -> the fake device buffer's tiles [chips, tiles, 32, 32]


class Copy:  # ttnn.from_device(blocking=False): the staging's tiles when the read is enqueued (queue order)
    def __init__(self, data):
        self.data = data.clone()


def buf(chips, tiles, dt, gen=None):
    a = 0x1000 * (len(MEM) + 1)
    MEM[a] = torch.zeros(chips, tiles, 32, 32, dtype=dt) if gen is None else torch.randn(
        chips, tiles, 32, 32, generator=gen).to(dt)
    return SimpleNamespace(padded_shape=(tiles, 1, 32, 32), dtype=ttnn.float32 if dt == torch.float32 else
                           ttnn.bfloat16, buffer_address=lambda a=a: a)


def run(ops):  # SlotOps: kind 1 = tile copy, kind 2 = conv row (src row << 16 | dst row) of every tile
    for kind, sa, s0, da, d0, n, _nb, rows in ops:
        if kind == 1:
            MEM[da][:, d0 : d0 + n] = MEM[sa][:, s0 : s0 + n]
        else:
            MEM[da][:, :n, rows & 0xFFFF, :] = MEM[sa][:, :n, rows >> 16, :]


ttnn.from_device = lambda t, blocking=True: Copy(MEM[t.buffer_address()])
ttnn.to_torch = lambda t, mesh_composer=None: (t.data if isinstance(t, Copy) else MEM[t.buffer_address()]).clone()
CHUNK, MB, ENTRIES, CH, N_S = 4, 4, 4, 2, 2  # 2 chips; the entries are virtual slots 4..7 (state chunk 1)
SEQ = [dict(L=0, sparse=None), dict(L=1, sparse="dsa"), dict(L=2, sparse=None), dict(L=3, sparse=None),
       dict(L=4, sparse="dsa")]


def rig(shard, rep, paged, seed):
    gen = torch.Generator().manual_seed(seed)
    MEM.clear()
    states = [[None if e["sparse"] else (buf(CH, CHUNK * N_S, torch.float32, gen), buf(CH, 3, torch.bfloat16, gen))
               for e in SEQ] for _c in range(2)]
    dsa = {i: dict(pools=buf(CH, 2 * MB * 8, torch.bfloat16, gen), ring=buf(CH, 2 * MB * 2, torch.bfloat16, gen),
                   slots=2 * MB, paged=paged) for i, e in enumerate(SEQ) if e["sparse"]}
    if rep:  # every chip holds the same DSA rows
        for st in dsa.values():
            for t in (st["pools"], st["ring"]):
                MEM[t.buffer_address()][1:] = MEM[t.buffer_address()][:1]
    extra = dict(snapshot_entries=ENTRIES, dsa=dsa, kda_stage_s=buf(CH, 2 * N_S, torch.float32),
                 kda_stage_c=buf(CH, 3, torch.bfloat16), dsa_stage=buf(CH, 6, torch.bfloat16))
    if shard:
        extra["dsa_shard"] = (CH, 10**6, 2)
    kv = SimpleNamespace(extra=extra, states=states, max_batch=MB)
    be = STM.StackBackend.__new__(STM.StackBackend)
    be.mesh, be.tp, be.chunk, be.seq, be.counters = SimpleNamespace(), CH, CHUNK, SEQ, {"slot_ops": 0}
    be.slot_ops = SimpleNamespace(run=run)
    return be, kv


def same(a, b, where="blob"):
    assert type(a) is type(b), (where, type(a), type(b))
    if isinstance(a, dict):
        assert a.keys() == b.keys(), (where, a.keys(), b.keys())
        for k in a:
            same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), (where, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b), (where, a.shape, b.shape)
    else:
        assert a == b, (where, a, b)


KEYS = ("dsa_exports", "dsa_export_bytes", "dsa_export_replicated", "kda_pack_exports", "slot_ops")
n_cases = 0
for shard in (False, True):
    for rep in (False, True):
        for paged in (False, True):
            for n_tokens in (300, None):
                be, kv = rig(shard, rep, paged, seed=n_cases)
                c0 = {k: be.counters.get(k, 0) for k in KEYS}
                blob = be.export_snapshot(kv, 2, n_tokens=n_tokens)
                c1 = {k: be.counters.get(k, 0) for k in KEYS}
                finish, bound = be.export_snapshot_async(kv, 2, n_tokens=n_tokens)
                for t in MEM.values():  # a later prefill / import rewrites the entry and the staging: the reads
                    t.add_(1)  # were taken in queue order at the enqueue
                got = finish()
                c2 = {k: be.counters.get(k, 0) for k in KEYS}
                same(blob, got)
                assert [k for d in blob for k in d] == ["kda", "dsa"]
                dsa = blob[1]["dsa"]
                assert dsa["layout"] == ("unit" if shard else "chip") and dsa["pools"] == (not paged)
                assert dsa["units"] == (0 if paged else None if n_tokens is None else 3)
                assert dsa["replicated"] == rep, (shard, rep, paged, dsa["replicated"])
                assert all(c2[k] - c1[k] == c1[k] - c0[k] for k in KEYS), (c0, c1, c2)
                assert c1["dsa_export_bytes"] - c0["dsa_export_bytes"] == sum(
                    v.numel() * v.element_size() for v in dsa["tensors"].values()) > 0
                kda = blob[0]["kda"]
                assert bound >= sum(v.numel() * v.element_size() for v in (kda["S"], kda["C"])) + c1[
                    "dsa_export_bytes"] - c0["dsa_export_bytes"], "the admission bound covers the blob"
                n_cases += 1
assert n_cases == 16
print("ASYNC_EXPORT_OK")
"""


def test_prefix_cache_async_export_ordering(monkeypatch):
    """GLM53_PREFIX_ASYNC_EXPORT: a full pool's LRU entry is spilled with its transfer in flight -- the snapshot that
    evicted it reuses the device entry at once, yet the blob holds the evicted state (taken in queue order at the
    enqueue, assembled later off the engine thread); the host tier admits it at the backend's bound and takes the
    blob's size when it lands; a restore of the entry waits for its blob before the import and serves the cold
    prefill's first token; a failed in-flight export drops its entry; nothing blocks on the export otherwise"""
    from concurrent.futures import wait

    monkeypatch.setattr(gvs, "PREFIX_CACHE", True)
    monkeypatch.setattr(gvs, "PREFIX_ASYNC_EXPORT", True)
    monkeypatch.setenv("GLM53_PREFIX_HOST_BYTES", str(1 << 30))
    monkeypatch.setenv("GLM53_PREFIX_SPILL_MIN", "0")
    backend = _MockAsyncHostTierBackend(B)
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
    ad = GLM53StackForCausalLM(gen, B, SEQ)
    kv = ad.allocate_kv_cache((NUM_BLOCKS, 1, BLOCK_SIZE, LATENT_DIM), torch.bfloat16, 45)
    ad.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
    ad.warmup_model_decode(kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=BPS, can_sample_on_device=True)
    pc = ad.prefix
    assert pc.host and pc.async_export
    doc = _prompt(256, 5)
    a, b = doc + _prompt(40, 50), doc + _prompt(30, 20)
    row_a = _blocks(0, len(a))
    _prefill_steps(ad, kv, a, row_a, slot=0, ends=[128, 256, len(a)])
    ad.release_request(0)
    key128, key256 = pc.store.key(a, 128), pc.store.key(a, 256)
    # ---- C needs three entries: A's 128 and 256 are spilled in flight (the gate holds their blobs back) and their
    #      device entries take C's 256 / 384 snapshots at once
    c = _prompt(384, 9) + _prompt(10, 33)
    _prefill_steps(ad, kv, c, _blocks(1, len(c)), slot=1, ends=[128, 256, 384, len(c)])
    ad.release_request(1)
    assert [x[0] for x in backend.async_exports] == [0, 1] and backend.exports == []  # no blocking export
    assert pc.stats["spills_async"] == 2 and len(pc._exports) == 2
    e128, e256 = pc.store.entries[key128], pc.store.entries[key256]
    assert e128.tier == e256.tier == "host" and isinstance(e256.host_blob, gvs._PendingExport)
    assert kv.state[B + 0] == c[:256] and kv.state[B + 1] == c[:384]  # the entries hold C's snapshots now
    assert pc.store.host_used == 2 * backend.BOUND  # admitted at the bound until they land
    # ---- B's hit on the in-flight 256 is confirmed (its pages are resident); its restore waits for the blob (the
    #      gate opens 0.2 s into B's prefill) and imports A's state -- not C's 384, which took A's device entry
    assert ad.confirm_prefix_cache_hit(b, [], 256) == 256
    threading.Timer(0.2, backend.gate.set).start()
    row_b = row_a[:4] + _blocks(2, len(b))[4:]
    out = ad.prefill_forward(_history([b]), page_table=_table([row_b], 1), kv_cache=kv, prompt_lens=[len(b)],
                             start_pos=[256], empty_slots=[2], sampling_params=_greedy(1))  # fmt: skip
    g = ad._alias[2]
    e256 = pc.store.entries[key256]
    assert pc.stats["export_waits"] == 1 and pc.stats["export_wait_s"] >= 0.1
    # (256's export was still queued behind 128's, which held the one thread at the gate: the restore took it out of
    # the queue and assembled it in place once 128's assembly finished, instead of waiting for the thread to reach it)
    assert pc.stats["export_jumps"] == 1
    assert e256.tier == "device" and backend.imports == [e256.device_entry] and kv.state[B + e256.device_entry] == doc
    assert kv.state[g] == b and int(out[0][0, 0]) == (b[-1] + 1) % VOCAB
    assert pc.stats["restores"] == 1 and pc.stats["attaches"] == 1 and pc.stats["fallbacks"] == 0
    ad.release_request(2)
    # ---- the rest land: the evicted states, their real sizes in the host accounting
    wait([p.future for p in pc._exports.values()], timeout=30)
    with pc.lock:
        pc._settle_exports()
    assert not pc._exports and pc.stats["exports_landed"] == pc.stats["spills_async"] == 3
    assert torch.equal(pc.store.entries[key128].host_blob["state"], torch.tensor(doc[:128]))
    assert pc.store.entries[key128].bytes == 128 * 8
    assert pc.store.host_used == sum(e.bytes for e in pc.store.entries.values() if e.tier == "host")
    # ---- failed in-flight exports drop their entries (their next hits prefill from 0, as after a blocking failure)
    backend.fail = True
    before = {k for k, e in pc.store.entries.items() if e.tier == "device"}
    d = _prompt(256, 21) + _prompt(10, 1)
    _prefill_steps(ad, kv, d, _blocks(3, len(d)), slot=3, ends=[128, 256, len(d)])
    ad.release_request(3)
    assert pc.stats["spills_async"] == 5  # D's two snapshots spilled two device entries
    wait([p.future for p in pc._exports.values()], timeout=30)
    with pc.lock:
        pc._settle_exports()
    gone = before - set(pc.store.entries)
    assert pc.stats["export_failed"] == 2 and len(gone) == 2 and not pc._exports
    assert pc.store.host_used == sum(e.bytes for e in pc.store.entries.values() if e.tier == "host")
    assert backend.exports == [] and "ttnn" not in sys.modules
    # ---- the device side (StackBackend over a fake device, a stub-ttnn child): export_snapshot_async's finish() returns
    #      export_snapshot's blob -- keys, shapes, bytes, units, layout, replicated flag; shard / non-shard, replicated
    #      or not, per-slot or paged pools, a cut or the whole slot -- and moves the same counters (dsa_export_bytes /
    #      dsa_export_replicated included); its reads are the staging at the enqueue, and its bound covers the blob
    _stub_child(_ASYNC_EXPORT_CHILD, "ASYNC_EXPORT_OK")
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- GLM53_PF_TAIL_SPLIT (Glm53Generator._tail_split)
def test_prefill_tail_split_pieces(monkeypatch):
    """GLM53_PF_TAIL_SPLIT on the TP prefill's buckets 2,048 / 8,192: "auto" splits a 4,150-token chunk at 64 K into
    2,048-row pieces (cheaper than one 8,192 replay there by the cost table) and keeps it whole at 1 M (the pieces cost
    more); a k cap splits the LAST chunk only; the F27 guard keeps a chunk whole when a piece would start past the
    backend's prefill_resume_max; every split covers its chunk exactly (starts, offsets, lengths, padded lengths). The
    knobs are checked when the generator is built"""
    from models.autoports.zai_org_glm_5_3_flash.tt.generator import ChunkPlan, round_up

    backend = _MockStackBackend(B)
    backend.tpp = SimpleNamespace(buckets=(2048, 8192))
    rmax = [1046528]  # (TPPrefill.resume_max at 1 M: the top window - 2,048)
    backend.prefill_resume_max = lambda: rmax[0]
    gen = Glm53Generator(backend, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)

    def chunk(start, n, off=7):
        return ChunkPlan(start=start, offset=off, n_valid=n, physical=round_up(n, backend.prefill_align))

    def covers(c, pieces):
        assert (
            pieces[0].start == c.start and pieces[0].offset == c.offset and sum(p.n_valid for p in pieces) == c.n_valid
        )
        for p, q in zip(pieces, pieces[1:]):
            assert q.start == p.start + p.n_valid and q.offset == p.offset + p.n_valid and p.n_valid == 2048
        assert all(p.physical == round_up(p.n_valid, backend.prefill_align) for p in pieces)

    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT", "auto")
    c64 = chunk(65536, 4150)
    out = gen._tail_split([c64])
    assert [p.n_valid for p in out] == [2048, 2048, 54] and gen.counters["prefill_tail_splits"] == 1
    covers(c64, out)
    assert gen.counters["prefill_split_ms_saved"] > 0
    c1m = chunk(1040384, 4150)  # 3 x 2,048 at ~1 M (~3.0 s) lose to one 8,192 replay (~2.5 s)
    assert gen._tail_split([c1m]) == [c1m] and gen.counters["prefill_tail_splits"] == 1
    assert gen._tail_split([chunk(65536, 2000)]) == [chunk(65536, 2000)]  # (fits the smallest bucket: whole)
    out = gen._tail_split([c64, chunk(69686, 4150, off=4157)])  # auto: every chunk of the call
    assert [p.n_valid for p in out] == [2048, 2048, 54] * 2
    covers(c64, out[:3])
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT", "2")  # a k cap: the LAST chunk only, at most k pieces
    tail = chunk(65536, 2100)
    out = gen._tail_split([c64, tail])
    assert out[0] == c64 and [p.n_valid for p in out[1:]] == [2048, 52]
    covers(tail, out[1:])
    assert gen._tail_split([c64]) == [c64]  # (3 pieces > k 2)
    # the F27 guard: a 2,100-token tail resumed at 1,046,400 -- its second piece would start at 1,048,448, past
    # resume_max (no bucket has a key there): kept whole (TPPrefill.clamp_lengths plans it tail-first)
    near = chunk(1046400, 2100)
    assert gen._tail_split([near]) == [near]
    rmax[0] = None  # (a backend without the bound: split)
    assert [p.n_valid for p in gen._tail_split([near])] == [2048, 52]
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT", "0")
    assert gen._tail_split([c64]) == [c64]
    # the knobs are validated at construction (a typo stops the launch, not its first long prefill)
    for knob, bad in (("GLM53_PF_TAIL_SPLIT", "on"), ("GLM53_PF_TAIL_SPLIT", "-2"),
                      ("GLM53_PF_SPLIT_COSTS", "2048:0=x"), ("GLM53_PF_SPLIT_COSTS", "2048:0")):  # fmt: skip
        monkeypatch.setenv(knob, bad)
        with expect_error(ValueError, knob):
            Glm53Generator(_MockStackBackend(B), tokenizer=None, max_batch=B, max_seq_len=SEQ)
        monkeypatch.delenv(knob)
    monkeypatch.setenv("GLM53_PF_SPLIT_COSTS", "2048:0=10,1000000=20;8192:0=15")
    g2 = Glm53Generator(_MockStackBackend(B), tokenizer=None, max_batch=B, max_seq_len=SEQ)
    assert g2._split_cost_ms(2048, 500000) == 15.0 and g2._split_cost_ms(1024, 0) is None
    # a 1,024 bucket (GLM53_PREFILL_BUCKETS=1024,2048,8192), default cost table: auto takes the cheapest MIX of replay
    # sizes -- an exact-snapshot chunk of 2,176 tokens at ~80 K replays 1,024 + 2,048 rows instead of 2 x 2,048, a
    # 4,150-token chunk keeps two 2,048 replays (five 1,024 pieces cost more), ~1 M still prefers one 8,192 replay
    monkeypatch.delenv("GLM53_PF_SPLIT_COSTS")
    monkeypatch.setenv("GLM53_PF_TAIL_SPLIT", "auto")
    b3 = _MockStackBackend(B)
    b3.tpp = SimpleNamespace(buckets=(1024, 2048, 8192))
    b3.prefill_resume_max = lambda: 1046528
    g3 = Glm53Generator(b3, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)

    def replays(pieces):  # (the bucket each piece replays in)
        return sorted(next(c for c in (1024, 2048, 8192) if c >= p.n_valid) for p in pieces)

    snap = chunk(81920, 2176)
    out = g3._tail_split([snap])
    assert replays(out) == [1024, 2048] and out[1].start == 81920 + out[0].n_valid and out[0].n_valid % 1024 == 0
    assert sum(p.n_valid for p in out) == 2176 and out[1].offset == snap.offset + out[0].n_valid
    assert g3.counters["prefill_tail_rows_saved"] == 8192 - 1024 - 2048 and g3.counters["prefill_split_ms_saved"] > 0
    out = g3._tail_split([c64])
    assert replays(out) == [1024, 2048, 2048] and sum(p.n_valid for p in out) == 4150
    assert g3._tail_split([chunk(65536, 1000)]) == [chunk(65536, 1000)]  # (fits the smallest bucket: whole)
    assert g3._tail_split([c1m]) == [c1m]
    assert "ttnn" not in sys.modules


# ------------------------------------------------------------------- AllocPool / SpecSwitch (stub-ttnn children)
_POOL_SPEC_CHILD = r"""
import sys
from types import SimpleNamespace
from unittest import mock

sys.modules["ttnn"] = mock.MagicMock()  # mk_stack / spec_mtp / stack_model module-level ttnn imports only
from models.autoports.zai_org_glm_5_3_flash.tt import mk_stack as MS
from models.autoports.zai_org_glm_5_3_flash.tt import spec_mtp as SM
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as STM

# ---- AllocPool: frozen after the first capture -- a later build of the same programs allocates nothing; a program
#      first built after it allocates (not refused), logged once and counted
pool = MS.AllocPool()
made = []
pool.orig = lambda shape, *a, **kw: made.append(tuple(shape)) or SimpleNamespace(shape=tuple(shape))
with pool:
    x, y = pool((1, 1, 32, 64), "bf16", "tile"), pool((1, 1, 32, 64), "bf16", "tile")
child = pool.child()
assert len(made) == 2 and not pool.frozen and not child.frozen
pool.freeze()
assert pool.frozen and child.frozen and pool.post_capture_allocs == 0
with pool:
    assert pool((1, 1, 32, 64), "bf16", "tile") is x and pool((1, 1, 32, 64), "bf16", "tile") is y
assert len(made) == 2 and pool.post_capture_allocs == 0 and not MS._POST_CAPTURE_LOGGED
with pool:
    pool((1, 1, 32, 128), "bf16", "tile")
    pool((1, 1, 32, 256), "bf16", "tile")
assert len(made) == 4 and pool.post_capture_allocs == 2 and MS._POST_CAPTURE_LOGGED == [True]
p2 = MS.AllocPool()
p2.freeze()
assert p2.child().frozen  # (a child made after the freeze)


# ---- SpecSwitch: every step captured, all or nothing
class Step:
    def __init__(self, fail=False):
        self.groups, self.traces, self.fail, self.released = 4, {}, fail, 0

    def capture(self):
        if self.fail:
            raise RuntimeError("trace region full")
        self.traces = {"step": ("mock", lambda: None), "boot": ("mock", lambda: None)}
        return self.traces

    def release(self):
        self.released += 1
        self.traces = {}


a, b = Step(), Step(fail=True)
sw = SM.SpecSwitch([a, b])
try:
    sw.capture()
    raise AssertionError("SpecSwitch.capture swallowed a step's failure")
except RuntimeError:
    pass
assert a.traces == {} and a.released == 1 and b.released == 1 and not sw.captured()
b.fail = False
assert sw.capture() is a.traces and sw.captured()
# StackBackend._capture_spec: a switch whose CURRENT step is captured but another is not -> captures the rest
a2, b2 = Step(), Step()
a2.capture()
sw2 = SM.SpecSwitch([a2, b2])
assert sw2.traces and not sw2.captured()
STM.StackBackend._capture_spec(SimpleNamespace(spec=sw2, mesh=None))
assert b2.traces and sw2.captured()
STM.StackBackend._capture_spec(SimpleNamespace(spec=sw2, mesh=None))  # (captured: nothing)
assert a2.released == b2.released == 0
# SpecMTP._replay before any capture: a LAZY capture -- counted, warned once (no assert)
fake = SimpleNamespace(traces={}, counters={}, K=3, groups=4, ran=[])
fake.capture = lambda: fake.traces.setdefault("step", ("mock", lambda: fake.ran.append(1))) and fake.traces
SM.SpecMTP._replay(fake, "step")
SM.SpecMTP._replay(fake, "step")
assert fake.counters == {"lazy_captures": 1} and fake.ran == [1, 1] and SM._LAZY_CAPTURE_LOGGED == [True]
print("POOL_SPEC_OK")
"""


def test_alloc_pool_frozen_and_spec_switch_capture():
    """mk_stack.AllocPool.freeze (StackBackend._capture): after the first trace capture a pool tensor allocated by a
    program first built then is logged once and counted (not refused: unvalidated on device). spec_mtp.SpecSwitch:
    capture is all or nothing (a step's failure releases every step and re-raises), captured() needs EVERY step, and
    StackBackend._capture_spec captures a switch whose current step alone holds traces; SpecMTP._replay before any
    capture counts / warns a lazy capture. The modules import ttnn at module level: a child with a stub ttnn"""
    _stub_child(_POOL_SPEC_CHILD, "POOL_SPEC_OK")
    assert "ttnn" not in sys.modules


# ==================================== page-table refresh (GLM53_PT_CHANGED_ROWS, GLM53_SPEC_PT_LAZY)
def test_page_rows_changed_only(monkeypatch):
    """GLM53_PT_CHANGED_ROWS (default 1): _write_page_rows' full path over the rows that changed only -- with every
    live row passed while some grew (the served verify / decode), a new request's row replaced and a row cut at a
    repeated block, the published logical / physical tables, the coverage and the owner-block map equal the full path's
    over every passed row (=0)"""

    def run(flag):
        monkeypatch.setenv("GLM53_PT_CHANGED_ROWS", flag)
        be = _MockPrefixBackend(B)
        gen = Glm53Generator(be, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=B, max_seq_len=SEQ)
        kv = gen.allocate_kv_cache(B, SEQ, num_blocks=NUM_BLOCKS, null_block=0)
        gen.shared_pages = True
        b = gen._binding(kv)
        n = {0: 5, 1: 9, 2: 3, 3: 12}

        def row(s, blocks=None):
            r = torch.full((BPS,), -1, dtype=torch.int64)
            bl = blocks if blocks is not None else _blocks(s, n[s] * BLOCK_SIZE)
            r[: len(bl)] = torch.tensor(bl, dtype=torch.int64)
            return r

        used = lambda: {s: (n[s] - 1) * BLOCK_SIZE + 1 for s in n}  # noqa: E731
        gen._write_page_rows(b, {s: row(s) for s in n}, used())
        snaps = []
        for grow in ([1], [0, 3], [], [2]):
            for s in grow:
                n[s] += 1
            gen._write_page_rows(b, {s: row(s) for s in n}, used())
            snaps.append((b.pt_shadow.clone(), b.phys.clone(), b.coverage.clone()))
        n[2] = 4  # a new request in slot 2: another pool's blocks (the old row's released by its owner first)
        gen.release_slot(kv, 2)
        new2 = [1 + 3 * BPS + 20 + j for j in range(4)]
        gen._write_page_rows(b, {0: row(0), 1: row(1), 2: row(2, new2), 3: row(3)}, used())
        snaps.append((b.pt_shadow.clone(), b.phys.clone(), b.coverage.clone()))
        rep = _blocks(1, n[1] * BLOCK_SIZE)
        gen._write_page_rows(b, {0: row(0), 1: row(1, rep + [rep[2]]), 2: row(2, new2), 3: row(3)}, used())
        snaps.append((b.pt_shadow.clone(), b.phys.clone(), b.coverage.clone()))
        o = b.obm
        return snaps, (dict(o.rows), dict(o.phys), dict(o.refs)), b.io.page_table.t.clone(), dict(gen.counters)

    s0, m0, t0, c0 = run("0")
    s1, m1, t1, c1 = run("1")
    assert len(s0) == len(s1) == 6
    for a, b_ in zip(s0, s1):
        assert all(torch.equal(x, y) for x, y in zip(a, b_))
    assert m0 == m1 and torch.equal(t0, t1)
    assert c0.get("page_rows_skipped", 0) == 0 and c1["page_rows_skipped"] == 3 + 2 + 3 + 3 + 3
    assert c0["page_rows_unchanged"] == c1["page_rows_unchanged"] == 1  # (the round where nothing grew)
    assert "ttnn" not in sys.modules


_PT_LAZY_CHILD = r"""
import os
import random
import sys
from types import SimpleNamespace
from unittest import mock

import torch

# (the host test module and the generator first, ttnn-free; then the stub for spec_mtp / stack_model's module-level
# ttnn imports only)
import models.demos.glm53_flash.tests.test_stack_adapter_host as T
from models.autoports.zai_org_glm_5_3_flash.tt import dsa_shard as DSH
from models.autoports.zai_org_glm_5_3_flash.tt.generator import Glm53Generator

assert "ttnn" not in sys.modules
sys.modules["ttnn"] = mock.MagicMock()
import ttnn  # noqa: E402

from models.autoports.zai_org_glm_5_3_flash.tt import spec_mtp as SM  # noqa: E402
from models.autoports.zai_org_glm_5_3_flash.tt import stack_model as STM  # noqa: E402

BPU, BS, MB, CHUNK = 2, 64, 16, 4
SERVED = (("lone", 1, 3), ("primary", 4, 3), ("alt", 8, 1))  # the served 1 M steps (name, G users, K drafts)


class Dev:
    # a device buffer: data = the last host tensor copied in (every copy logged by name)
    def __init__(self, rig, name, rows):
        self.rig, self.name, self.shape, self.data = rig, name, (rows, rig.bl), None
        rig.devs[name] = self


def _copy(host, dev):
    dev.rig.log.append(dev.name)
    dev.data = host.clone()


ttnn.from_torch = lambda t, **kw: t.clone()
ttnn.copy_host_to_device_tensor = _copy
CALLS, RIG = [], [None]


def _prepare(self):  # (the eager compile runs read the step's tables: they must be the published ones)
    CALLS.append(("prepare", self.name, RIG[0].holds(self)))


def _capture(self):
    CALLS.append(("capture", self.name, RIG[0].holds(self)))
    return self.traces


SM.SpecMTP.prepare, SM.SpecMTP.capture = _prepare, _capture


class Rig:
    # the real StackBackend.write of the page table on the sharded pool (C owner chips) + real SpecMTP / SpecSwitch
    # tables, every device buffer a Dev
    def __init__(self, C, bps, steps=SERVED, switch=True, pool=False):
        self.C, self.bps, self.log, self.devs, self.value = C, bps, [], {}, None
        self.bl = DSH.ShardMap(C, BS).local_cap(bps * BS) // BS
        be = self.be = STM.StackBackend.__new__(STM.StackBackend)
        be.mesh, be.tp, be.chunk = SimpleNamespace(get_num_devices=lambda: C), C, CHUNK
        self.io = SimpleNamespace(page_table=SimpleNamespace(name="page_table", t=Dev(self, "pt", MB)))
        be._kv = SimpleNamespace(extra={"dsa_shard": (C, 10**6, BPU), "bps_local": self.bl, "io": self.io,
                                        "pt_c": [Dev(self, f"pt_c{c}", CHUNK) for c in range(MB // CHUNK)]})
        self.steps = [self.step(n, G, K, pool) for n, G, K in steps]
        be.spec = SM.SpecSwitch(self.steps) if switch else self.steps[0]
        RIG[0] = self

    def step(self, name, G, K, pool):
        x = SM.SpecMTP.__new__(SM.SpecMTP)
        x.__dict__.update(name=name, groups=G, K=K, slots=tuple(u * (K + 1) for u in range(max(1, G))),
                          shard=(self.C, 10**6, BPU), be=self.be, mesh=self.be.mesh, mtp_private=False,
                          mtp_sparse=not pool, drafter="mtp", _pt_rows=None, traces={})
        x.pt_v = Dev(self, name + ".pt_v", 32)
        x.pt_vc = Dev(self, name + ".pt_vc", 32) if G and K >= 2 and not pool else None
        x.pt_m = Dev(self, name + ".pt_m", 32) if pool else x.pt_v  # (GLM53_SPEC_MTP_CACHE=pool: global rows)
        x.pt_mc = Dev(self, name + ".pt_mc", 32) if pool and G else x.pt_m
        return x

    def publish(self, value):  # (Glm53Generator._publish_page_table's two backend calls)
        value = torch.as_tensor(value).to(torch.int32)
        STM.StackBackend.write(self.be, self.io.page_table, value)
        STM.StackBackend.set_page_table_host(self.be, self.be._kv, value)
        self.value = value.clone()

    def cur(self):
        sp = self.be.spec
        return sp.steps[sp.cur] if isinstance(sp, SM.SpecSwitch) else sp

    def ref(self, x):
        # the bytes the old code (every step written, localized on its own) leaves in x's tables: built here from
        # dsa_shard.local_rows directly
        G, g = max(1, x.groups), x.K + 1
        rows = self.value[list(x.slots)].to(torch.int64)
        loc = torch.stack(DSH.local_rows(rows, G, self.C, BPU, self.bl))
        by_row = [min(r // g, G - 1) for r in range(32)] if x.groups else [0] * 32
        by_user = [min(r, G - 1) for r in range(32)]
        out = {x.pt_v.name: loc[:, by_row].reshape(-1, self.bl)}
        if x.pt_vc is not None:
            out[x.pt_vc.name] = loc[:, by_user].reshape(-1, self.bl)
        if x.pt_m is not x.pt_v:
            out[x.pt_m.name] = rows[by_row]
            if x.groups:
                out[x.pt_mc.name] = rows[by_user]
        return out

    def holds(self, x):
        return all(self.devs[k].data is not None and torch.equal(self.devs[k].data.to(torch.int64), v.to(torch.int64))
                   for k, v in self.ref(x).items())

    def main_ok(self):
        loc = DSH.local_rows(self.value.to(torch.int64), MB, self.C, BPU, self.bl)
        assert torch.equal(self.devs["pt"].data, torch.cat(loc))
        for c in range(MB // CHUNK):
            assert torch.equal(self.devs[f"pt_c{c}"].data, torch.cat([lt[c * CHUNK : (c + 1) * CHUNK] for lt in loc]))

    def snap(self):  # (what the next replay reads: the main tables + the current step's)
        x = self.cur()
        names = ["pt"] + [f"pt_c{c}" for c in range(MB // CHUNK)] + list(self.ref(x))
        return {k: self.devs[k].data.clone() for k in names}


def script(C, bps, n, seed, nsteps):
    # a random op sequence over physical tables (column j on owner (j // BPU) % C, physical block owner + C x id, id 0 =
    # the owner's null block): block crossings, releases, new requests, slot remaps, prefix attaches (shared blocks),
    # same-table publishes, step switches, entries (the current step's rows set directly), prepare, capture
    rng, ids, nxt = random.Random(seed), {}, [1]
    own = (torch.arange(bps) // BPU) % C

    def fresh(k):
        nxt[0] += k
        return list(range(nxt[0] - k, nxt[0]))

    def table():
        v = own.expand(MB, -1).clone()
        for s, i in ids.items():
            v[s, : len(i)] = own[: len(i)] + C * torch.tensor(i, dtype=torch.int64)
        return v

    for s in (0, 2, 4):
        ids[s] = fresh(rng.randint(2, bps // 2))
    ev = [("publish", table())]
    kinds = ["grow"] * 6 + ["release", "new", "new", "remap", "attach", "same", "select", "select", "enter",
                            "prepare", "capture"]  # fmt: skip
    for _ in range(n):
        k = rng.choice(kinds)
        if k in ("grow", "release", "remap", "attach") and not ids:
            k = "new"
        if k == "grow":
            s = rng.choice(sorted(ids))
            if len(ids[s]) < bps:
                ids[s].append(fresh(1)[0])
        elif k == "release":
            del ids[rng.choice(sorted(ids))]
        elif k == "new":
            ids[rng.randrange(MB)] = fresh(rng.randint(1, bps // 2))
        elif k == "remap":  # (the adapter's slot remap: two slots trade rows)
            a, b = rng.sample(range(MB), 2)
            ra, rb = ids.pop(a, None), ids.pop(b, None)
            if ra is not None:
                ids[b] = ra
            if rb is not None:
                ids[a] = rb
        elif k == "attach":  # (a prefix hit: another live row's first blocks, then fresh ones)
            t = rng.choice(sorted(ids))
            h = rng.randint(1, len(ids[t]))
            ids[rng.randrange(MB)] = ids[t][:h] + fresh(rng.randint(0, bps - h))
        if k in ("select",):
            ev.append(("select", rng.randrange(nsteps)))
        elif k in ("enter", "prepare", "capture"):
            ev.append((k,))
        else:
            ev.append(("publish", table()))
    return ev + [("prepare",), ("capture",)]


def run(rig, ev, lazy):
    os.environ["GLM53_SPEC_PT_LAZY"] = lazy
    snaps, stale = [], 0
    CALLS.clear()
    for e in ev:
        sp = rig.be.spec
        if e[0] == "publish":
            rig.publish(e[1])
        elif e[0] == "select":
            if isinstance(sp, SM.SpecSwitch):
                sp.select(e[1])
        elif e[0] == "enter":  # (SpecMTP.enter: the current step's users' rows, set directly, then the boot replay)
            x = rig.cur()
            if x.groups:
                sp.set_page_tables([rig.value[s] for s in x.slots])
            else:
                sp.set_page_table(rig.value[0])
        elif e[0] == "prepare":
            sp.prepare()
        else:
            sp.capture()
        rig.main_ok()
        assert rig.holds(rig.cur()), (lazy, e[0], rig.cur().name)  # (what the next replay of the current step reads)
        stale += sum(not rig.holds(x) for x in rig.steps)
        snaps.append(rig.snap())
    assert CALLS and all(ok for _, _, ok in CALLS), CALLS  # (prepare / capture met every step up to date)
    if isinstance(rig.be.spec, SM.SpecSwitch):  # (every step brought up to date once it is current)
        for i in range(len(rig.steps)):
            rig.be.spec.select(i)
            assert rig.holds(rig.steps[i]), (lazy, i)
    return snaps, stale, len(rig.log)


# ---- 1. fuzz: the served steps on the sharded pool (and a pool-MTP-cache variant, a lone step without a switch):
#         after every op the main tables and the CURRENT step's tables hold exactly the old full write's bytes, and
#         GLM53_SPEC_PT_LAZY=1 snapshots equal =0's; lazy leaves non-current steps stale and writes less
for cfg in (dict(), dict(pool=True), dict(switch=False, steps=(("one", 4, 3),)), dict(switch=False, steps=(("one", 0, 3),)),
            dict(switch=False, steps=(("one", 0, 3),), pool=True)):  # fmt: skip
    for seed in (1, 2, 3):
        ev = script(8, 48, 220, seed, len(cfg.get("steps", SERVED)))
        s0, st0, w0 = run(Rig(8, 48, **cfg), ev, "0")
        s1, st1, w1 = run(Rig(8, 48, **cfg), ev, "1")
        assert len(s0) == len(s1) and all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
                                          for a, b in zip(s0, s1)), (cfg, seed)  # fmt: skip
        assert st0 == 0, (cfg, seed, st0)
        if cfg.get("switch", True):
            assert st1 > 0 and w1 < w0, (cfg, seed, st1, w1, w0)
        else:
            assert st1 == 0 and w1 == w0, (cfg, seed, st1, w1, w0)

# ---- 2. the write counts of the served pattern (lone current; users on slots 0 / 2 / 4): a block crossing in slot 0
#         writes the main table, its chunk table and every step whose bases hold slot 0 (old) or the lone step only
#         (lazy); a select writes the stale step once
for lazy, want in (("0", [["pt", "pt_c0", "lone.pt_v", "lone.pt_vc", "primary.pt_v", "primary.pt_vc", "alt.pt_v"],
                          ["pt", "pt_c0", "alt.pt_v"], ["pt", "pt_c1", "primary.pt_v", "primary.pt_vc", "alt.pt_v"],
                          [], [], []]),
                   ("1", [["pt", "pt_c0", "lone.pt_v", "lone.pt_vc"], ["pt", "pt_c0"], ["pt", "pt_c1"],
                          ["alt.pt_v"], ["primary.pt_v", "primary.pt_vc"], []])):  # fmt: skip
    os.environ["GLM53_SPEC_PT_LAZY"] = lazy
    rig = Rig(8, 48)
    own = (torch.arange(48) // BPU) % 8
    n, base = {0: 5, 2: 9, 4: 3}, {0: 1, 2: 100, 4: 200}

    def table():
        v = own.expand(MB, -1).clone()
        for s, k in n.items():
            v[s, :k] = own[:k] + 8 * (base[s] + torch.arange(k))
        return v

    rig.publish(table())
    got = []
    for act in ("grow 0", "grow 2", "grow 4", "select 2", "select 1", "select 0"):
        rig.log.clear()
        what, s = act.split()
        if what == "grow":
            n[int(s)] += 1
            rig.publish(table())
        else:
            rig.be.spec.select(int(s))
        got.append(list(rig.log))
        assert rig.holds(rig.cur())
    assert got == want, (lazy, got)


# ---- 3. the generator's publishes (Glm53Generator._write_page_rows on the owner-block map, GLM53_PT_CHANGED_ROWS):
#         rows growing a block, a release + a new request, two slots trading rows, a repeated-block cut, step
#         switches; the shim backend forwards every page-table write to the real StackBackend.write
def gen_run(lazy):
    os.environ["GLM53_SPEC_PT_LAZY"] = lazy
    hold = [None]

    class Be(T._MockPrefixBackend):
        def allocate_decode_io(self, kv):
            io = super().allocate_decode_io(kv)
            self.pt_buf = io.page_table
            return io

        def write(self, buf, value):
            super().write(buf, value)
            if buf is getattr(self, "pt_buf", None):
                if hold[0] is None:
                    hold[0] = Rig(self.C, int(torch.as_tensor(value).shape[1]))
                hold[0].publish(value)

    be = Be(MB)
    gen = Glm53Generator(be, tokenizer=SimpleNamespace(stop_token_ids=()), max_batch=MB, max_seq_len=T.SEQ)
    kv = gen.allocate_kv_cache(MB, T.SEQ, num_blocks=MB * T.BPS + MB, null_block=0)
    gen.shared_pages = True
    b, rig = gen._binding(kv), hold[0]
    n = {0: 5, 2: 9, 4: 3, 8: 12, 12: 4, 14: 6}

    def row(s, blocks=None):
        r = torch.full((T.BPS,), -1, dtype=torch.int64)
        bl = blocks if blocks is not None else T._blocks(s, n[s] * T.BLOCK_SIZE)
        r[: len(bl)] = torch.tensor(bl, dtype=torch.int64)
        return r

    used = lambda: {s: (n[s] - 1) * T.BLOCK_SIZE + 1 for s in n}  # noqa: E731
    snaps = []

    def pub(rows=None):
        gen._write_page_rows(b, rows if rows is not None else {s: row(s) for s in n}, used())
        rig.main_ok()
        assert rig.holds(rig.cur())
        snaps.append(rig.snap())

    pub()
    for grow, sel in (([0], None), ([2, 8], 2), ([], None), ([4], 1), ([0, 12, 14], 0), ([2], 2)):
        for s in grow:
            n[s] += 1
        pub()
        if sel is not None:
            rig.be.spec.select(sel)
            assert rig.holds(rig.cur())
            snaps.append(rig.snap())
    n[4] = 4  # a new request in slot 4 (the old row released by its owner first): another pool's blocks
    gen.release_slot(kv, 4)
    new4 = [1 + 15 * T.BPS + j for j in range(4)]
    pub({s: row(s, new4 if s == 4 else None) for s in n})
    rows = {s: row(s, new4 if s == 4 else None) for s in n}
    rows[0], rows[2], n[0], n[2] = rows[2], rows[0], n[2], n[0]  # (slots 0 and 2 trade rows: a remap)
    pub(rows)
    rep = T._blocks(8, n[8] * T.BLOCK_SIZE)
    rows[8] = row(8, rep + [rep[2]])  # (a repeated block: the row is cut there)
    pub(rows)
    rig.be.spec.prepare()
    for i in range(len(rig.steps)):
        rig.be.spec.select(i)
        assert rig.holds(rig.steps[i])
    return snaps, len(rig.log), int(gen.counters["page_table_refresh"])


CALLS.clear()
g0, w0, r0 = gen_run("0")
g1, w1, r1 = gen_run("1")
assert len(g0) == len(g1) and r0 == r1 and w1 < w0, (len(g0), len(g1), r0, r1, w0, w1)
assert all(a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) for a, b in zip(g0, g1))
assert CALLS and all(ok for _, _, ok in CALLS), CALLS
print("PT_LAZY_OK")
"""


def test_spec_pt_lazy_tables():
    """GLM53_SPEC_PT_LAZY (default 1): a page-table publish writes the CURRENT speculative step's tables only, its
    chip-local rows gathered from the backend's localized table; another step is written when made current (select) or
    before prepare / capture. Against the old path (=0: every step written at every publish, each localizing its own
    rows), on the real StackBackend.write + SpecMTP / SpecSwitch table code with stub device buffers: (1) random op
    sequences on the served lone 1x3 + primary 4x3 + alternate 8x1 steps (the sharded pool; a pool-MTP-cache variant; a
    lone step without a switch) -- block crossings, releases, new requests, slot remaps, prefix attaches, same-table
    publishes, step switches, entries, prepare, capture: after every op the main / chunk tables and the current step's
    tables hold exactly the bytes dsa_shard.local_rows gives (=0 too), the two runs' snapshots are equal, lazy leaves
    non-current steps stale and writes fewer tensors; (2) the served write counts (a slot-0 crossing: 7 tensor writes ->
    4, a crossing on another step's base: 3 -> 2, the stale step written once at its select); (3) the generator's own
    publishes (Glm53Generator._write_page_rows on the owner-block map) through the real backend write, with a release +
    new request, a slot remap and a repeated-block cut. spec_mtp / stack_model import ttnn: a child with a stub ttnn"""
    _stub_child(_PT_LAZY_CHILD, "PT_LAZY_OK")
    assert "ttnn" not in sys.modules
