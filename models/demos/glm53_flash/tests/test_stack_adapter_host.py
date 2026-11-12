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
on shared pages, the release, the LRU reuse of unpinned entries, and the fallbacks to a prefill from 0.

    PYTHONPATH=<worktree> <tt-metal>/python_env/bin/pytest --noconftest -p no:cacheprovider -q \\
        models/demos/glm53_flash/tests/test_stack_adapter_host.py
"""

from __future__ import annotations

import contextlib
import inspect
import sys
import types
from dataclasses import dataclass, field
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
    row_params,
    trim_page_rows,
)

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
    chunks). Model: next token = (token + 1) % VOCAB; the per-slot ledger asserts positions and page coverage."""

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

    def __init__(self, max_batch: int):
        self.max_batch = max_batch
        self.chunks: list = []  # (slot, chunk_start, n_valid, logits_rows, sampled)
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

    # ---- prefill
    def prepare_prefill(self, kv):
        self.prepared = (True, self.prepared[1])

    def prefill_chunk(self, tokens, *, kv_cache, slot, chunk_start, n_valid, logits_rows, sample=None, io=None, **sio):
        st = kv_cache.state[slot]
        if chunk_start == 0:
            st.clear()
        assert len(st) == chunk_start, f"slot {slot}: state holds {len(st)} tokens, chunk starts at {chunk_start}"
        st.extend(int(t) for t in tokens[:n_valid].tolist())
        self.chunks.append((slot, chunk_start, n_valid, logits_rows, sample is not None))
        if logits_rows == "none":
            return None
        nxt = (st[-1] + 1) % VOCAB
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
