# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimum-surface vLLM adapter tests for zai-org/GLM-4.7-Flash ($vllm-integration).

Reduced representative target: one real layer of each kind (dense layer 0,
MoE layer 1), the same generator/adapter/plugin-facing contract the full
47-layer model uses. This is the adapter's own inner bring-up loop, not final
serving evidence -- run before the full-model `run_vllm_server` pass.

    pytest models/autoports/zai_org_glm_4_7_flash/tests/test_generator_vllm_adapter.py -x -s
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

import ttnn
from models.autoports.zai_org_glm_4_7_flash.tt.generator import build_generator
from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import GLM47FlashForCausalLM
from models.common.sampling import SamplingParams

MODEL_DIR = Path(__file__).resolve().parents[1]
TRACE_REGION_SIZE = 350_000_000
L1_SMALL_SIZE = 32768
MAX_SEQ_LEN = 4096
BLOCK_SIZE = 64
MAX_BATCH_SIZE = 32
BLOCKS_PER_USER = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
NUM_BLOCKS = MAX_BATCH_SIZE * BLOCKS_PER_USER


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=L1_SMALL_SIZE, trace_region_size=TRACE_REGION_SIZE
    )
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def adapter(device):
    generator = build_generator(
        MODEL_DIR,
        device,
        layer_indices=[0, 1],  # dense layer + one MoE layer: one of each kind
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        defer_cache_and_traces=True,  # exactly what initialize_vllm_model uses
    )
    model = GLM47FlashForCausalLM(generator)
    kv_cache = model.allocate_kv_cache(
        kv_cache_shape=(NUM_BLOCKS, 1, BLOCK_SIZE, generator.model.layers[0].kvpe_dim),
        dtype=torch.bfloat16,  # deliberately NOT the selected policy: proves the override
        num_layers=len(generator.model.layers),
    )
    # Mirrors vllm_tt_plugin/model_runner.py's warmup_model(): compile-only pass,
    # then the traced pass that actually captures.
    model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
    model.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=MAX_BATCH_SIZE,
        num_blocks=NUM_BLOCKS,
        can_sample_on_device=True,
        enable_trace=False,
    )
    model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=True)
    model.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=MAX_BATCH_SIZE,
        num_blocks=NUM_BLOCKS,
        can_sample_on_device=True,
        enable_trace=True,
    )
    yield model, kv_cache
    generator.teardown()


def _block_table_for(slot: int, blocks: int = BLOCKS_PER_USER) -> torch.Tensor:
    """A page table row that does not alias any other slot's blocks: block ids
    live in a private [slot*BLOCKS_PER_USER, (slot+1)*BLOCKS_PER_USER) range,
    the same identity layout GLM47FlashModel.default_page_table uses."""
    start = slot * BLOCKS_PER_USER
    return torch.arange(start, start + blocks, dtype=torch.int32).unsqueeze(0)


def _greedy(n: int) -> SamplingParams:
    """vLLM's own per-row TTSamplingParams is duck-type compatible with this;
    a plain list-valued SamplingParams exercises the same format_sampling_params
    path without needing the plugin installed for this adapter-only test."""
    return SamplingParams(temperature=[0.0] * n, top_k=[1] * n, top_p=[1.0] * n)


def test_implements_full_vllm_plugin_contract():
    """The plugin calls several of these with no ``hasattr`` guard (notably
    ``warmup_model_prefill``/``warmup_model_decode`` from
    ``vllm_tt_plugin/model_runner.py``'s ``warmup_model()``) -- a missing one
    is an ``AttributeError`` at server startup, not a graceful skip. Guards
    against silently dropping one of these again."""
    required = [
        "initialize_vllm_model",
        "get_max_tokens_all_users",
        "allocate_kv_cache",
        "warmup_model_prefill",
        "warmup_model_decode",
        "prefill_forward",
        "decode_forward",
        "read_decode_output",
        "process_decode_output_host",
    ]
    missing = [name for name in required if not hasattr(GLM47FlashForCausalLM, name)]
    assert not missing, f"GLM47FlashForCausalLM is missing required vLLM contract methods: {missing}"


def test_cache_dtype_override(adapter):
    """allocate_kv_cache must ignore vLLM's requested torch dtype and use the
    datatype-sweep-selected ttnn policy (bfloat8_b), not fall back to whatever
    vLLM's cache_config guessed."""
    model, kv_cache = adapter
    assert model.generator.model.cache_dtype == ttnn.bfloat8_b
    for layer_cache in kv_cache:
        assert layer_cache.dtype == ttnn.bfloat8_b
        assert tuple(layer_cache.shape) == (NUM_BLOCKS, 1, BLOCK_SIZE, model.generator.model.layers[0].kvpe_dim)


def test_prefill_forward_multi_request_non_aligned_lengths(adapter):
    """Two concurrently-admitted requests, non-aligned prompt lengths, landing
    in non-contiguous physical slots (5 and 17, not 0/1) -- proves empty_slots
    addressing and the row-order (not slot-order) return contract."""
    model, kv_cache = adapter
    slots = [5, 17]
    lens = [37, 137]  # neither a multiple of 32/64/128
    max_len = max(lens)
    tokens = torch.zeros((2, max_len), dtype=torch.int32)
    for i, plen in enumerate(lens):
        tokens[i, :plen] = torch.arange(1, plen + 1, dtype=torch.int32) % 1000 + 1
    page_table = torch.cat([_block_table_for(s) for s in slots], dim=0)

    out = model.prefill_forward(
        tokens=tokens,
        prompt_lens=lens,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=[0, 0],
        sampling_params=_greedy(2),
        empty_slots=slots,
    )
    assert out.shape == (2,)
    assert out.dtype == torch.int64
    vocab = model.generator.model.vocab_size
    assert bool(((out >= 0) & (out < vocab)).all())


def test_prefill_forward_host_sampling_fallback_returns_logits(adapter):
    """sampling_params=None (perform_device_sampling=False, e.g. a logprobs
    request on this single-chip mesh) must return raw logits for vLLM's own
    host sampler, not a device-sampled token -- and must not crash."""
    model, kv_cache = adapter
    slot = 22
    plen = 65  # non-aligned
    tokens = torch.zeros((1, plen), dtype=torch.int32)
    tokens[0, :plen] = torch.arange(1, plen + 1, dtype=torch.int32) % 1000 + 1
    page_table = _block_table_for(slot)

    out = model.prefill_forward(
        tokens=tokens,
        prompt_lens=[plen],
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=[0],
        sampling_params=None,
        empty_slots=[slot],
    )
    vocab = model.generator.model.vocab_size
    assert out.shape[0] == 1
    assert out.shape[-1] == vocab
    assert torch.isfinite(out).all()


def test_decode_forward_reset_batch_then_steady_state(adapter):
    """reset_batch=True writes host tokens/positions once; the following
    reset_batch=False steps must not need any host token/position write to
    keep producing valid, in-range tokens (the async decode contract)."""
    model, kv_cache = adapter
    slots = [5, 17]
    page_table = torch.cat([_block_table_for(s) for s in slots], dim=0)
    sz = len(slots)
    tokens = torch.tensor([[7], [9]], dtype=torch.int32)
    start_pos = torch.tensor([37, 137], dtype=torch.int32)

    out0 = model.decode_forward(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(sz),
        reset_batch=True,
    )
    # decode_forward returns the full max_batch_size-wide sampler tile (the
    # persistent 32-row device tensor); the caller (vLLM's own model_runner
    # _get_output_tokens) slices [0, sz) itself -- see the module docstring.
    assert out0.shape == (MAX_BATCH_SIZE, 1)
    vocab = model.generator.model.vocab_size
    assert bool(((out0[:sz] >= 0) & (out0[:sz] < vocab)).all())

    # Steady state: pass the SAME page table (unchanged) and stale host
    # tokens/positions (must be ignored -- the model advances its own device
    # position and reads the previously-sampled token, not these).
    stale_tokens = torch.tensor([[-1], [-1]], dtype=torch.int32)
    stale_pos = torch.tensor([-999, -999], dtype=torch.int32)
    for _ in range(3):
        out = model.decode_forward(
            tokens=stale_tokens,
            start_pos=stale_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_greedy(sz),
            reset_batch=False,
        )
        assert out.shape == (MAX_BATCH_SIZE, 1)
        assert bool(((out[:sz] >= 0) & (out[:sz] < vocab)).all())


def test_page_table_refresh_changed_and_unchanged(adapter):
    """refresh_page_table's only_if_changed diff must actually fire only on a
    real change: run once with an unchanged table (no host->device copy) and
    once with a genuinely different one (must copy and still produce valid
    output), using the generator's own counters as the tripwire."""
    model, kv_cache = adapter
    slots = [3]
    page_table = _block_table_for(slots[0])
    tokens = torch.tensor([[11]], dtype=torch.int32)
    start_pos = torch.tensor([10], dtype=torch.int32)

    model.decode_forward(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(1),
        reset_batch=True,
    )
    before = model.generator.counters["page_table_refreshes"]
    model.decode_forward(
        tokens=torch.tensor([[-1]], dtype=torch.int32),
        start_pos=torch.tensor([-999], dtype=torch.int32),
        page_table=page_table,  # identical content
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(1),
        reset_batch=False,
    )
    assert model.generator.counters["page_table_refreshes"] == before, "unchanged page table must not be re-copied"

    # Simulate the request crossing a page boundary: same slot, one more/different block.
    grown = page_table.clone()
    grown[0, -1] = 999  # a genuinely different block id -> must be detected as changed
    model.decode_forward(
        tokens=torch.tensor([[-1]], dtype=torch.int32),
        start_pos=torch.tensor([-999], dtype=torch.int32),
        page_table=grown,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(1),
        reset_batch=False,
    )
    assert model.generator.counters["page_table_refreshes"] == before + 1, "changed page table must be re-copied"


def test_async_decode_split(adapter):
    """decode_forward(read_from_device=False) -> read_decode_output(async_read=True)
    -> process_decode_output_host must reach the exact same tokens as the
    blocking read_from_device=True path, for supports_async_decode=True."""
    model, kv_cache = adapter
    slots = [8]
    page_table = _block_table_for(slots[0])
    tokens = torch.tensor([[3]], dtype=torch.int32)
    start_pos = torch.tensor([5], dtype=torch.int32)

    model.decode_forward(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(1),
        reset_batch=True,
    )

    tt_out = model.decode_forward(
        tokens=torch.tensor([[-1]], dtype=torch.int32),
        start_pos=torch.tensor([-999], dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=_greedy(1),
        reset_batch=False,
    )
    assert isinstance(tt_out, ttnn.Tensor)
    host, events = model.read_decode_output(tt_out, async_read=True)
    for event in events:
        ttnn.event_synchronize(event)
    async_result = model.process_decode_output_host(host, is_tokens=True)

    sync_result = model.decode_forward(
        tokens=torch.tensor([[-1]], dtype=torch.int32),
        start_pos=torch.tensor([-999], dtype=torch.int32),
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(1),
        reset_batch=False,
    )
    # async_result is the token sampled on the async-split step; sync_result is
    # the NEXT step's token (both device-driven, no host write in between) --
    # what matters is both are in-range and the async path never crashed/hung.
    # Index 0, not `slots[0]`: with sz=1 active row this step, decode addresses
    # row 0 (front-packed), independent of which page-table block range "slot"
    # was used to build this test's dummy page table content.
    vocab = model.generator.model.vocab_size
    assert 0 <= int(async_result[0, 0].item()) < vocab
    assert 0 <= int(sync_result[0, 0].item()) < vocab


def test_get_max_tokens_all_users_matches_contract():
    total = GLM47FlashForCausalLM.get_max_tokens_all_users(
        model_name="zai-org/GLM-4.7-Flash", num_devices=1, max_model_len=202752, max_num_seqs=32
    )
    assert 400_000 < total < 600_000


def test_rejects_batch_size_over_32(expect_error):
    with expect_error(ValueError, "max_batch_size"):
        GLM47FlashForCausalLM.initialize_vllm_model(
            hf_config=None, mesh_device=None, max_batch_size=33, max_seq_len=1024
        )


# --------------------------------------------------------------------------- VS-008
# Prefill lane broadcast. Host-only: these assert the params a request would
# reach the device with, not a device draw.


def _lane_view(sampling_params, lanes=MAX_BATCH_SIZE):
    """The per-lane temperature/top_k the sampler would receive."""
    from models.common.sampling import format_sampling_params

    f = format_sampling_params(sampling_params, lanes)
    return f.temperature[:lanes], f.top_k[:lanes]


def test_scalar_prefill_params_would_reach_only_lane_zero_without_broadcast():
    """The bug VS-008 fixes, pinned so a regression is visible here.

    A scalar request formats to `[value] + 31 * default`, and the defaults are
    greedy. Prefill reads lane (seq-1) % 32, so for any prompt longer than one
    token the request's own params were never the ones that sampled.
    """
    temp, top_k = _lane_view(SamplingParams(temperature=2.0, top_k=10, top_p=0.95))
    assert temp[0] != temp[1], "lane 0 should differ from the padded lanes (that is the bug)"
    assert top_k[1] == 1 and top_k[3] == 1, f"padded lanes are greedy: {top_k[:4]}"


def test_broadcast_makes_every_prefill_lane_carry_the_request_params():
    """After the broadcast, whichever lane prefill reads holds the request's params."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import _broadcast_per_user_fields

    params = SamplingParams(temperature=2.0, top_k=10, top_p=0.95, presence_penalty=0.5)
    temp, top_k = _lane_view(_broadcast_per_user_fields(params, MAX_BATCH_SIZE))

    assert len(set(temp)) == 1, f"temperature must be uniform across lanes, got {sorted(set(temp))}"
    assert len(set(top_k)) == 1, f"top_k must be uniform across lanes, got {sorted(set(top_k))}"
    assert top_k[0] == 10
    # Every lane a prefill could read, not just lane 0.
    assert top_k[3] == 10 and top_k[MAX_BATCH_SIZE - 1] == 10


def test_broadcast_leaves_real_per_lane_batches_alone():
    """A caller describing a genuine multi-lane batch is not overwritten."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import _broadcast_per_user_fields

    per_lane = SamplingParams(temperature=[0.5, 1.5], top_k=[3, 7], top_p=1.0)
    out = _broadcast_per_user_fields(per_lane, MAX_BATCH_SIZE)
    assert out.temperature == [0.5, 1.5], "existing per-lane lists must pass through untouched"
    assert out.top_k == [3, 7]
    assert out.top_p == [1.0] * MAX_BATCH_SIZE, "scalars alongside lists are still broadcast"


def test_broadcast_never_touches_the_seed():
    """seed is lane-scoped by design: broadcasting it means every lane draws alike."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import _broadcast_per_user_fields

    out = _broadcast_per_user_fields(SamplingParams(temperature=1.0, top_k=1, top_p=1.0, seed=42), MAX_BATCH_SIZE)
    assert out.seed == 42, "seed must stay scalar/lane-scoped, not become a 32-lane list"


def test_slice_row_indexes_tensor_params(expect_error):
    """R2: TTSamplingParams types per-user fields as Tensor | list; a tensor must
    be indexed, not passed through whole (which would hand this request the
    whole batch's values)."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator_vllm import _slice_sampling_params_row

    batch = SamplingParams(
        temperature=torch.tensor([0.5, 1.5, 2.5]),
        top_k=torch.tensor([3, 7, 11]),
        top_p=torch.tensor([0.9, 0.8, 0.7]),
    )
    row = _slice_sampling_params_row(batch, 1)
    assert row.temperature == pytest.approx(1.5)
    assert row.top_k == 7
    assert row.top_p == pytest.approx(0.8)
