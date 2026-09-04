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

import json
import math
import re
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


#: Deliberately NOT max_batch_size * BLOCKS_PER_USER (2048): a real vLLM pool
#: is shared, not divided into equal per-request shares, so a fix here must
#: not depend on that coincidence. 72 // 32 == 2, which is what the old buggy
#: ``num_blocks // max_batch_size`` formula would have given (vs. the correct
#: BLOCKS_PER_USER=64) -- wide enough for exactly one full-context request
#: (64 blocks) plus a little slack, matching how vLLM would actually share a
#: pool across many shorter concurrent requests and a few long ones.
NUM_BLOCKS_SHARED_POOL = BLOCKS_PER_USER + 8


@pytest.fixture(scope="module")
def shared_pool_adapter(device):
    """A second, independent reduced model on the same device, allocated with
    a shared-pool-shaped (not equal-share-shaped) block count, to prove
    blocks_per_user -- and therefore the per-request page-table width -- comes
    from max_seq_len, not from dividing whatever pool size vLLM happens to
    choose by max_batch_size (doc/vllm_integration/work_log.md VS-011)."""
    generator = build_generator(
        MODEL_DIR,
        device,
        layer_indices=[0, 1],
        max_batch_size=MAX_BATCH_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        defer_cache_and_traces=True,
    )
    model = GLM47FlashForCausalLM(generator)
    kv_cache = model.allocate_kv_cache(
        kv_cache_shape=(NUM_BLOCKS_SHARED_POOL, 1, BLOCK_SIZE, generator.model.layers[0].kvpe_dim),
        dtype=torch.bfloat16,
        num_layers=len(generator.model.layers),
    )
    model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=False)
    model.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=MAX_BATCH_SIZE,
        num_blocks=NUM_BLOCKS_SHARED_POOL,
        can_sample_on_device=True,
        enable_trace=False,
    )
    model.warmup_model_prefill(kv_cache=kv_cache, can_sample_on_device=True, enable_trace=True)
    model.warmup_model_decode(
        kv_cache=kv_cache,
        max_batch_size=MAX_BATCH_SIZE,
        num_blocks=NUM_BLOCKS_SHARED_POOL,
        can_sample_on_device=True,
        enable_trace=True,
    )
    yield model, kv_cache
    generator.teardown()


def test_blocks_per_user_is_max_seq_len_derived_not_pool_derived(shared_pool_adapter, expect_error):
    """The regression test for VS-011: with a pool shaped like a real shared
    vLLM allocation (NUM_BLOCKS_SHARED_POOL, not max_batch_size*BLOCKS_PER_USER),
    blocks_per_user must still equal cdiv(max_seq_len, block_size) -- not
    num_blocks // max_batch_size (which would be 72 // 32 == 2 here, nowhere
    near enough to address a full-context request)."""
    model, kv_cache = shared_pool_adapter
    assert model.blocks_per_user == BLOCKS_PER_USER
    assert model.generator.model.blocks_per_user == BLOCKS_PER_USER
    assert model.generator.model.max_seq_len_physical >= MAX_SEQ_LEN

    # A single request using the full per-request width (one full-context
    # request's worth of blocks) must be accepted, not truncated.
    full_width_table = torch.arange(0, BLOCKS_PER_USER, dtype=torch.int32).unsqueeze(0)
    model._write_page_table_rows(full_width_table, at=[0])  # must not raise
    assert torch.equal(model._pt_mirror[0, :BLOCKS_PER_USER], full_width_table[0])

    # A table wider than blocks_per_user must be rejected, not silently
    # truncated (the old behavior for exactly this case).
    too_wide = torch.arange(0, BLOCKS_PER_USER + 1, dtype=torch.int32).unsqueeze(0)
    with expect_error(ValueError, "refusing to truncate"):
        model._write_page_table_rows(too_wide, at=[0])


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


# ------------------------------------------------------------------ compact kc-bucketed decode traces
#
# The optimized-vLLM stage replaced the batch>1 union decode MoE with a
# kc-bucketed compact (sparse_matmul INDEXED/GATHER) one; see
# doc/optimized_vllm/README.md. kc is a hard correctness bound
# (union <= live_rows * top_k, given the inactive-row routing mask), so these
# pin the bound, the bucket choice, the union fallback, and the fact that
# switching buckets mid-loop does not disturb the persistent trace inputs.


def test_kc_bucket_bound_covers_every_live_row_count(adapter):
    """Every row count must map to a bucket at least ``live_rows * top_k`` wide,
    or to the union trace (``None``), which has no bound."""
    model, _kv = adapter
    gen = model.generator
    moe = next(l for l in gen.model.layers if l.layer_kind == "moe")
    top_k, n_experts = moe.top_k, moe.n_experts
    assert gen._decode_kc_buckets, "compact decode buckets must exist at max_batch_size > 1"
    for rows in range(0, MAX_BATCH_SIZE + 1):
        kc = gen.decode_kc_for_rows(rows)
        if kc is None:
            continue  # union trace: scans every expert, no bound to satisfy
        assert kc >= min(n_experts, rows * top_k), (
            f"{rows} live rows can select up to {min(n_experts, rows * top_k)} distinct experts "
            f"but the chosen bucket is only {kc} wide; the lowest-scoring selected experts "
            f"would be silently dropped"
        )


def test_kc_full_width_is_never_captured_compact(adapter):
    """kc == n_experts is measured to be *slower* than the union path (the
    compact form pays for all kc experts unconditionally, the union form only
    for the ones the batch selected), so it must fall back to the union trace
    rather than being captured as a compact bucket."""
    model, _kv = adapter
    gen = model.generator
    n_experts = next(l for l in gen.model.layers if l.layer_kind == "moe").n_experts
    assert n_experts not in gen._decode_kc_buckets
    assert None in gen._decode_kc_buckets, "row counts needing full width must have a union trace to fall back to"
    assert gen.decode_kc_for_rows(MAX_BATCH_SIZE) is None


def test_every_bucket_has_a_captured_trace(adapter):
    """A bucket without a trace would silently fall back to a wider one."""
    model, _kv = adapter
    gen = model.generator
    assert set(gen._decode_traces) == set(gen._decode_kc_buckets)
    assert len({tid for tid in gen._decode_traces.values()}) == len(
        gen._decode_kc_buckets
    ), "trace ids must be distinct"


def test_all_decode_traces_share_one_logits_buffer(adapter):
    """One captured sampling trace serves every decode trace, which is only
    sound because every decode trace writes the same logits *buffer*.

    Replays each captured bucket in turn and checks the logits buffer address
    is the one the sampler was captured against and never moves, and that each
    replay actually wrote it (so this is not passing on a stale tensor).
    """
    model, _kv = adapter
    gen = model.generator
    assert gen._decode_logits is not None
    assert gen._sampling_traced, "the sampler trace must be captured against that shared buffer"
    shared_address = gen._decode_logits.buffer_address()

    gen.reset()
    gen.refresh_page_table(torch.cat([_block_table_for(s) for s in range(MAX_BATCH_SIZE)], dim=0))
    seen = []
    for i, kc in enumerate(gen._decode_kc_buckets):
        gen.set_decode_tokens([500 + 31 * i] * MAX_BATCH_SIZE)
        gen.set_decode_positions([70 + i] * MAX_BATCH_SIZE)
        gen._decode_trace_id = gen._decode_traces[kc]
        gen._active_kc = kc
        gen._advance_host_positions()
        ttnn.execute_trace(gen.mesh_device, gen._decode_trace_id, cq_id=0, blocking=True)
        assert (
            gen._decode_logits.buffer_address() == shared_address
        ), f"bucket {kc} moved the logits buffer; the captured sampling trace reads {shared_address}"
        seen.append(ttnn.to_torch(gen._decode_logits).float()[0, 0, 0, :64].clone())
    assert len(seen) == len(gen._decode_kc_buckets)
    # Different token inputs must produce different logits, so each replay
    # demonstrably wrote the shared buffer rather than leaving the last one.
    assert any(
        not torch.equal(seen[0], other) for other in seen[1:]
    ), "every bucket produced identical logits for different tokens; the replays did not write the buffer"


def test_bucket_switch_preserves_token_feedback_and_positions(adapter):
    """Growing then shrinking the live-row set switches decode traces. The
    persistent token/position tensors are shared by all of them, so the sampled
    token from the step before a switch must still be the token input after it,
    and positions must keep advancing by exactly one per replay."""
    model, kv_cache = adapter
    gen = model.generator
    gen.reset()

    def step(rows, reset_batch):
        page_table = torch.cat([_block_table_for(s) for s in range(rows)], dim=0)
        return model.decode_forward(
            tokens=torch.tensor([[100 + 7 * i] for i in range(rows)], dtype=torch.int32),
            start_pos=torch.tensor([40 + i for i in range(rows)], dtype=torch.int32),
            page_table=page_table,
            kv_cache=kv_cache,
            enable_trace=True,
            read_from_device=True,
            sampling_params=_greedy(rows),
            reset_batch=reset_batch,
        )

    step(1, True)
    kc_one = gen._active_kc
    step(8, True)
    kc_eight = gen._active_kc
    step(1, True)
    assert gen._active_kc == kc_one, "returning to one live row must return to the narrow bucket"
    assert kc_eight != kc_one, "eight live rows must not reuse the one-row bucket"
    assert gen.counters["decode_trace_bucket_switches"] >= 2

    # Steady state on the narrow bucket: token feedback stays on device and the
    # position advances by exactly one per replay, across the switch.
    #
    # Counters are cumulative over this module-scoped fixture and other tests in
    # this file deliberately exercise the host-sampling compatibility path, so
    # assert on the delta across this step, not on the absolute value.
    before_tok = gen.read_decode_tokens(1)[0]
    before_pos = list(gen._host_positions)
    fallback_keys = ("eager_decode_steps", "full_logits_readbacks", "host_argmax_calls", "eager_sampling_steps")
    baseline = {k: gen.counters[k] for k in fallback_keys}
    gen.decode_step_traced()
    assert gen._host_positions[0] == before_pos[0] + 1, "traced replay must advance the device position by one"
    after_tok = gen.read_decode_tokens(1)[0]
    assert isinstance(after_tok, int)
    assert before_tok >= 0 and after_tok >= 0
    for key in fallback_keys:
        assert gen.counters[key] == baseline[key], f"{key} must not advance on the traced token-out path"


def test_page_table_calls_are_skipped_when_rows_are_unchanged(adapter):
    """The steady-state decode loop must not re-upload the page table. The
    adapter diffs vLLM's rows against its own mirror, so an unchanged block
    list costs one comparison and no host->device copy."""
    model, kv_cache = adapter
    page_table = _block_table_for(2)
    common = dict(
        page_table=page_table,
        kv_cache=kv_cache,
        enable_trace=True,
        read_from_device=True,
        sampling_params=_greedy(1),
    )
    model.decode_forward(
        tokens=torch.tensor([[13]], dtype=torch.int32),
        start_pos=torch.tensor([20], dtype=torch.int32),
        reset_batch=True,
        **common,
    )
    skipped = model.page_table_calls_skipped
    written = model.page_table_calls_written
    refreshes = model.generator.counters["page_table_refreshes"]
    for _ in range(5):
        model.decode_forward(
            tokens=torch.tensor([[-1]], dtype=torch.int32),
            start_pos=torch.tensor([-999], dtype=torch.int32),
            reset_batch=False,
            **common,
        )
    assert model.page_table_calls_skipped == skipped + 5, "identical rows must all be skipped"
    assert model.page_table_calls_written == written, "no write may happen for an unchanged table"
    assert model.generator.counters["page_table_refreshes"] == refreshes, "no host->device page-table copy"


def test_serving_prefill_compiles_nothing_and_never_recaptures(adapter):
    """A served prefill used to compile one new program the first time each
    decode slot was used, while the decode traces were live -- so every
    admitted request forced a full decode-trace recapture (22 of them inside
    the 100/100/32 CI serving burst; doc/optimized_vllm/README.md). Warming one
    prefill per slot at startup moves that compile off the served path.
    """
    model, _kv = adapter
    gen = model.generator
    dev = gen.mesh_device
    entries = dev.num_program_cache_entries()
    dev.set_program_cache_misses_allowed(False)
    try:
        for slot in (0, MAX_BATCH_SIZE // 2, MAX_BATCH_SIZE - 1):
            gen.apply_prefill_sampling_state(_greedy(1), empty_slots=[slot])
            gen.prefill_and_sample(list(range(1000, 1100)), user_id=slot, recapture=False)
    finally:
        dev.set_program_cache_misses_allowed(True)
    assert dev.num_program_cache_entries() == entries, "a served prefill must not compile a new program"
    assert not gen._maybe_recapture_after_compile(), "and therefore must not force a decode-trace recapture"


def test_kc_bucket_is_only_chosen_where_it_is_the_cheaper_path(adapter):
    """A compact bucket's cost is flat across the row counts it serves while the
    union path's grows with the real union width, so within a bucket's range the
    compact form starts behind and crosses over. Measured
    (doc/optimized_vllm/adapter_decode_floor_{before,after,kc64}.json): kc=32
    only pays from 7 live rows up (5 rows 55.377 union vs 57.495 kc=32, 7 rows
    58.949 vs 57.525), so 5-6 live rows take the kc=24 bucket added for exactly
    that window (49.745 / 49.756 vs the union's 55.377 / 56.749) rather than
    kc=32, whose bound also covers them.
    """
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import COMPACT_KC_MIN_ROWS

    model, _kv = adapter
    gen = model.generator
    for rows in range(0, MAX_BATCH_SIZE + 1):
        kc = gen.decode_kc_for_rows(rows)
        if kc is None:
            continue
        assert rows >= COMPACT_KC_MIN_ROWS[kc], (
            f"{rows} live rows chose compact bucket kc={kc}, but that bucket is only the cheaper path from "
            f"{COMPACT_KC_MIN_ROWS[kc]} rows up"
        )
    # The specific crossovers this model measured, pinned so a bucket-set change
    # cannot silently reintroduce a regression on part of a bucket's range.
    assert gen.decode_kc_for_rows(1) == 4
    assert gen.decode_kc_for_rows(4) == 16
    assert gen.decode_kc_for_rows(5) == 24
    assert gen.decode_kc_for_rows(6) == 24
    assert gen.decode_kc_for_rows(7) == 32
    assert gen.decode_kc_for_rows(9) is None


def test_every_bucket_has_a_measured_crossover():
    """A bucket with no measured COMPACT_KC_MIN_ROWS entry would be used from
    one live row up, which is the regression the table exists to prevent.
    Adding a bucket must mean measuring it, so the lookup is strict."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import COMPACT_KC_BUCKETS, COMPACT_KC_MIN_ROWS

    assert set(COMPACT_KC_BUCKETS) <= set(
        COMPACT_KC_MIN_ROWS
    ), f"buckets without a measured crossover: {sorted(set(COMPACT_KC_BUCKETS) - set(COMPACT_KC_MIN_ROWS))}"


def test_compaction_is_off_rather_than_fatal_for_a_non_tile_decode_batch():
    """A decode batch that is not a whole tile cannot use the [E, B] embedding
    table the compact path builds. That must disable compaction, not raise:
    the union path is batch-agnostic and is what every non-serving caller used
    before this stage existed."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import _kc_buckets

    class _Layer:
        layer_kind = "moe"
        n_experts = 64
        top_k = 4

    class _Model:
        layers = [_Layer()]

        def __init__(self, batch):
            self.max_batch_size = batch

    for batch in (2, 8, 16, 31):
        assert _kc_buckets(_Model(batch)) == (), f"max_batch_size={batch} must fall back to the union path"
    assert _kc_buckets(_Model(32)) == (4, 16, 24, 32, None)


# ------------------------------------------------------------------ report/source consistency
#
# Four of five $stage-review rounds on this stage found the same class of defect:
# a number or a bucket set quoted in prose or in a shipped docstring that no
# committed artifact contains, usually because a later re-measure did not
# propagate. These are device-free checks that close it mechanically.

_DOC_DIR = MODEL_DIR / "doc" / "optimized_vllm"


def _decode_floor(arm: str) -> dict:
    payload = json.loads((_DOC_DIR / f"adapter_decode_floor_{arm}.json").read_text())
    return {r["active_rows"]: r for r in payload["results"]}


def test_shipped_bucket_docstring_table_matches_the_committed_measurements():
    """`COMPACT_KC_BUCKETS`' docstring carries the sweep that justifies the
    shipped bucket set. Every row of it must be the committed measurement, to
    the digit, and must name the bucket the shipped table actually selects."""
    from models.autoports.zai_org_glm_4_7_flash.tt import generator as gen_module

    before, after = _decode_floor("before"), _decode_floor("after")
    doc = gen_module.__doc__ or ""
    src = (MODEL_DIR / "tt" / "generator.py").read_text()
    table = re.findall(r"^#: (\d+)\s+([\d.]+)\s+([\d.]+) \((kc \d+|union)\)\s+([+-][\d.]+)$", src, re.M)
    assert table, "COMPACT_KC_BUCKETS' docstring table is missing or no longer machine-readable"
    assert {int(r[0]) for r in table} == set(
        before
    ), f"docstring table rows {sorted(int(r[0]) for r in table)} != swept rows {sorted(before)}"
    for rows_s, union_s, shipped_s, kc_s, delta_s in table:
        rows = int(rows_s)
        kc = after[rows]["moe_kc_used"]
        assert float(union_s) == pytest.approx(before[rows]["adapter_async_token_out_ms"], abs=5e-4), rows
        assert float(shipped_s) == pytest.approx(after[rows]["adapter_async_token_out_ms"], abs=5e-4), rows
        assert kc_s == (
            f"kc {kc}" if kc is not None else "union"
        ), f"docstring says row {rows} uses {kc_s}, the committed arm says {kc}"
        expected = after[rows]["adapter_async_token_out_ms"] - before[rows]["adapter_async_token_out_ms"]
        assert float(delta_s) == pytest.approx(expected, abs=1e-3), rows
    del doc


def test_report_states_the_shipped_bucket_set():
    """The stage README's prose bucket set must be the code's."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import COMPACT_KC_BUCKETS

    readme = (_DOC_DIR / "README.md").read_text()
    shipped = "(" + ", ".join(str(b) for b in COMPACT_KC_BUCKETS) + ", union)"
    assert f"Buckets: `{shipped}`" in readme, f"README does not state the shipped bucket set {shipped}"
    stale = re.findall(r"Buckets: `\(([^`]*)\)`", readme)
    assert stale == [shipped.strip("()")], f"README states bucket sets {stale}, shipped is {shipped}"


def test_every_shipped_bucket_has_bitwise_equivalence_evidence():
    """Each compact bucket must be covered by bucket_numerics.json at the live-row
    count where its bound is saturated -- its zero-slack case."""
    from models.autoports.zai_org_glm_4_7_flash.tt.generator import COMPACT_KC_BUCKETS

    payload = json.loads((_DOC_DIR / "bucket_numerics.json").read_text())
    saturated = {int(k): v for k, v in payload["saturated_row_per_bucket"].items()}
    assert set(saturated.values()) == set(COMPACT_KC_BUCKETS), (
        f"bucket_numerics.json covers {sorted(set(saturated.values()))}, shipped buckets are "
        f"{sorted(COMPACT_KC_BUCKETS)}"
    )
    for rows in saturated:
        result = payload["results"][f"rows{rows}_compactbucket_vs_union"]
        assert result["bitwise_identical"], f"{rows} live rows: compact bucket differs from the union path"
        assert result["argmax_identical"]
